"""
The validation gate: is a newly made agent able to run at all?

A meta-improvement step can leave the agent's code broken. Benchmarking a
broken agent costs a full evaluation, and yields a score of zero that says
nothing about the idea that was tried. The gate runs a few cheap checks in
the agent's own sandbox first:

  import  the agent's code can be imported
  tests   its unit tests pass
  smoke   it can solve a trivial problem end to end

The checks run in a container in which the agent's code has been copied to
AGENT_CODE_DIR. They are given an `exec` function that runs a command in it.
"""

from typing import Awaitable, Callable, Protocol
from dataclasses import dataclass, field

AGENT_CODE_DIR = "/home/agent/agent_code"
CHECKS = ("import", "tests", "smoke")

TESTS_TIMEOUT_SECONDS = 15 * 60
SMOKE_TIMEOUT_SECONDS = 120  # for the agent itself; the check allows a margin
SMOKE_COST_THRESHOLD_USD = 1.0
SMOKE_PROMPT = "What is 2 + 2? Submit just the number as your answer."
SMOKE_PROMPT_FILE = "/tmp/gate_prompt.txt"
SMOKE_LOGDIR = "/tmp/gate_logs"

OUTPUT_TAIL_CHARS = 3000


class Exec(Protocol):
    async def __call__(
        self, *cmd: str, env: dict[str, str] | None = None, timeout: float | None = None
    ) -> tuple[bool, str, str]:
        """Run a command in the gate container: (success, stdout, stderr)"""
        ...


@dataclass(frozen=True)
class GateConfig:
    # Which of CHECKS to run, in order; empty to switch the gate off
    checks: tuple[str, ...] = CHECKS
    # How often the meta agent gets to fix an agent that fails the gate
    retries: int = 0
    # Model for the smoke run (default: the agent's own configured model)
    model: str | None = None

    def __post_init__(self):
        unknown = [c for c in self.checks if c not in CHECKS]
        if unknown:
            raise ValueError(
                f"Unknown gate check(s) {', '.join(unknown)}; choose from {', '.join(CHECKS)}"
            )
        if self.retries < 0:
            raise ValueError("retries must not be negative")
        # Always run in the canonical order: cheapest and most basic first
        object.__setattr__(self, "checks", tuple(c for c in CHECKS if c in self.checks))

    @property
    def enabled(self) -> bool:
        return bool(self.checks)

    @classmethod
    def parse_checks(cls, value: str) -> tuple[str, ...]:
        """'import,tests' -> ('import', 'tests'); 'none' -> ()"""
        if value.strip().lower() in ("", "none"):
            return ()
        return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass
class Check:
    name: str
    passed: bool  # whether the agent gets through this check
    command: str
    output: str = ""  # the end of what the command printed
    note: str | None = None
    # What the command itself reported, where that differs from `passed`
    raw_passed: bool | None = None

    def to_dict(self) -> dict:
        data = {"name": self.name, "passed": self.passed, "command": self.command}
        if self.note:
            data["note"] = self.note
        if not self.passed:
            data["output"] = self.output
        return data


@dataclass
class GateResult:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_check(self) -> Check | None:
        return next((check for check in self.checks if not check.passed), None)

    @property
    def tests_passed(self) -> bool | None:
        """Whether the agent's own test suite passes (None: it was not run)"""
        check = next((c for c in self.checks if c.name == "tests"), None)
        if check is None:
            return None
        return check.raw_passed if check.raw_passed is not None else check.passed

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [c.to_dict() for c in self.checks]}

    def failure_reason(self) -> str | None:
        check = self.failed_check
        if check is None:
            return None
        return f"Validation gate failed at '{check.name}' ({check.command}): {check.output}"


def _tail(stdout: str, stderr: str) -> str:
    text = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
    return text[-OUTPUT_TAIL_CHARS:]


async def check_import(exec: Exec) -> Check:
    command = "python -c 'import agent_code.agent'"
    success, stdout, stderr = await exec("python", "-c", "import agent_code.agent", timeout=120)
    return Check("import", success, command, _tail(stdout, stderr))


async def check_tests(
    exec: Exec, parent_tests_pass: Callable[[], Awaitable[bool | None]] | None = None
) -> Check:
    """The agent's unit tests must pass, unless its parent's did not either.

    A new agent is not held to a standard that the code it started from did
    not meet: a suite that was already failing (in this sandbox, say) would
    otherwise disqualify every agent ever made. `parent_tests_pass` is only
    called when the tests fail.
    """
    command = f"cd {AGENT_CODE_DIR} && pytest"
    success, stdout, stderr = await exec(
        "bash", "-c", command, timeout=TESTS_TIMEOUT_SECONDS
    )
    check = Check("tests", success, command, _tail(stdout, stderr), raw_passed=success)
    if not success and parent_tests_pass is not None:
        if await parent_tests_pass() is False:
            check.passed = True
            check.note = "The tests fail, as they already did for the parent agent"
    return check


async def check_smoke(exec: Exec, model: str | None = None) -> Check:
    """The agent must run on a trivial problem and submit an answer."""
    command = (
        f"python -m agent_code benchmark --prompt-file {SMOKE_PROMPT_FILE} "
        f"--logdir {SMOKE_LOGDIR} --timeout {SMOKE_TIMEOUT_SECONDS} "
        f"--cost-threshold {SMOKE_COST_THRESHOLD_USD}"
    )
    success, stdout, stderr = await exec(
        "bash", "-c", f"cat > {SMOKE_PROMPT_FILE} << 'EOF'\n{SMOKE_PROMPT}\nEOF", timeout=60
    )
    if not success:
        return Check("smoke", False, command, _tail(stdout, stderr), "Could not write the prompt")

    success, stdout, stderr = await exec(
        *command.split(),
        env={"AGENT_MODEL": model} if model else None,
        timeout=SMOKE_TIMEOUT_SECONDS + 60,
    )
    if not success:
        return Check("smoke", False, command, _tail(stdout, stderr))

    has_answer, answer, _ = await exec("cat", f"{SMOKE_LOGDIR}/answer.txt", timeout=60)
    if not has_answer or not answer.strip():
        return Check(
            "smoke", False, command, _tail(stdout, stderr), "The agent ran, but submitted no answer"
        )
    return Check("smoke", True, command)


async def run_gate(
    exec: Exec,
    config: GateConfig = GateConfig(),
    parent_tests_pass: Callable[[], Awaitable[bool | None]] | None = None,
) -> GateResult:
    """Run the configured checks, stopping at the first that fails."""
    result = GateResult()
    for name in config.checks:
        if name == "import":
            check = await check_import(exec)
        elif name == "tests":
            check = await check_tests(exec, parent_tests_pass)
        else:
            check = await check_smoke(exec, config.model)
        result.checks.append(check)
        if not check.passed:
            break
    return result


def fix_prompt(result: GateResult) -> str:
    """What the meta agent is asked to do about an agent that failed the gate."""
    check = result.failed_check
    assert check is not None
    note = f"\n{check.note}\n" if check.note else ""
    return f"""The agent code in /home/agent/workdir was modified a moment ago, and it no longer passes validation. It cannot be used until it does.

The check that failed is '{check.name}'. With the agent code placed at {AGENT_CODE_DIR}, this command was run:

{check.command}
{note}
It ended with:

---
{check.output}
---

Find the cause and fix the code in /home/agent/workdir, changing as little as possible: keep the feature that was being added if you can, and remove it only if it cannot be made to work. Do not delete, skip or weaken tests to get them to pass. Make sure any new dependencies are in requirements.txt. Before you finish, run the failing check yourself (with /home/agent/workdir in place of {AGENT_CODE_DIR}) to confirm that it passes."""
