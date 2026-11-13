"""Tests for the validation gate checks."""
import asyncio

import pytest

from search.gate import (
    CHECKS,
    GateConfig,
    GateResult,
    Check,
    fix_prompt,
    run_gate,
)


class FakeExec:
    """A gate container in which the commands containing any of `failing` fail."""

    def __init__(self, failing=(), answer="4"):
        self.failing, self.answer = tuple(failing), answer
        self.calls: list[dict] = []

    async def __call__(self, *cmd, env=None, timeout=None):
        text = " ".join(cmd)
        self.calls.append({"cmd": text, "env": env, "timeout": timeout})
        if any(marker in text for marker in self.failing):
            return False, "some output", f"error in: {text[:40]}"
        if text.startswith("cat "):
            return True, self.answer, ""
        return True, "", ""

    def commands(self):
        return [call["cmd"] for call in self.calls]


def gate(exec, **kwargs):
    return asyncio.run(run_gate(exec, **kwargs))


def test_all_checks_pass():
    exec = FakeExec()

    result = gate(exec)

    assert result.passed
    assert [c.name for c in result.checks] == ["import", "tests", "smoke"]
    assert result.failed_check is None and result.failure_reason() is None
    assert result.tests_passed is True
    commands = exec.commands()
    assert commands[0] == "python -c import agent_code.agent"
    assert commands[1] == "bash -c cd /home/agent/agent_code && pytest"
    assert "What is 2 + 2?" in commands[2]
    assert commands[3].startswith("python -m agent_code benchmark --prompt-file /tmp/gate_prompt.txt")
    assert commands[4] == "cat /tmp/gate_logs/answer.txt"
    assert result.to_dict() == {
        "passed": True,
        "checks": [
            {"name": name, "passed": True, "command": check.command}
            for name, check in zip(CHECKS, result.checks)
        ],
    }


def test_stops_at_the_first_failing_check():
    exec = FakeExec(failing=["import agent_code"])

    result = gate(exec)

    assert not result.passed
    assert [c.name for c in result.checks] == ["import"]
    assert len(exec.calls) == 1  # no tests, no smoke run
    assert result.tests_passed is None
    assert "Validation gate failed at 'import'" in result.failure_reason()
    assert "error in: python -c import" in result.failure_reason()
    assert "error in" in result.to_dict()["checks"][0]["output"]


def test_failing_tests_fail_the_gate():
    result = gate(FakeExec(failing=["pytest"]))

    assert not result.passed
    assert result.failed_check.name == "tests"
    assert result.tests_passed is False


def test_tests_that_already_failed_for_the_parent_are_not_held_against_the_agent():
    asked = []

    async def parent_tests_pass():
        asked.append(True)
        return False

    result = gate(FakeExec(failing=["pytest"]), parent_tests_pass=parent_tests_pass)

    assert result.passed
    assert [c.name for c in result.checks] == ["import", "tests", "smoke"]
    assert asked == [True]
    tests = result.checks[1]
    assert "already did for the parent" in tests.note
    # Its own suite still fails, which is what its children are compared with
    assert result.tests_passed is False
    assert result.to_dict()["checks"][1]["note"] == tests.note


@pytest.mark.parametrize("parent_result", [True, None])
def test_regressed_tests_fail_the_gate(parent_result):
    async def parent_tests_pass():
        return parent_result  # passed, or could not be determined

    result = gate(FakeExec(failing=["pytest"]), parent_tests_pass=parent_tests_pass)

    assert not result.passed
    assert result.failed_check.name == "tests"


def test_parent_is_not_consulted_when_tests_pass():
    async def parent_tests_pass():
        raise AssertionError("should not be needed")

    assert gate(FakeExec(), parent_tests_pass=parent_tests_pass).passed


def test_smoke_run_that_errors_fails_the_gate():
    result = gate(FakeExec(failing=["agent_code benchmark"]))
    assert result.failed_check.name == "smoke"


@pytest.mark.parametrize("answer", ["", "  \n"])
def test_smoke_run_without_an_answer_fails_the_gate(answer):
    result = gate(FakeExec(answer=answer))
    assert result.failed_check.name == "smoke"
    assert "submitted no answer" in result.failed_check.note


def test_smoke_run_without_answer_file_fails_the_gate():
    result = gate(FakeExec(failing=["cat /tmp/gate_logs"]))
    assert result.failed_check.name == "smoke"


def test_smoke_model_override():
    exec = FakeExec()
    gate(exec, config=GateConfig(model="HAIKU_35"))
    smoke = next(c for c in exec.calls if "agent_code benchmark" in c["cmd"])
    assert smoke["env"] == {"AGENT_MODEL": "HAIKU_35"}

    exec = FakeExec()
    gate(exec)
    smoke = next(c for c in exec.calls if "agent_code benchmark" in c["cmd"])
    assert smoke["env"] is None
    assert smoke["timeout"] > 120  # a margin over the agent's own time limit


def test_only_configured_checks_run():
    exec = FakeExec(failing=["pytest"])
    result = gate(exec, config=GateConfig(checks=("smoke", "import")))
    assert result.passed
    assert [c.name for c in result.checks] == ["import", "smoke"]  # canonical order

    exec = FakeExec()
    result = gate(exec, config=GateConfig(checks=()))
    assert result.passed and exec.calls == []
    assert not GateConfig(checks=()).enabled


def test_config():
    assert GateConfig().checks == ("import", "tests", "smoke")
    assert GateConfig().enabled and GateConfig().retries == 0
    assert GateConfig.parse_checks("import, tests") == ("import", "tests")
    assert GateConfig.parse_checks("none") == ()
    assert GateConfig.parse_checks("") == ()
    with pytest.raises(ValueError, match="Unknown gate check"):
        GateConfig(checks=("import", "lint"))
    with pytest.raises(ValueError):
        GateConfig(retries=-1)


def test_fix_prompt_carries_the_failure():
    result = GateResult(
        [
            Check("import", True, "python -c 'import agent_code.agent'"),
            Check("tests", False, "cd /home/agent/agent_code && pytest", "E  NameError: name 'foo'"),
        ]
    )
    prompt = fix_prompt(result)
    assert "'tests'" in prompt
    assert "cd /home/agent/agent_code && pytest" in prompt
    assert "NameError: name 'foo'" in prompt
    assert "/home/agent/workdir" in prompt
    assert "Do not delete, skip or weaken tests" in prompt
