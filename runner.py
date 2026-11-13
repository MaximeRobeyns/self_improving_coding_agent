# Self-Improving Coding Agent runner
"""
An agent runner that handles benchmark evaluation and self-improvement cycles.
"""

import os
import re
import sys
import json
import shutil
import random
import logging
import asyncio
import platform
import argparse
import subprocess

from uuid import uuid4
from contextlib import asynccontextmanager
from typing import Type
from pathlib import Path
from datetime import datetime
from asyncio.subprocess import Process

from base_agent.src.benchmarks import benchmark_registry
from base_agent.src.benchmarks.base import BaseBenchmark, BenchmarkTracker, Problem
from base_agent.src.utils.archive_analysis import (
    ArchiveAnalyzer,
    base_problem_id,
    result_id,
)
from base_agent.src.utils.usage import USAGE_FILENAME, read_usage
from base_agent.src.utils.lineage import (
    STATUS_OK,
    STATUS_FAILED,
    STATUS_INVALID,
    STATUS_PENDING,
    FINISHED_STATUSES,
    read_lineage,
    write_lineage,
    child_generation,
    get_failed_iterations,
    get_pruned_iterations,
)
from base_agent.src.llm.api import create_completion
from base_agent.src.llm.base import Message
from base_agent.src.types.llm_types import Model, TextContent
from search.selection import (
    DEFAULT_SELECTION_STRATEGY,
    SELECTION_STRATEGIES,
    ArchiveView,
    get_selector,
    selection_rng,
)
from search.changelog import render_other_branches, update_change_log
from runner_config import META_AGENT_CHOICES, RunnerSettings, resolve_settings
from search.gate import (
    AGENT_CODE_DIR,
    GateConfig,
    GateResult,
    check_tests,
    fix_prompt,
    run_gate,
)
from search.evaluation import (
    EvaluationConfig,
    holdout_dir,
    should_promote,
    split_problems,
    stage1_problems,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global locks
benchmark_trackers = {}
benchmark_locks = {}

# The image that agents run in. Agents made during an experiment get an image
# of their own, tagged SANDBOX_IMAGE:run{exp_id}_agent{agent_id}.
SANDBOX_IMAGE = "sica_sandbox"

# Credentials are not built into the sandbox image, but handed to each
# container when it is started. These variables are passed on if set:
SANDBOX_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "FIREWORKS_AI_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "VERTEX_PROJECT_ID",
    "VERTEX_REGION",
)
# and this file, if it exists, is mounted read-only:
GOOGLE_CREDENTIALS_FILE = Path("sandbox/GOOGLE_APPLICATION_CREDENTIALS.json")
CONTAINER_GOOGLE_CREDENTIALS = "/tmp/GOOGLE_APPLICATION_CREDENTIALS.json"

# How long a benchmark run may go over its time limit before it is cut off
TIMEOUT_GRACE_SECONDS = 10

# Limits for the meta agent when it is asked to fix an agent that failed the
# validation gate
GATE_FIX_TIMEOUT_SECONDS = 30 * 60
GATE_FIX_COST_THRESHOLD_USD = 5.0

# The agent's monitoring web server listens on this port inside the container
MONITOR_CONTAINER_PORT = 8080
# Publish it on a free host port chosen by docker (loopback only), so that any
# number of containers can run concurrently without port collisions.
MONITOR_PORT_MAPPING = f"127.0.0.1::{MONITOR_CONTAINER_PORT}"

# Utility functions ------------------------------------------------------------


def parse_meta_agent(value: str) -> str | int:
    """A --meta-agent value: one of META_AGENT_CHOICES, or an agent id"""
    if value in META_AGENT_CHOICES:
        return value
    try:
        agent_id = int(value)
    except ValueError:
        agent_id = -1
    if agent_id < 0:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not one of {', '.join(META_AGENT_CHOICES)} or an agent id"
        )
    return agent_id


def setup_argparse() -> argparse.ArgumentParser:
    """The command line. Options have no defaults here: an option that is not
    given takes its value from the config file, or else its default, both of
    which are in runner_config.RunnerSettings (see settings_from_args)."""
    parser = argparse.ArgumentParser(
        description="Self referential agent",
        epilog="Every option can also be set in a YAML file given with --config "
        "(see run.example.yaml); options on the command line take precedence.",
    )
    parser.add_argument(
        "--experiment-id", "-id", type=int, help="ID of experiment to resume"
    )
    parser.add_argument("--config", type=Path, help="YAML file with settings")

    search = parser.add_argument_group("search")
    search.add_argument(
        "--iterations", "-n", type=int, help="Number of iterations to run (default: 20)"
    )
    search.add_argument(
        "--workers", type=int, help="Number of parallel problem workers (default: 8)"
    )
    search.add_argument(
        "--children",
        type=int,
        help="Number of new agents to create per iteration, each from its own selected parent (default: 1)",
    )
    search.add_argument(
        "--selection",
        choices=SELECTION_STRATEGIES,
        help=f"How to select the agent that the next agent is built from (default: {DEFAULT_SELECTION_STRATEGY})",
    )
    search.add_argument(
        "--seed",
        type=int,
        help="Seed for parent selection (default: the experiment's stored seed, or a new random one)",
    )
    search.add_argument(
        "--meta-agent",
        type=parse_meta_agent,
        metavar="{best,parent,seed,<id>}",
        help="Which agent carries out the improvement of a new agent: the best-scoring "
        "agent (default), the new agent's parent, the initial agent, or a fixed agent id",
    )
    search.add_argument(
        "--meta-model",
        choices=[model.name for model in Model],
        metavar="MODEL",
        help="Model that the meta agent uses while improving an agent (default: the "
        "meta agent's own configured model). Benchmarks are unaffected.",
    )

    gate = parser.add_argument_group("validation gate")
    gate.add_argument(
        "--gate-checks",
        metavar="CHECKS",
        help="Checks that a new agent must pass before it is evaluated, comma separated: "
        "import (its code imports), tests (its unit tests pass, unless its parent's "
        "did not either), smoke (it solves a trivial problem). 'none' switches the "
        "gate off (default: import,tests,smoke)",
    )
    gate.add_argument(
        "--gate-retries",
        type=int,
        help="How often the meta agent is asked to fix a new agent that fails the gate (default: 0)",
    )
    gate.add_argument(
        "--gate-model",
        choices=[model.name for model in Model],
        metavar="MODEL",
        help="Model for the gate's smoke run (default: the agent's own configured model)",
    )

    evaluation = parser.add_argument_group(
        "evaluation",
        "Stored with the experiment: when resuming, options that are not given keep "
        "their stored value. The choice of problems (--subset-size, --benchmark-seed, "
        "--holdout-fraction) cannot be changed once agents have been evaluated.",
    )
    evaluation.add_argument(
        "--benchmarks",
        nargs="+",
        metavar="NAME",
        help="Benchmarks to run (default: gsm8k). Available: "
        + ", ".join(benchmark_registry),
    )
    evaluation.add_argument(
        "--subset-size",
        type=int,
        help="Number of problems to take from each benchmark; 0 for all (default: 18)",
    )
    evaluation.add_argument(
        "--benchmark-seed", type=int, help="Seed of the choice of problems (default: 1)"
    )
    evaluation.add_argument(
        "--timeout-s", type=int, help="Time limit per problem, in seconds (default: 600)"
    )
    evaluation.add_argument(
        "--cost-limit-usd", type=float, help="Cost limit per problem, in USD (default: 3)"
    )
    evaluation.add_argument(
        "--repeats",
        type=int,
        help="Number of times to run each benchmark problem (default: 1)",
    )
    evaluation.add_argument(
        "--holdout-fraction",
        type=float,
        help="Share of each benchmark's problems to keep out of the archive that agents see; "
        "can only be set when an experiment is started (default: 0)",
    )
    evaluation.add_argument(
        "--stage1-size",
        type=int,
        help="Evaluate new agents on this many problems per benchmark first, and in full "
        "only if they are not clearly worse than their parent (default: 0, always in full)",
    )
    evaluation.add_argument(
        "--promote-threshold",
        type=float,
        help="With --stage1-size: evaluate in full unless the agent is worse than its "
        "parent by more than this, at 95%% confidence (default: 0)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to perform")

    # Just run through a single benchmark for testing
    test_parser = subparsers.add_parser(
        "test", help="Just run the benchmark on the latest agent iteration"
    )
    test_parser.add_argument("--name", default="gsm8k", help="Benchmark ID to run")

    return parser


def settings_from_args(args: argparse.Namespace) -> RunnerSettings:
    """The settings of this invocation: defaults, overridden by the config
    file, overridden by the command line.

    Raises:
        ValueError: if a setting is unknown or invalid
    """
    cli = {
        name: getattr(args, name, None) for name in RunnerSettings.model_fields
    }
    return resolve_settings(args.config, cli)


def get_next_dir_number(base_dir: str | Path) -> int:
    """
    Returns the next free directory name of the form `run_{i}` in some base_dir.
    """
    base_path = Path(base_dir)
    numbered_dirs = []
    for path in base_path.iterdir():
        if path.is_dir():
            try:
                if path.name.startswith("run_"):
                    numbered_dirs.append(int(path.name.lstrip("run_")))
            except ValueError:
                continue
    # If no numbered directories exist, start with 1
    if not numbered_dirs:
        return 1
    return max(numbered_dirs) + 1


def sandbox_secret_args() -> list[str]:
    """`docker run` arguments that give a container its credentials.

    Variables are passed by name only (`-e NAME`): docker takes the value
    from this process's environment, so that it never appears on a command
    line, in a log, or in a file. Commands run later with `docker exec`
    inherit them.
    """
    args: list[str] = []
    for name in SANDBOX_ENV_VARS:
        if os.environ.get(name):
            args += ["-e", name]
    if GOOGLE_CREDENTIALS_FILE.is_file():
        args += [
            "-v",
            f"{GOOGLE_CREDENTIALS_FILE.absolute()}:{CONTAINER_GOOGLE_CREDENTIALS}:ro",
            "-e",
            f"GOOGLE_APPLICATION_CREDENTIALS={CONTAINER_GOOGLE_CREDENTIALS}",
        ]
    return args


async def run_docker_command(*args) -> tuple[bool, str, str]:
    """Run a command inside a docker container"""
    logger.debug("Running docker command:")
    logger.debug(" ".join(args))
    proc: Process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    success = True
    if proc.returncode != 0:
        logger.debug(f"Command {args[0]} failed: {stderr.decode()}")
        success = False
    return success, stdout.decode().strip(), stderr.decode().strip()


async def get_container_host_port(
    container_name: str, container_port: int = MONITOR_CONTAINER_PORT
) -> int | None:
    """Returns the host port docker assigned to a published container port,
    or None if it could not be determined."""
    success, stdout, _ = await run_docker_command(
        "docker", "port", container_name, f"{container_port}/tcp"
    )
    if not success or not stdout:
        return None
    # One line per address family, e.g. '127.0.0.1:49153' or '[::]:49153'
    try:
        return int(stdout.splitlines()[0].rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


async def wait_for_container_ready(container_name: str, timeout: float = 30):
    """Waits until a container is ready after starting it up"""
    start_time = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - start_time) < timeout:
        try:
            # Get container status in JSON format
            _, stdout, _ = await run_docker_command(
                "docker",
                "inspect",
                "--format",
                "{{json .State.Status}}",
                container_name,
            )
            status = json.loads(stdout)

            if status == "running":
                # Additional health check - try a basic command
                try:
                    await run_docker_command("docker", "exec", container_name, "ps")
                    return  # Container is truly ready
                except Exception:
                    pass  # Container not quite ready yet

        except Exception:
            pass  # Container might not exist yet

        await asyncio.sleep(0.1)  # Short delay before retry

    raise TimeoutError(
        f"Container {container_name} did not become ready within {timeout} seconds"
    )


def write_json_atomic(path: Path, data) -> None:
    """Write a JSON file so that it is either the old or the new version in
    full: a crash while writing cannot leave a truncated file behind."""
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_metadata(exp_dir: Path) -> dict:
    """Load the experiment metadata, creating if doesn't exist"""
    metadata_file = exp_dir / "metadata.json"
    if not metadata_file.exists():
        metadata = {
            "experiment_start_timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "executable": sys.executable,
            # Number of completed generations (evaluate, then create children)
            "generation": 0,
            # Agents that exist but still have to be evaluated
            "pending_agents": [0],
            "git_commit": None,
            # NOTE: the current_benchmark_idx field will be updated later, and
            # is set to -1 for improvement tasks, and 0, 1, 2, ... otherwise.
            # "current_benchmark_idx": None,
        }
        try:
            metadata["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except subprocess.CalledProcessError:
            pass

        write_json_atomic(metadata_file, metadata)
    else:
        with open(metadata_file) as f:
            metadata = json.load(f)
    return metadata


def update_metadata(exp_dir: Path, **kwargs) -> None:
    """Update specific metadata fields"""
    metadata = load_metadata(exp_dir)
    metadata.update(kwargs)
    write_json_atomic(exp_dir / "metadata.json", metadata)


def resolve_selection_seed(exp_dir: Path, seed: int | None = None) -> int:
    """The seed for parent selection in this experiment.

    An explicitly given seed wins. Otherwise the experiment keeps the seed it
    already has, so that resuming it does not change its random choices; a
    new experiment gets a random one. The seed in use is stored in the
    experiment metadata.
    """
    stored = load_metadata(exp_dir).get("selection_seed")
    if seed is None:
        seed = stored if stored is not None else random.SystemRandom().randrange(2**32)
    elif stored is not None and stored != seed:
        logger.warning(f"Selection seed changed from {stored} to {seed}")
    if seed != stored:
        update_metadata(exp_dir, selection_seed=seed)
    return seed


# Evaluation settings that decide which problems agents are run on
PROBLEM_CHOICE_SETTINGS = ("subset_size", "benchmark_seed", "holdout_fraction")


def resolve_evaluation_config(
    exp_dir: Path, seed: int, **given
) -> EvaluationConfig:
    """How agents are evaluated in this experiment.

    `given` holds evaluation settings (fields of EvaluationConfig) asked for
    in this invocation; None or absent means not asked for. Those keep the
    value stored with the experiment, or the default for a new experiment.

    The choice of problems is fixed once agents have been evaluated: with
    other problems, results would no longer be comparable between agents,
    and held-out problems could end up in the archive that agents see.

    Raises:
        ValueError: for invalid settings, unknown benchmarks, or an attempt
            to change the choice of problems
    """
    stored = load_metadata(exp_dir).get("evaluation")
    current = EvaluationConfig(**stored) if stored else EvaluationConfig(split_seed=seed)
    asked = {key: value for key, value in given.items() if value is not None}
    config = EvaluationConfig(**{**current.to_dict(), **asked})

    unknown = [name for name in config.benchmarks if name not in benchmark_registry]
    if unknown:
        raise ValueError(
            f"Unknown benchmark(s) {', '.join(unknown)}; available: {', '.join(benchmark_registry)}"
        )

    if any(exp_dir.glob("agent_*/benchmarks")):
        for name in PROBLEM_CHOICE_SETTINGS:
            old, new = getattr(current, name), getattr(config, name)
            if old != new:
                raise ValueError(
                    f"{name} of this experiment is {old} and cannot be changed "
                    f"(to {new}) now that agents have been evaluated"
                )
        if config.benchmarks != current.benchmarks:
            logger.warning(
                f"Benchmarks changed from {list(current.benchmarks)} to {list(config.benchmarks)}: "
                "overall scores of agents evaluated before and after are not comparable"
            )
    if config.to_dict() != stored:
        update_metadata(exp_dir, evaluation=config.to_dict())
    return config


async def update_benchmark_progress(
    exp_dir: Path, benchmark_name: str, problem_id: str, total_problems: int
) -> None:
    """Update the metadata to mark a problem as completed"""
    metadata_file = exp_dir / "metadata.json"

    # Use a file lock to ensure thread safety
    async with asyncio.Lock():
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Initialize benchmark progress if not exists
        if "benchmark_progress" not in metadata:
            metadata["benchmark_progress"] = {}

        if benchmark_name not in metadata["benchmark_progress"]:
            metadata["benchmark_progress"][benchmark_name] = {
                "total": total_problems,
                "completed": 0,
                "problems_completed": [],
            }

        # Ensure total is up-to-date
        metadata["benchmark_progress"][benchmark_name]["total"] = total_problems

        # Update the progress
        if (
            problem_id
            not in metadata["benchmark_progress"][benchmark_name]["problems_completed"]
        ):
            metadata["benchmark_progress"][benchmark_name]["problems_completed"].append(
                problem_id
            )
            metadata["benchmark_progress"][benchmark_name]["completed"] += 1

        # Save metadata
        write_json_atomic(metadata_file, metadata)


async def generate_contextual_summary(
    problem_statement: str,
    llm_answer: str,
    trace: str,
    score: float,
    parse_errors: str | None = None,
    answer_discussion: str | None = None,
) -> str:
    """
    Generates a summary of the long agent trace for archival and indexing purposes.

    Incorporates the original problem statement, the agent's final answer,
    score, any parsing errors and answer discussion to help better
    contextualise the trace summary.
    """
    scoring_context = (
        f"The agent's answer got a score of {score}"
        "(For binary correct/incorrect answers 0 represents a wrong answer and 1 represents a correct answer. "
        "For other problem types, a higher score is better.)"
    )

    if parse_errors:
        scoring_context += f"\nThere were issues parsing the answer: {parse_errors}"

    if answer_discussion:
        scoring_context += f"\nHere is some additional information about the answer to this problem:\n{answer_discussion}\n"

    summary_prompt = f"""Below is a trace of an agent's execution on the following problem:

Problem:
{problem_statement}

Agent's Answer:
{llm_answer}

Execution Result:
{scoring_context}

Trace:
{trace}

Please write a critical analysis of the agent's performance, taking into account both the solution process and the final outcome. Consider:
- Did the agent follow a logical approach?
- Were there unnecessary steps or inefficiencies?
- If the answer was wrong, where did the agent's reasoning fail?
- If there were parsing errors, what caused them?
- What specific improvements could make the agent more effective?

Keep your analysis concise (no more than 1-2 paragraphs max) but thorough.
"""

    summary = await create_completion(
        messages=[
            Message(
                role="system",
                content=[
                    TextContent(
                        text="You are a critical yet constructive and creative evaluator of AI agent performance."
                    )
                ],
            ),
            Message(role="user", content=[TextContent(text=summary_prompt)]),
        ],
        model=Model.GEMINI_FLASH_2,
    )

    # Assume we'll get a single text completion
    return summary.content[0].text


# New Job class for the queue -------------------------------------------------
class Job:
    """Represents a benchmark problem to be processed"""

    def __init__(
        self,
        benchmark_name: str,
        problem: Problem,
        benchmark: BaseBenchmark,
        repeat: int = 0,
        total: int | None = None,
    ):
        self.benchmark_name = benchmark_name
        self.problem = problem
        self.benchmark = benchmark
        # Which run of the problem this is, and the id its result is stored under
        self.repeat = repeat
        self.result_id = result_id(problem.problem_id, repeat)
        # Number of jobs of this benchmark in the current run
        self.total = total if total is not None else len(benchmark.problems)
        self.id = f"{benchmark_name}:{self.result_id}"

    def __str__(self):
        return self.id


# Process a single job --------------------------------------------------------
async def process_job(
    exp_dir: Path,
    agent_dir: Path,
    job: Job,
    worker_id: int,
    image: str = SANDBOX_IMAGE,
    results_exp_dir: Path | None = None,
    timeout_s: int = 600,
    cost_limit_usd: float = 3.0,
) -> bool:
    """Process a single benchmark problem job

    The results go to the agent's directory in `results_exp_dir`, which
    mirrors the layout of the experiment directory (and defaults to it).
    `timeout_s` and `cost_limit_usd` are the limits for this one problem.
    """
    TIMEOUT_SECONDS = timeout_s
    COST_THRESHOLD_USD = cost_limit_usd

    benchmark = job.benchmark
    problem = job.problem

    results_exp_dir = results_exp_dir or exp_dir
    results_agent_dir = results_exp_dir / agent_dir.name
    benchmark_dir = results_agent_dir / "benchmarks" / benchmark.name
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    benchmark_lock = benchmark_locks[job.benchmark_name]
    results_tracker = benchmark_trackers[job.benchmark_name]

    # Check if the problem is already completed
    async with benchmark_lock:
        if job.result_id in results_tracker.results:
            if results_tracker.results[job.result_id].is_complete():
                logger.info(f"Problem {job.result_id} already completed, skipping")
                return True
            else:
                logger.info(f"Re-doing incomplete problem {job.result_id}")
        else:
            results_tracker.start_problem(job.result_id, repeat=job.repeat)

    logger.info(f"Starting {job.result_id} (worker {worker_id})")

    logdir = benchmark_dir / "traces" / job.result_id

    # Create problem data directory
    dirname = re.sub(r"[^a-zA-Z0-9_]", "", f"{benchmark.name}_{job.result_id}")
    problem_data_dir = results_exp_dir / "problem_data" / dirname
    if problem_data_dir.exists():
        shutil.rmtree(problem_data_dir)
    problem_data_dir.mkdir(parents=True)

    timed_out = False
    cost_threshold_exceeded = False

    async def record_failure(reason: str, **fields) -> None:
        """Keep the reason with the problem's result. Its score stays unset,
        so that it is attempted again when the experiment is resumed."""
        async with benchmark_lock:
            results_tracker.update_problem(
                job.result_id, error=reason[-2000:], **fields
            )

    # Create unique container name
    local_id = f"agent_{uuid4().hex[:8]}"
    container_name = f"sica_{benchmark.name}_{job.result_id}_{local_id}"
    container_name = re.sub(f"[^a-zA-Z0-9_]", "", container_name)

    try:
        # Ensure answer directory exists
        logdir.mkdir(parents=True, exist_ok=True)

        # Start the container
        container_cmd = [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
        ]
        container_cmd += ["-p", MONITOR_PORT_MAPPING]
        container_cmd += sandbox_secret_args()
        # Mount the experiment directory (results/run_{i})
        container_cmd += [
            "-v",
            f"{exp_dir.absolute()}:/home/agent/{exp_dir}:rw",
        ]
        if results_exp_dir != exp_dir:
            # Results are kept outside of the experiment directory
            container_cmd += [
                "-v",
                f"{results_exp_dir.absolute()}:/home/agent/{results_exp_dir}:rw",
            ]
        # Mount the problem data directory to the workdir
        container_cmd += [
            "-v",
            f"{problem_data_dir.absolute()}:/home/agent/workdir:rw",
        ]
        container_cmd += [image, "tail", "-f", "/dev/null"]
        success, _, stderr = await run_docker_command(*container_cmd)
        if not success:
            logger.error(
                f"Failed to start container for {job.result_id}: {stderr}"
            )
            await record_failure(f"Could not start container: {stderr}")
            return False

        await wait_for_container_ready(container_name)
        monitor_port = await get_container_host_port(container_name)
        logger.info(
            f"{job.result_id} (worker {worker_id}) monitor: http://localhost:{monitor_port}"
        )

        try:
            # Setup the problem environment
            await benchmark.setup_problem(problem, problem_data_dir, container_name)

            # Write content directly using echo
            container_prompt_file = "/tmp/prompt.txt"

            # Create the file in the container with explicit error checking
            create_file_cmd = [
                "docker",
                "exec",
                "-u",
                "agent:agent",
                container_name,
                "bash",
                "-c",
                f"cat > {container_prompt_file} << 'EOF'\n{problem.statement}\nEOF",
            ]
            success, stdout, stderr = await run_docker_command(*create_file_cmd)
            if not success:
                logger.error(f"Failed to create prompt file: {stderr}")

            await asyncio.sleep(
                random.random() * 2
            )  # stagger the start of the agents to avoid API rate limit issues

            # Execute the agent command and get stdout
            agent_module = str(agent_dir / "agent_code").replace("/", ".")
            agent_cmd = ["python", "-m", agent_module, "benchmark"]
            agent_cmd += ["--prompt-file", container_prompt_file]
            agent_cmd += ["--logdir", str(logdir)]
            agent_cmd += ["--timeout", str(TIMEOUT_SECONDS)]
            agent_cmd += ["--cost-threshold", str(COST_THRESHOLD_USD)]

            # A report left by an earlier attempt must not be taken for this one's
            (logdir / USAGE_FILENAME).unlink(missing_ok=True)

            try:
                # An additional safety feature; if timeout cancellation
                # doesn't happen at the application layer, we cancel
                # the docker container
                async with asyncio.timeout(TIMEOUT_SECONDS + TIMEOUT_GRACE_SECONDS):
                    success, stdout, stderr = await run_docker_command(
                        "docker", "exec", container_name, *agent_cmd
                    )
            except asyncio.TimeoutError:
                # Timeout exceeded the grace period
                logger.warning(
                    f"Problem {job.result_id} timed out after {TIMEOUT_SECONDS} seconds"
                )
                timed_out = True
                success = False
                stderr = f"Execution timed out after {TIMEOUT_SECONDS} seconds"

                # Write timeout information to trace file
                timeout_msg = (
                    f"\n\nEXECUTION TIMED OUT AFTER {TIMEOUT_SECONDS} SECONDS\n"
                )
                trace_path = logdir / "trace.txt"
                if trace_path.exists():
                    with open(trace_path, "a") as f:
                        f.write(timeout_msg)
                else:
                    trace_path.write_text(timeout_msg)

            # A run that had to be cut off is not a failed attempt to be
            # repeated: it is scored on whatever answer it left, below.
            if not success and not timed_out:
                logger.error(f"Problem {job.result_id} failed: {stderr}")
                await record_failure(stderr or "The agent exited with an error")
                return False

            if not timed_out:
                usage = read_agent_usage(logdir, stdout)
                if usage is None:
                    raise ValueError("The agent did not report its usage")
                tokens, cached, cost, time = (
                    usage["tokens"], usage["cached_tokens"], usage["cost"], usage["time"]
                )  # fmt: skip
                percent_cached = (
                    int(cached) / int(tokens) * 100 if int(tokens) > 0 else 0
                )

                if float(cost) >= COST_THRESHOLD_USD:
                    cost_threshold_exceeded = True

                if float(time) >= TIMEOUT_SECONDS - 0.5:
                    timed_out = True
                    logger.info(
                        f"Execution of {job.result_id} gracefully timed out after {TIMEOUT_SECONDS} seconds"
                    )

                logger.info(
                    f"{job.result_id} | tokens: {tokens} (cached: {percent_cached:.2f}%), cost: ${float(cost):.4f}, duration: {float(time):.2f}s"
                )
            else:
                # The run was cut off, so it did not report what it used: the
                # time is known, tokens and cost are not
                tokens, cached, cost, time = 0, 0, 0, str(TIMEOUT_SECONDS)

            # Score the answer
            score, parse_errors, answer_discussion = await benchmark.score_problem(
                problem,
                str(problem_data_dir.absolute()),
                str(logdir.absolute()),
                container_name,
            )

            score_path = logdir / "score.txt"
            score_path.write_text(str(score))

            async with benchmark_lock:
                results_tracker.update_problem(
                    job.result_id,
                    tokens_used=int(tokens),
                    num_cached_tokens=int(cached),
                    cost_estimate=float(cost),
                    wall_time=float(time),
                    score=float(score),
                    timed_out=timed_out,
                    cost_threshold_exceeded=cost_threshold_exceeded,
                    error=None,
                )

            answer_path = logdir / "answer.txt"
            if answer_path.exists():
                llm_answer = answer_path.read_text()
            else:
                llm_answer = "No LLM answer provided"
            trace_path = logdir / "trace.txt"
            if trace_path.exists():
                trace = trace_path.read_text()
            else:
                trace = "No trace was available"

            # Add timeout information to the summary context
            timeout_context = ""
            if timed_out:
                timeout_context = f"\nNOTE: This execution was terminated after reaching the {TIMEOUT_SECONDS} second time budget limit."

            summary = await generate_contextual_summary(
                problem.statement,
                llm_answer,
                trace,
                score,
                parse_errors,
                (
                    answer_discussion + timeout_context
                    if answer_discussion
                    else timeout_context
                ),
            )

            summary_path = logdir / "summary.txt"
            summary_path.write_text(summary)

            # Update the benchmark progress in metadata
            await update_benchmark_progress(
                results_exp_dir, benchmark.name, job.result_id, job.total
            )

            return True

        except Exception as e:
            logger.error(f"Error processing problem {job.result_id}: {e}")
            await record_failure(f"{type(e).__name__}: {e}")
            return False
    finally:
        # Cleanup running docker container
        await run_docker_command("docker", "rm", "-f", container_name)

        # Cleanup problem data directory
        shutil.rmtree(problem_data_dir)
        logger.info(f"Completed {job.result_id}")


async def generate_benchmark_statistics(agent_dir: Path, benchmark_name: str) -> None:
    """Generate and save benchmark statistics"""
    benchmark_dir = agent_dir / "benchmarks" / benchmark_name
    results_tracker = BenchmarkTracker(benchmark_dir / "results.jsonl")

    # Aggregate scores and metrics across all problems to calculate overall metrics
    scores_by_problem: dict[str, list[float]] = {}
    tokens, cached, cost, time = 0, 0, 0, 0
    for result in results_tracker.results.values():
        # Default invalid score to 0
        score = result.score if result.score is not None else 0
        problem_id = base_problem_id(result.problem_id, result.repeat)
        scores_by_problem.setdefault(problem_id, []).append(score)
        if result.tokens_used:
            tokens += result.tokens_used
        if result.num_cached_tokens:
            cached += result.num_cached_tokens
        if result.cost_estimate:
            cost += result.cost_estimate
        if result.wall_time:
            time += result.wall_time

    # Each problem counts once, however often it was run
    scores = [sum(runs) / len(runs) for runs in scores_by_problem.values()]
    avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0
    aggregates = dict(
        avg_score=avg_score,
        tokens=tokens,
        num_cached_tokens=cached,
        cost=cost,
        time=time,
    )
    with open(benchmark_dir / "perf.json", "w") as f:
        f.write(json.dumps(aggregates))

    logger.info(
        f"Generated performance statistics for {benchmark_name}: avg_score={avg_score:.4f}"
    )


# Run benchmarks with job queue -----------------------------------------------
async def run_benchmarks_with_job_queue(
    exp_dir: Path,
    agent_dir: Path,
    benchmarks: list[Type[BaseBenchmark]],
    max_workers: int = 6,
    image: str = SANDBOX_IMAGE,
    results_exp_dir: Path | None = None,
    select_problems=None,
    repeats: int = 1,
    subset_size: int = 18,
    benchmark_seed: int = 1,
    timeout_s: int = 600,
    cost_limit_usd: float = 3.0,
) -> None:
    """Run multiple benchmarks concurrently using a job queue approach

    Args:
        results_exp_dir: Where results go (see process_job)
        select_problems: Given a benchmark, returns the problems of it to run
            (default: all of them)
        repeats: Number of times to run each problem
        subset_size: Number of problems to take from each benchmark (0: all)
        benchmark_seed: Seed of that choice of problems
        timeout_s: Time limit for a problem
        cost_limit_usd: Cost limit for a problem
    """
    results_exp_dir = results_exp_dir or exp_dir
    results_agent_dir = results_exp_dir / agent_dir.name
    results_exp_dir.mkdir(parents=True, exist_ok=True)

    # Create worker ID queue (used for port assignment)
    id_queue = asyncio.Queue()
    for i in range(max_workers):
        id_queue.put_nowait(i)

    # Job queue
    job_queue = asyncio.Queue()

    # Load benchmark progress from metadata
    metadata = load_metadata(results_exp_dir)
    benchmark_progress = metadata.get("benchmark_progress", {})

    # Initialize benchmarks and enqueue jobs
    benchmarks_dict = {}

    for benchmark_cls in benchmarks:
        benchmark = benchmark_cls(seed=benchmark_seed, subset_size=subset_size or None)
        logger.info(f"Initializing benchmark: {benchmark.name}")

        benchmark_dir = results_agent_dir / "benchmarks" / benchmark.name
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        # Create a global results tracker and lock for this benchmark
        benchmark_locks[benchmark.name] = asyncio.Lock()
        benchmark_trackers[benchmark.name] = BenchmarkTracker(
            benchmark_dir / "results.jsonl"
        )

        # Get previously completed problems
        completed_problems = set(
            benchmark_progress.get(benchmark.name, {}).get("problems_completed", [])
        )

        # Add problems to the queue
        problems = select_problems(benchmark) if select_problems else benchmark.problems
        total = len(problems) * repeats
        problems_to_process = 0
        for repeat in range(repeats):
            for problem in problems:
                job = Job(benchmark.name, problem, benchmark, repeat, total)
                if job.result_id not in completed_problems:
                    await job_queue.put(job)
                    problems_to_process += 1

        logger.info(
            f"Added {problems_to_process} problems from {benchmark.name} to queue"
        )
        benchmarks_dict[benchmark.name] = benchmark

    # If no jobs to process, we're done
    if job_queue.empty():
        logger.info("No problems to process, all benchmarks are complete")
        # Generate final perf reports for each benchmark
        for benchmark_name, benchmark in benchmarks_dict.items():
            await generate_benchmark_statistics(results_agent_dir, benchmark_name)
        return

    async def worker():
        worker_id = await id_queue.get()
        try:
            while True:
                try:
                    job = await job_queue.get()
                except asyncio.QueueEmpty:
                    # No more jobs
                    break

                try:
                    # Process the job
                    logger.info(f"Worker {worker_id} processing job {job}")
                    await process_job(
                        exp_dir,
                        agent_dir,
                        job,
                        worker_id,
                        image,
                        results_exp_dir,
                        timeout_s,
                        cost_limit_usd,
                    )
                except Exception as e:
                    logger.error(f"Error in worker {worker_id} processing {job}: {e}")
                finally:
                    job_queue.task_done()
        finally:
            id_queue.put_nowait(worker_id)

    # Start workers
    logger.info(f"Starting {max_workers} workers")
    workers = []
    for _ in range(max_workers):
        task = asyncio.create_task(worker())
        workers.append(task)

    # Wait for all jobs to complete
    await job_queue.join()

    # Cancel workers
    for task in workers:
        task.cancel()

    # Wait for workers to be cancelled
    await asyncio.gather(*workers, return_exceptions=True)

    # Generate final perf reports for each benchmark
    for benchmark_name, benchmark in benchmarks_dict.items():
        await generate_benchmark_statistics(results_agent_dir, benchmark_name)


def agent_image_tag(exp_id: int, agent_id: int) -> str:
    return f"{SANDBOX_IMAGE}:run{exp_id}_agent{agent_id}"


async def resolve_agent_image(exp_id: int, agent_id: int) -> str:
    """The image to run an agent in: its own if one was built, else the base
    image (the initial agent, and agents from before per-agent images)."""
    tag = agent_image_tag(exp_id, agent_id)
    exists, _, _ = await run_docker_command("docker", "image", "inspect", tag)
    return tag if exists else SANDBOX_IMAGE


async def build_agent_image(tag: str, agent_code_dir: Path) -> tuple[bool, str]:
    """Build the sandbox image for an agent, installing the requirements.txt
    of that agent's code. Returns (success, stderr).

    No credentials go into the image: see sandbox_secret_args.
    """
    target_arch = "x86_64"  # Default architecture
    # Detect if the script is running on a Mac with Apple Silicon
    if platform.system() == "Darwin" and platform.processor() == "arm":
        target_arch = "aarch64"
    build_cmd = ["docker", "buildx", "build"]
    build_cmd += ["--build-context", f"base_agent={agent_code_dir.absolute()}"]
    build_cmd += ["--build-arg", f"TARGET_ARCH={target_arch}"]
    build_cmd += ["-f", "sandbox/Dockerfile", "-t", tag, "--load", "sandbox"]
    success, _, stderr = await run_docker_command(*build_cmd)
    return success, stderr


def read_agent_usage(logdir: Path, stdout: str) -> dict | None:
    """What an agent run consumed: from the usage report in its log directory
    or, for agents from before that existed, the last line of its output."""
    return read_usage(logdir) or parse_usage(stdout)


def parse_usage(stdout: str) -> dict | None:
    """The usage that an agent reports on the last line of its output, as
    'tokens|cached tokens|cost|duration', or None if it is not there."""
    try:
        tokens, cached, cost, time = stdout.splitlines()[-1].split("|")
        return {
            "tokens": int(tokens),
            "cached_tokens": int(cached),
            "cost": float(cost),
            "time": float(time),
        }
    except (IndexError, ValueError):
        return None


# Validation gate --------------------------------------------------------------


@asynccontextmanager
async def gate_container(exp_id: int, agent_id: int, image: str, agent_code_dir: Path):
    """A throwaway container holding a copy of an agent's code at
    AGENT_CODE_DIR. Yields a function that runs a command in it."""
    name = f"sica_run_{exp_id}_agent_{agent_id}_gate_{uuid4().hex[:8]}"
    try:
        # The code is copied rather than mounted read-write: the checks leave
        # caches and other files behind, which must not end up in the agent
        success, _, stderr = await run_docker_command(
            "docker", "run", "--rm", "-d", "--name", name,
            *sandbox_secret_args(),
            "-v", f"{agent_code_dir.absolute()}:/mnt/agent_code:ro",
            image, "tail", "-f", "/dev/null",
        )  # fmt: skip
        if not success:
            raise RuntimeError(f"Could not start the validation container: {stderr}")
        await wait_for_container_ready(name)
        success, _, stderr = await run_docker_command(
            "docker", "exec", name, "cp", "-r", "/mnt/agent_code", AGENT_CODE_DIR
        )
        if not success:
            raise RuntimeError(f"Could not copy the agent code for validation: {stderr}")

        async def exec(
            *cmd: str, env: dict[str, str] | None = None, timeout: float | None = None
        ) -> tuple[bool, str, str]:
            docker_cmd = ["docker", "exec"]
            for key, value in (env or {}).items():
                docker_cmd += ["-e", f"{key}={value}"]
            try:
                async with asyncio.timeout(timeout):
                    return await run_docker_command(*docker_cmd, name, *cmd)
            except asyncio.TimeoutError:
                return False, "", f"Timed out after {timeout} seconds"

        yield exec
    finally:
        await run_docker_command("docker", "rm", "-f", name)


async def agent_tests_pass(exp_id: int, exp_dir: Path, agent_id: int) -> bool | None:
    """Whether an agent's own unit tests pass in its sandbox. Taken from its
    lineage record if known, and otherwise run now and recorded."""
    agent_dir = exp_dir / f"agent_{agent_id}"
    known = (read_lineage(agent_dir) or {}).get("tests_passed")
    if known is not None:
        return known
    image = await resolve_agent_image(exp_id, agent_id)
    async with gate_container(exp_id, agent_id, image, agent_dir / "agent_code") as exec:
        check = await check_tests(exec)
    write_lineage(agent_dir, tests_passed=check.raw_passed)
    return check.raw_passed


async def validate_agent(
    exp_id: int, exp_dir: Path, agent_id: int, gate: GateConfig, image: str
) -> GateResult:
    """Run the validation gate on an agent, in the given image."""
    agent_dir = exp_dir / f"agent_{agent_id}"
    parent_id = (read_lineage(agent_dir) or {}).get("parent_id")

    async def parent_tests_pass() -> bool | None:
        if parent_id is None:
            return None
        return await agent_tests_pass(exp_id, exp_dir, parent_id)

    async with gate_container(exp_id, agent_id, image, agent_dir / "agent_code") as exec:
        return await run_gate(exec, gate, parent_tests_pass)


async def check_initial_agent(exp_id: int, exp_dir: Path, gate: GateConfig) -> None:
    """Run the validation gate on the initial agent, once per experiment.

    If even the initial agent does not get through, the setup is at fault
    (image, API keys) and every agent made from it would be thrown out, each
    at the cost of a meta-improvement step. Its unit tests are the exception:
    they set the standard for its descendants rather than having to pass.

    Raises:
        RuntimeError: if the initial agent does not pass the gate
    """
    agent_dir = exp_dir / "agent_0"
    if not gate.enabled or (read_lineage(agent_dir) or {}).get("gate") is not None:
        return
    logger.info("Checking that the initial agent passes the validation gate")

    async def no_standard_yet() -> bool:
        return False

    image = await resolve_agent_image(exp_id, 0)
    async with gate_container(exp_id, 0, image, agent_dir / "agent_code") as exec:
        result = await run_gate(exec, gate, no_standard_yet)
    if not result.passed:
        raise RuntimeError(
            "The initial agent does not pass the validation gate, so no agent made "
            "from it would. Check the setup (is the sandbox image built, are the API "
            "keys exported?), or switch the gate off with --gate-checks none.\n"
            f"{result.failure_reason()}"
        )
    write_lineage(agent_dir, gate=result.to_dict(), tests_passed=result.tests_passed)
    if result.tests_passed is False:
        logger.warning(
            "The unit tests of the initial agent fail in the sandbox; new agents "
            "will not be required to pass theirs"
        )


def set_aside(exp_dir: Path, agent_dir: Path) -> None:
    """Move the remains of an interrupted meta-improvement out of the way
    (to `_interrupted/`), so that the agent can be created afresh."""
    target_root = exp_dir / "_interrupted"
    target_root.mkdir(exist_ok=True)
    target = target_root / f"{agent_dir.name}_{uuid4().hex[:8]}"
    logger.warning(f"Setting aside incomplete {agent_dir} as {target}")
    shutil.move(str(agent_dir), str(target))


async def run_meta_agent_benchmark(
    exp_id: int, exp_dir: Path, child: dict, gate: GateConfig = GateConfig()
) -> None:
    """Create a new agent version from an archived one

    We can conceptually think of this as a 'meta agent improvement' benchmark.

    Args:
        exp_id: The experiment id
        exp_dir: The experiment directory (results/run_{exp_id})
        child: The planned agent (see plan_generation): its `id`, the
            `parent_id` whose code it starts from, the `meta_agent_id` that
            makes the changes (using `meta_model`, if given), and the
            `selection` record of the parent choice
        gate: The validation that the new agent has to pass to be usable
    """
    child_id: int = child["id"]
    parent_id: int = child["parent_id"]
    meta_agent_id: int = child["meta_agent_id"]
    meta_model: str | None = child.get("meta_model")
    child_dir = exp_dir / f"agent_{child_id}"
    child_code_dir = child_dir / "agent_code"
    parent_code_dir = exp_dir / f"agent_{parent_id}" / "agent_code"
    logger.info(
        f"Creating agent {child_id} from agent {parent_id} (meta agent: {meta_agent_id})"
    )

    if child_dir.exists():
        existing = read_lineage(child_dir) or {}
        if existing.get("status") in FINISHED_STATUSES:
            raise ValueError(f"Agent {child_id} has already been created")
        set_aside(exp_dir, child_dir)
    child_dir.mkdir(parents=True)

    # Record where the new agent comes from. It stays 'pending' until the meta
    # agent has run, and is marked 'failed' if it does not complete, or
    # 'invalid' if what it made cannot run.
    write_lineage(
        child_dir,
        id=child_id,
        parent_id=parent_id,
        meta_agent_id=meta_agent_id,
        meta_model=meta_model,
        generation=child_generation(exp_dir, parent_id),
        selection=child.get("selection"),
        status=STATUS_PENDING,
    )
    status = STATUS_FAILED
    failure_reason: str | None = "Meta-improvement did not complete"
    interrupted = False

    # Create a unique container name
    local_id = f"agent_{uuid4().hex[:8]}"
    container_name = f"sica_run_{exp_id}_agent_{child_id}_improvement_{local_id}"
    container_name = re.sub(f"[^a-zA-Z0-9_]", "", container_name)

    try:
        # 1. Copy the parent's code as the starting point. Its change log
        # comes along, and describes exactly the changes present in this code.
        shutil.copytree(parent_code_dir, child_code_dir)

        # 2. What was tried on other branches goes in a generated section
        update_change_log(
            child_code_dir / "agent_change_log.md",
            render_other_branches(ArchiveAnalyzer(exp_dir), parent_id),
        )

        # Setup meta-improvement logs directory
        meta_log_dir = child_dir / "meta_improvement_logs"
        container_meta_log_path = f"/home/agent/meta_improvement_logs_{child_id}"
        meta_log_dir.mkdir(parents=True, exist_ok=True)

        # 3. Start the container, from the image of the agent that will run
        image = await resolve_agent_image(exp_id, meta_agent_id)
        container_cmd = ["docker", "run", "--rm", "-d", "--name", container_name]
        # Publish the monitoring server on a free host port
        container_cmd += ["-p", MONITOR_PORT_MAPPING]
        container_cmd += sandbox_secret_args()
        # Mount the full agent archive as read-only
        container_cmd += ["-v", f"{exp_dir.absolute()}:/home/agent/archive:ro"]
        # Mount the new agent directory as the workdir
        container_cmd += ["-v", f"{child_code_dir.absolute()}:/home/agent/workdir:rw"]
        # Mount meta agent log directory
        container_cmd += ["-v", f"{meta_log_dir.absolute()}:{container_meta_log_path}:rw"]
        container_cmd += [image, "tail", "-f", "/dev/null"]
        success, _, stderr = await run_docker_command(*container_cmd)
        if not success:
            logger.error(f"Could not start container: {stderr}")
            failure_reason = f"Could not start container: {stderr[-2000:]}"
            return

        await wait_for_container_ready(container_name)
        monitor_port = await get_container_host_port(container_name)
        logger.info(
            f"Meta-improvement monitor (agent {child_id}): http://localhost:{monitor_port}"
        )

        # 4. Execute the (meta) agent command and get stdout
        agent_module = f"archive.agent_{meta_agent_id}.agent_code"
        agent_cmd = ["python", "-m", agent_module, "improve"]
        agent_cmd += ["--workdir", "/home/agent/workdir"]
        agent_cmd += ["--logdir", container_meta_log_path]
        # The agent whose code is in the workdir, and the agent being created
        agent_cmd += ["--best-iter", str(parent_id)]
        agent_cmd += ["--current-iter", str(child_id)]
        exec_cmd = ["docker", "exec"]
        if meta_model:
            # Read by the agent's settings (AGENT_ prefix), for this run only
            exec_cmd += ["-e", f"AGENT_MODEL={meta_model}"]
        success, stdout, stderr = await run_docker_command(
            *exec_cmd, container_name, *agent_cmd
        )

        if not success:
            logger.error(f"Meta agent failed creating agent {child_id}: {stderr}")
            logger.error(stdout)
            failure_reason = f"Meta agent exited with an error: {stderr[-2000:]}"
            return

        # What it took to create this agent, for cost accounting
        usage = read_agent_usage(meta_log_dir, stdout)
        if usage is None:
            logger.warning(f"Meta agent reported no usage for agent {child_id}")
            usage = {"tokens": 0, "cached_tokens": 0, "cost": 0.0, "time": 0.0}
        else:
            write_lineage(child_dir, usage=usage)
        tokens, cached, cost, time = (
            usage["tokens"], usage["cached_tokens"], usage["cost"], usage["time"]
        )  # fmt: skip
        percent_cached = cached / tokens * 100 if tokens > 0 else 0
        logger.info(
            f"meta-improvement (agent {child_id}) || tokens: {tokens} (cached: {percent_cached:.2f}%), cost: ${cost:.4f}, duration: {time:.2f}s"
        )

        fix_attempts = 0
        while True:
            # 5. Build the new agent's image, with its (possibly changed)
            # requirements.txt. An agent whose image does not build cannot run.
            image_tag = agent_image_tag(exp_id, child_id)
            built, build_stderr = await build_agent_image(image_tag, child_code_dir)
            if not built:
                logger.error(f"Image build failed for agent {child_id}: {build_stderr}")
                failure_reason = f"Image build failed: {build_stderr[-2000:]}"
                return

            # 6. Validate it: an agent that cannot run is not worth evaluating
            if not gate.enabled:
                break
            result = await validate_agent(exp_id, exp_dir, child_id, gate, image_tag)
            write_lineage(
                child_dir,
                gate={**result.to_dict(), "fix_attempts": fix_attempts},
                tests_passed=result.tests_passed,
            )
            if result.passed:
                break
            logger.error(f"Agent {child_id}: {result.failure_reason()}")
            if fix_attempts >= gate.retries:
                status = STATUS_INVALID
                failure_reason = result.failure_reason()
                return

            # Ask the meta agent to repair what it made, then build and
            # validate again
            fix_attempts += 1
            logger.info(
                f"Asking the meta agent to fix agent {child_id} (attempt {fix_attempts} of {gate.retries})"
            )
            fix_cmd = ["python", "-m", f"{agent_module}.agent"]
            fix_cmd += ["--prompt", fix_prompt(result)]
            fix_cmd += ["--workdir", "/home/agent/workdir"]
            fix_cmd += ["--logdir", f"{container_meta_log_path}/gate_fix_{fix_attempts}"]
            fix_cmd += ["--timeout", str(GATE_FIX_TIMEOUT_SECONDS)]
            fix_cmd += ["--cost-threshold", str(GATE_FIX_COST_THRESHOLD_USD)]
            fixed, _, fix_stderr = await run_docker_command(
                *exec_cmd, container_name, *fix_cmd
            )
            if not fixed:
                logger.warning(f"The fix attempt for agent {child_id} errored: {fix_stderr}")
            # The fix is part of what it took to create this agent
            fix_usage = read_usage(meta_log_dir / f"gate_fix_{fix_attempts}")
            if fix_usage is not None:
                usage = {key: usage[key] + fix_usage[key] for key in usage}
                write_lineage(child_dir, usage=usage)

        # The agent is usable; anything that goes wrong from here on
        # (summaries) does not invalidate it.
        status, failure_reason = STATUS_OK, None

        # 7. Generate and save a summary for the meta-improvement logs
        trace_path = meta_log_dir / "execution_tree.txt"
        if trace_path.exists():
            trace = trace_path.read_text()
        else:
            trace = "No trace was available"

        # Define a simple problem statement and answer for context
        problem_statement = (
            f"Improve the agent from iteration {parent_id} to create iteration {child_id}"
        )
        llm_answer = "Agent code updated"
        score = 1.0  # Binary success/failure

        # Additional context for the summary
        meta_context = (
            f"The 'answer' actually pertains to whether the agent was successful (1) or errored out (0) during the meta-improvement task"
            f"Meta-improvement process used {tokens} tokens ({percent_cached:.2f}% cached), "
            f"cost ${cost:.4f}, and took {time:.2f} seconds."
        )
        if stderr:
            meta_context += f"\nErrors encountered: {stderr}"

        summary = await generate_contextual_summary(
            problem_statement=problem_statement,
            llm_answer=llm_answer,
            trace=trace,
            score=score,
            parse_errors=None,
            answer_discussion=meta_context,
        )

        summary_path = meta_log_dir / "summary.txt"
        summary_path.write_text(summary)
        logger.info(f"Meta-improvement summary saved to {summary_path}")

    except Exception as e:
        if status != STATUS_OK:
            failure_reason = f"{type(e).__name__}: {e}"
        raise

    except BaseException:
        # The runner itself is being stopped (Ctrl-C, cancellation)
        interrupted = True
        raise

    finally:

        # Cleanup running docker container
        await run_docker_command("docker", "rm", "-f", container_name)

        if status == STATUS_OK:
            write_lineage(child_dir, status=status)
        elif interrupted:
            # Left 'pending', so that it is created afresh on resumption
            logger.warning(f"Creation of agent {child_id} was interrupted")
        else:
            logger.error(f"Marking agent {child_id} as {status}: {failure_reason}")
            write_lineage(child_dir, status=status, failure_reason=failure_reason)


# Generations -----------------------------------------------------------------


def load_search_state(exp_dir: Path) -> dict:
    """The experiment metadata, with the search state fields present.

    Experiments from before generations only stored `agent_iteration`: the
    agent that was being evaluated, all agents up to it being complete. Those
    agents may have no lineage record, and get one here (of unknown ancestry).
    """
    metadata = load_metadata(exp_dir)
    if "pending_agents" not in metadata:
        current = metadata.get("agent_iteration", 0)
        for agent_id in get_agent_ids(exp_dir):
            agent_dir = exp_dir / f"agent_{agent_id}"
            if agent_id <= current and read_lineage(agent_dir) is None:
                write_lineage(agent_dir, id=agent_id, status=STATUS_OK)
        update_metadata(exp_dir, generation=current, pending_agents=[current])
        metadata = load_metadata(exp_dir)
    return metadata


def get_agent_ids(exp_dir: Path) -> list[int]:
    """The ids of all agent directories in an experiment"""
    ids = []
    for agent_dir in exp_dir.glob("agent_*"):
        try:
            ids.append(int(agent_dir.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids)


def set_aside_incomplete_agents(exp_dir: Path) -> None:
    """Set aside agent directories left behind by an interrupted
    meta-improvement step: those never recorded as finished."""
    for agent_id in get_agent_ids(exp_dir):
        agent_dir = exp_dir / f"agent_{agent_id}"
        record = read_lineage(agent_dir) or {}
        if record.get("status") not in FINISHED_STATUSES:
            set_aside(exp_dir, agent_dir)


def check_meta_agent(exp_dir: Path, meta_agent: str | int) -> None:
    """Raises ValueError if a fixed meta agent cannot do the job: it has to
    exist, and to have been created successfully."""
    if isinstance(meta_agent, str):
        if meta_agent not in META_AGENT_CHOICES:
            raise ValueError(f"Unknown meta agent {meta_agent!r}")
        return
    if meta_agent not in get_agent_ids(exp_dir):
        raise ValueError(f"Meta agent {meta_agent} does not exist in {exp_dir}")
    if meta_agent in get_failed_iterations(exp_dir):
        raise ValueError(f"Meta agent {meta_agent} was not created successfully")


def plan_generation(
    exp_dir: Path,
    generation: int,
    n_children: int,
    selection: str = DEFAULT_SELECTION_STRATEGY,
    seed: int = 0,
    meta_agent: str | int = "best",
    meta_model: str | None = None,
) -> list[dict]:
    """Decide which agents to create in a generation, and from which parents.

    `meta_agent` says which agent carries out each improvement: 'best' (the
    best-scoring agent), 'parent' (the new agent's own parent), 'seed' (the
    initial agent) or a fixed agent id. `meta_model` optionally overrides the
    model that it uses.

    The plan is made once per generation and stored, so that a resumed
    experiment creates the same agents as an uninterrupted one.
    """
    # Agents whose own meta-improvement step failed are never built upon, nor
    # used as the meta agent. Nor are agents that were dropped after the first
    # evaluation stage: they were clearly worse than their parent, and their
    # scores on a subset of the problems are not comparable with the others'.
    failed = get_failed_iterations(exp_dir) | get_pruned_iterations(exp_dir)
    analyzer = ArchiveAnalyzer(exp_dir)
    agent_ids = get_agent_ids(exp_dir) or [0]

    archive_view = ArchiveView.from_analyzer(
        analyzer, max(agent_ids), score_type="mean_score", excluded=failed
    )
    selected = get_selector(selection).select(
        archive_view, k=n_children, rng=selection_rng(seed, generation)
    )

    check_meta_agent(exp_dir, meta_agent)
    try:
        best_agent_id = analyzer.get_best_agent_iteration(exclude=failed)
    except Exception:
        best_agent_id = 0

    def meta_agent_for(parent_id: int) -> int:
        if meta_agent == "best":
            return best_agent_id
        if meta_agent == "parent":
            return parent_id
        if meta_agent == "seed":
            return 0
        return int(meta_agent)

    weights = {str(i): w for i, w in selected.weights.items()}
    next_id = max(agent_ids) + 1
    return [
        {
            "id": next_id + index,
            "parent_id": parent_id,
            "meta_agent_id": meta_agent_for(parent_id),
            "meta_model": meta_model,
            # Enough to reproduce the choice of parent: it is
            # select(k=n_children, rng=selection_rng(seed, generation)).parents[index]
            "selection": {
                "strategy": selection,
                "seed": seed,
                "generation": generation,
                "n_children": n_children,
                "index": index,
                "weights": weights,
            },
        }
        for index, parent_id in enumerate(selected.parents)
    ]


async def evaluate_agent(
    exp_id: int,
    exp_dir: Path,
    agent_id: int,
    workers: int,
    config: EvaluationConfig,
) -> None:
    """Run the benchmarks for one agent.

    With stages, the agent is first run on a few problems per benchmark, and
    evaluated in full only if that does not show it to be clearly worse than
    its parent. With a holdout fraction, the held-out problems are run last,
    with their results kept outside of the experiment directory.
    """
    agent_dir = exp_dir / f"agent_{agent_id}"
    unknown = [name for name in config.benchmarks if name not in benchmark_registry]
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown)}")
    benchmarks = [benchmark_registry[name] for name in config.benchmarks]
    image = await resolve_agent_image(exp_id, agent_id)
    seed = config.split_seed
    # How every benchmark run of this agent is made
    run_settings = dict(
        image=image,
        repeats=config.repeats,
        subset_size=config.subset_size,
        benchmark_seed=config.benchmark_seed,
        timeout_s=config.timeout_s,
        cost_limit_usd=config.cost_limit_usd,
    )

    def dev(benchmark: BaseBenchmark) -> list[Problem]:
        return split_problems(
            benchmark.problems, benchmark.name, config.holdout_fraction, seed
        )[0]

    def holdout(benchmark: BaseBenchmark) -> list[Problem]:
        return split_problems(
            benchmark.problems, benchmark.name, config.holdout_fraction, seed
        )[1]

    def stage1(benchmark: BaseBenchmark) -> list[Problem]:
        return stage1_problems(dev(benchmark), benchmark.name, config.stage1_size, seed)

    record = read_lineage(agent_dir) or {}
    parent_id = record.get("parent_id")
    evaluation = record.get("evaluation") or {}

    # First stage, unless its outcome was already decided (when resuming)
    if config.stage1_size > 0 and parent_id is not None and "promoted" not in evaluation:
        logger.info(f"Running first-stage benchmarks for agent {agent_id}")
        await run_benchmarks_with_job_queue(
            exp_dir,
            agent_dir,
            benchmarks,
            workers,
            select_problems=stage1,
            **run_settings,
        )
        delta = ArchiveAnalyzer(exp_dir).paired_delta(agent_id, parent_id)
        evaluation = {
            "promoted": should_promote(delta, config.promote_threshold),
            "promote_threshold": config.promote_threshold,
            "stage1_delta": (
                None
                if delta is None
                else {
                    "mean": delta.mean,
                    "ci_lower": delta.ci_lower,
                    "ci_upper": delta.ci_upper,
                    "n_problems": delta.n_problems,
                }
            ),
        }
        write_lineage(agent_dir, evaluation=evaluation)
        logger.info(
            f"Agent {agent_id} vs parent {parent_id} after the first stage: {delta}"
        )

    if evaluation.get("promoted") is False:
        logger.warning(
            f"Agent {agent_id} is clearly worse than its parent; not evaluating it in full"
        )
        return

    logger.info(f"Running benchmarks for agent {agent_id}")
    await run_benchmarks_with_job_queue(
        exp_dir,
        agent_dir,
        benchmarks,
        workers,
        select_problems=dev,
        **run_settings,
    )
    if config.holdout_fraction > 0:
        logger.info(f"Running held-out benchmarks for agent {agent_id}")
        await run_benchmarks_with_job_queue(
            exp_dir,
            agent_dir,
            benchmarks,
            workers,
            results_exp_dir=holdout_dir(exp_dir),
            select_problems=holdout,
            **run_settings,
        )


async def evaluate_pending_agents(
    exp_id: int,
    exp_dir: Path,
    workers: int,
    config: EvaluationConfig = EvaluationConfig(),
) -> None:
    """Run the benchmarks for every agent that has not been evaluated yet"""
    pending: list[int] = list(load_metadata(exp_dir).get("pending_agents", []))
    while pending:
        agent_id = pending[0]
        if agent_id in get_failed_iterations(exp_dir):
            # Its meta-improvement step failed, so the code is in an unknown
            # state: don't spend benchmark budget on it.
            logger.warning(f"Agent {agent_id} is marked as failed; skipping its benchmarks")
        else:
            await evaluate_agent(exp_id, exp_dir, agent_id, workers, config)
        pending = pending[1:]
        # Important: reset benchmark progress for the next agent
        update_metadata(exp_dir, pending_agents=pending, benchmark_progress=dict())
        if config.holdout_fraction > 0 and holdout_dir(exp_dir).exists():
            update_metadata(holdout_dir(exp_dir), benchmark_progress=dict())


async def run_search(
    exp_id: int,
    exp_dir: Path,
    iterations: int,
    workers: int,
    n_children: int = 1,
    selection: str = DEFAULT_SELECTION_STRATEGY,
    seed: int = 0,
    config: EvaluationConfig = EvaluationConfig(),
    meta_agent: str | int = "best",
    meta_model: str | None = None,
    gate: GateConfig = GateConfig(),
) -> None:
    """The self-improvement loop. In each generation: evaluate the agents that
    have not been evaluated yet, then create `n_children` new agents."""
    start_generation = load_search_state(exp_dir)["generation"]
    logger.info(f"Starting from generation {start_generation}")
    if start_generation < iterations:
        await check_initial_agent(exp_id, exp_dir, gate)

    for generation in range(start_generation, iterations):
        await evaluate_pending_agents(exp_id, exp_dir, workers, config)

        # Plan the new agents, unless resuming a generation already planned
        plan = load_metadata(exp_dir).get("planned_children")
        if not plan:
            set_aside_incomplete_agents(exp_dir)
            plan = plan_generation(
                exp_dir, generation, n_children, selection, seed, meta_agent, meta_model
            )
            update_metadata(exp_dir, planned_children=plan)

        def is_created(child: dict) -> bool:
            record = read_lineage(exp_dir / f"agent_{child['id']}") or {}
            return record.get("status") in FINISHED_STATUSES

        todo = [child for child in plan if not is_created(child)]
        logger.info(
            f"Generation {generation}: creating agents {[c['id'] for c in todo]} "
            f"from parents {[c['parent_id'] for c in todo]}"
        )
        results = await asyncio.gather(
            *(run_meta_agent_benchmark(exp_id, exp_dir, child, gate) for child in todo),
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        for child, result in zip(todo, results):
            if isinstance(result, BaseException):
                logger.error(f"Creating agent {child['id']} raised {result!r}")
        interruption = next((e for e in errors if not isinstance(e, Exception)), None)
        if interruption is not None:
            raise interruption
        if errors and len(errors) == len(todo):
            # Nothing worked: more likely a broken setup than bad agents
            raise errors[0]

        update_metadata(
            exp_dir,
            generation=generation + 1,
            pending_agents=[child["id"] for child in plan],
            planned_children=None,
            benchmark_progress=dict(),
        )
        logger.info(f"Completed generation {generation}")


def initialise_experiment(exp_dir: Path) -> None:
    """Create a new experiment directory, with the base agent as agent_0"""
    exp_dir.mkdir(parents=True)
    load_metadata(exp_dir)  # Creates if doesn't exist

    # Copy the initial agent
    agent_0_dir = exp_dir / "agent_0"
    agent_0_dir.mkdir()
    shutil.copytree("base_agent", agent_0_dir / "agent_code")
    write_lineage(
        agent_0_dir,
        id=0,
        parent_id=None,
        meta_agent_id=None,
        generation=0,
        status=STATUS_OK,
    )


async def main():
    """Main entry point for the agent runner"""
    parser = setup_argparse()
    args = parser.parse_args()
    logger.debug(f"Setup parser: {args}")
    try:
        settings = settings_from_args(args)
    except ValueError as e:
        parser.error(str(e))

    # Either get the specified experiment id, or get the next one
    exp_id = args.experiment_id or get_next_dir_number("results")
    exp_dir = Path("results") / f"run_{exp_id}"

    # Setup experiment directory
    if not exp_dir.exists():
        initialise_experiment(exp_dir)
        logger.info(f"Initialized experiment {exp_id} in {exp_dir}")

    selection_seed = resolve_selection_seed(exp_dir, settings.seed)
    try:
        config = resolve_evaluation_config(
            exp_dir,
            selection_seed,
            **{
                name: getattr(settings, name)
                for name in EvaluationConfig.__dataclass_fields__
                if name in RunnerSettings.model_fields
            },
        )
        check_meta_agent(exp_dir, settings.meta_agent)
        gate = settings.gate
    except ValueError as e:
        parser.error(str(e))

    # Handle test mode
    if args.command == "test":
        # Find the latest agent iteration
        latest_agent = max(
            (d for d in exp_dir.iterdir() if d.name.startswith("agent_")),
            key=lambda d: int(d.name.split("_")[1]),
        )

        # Get the benchmark class
        benchmark_cls = benchmark_registry.get(args.name)
        if not benchmark_cls:
            logger.error(f"Benchmark {args.name} not found")
            return

        # Run the specified benchmark using the job queue approach
        await run_benchmarks_with_job_queue(
            exp_dir,
            latest_agent,
            [benchmark_cls],
            settings.workers,
            image=await resolve_agent_image(
                exp_id, int(latest_agent.name.split("_")[1])
            ),
            repeats=config.repeats,
            subset_size=config.subset_size,
            benchmark_seed=config.benchmark_seed,
            timeout_s=config.timeout_s,
            cost_limit_usd=config.cost_limit_usd,
        )
        return

    passed_on = [name for name in SANDBOX_ENV_VARS if os.environ.get(name)]
    if passed_on:
        logger.info(f"Passed on to the sandbox: {', '.join(passed_on)}")
    else:
        logger.warning(
            "None of the provider API keys are set in this shell, so agents will "
            f"not be able to call any model ({', '.join(SANDBOX_ENV_VARS)})"
        )

    # Keep a record of what this invocation was run with
    update_metadata(
        exp_dir, runner_settings={**settings.model_dump(), "seed": selection_seed}
    )
    logger.info(
        f"Parent selection: {settings.selection} (seed {selection_seed}), "
        f"{settings.children} new agent(s) per iteration"
    )
    logger.info(f"Evaluation: {config}")
    logger.info(
        f"Meta agent: {settings.meta_agent}, model: {settings.meta_model or 'its own default'}"
    )
    logger.info(f"Validation gate: {gate}")
    await run_search(
        exp_id,
        exp_dir,
        settings.iterations,
        settings.workers,
        n_children=settings.children,
        selection=settings.selection,
        seed=selection_seed,
        config=config,
        meta_agent=settings.meta_agent,
        meta_model=settings.meta_model,
        gate=gate,
    )


if __name__ == "__main__":
    asyncio.run(main())
