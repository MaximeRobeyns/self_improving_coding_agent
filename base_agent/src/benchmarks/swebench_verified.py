# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import random
import logging
import asyncio
import subprocess

from datasets import load_dataset
from pathlib import Path
from dataclasses import dataclass

from .base import BaseBenchmark, Problem

# The SWE-bench harness is imported where it is used (setup_problem), not
# here: this module is imported whenever any benchmark is, and the harness is
# a large dependency whose layout differs between versions. Only running this
# benchmark needs it.

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class SWEBenchExample:
    """A single SWE-bench example."""

    repo: str
    instance_id: str
    base_commit: str
    environment_setup_commit: str
    problem_statement: str
    patch: str
    test_patch: str
    hints_text: str
    fail_to_pass: str
    pass_to_pass: str
    version: str = None

    def to_dict(self) -> dict:
        """Convert to the format expected by SWEbenchInstance TypedDict."""
        return {
            "repo": self.repo,
            "instance_id": self.instance_id,
            "base_commit": self.base_commit,
            "environment_setup_commit": self.environment_setup_commit,
            "problem_statement": self.problem_statement,
            "patch": self.patch,
            "test_patch": self.test_patch,
            "hints_text": self.hints_text,
            "FAIL_TO_PASS": self.fail_to_pass,
            "PASS_TO_PASS": self.pass_to_pass,
            "version": self.version,
            "created_at": "",  # Add a default empty string since it's required by TypedDict
        }

    @classmethod
    def from_row(cls, row: dict) -> "SWEBenchExample":
        """Create a SWEBenchExample from a dataset row."""
        return cls(
            repo=row.get("repo"),
            instance_id=row.get("instance_id"),
            base_commit=row.get("base_commit"),
            environment_setup_commit=row.get("environment_setup_commit"),
            problem_statement=row.get("problem_statement"),
            patch=row.get("patch"),
            test_patch=row.get("test_patch"),
            hints_text=row.get("hints_text"),
            fail_to_pass=row.get("FAIL_TO_PASS"),
            pass_to_pass=row.get("PASS_TO_PASS"),
            version=row.get("version"),
        )

    def format_problem(self, allow_hints: bool = True) -> str:
        """Format the example as a problem statement."""
        lines = []

        lines.append(f"You are an expert maintainer of the `{self.repo}` project")
        lines.append(f"There has just been a github issue that you need to resolve")
        lines.append(f"Issue:\n\n{self.problem_statement}")

        if self.hints_text and allow_hints:
            lines.append(f"\n\n{self.hints_text}")

        lines.append(f"\n\nMake sure that this problem is fixed.")
        lines.append(
            f"Everything should already have been cloned and set up for you in your current work directory."
        )
        lines.append(
            f"Work at pace and try to be efficient. Don't spend too much time trying to get tests to work if they're not working immediately."
        )

        return "\n".join(lines)

    def format_problem_full(self, allow_hints: bool = True) -> str:
        """Format the example as a problem statement (without manual setup)."""
        dirname = self.repo.split("/")[-1]
        lines = []

        lines.append(f"You are an expert maintainer of the `{self.repo}` project")
        lines.append(f"There has just been a github issue that you need to resolve")
        lines.append(f"Issue:\n\n{self.problem_statement}")

        if self.hints_text and allow_hints:
            lines.append(f"\n\n{self.hints_text}")

        lines.append(f"\n\nSetup steps to reproduce that you should follow:")
        lines.append(f"1. Clone the repository:")
        lines.append(f"git clone https://github.com/{self.repo}")
        lines.append(f"2. checkout the environment setup commit:")
        lines.append(f"cd {dirname} && git checkout {self.environment_setup_commit}")
        lines.append(f"3. Install the project")
        lines.append(f"pip install -e .")
        lines.append(f"3. Checkout the problematic commit")
        lines.append(f"git checkout {self.base_commit}")
        lines.append(
            f"\n\nPlease identify the minimal fix, and run the test suite to ensure the test works"
        )

        return "\n".join(lines)


class SWEBenchBenchmark(BaseBenchmark):
    """
    SWE-bench benchmark implementation.

    This benchmark evaluates an agent's ability to fix software issues based on
    the princeton-nlp/SWE-bench_Verified dataset.
    """

    name = "swebench"

    def __init__(self, seed: int | None = 1, subset_size: int | None = 100):
        """Initialize the SWE-bench benchmark.

        Args:
            seed: Random seed for shuffling problems (default: 1)
            subset_size: Number of problems to use (default: 7)
        """
        super().__init__(seed, subset_size)

        all_repos = {
            "scikit-learn/scikit-learn",
            "pylint-dev/pylint",
            "pydata/xarray",
            "psf/requests",
            "mwaskom/seaborn",
            "sympy/sympy",
            "matplotlib/matplotlib",
            "pytest-dev/pytest",
            "django/django",
            "sphinx-doc/sphinx",
        }
        fast_repos = {
            "psf/requests",
            "mwaskom/seaborn",
            "sympy/sympy",
            "pytest-dev/pytest",
            "django/django",
            "sphinx-doc/sphinx",
        }
        # For quick debugging
        fast_repos_only: bool = False

        # Store seed and subset_size as instance variables
        self.seed = seed
        self.subset_size = subset_size

        try:
            # Load SWE-bench dataset from HuggingFace
            dataset = load_dataset("princeton-nlp/SWE-bench_Verified")
            split_data = dataset["test"]

            # Convert examples to problems
            self._all_problems = []
            for i, row in enumerate(split_data):
                try:
                    example = SWEBenchExample.from_row(row)

                    if fast_repos_only and example.repo not in fast_repos:
                        continue

                    self._all_problems.append(
                        Problem(
                            problem_id=example.instance_id,
                            statement=example.format_problem(),
                            answer=example,  # Store the full example as the answer
                            answer_discussion=None,
                        )
                    )

                except Exception as e:
                    logger.error(f"Failed to process example {i}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error loading SWE-bench dataset: {e}")
            raise e

        self._problems_shuffled = False
        self._data = None

    def _shuffle_problems(self) -> None:
        """Shuffle the problems list using the specified seed"""
        if not self._problems_shuffled:
            all_problems = self._all_problems.copy()
            random.seed(self.seed)
            random.shuffle(all_problems)
            self._problems_shuffled = True
            # Apply subset if specified
            if self.subset_size is not None:
                self._data = all_problems[: min(self.subset_size, len(all_problems))]
            else:
                self._data = all_problems

    @property
    def problems(self) -> list[Problem]:
        """Return a subset of shuffled problems from the benchmark"""
        if self._data is None:
            self._shuffle_problems()
        return self._data

    async def run_docker_command(self, *args) -> tuple[bool, str, str]:
        """Run a command inside a docker container"""
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        success = proc.returncode == 0
        return success, stdout.decode().strip(), stderr.decode().strip()

    async def setup_problem(
        self, problem: Problem, problem_data_dir: Path, container_name: str
    ) -> None:
        """Setup the git repository for this problem inside the Docker container."""
        if not isinstance(problem.answer, SWEBenchExample):
            raise ValueError("Problem answer must be a SWEBenchExample instance")

        example = problem.answer
        example_dict = example.to_dict()

        try:
            # Add the Git safe directory configuration first
            await self.run_docker_command(
                "docker",
                "exec",
                container_name,
                "git",
                "config",
                "--global",
                "--add",
                "safe.directory",
                "/home/agent/workdir",
            )

            # Run setup commands inside the container
            async def run_in_container(cmd: str, workdir: str | None = None):
                full_cmd = cmd
                if workdir:
                    full_cmd = f"cd {workdir} && {cmd}"
                logger.debug(f"Running in container: {full_cmd}")
                success, stdout, stderr = await self.run_docker_command(
                    "docker", "exec", container_name, "bash", "-c", full_cmd
                )
                if not success:
                    raise RuntimeError(f"Command failed: {stderr}")
                if stdout.strip():
                    logger.debug(f"stdout:\n{stdout}")
                return stdout

            # Install the harness from source: scripts/install_swebench_harness.sh
            from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
            from swebench.harness.test_spec.python import (
                make_env_script_list_py,
                make_repo_script_list_py,
            )

            # Get repository-specific setup from SWE-bench constants
            repo_specs = MAP_REPO_VERSION_TO_SPECS.get(example.repo, {})
            version_specs = repo_specs.get(example.version, {})

            # Create environment and repository setup scripts
            env_script_list = make_env_script_list_py(
                example_dict, version_specs, "testbed"
            )

            # Use workdir directly since it's problem-specific
            repo_script_list = make_repo_script_list_py(
                version_specs,
                example.repo,
                "/home/agent/workdir",  # Clone directly to workdir
                example.base_commit,
                "testbed",
            )

            # Run environment setup scripts
            env_setup = "\n".join(env_script_list)
            logger.debug(f"Running env setup for {problem.problem_id}:\n\n{env_setup}")
            await run_in_container(env_setup)

            # Run repository setup scripts
            repo_setup = "\n".join(repo_script_list)
            logger.debug(
                f"Running repo setup for {problem.problem_id}:\n\n{repo_setup}"
            )
            await run_in_container(repo_setup)

            # Finally, checkout the problematic commit
            await run_in_container(
                f"git checkout {example.base_commit}", workdir="/home/agent/workdir"
            )

        except Exception as e:
            raise RuntimeError(f"Failed to setup repository in container: {str(e)}")

    async def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> tuple[float, str | None, str | None]:
        """Score a submitted answer for the SWE-bench problem."""
        try:
            if not isinstance(problem.answer, SWEBenchExample):
                raise ValueError("Problem answer must be a SWEBenchExample instance")

            example = problem.answer
            repo_path = Path(agent_workdir)

            # Get the diff/patch of changes
            patch_content = ""
            try:
                # Ensure we're in the repo root
                completed_process = subprocess.run(
                    ["git", "diff", "HEAD"],  # Match gold patch format
                    cwd=str(repo_path),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                patch_content = completed_process.stdout
            except subprocess.CalledProcessError as e:
                logger.error(f"Error getting git diff: {e.stderr}")
                return 0.0, f"Error getting git diff: {e.stderr}", None

            # If there's no diff, the agent hasn't made any changes
            if not patch_content.strip():
                logger.error("No changes were made to the repository")
                return 0.0, "No changes were made to the repository", None

            # Save the patch to the answer directory
            patch_file = Path(agent_answer_dir) / "answer.patch"
            patch_file.write_text(patch_content)

            # Create a predictions file for swebench
            prediction_file = Path(agent_answer_dir) / "prediction.json"
            prediction = {
                "model_name_or_path": "agent",
                "instance_id": example.instance_id,
                "model_patch": patch_content,
            }
            with prediction_file.open("w") as f:
                json.dump([prediction], f)

            # Run SWE-bench evaluation
            eval_cmd = ["python3.12", "-m", "swebench.harness.run_evaluation"]
            eval_cmd += ["--dataset_name", "princeton-nlp/SWE-bench_Verified"]
            eval_cmd += ["--predictions_path", "prediction.json"]
            eval_cmd += ["--instance_ids", example.instance_id, "--run_id", "eval"]
            logger.debug(f"Eval cmd: {eval_cmd}")
            subprocess.run(
                eval_cmd,
                cwd=agent_answer_dir,
                check=True,
                capture_output=True,
                text=True,
            )

            # Look for results in the expected location
            results_path = Path(agent_answer_dir) / f"agent.eval.json"
            if results_path.exists():
                with open(results_path, "r") as f:
                    results = json.load(f)
                    correct = results.get("resolved_instances", 0) == 1
                    return 1.0 if correct else 0.0, None, None

            logger.error("Could not find evaluation results")
            return 0.0, "Could not find evaluation results", None

        except Exception as e:
            logger.error(f"Error while scoring swebench problem: {e}")
            return 0.0, str(e), None


# Testing ----------------------------------------------------------------------

import asyncio
import logging
import tempfile
import re
import time
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Existing helper functions remain the same
async def run_docker_command(*args) -> tuple[bool, str, str]:
    """Run a docker command and return success, stdout, stderr"""
    logger.debug(f"Running docker command: {' '.join(args)}")
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode == 0, stdout.decode(), stderr.decode()


async def wait_for_container_ready(container_name: str, timeout: float = 30):
    """Wait until container is ready to accept commands"""
    start_time = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - start_time) < timeout:
        try:
            _, stdout, _ = await run_docker_command(
                "docker",
                "inspect",
                "--format",
                "{{json .State.Status}}",
                container_name,
            )
            if "running" in stdout.lower():
                try:
                    await run_docker_command("docker", "exec", container_name, "ps")
                    return
                except Exception:
                    pass
        except Exception:
            pass
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Container {container_name} not ready within {timeout}s")


async def run_in_container(
    container_name: str, cmd: str, workdir: str | None = None
) -> tuple[bool, str, str]:
    """Run a command in the container with detailed logging"""
    full_cmd = cmd
    if workdir:
        full_cmd = f"cd {workdir} && {cmd}"

    logger.info(f"Running in container: {full_cmd}")
    success, stdout, stderr = await run_docker_command(
        "docker", "exec", container_name, "bash", "-c", full_cmd
    )

    if stdout.strip():
        logger.info(f"stdout:\n{stdout}")
    if stderr.strip():
        logger.info(f"stderr:\n{stderr}")

    return success, stdout, stderr


async def run_single_test(
    problem: Problem,
    benchmark: SWEBenchBenchmark,
    base_temp_dir: Path,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, float, str | None]:
    """Run a single test case with resource management"""
    async with semaphore:  # Control concurrent Docker container creation
        # Create problem-specific temporary directory
        dirname = re.sub(r"[^a-zA-Z0-9_]", "", f"swebench_{problem.problem_id}")
        problem_data_dir = base_temp_dir / "problem_data" / dirname
        problem_data_dir.mkdir(parents=True, exist_ok=True)

        results_dir = base_temp_dir / "results" / dirname
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create unique container name
        container_name = f"swebench_test_{problem.problem_id}"
        container_name = container_name.replace("/", "_")

        try:
            # Start container
            container_cmd = [
                "docker",
                "run",
                "--rm",
                "-d",
                "--name",
                container_name,
                "-v",
                f"{problem_data_dir}:/home/agent/workdir:rw",
                "-v",
                f"{Path('.env').absolute()}:/home/agent/.env:ro",
                "adas_sandbox",
                "tail",
                "-f",
                "/dev/null",
            ]

            success, stdout, stderr = await run_docker_command(*container_cmd)
            if not success:
                return problem.problem_id, 0.0, f"Failed to start container: {stderr}"

            await wait_for_container_ready(container_name)

            # Setup the problem
            start = time.time()
            try:
                await benchmark.setup_problem(problem, problem_data_dir, container_name)
            except Exception as e:
                return problem.problem_id, 0.0, f"Problem setup failed: {str(e)}"

            print(f"Setup for {problem.problem_id} took {time.time() - start:.2f}s")

            # Apply the gold standard patch
            patch_file = problem_data_dir / "changes.patch"
            patch_file.write_text(problem.answer.patch)

            success, stdout, stderr = await run_in_container(
                container_name, "git apply changes.patch", workdir="/home/agent/workdir"
            )
            if not success:
                return problem.problem_id, 0.0, f"Failed to apply patch: {stderr}"

            # Score the solution
            score, errors, discussion = benchmark.score_problem(
                problem, str(problem_data_dir), str(results_dir), container_name
            )

            return problem.problem_id, score, errors

        finally:
            # Cleanup container
            await run_docker_command("docker", "rm", "-f", container_name)


async def main():
    # Initialize benchmark
    t0 = time.time()
    logger.info("Initializing benchmark...")
    benchmark = SWEBenchBenchmark(seed=1, subset_size=100)
    problems = benchmark.problems

    # Get unique repos
    blacklist = {"scikit-learn/scikit-learn"}
    repos = blacklist
    problem_set = []
    for p in problems:
        if p.answer.repo not in repos:
            repos.add(p.answer.repo)
            problem_set.append(p)

    print(f"Found these unique repos: {repos}")
    logger.info(f"Benchmark initialized in {time.time() - t0:.2f}s")

    # Create base temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        base_temp_dir = Path(temp_dir)

        # Control concurrent Docker containers (adjust based on system resources)
        max_concurrent = 4
        semaphore = asyncio.Semaphore(max_concurrent)

        # Run tests in parallel
        tasks = [
            run_single_test(problem, benchmark, base_temp_dir, semaphore)
            for problem in problem_set
        ]

        # Wait for all tests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        all_passed = True
        for problem_id, score, errors in results:
            print(f"\nResults for problem {problem_id}:")
            print(f"Score: {score}")
            if errors:
                print(f"Errors: {errors}")
                all_passed = False

            # Assert expected score
            try:
                assert score == 1.0, f"Expected score of 1.0 but got {score}"
            except AssertionError as e:
                print(f"Assertion failed: {str(e)}")
                all_passed = False

        if all_passed:
            print("\nAll verification tests passed successfully!")
        else:
            print("\nSome tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())
