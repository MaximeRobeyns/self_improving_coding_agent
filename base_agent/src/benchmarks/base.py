# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import jsonlines

from abc import abstractmethod
from typing import Any, ClassVar
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class Problem:
    """A single benchmark problem, containing a problem_id, problem statement and answer"""

    problem_id: str
    statement: str
    answer: Any
    answer_discussion: str | None


@dataclass
class ProblemResult:
    """Complete record of a single problem attempt"""

    problem_id: str
    timestamp: str | None = None
    score: float | None = None
    tokens_used: int | None = None
    num_cached_tokens: int | None = None
    cost_estimate: float | None = None
    wall_time: float | None = None
    timed_out: bool = False
    cost_threshold_exceeded: bool = False
    # Which run of the problem this is, when problems are run more than once.
    # The problem_id of runs after the first ends in '#<repeat>'.
    repeat: int = 0

    def is_complete(self) -> bool:
        # Considered complete if it has been scored
        return self.score is not None

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid field {key} in ProblemResult update")


class BenchmarkTracker:
    def __init__(self, results_path: Path):
        self.results_path = results_path
        self.results: dict[str, ProblemResult] = self._load_or_create()

    def _load_or_create(self) -> dict[str, ProblemResult]:
        results = {}
        if self.results_path.exists():
            with jsonlines.open(self.results_path) as reader:
                for line in reader:
                    results[line["problem_id"]] = ProblemResult(**line)
        return results

    def start_problem(self, problem_id: str, repeat: int = 0) -> None:
        result = ProblemResult(
            problem_id=problem_id,
            timestamp=datetime.now().isoformat(),
            repeat=repeat,
        )
        self.results[problem_id] = result
        with jsonlines.open(self.results_path, mode="a") as writer:
            writer.write(asdict(result))

    def update_problem(self, problem_id: str, **kwargs) -> None:
        if problem_id not in self.results:
            raise KeyError(f"Problem {problem_id} not found")

        self.results[problem_id].update(**kwargs)

        # Rewrite the file with updated results
        with jsonlines.open(self.results_path, mode="w") as writer:
            writer.write_all(asdict(result) for result in self.results.values())


class BaseBenchmark:

    name: ClassVar[str]

    def __init__(self, seed: int | None = None, subset_size: int | None = None):
        self.problem_idx: int = 0
        self.seed = seed
        self.subset_size = subset_size

    @property
    @abstractmethod
    def problems(self) -> list[Problem]:
        pass

    @abstractmethod
    async def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> tuple[float, str | None, str | None]:
        """
        Score the answer to the problem; the agent_workdir is an absolute path
        to the mapped /home/agent/workdir in the docker container, while the
        agent_answer_dir is the absolute path to the mapped logdir in the
        docker container, which should contain an answer.txt file.

        To get the submitted answer (if relevant):

        answer_path = Path(agent_answer_dir) / "answer.txt"
        llm_answer = answer_path.read_text().strip()

        Return the score (as a float), any parsing errors, and any additional
        discussion or information about the answer that can assist the summary.
        """
        pass

    def get_problem(self, problem_id: str) -> Problem | None:
        """Retrieve a specific problem by ID
        Overload this method if there is a more efficient way of locating the
        problem by problem_id.
        """
        return next((p for p in self.problems if p.problem_id == problem_id), None)

    async def setup_problem(
        self, problem: Problem, problem_data_dir: Path, container_name: str
    ) -> None:
        """Optional hook for performing problem-specific setup.

        This is called before each problem is run. The problem_data_dir
        will be mounted in the agent's container at /home/agent/workdir.

        Args:
            problem: The problem being run
            problem_data_dir: Path to a temporary directory for problem data.
                This directory will be mounted in the agent's container.
            container_name: The name of the container that the problem will run in
        """
        pass  # Default no-op implementation
