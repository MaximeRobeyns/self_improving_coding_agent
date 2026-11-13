"""
How agents are evaluated: which problems they are run on, how often, and
whether an agent is worth evaluating in full.

All choices of problems are deterministic functions of a seed and the problem
ids, so that every agent of an experiment is evaluated on the same problems
and its results can be compared problem by problem.
"""

import hashlib

from pathlib import Path
from typing import TYPE_CHECKING, Sequence
from dataclasses import asdict, dataclass

from base_agent.src.utils.archive_analysis import PairedDelta

if TYPE_CHECKING:
    # Only needed for annotations: importing the benchmarks package loads
    # every benchmark and its dependencies
    from base_agent.src.benchmarks.base import Problem


@dataclass(frozen=True)
class EvaluationConfig:
    # Number of times each problem is run
    repeats: int = 1
    # Share of each benchmark's problems that is kept out of the archive, and
    # so out of sight of the agents doing the meta-improvement
    holdout_fraction: float = 0.0
    # Problems per benchmark in the first evaluation stage (0: no stages)
    stage1_size: int = 0
    # An agent is evaluated in full unless the first stage shows that it is
    # worse than its parent by more than this (see should_promote)
    promote_threshold: float = 0.0
    # Seed of the problem splits; fixed for the life of an experiment
    split_seed: int = 0

    def __post_init__(self):
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1")
        if not 0.0 <= self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be in [0, 1)")
        if self.stage1_size < 0:
            raise ValueError("stage1_size must not be negative")

    def to_dict(self) -> dict:
        return asdict(self)


def holdout_dir(exp_dir: Path) -> Path:
    """Where the results on held-out problems of an experiment go. Mirrors the
    layout of the experiment directory, but lies outside of it: the experiment
    directory is the archive that agents get to see."""
    return exp_dir.parent / "holdout" / exp_dir.name


def _order(
    problems: Sequence["Problem"], benchmark_name: str, seed: int, purpose: str
) -> list["Problem"]:
    """A fixed pseudo-random order of the problems. It depends only on the
    problem ids, not on the order or the number of problems given."""

    def rank(problem: "Problem") -> str:
        key = f"{seed}:{purpose}:{benchmark_name}:{problem.problem_id}"
        return hashlib.sha256(key.encode()).hexdigest()

    return sorted(problems, key=rank)


def split_problems(
    problems: Sequence["Problem"],
    benchmark_name: str,
    holdout_fraction: float,
    seed: int,
) -> tuple[list["Problem"], list["Problem"]]:
    """Split a benchmark's problems into (dev, holdout), keeping their order.

    With a positive fraction, both parts get at least one problem whenever
    there are two or more problems.
    """
    problems = list(problems)
    if holdout_fraction <= 0.0 or len(problems) < 2:
        return problems, []
    n_holdout = round(len(problems) * holdout_fraction)
    n_holdout = max(1, min(len(problems) - 1, n_holdout))
    ordered = _order(problems, benchmark_name, seed, "holdout")
    holdout_ids = {p.problem_id for p in ordered[:n_holdout]}
    dev = [p for p in problems if p.problem_id not in holdout_ids]
    holdout = [p for p in problems if p.problem_id in holdout_ids]
    return dev, holdout


def stage1_problems(
    problems: Sequence["Problem"], benchmark_name: str, size: int, seed: int
) -> list["Problem"]:
    """The problems of the first evaluation stage: a fixed subset of `size`
    problems (all of them, if there are no more), keeping their order."""
    problems = list(problems)
    if size >= len(problems):
        return problems
    chosen = {p.problem_id for p in _order(problems, benchmark_name, seed, "stage1")[:size]}
    return [p for p in problems if p.problem_id in chosen]


def should_promote(delta: PairedDelta | None, threshold: float = 0.0) -> bool:
    """Whether an agent goes on to the full evaluation after the first stage.

    `delta` is its score relative to its parent on the first-stage problems.
    The agent is dropped only when even the upper end of the confidence
    interval is below the threshold: with the default of 0, when the data
    rule out that it is as good as its parent. The first stage is small, so
    anything less clear-cut is given the benefit of the doubt, as is an agent
    that cannot be compared with its parent at all.
    """
    if delta is None or delta.n_problems < 2:
        return True
    return delta.ci_upper >= threshold
