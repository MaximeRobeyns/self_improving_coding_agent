"""
Parent selection: which archived agent(s) the next agent should be built from.

Strategies share one interface (`ParentSelector`) and are looked up by name
with `get_selector`. They only see an `ArchiveView`: the agents that are
usable as parents, with their scores and how often they were already built on.
"""

import math
import random
import logging

from typing import Protocol
from dataclasses import dataclass, field

from base_agent.src.utils.archive_analysis import (
    ArchiveAnalyzer,
    ScoreType,
    compute_statistics,
    per_problem_scores,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """An archived agent that may be selected as a parent."""

    id: int
    mean: float  # the agent's overall score
    ci_lower: float  # lower 95% confidence bound of `mean` (may be NaN)
    scores: tuple[float, ...] = ()  # individual problem scores
    n_children: int = 0  # agents already built from this one


@dataclass(frozen=True)
class ArchiveView:
    """What a selection strategy gets to see of the archive."""

    candidates: dict[int, Candidate]
    current_iteration: int

    @classmethod
    def from_analyzer(
        cls,
        analyzer: ArchiveAnalyzer,
        current_iteration: int,
        score_type: ScoreType = "mean_score",
        excluded: set[int] | frozenset[int] = frozenset(),
    ) -> "ArchiveView":
        """Build the view from an archive.

        Only agents with benchmark results are candidates. `excluded` agents
        (e.g. those whose meta-improvement step failed) never are.
        """
        scores_df, summaries_df = analyzer.get_problem_scores_by_iteration()
        if scores_df.empty or summaries_df.empty:
            return cls({}, current_iteration)
        stats = compute_statistics(scores_df, summaries_df, score_type=score_type)
        if stats.empty:
            return cls({}, current_iteration)

        score_key = "score" if score_type == "mean_score" else "utility"
        problem_scores = (
            per_problem_scores(scores_df, score_key)
            .groupby("iteration")[score_key]
            .apply(tuple)
        )

        n_children: dict[int, int] = {}
        for record in analyzer.get_lineage().values():
            parent_id = record.get("parent_id")
            if parent_id is not None:
                n_children[parent_id] = n_children.get(parent_id, 0) + 1

        candidates = {}
        for _, row in stats.iterrows():
            iteration = int(row["iteration"])
            if iteration in excluded:
                continue
            candidates[iteration] = Candidate(
                id=iteration,
                mean=float(row["target_score"]),
                ci_lower=float(row["ci_lower"]),
                scores=tuple(float(s) for s in problem_scores.get(iteration, ())),
                n_children=n_children.get(iteration, 0),
            )
        return cls(candidates, current_iteration)


@dataclass(frozen=True)
class Selection:
    """The outcome of a parent selection."""

    parents: list[int]
    # For each candidate, the probability with which one pick selects it
    weights: dict[int, float] = field(default_factory=dict)


class ParentSelector(Protocol):
    name: str

    def select(self, archive: ArchiveView, k: int, rng: random.Random) -> Selection:
        """Select `k` parents (with replacement) from the archive."""
        ...


FALLBACK_PARENT = 0  # the base agent, when the archive has no candidates


def _fallback(k: int) -> Selection:
    logger.info("No parent candidates; defaulting to base agent iteration 0")
    return Selection([FALLBACK_PARENT] * k)


class GreedyCI:
    """The latest agent that is not significantly worse than the best one.

    Walks back from the current iteration and takes the first agent whose
    score reaches the lower confidence bound of the best agent, falling back
    to the best agent itself. Deterministic: all `k` parents are the same.
    """

    name = "greedy_ci"

    def select(self, archive: ArchiveView, k: int, rng: random.Random) -> Selection:
        candidates = archive.candidates
        if not candidates:
            return _fallback(k)
        parent = self._select_one(archive)
        weights = {i: 1.0 if i == parent else 0.0 for i in candidates}
        return Selection([parent] * k, weights)

    @staticmethod
    def _select_one(archive: ArchiveView) -> int:
        candidates = archive.candidates
        # First of the best-scoring agents, in iteration order
        best = max(sorted(candidates), key=lambda i: candidates[i].mean)
        best_lower_bound = candidates[best].ci_lower

        for i in range(archive.current_iteration, -1, -1):
            if i not in candidates:
                continue
            logger.info(
                f"Agent {i} mean score: {candidates[i].mean}; best lower bound: {best_lower_bound}"
            )
            if candidates[i].mean >= best_lower_bound:
                return i
            # If we've gone past the best iteration and haven't found a
            # suitable agent, use the best
            if i <= best:
                return best
        return FALLBACK_PARENT


class ScoreChildProp:
    """Sample in proportion to score, discounted by the number of children.

    weight = sigmoid(steepness * (score - midpoint)) / (1 + n_children)

    Favours high-scoring agents, while steering away from agents that were
    already built on many times so that the search branches out.
    """

    name = "score_child_prop"

    def __init__(self, steepness: float = 10.0, midpoint: float = 0.5):
        self.steepness = steepness
        self.midpoint = midpoint

    def select(self, archive: ArchiveView, k: int, rng: random.Random) -> Selection:
        candidates = archive.candidates
        if not candidates:
            return _fallback(k)
        ids = sorted(candidates)
        raw = [self._weight(candidates[i]) for i in ids]
        total = sum(raw)
        weights = {i: w / total for i, w in zip(ids, raw)}
        parents = rng.choices(ids, weights=[weights[i] for i in ids], k=k)
        return Selection(parents, weights)

    def _weight(self, candidate: Candidate) -> float:
        z = self.steepness * (candidate.mean - self.midpoint)
        z = max(-60.0, min(60.0, z))  # keep exp() in range and weights positive
        return 1.0 / (1.0 + math.exp(-z)) / (1 + candidate.n_children)


class Thompson:
    """Thompson sampling over each agent's success rate.

    Each agent's problem scores (in [0, 1]) give a Beta posterior over its
    true score. A pick draws one value per agent and takes the largest, so
    agents evaluated on few problems are still explored.
    """

    name = "thompson"
    WEIGHT_ESTIMATE_DRAWS = 2000

    def select(self, archive: ArchiveView, k: int, rng: random.Random) -> Selection:
        candidates = archive.candidates
        if not candidates:
            return _fallback(k)
        ids = sorted(candidates)
        posteriors = [self._posterior(candidates[i]) for i in ids]

        def pick() -> int:
            draws = [rng.betavariate(a, b) for a, b in posteriors]
            return ids[draws.index(max(draws))]

        parents = [pick() for _ in range(k)]

        # The pick probabilities have no closed form; estimate them
        counts = dict.fromkeys(ids, 0)
        for _ in range(self.WEIGHT_ESTIMATE_DRAWS):
            counts[pick()] += 1
        weights = {i: c / self.WEIGHT_ESTIMATE_DRAWS for i, c in counts.items()}
        return Selection(parents, weights)

    @staticmethod
    def _posterior(candidate: Candidate) -> tuple[float, float]:
        scores = candidate.scores or (candidate.mean,)
        scores = [min(1.0, max(0.0, s)) for s in scores if not math.isnan(s)]
        successes = sum(scores)
        return 1.0 + successes, 1.0 + len(scores) - successes


_SELECTORS: dict[str, type] = {
    cls.name: cls for cls in (GreedyCI, ScoreChildProp, Thompson)
}
SELECTION_STRATEGIES = tuple(_SELECTORS)
DEFAULT_SELECTION_STRATEGY = GreedyCI.name


def get_selector(name: str) -> ParentSelector:
    try:
        return _SELECTORS[name]()
    except KeyError:
        raise ValueError(
            f"Unknown selection strategy {name!r}; choose from {', '.join(SELECTION_STRATEGIES)}"
        ) from None


def selection_rng(seed: int, iteration: int) -> random.Random:
    """The random stream for the selection made at a given iteration.

    Derived from the experiment seed alone, so that a resumed experiment
    makes the same selection as an uninterrupted one.
    """
    return random.Random(f"{seed}:{iteration}")
