"""Tests for parent selection strategies."""
import math
import random

import pandas as pd
import pytest

from base_agent.src.utils.archive_analysis import compute_statistics
from search.selection import (
    DEFAULT_SELECTION_STRATEGY,
    SELECTION_STRATEGIES,
    ArchiveView,
    Candidate,
    GreedyCI,
    ScoreChildProp,
    Thompson,
    get_selector,
    selection_rng,
)


class FakeAnalyzer:
    """Stands in for ArchiveAnalyzer: one benchmark's problem scores per agent."""

    def __init__(self, scores_by_iteration, parents=None):
        self._scores = scores_by_iteration
        self._parents = parents or {}

    def get_problem_scores_by_iteration(self):
        score_rows, summary_rows = [], []
        for iteration, scores in self._scores.items():
            for score in scores:
                score_rows.append(
                    {"iteration": iteration, "score": score, "utility": score / 2}
                )
            mean = sum(scores) / len(scores)
            summary_rows.append(
                {
                    "iteration": iteration,
                    "avg_score": mean,
                    "utility_score": mean / 2,
                    "total_tokens": 0,
                    "total_cached_tokens": 0,
                    "total_cost": 0.0,
                    "total_time": 0.0,
                }
            )
        return pd.DataFrame(score_rows), pd.DataFrame(summary_rows)

    def get_lineage(self):
        return {i: {"id": i, "parent_id": p} for i, p in self._parents.items()}


def view(scores_by_iteration, current_iteration=None, excluded=frozenset(), parents=None):
    if current_iteration is None:
        current_iteration = max(scores_by_iteration)
    return ArchiveView.from_analyzer(
        FakeAnalyzer(scores_by_iteration, parents), current_iteration, excluded=excluded
    )


def candidates(**means_by_name):
    """ArchiveView from e.g. a0=0.5, a1=(0.9, n_children)."""
    result = {}
    for name, spec in means_by_name.items():
        i = int(name[1:])
        mean, n_children = spec if isinstance(spec, tuple) else (spec, 0)
        result[i] = Candidate(i, mean, mean, (mean,) * 10, n_children)
    return ArchiveView(result, max(result))


# ArchiveView ------------------------------------------------------------------


def test_view_holds_scores_and_statistics():
    v = view({0: [0, 0, 0, 1], 1: [1, 1, 1, 0]})

    assert sorted(v.candidates) == [0, 1]
    assert v.current_iteration == 1
    assert v.candidates[1].mean == 0.75
    assert v.candidates[1].scores == (1.0, 1.0, 1.0, 0.0)
    # 95% interval of the mean: 0.75 - 1.96 * std / sqrt(n)
    assert v.candidates[1].ci_lower == pytest.approx(0.75 - 1.96 * 0.5 / 2)


def test_view_excludes_agents():
    v = view({0: [0, 1], 1: [1, 1], 2: [1, 0]}, excluded={1})
    assert sorted(v.candidates) == [0, 2]


def test_view_counts_children():
    v = view(
        {0: [0, 1], 1: [1, 1], 2: [1, 0]},
        # agent 3 has no results yet, but was still built from agent 1
        parents={0: None, 1: 0, 2: 0, 3: 1},
    )
    assert {i: c.n_children for i, c in v.candidates.items()} == {0: 2, 1: 1, 2: 0}


def test_view_uses_utility_scores_when_asked():
    v = ArchiveView.from_analyzer(
        FakeAnalyzer({0: [1, 1, 0, 0]}), 0, score_type="utility_score"
    )
    assert v.candidates[0].mean == 0.25
    assert v.candidates[0].scores == (0.5, 0.5, 0.0, 0.0)


def test_view_of_empty_archive():
    assert view({}, current_iteration=0).candidates == {}


# All strategies ---------------------------------------------------------------


@pytest.mark.parametrize("name", SELECTION_STRATEGIES)
def test_empty_archive_falls_back_to_base_agent(name):
    selection = get_selector(name).select(ArchiveView({}, 3), 2, random.Random(0))
    assert selection.parents == [0, 0]
    assert selection.weights == {}


@pytest.mark.parametrize("name", SELECTION_STRATEGIES)
def test_same_seed_same_selection(name):
    archive = view({0: [0, 0, 1, 1], 1: [1, 1, 1, 0], 2: [1, 0, 1, 0], 3: [1, 1, 0, 0]})
    selector = get_selector(name)

    first = selector.select(archive, 5, selection_rng(7, 3))
    again = selector.select(archive, 5, selection_rng(7, 3))

    assert first == again
    assert len(first.parents) == 5


@pytest.mark.parametrize("name", SELECTION_STRATEGIES)
def test_never_selects_an_excluded_agent(name):
    archive = view({0: [0, 0, 0, 1], 1: [1, 1, 1, 1], 2: [1, 1, 1, 1]}, excluded={1, 2})
    selection = get_selector(name).select(archive, 50, random.Random(1))
    assert set(selection.parents) == {0}
    assert set(selection.weights) == {0}


@pytest.mark.parametrize("name", SELECTION_STRATEGIES)
def test_weights_are_probabilities_over_candidates(name):
    archive = view({0: [0, 0, 1, 1], 1: [1, 1, 1, 0], 2: [1, 0, 1, 0]})
    selection = get_selector(name).select(archive, 1, random.Random(3))
    assert set(selection.weights) == {0, 1, 2}
    assert sum(selection.weights.values()) == pytest.approx(1.0)
    assert all(0.0 <= w <= 1.0 for w in selection.weights.values())


def test_registry():
    assert SELECTION_STRATEGIES == ("greedy_ci", "score_child_prop", "thompson")
    assert DEFAULT_SELECTION_STRATEGY == "greedy_ci"
    assert isinstance(get_selector("thompson"), Thompson)
    with pytest.raises(ValueError, match="greedy_ci, score_child_prop, thompson"):
        get_selector("nope")


def test_selection_rng_depends_on_seed_and_iteration():
    def stream(seed, iteration):
        return selection_rng(seed, iteration).random()

    assert stream(1, 2) == stream(1, 2)
    assert stream(1, 2) != stream(1, 3)
    assert stream(1, 2) != stream(2, 2)


# GreedyCI ---------------------------------------------------------------------


def legacy_select_base_agent(analyzer, current_iteration, excluded=frozenset()):
    """The selection rule as it was in runner.py before strategies existed."""
    scores_df, summaries_df = analyzer.get_problem_scores_by_iteration()
    if scores_df.empty or summaries_df.empty:
        return 0
    stats = compute_statistics(scores_df, summaries_df, score_type="mean_score")
    if stats.empty:
        return 0
    stats = stats[~stats["iteration"].isin(excluded)]
    stats = stats.set_index("iteration", drop=False)
    if stats.empty:
        return 0
    best_idx = stats["target_score"].idxmax()
    best_lower_bound = stats.loc[best_idx]["ci_lower"]
    for i in range(current_iteration, -1, -1):
        if i not in stats.index:
            continue
        if stats.loc[i, "target_score"] >= best_lower_bound:
            return i
        if i <= best_idx:
            return best_idx
    return 0


def greedy(scores_by_iteration, current_iteration, excluded=frozenset()):
    archive = view(scores_by_iteration, current_iteration, excluded)
    return GreedyCI().select(archive, 1, random.Random(0)).parents[0]


def test_greedy_picks_latest_good_agent():
    assert greedy({0: [0, 0, 0, 1], 1: [1, 1, 1, 0], 2: [1, 1, 1, 1]}, 2) == 2


def test_greedy_rolls_back_to_the_best_agent():
    # Agent 2 regressed to well below agent 1's lower bound
    assert greedy({0: [0, 0, 0, 0], 1: [1, 1, 1, 1], 2: [1, 0, 0, 0]}, 2) == 1


def test_greedy_skips_excluded_agent():
    scores = {0: [0, 0, 0, 1], 1: [1, 1, 1, 0], 2: [1, 1, 1, 1]}
    assert greedy(scores, 2, excluded={2}) == 1


def test_greedy_excluded_best_does_not_set_the_bar():
    scores = {0: [0, 0, 0, 0], 1: [1, 1, 1, 1], 2: [1, 0, 1, 0]}
    assert greedy(scores, 2, excluded={1}) == 2


def test_greedy_is_deterministic_for_all_k():
    archive = view({0: [0, 0, 0, 1], 1: [1, 1, 1, 0]})
    selection = GreedyCI().select(archive, 3, random.Random(0))
    assert selection.parents == [1, 1, 1]
    assert selection.weights == {0: 0.0, 1: 1.0}


def test_greedy_matches_the_legacy_rule_on_random_archives():
    rng = random.Random(20261112)
    for _ in range(500):
        n_agents = rng.randint(1, 8)
        # Agents may lack results (non-contiguous iterations), scores may tie
        iterations = [i for i in range(n_agents) if rng.random() < 0.85] or [0]
        n_problems = rng.choice([1, 2, 4, 18])
        p = {i: rng.choice([0.0, 0.2, 0.5, 0.5, 0.8, 1.0]) for i in iterations}
        scores = {
            i: [float(rng.random() < p[i]) for _ in range(n_problems)]
            for i in iterations
        }
        excluded = {i for i in iterations if rng.random() < 0.15}
        current = rng.randint(max(iterations), n_agents)

        expected = legacy_select_base_agent(FakeAnalyzer(scores), current, excluded)
        assert greedy(scores, current, excluded) == expected, (scores, current, excluded)


# ScoreChildProp ---------------------------------------------------------------


def test_score_child_prop_weight_formula():
    archive = candidates(a0=0.5, a1=(0.5, 1), a2=0.7)
    weights = ScoreChildProp().select(archive, 1, random.Random(0)).weights

    raw = {0: 0.5, 1: 0.5 / 2, 2: 1 / (1 + math.exp(-2.0))}
    total = sum(raw.values())
    assert weights == pytest.approx({i: w / total for i, w in raw.items()})


def test_score_child_prop_prefers_higher_scores():
    weights = ScoreChildProp().select(candidates(a0=0.3, a1=0.6), 1, random.Random(0)).weights
    assert weights[1] > weights[0]


def test_score_child_prop_discounts_agents_with_many_children():
    weights = ScoreChildProp().select(
        candidates(a0=(0.6, 4), a1=(0.6, 0)), 1, random.Random(0)
    ).weights
    assert weights[1] == pytest.approx(5 * weights[0])


def test_score_child_prop_samples_according_to_weights():
    archive = candidates(a0=0.4, a1=0.6, a2=(0.6, 3))
    selection = ScoreChildProp().select(archive, 20000, random.Random(5))
    for i, weight in selection.weights.items():
        assert selection.parents.count(i) / 20000 == pytest.approx(weight, abs=0.015)


def test_score_child_prop_handles_extreme_scores():
    weights = ScoreChildProp().select(
        candidates(a0=-1000.0, a1=1000.0), 1, random.Random(0)
    ).weights
    assert weights[0] > 0.0
    assert sum(weights.values()) == pytest.approx(1.0)


# Thompson ---------------------------------------------------------------------


def test_thompson_prefers_the_better_agent():
    archive = view({0: [0.0] * 18 + [1.0] * 2, 1: [1.0] * 18 + [0.0] * 2})
    selection = Thompson().select(archive, 200, random.Random(0))
    assert selection.parents.count(1) == 200
    assert selection.weights[1] > 0.99


def test_thompson_still_explores_agents_with_few_results():
    # Agent 1 failed its only problem, yet may well be better than agent 0
    archive = view({0: [1.0] * 10 + [0.0] * 10, 1: [0.0]})
    selection = Thompson().select(archive, 1, random.Random(0))
    assert 0.1 < selection.weights[1] < 0.5


def test_thompson_posterior():
    assert Thompson._posterior(Candidate(0, 0.5, 0.0, (1.0, 0.0, 0.5, 1.0))) == (3.5, 2.5)
    # Scores outside [0, 1] are clipped; missing scores fall back to the mean
    assert Thompson._posterior(Candidate(0, 0.5, 0.0, (2.0, -1.0))) == (2.0, 2.0)
    assert Thompson._posterior(Candidate(0, 0.25, 0.0)) == (1.25, 1.75)
