"""Tests for the choice of evaluation problems and the promotion rule."""
from dataclasses import dataclass

import pytest

from base_agent.src.utils.archive_analysis import PairedDelta
from search.evaluation import (
    EvaluationConfig,
    should_promote,
    split_problems,
    stage1_problems,
)


@dataclass
class Problem:
    """Like a benchmark problem: all that matters here is its id."""

    problem_id: str


def problems(n, prefix="p"):
    return [Problem(f"{prefix}{i}") for i in range(n)]


def ids(problem_list):
    return [p.problem_id for p in problem_list]


# split_problems ---------------------------------------------------------------


def test_no_holdout_by_default():
    dev, holdout = split_problems(problems(10), "bench", 0.0, seed=1)
    assert ids(dev) == ids(problems(10))
    assert holdout == []


def test_split_sizes_and_order():
    dev, holdout = split_problems(problems(20), "bench", 0.3, seed=1)

    assert len(holdout) == 6
    assert len(dev) == 14
    assert sorted(ids(dev) + ids(holdout)) == sorted(ids(problems(20)))
    # Both parts keep the benchmark's order
    position = {pid: n for n, pid in enumerate(ids(problems(20)))}
    for part in (dev, holdout):
        assert [position[i] for i in ids(part)] == sorted(position[i] for i in ids(part))


def test_split_is_stable():
    first = split_problems(problems(20), "bench", 0.3, seed=1)
    again = split_problems(list(reversed(problems(20))), "bench", 0.3, seed=1)
    assert set(ids(first[1])) == set(ids(again[1]))


def test_split_depends_on_seed_and_benchmark():
    base = set(ids(split_problems(problems(40), "bench", 0.5, seed=1)[1]))
    assert base != set(ids(split_problems(problems(40), "bench", 0.5, seed=2)[1]))
    assert base != set(ids(split_problems(problems(40), "other", 0.5, seed=1)[1]))


@pytest.mark.parametrize("fraction", [0.01, 0.5, 0.99])
def test_both_parts_get_a_problem(fraction):
    dev, holdout = split_problems(problems(3), "bench", fraction, seed=1)
    assert len(dev) >= 1 and len(holdout) >= 1


def test_single_problem_cannot_be_split():
    dev, holdout = split_problems(problems(1), "bench", 0.5, seed=1)
    assert (len(dev), len(holdout)) == (1, 0)


# stage1_problems --------------------------------------------------------------


def test_stage1_is_a_fixed_subset():
    subset = stage1_problems(problems(18), "bench", 8, seed=1)

    assert len(subset) == 8
    assert set(ids(subset)) < set(ids(problems(18)))
    assert ids(subset) == ids(stage1_problems(problems(18), "bench", 8, seed=1))
    assert ids(subset) != ids(stage1_problems(problems(18), "bench", 8, seed=2))


def test_stage1_of_a_small_benchmark_is_all_of_it():
    assert ids(stage1_problems(problems(5), "bench", 8, seed=1)) == ids(problems(5))


def test_stage1_is_independent_of_the_holdout_split():
    # Different orderings: the first stage is not just the first few dev problems
    dev, _ = split_problems(problems(40), "bench", 0.5, seed=1)
    subset = stage1_problems(dev, "bench", 10, seed=1)
    assert set(ids(subset)) <= set(ids(dev))
    assert ids(subset) != ids(dev)[:10]


# should_promote ---------------------------------------------------------------


@pytest.mark.parametrize(
    "delta, threshold, expected",
    [
        (PairedDelta(0.2, 0.0, 0.4, 8, 1), 0.0, True),  # better
        (PairedDelta(-0.1, -0.4, 0.2, 8, 1), 0.0, True),  # unclear: benefit of the doubt
        (PairedDelta(0.0, 0.0, 0.0, 8, 1), 0.0, True),  # identical to its parent
        (PairedDelta(-0.5, -0.8, -0.2, 8, 1), 0.0, False),  # clearly worse
        (PairedDelta(-1.0, -1.0, -1.0, 8, 1), 0.0, False),  # broken
        (PairedDelta(-0.5, -0.8, -0.2, 8, 1), -0.3, True),  # more lenient
        (PairedDelta(0.05, -0.1, 0.08, 8, 1), 0.1, False),  # stricter
        (PairedDelta(-1.0, -1.0, -1.0, 1, 1), 0.0, True),  # one problem proves nothing
        (None, 0.0, True),  # cannot be compared
    ],
)
def test_should_promote(delta, threshold, expected):
    assert should_promote(delta, threshold) is expected


# EvaluationConfig -------------------------------------------------------------


def test_config_defaults_change_nothing():
    config = EvaluationConfig()
    assert (config.repeats, config.holdout_fraction, config.stage1_size) == (1, 0.0, 0)
    assert EvaluationConfig(**config.to_dict()) == config


@pytest.mark.parametrize(
    "kwargs",
    [{"repeats": 0}, {"holdout_fraction": 1.0}, {"holdout_fraction": -0.1}, {"stage1_size": -1}],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        EvaluationConfig(**kwargs)
