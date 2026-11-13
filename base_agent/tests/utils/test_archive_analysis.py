# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the archive analysis utility."""
import os
import json
import pytest
from pathlib import Path
from datetime import datetime, timedelta

from src.utils.archive_analysis import (
    ArchiveAnalyzer,
    compute_statistics,
    ScoreType,
)

@pytest.fixture
def test_archive(tmp_path):
    """Create a minimal test archive with realistic data."""
    archive_root = tmp_path / "results" / "run_1"
    archive_root.mkdir(parents=True)

    # Create two agent iterations with two benchmarks each
    for agent_i in range(2):  # agent_0 and agent_1
        agent_dir = archive_root / f"agent_{agent_i}"
        agent_dir.mkdir()

        # Add description
        desc_file = agent_dir / "agent_code" / "description.txt"
        desc_file.parent.mkdir(parents=True)
        desc_file.write_text(f"Test agent version {agent_i}")

        # Create metadata.json
        metadata = {
            "experiment_start_timestamp": "2025-03-15T11:24:37.912936",
            "python_version": "3.12.9 | packaged by conda-forge | (main, Mar 4 2025, 22:48:41) [GCC 13.3.0]",
            "executable": "/usr/bin/python",
            "agent_iteration": agent_i,
            "git_commit": "21c6c657b2835445e8824c04b7527a655a215751",
            "current_benchmark_idx": -1
        }
        with open(agent_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        for bench_name in ["gsm8k", "math"]:
            bench_dir = agent_dir / "benchmarks" / bench_name
            bench_dir.mkdir(parents=True)

            # Create realistic results.jsonl based on provided example
            results = [
                {
                    "problem_id": f"{bench_name}_{i}",
                    "timestamp": (datetime.now() + timedelta(seconds=i)).isoformat(),
                    "score": float(i % 2),  # Alternate between 0.0 and 1.0
                    "tokens_used": 57697 + i * 1000,
                    "num_cached_tokens": i * 500,
                    "cost_estimate": 0.05297255 + i * 0.01,
                    "wall_time": 120.31996870040894 - i * 10,  # Vary the times
                    "timed_out": i == 3,  # Make one problem timeout
                    "cost_threshold_exceeded": False
                }
                for i in range(4)  # 4 problems per benchmark
            ]

            with open(bench_dir / "results.jsonl", "w") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")

            # Create corresponding perf.json based on provided example
            perf_data = {
                "avg_score": sum(r["score"] for r in results) / len(results),
                "tokens": sum(r["tokens_used"] for r in results),
                "num_cached_tokens": sum(r["num_cached_tokens"] for r in results),
                "cost": sum(r["cost_estimate"] for r in results),
                "time": sum(r["wall_time"] for r in results)
            }

            with open(bench_dir / "perf.json", "w") as f:
                json.dump(perf_data, f)

            # Create trace directory with summaries
            for result in results:
                prob_dir = bench_dir / "traces" / result["problem_id"]
                prob_dir.mkdir(parents=True)

                # Create a meaningful summary based on performance
                if result["score"] > 0:
                    summary = (
                        f"Analysis for problem {result['problem_id']}\n"
                        f"Score: {result['score']}\n"
                        "The agent successfully solved this problem using an efficient approach.\n"
                        f"Used {result['tokens_used']} tokens ({result['num_cached_tokens']} cached)\n"
                        f"Cost: ${result['cost_estimate']:.4f}\n"
                        f"Time: {result['wall_time']:.2f}s\n"
                        "Key steps: problem decomposition, accurate calculation, clear answer format."
                    )
                else:
                    summary = (
                        f"Analysis for problem {result['problem_id']}\n"
                        f"Score: {result['score']}\n"
                        "The agent failed to solve this problem due to:\n"
                        "- Incorrect problem interpretation\n"
                        "- Calculation errors\n"
                        f"Used {result['tokens_used']} tokens ({result['num_cached_tokens']} cached)\n"
                        f"Cost: ${result['cost_estimate']:.4f}\n"
                        f"Time: {result['wall_time']:.2f}s"
                    )
                    if result["timed_out"]:
                        summary += "\nExecution timed out before completion."

                (prob_dir / "summary.txt").write_text(summary)
                (prob_dir / "execution_tree.txt").write_text(
                    "Mock execution tree for test purposes...\n"
                    "agent_call -> tool_call -> result"
                )

    return archive_root

def test_core_statistics(test_archive):
    """Test basic statistical analysis of archive runs."""
    analyzer = ArchiveAnalyzer(test_archive)
    df = analyzer.analyze_run()

    # Check basic structure
    assert len(df) == 2  # Two iterations
    assert "gsm8k_score" in df.columns
    assert "math_score" in df.columns

    # Verify metric calculations
    assert all(df["avg_tokens"] > 0)
    assert all(df["avg_cost"] > 0)
    assert all(df["avg_time"] > 0)
    assert all(df["utility_score"] >= 0)

    # Check specific values based on our synthetic data
    # Each benchmark has 4 problems, alternating scores of 0 and 1
    assert abs(df.loc[0, "gsm8k_score"] - 0.5) < 1e-6  # Should be 0.5
    assert abs(df.loc[0, "math_score"] - 0.5) < 1e-6  # Should be 0.5

def test_problem_analysis(test_archive):
    """Test problem-level analysis functions."""
    analyzer = ArchiveAnalyzer(test_archive)

    # Test worst problems analysis
    worst = analyzer.get_worst_performing_problems(
        iteration=0,
        n=2,
        benchmark="gsm8k"
    )

    assert len(worst) == 2
    assert all(row["score"] == 0.0 for _, row in worst.iterrows())
    assert all("failed to solve" in row["summary"] for _, row in worst.iterrows())

    # Test best problems analysis
    best = analyzer.get_best_performing_problems(
        iteration=0,
        n=2,
        benchmark="gsm8k"
    )

    assert len(best) == 2
    assert all(row["score"] == 1.0 for _, row in best.iterrows())
    assert all("successfully solved" in row["summary"] for _, row in best.iterrows())

def test_trend_analysis(test_archive):
    """Test trend analysis functions."""
    analyzer = ArchiveAnalyzer(test_archive)
    trend_df = analyzer.get_improvement_trend()

    assert len(trend_df) == 2  # Two iterations
    assert "mean_score" in trend_df.columns
    assert "rolling_avg_3" in trend_df.columns
    assert "rolling_avg_5" in trend_df.columns

    # Check rolling averages
    # With only two points, rolling_avg_3 should equal mean_score
    assert all(trend_df["rolling_avg_3"] == trend_df["mean_score"])

def test_statistical_measures(test_archive):
    """Test statistical measures and confidence intervals."""
    analyzer = ArchiveAnalyzer(test_archive)
    scores_df, summaries_df = analyzer.get_problem_scores_by_iteration()

    stats = compute_statistics(scores_df, summaries_df)

    # Check basic statistics properties
    assert all(stats["ci_lower"] <= stats["target_score"])
    assert all(stats["target_score"] <= stats["ci_upper"])
    assert all(stats["std_score"] >= 0)  # Standard deviation should be non-negative

    # Check that CI width is reasonable (not too wide or narrow)
    ci_width = stats["ci_upper"] - stats["ci_lower"]
    assert all(ci_width >= 0)  # CIs should have positive width
    assert all(ci_width <= 1)  # For 0-1 scores, CI width should be <= 1

def test_edge_cases(test_archive):
    """Test handling of edge cases."""
    analyzer = ArchiveAnalyzer(test_archive)

    # Test non-existent iteration
    with pytest.raises(ValueError):
        analyzer.get_worst_performing_problems(999)

    # Test non-existent benchmark
    with pytest.raises(ValueError):
        analyzer.get_worst_performing_problems(0, benchmark="nonexistent")

    # Test empty directories
    empty_dir = test_archive / "agent_2" / "benchmarks" / "empty"
    empty_dir.mkdir(parents=True)

    # Should handle empty benchmark gracefully
    df = analyzer.analyze_run()
    assert "empty_score" not in df.columns


# Lineage ----------------------------------------------------------------------

from src.utils.lineage import write_lineage


@pytest.fixture
def branching_archive(tmp_path):
    """An archive whose agents form a tree rather than a line:

        0 ── 1 ── 3
        │    └─── 4
        └─── 2 (failed)
        5 (no lineage record, as in archives made before lineage tracking)
    """
    root = tmp_path / "run_1"
    for i in range(6):
        (root / f"agent_{i}").mkdir(parents=True)
    write_lineage(root / "agent_0", id=0, parent_id=None, status="ok", generation=0)
    write_lineage(root / "agent_1", id=1, parent_id=0, meta_agent_id=0, status="ok", generation=1)
    write_lineage(root / "agent_2", id=2, parent_id=0, meta_agent_id=0, status="failed", generation=1)
    write_lineage(root / "agent_3", id=3, parent_id=1, meta_agent_id=1, status="ok", generation=2)
    write_lineage(root / "agent_4", id=4, parent_id=1, meta_agent_id=3, status="ok", generation=2)
    # Things that are not agents
    (root / "problem_data").mkdir()
    (root / "agent_notes").mkdir()
    return root


def test_get_lineage(branching_archive):
    lineage = ArchiveAnalyzer(branching_archive).get_lineage()

    assert sorted(lineage) == [0, 1, 2, 3, 4, 5]
    assert lineage[0]["parent_id"] is None
    assert lineage[0]["generation"] == 0
    assert lineage[4]["parent_id"] == 1
    assert lineage[4]["meta_agent_id"] == 3
    assert lineage[2]["status"] == "failed"
    # An agent with no record is usable, with unknown ancestry
    assert lineage[5] == {
        "id": 5,
        "parent_id": None,
        "meta_agent_id": None,
        "status": "ok",
        "generation": None,
    }


def test_get_lineage_legacy_root_is_generation_zero(tmp_path):
    (tmp_path / "agent_0").mkdir()
    assert ArchiveAnalyzer(tmp_path).get_lineage()[0]["generation"] == 0


def test_get_children(branching_archive):
    analyzer = ArchiveAnalyzer(branching_archive)
    assert analyzer.get_children(0) == [1, 2]
    assert analyzer.get_children(1) == [3, 4]
    assert analyzer.get_children(3) == []
    assert analyzer.get_children(99) == []


def test_get_ancestors(branching_archive):
    analyzer = ArchiveAnalyzer(branching_archive)
    assert analyzer.get_ancestors(4) == [1, 0]  # nearest first
    assert analyzer.get_ancestors(2) == [0]
    assert analyzer.get_ancestors(0) == []
    assert analyzer.get_ancestors(5) == []
    assert analyzer.get_ancestors(99) == []


def test_get_ancestors_stops_on_a_cycle(tmp_path):
    for i in (1, 2):
        (tmp_path / f"agent_{i}").mkdir()
    write_lineage(tmp_path / "agent_1", id=1, parent_id=2, status="ok")
    write_lineage(tmp_path / "agent_2", id=2, parent_id=1, status="ok")
    assert ArchiveAnalyzer(tmp_path).get_ancestors(1) == [2]


def test_formatted_overview_shows_parent(test_archive):
    analyzer = ArchiveAnalyzer(test_archive)

    def rows():
        lines = [
            line
            for line in analyzer.analyze_run_formatted().splitlines()
            if line.startswith("|")
        ]
        cells = [[c.strip() for c in line.strip("|").split("|")] for line in lines]
        return cells[0], cells[2:]

    # Without lineage records the parent is unknown
    headers, body = rows()
    assert headers[:2] == ["Iteration", "Parent"]
    assert [r[1] for r in body] == ["-", "-"]

    write_lineage(test_archive / "agent_0", id=0, parent_id=None, status="ok")
    write_lineage(test_archive / "agent_1", id=1, parent_id=0, status="ok")
    headers, body = rows()
    assert [r[:2] for r in body] == [["0", "-"], ["1", "0"]]


# Paired comparison ------------------------------------------------------------

import numpy as np
import pandas as pd

from src.utils.archive_analysis import PairedDelta, paired_delta


def scores_frame(scores):
    """{iteration: {benchmark: [score per problem]}} -> a problem scores frame."""
    rows = []
    for iteration, benchmarks in scores.items():
        for benchmark, values in benchmarks.items():
            for n, value in enumerate(values):
                if value is None:  # this agent did not run this problem
                    continue
                rows.append(
                    {
                        "iteration": iteration,
                        "benchmark": benchmark,
                        "problem_id": f"{benchmark}_{n}",
                        "score": float(value),
                        "utility": float(value) / 2,
                    }
                )
    return pd.DataFrame(rows)


def test_paired_delta_point_estimate():
    df = scores_frame({0: {"a": [0, 0, 1, 1]}, 1: {"a": [1, 0, 1, 1]}})

    delta = paired_delta(df, 1, 0)

    assert delta.mean == pytest.approx(0.25)
    assert delta.ci_lower <= delta.mean <= delta.ci_upper
    assert (delta.n_problems, delta.n_benchmarks) == (4, 1)
    # The comparison is of the first agent relative to the second
    assert paired_delta(df, 0, 1).mean == pytest.approx(-0.25)


def test_paired_delta_weights_benchmarks_equally():
    # Like the overall score: the mean over benchmarks of the benchmark means
    df = scores_frame(
        {
            0: {"small": [0, 0], "large": [1] * 8},
            1: {"small": [1, 1], "large": [1] * 8},
        }
    )
    delta = paired_delta(df, 1, 0)
    assert delta.mean == pytest.approx(0.5)  # not 2 / 10
    assert (delta.n_problems, delta.n_benchmarks) == (10, 2)


def test_paired_delta_matches_difference_of_overall_scores(test_archive):
    analyzer = ArchiveAnalyzer(test_archive)
    scores_df, summaries_df = analyzer.get_problem_scores_by_iteration()
    stats = compute_statistics(scores_df, summaries_df).set_index("iteration")

    delta = analyzer.paired_delta(1, 0)

    assert delta.mean == pytest.approx(
        stats.loc[1, "target_score"] - stats.loc[0, "target_score"]
    )
    assert delta.n_problems == 8


def test_paired_delta_uses_only_shared_problems():
    df = scores_frame(
        {
            0: {"a": [0, 0, 1, None], "b": [1, 1]},
            1: {"a": [1, None, 1, 1]},
        }
    )
    delta = paired_delta(df, 1, 0)
    # Shared: a_0 (0 -> 1) and a_2 (1 -> 1)
    assert delta.mean == pytest.approx(0.5)
    assert (delta.n_problems, delta.n_benchmarks) == (2, 1)


def test_paired_delta_averages_repeated_runs_of_a_problem():
    df = scores_frame({0: {"a": [0, 0]}, 1: {"a": [1, 1]}})
    repeat = df[(df["iteration"] == 1) & (df["problem_id"] == "a_0")].assign(score=0.0)
    df = pd.concat([df, repeat])

    delta = paired_delta(df, 1, 0)

    assert delta.mean == pytest.approx(0.75)  # a_0: 0.5 - 0, a_1: 1 - 0
    assert delta.n_problems == 2


def test_paired_delta_without_shared_problems_is_none():
    df = scores_frame({0: {"a": [1, 0]}, 1: {"b": [1, 0]}})
    assert paired_delta(df, 1, 0) is None
    assert paired_delta(df, 1, 7) is None
    assert paired_delta(pd.DataFrame(), 1, 0) is None


def test_paired_delta_is_deterministic():
    rng = np.random.default_rng(0)
    # Partial-credit scores, so that interval bounds are not confined to a grid
    df = scores_frame({i: {"a": rng.random(30).tolist()} for i in (0, 1)})
    assert paired_delta(df, 1, 0) == paired_delta(df, 1, 0)
    assert paired_delta(df, 1, 0, seed=1) != paired_delta(df, 1, 0, seed=2)


def test_paired_delta_score_type():
    df = scores_frame({0: {"a": [0, 0, 1, 1]}, 1: {"a": [1, 1, 1, 1]}})
    assert paired_delta(df, 1, 0, score_type="utility_score").mean == pytest.approx(0.25)


def test_paired_interval_is_narrower_than_the_unpaired_one():
    # Both agents solve the easy problems and fail the hard ones; agent 1 also
    # solves 6 problems that agent 0 does not.
    base = [1.0] * 20 + [0.0] * 20
    improved = [1.0] * 26 + [0.0] * 14
    df = scores_frame({0: {"a": base}, 1: {"a": improved}})

    delta = paired_delta(df, 1, 0)

    assert delta.mean == pytest.approx(0.15)
    assert delta.significant
    # The independent-samples interval for the same data includes zero
    se_unpaired = np.sqrt(np.var(base, ddof=1) / 40 + np.var(improved, ddof=1) / 40)
    assert 0.15 - 1.96 * se_unpaired < 0
    assert delta.ci_upper - delta.ci_lower < 2 * 1.96 * se_unpaired / 1.5


def test_paired_interval_covers_the_true_difference():
    # Problems vary in difficulty; agent 1 is truly better by 0.1 on each
    rng = np.random.default_rng(42)
    n_problems, n_trials, covered = 60, 300, 0
    for trial in range(n_trials):
        difficulty = rng.uniform(0.1, 0.8, n_problems)
        df = scores_frame(
            {
                0: {"a": (rng.random(n_problems) < difficulty).astype(float)},
                1: {"a": (rng.random(n_problems) < difficulty + 0.1).astype(float)},
            }
        )
        delta = paired_delta(df, 1, 0, n_resamples=500, seed=trial)
        covered += delta.ci_lower <= 0.1 <= delta.ci_upper
    assert 0.90 <= covered / n_trials <= 0.99


def test_paired_delta_significance():
    assert PairedDelta(0.2, 0.05, 0.35, 10, 1).significant
    assert PairedDelta(-0.2, -0.35, -0.05, 10, 1).significant
    assert not PairedDelta(0.2, -0.05, 0.35, 10, 1).significant
    assert not PairedDelta(0.0, 0.0, 0.0, 10, 1).significant
    # One problem cannot say anything about variability
    assert not PairedDelta(1.0, 1.0, 1.0, 1, 1).significant


def test_paired_delta_formatting():
    assert str(PairedDelta(0.125, -0.01, 0.26, 18, 1)) == "+0.125 [-0.010, +0.260]"
    assert str(PairedDelta(-0.25, -0.4, -0.1, 18, 1)) == "-0.250 [-0.400, -0.100] *"


def test_formatted_overview_shows_change_against_parent(test_archive):
    analyzer = ArchiveAnalyzer(test_archive)
    write_lineage(test_archive / "agent_1", id=1, parent_id=0, status="ok")

    output = analyzer.analyze_run_formatted()

    table = [line for line in output.splitlines() if line.startswith("|")]
    cells = [[c.strip() for c in line.strip("|").split("|")] for line in table]
    assert cells[0][:3] == ["Iteration", "Parent", "vs Parent"]
    assert cells[2][2] == "-"  # the root agent has no parent
    assert cells[3][2] == str(analyzer.paired_delta(1, 0))
    assert "vs Parent:" in output.splitlines()[-1]


def test_formatted_comparison_includes_paired_differences(test_archive):
    analyzer = ArchiveAnalyzer(test_archive)

    output = analyzer.compare_iterations_formatted([0, 1])

    assert "Paired Score Differences" in output
    assert f"Iteration 1 vs 0: {analyzer.paired_delta(1, 0)} (8 shared problems)" in output


# Repeated runs ----------------------------------------------------------------

from src.utils.archive_analysis import base_problem_id, per_problem_scores, result_id


def test_result_ids():
    assert result_id("p1") == "p1"
    assert result_id("p1", 2) == "p1#2"
    assert base_problem_id("p1#2", 2) == "p1"
    assert base_problem_id("p1", 0) == "p1"
    # A '#' in a problem's own id is not a repeat marker
    assert base_problem_id("issue#12", 0) == "issue#12"
    assert base_problem_id(result_id("issue#12", 3), 3) == "issue#12"


def add_repeat(archive, agent, benchmark, problem_id, repeat, score):
    path = archive / f"agent_{agent}" / "benchmarks" / benchmark / "results.jsonl"
    row = {
        "problem_id": f"{problem_id}#{repeat}",
        "repeat": repeat,
        "score": score,
        "tokens_used": 1,
        "num_cached_tokens": 0,
        "cost_estimate": 0.01,
        "wall_time": 1.0,
        "timed_out": False,
    }
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def test_repeats_are_runs_of_the_same_problem(test_archive):
    # gsm8k_0 scored 0 in its first run; add two more runs of it
    add_repeat(test_archive, 1, "gsm8k", "gsm8k_0", 1, 1.0)
    add_repeat(test_archive, 1, "gsm8k", "gsm8k_0", 2, 1.0)
    analyzer = ArchiveAnalyzer(test_archive)

    scores_df, summaries_df = analyzer.get_problem_scores_by_iteration()

    runs = scores_df[(scores_df["iteration"] == 1) & (scores_df["problem_id"] == "gsm8k_0")]
    assert sorted(runs["repeat"]) == [0, 1, 2]
    means = per_problem_scores(scores_df)
    assert len(means[means["iteration"] == 1]) == 8  # still 8 problems
    row = means[(means["iteration"] == 1) & (means["problem_id"] == "gsm8k_0")]
    assert row["score"].item() == pytest.approx(2 / 3)
    # The confidence interval counts problems, not runs
    stats = compute_statistics(scores_df, summaries_df).set_index("iteration")
    assert stats.loc[1, "num_problems"] == 8
    assert stats.loc[0, "num_problems"] == 8
    # And the paired comparison is of the problems' mean scores
    delta = analyzer.paired_delta(1, 0)
    assert delta.n_problems == 8
    assert delta.mean == pytest.approx((2 / 3) / 4 / 2)
