"""Tests for the experiment report."""
import csv
import json

import pytest

from base_agent.src.utils.lineage import write_lineage
from search import report
from search.evaluation import holdout_dir


def add_results(root, agent_id, scores, cost_per_problem=0.5):
    bench_dir = root / f"agent_{agent_id}" / "benchmarks" / "gsm8k"
    bench_dir.mkdir(parents=True)
    rows = [
        {
            "problem_id": f"p{n}",
            "score": float(score),
            "tokens_used": 100,
            "num_cached_tokens": 0,
            "cost_estimate": cost_per_problem,
            "wall_time": 1.0,
            "timed_out": False,
        }
        for n, score in enumerate(scores)
    ]
    (bench_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    perf = {
        "avg_score": sum(scores) / len(scores),
        "tokens": 100 * len(scores),
        "num_cached_tokens": 0,
        "cost": cost_per_problem * len(scores),
        "time": float(len(scores)),
    }
    (bench_dir / "perf.json").write_text(json.dumps(perf))


def agent(exp_dir, agent_id, parent_id, generation, scores=None, holdout=None, **lineage):
    (exp_dir / f"agent_{agent_id}").mkdir(parents=True)
    lineage.setdefault("status", "ok")
    write_lineage(
        exp_dir / f"agent_{agent_id}",
        id=agent_id,
        parent_id=parent_id,
        generation=generation,
        **lineage,
    )
    if scores is not None:
        add_results(exp_dir, agent_id, scores)
    if holdout is not None:
        add_results(holdout_dir(exp_dir), agent_id, holdout)


def selection(parent_weights, generation, strategy="score_child_prop"):
    return {"strategy": strategy, "seed": 7, "generation": generation, "weights": parent_weights}


@pytest.fixture
def exp_dir(tmp_path):
    """
    0 (0.50) ── 1 (0.75) ── 3 (1.00, best)
           │           └─── 4 (dropped after the first stage)
           └─── 2 (failed)
    """
    exp = tmp_path / "results" / "run_1"
    agent(exp, 0, None, 0, [1, 1, 0, 0], holdout=[1, 0])
    agent(
        exp, 1, 0, 1, [1, 1, 1, 0], holdout=[1, 0],
        meta_agent_id=0, usage={"cost": 2.0},
        selection=selection({"0": 1.0}, 0),
    )
    agent(exp, 2, 0, 1, status="failed", meta_agent_id=0, selection=selection({"0": 1.0}, 0))
    agent(
        exp, 3, 1, 2, [1, 1, 1, 1], holdout=[1, 0],
        meta_agent_id=1, meta_model="GPT4O", usage={"cost": 3.0},
        selection=selection({"0": 0.25, "1": 0.75}, 1),
    )
    agent(
        exp, 4, 1, 2, [0, 0], meta_agent_id=1, usage={"cost": 1.0},
        selection=selection({"0": 0.25, "1": 0.75}, 1),
        evaluation={"promoted": False},
    )
    return exp


def test_collect(exp_dir):
    rows = {row.id: row for row in report.collect(exp_dir)}

    assert sorted(rows) == [0, 1, 2, 3, 4]
    assert (rows[1].score, rows[1].holdout_score) == (0.75, 0.5)
    assert rows[1].vs_parent.mean == pytest.approx(0.25)
    assert rows[0].vs_parent is None
    assert (rows[2].status, rows[2].score, rows[2].usable) == ("failed", None, False)
    assert (rows[4].pruned, rows[4].usable) == (True, False)
    assert (rows[3].meta_agent_id, rows[3].meta_model) == (1, "GPT4O")
    # Benchmarks (archive and held-out problems), and the meta-improvement step
    assert rows[1].eval_cost == pytest.approx(4 * 0.5 + 2 * 0.5)
    assert rows[1].meta_cost == 2.0
    assert report.best_agent(list(rows.values())).id == 3


def test_tree(exp_dir):
    assert report.render_tree(report.collect(exp_dir)).splitlines() == [
        "agent 0  0.500  (holdout 0.500)",
        "├── agent 1  0.750  (holdout 0.500)  vs parent +0.250 [+0.000, +0.750]",
        "│   ├── agent 3  1.000  (holdout 0.500)  vs parent +0.250 [+0.000, +0.750]  <- best",
        "│   └── agent 4  0.000  vs parent -1.000 [-1.000, -1.000] *  [dropped after first stage]",
        "└── agent 2  [failed]",
    ]


def test_tree_of_agents_with_unknown_parents(tmp_path):
    # As in archives from before lineage tracking: every agent is a root
    for agent_id in (0, 1):
        (tmp_path / f"agent_{agent_id}").mkdir()
    add_results(tmp_path, 0, [1, 0])
    assert report.render_tree(report.collect(tmp_path)).splitlines() == [
        "agent 0  0.500  <- best",
        "agent 1  [not evaluated]",
    ]


def test_generations(exp_dir):
    entries = report.generations(report.collect(exp_dir))

    assert [e["generation"] for e in entries] == [0, 1, 2]
    assert [e["agents"] for e in entries] == [[0], [1, 2], [3, 4]]
    assert [e["failed"] for e in entries] == [0, 1, 0]
    assert [e["dropped"] for e in entries] == [0, 0, 1]
    assert [e["best_so_far"] for e in entries] == [0.5, 0.75, 1.0]
    assert [e["best_agent"] for e in entries] == [0, 1, 3]
    assert [e["meta_cost"] for e in entries] == [0.0, 2.0, 4.0]
    assert [e["eval_cost"] for e in entries] == [3.0, 3.0, 4.0]
    assert [e["cumulative_cost"] for e in entries] == [3.0, 8.0, 16.0]


def test_best_so_far_never_decreases_and_ignores_dropped_agents(tmp_path):
    agent(tmp_path, 0, None, 0, [1, 1, 1, 0])
    agent(tmp_path, 1, 0, 1, [1, 0, 0, 0])
    # Scored 1.0, but only on the two first-stage problems
    agent(tmp_path, 2, 1, 2, [1, 1], evaluation={"promoted": False})

    entries = report.generations(report.collect(tmp_path))

    assert [e["best_score"] for e in entries] == [0.75, 0.25, None]
    assert [e["best_so_far"] for e in entries] == [0.75, 0.75, 0.75]
    assert [e["best_agent"] for e in entries] == [0, 0, 0]


def test_legacy_agents_count_as_one_generation_each(tmp_path):
    for agent_id, scores in enumerate([[0, 1], [1, 1], [0, 0]]):
        (tmp_path / f"agent_{agent_id}").mkdir()
        add_results(tmp_path, agent_id, scores)
    entries = report.generations(report.collect(tmp_path))
    assert [(e["generation"], e["best_so_far"]) for e in entries] == [(0, 0.5), (1, 1.0), (2, 1.0)]


def test_selection_log(exp_dir):
    text = report.render_selections(report.collect(exp_dir))
    lines = [[c.strip() for c in line.strip("|").split("|")] for line in text.splitlines()]

    assert lines[0] == [
        "Agent", "Gen", "Strategy", "Seed", "Parent", "P(parent)", "Candidates", "Meta agent",
    ]
    by_agent = {line[0]: line for line in lines[2:]}
    assert sorted(by_agent) == ["1", "2", "3", "4"]  # the root agent was not selected
    assert by_agent["3"] == [
        "3", "1", "score_child_prop", "7", "1", "0.75", "0: 0.25, 1: 0.75", "1 (GPT4O)",
    ]


def test_holdout_comparison(exp_dir):
    text = report.render_holdout(report.collect(exp_dir))
    lines = [[c.strip() for c in line.strip("|").split("|")] for line in text.splitlines()[:5]]

    # Archive scores rise from 0.5 to 1.0; held-out scores do not move
    assert lines[2] == ["0", "0.500", "0.500", "+0.000", "+0.000"]
    assert lines[4] == ["3", "1.000", "0.500", "+0.500", "+0.000"]
    assert "overfitting" in text


def test_no_holdout_section_without_holdout_results(tmp_path):
    agent(tmp_path, 0, None, 0, [1, 0])
    assert report.render_holdout(report.collect(tmp_path)) is None
    text = report.render_experiment(tmp_path)
    assert "## Archive vs holdout" not in text
    assert "Its holdout" not in text


def test_dot(exp_dir):
    dot = report.to_dot(report.collect(exp_dir))

    assert dot.startswith("digraph agents {")
    assert 'a3 [label="agent 3\\n1.000 / holdout 0.500", penwidth=3];' in dot
    assert 'a2 [label="agent 2\\nfailed", style=dashed' in dot
    assert 'a1 -> a3 [label="+0.250"];' in dot
    assert 'a1 -> a4 [label="-1.000*"];' in dot
    assert "a0 -> a2;" in dot


@pytest.fixture
def three_runs(tmp_path):
    """Runs whose best scores end at 1.0, 0.5 and 0.75"""
    dirs = []
    for n, curve in enumerate([[0.5, 0.75, 1.0], [0.5, 0.5, 0.25], [0.5, 0.75]]):
        exp = tmp_path / f"run_{n + 1}"
        for generation, score in enumerate(curve):
            scores = [1.0] * int(score * 4) + [0.0] * (4 - int(score * 4))
            agent(exp, generation, generation - 1 if generation else None, generation, scores)
        dirs.append(exp)
    return dirs


def test_runs_are_compared_by_their_best_score_curves(three_runs):
    comparison = report.compare_runs(three_runs)

    assert [row["scores"] for row in comparison["generations"]] == [
        [0.5, 0.5, 0.5],
        [0.75, 0.5, 0.75],
        [1.0, 0.5, 0.75],  # the third run stopped a generation earlier
    ]
    assert comparison["generations"][0]["std"] == 0.0
    assert comparison["final_mean"] == pytest.approx(0.75)
    assert comparison["final_std"] == pytest.approx(0.25)
    assert "standard deviation 0.250" in report.render_comparison(comparison)


def test_command_line(exp_dir, three_runs, tmp_path, capsys):
    assert report.main([str(exp_dir), "--dot", str(tmp_path / "tree.dot")]) == 0
    out = capsys.readouterr().out
    for heading in ("## Agent tree", "## Progress per generation", "## Archive vs holdout", "## Parent selection"):
        assert heading in out
    assert (tmp_path / "tree.dot").read_text().startswith("digraph agents {")

    curves = tmp_path / "curves.csv"
    assert report.main([*map(str, three_runs), "--csv", str(curves)]) == 0
    assert "Final best score over 3 runs" in capsys.readouterr().out
    rows = list(csv.reader(curves.open()))
    assert rows[0][0] == "generation" and rows[0][-2:] == ["mean", "std"]
    assert rows[3][1:4] == ["1.0", "0.5", "0.75"]

    assert report.main([str(exp_dir), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["experiments"][str(exp_dir)]["agents"]) == 5

    with pytest.raises(SystemExit):
        report.main([str(tmp_path / "missing")])
    with pytest.raises(SystemExit):
        report.main([*map(str, three_runs), "--dot", str(tmp_path / "x.dot")])
