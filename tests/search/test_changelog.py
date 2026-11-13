"""Tests for the generated 'other branches' section of agent change logs."""
import json

from base_agent.src.utils.archive_analysis import ArchiveAnalyzer
from base_agent.src.utils.lineage import write_lineage
from search.changelog import (
    BLOCK_END,
    BLOCK_START,
    render_other_branches,
    update_change_log,
)


def make_agent(root, agent_id, parent_id, status="ok", scores=None, description=None):
    agent_dir = root / f"agent_{agent_id}"
    (agent_dir / "agent_code").mkdir(parents=True)
    if description is not None:
        (agent_dir / "agent_code" / "description.txt").write_text(description)
    if parent_id is not ...:
        write_lineage(agent_dir, id=agent_id, parent_id=parent_id, status=status)
    if scores is not None:
        bench_dir = agent_dir / "benchmarks" / "gsm8k"
        bench_dir.mkdir(parents=True)
        results = [
            {
                "problem_id": f"p{n}",
                "score": float(s),
                "tokens_used": 100,
                "num_cached_tokens": 0,
                "cost_estimate": 0.01,
                "wall_time": 1.0,
                "timed_out": False,
            }
            for n, s in enumerate(scores)
        ]
        (bench_dir / "results.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in results)
        )
        perf = {
            "avg_score": sum(scores) / len(scores),
            "tokens": 100 * len(scores),
            "num_cached_tokens": 0,
            "cost": 0.01 * len(scores),
            "time": 1.0 * len(scores),
        }
        (bench_dir / "perf.json").write_text(json.dumps(perf))


def tree(tmp_path):
    """0 -> 1 -> 3, 0 -> 2 (failed), 1 -> 4 (not evaluated), 1 -> 5 (being made)"""
    make_agent(tmp_path, 0, None, scores=[0, 0, 1, 1], description="Base")
    make_agent(tmp_path, 1, 0, scores=[1, 0, 1, 1], description="Adds a diff tool")
    make_agent(tmp_path, 2, 0, status="failed", description="Half an LSP client")
    make_agent(tmp_path, 3, 1, scores=[0, 0, 0, 0], description="Rewrites prompts")
    make_agent(tmp_path, 4, 1, description="Adds a linter")
    make_agent(tmp_path, 5, 1, status="pending")
    return ArchiveAnalyzer(tmp_path)


def test_lists_agents_off_the_line_of_descent(tmp_path):
    text = render_other_branches(tree(tmp_path), parent_id=3)

    assert "This agent was built from agent 3." in text
    # Ancestors (0, 1) and the parent itself (3) are not other branches
    assert "### Agent 1 " not in text
    assert "### Agent 3 " not in text
    assert "### Agent 0 " not in text
    assert "### Agent 2 (built from agent 0): failed to build\n\nHalf an LSP client" in text
    assert "### Agent 4 (built from agent 1): not evaluated yet\n\nAdds a linter" in text
    # Agents still being created (siblings of this one) are left out
    assert "Agent 5" not in text


def test_reports_scores_relative_to_each_agents_parent(tmp_path):
    text = render_other_branches(tree(tmp_path), parent_id=2)
    assert "### Agent 1 (built from agent 0): vs parent: +0.250 [" in text
    assert "### Agent 3 (built from agent 1): vs parent: -0.750 [" in text


def test_nothing_to_list_on_a_single_line(tmp_path):
    make_agent(tmp_path, 0, None)
    make_agent(tmp_path, 1, 0)
    make_agent(tmp_path, 2, 1)
    assert render_other_branches(ArchiveAnalyzer(tmp_path), parent_id=2) is None


def test_agents_of_unknown_ancestry_are_not_listed(tmp_path):
    for agent_id in (0, 1, 2):
        make_agent(tmp_path, agent_id, ...)  # no lineage records
    assert render_other_branches(ArchiveAnalyzer(tmp_path), parent_id=2) is None


def test_long_descriptions_are_cut(tmp_path):
    make_agent(tmp_path, 0, None)
    make_agent(tmp_path, 1, 0)
    make_agent(tmp_path, 2, 0, description="x" * 5000)
    text = render_other_branches(ArchiveAnalyzer(tmp_path), parent_id=1)
    assert "x" * 600 + " [...]" in text
    assert "x" * 601 not in text


def test_section_is_appended(tmp_path):
    path = tmp_path / "agent_change_log.md"
    path.write_text("# Log\n\n## Iteration 1\n")

    update_change_log(path, "## Attempts on other branches\n\nstuff")

    assert path.read_text() == (
        "# Log\n\n## Iteration 1\n\n"
        f"{BLOCK_START}\n## Attempts on other branches\n\nstuff\n{BLOCK_END}\n"
    )


def test_section_is_replaced_not_accumulated(tmp_path):
    path = tmp_path / "agent_change_log.md"
    path.write_text("# Log\n")
    update_change_log(path, "old section")
    # The agent then appended its own entry below the generated section
    path.write_text(path.read_text() + "\n## Iteration 2\n\nNew feature\n")

    update_change_log(path, "new section")

    text = path.read_text()
    assert "old section" not in text
    assert text.count(BLOCK_START) == 1
    assert text == (
        "# Log\n\n## Iteration 2\n\nNew feature\n\n"
        f"{BLOCK_START}\nnew section\n{BLOCK_END}\n"
    )


def test_section_is_removed_when_there_is_nothing_to_list(tmp_path):
    path = tmp_path / "agent_change_log.md"
    path.write_text("# Log\n")
    update_change_log(path, "section")

    update_change_log(path, None)

    assert path.read_text() == "# Log\n"


def test_log_without_a_section_is_left_alone(tmp_path):
    path = tmp_path / "agent_change_log.md"
    path.write_text("# Log\n\n\n## Entry\n")
    update_change_log(path, None)
    assert path.read_text() == "# Log\n\n\n## Entry\n"
    # And a missing log is not created for nothing
    update_change_log(tmp_path / "missing.md", None)
    assert not (tmp_path / "missing.md").exists()
