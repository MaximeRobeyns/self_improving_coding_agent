"""Tests for the per-agent lineage record."""
import json

from src.utils.lineage import (
    LINEAGE_FILENAME,
    get_failed_iterations,
    read_lineage,
    write_lineage,
)


def test_write_then_read_roundtrip(tmp_path):
    agent_dir = tmp_path / "agent_3"
    agent_dir.mkdir()

    write_lineage(agent_dir, id=3, parent_id=1, meta_agent_id=2, status="ok")

    record = read_lineage(agent_dir)
    assert record["id"] == 3
    assert record["parent_id"] == 1
    assert record["meta_agent_id"] == 2
    assert record["status"] == "ok"
    assert "created_at" in record


def test_write_merges_and_keeps_created_at(tmp_path):
    agent_dir = tmp_path / "agent_1"
    agent_dir.mkdir()
    write_lineage(agent_dir, id=1, parent_id=0, status="pending")
    created_at = read_lineage(agent_dir)["created_at"]

    write_lineage(agent_dir, status="failed", failure_reason="boom")

    record = read_lineage(agent_dir)
    assert record["status"] == "failed"
    assert record["failure_reason"] == "boom"
    assert record["parent_id"] == 0
    assert record["created_at"] == created_at


def test_read_missing_or_corrupt_returns_none(tmp_path):
    agent_dir = tmp_path / "agent_0"
    agent_dir.mkdir()
    assert read_lineage(agent_dir) is None

    (agent_dir / LINEAGE_FILENAME).write_text("{not json")
    assert read_lineage(agent_dir) is None


def test_write_leaves_no_temp_files(tmp_path):
    agent_dir = tmp_path / "agent_0"
    agent_dir.mkdir()
    write_lineage(agent_dir, id=0, status="ok")
    assert [p.name for p in agent_dir.iterdir()] == [LINEAGE_FILENAME]


def test_get_failed_iterations(tmp_path):
    # agent_0: no lineage file (legacy archive) -> not failed
    (tmp_path / "agent_0").mkdir()
    # agent_1: ok
    (tmp_path / "agent_1").mkdir()
    write_lineage(tmp_path / "agent_1", id=1, status="ok")
    # agent_2: failed
    (tmp_path / "agent_2").mkdir()
    write_lineage(tmp_path / "agent_2", id=2, status="failed")
    # agent_3: still pending (e.g. runner killed mid-improvement) -> not usable
    (tmp_path / "agent_3").mkdir()
    write_lineage(tmp_path / "agent_3", id=3, status="pending")
    # Unrelated directories are ignored
    (tmp_path / "problem_data").mkdir()
    (tmp_path / "agent_x").mkdir()
    (tmp_path / "metadata.json").write_text(json.dumps({}))

    assert get_failed_iterations(tmp_path) == {2, 3}


def test_get_failed_iterations_missing_root(tmp_path):
    assert get_failed_iterations(tmp_path / "nope") == set()


def test_child_generation(tmp_path):
    from src.utils.lineage import child_generation

    (tmp_path / "agent_0").mkdir()
    (tmp_path / "agent_1").mkdir()
    (tmp_path / "agent_2").mkdir()
    write_lineage(tmp_path / "agent_1", id=1, parent_id=0, status="ok", generation=1)

    # The root agent is generation 0 even without a record
    assert child_generation(tmp_path, 0) == 1
    assert child_generation(tmp_path, 1) == 2
    # Unknown ancestry stays unknown
    assert child_generation(tmp_path, 2) is None
    assert child_generation(tmp_path, 99) is None


def test_get_pruned_iterations(tmp_path):
    from src.utils.lineage import get_pruned_iterations

    for i in range(4):
        (tmp_path / f"agent_{i}").mkdir()
    write_lineage(tmp_path / "agent_1", id=1, status="ok", evaluation={"promoted": True})
    write_lineage(tmp_path / "agent_2", id=2, status="ok", evaluation={"promoted": False})
    write_lineage(tmp_path / "agent_3", id=3, status="ok")

    assert get_pruned_iterations(tmp_path) == {2}
    assert get_pruned_iterations(tmp_path / "nope") == set()
