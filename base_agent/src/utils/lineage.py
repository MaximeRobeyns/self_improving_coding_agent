# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Per-agent lineage record, stored at `agent_{i}/lineage.json` in the archive.

Records where an agent iteration came from and whether the meta-improvement
step that produced it succeeded.
"""
import os
import json
import logging

from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

LINEAGE_FILENAME = "lineage.json"

STATUS_PENDING = "pending"  # meta-improvement started but has not finished
STATUS_OK = "ok"
STATUS_FAILED = "failed"


def read_lineage(agent_dir: str | Path) -> dict | None:
    """Return the lineage record for an agent directory, or None if it has
    none (archives created before lineage tracking) or it is unreadable."""
    path = Path(agent_dir) / LINEAGE_FILENAME
    try:
        with open(path) as f:
            record = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Could not read lineage record {path}: {e}")
        return None
    return record if isinstance(record, dict) else None


def write_lineage(agent_dir: str | Path, **fields) -> dict:
    """Create or update the lineage record for an agent directory.

    The given fields are merged into any existing record. The write is atomic,
    so a crash cannot leave a truncated file behind.
    """
    agent_dir = Path(agent_dir)
    record = read_lineage(agent_dir) or {}
    record.setdefault("created_at", datetime.now().isoformat())
    record.update(fields)

    path = agent_dir / LINEAGE_FILENAME
    tmp_path = agent_dir / f".{LINEAGE_FILENAME}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp_path, path)
    return record


def child_generation(archive_root: str | Path, parent_id: int) -> int | None:
    """The generation (distance from the root agent) of a child of `parent_id`,
    or None if the parent's own generation is unknown."""
    record = read_lineage(Path(archive_root) / f"agent_{parent_id}") or {}
    parent_generation = record.get("generation")
    if parent_generation is None and parent_id == 0:
        parent_generation = 0  # agent_0 is always the root
    if not isinstance(parent_generation, int):
        return None
    return parent_generation + 1


def get_pruned_iterations(archive_root: str | Path) -> set[int]:
    """Agent iterations whose evaluation was stopped after the first stage,
    because they were clearly worse than their parent. They only have results
    for a subset of the problems, so their overall scores are not comparable
    with those of other agents."""
    archive_root = Path(archive_root)
    pruned: set[int] = set()
    if not archive_root.is_dir():
        return pruned
    for agent_dir in archive_root.glob("agent_*"):
        try:
            iteration = int(agent_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        evaluation = (read_lineage(agent_dir) or {}).get("evaluation") or {}
        if evaluation.get("promoted") is False:
            pruned.add(iteration)
    return pruned


def get_failed_iterations(archive_root: str | Path) -> set[int]:
    """Agent iterations in an archive that must not be evaluated or built upon.

    That is every agent whose meta-improvement step failed, or never finished
    ('pending'). Agents with no lineage record are treated as usable.
    """
    archive_root = Path(archive_root)
    failed: set[int] = set()
    if not archive_root.is_dir():
        return failed
    for agent_dir in archive_root.glob("agent_*"):
        try:
            iteration = int(agent_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        record = read_lineage(agent_dir)
        if record is not None and record.get("status", STATUS_OK) != STATUS_OK:
            failed.add(iteration)
    return failed
