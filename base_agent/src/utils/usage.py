# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
The usage report of an agent run: `usage.json` in the run's log directory.

This is how the agent tells whoever started it what the run consumed. (It
also still prints the same numbers as the last line of its output, as
'tokens|cached tokens|cost|duration', which is all that older runners read.)
"""
import os
import json
import logging

from pathlib import Path

logger = logging.getLogger(__name__)

USAGE_FILENAME = "usage.json"


def write_usage(
    logdir: str | Path, tokens: int, cached_tokens: int, cost: float, time: float
) -> None:
    """Write the usage report of a run. Never raises: failing to report usage
    must not fail a run that has otherwise completed."""
    try:
        logdir = Path(logdir)
        logdir.mkdir(parents=True, exist_ok=True)
        usage = {
            "tokens": int(tokens),
            "cached_tokens": int(cached_tokens),
            "cost": float(cost),
            "time": float(time),
        }
        tmp_path = logdir / f".{USAGE_FILENAME}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(usage, f)
        os.replace(tmp_path, logdir / USAGE_FILENAME)
    except Exception as e:
        logger.warning(f"Could not write the usage report to {logdir}: {e}")


def read_usage(logdir: str | Path) -> dict | None:
    """The usage report in a log directory, or None if there is no valid one."""
    try:
        with open(Path(logdir) / USAGE_FILENAME) as f:
            data = json.load(f)
        return {
            "tokens": int(data["tokens"]),
            "cached_tokens": int(data["cached_tokens"]),
            "cost": float(data["cost"]),
            "time": float(data["time"]),
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None
