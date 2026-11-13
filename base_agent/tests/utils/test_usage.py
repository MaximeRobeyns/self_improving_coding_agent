"""Tests for the usage report of an agent run."""
import json

import pytest

from src.utils.usage import USAGE_FILENAME, read_usage, write_usage


def test_roundtrip(tmp_path):
    write_usage(tmp_path / "logs", tokens=1200, cached_tokens=300, cost=0.0421, time=17.5)

    assert read_usage(tmp_path / "logs") == {
        "tokens": 1200,
        "cached_tokens": 300,
        "cost": 0.0421,
        "time": 17.5,
    }
    assert [p.name for p in (tmp_path / "logs").iterdir()] == [USAGE_FILENAME]


def test_write_replaces_an_earlier_report(tmp_path):
    write_usage(tmp_path, 1, 0, 0.1, 1.0)
    write_usage(tmp_path, 2, 1, 0.2, 2.0)
    assert read_usage(tmp_path)["tokens"] == 2


@pytest.mark.parametrize(
    "content",
    [None, "", "{not json", "[]", '{"tokens": 1}', '{"tokens": "many", "cached_tokens": 0, "cost": 0, "time": 0}'],
)
def test_missing_or_invalid_report_reads_as_none(tmp_path, content):
    if content is not None:
        (tmp_path / USAGE_FILENAME).write_text(content)
    assert read_usage(tmp_path) is None


def test_write_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    write_usage(blocker / "logs", 1, 0, 0.1, 1.0)  # cannot create the directory
    assert read_usage(blocker / "logs") is None
