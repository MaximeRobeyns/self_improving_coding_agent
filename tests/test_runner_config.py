"""Tests for runner settings: defaults, config file, command line."""
from pathlib import Path

import pytest

from runner_config import RunnerSettings, load_config_file, resolve_settings
from search.evaluation import EvaluationConfig
from search.gate import GateConfig

REPO_ROOT = Path(__file__).parent.parent


def write(tmp_path, text):
    path = tmp_path / "run.yaml"
    path.write_text(text)
    return path


def test_defaults_are_the_original_behaviour():
    settings = resolve_settings()

    assert (settings.iterations, settings.workers, settings.children) == (20, 8, 1)
    assert (settings.selection, settings.seed, settings.meta_agent) == ("greedy_ci", None, "best")
    assert settings.gate == GateConfig()
    # Evaluation settings are left to the experiment, or EvaluationConfig
    for name in ("benchmarks", "subset_size", "timeout_s", "cost_limit_usd", "repeats"):
        assert getattr(settings, name) is None
    defaults = EvaluationConfig()
    assert (defaults.benchmarks, defaults.subset_size, defaults.benchmark_seed) == (("gsm8k",), 18, 1)
    assert (defaults.timeout_s, defaults.cost_limit_usd) == (600, 3.0)


def test_config_file(tmp_path):
    path = write(
        tmp_path,
        """
        children: 3
        selection: thompson
        meta-agent: seed          # dashes are fine too
        gate_checks: [import, smoke]
        benchmarks: [gsm8k, humaneval]
        subset_size: 0
        timeout_s: 900
        cost_limit_usd: 1.5
        """,
    )

    settings = resolve_settings(path)

    assert (settings.children, settings.selection, settings.meta_agent) == (3, "thompson", "seed")
    assert settings.gate.checks == ("import", "smoke")
    assert settings.benchmarks == ["gsm8k", "humaneval"]
    assert (settings.subset_size, settings.timeout_s, settings.cost_limit_usd) == (0, 900, 1.5)
    assert settings.iterations == 20  # not in the file


def test_command_line_wins_over_the_file(tmp_path):
    path = write(tmp_path, "children: 3\nselection: thompson\nrepeats: 2\n")

    settings = resolve_settings(path, {"children": 5, "selection": None, "workers": 2})

    assert settings.children == 5  # command line
    assert settings.selection == "thompson"  # file: not given on the command line
    assert settings.repeats == 2
    assert settings.workers == 2


def test_empty_file_is_all_defaults(tmp_path):
    assert resolve_settings(write(tmp_path, "# nothing set\n")) == RunnerSettings()


@pytest.mark.parametrize(
    "text, problem",
    [
        ("childern: 3", "childern"),  # misspelt
        ("children: 0", "children"),
        ("children: many", "children"),
        ("selection: best_first", "selection"),
        ("meta_agent: strongest", "meta_agent"),
        ("meta_agent: -2", "meta_agent"),
        ("gate_checks: [import, lint]", "gate_checks"),
        ("holdout_fraction: 1.0", "holdout_fraction"),
        ("timeout_s: 0", "timeout_s"),
        ("benchmarks: []", "benchmarks"),
        ("subset_size: -1", "subset_size"),
    ],
)
def test_invalid_settings_are_named(tmp_path, text, problem):
    with pytest.raises(ValueError, match=problem):
        resolve_settings(write(tmp_path, text))


def test_unreadable_config_files(tmp_path):
    with pytest.raises(ValueError, match="Cannot read config file"):
        load_config_file(tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_config_file(write(tmp_path, "children: [3"))
    with pytest.raises(ValueError, match="name: value"):
        load_config_file(write(tmp_path, "- children\n- 3\n"))


def test_meta_agent_forms():
    assert resolve_settings(cli={"meta_agent": 4}).meta_agent == 4
    assert resolve_settings(cli={"meta_agent": "4"}).meta_agent == 4
    assert resolve_settings(cli={"meta_agent": "parent"}).meta_agent == "parent"


def test_example_config_is_valid():
    settings = resolve_settings(REPO_ROOT / "run.example.yaml")

    assert settings.children == 3 and settings.selection == "score_child_prop"
    assert settings.gate.retries == 1
    # Its evaluation section spells out the defaults
    given = {
        name: getattr(settings, name)
        for name in EvaluationConfig.__dataclass_fields__
        if name in RunnerSettings.model_fields
    }
    assert EvaluationConfig(**given) == EvaluationConfig()
    # And it documents every setting
    text = (REPO_ROOT / "run.example.yaml").read_text()
    for name in RunnerSettings.model_fields:
        assert f"{name}:" in text, name
