"""
Settings of a runner invocation.

Every setting can be given on the command line or in a YAML file passed with
--config; the command line wins. See run.example.yaml.
"""

from pathlib import Path

import yaml

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from search.gate import GateConfig
from search.selection import DEFAULT_SELECTION_STRATEGY, SELECTION_STRATEGIES

META_AGENT_CHOICES = ("best", "parent", "seed")


class RunnerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The search
    iterations: int = Field(20, ge=0)
    workers: int = Field(8, ge=1)
    children: int = Field(1, ge=1)
    selection: str = DEFAULT_SELECTION_STRATEGY
    seed: int | None = None
    meta_agent: str | int = "best"
    meta_model: str | None = None

    # The validation gate
    gate_checks: str = "import,tests,smoke"
    gate_retries: int = Field(0, ge=0)
    gate_model: str | None = None

    # The evaluation. Unset means: what the experiment already uses, or the
    # default for a new experiment (see search.evaluation.EvaluationConfig).
    benchmarks: list[str] | None = None
    subset_size: int | None = Field(None, ge=0)  # 0: all problems
    benchmark_seed: int | None = None
    timeout_s: int | None = Field(None, gt=0)
    cost_limit_usd: float | None = Field(None, gt=0)
    repeats: int | None = Field(None, ge=1)
    holdout_fraction: float | None = Field(None, ge=0, lt=1)
    stage1_size: int | None = Field(None, ge=0)
    promote_threshold: float | None = None

    @field_validator("selection")
    @classmethod
    def _known_selection(cls, value: str) -> str:
        if value not in SELECTION_STRATEGIES:
            raise ValueError(f"must be one of {', '.join(SELECTION_STRATEGIES)}")
        return value

    @field_validator("meta_agent")
    @classmethod
    def _known_meta_agent(cls, value: str | int) -> str | int:
        if isinstance(value, bool) or (isinstance(value, int) and value < 0):
            raise ValueError("must be an agent id")
        if isinstance(value, str) and value not in META_AGENT_CHOICES:
            if value.isdigit():
                return int(value)
            raise ValueError(f"must be one of {', '.join(META_AGENT_CHOICES)} or an agent id")
        return value

    @field_validator("gate_checks", mode="before")
    @classmethod
    def _gate_checks(cls, value) -> str:
        # In a file, the checks may be written as a list
        if isinstance(value, (list, tuple)):
            value = ",".join(map(str, value))
        if not isinstance(value, str):
            raise ValueError("must be a comma separated string or a list")
        GateConfig(checks=GateConfig.parse_checks(value))  # raises for unknown checks
        return value

    @field_validator("benchmarks")
    @classmethod
    def _some_benchmarks(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and not value:
            raise ValueError("must name at least one benchmark")
        return value

    @property
    def gate(self) -> GateConfig:
        return GateConfig(
            checks=GateConfig.parse_checks(self.gate_checks),
            retries=self.gate_retries,
            model=self.gate_model,
        )


def load_config_file(path: Path) -> dict:
    """The settings in a YAML config file, with '-' in names read as '_'.

    Raises:
        ValueError: if the file cannot be read or is not a mapping
    """
    try:
        data = yaml.safe_load(Path(path).read_text())
    except OSError as e:
        raise ValueError(f"Cannot read config file {path}: {e}") from None
    except yaml.YAMLError as e:
        raise ValueError(f"Config file {path} is not valid YAML: {e}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain settings as 'name: value' pairs")
    return {str(key).replace("-", "_"): value for key, value in data.items()}


def resolve_settings(config_path: Path | None = None, cli: dict | None = None) -> RunnerSettings:
    """Combine defaults, the config file and the command line, in rising
    order of precedence. `cli` holds the options given on the command line;
    those that were not given are absent or None.

    Raises:
        ValueError: naming each setting that is unknown or invalid
    """
    values = load_config_file(config_path) if config_path else {}
    values.update({key: value for key, value in (cli or {}).items() if value is not None})
    try:
        return RunnerSettings(**values)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or 'settings'}: {error['msg']}"
            for error in e.errors()
        )
        source = f" (from {config_path} and the command line)" if config_path else ""
        raise ValueError(f"Invalid settings{source}: {problems}") from None
