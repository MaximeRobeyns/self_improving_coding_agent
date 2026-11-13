# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pydantic import field_validator
from pydantic_settings import BaseSettings

from .types.llm_types import Model


class Settings(BaseSettings):
    # Basic Agent Configuration
    NAME: str = "self_referential_agent"
    LOG_LEVEL: str = "INFO"

    MODEL: Model = Model.SONNET_55
    REASONING_MODEL: Model = Model.O3_MINI
    OVERSIGHT_MODEL: Model = Model.SONNET_55

    @field_validator("MODEL", "REASONING_MODEL", "OVERSIGHT_MODEL", mode="before")
    def parse_model(cls, value):
        """Convert a string model name into a Model enum instance."""
        if isinstance(value, str):
            return Model.from_name(value)
        elif isinstance(value, Model):
            return value
        raise ValueError(f"Invalid model value: {value!r}")

    model_config = {
        "env_prefix": "AGENT_",
        "case_sensitive": True,
        "extra": "allow",  # Allow extra fields from environment
    }


settings = Settings()
