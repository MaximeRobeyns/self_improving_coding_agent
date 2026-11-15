# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

from abc import ABC, abstractmethod
from typing import Any, ClassVar
from pydantic import BaseModel, Field
from ..schemas import dumps
from ..types.common import ArgFormat


class ToolResult(BaseModel):
    """Represents the result of a tool execution."""

    tool_name: str
    success: bool
    duration: float = 0.0  # on tool error paths, duration is often 0
    output: dict[str, Any] | str | None = None
    warnings: str | None = None
    errors: str | None = None
    invocation_id: str = Field(default_factory=lambda: os.urandom(4).hex())

    def __str__(self):
        str_output = self.output if isinstance(self.output, str) else None
        if isinstance(self.output, dict):
            str_output = dumps(self.output, ArgFormat.XML, indent=2)

        tool_response_str = "<TOOL_RESPONSE>"
        tool_response_str += (
            f"\n<STATUS>{'SUCCESS' if self.success else 'FAILURE'}</STATUS>"
        )
        if str_output is not None:
            tool_response_str += f"\n<OUTPUT>{str_output}</OUTPUT>"
        if self.warnings is not None:
            tool_response_str += f"\n<WARNINGS>{self.warnings}</WARNINGS>"
        if self.errors is not None:
            tool_response_str += f"\n<ERRORS>{self.errors}</ERRORS>"
        if self.duration is not None:
            tool_response_str += f"\n<DURATION>{self.duration:.3f}</DURATION>"
        tool_response_str += "\n</TOOL_RESPONSE>"

        return tool_response_str

    def to_plain_string(self):
        str_output = self.output if isinstance(self.output, str) else None
        if isinstance(self.output, dict):
            str_output = dumps(self.output, ArgFormat.JSON, indent=2)

        tool_response_str = f"{self.tool_name} response:"
        tool_response_str += f"\nSuccess: {self.success}"
        if str_output is not None:
            tool_response_str += f"\nResult: {str_output}"
        if self.warnings is not None:
            tool_response_str += f"\nWarnings: {self.warnings}"
        if self.errors is not None:
            tool_response_str += f"\nErrors: {self.errors}"
        if self.duration is not None:
            tool_response_str += f"\nDuration: {self.duration:.3f}"

        return tool_response_str


class ToolInterface(BaseModel, ABC):
    """Abstract interface for all tools"""

    # Class variables
    TOOL_NAME: ClassVar[str]
    TOOL_DESCRIPTION: ClassVar[str]
    EPHEMERAL: ClassVar[bool] = False

    class Config:
        extra = "forbid"

    @abstractmethod
    async def run(self) -> ToolResult:
        """Execute the tool's functionality"""
        pass

    @classmethod
    @abstractmethod
    def generate_examples(cls) -> list[tuple["BaseTool", ToolResult]]:  # noqa: F821
        """Generate example uses of the tool with their expected outputs"""
        pass

    @classmethod
    @abstractmethod
    def to_prompt_format(cls, arg_format: ArgFormat = ArgFormat.XML) -> str:
        """Convert the tool definition to XML format for the unconstrained tool use prompt."""
        pass

    @classmethod
    @abstractmethod
    def to_plain_prompt_format(cls, arg_format: ArgFormat = ArgFormat.JSON) -> str:
        """Convert the tool definition to a formatted string for the constrained tool use prompt.

        NOTE: most providers use JSON-like syntax in their prompts, so
        generating few-shot examples like this tends to work better.
        """
        pass
