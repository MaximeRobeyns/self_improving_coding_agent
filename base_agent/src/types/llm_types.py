# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Contents:
    - Abstraction classes between providers
    - Model enum
"""
from enum import Enum, auto
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass

from ..types.common import ArgFormat


class FCI(int, Enum):
    """Function calling interface for LLM models."""

    # native tool calling, often using constrained generation, from inference providers
    CONSTRAINED = auto()
    # our custom, unconstrainted XML or JSON-based tool calling mechanism
    UNCONSTRAINED = auto()


class StopReason(str, Enum):
    COMPLETE = "complete"  # Normal completion
    LENGTH = "length"  # Hit token limit (or stop sequence - if reported)
    STOP_TOKEN = "stop_token"  # Hit a defined stop token
    TOOL_CALL = "tool_call"  # tool was called and needs to be evaluated (only for native tool calling)
    ERROR = "error"  # Any error condition


class ReasoningEffort(str, Enum):
    """Reasoning effort for reasoning models"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TextContent(BaseModel):
    text: str
    cache_control: Optional[Dict[str, str]] = None

    def __str__(self) -> str:
        return f"Text Content: {self.text}"


class ReasoningContent(BaseModel):
    text: str
    cache_control: Optional[Dict[str, str]] = None

    def __str__(self) -> str:
        return f"Reasoning CoT: {self.text}"


class ToolCallContent(BaseModel):
    call_id: str
    tool_name: str
    tool_args: Dict[str, Any]
    call_type: FCI | int
    parse_errors: str | None = None
    cache_control: Optional[Dict[str, str]] = None

    def __str__(self) -> str:
        return f"Tool Call for `{self.tool_name}` (id {self.call_id}) with args:\n{self.tool_args}"

class ToolResultContent(BaseModel):
    call_id: str
    content: str
    tool_name: str
    cache_control: Optional[Dict[str, str]] = None

    def __str__(self) -> str:
        return f"Tool Result for `{self.tool_name}` (id {self.call_id}):\n{self.content}"

# TODO: add other content types (e.g. for images, audio and other media) here...


ContentTypes = TextContent | ReasoningContent | ToolCallContent | ToolResultContent

# -------------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenCost:
    """Token cost information for a model.

    All costs are in USD per million tokens.
    """

    input_uncached: float  # Cost per 1M uncached input tokens
    input_cached: float  # Cost per 1M cached input tokens
    cache_write: float  # Cost per 1M tokens for cache writes
    output: float  # Cost per 1M output tokens


class TokenUsage(BaseModel):
    """Standardized token usage tracking for LLM interactions."""

    # Inputs:
    #
    # NOTE: for a given completion, each input token may only be counted in one
    # of the following:
    uncached_prompt_tokens: int = Field(
        default=0, description="Number of uncached input tokens"
    )
    cache_write_prompt_tokens: int = Field(
        default=0, description="Number of uncached input tokens written to cache"
    )
    cached_prompt_tokens: int = Field(
        default=0, description="Number of input tokens with cache hits"
    )

    # Outputs
    completion_tokens: int = Field(
        default=0, description="Total number of output tokens"
    )

    @property
    def input_tokens(self) -> int:
        return (
            self.uncached_prompt_tokens
            + self.cache_write_prompt_tokens
            + self.cached_prompt_tokens
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens for this interaction"""
        return (
            self.uncached_prompt_tokens
            + self.cache_write_prompt_tokens
            + self.cached_prompt_tokens
            + self.completion_tokens
        )

    @property
    def cache_metrics(self) -> dict:
        return {
            "cache_hits": self.cached_prompt_tokens,
            "cache_misses": self.uncached_prompt_tokens - self.cached_prompt_tokens,
            "cache_writes": self.cache_write_prompt_tokens,
        }

    def calculate_cost(self, model_cost: TokenCost) -> float:
        """Calculate the total cost for this usage based on the model's pricing."""
        scale = 1_000_000

        # Calculate input costs
        input_cost = (
            (self.uncached_prompt_tokens * model_cost.input_uncached / scale)
            + (self.cached_prompt_tokens * model_cost.input_cached / scale)
            + (self.cache_write_prompt_tokens * model_cost.cache_write / scale)
        )

        # Calculate output costs
        output_cost = self.completion_tokens * model_cost.output / scale

        return input_cost + output_cost

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented

        return TokenUsage(
            uncached_prompt_tokens=self.uncached_prompt_tokens + other.uncached_prompt_tokens,
            cache_write_prompt_tokens=self.cache_write_prompt_tokens + other.cache_write_prompt_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens + other.cached_prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class Provider(Enum):
    """Supported LLM providers."""

    ANTHROPIC = auto()
    OPENAI = auto()
    DEEPSEEK = auto()
    FIREWORKS = auto()
    GOOGLE = auto()
    GOOGLE_OAI = auto()
    GOOGLE_REST = auto()
    VERTEX = auto()


@dataclass(frozen=True)
class ModelInfo:
    """Complete information about an LLM model."""

    api_name: str  # The model name used in API calls
    provider: Provider  # The LLM provider
    costs: TokenCost  # Token cost information
    max_tokens: int  # Maximum output tokens supported by the model
    supports_caching: bool = False  # Whether the model supports prompt caching
    reasoning_model: bool = False  # Whether this is a reasoning model
    context_window: int = 0  # Maximum total tokens (prompt + completion) supported
    function_calling_interface: FCI = FCI.UNCONSTRAINED  # function calling interface
    preferred_arg_format: ArgFormat = ArgFormat.XML  # XML or JSON preferred
    # What the model's API accepts. The defaults describe the older models;
    # current Claude models reject all three of these with a 400:
    # - a request whose last message is an assistant turn to be continued
    supports_prefill: bool = True
    # - temperature / top_p
    supports_sampling_params: bool = True
    # - a thinking token budget. Models with adaptive thinking decide how much
    #   to think themselves, steered by an effort level.
    adaptive_thinking: bool = False


class Model(Enum):
    """Available LLM models and their configurations.

    Note, all costs are reported in USD per million tokens.
    """

    # Claude 3.x models. These have been retired from the Claude API (Sonnet
    # 3.5 on 2025-10-28, Sonnet 3.7 and Haiku 3.5 on 2026-02-19): requests to
    # them fail. They are kept so that results of past runs can still be
    # costed and analysed.
    HAIKU_35 = ModelInfo(
        api_name="claude-3-5-haiku-20241022",
        provider=Provider.ANTHROPIC,
        costs=TokenCost(
            input_uncached=0.80,  # Base input cost
            input_cached=0.08,  # Cached input cost
            cache_write=1.00,  # Cache write cost
            output=4.00,  # Output cost
        ),
        max_tokens=8192,  # Maximum output tokens for 3.5 Haiku
        supports_caching=True,
        context_window=200000,
        function_calling_interface=FCI.CONSTRAINED,
        preferred_arg_format=ArgFormat.JSON,
    )

    SONNET_35 = ModelInfo(
        api_name="claude-3-5-sonnet-20241022",
        provider=Provider.ANTHROPIC,
        costs=TokenCost(
            input_uncached=3.00,  # Base input cost
            input_cached=0.30,  # Cached input cost
            cache_write=3.75,  # Cache write cost
            output=15.00,  # Output cost
        ),
        max_tokens=8192,  # Maximum output tokens for Claude 3.5 Sonnet
        supports_caching=True,
        context_window=200000,  # 200k context window
        function_calling_interface=FCI.UNCONSTRAINED,
        preferred_arg_format=ArgFormat.XML,
    )

    SONNET_37 = ModelInfo(
        api_name="claude-3-7-sonnet-20250219",
        provider=Provider.ANTHROPIC,
        costs=TokenCost(
            input_uncached=3.00,  # Base input cost
            input_cached=0.30,  # Cached input cost
            cache_write=3.75,  # Cache write cost
            output=15.00,  # Output cost
        ),
        max_tokens=64000,
        # max_tokens=128000,  # Need to set the output-128k-2025-01-30 header for this
        supports_caching=True,
        context_window=200000,  # 200k context window
        reasoning_model=True,
        function_calling_interface=FCI.UNCONSTRAINED,
        preferred_arg_format=ArgFormat.XML,
    )

    # Current Claude models. Prices are Anthropic's first-party API rates in
    # USD per million tokens; cache writes are 1.25x the input price.
    HAIKU_55 = ModelInfo(
        api_name="claude-haiku-5-5",
        provider=Provider.ANTHROPIC,
        costs=TokenCost(
            # For prompts of up to 100K tokens; longer prompts are billed at
            # 0.50 / 2.50, which is not modelled here
            input_uncached=0.10,
            input_cached=0.01,
            cache_write=0.125,
            output=0.50,
        ),
        # The model allows up to 128K output tokens, but only when streaming
        max_tokens=16000,
        supports_caching=True,
        context_window=1_000_000,
        reasoning_model=True,
        function_calling_interface=FCI.UNCONSTRAINED,
        preferred_arg_format=ArgFormat.XML,
        supports_prefill=False,
        supports_sampling_params=False,
        adaptive_thinking=True,
    )

    SONNET_55 = ModelInfo(
        api_name="claude-sonnet-5-5",
        provider=Provider.ANTHROPIC,
        costs=TokenCost(
            input_uncached=2.00,
            input_cached=0.10,
            cache_write=2.50,
            output=10.00,
        ),
        max_tokens=16000,
        supports_caching=True,
        context_window=1_000_000,
        reasoning_model=True,
        function_calling_interface=FCI.UNCONSTRAINED,
        preferred_arg_format=ArgFormat.XML,
        supports_prefill=False,
        supports_sampling_params=False,
        adaptive_thinking=True,
    )

    OPUS_55 = ModelInfo(
        api_name="claude-opus-5-5",
        provider=Provider.ANTHROPIC,
        costs=TokenCost(
            input_uncached=4.00,
            input_cached=0.20,
            cache_write=5.00,
            output=20.00,
        ),
        max_tokens=16000,
        supports_caching=True,
        context_window=1_000_000,
        reasoning_model=True,
        function_calling_interface=FCI.UNCONSTRAINED,
        preferred_arg_format=ArgFormat.XML,
        supports_prefill=False,
        supports_sampling_params=False,
        adaptive_thinking=True,
    )

    SONNET_37_VERTEX = ModelInfo(
        api_name="claude-3-7-sonnet@20250219",
        provider=Provider.VERTEX,
        costs=TokenCost(
            input_uncached=3.00,  # Base input cost
            input_cached=0.30,  # Cached input cost - assuming similar to direct API
            cache_write=3.75,  # Cache write cost - assuming similar to direct API
            output=15.00,  # Output cost - assuming similar to direct API
        ),
        max_tokens=64000,
        supports_caching=True,
        context_window=200000,  # 200k context window
        reasoning_model=True,
        function_calling_interface=FCI.UNCONSTRAINED,
        preferred_arg_format=ArgFormat.XML,
    )

    # GPT-4o
    GPT4O = ModelInfo(
        api_name="gpt-4o-2024-11-20",
        provider=Provider.OPENAI,
        costs=TokenCost(
            input_uncached=2.50,  # Base input cost
            input_cached=1.25,  # Cached input cost
            cache_write=2.50,  # No separate cache write cost
            output=10.00,  # Output cost
        ),
        max_tokens=16384,  # Maximum output tokens for GPT-4o
        supports_caching=True,
        context_window=128000,  # 128k context window
        function_calling_interface=FCI.CONSTRAINED,
        preferred_arg_format=ArgFormat.JSON,
    )

    # GPT-4o-mini
    GPT4O_MINI = ModelInfo(
        api_name="gpt-4o-mini-2024-07-18",
        provider=Provider.OPENAI,
        costs=TokenCost(
            input_uncached=0.150,  # Base input cost
            input_cached=0.075,  # Cached input cost
            cache_write=0.150,  # No separate cache write cost
            output=0.60,  # Output cost
        ),
        max_tokens=16384,  # Maximum output tokens for GPT-4o
        supports_caching=True,
        context_window=128000,  # 128k context window
        function_calling_interface=FCI.CONSTRAINED,
        preferred_arg_format=ArgFormat.JSON,
    )

    # OpenAI o1
    O1 = ModelInfo(
        api_name="o1-2024-12-17",
        provider=Provider.OPENAI,
        costs=TokenCost(
            input_uncached=15.00,
            input_cached=7.50,
            cache_write=15.00,
            output=60.00,
        ),
        max_tokens=100000,
        supports_caching=True,
        reasoning_model=True,
        context_window=200000,
        function_calling_interface=FCI.CONSTRAINED,
        preferred_arg_format=ArgFormat.JSON,
    )

    # OpenAI o3 mini
    O3_MINI = ModelInfo(
        api_name="o3-mini-2025-01-31",
        provider=Provider.OPENAI,
        costs=TokenCost(
            input_uncached=1.10,
            input_cached=0.55,
            cache_write=1.10,
            output=4.40,
        ),
        max_tokens=100000,
        supports_caching=True,
        reasoning_model=True,
        context_window=200000,
        function_calling_interface=FCI.CONSTRAINED,
        preferred_arg_format=ArgFormat.JSON,
    )

    # DeepSeek Chat
    DEEPSEEK_V3 = ModelInfo(
        api_name="deepseek-chat",
        provider=Provider.DEEPSEEK,
        # NOTE: prices increase after 2025-02-08
        costs=TokenCost(
            input_uncached=0.14,  # Base input cost
            input_cached=0.014,  # Cached input cost
            cache_write=0.14,  # Cache write cost
            output=0.28,  # Base output cost
        ),
        max_tokens=8192,  # Maximum output tokens for DeepSeek V3
        supports_caching=True,
        context_window=64000,  # 64k context window on deepseek.com
        function_calling_interface=FCI.UNCONSTRAINED,
        preferred_arg_format=ArgFormat.XML,
    )

    # DeepSeek Reasoner
    DEEPSEEK_R1 = ModelInfo(
        api_name="deepseek-reasoner",
        provider=Provider.DEEPSEEK,
        costs=TokenCost(
            input_uncached=0.55,  # Base input cost
            input_cached=0.14,  # Cached input cost
            cache_write=0.55,  # Cache write cost
            output=2.19,  # Base output cost
        ),
        max_tokens=8192,  # Maximum output tokens for DeepSeek V3
        supports_caching=True,
        reasoning_model=True,
        context_window=64000,  # 64k context window on deepseek.com
        function_calling_interface=FCI.UNCONSTRAINED,
        preferred_arg_format=ArgFormat.XML,
    )

    FIREWORKS_DEEPSEEK_V3 = ModelInfo(
        api_name="accounts/fireworks/models/deepseek-v3",
        provider=Provider.FIREWORKS,
        costs=TokenCost(
            input_uncached=0.9,
            input_cached=0.9,  # no price difference :(
            cache_write=0.9,
            output=0.9,  # same input as output costs
        ),
        max_tokens=16384,
        supports_caching=True,  # yes, and reduces TTFT by ~80%, but no cost savings
        context_window=128000,
        function_calling_interface=FCI.UNCONSTRAINED,
        preferred_arg_format=ArgFormat.XML,
    )

    FIREWORKS_DEEPSEEK_R1 = ModelInfo(
        api_name="accounts/fireworks/models/deepseek-r1",
        provider=Provider.FIREWORKS,
        costs=TokenCost(
            input_uncached=8.00,
            input_cached=8.00,  # no price difference :(
            cache_write=8.00,
            output=8.00,  # same input as output costs
        ),
        max_tokens=20480,
        supports_caching=True,  # yes, and reduces TTFT by ~80%, but no cost savings
        reasoning_model=True,
        context_window=160000,
        function_calling_interface=FCI.UNCONSTRAINED,
        preferred_arg_format=ArgFormat.XML,
    )

    GEMINI_FLASH_2 = ModelInfo(
        api_name="gemini-2.0-flash",
        provider=Provider.GOOGLE_REST,
        costs=TokenCost(
            input_uncached=0.10,
            input_cached=0.025,
            cache_write=0.10,
            output=0.40,
        ),
        max_tokens=8192,
        supports_caching=False,  # coming 31 March
        reasoning_model=False,
        context_window=1_048_576,
        function_calling_interface=FCI.CONSTRAINED,
        preferred_arg_format=ArgFormat.JSON,
    )

    GEMINI_25_PRO = ModelInfo(
        api_name="gemini-2.5-pro-preview-03-25",
        provider=Provider.GOOGLE_REST,
        # TODO: update abstraction to account for 2.50 in > 200k, 15.00 out > 200k
        costs=TokenCost(
            input_uncached=1.25,
            input_cached=1.25,  # caching unavailable
            cache_write=1.25,
            output=10.00,
        ),
        max_tokens=65536,
        supports_caching=False,
        reasoning_model=True,
        context_window=1_048_576,
        function_calling_interface=FCI.CONSTRAINED,
        preferred_arg_format=ArgFormat.JSON,
    )

    @property
    def id(self) -> str:
        """Get the model identifier string used in API calls."""
        return self.value.api_name

    @property
    def provider(self) -> Provider:
        """Get the model's provider."""
        return self.value.provider

    @property
    def token_cost(self) -> TokenCost:
        """Get the model's token cost information."""
        return self.value.costs

    @property
    def max_output_tokens(self) -> int:
        """The maximum number of output tokens this model supports."""
        return self.value.max_tokens

    @property
    def supports_caching(self) -> bool:
        """Check if this model supports prompt caching."""
        return self.value.supports_caching

    @property
    def is_reasoner(self) -> bool:
        """Check if this model supports prompt caching."""
        return self.value.reasoning_model

    @property
    def context_window_size(self) -> int:
        """Get the model's total context window size."""
        return self.value.context_window

    @property
    def supports_prefill(self) -> bool:
        """Whether a request may end in an assistant turn for the model to continue."""
        return self.value.supports_prefill

    @property
    def supports_sampling_params(self) -> bool:
        """Whether the model accepts temperature / top_p."""
        return self.value.supports_sampling_params

    @property
    def adaptive_thinking(self) -> bool:
        """Whether the model sets its own amount of thinking, steered by an
        effort level, rather than taking a thinking token budget."""
        return self.value.adaptive_thinking

    @property
    def fci(self) -> FCI:
        """Get the model's function calling interface"""
        return self.value.function_calling_interface

    @property
    def arg_format(self) -> ArgFormat:
        """Get the model's preferred arg format; XML or JSON"""
        return self.value.preferred_arg_format

    @classmethod
    def from_api_name(cls, api_name: str) -> "Model":
        """Get Model enum from API name."""
        for model in cls:
            if model.value.api_name == api_name:
                return model
        raise ValueError(f"{api_name!r} is not a valid Model")

    @classmethod
    def from_name(cls, name: str) -> "Model":
        """Get Model enum from its string name (e.g., 'SONNET_35')."""
        try:
            return cls[name]
        except KeyError:
            raise ValueError(f"{name!r} is not a valid Model name")

    def __init__(self, info: ModelInfo):
        """Initialize with model info."""
        self._value_ = info  # For compatibility with Enum


MODEL_FAILOVER_MAP = {
    Model.HAIKU_55: Model.SONNET_55,
    Model.SONNET_55: Model.OPUS_55,
    Model.OPUS_55: Model.SONNET_55,
    # The retired Claude models fail over to their current counterparts
    Model.HAIKU_35: Model.HAIKU_55,
    Model.SONNET_35: Model.SONNET_55,
    Model.SONNET_37: Model.SONNET_55,
    Model.SONNET_37_VERTEX: Model.SONNET_55,
    Model.GPT4O: Model.SONNET_55,
    Model.O1: Model.FIREWORKS_DEEPSEEK_R1,
    Model.O3_MINI: Model.FIREWORKS_DEEPSEEK_R1,
    Model.DEEPSEEK_V3: Model.FIREWORKS_DEEPSEEK_V3,
    Model.DEEPSEEK_R1: Model.FIREWORKS_DEEPSEEK_R1,
    Model.FIREWORKS_DEEPSEEK_V3: Model.SONNET_55,
    Model.FIREWORKS_DEEPSEEK_R1: Model.O1,
    Model.GEMINI_FLASH_2: Model.HAIKU_55,
    Model.GEMINI_25_PRO: Model.GEMINI_25_PRO,
}
