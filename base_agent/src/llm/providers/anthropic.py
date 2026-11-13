# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Anthropic-specific LLM provider implementation."""

import logging
import asyncio
import time
import threading
from typing import List, Optional, Union, AsyncGenerator, Type, Dict, Any
from datetime import datetime, timezone

from pydantic import BaseModel
from anthropic.types import Message as AntMessage

from .base_provider import (
    Message,
    Completion,
    CompletionChunk,
    TimingInfo,
    BaseProvider,
)
from ...types.llm_types import TokenUsage, FCI, Model, StopReason, ReasoningEffort, TextContent, ReasoningContent, ToolCallContent, ToolResultContent, ContentTypes
from ...types.agent_types import AgentInterface
from ...types.tool_types import ToolInterface

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Initialize tokenizer for token estimation if available
try:
    import tiktoken
    TOKENIZER = tiktoken.encoding_for_model("cl100k_base")  # Claude uses cl100k_base encoding
except Exception:
    TOKENIZER = None


def estimate_token_count(text: str) -> int:
    """Estimate the number of tokens in the text."""
    if TOKENIZER:
        return len(TOKENIZER.encode(text))
    else:
        # Rough approximation: 4 characters per token on average
        return len(text) // 4


def get_suffix_difference(shorter: str, longer: str) -> str:
    """For debugging cache breaks"""
    # Find the index where the prefix ends
    common_length = len(shorter)

    if not longer.startswith(shorter):
        # Find actual common prefix length if shorter is not fully a prefix
        common_length = next((i for i, (a, b) in enumerate(zip(shorter, longer)) if a != b), 0)

    return longer[common_length:]


class AnthropicRateLimiter:
    """
    Global rate limiter for Anthropic API that tracks and enforces rate limits
    based on response headers.
    """

    # Singleton instance
    _instance = None
    _lock = threading.RLock()  # Reentrant lock for thread safety

    def __new__(cls):
        """Ensure only one instance of the rate limiter exists."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AnthropicRateLimiter, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """Initialize the rate limiter if it hasn't been initialized yet."""
        with self._lock:
            if getattr(self, "_initialized", False):
                return

            # Rate limit tracking
            self._requests_limit = float('inf')  # Number of requests
            self._requests_remaining = float('inf')
            self._requests_reset = datetime.max.replace(tzinfo=timezone.utc)

            self._input_tokens_limit = float('inf')  # Input tokens
            self._input_tokens_remaining = float('inf')
            self._input_tokens_reset = datetime.max.replace(tzinfo=timezone.utc)

            self._output_tokens_limit = float('inf')  # Output tokens
            self._output_tokens_remaining = float('inf')
            self._output_tokens_reset = datetime.max.replace(tzinfo=timezone.utc)

            self._tokens_limit = float('inf')  # Total tokens
            self._tokens_remaining = float('inf')
            self._tokens_reset = datetime.max.replace(tzinfo=timezone.utc)

            self._initialized = True

    def update_from_headers(self, headers: Dict[str, str]) -> None:
        """
        Update rate limit information based on response headers.

        Args:
            headers: The response headers from an Anthropic API request
        """
        with self._lock:
            # Update request limits
            if 'anthropic-ratelimit-requests-limit' in headers and 'anthropic-ratelimit-requests-remaining' in headers:
                self._requests_limit = int(headers.get('anthropic-ratelimit-requests-limit', self._requests_limit))
                self._requests_remaining = int(headers.get('anthropic-ratelimit-requests-remaining', self._requests_remaining))
                reset_time_str = headers.get('anthropic-ratelimit-requests-reset')
                if reset_time_str:
                    self._requests_reset = datetime.fromisoformat(reset_time_str.replace('Z', '+00:00'))

            # Update input token limits
            if 'anthropic-ratelimit-input-tokens-limit' in headers and 'anthropic-ratelimit-input-tokens-remaining' in headers:
                self._input_tokens_limit = int(headers.get('anthropic-ratelimit-input-tokens-limit', self._input_tokens_limit))
                self._input_tokens_remaining = int(headers.get('anthropic-ratelimit-input-tokens-remaining', self._input_tokens_remaining))
                reset_time_str = headers.get('anthropic-ratelimit-input-tokens-reset')
                if reset_time_str:
                    self._input_tokens_reset = datetime.fromisoformat(reset_time_str.replace('Z', '+00:00'))

            # Update output token limits
            if 'anthropic-ratelimit-output-tokens-limit' in headers and 'anthropic-ratelimit-output-tokens-remaining' in headers:
                self._output_tokens_limit = int(headers.get('anthropic-ratelimit-output-tokens-limit', self._output_tokens_limit))
                self._output_tokens_remaining = int(headers.get('anthropic-ratelimit-output-tokens-remaining', self._output_tokens_remaining))
                reset_time_str = headers.get('anthropic-ratelimit-output-tokens-reset')
                if reset_time_str:
                    self._output_tokens_reset = datetime.fromisoformat(reset_time_str.replace('Z', '+00:00'))

            # Update total token limits
            if 'anthropic-ratelimit-tokens-limit' in headers and 'anthropic-ratelimit-tokens-remaining' in headers:
                self._tokens_limit = int(headers.get('anthropic-ratelimit-tokens-limit', self._tokens_limit))
                self._tokens_remaining = int(headers.get('anthropic-ratelimit-tokens-remaining', self._tokens_remaining))
                reset_time_str = headers.get('anthropic-ratelimit-tokens-reset')
                if reset_time_str:
                    self._tokens_reset = datetime.fromisoformat(reset_time_str.replace('Z', '+00:00'))

            # Log updated limits
            logger.debug(f"Updated rate limits - Requests: {self._requests_remaining}/{self._requests_limit}, "
                        f"Input tokens: {self._input_tokens_remaining}/{self._input_tokens_limit}, "
                        f"Output tokens: {self._output_tokens_remaining}/{self._output_tokens_limit}, "
                        f"Total tokens: {self._tokens_remaining}/{self._tokens_limit}")

    def check_and_reserve_request(self) -> tuple[bool, Optional[float]]:
        """
        Check if a request can be made without exceeding rate limits and reserve capacity.

        Returns:
            tuple[bool, Optional[float]]: (True if request is allowed, None or seconds to wait if not allowed)
        """
        with self._lock:
            # Check if any rate limits have reset
            now = datetime.now(timezone.utc)

            # Check request limit
            if now >= self._requests_reset:
                # Reset window has passed, reset the limit
                self._requests_remaining = self._requests_limit
                self._requests_reset = datetime.max.replace(tzinfo=timezone.utc)

            if self._requests_remaining < 1:
                wait_time = (self._requests_reset - now).total_seconds()
                return False, max(0, wait_time)

            # Reserve a request
            self._requests_remaining -= 1
            return True, None

    def check_and_reserve_tokens(
        self, estimated_input_tokens: int, estimated_output_tokens: int
    ) -> tuple[bool, Optional[float]]:
        """
        Check if token usage is within limits and reserve capacity.

        Args:
            estimated_input_tokens: Estimated number of input tokens
            estimated_output_tokens: Estimated number of output tokens

        Returns:
            tuple[bool, Optional[float]]: (True if allowed, None or seconds to wait if not allowed)
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Check input token limit
            if now >= self._input_tokens_reset:
                self._input_tokens_remaining = self._input_tokens_limit
                self._input_tokens_reset = datetime.max.replace(tzinfo=timezone.utc)

            if self._input_tokens_remaining < estimated_input_tokens:
                wait_time = (self._input_tokens_reset - now).total_seconds()
                return False, max(0, wait_time)

            # Check output token limit
            if now >= self._output_tokens_reset:
                self._output_tokens_remaining = self._output_tokens_limit
                self._output_tokens_reset = datetime.max.replace(tzinfo=timezone.utc)

            if self._output_tokens_remaining < estimated_output_tokens:
                wait_time = (self._output_tokens_reset - now).total_seconds()
                return False, max(0, wait_time)

            # Check total token limit
            total_tokens = estimated_input_tokens + estimated_output_tokens

            if now >= self._tokens_reset:
                self._tokens_remaining = self._tokens_limit
                self._tokens_reset = datetime.max.replace(tzinfo=timezone.utc)

            if self._tokens_remaining < total_tokens:
                wait_time = (self._tokens_reset - now).total_seconds()
                return False, max(0, wait_time)

            # Reserve tokens
            self._input_tokens_remaining -= estimated_input_tokens
            self._output_tokens_remaining -= estimated_output_tokens
            self._tokens_remaining -= total_tokens

            return True, None

    def report_actual_usage(self, input_tokens: int, output_tokens: int) -> None:
        """
        Report the actual token usage for a request, adjusting the reserved amounts if needed.

        This method should be called after a request is completed to correct any over-reservations.

        Args:
            input_tokens: Actual number of input tokens used
            output_tokens: Actual number of output tokens used
        """
        # We rely on the headers for the most accurate state,
        # so this method doesn't need to do anything currently
        pass

    def get_rate_limit_status(self) -> Dict[str, Dict[str, any]]:
        """Get the current rate limit status."""
        with self._lock:
            now = datetime.now(timezone.utc)

            return {
                'requests': {
                    'limit': self._requests_limit,
                    'remaining': self._requests_remaining,
                    'reset_in_seconds': max(0, (self._requests_reset - now).total_seconds())
                               if self._requests_reset != datetime.max.replace(tzinfo=timezone.utc) else 0,
                },
                'input_tokens': {
                    'limit': self._input_tokens_limit,
                    'remaining': self._input_tokens_remaining,
                    'reset_in_seconds': max(0, (self._input_tokens_reset - now).total_seconds())
                                if self._input_tokens_reset != datetime.max.replace(tzinfo=timezone.utc) else 0,
                },
                'output_tokens': {
                    'limit': self._output_tokens_limit,
                    'remaining': self._output_tokens_remaining,
                    'reset_in_seconds': max(0, (self._output_tokens_reset - now).total_seconds())
                                 if self._output_tokens_reset != datetime.max.replace(tzinfo=timezone.utc) else 0,
                },
                'total_tokens': {
                    'limit': self._tokens_limit,
                    'remaining': self._tokens_remaining,
                    'reset_in_seconds': max(0, (self._tokens_reset - now).total_seconds())
                                if self._tokens_reset != datetime.max.replace(tzinfo=timezone.utc) else 0,
                },
            }


# Create the global singleton instance
anthropic_rate_limiter = AnthropicRateLimiter()


# Sent as the closing user turn to models that cannot be handed an unfinished
# assistant turn to continue (see _prepare_messages)
CONTINUATION_PROMPT = (
    "Continue from exactly where the above leaves off. Do not repeat or "
    "summarise what is already there: take the next step."
)


class AnthropicProvider(BaseProvider):
    """Provider implementation for Anthropic's Claude models."""

    def __init__(self, client, max_retries: int = 3, retry_base_delay: float = 1.0):
        self.client = client
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._last_sys = ""
        self._last_msg = ""

    def map_stop_reason(self, response: AntMessage) -> tuple[StopReason, Optional[str]]:
        """Map Anthropic-specific stop reasons to our standard format.

        Returns the standardised stop reason, and optionally the stop sequence
        that caused it.
        """
        # Handle streaming chunks
        # TODO: finish handling the streaming case
        if hasattr(response, "type") and response.type == "message_delta":
            if hasattr(response.delta, "stop_reason"):
                raw_stop = response.delta.stop_reason
            else:
                return StopReason.COMPLETE, None

        raw_stop_reason = response.stop_reason
        if raw_stop_reason == "end_turn":
            return StopReason.COMPLETE, None  # Natural termination
        elif raw_stop_reason == "max_tokens":
            return StopReason.LENGTH, None
        elif raw_stop_reason == "stop_sequence":
            return StopReason.STOP_TOKEN, response.stop_sequence
        elif raw_stop_reason == "tool_use":
            return StopReason.TOOL_CALL, None
        elif raw_stop_reason == "refusal":
            # The request was declined by a safety classifier; there is no
            # usable content. `stop_details` says which category, if any.
            details = getattr(response, "stop_details", None)
            logger.warning(f"Anthropic declined the request (refusal): {details}")
            return StopReason.ERROR, None
        else:
            return StopReason.ERROR, None

    def _content_mapping(self, block: ContentTypes) -> dict:
        """Maps our message content types, into provider-specific message formats"""
        if isinstance(block, TextContent):
            return {"type": "text", "text": block.text,
                    **({"cache_control": block.cache_control} if block.cache_control else {})}
        elif isinstance(block, ReasoningContent):
            return {"type": "thinking", "thinking": block.text,
                    **({"cache_control": block.cache_control} if block.cache_control else {})}
            # TODO: handle redacted thinking
        elif isinstance(block, ToolCallContent):
            return {"type": "tool_use", "id": block.call_id, "name": block.tool_name, "input": block.tool_args,
                    **({"cache_control": block.cache_control} if block.cache_control else {})}
        elif isinstance(block, ToolResultContent):
            return {"type": "tool_result", "tool_use_id": block.call_id, "content": block.content,
                    **({"cache_control": block.cache_control} if block.cache_control else {})}
        else:
            raise ValueError(f"Unhandled content type in provider Anthropic: {block}")

    def _prepare_messages(
        self, messages: list[Message], model: Model | None = None
    ) -> tuple[Optional[list[dict]], list[dict]]:
        """Process and prepare messages for the Anthropic API.

        Anthropic has specific requirements for message formatting:
        1. System messages must be separated and structured
        2. Assistant messages need content trimming
        3. Names and cache control flags must be properly formatted
        4. For models that do not accept a request ending in an assistant
           turn (see CONTINUATION_PROMPT), a user turn is appended

        Returns:
            Tuple of (system_content, anthropic_messages)
        """
        system_content = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content = [self._content_mapping(block) for block in msg.content]
            else:
                msg_content = [self._content_mapping(block) for block in msg.content]
                anthropic_messages.append(
                    {
                        "role": msg.role,
                        "content": msg_content,
                        **({"name": msg.name} if msg.name else {}),
                    }
                )

        for i in range(len(anthropic_messages)-1, 0, -1):
            # Remove potential empty message blocks from the end
            while hasattr(anthropic_messages[i]["content"][-1], "text") and anthropic_messages[i]["content"][-1].text == "":
                messages[i].content.pop()

            if len(messages[i].content) == 0:
                messages.pop()

            elif hasattr(anthropic_messages[i]["content"][-1], "text") and anthropic_messages[i]["content"][-1]["text"] != "":
                break

        # Ensure the last assistant message in the prefill doesn't have
        # trailing whitespace
        if anthropic_messages[-1]["role"] == "assistant":
            for block in reversed(anthropic_messages[-1]["content"]):
                if block["type"] == "text":
                    block["text"] = block["text"].rstrip()

        # Default to setting a cache break point at the last message
        anthropic_messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        # The agents send their work so far as an assistant turn for the model
        # to carry on from. Models that reject that get a user turn asking
        # them to carry on instead. It goes after the cache break point: the
        # work so far is then a prefix of the next request, which this
        # closing turn would not be.
        if (
            model is not None
            and not model.supports_prefill
            and anthropic_messages[-1]["role"] == "assistant"
        ):
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": CONTINUATION_PROMPT}],
                }
            )

        if isinstance(system_content, list) and len(system_content) > 0:
            system_content[-1]["cache_control"] = {"type": "ephemeral"}

        return system_content, anthropic_messages

    def _create_token_usage(self, usage_data) -> TokenUsage:
        """Create TokenUsage object from response usage data."""

        empty_usage = TokenUsage()
        if not usage_data:
            return empty_usage

        try:
            usage_dict = (
                usage_data.model_dump()
                if hasattr(usage_data, "model_dump")
                else vars(usage_data)
            )
            return TokenUsage(
                uncached_prompt_tokens=usage_dict.get("input_tokens", 0),
                cache_write_prompt_tokens=usage_dict.get(
                    "cache_creation_input_tokens", 0
                ),
                cached_prompt_tokens=usage_dict.get("cache_read_input_tokens", 0),
                completion_tokens=usage_dict.get("output_tokens", 0),
            )
        except Exception as e:
            logger.warning(f"Failed to create token usage: {e}")
            return empty_usage

    def pydantic_to_native_tool(self, tool: ToolInterface | AgentInterface) -> dict:
        if hasattr(tool, 'TOOL_NAME') and hasattr(tool, 'TOOL_DESCRIPTION'):
            return {
                "name": tool.TOOL_NAME,
                "description": tool.TOOL_DESCRIPTION,
                "input_schema": tool.model_json_schema()
            }
        elif hasattr(tool, 'AGENT_NAME') and hasattr(tool, 'AGENT_DESCRIPTION'):
            return {
                "name": tool.AGENT_NAME,
                "description": tool.AGENT_DESCRIPTION,
                "input_schema": tool.model_json_schema()
            }
        else:
            # Fall back to guess
            return {
                "name": tool.__class__.__name__,
                "description": "No description provided.",
                "input_schema": tool.model_json_schema()
            }

        raise ValueError("tool is not BaseTool or BaseAgent")

    def _model_args(
        self,
        model: Model,
        temperature: float,
        top_p: float,
        reasoning_effort: ReasoningEffort | None,
    ) -> dict:
        """The request arguments that depend on what the model accepts:
        sampling parameters and thinking."""
        args: dict = {}
        if model.supports_sampling_params:
            args["temperature"] = temperature
            args["top_p"] = top_p

        if model.adaptive_thinking:
            # Thinking is on and sized by the model itself; a thinking budget
            # is rejected. How hard it works is steered by the effort level,
            # which is left at the model's default unless one is asked for.
            if reasoning_effort is not None:
                args["extra_body"] = {"output_config": {"effort": reasoning_effort.value}}
        elif model.is_reasoner and reasoning_effort is not None:
            args["temperature"] = 1.0
            args.pop("top_p", None)
            args["extra_body"] = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": self._reasoning_budgets_to_tokens(reasoning_effort),
                }
            }
        return args

    def _reasoning_budgets_to_tokens(self, reasoning_effort: ReasoningEffort) -> int:
        if reasoning_effort == ReasoningEffort.LOW:
            return 16000
        elif reasoning_effort == ReasoningEffort.MEDIUM:
            return 32000
        elif reasoning_effort == ReasoningEffort.HIGH:
            return 60000

    def _estimate_request_tokens(self, messages: List[Message]) -> int:
        """
        Estimate the number of tokens in the messages.

        Args:
            messages: The list of messages to estimate tokens for

        Returns:
            int: Estimated token count
        """
        total_tokens = 0

        for msg in messages:
            # Add tokens for role
            total_tokens += 4  # Average tokens used for role

            # Add tokens for each content block
            for block in msg.content:
                if hasattr(block, "text"):
                    total_tokens += estimate_token_count(block.text)
                elif hasattr(block, "tool_name"):
                    # Tool calls have a JSON structure
                    tool_tokens = (
                        estimate_token_count(block.tool_name) +
                        estimate_token_count(str(block.tool_args))
                    )
                    total_tokens += tool_tokens + 10  # Extra tokens for JSON structure

        # Add a buffer for any metadata or other tokens
        return int(total_tokens * 1.1)  # 10% buffer

    async def _update_rate_limits_from_direct_request(self, args: Dict[str, Any]) -> None:
        """
        Make a direct HTTP request to get rate limit headers.

        Args:
            args: The arguments being used for the main request
        """
        import httpx

        async with httpx.AsyncClient() as client:
            headers = {
                "x-api-key": self.client.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            # Make a small request to get headers
            small_args = args.copy()
            if "max_tokens" in small_args:
                small_args["max_tokens"] = 1  # Minimize usage

            try:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=small_args,
                    timeout=5.0
                )

                # Update rate limiter with the headers
                anthropic_rate_limiter.update_from_headers(dict(response.headers))
            except Exception as e:
                logger.warning(f"Failed to update rate limits: {e}")

    async def create_completion(
        self,
        messages: List[Message],
        model: Model,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        top_p: float = 1.0,
        reasoning_effort: ReasoningEffort | None = None,
        available_tools: list[Type[BaseModel]] | None = None,
        num_completions: int = 1,
    ) -> Completion:
        """Create a completion using Anthropic's API with rate limit handling."""
        # Estimate token usage for pre-flight rate limit check
        estimated_input_tokens = self._estimate_request_tokens(messages)
        # Estimate a conservative number for output tokens
        estimated_output_tokens = max_tokens or model.max_output_tokens

        # Check rate limits before making the request
        allowed, wait_time = anthropic_rate_limiter.check_and_reserve_tokens(
            estimated_input_tokens, estimated_output_tokens
        )

        if not allowed:
            # Check if we should wait or fail
            if wait_time > 60:  # If wait time is more than a minute, fail immediately
                raise Exception(f"Rate limit exceeded. Try again in {wait_time:.1f} seconds")

            # Wait a bit and try again
            logger.info(f"Token rate limit approaching, waiting {wait_time:.2f} seconds before request")
            print(f"\n\n\nToken rate limit approaching, waiting {wait_time:.2f} seconds before request\n\n\n")
            await asyncio.sleep(wait_time + 0.1)  # Add a small buffer

            # Check again after waiting
            allowed, _ = anthropic_rate_limiter.check_and_reserve_tokens(
                estimated_input_tokens, estimated_output_tokens
            )

            if not allowed:
                raise Exception("Token rate limit still exceeded after waiting")

        # Also check request limit
        allowed, wait_time = anthropic_rate_limiter.check_and_reserve_request()

        if not allowed:
            # Handle request limit similarly to token limit
            if wait_time > 60:
                raise Exception(f"Request rate limit exceeded. Try again in {wait_time:.1f} seconds")

            logger.info(f"Request rate limit approaching, waiting {wait_time:.2f} seconds")
            print(f"\n\n\nRequest rate limit approaching, waiting {wait_time:.2f} seconds\n\n\n")
            await asyncio.sleep(wait_time + 0.1)

            allowed, _ = anthropic_rate_limiter.check_and_reserve_request()
            if not allowed:
                raise Exception("Request rate limit still exceeded after waiting")

        # Execute the request with retries for rate limit errors
        for attempt in range(self.max_retries + 1):
            try:
                start_time = datetime.now()

                logger.debug(f"Creating completion with stop tokens: {stop}")

                # Process messages
                system_content, anthropic_messages = self._prepare_messages(messages, model)

                # Prepare arguments
                args = {
                    "messages": anthropic_messages,
                    "model": model.id,
                    "max_tokens": max_tokens or model.max_output_tokens,
                    **self._model_args(model, temperature, top_p, reasoning_effort),
                }

                if system_content:
                    args["system"] = system_content

                if stop:
                    stop_sequences = [stop] if isinstance(stop, str) else stop
                    args["stop_sequences"] = stop_sequences

                if available_tools and model.fci == FCI.CONSTRAINED:
                    args["tools"] = [self.pydantic_to_native_tool(t) for t in available_tools]


                # Execute the request with raw response to get headers
                try:
                    # For newer versions of the Anthropic SDK (0.17.0+)
                    raw_response = await self.client.messages.with_raw_response.create(**args)
                    response = raw_response.parsed
                    headers = raw_response.headers

                    # Update rate limiter with the headers
                    anthropic_rate_limiter.update_from_headers(headers)
                except (AttributeError, TypeError):
                    # Fallback for older SDK versions
                    response = await self.client.messages.create(**args)
                    # No headers available in older SDKs, but we can use the usage data

                end_time = datetime.now()

                # Extract usage information
                usage_info = self._create_token_usage(response.usage)
                logger.debug(usage_info)

                # Report actual usage to rate limiter to correct any estimation errors
                if usage_info:
                    anthropic_rate_limiter.report_actual_usage(
                        usage_info.input_tokens, usage_info.completion_tokens
                    )

                duration = end_time - start_time
                tokens_per_second = (
                    usage_info.completion_tokens / duration.total_seconds()
                    if usage_info and duration.total_seconds() > 0
                    else None
                )

                timing_info = TimingInfo(
                    start_time=start_time,
                    end_time=end_time,
                    total_duration=duration,
                    first_token_time=None,
                    time_to_first_token=None,
                    tokens_per_second=tokens_per_second,
                )

                # Create cache metrics if available
                cache_metrics = None
                if usage_info:
                    cache_metrics = {
                        "cache_hits": usage_info.cached_prompt_tokens,
                        "cache_misses": usage_info.uncached_prompt_tokens,
                        "cache_writes": usage_info.cache_write_prompt_tokens,
                    }

                logger.debug(f"\n\nUsage: {response.usage}\n{cache_metrics}\n\n")

                # Map stop reason
                stop_reason, stop_sequence = self.map_stop_reason(response)

                response_content = []
                for block in response.content:
                    match block.type:
                        case 'text':
                            response_content.append(TextContent(text=block.text))
                        case 'tool_use':
                            response_content.append(
                                ToolCallContent(
                                    call_id=block.id, tool_name=block.name, tool_args=block.input,
                                    call_type=FCI.CONSTRAINED,
                                )
                            )
                        case 'thinking':
                            # Current models return their thinking blocks
                            # empty unless a summary is asked for
                            if getattr(block, "thinking", ""):
                                response_content.append(
                                    ReasoningContent(text=block.thinking)
                                )
                        case 'redacted_thinking':
                            pass
                        case _:
                            logger.warning(
                                f"Unhandled response block type {block.type} in Anthropic completion"
                            )

                completion = Completion(
                    id=response.id,
                    content=response_content,
                    model=model,
                    usage=usage_info,
                    timing=timing_info,
                    cache_metrics=cache_metrics,
                    stop_reason=stop_reason,
                    stop_sequence=stop_sequence,
                )

                # Store raw response data
                completion.raw_response = {
                    "stop_reason": getattr(response, "stop_reason", None),
                    "stop_sequence": getattr(response, "stop_sequence", None),
                }

                return completion

            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit_error = "rate_limit" in error_msg or "rate limit" in error_msg

                if is_rate_limit_error and attempt < self.max_retries:
                    # Calculate backoff delay with exponential backoff and jitter
                    delay = self.retry_base_delay * (2 ** attempt) * (0.5 + 0.5 * time.time() % 1)
                    logger.warning(f"Rate limit exceeded, retrying in {delay:.2f} seconds (attempt {attempt+1}/{self.max_retries})")
                    print(f"\n\n\nRate limit exceeded, retrying in {delay:.2f} seconds (attempt {attempt+1}/{self.max_retries})\n\n\n")
                    await asyncio.sleep(delay)
                    continue

                # Re-raise other errors or if we've exhausted retries
                logger.error(f"Error during completion: {str(e)}")
                raise

    async def create_streaming_completion(
        self,
        messages: List[Message],
        model: Model,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        top_p: float = 1.0,
        reasoning_effort: ReasoningEffort | None = None,
        available_tools: list[Type[BaseModel]] | None = None,
    ) -> AsyncGenerator[CompletionChunk, None]:
        """Create a streaming completion using Anthropic's API."""
        # Add rate limiting for streaming completion method too
        # Estimate token usage
        estimated_input_tokens = self._estimate_request_tokens(messages)
        estimated_output_tokens = max_tokens or model.max_output_tokens

        # Check rate limits
        tokens_allowed, tokens_wait_time = anthropic_rate_limiter.check_and_reserve_tokens(
            estimated_input_tokens, estimated_output_tokens
        )

        requests_allowed, requests_wait_time = anthropic_rate_limiter.check_and_reserve_request()

        # Handle rate limits
        if not tokens_allowed or not requests_allowed:
            wait_time = max(tokens_wait_time or 0, requests_wait_time or 0)

            if wait_time > 60:
                # Rather than making the user wait a long time, raise an error
                raise Exception(f"Rate limit exceeded. Try again in {wait_time:.1f} seconds")

            logger.info(f"Rate limit approaching, waiting {wait_time:.2f} seconds before request")
            await asyncio.sleep(wait_time + 0.1)  # Add a small buffer

        start_time = datetime.now()
        first_token_time = None
        message_id = None
        final_usage = None
        current_stop_reason = None
        current_stop_sequence = None
        usage_info = None
        initial_prompt_tokens = 0  # Track initial prompt tokens

        # Process messages
        system_content, anthropic_messages = self._prepare_messages(messages, model)

        # Prepare arguments
        args = {
            "messages": anthropic_messages,
            "model": model.id,
            "max_tokens": max_tokens or model.max_output_tokens,
            "stream": True,
            **self._model_args(model, temperature, top_p, None),
        }

        if system_content:
            args["system"] = system_content

        if stop:
            stop_sequences = [stop] if isinstance(stop, str) else stop
            args["stop_sequences"] = stop_sequences

        # Track cache metrics
        cache_metrics = None

        async for stream_resp in await self.client.messages.create(**args):
            current_time = datetime.now()

            # Handle message start
            if stream_resp.type == "message_start":
                message_id = stream_resp.message.id
                # Get initial usage info and store prompt tokens
                if stream_resp.message.usage:
                    usage_info = stream_resp.message.usage
                    initial_prompt_tokens = usage_info.input_tokens
                    # This is wrong: cache misses
                    cache_metrics = {
                        "cache_hits": usage_info.cache_read_input_tokens,
                        "cache_misses": usage_info.input_tokens
                        - usage_info.cache_read_input_tokens,
                        "cache_writes": usage_info.cache_creation_input_tokens,
                    }
                continue

            # Handle content updates
            elif stream_resp.type == "content_block_delta":
                if stream_resp.delta.type == "text_delta" and stream_resp.delta.text:
                    content = stream_resp.delta.text
                    if first_token_time is None:
                        first_token_time = current_time

                    timing_info = TimingInfo(
                        start_time=start_time,
                        end_time=current_time,
                        total_duration=current_time - start_time,
                        first_token_time=first_token_time,
                        time_to_first_token=(
                            (first_token_time - start_time).total_seconds()
                            if first_token_time
                            else None
                        ),
                        tokens_per_second=None,  # Calculate in final chunk
                    )

                    yield CompletionChunk(
                        id=message_id,
                        content=content,
                        model=model,
                        is_finished=False,
                        timing=timing_info,
                        usage=None,  # Usage only in final chunk
                    )

            # Handle message updates
            elif stream_resp.type == "message_delta":
                # Update usage info
                if hasattr(stream_resp, "usage"):
                    usage_info = stream_resp.usage
                current_stop_reason, current_stop_sequence = self.map_stop_reason(
                    stream_resp
                )

            # Handle final chunk
            elif stream_resp.type == "message_stop":
                # Process final usage info with initial prompt tokens
                usage_dict = (
                    usage_info.model_dump()
                    if hasattr(usage_info, "model_dump")
                    else vars(usage_info)
                )
                # NOTE: this is probably inaccurate
                final_usage = TokenUsage(
                    uncached_prompt_tokens=initial_prompt_tokens,  # Use stored prompt tokens
                    cache_write_prompt_tokens=usage_dict.get(
                        "cache_creation_input_tokens", 0
                    ),
                    cached_prompt_tokens=usage_dict.get("cache_read_input_tokens", 0),
                    completion_tokens=usage_dict.get("output_tokens", 0),
                )

                # Calculate timing
                duration = current_time - start_time
                tokens_per_second = (
                    final_usage.completion_tokens / duration.total_seconds()
                    if final_usage and duration.total_seconds() > 0
                    else None
                )

                timing_info = TimingInfo(
                    start_time=start_time,
                    end_time=current_time,
                    total_duration=duration,
                    first_token_time=first_token_time,
                    time_to_first_token=(
                        (first_token_time - start_time).total_seconds()
                        if first_token_time
                        else None
                    ),
                    tokens_per_second=tokens_per_second,
                )

                # Send final chunk
                completion = CompletionChunk(
                    id=message_id,
                    content="",  # Empty content in final chunk
                    model=model,
                    is_finished=True,
                    timing=timing_info,
                    usage=final_usage,
                    cache_metrics=cache_metrics,
                    stop_reason=current_stop_reason,
                    raw_response={
                        "stop_reason": (
                            current_stop_reason.value if current_stop_reason else None
                        ),
                        "stop_sequence": current_stop_sequence,
                    },
                )

                yield completion
