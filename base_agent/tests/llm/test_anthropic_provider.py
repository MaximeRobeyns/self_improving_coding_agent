"""Tests for the requests that the Anthropic provider builds.

No request is sent: the client is a fake that records what it is asked.
"""
from types import SimpleNamespace

import pytest

from src.llm.base import Message
from src.llm.providers.anthropic import CONTINUATION_PROMPT, AnthropicProvider
from src.types.llm_types import (
    MODEL_FAILOVER_MAP,
    Model,
    Provider,
    ReasoningContent,
    ReasoningEffort,
    StopReason,
    TextContent,
)

CURRENT = [Model.HAIKU_55, Model.SONNET_55, Model.OPUS_55]
RETIRED = [Model.HAIKU_35, Model.SONNET_35, Model.SONNET_37]


class FakeClient:
    """Stands in for AsyncAnthropic: records requests, returns a canned reply."""

    def __init__(self, content=None, stop_reason="end_turn"):
        self.requests: list[dict] = []
        self._content = content or [SimpleNamespace(type="text", text="done")]
        self._stop_reason = stop_reason
        self.messages = SimpleNamespace(with_raw_response=SimpleNamespace(create=self._create))

    async def _create(self, **args):
        self.requests.append(args)
        response = SimpleNamespace(
            id="msg_1",
            content=self._content,
            stop_reason=self._stop_reason,
            stop_sequence=None,
            usage=SimpleNamespace(
                input_tokens=100, output_tokens=20,
                cache_creation_input_tokens=0, cache_read_input_tokens=0,
            ),
        )
        return SimpleNamespace(parsed=response, headers={})


def text(role, value):
    return Message(role=role, content=[TextContent(text=value)])


# What an agent sends on every step after its first: its instructions, the
# task, and its work so far as an assistant turn to carry on from
AGENT_STEP = [
    text("system", "You are an agent."),
    text("user", "Solve the task."),
    text("assistant", "First I looked at the files.   "),
]


async def request_for(model, messages=AGENT_STEP, **kwargs):
    client = FakeClient()
    await AnthropicProvider(client).create_completion(messages=messages, model=model, **kwargs)
    (request,) = client.requests
    return request


# Current models ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("model", CURRENT)
async def test_current_models_get_no_sampling_parameters(model):
    request = await request_for(model, temperature=0.7, top_p=0.9)

    assert "temperature" not in request and "top_p" not in request
    assert request["model"] == model.id
    assert request["max_tokens"] == 16000


@pytest.mark.asyncio
@pytest.mark.parametrize("model", CURRENT)
async def test_request_to_a_current_model_never_ends_in_an_assistant_turn(model):
    request = await request_for(model)

    roles = [m["role"] for m in request["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert request["messages"][-1]["content"] == [{"type": "text", "text": CONTINUATION_PROMPT}]
    # The work so far is passed on as it was (bar trailing whitespace)
    assert request["messages"][1]["content"][0]["text"] == "First I looked at the files."


@pytest.mark.asyncio
async def test_cache_break_point_is_on_the_work_so_far_not_the_closing_turn():
    request = await request_for(Model.SONNET_55)

    work_so_far, closing_turn = request["messages"][-2:]
    assert work_so_far["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in closing_turn["content"][-1]
    assert request["system"][-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_first_step_needs_no_closing_turn():
    request = await request_for(Model.SONNET_55, messages=AGENT_STEP[:2])
    assert [m["role"] for m in request["messages"]] == ["user"]


@pytest.mark.asyncio
async def test_current_models_think_adaptively():
    # No thinking budget, whatever effort is asked for
    request = await request_for(Model.SONNET_55)
    assert "extra_body" not in request and "thinking" not in request

    request = await request_for(Model.SONNET_55, reasoning_effort=ReasoningEffort.LOW)
    assert request["extra_body"] == {"output_config": {"effort": "low"}}
    assert "temperature" not in request


@pytest.mark.asyncio
async def test_stop_sequences_are_still_sent():
    request = await request_for(Model.SONNET_55, stop=["</TOOL_CALL>"])
    assert request["stop_sequences"] == ["</TOOL_CALL>"]


@pytest.mark.asyncio
async def test_empty_thinking_blocks_are_dropped():
    content = [
        SimpleNamespace(type="thinking", thinking="", signature="abc"),
        SimpleNamespace(type="redacted_thinking", data="xyz"),
        SimpleNamespace(type="thinking", thinking="A summary of the reasoning"),
        SimpleNamespace(type="text", text="The answer is 4."),
    ]
    client = FakeClient(content=content)

    completion = await AnthropicProvider(client).create_completion(
        messages=AGENT_STEP[:2], model=Model.SONNET_55
    )

    assert [type(block) for block in completion.content] == [ReasoningContent, TextContent]
    assert completion.content[0].text == "A summary of the reasoning"
    assert completion.stop_reason == StopReason.COMPLETE


@pytest.mark.asyncio
async def test_refusal_is_an_error_not_a_completion():
    client = FakeClient(content=[], stop_reason="refusal")
    completion = await AnthropicProvider(client).create_completion(
        messages=AGENT_STEP[:2], model=Model.SONNET_55
    )
    assert completion.stop_reason == StopReason.ERROR


# Older models are sent what they always were ------------------------------------


@pytest.mark.asyncio
async def test_older_models_keep_their_request_shape():
    request = await request_for(Model.SONNET_37, temperature=0.7, top_p=0.9)

    assert (request["temperature"], request["top_p"]) == (0.7, 0.9)
    assert request["max_tokens"] == 64000
    # The assistant turn is left to be continued
    assert [m["role"] for m in request["messages"]] == ["user", "assistant"]
    assert request["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_older_reasoning_models_keep_their_thinking_budget():
    request = await request_for(Model.SONNET_37, reasoning_effort=ReasoningEffort.MEDIUM)

    assert request["extra_body"] == {"thinking": {"type": "enabled", "budget_tokens": 32000}}
    assert request["temperature"] == 1.0 and "top_p" not in request


# Model definitions ------------------------------------------------------------


@pytest.mark.parametrize(
    "model, api_name, input_price, output_price, cache_read",
    [
        (Model.HAIKU_55, "claude-haiku-5-5", 0.10, 0.50, 0.01),
        (Model.SONNET_55, "claude-sonnet-5-5", 2.00, 10.00, 0.10),
        (Model.OPUS_55, "claude-opus-5-5", 4.00, 20.00, 0.20),
    ],
)
def test_current_model_definitions(model, api_name, input_price, output_price, cache_read):
    assert model.id == api_name  # no date suffix
    assert model.provider == Provider.ANTHROPIC
    costs = model.token_cost
    assert (costs.input_uncached, costs.output, costs.input_cached) == (
        input_price, output_price, cache_read,
    )
    assert costs.cache_write == pytest.approx(1.25 * input_price)
    assert model.context_window_size == 1_000_000
    assert not model.supports_prefill and not model.supports_sampling_params
    assert model.adaptive_thinking and model.supports_caching


@pytest.mark.parametrize("model", RETIRED)
def test_retired_models_are_kept_and_fail_over_to_current_ones(model):
    assert model.supports_prefill and model.supports_sampling_params
    assert not model.adaptive_thinking
    assert MODEL_FAILOVER_MAP[model] in CURRENT


def test_no_failover_leads_to_a_retired_model():
    for source, target in MODEL_FAILOVER_MAP.items():
        assert target not in RETIRED, f"{source.name} fails over to retired {target.name}"


def test_defaults_are_current_models():
    from src.config import Settings
    from src.agents.base_agent import BaseAgent
    from src.tools.committee_design import ReviewCommittee  # noqa: F401  (imports cleanly)

    settings = Settings()
    assert settings.MODEL == Model.SONNET_55
    assert settings.OVERSIGHT_MODEL == Model.SONNET_55
    assert BaseAgent.MODEL == Model.SONNET_55
    assert Model.from_name("OPUS_55") == Model.OPUS_55
