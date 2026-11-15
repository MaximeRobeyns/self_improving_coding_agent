# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the agent calling mechanism."""
import pytest
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from src.agents.agent_calling import (
    execute_agent_call,
    parse_agent_content,
    await_agent_task,
    handle_parse_errors,
    agent_registry,
)
from src.agents.base_agent import BaseAgent
from src.events.event_bus import EventBus
from src.types.event_types import Event, EventType
from src.callgraph.manager import CallGraphManager
from src.types.agent_types import AgentInterface, AgentStatus, AgentResult, AgentMetrics, InheritanceConfig
from src.types.llm_types import FCI, ToolCallContent, Model
from src.types.common import ArgFormat

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SimpleTestAgent(BaseAgent):
    """A minimal test agent implementation."""

    AGENT_NAME = "test_agent"
    AGENT_DESCRIPTION = "A test agent for testing agent calling"
    SYSTEM_PROMPT = "You are a test agent."
    MODEL = Model.SONNET_37
    INHERITANCE = InheritanceConfig()

    # Empty set of available agents by default
    AVAILABLE_AGENTS = set()

    def __init__(self, parent: BaseAgent | None = None, workdir: Path | None = None,
                 logdir: Path | None = None, debug_mode: bool = False, **data):
        super().__init__(parent=parent, workdir=workdir, logdir=logdir, debug_mode=debug_mode, **data)
        # Configure available agents
        self._available_agent_names = {SimpleTestAgent.AGENT_NAME}
        self._available_agents = {SimpleTestAgent}

    async def construct_core_prompt(self) -> str:
        """Minimal prompt for testing."""
        return "Test prompt"

    @classmethod
    def generate_examples(cls):
        """No examples needed for testing."""
        return []

class TestAgentCalling:
    """Test suite for agent calling functionality."""

    @pytest.fixture(autouse=True)
    async def _setup_event_bus(self):
        """Set up and clean up event bus state."""
        bus = await EventBus.get_instance()
        bus.clear()
        yield
        bus.clear()

    @pytest.fixture(autouse=True)
    async def _setup_callgraph(self):
        """Set up and clean up callgraph state."""
        graph = await CallGraphManager.get_instance()
        graph.clear()
        yield
        graph.clear()

    @pytest.fixture(autouse=True)
    def _setup_agent_registry(self):
        """Set up and clean up agent registry."""
        saved_registry = dict(agent_registry)
        yield
        agent_registry.clear()
        agent_registry.update(saved_registry)

    @pytest.fixture
    def setup_dirs(self, tmp_path):
        """Set up temporary directories for agent execution."""
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        logdir = tmp_path / "logdir"
        logdir.mkdir()
        return workdir, logdir

    @pytest.fixture
    async def parent_agent(self, setup_dirs):
        """Create a real test agent instance."""
        workdir, logdir = setup_dirs
        agent = SimpleTestAgent(workdir=workdir, logdir=logdir)
        agent._available_agent_names = {SimpleTestAgent.AGENT_NAME}
        agent._available_agents = {SimpleTestAgent}
        return agent

    @pytest.mark.asyncio
    async def test_agent_content_parsing(self):
        """Test parsing of agent call content."""
        # Register our test agent
        class ParseTestAgent(SimpleTestAgent):
            @classmethod
            async def args_str_to_dict(cls, args_str: str, arg_format: ArgFormat = ArgFormat.XML):
                # For testing, we just extract the workdir and logdir values
                if "<workdir>/tmp/test</workdir>" in args_str:
                    return {
                        "workdir": "/tmp/test",
                        "logdir": "/tmp/logs",
                    }, None
                return await super().args_str_to_dict(args_str, arg_format)

        # Register our test agent
        agent_registry["test_agent"] = ParseTestAgent

        # Test valid XML agent content
        content = """
        <AGENT_CALL>
        <AGENT_NAME>test_agent</AGENT_NAME>
        <AGENT_ARGS>
        <workdir>/tmp/test</workdir>
        <logdir>/tmp/logs</logdir>
        </AGENT_ARGS>
        </AGENT_CALL>
        """
        name, args, errors = await parse_agent_content(content, ArgFormat.XML)

        assert name == "test_agent"
        assert "workdir" in args
        assert args["workdir"] == "/tmp/test"
        assert "logdir" in args
        assert args["logdir"] == "/tmp/logs"
        assert errors is None

        # Test invalid agent name
        content = """
        <AGENT_CALL>
        <AGENT_NAME>nonexistent_agent</AGENT_NAME>
        <AGENT_ARGS>
        <workdir>/tmp/test</workdir>
        </AGENT_ARGS>
        </AGENT_CALL>
        """
        name, args, errors = await parse_agent_content(content, ArgFormat.XML)
        assert "does not correspond to a registered agent" in errors.lower()

    @pytest.mark.asyncio
    async def test_successful_agent_execution(self, parent_agent):
        """Test complete flow of successful agent execution."""
        # Ensure we await our parent_agent fixture
        caller = parent_agent

        # Register test agent in registry
        agent_registry["test_agent"] = SimpleTestAgent

        # Create a complete sub-agent
        class SuccessAgent(SimpleTestAgent):
            async def execute(self):
                self._local_state["exec_complete"] = True
                self._local_state["return_value"] = "Task completed successfully"
                return AgentResult(
                    agent_name=self.AGENT_NAME,
                    status=AgentStatus.SUCCESS,
                    result="Task completed successfully",
                    metrics=self._metrics
                )

        # Create the sub-agent instance
        sub_agent = SuccessAgent(
            parent=caller,
            workdir=caller._workdir,
            logdir=caller._logdir
        )

        # Create the agent content
        agent_content = ToolCallContent(
            call_id="test_call",
            tool_name="test_agent",
            tool_args={
                "workdir": str(caller._workdir),
                "logdir": str(caller._logdir)
            },
            call_type=FCI.CONSTRAINED
        )

        # Get callgraph instance directly
        graph = await CallGraphManager.get_instance()

        # Register agent in callgraph first
        await graph.start_agent(
            agent_name=sub_agent.AGENT_NAME,
            node_id=sub_agent._id,
            args=sub_agent.model_dump()
        )

        # Execute the agent call
        result = await execute_agent_call(sub_agent, caller, agent_content)

        # Get the event bus instance directly
        bus = await EventBus.get_instance()

        # Check event bus for correct event publication
        agent_events = bus.get_events(caller._id)
        call_events = [e for e in agent_events if e.type == EventType.AGENT_CALL]
        result_events = [e for e in agent_events if e.type == EventType.AGENT_RESULT]

        assert len(call_events) > 0, "No agent call events were published"
        assert len(result_events) > 0, "No agent result events were published"

        # Verify the result
        assert result.status == AgentStatus.SUCCESS
        assert result.result == "Task completed successfully"

        # Verify node in callgraph
        node = graph.graph.get_node(sub_agent._id)  # Use get_node() directly
        assert node is not None, "Node not found in callgraph"
        assert node.result == "Task completed successfully"
        assert node.success

    @pytest.mark.asyncio
    async def test_error_handling(self, parent_agent):
        """Test handling of agent execution errors."""
        caller = parent_agent

        # Register test agent in registry
        agent_registry["test_agent"] = SimpleTestAgent

        # Create a sub-agent that will raise an error
        class ErrorAgent(SimpleTestAgent):
            async def execute(self):
                raise Exception("Test error")

        # Create the agent instance
        sub_agent = ErrorAgent(
            parent=caller,
            workdir=caller._workdir,
            logdir=caller._logdir
        )

        # Get callgraph instance directly
        graph = await CallGraphManager.get_instance()

        # Register agent in callgraph first
        await graph.start_agent(
            agent_name=sub_agent.AGENT_NAME,
            node_id=sub_agent._id,
            args=sub_agent.model_dump()
        )

        agent_content = ToolCallContent(
            call_id="error_call",
            tool_name="test_agent",
            tool_args={
                "workdir": str(caller._workdir),
                "logdir": str(caller._logdir)
            },
            call_type=FCI.CONSTRAINED
        )

        # Execute and verify error handling
        result = await execute_agent_call(sub_agent, caller, agent_content)

        assert result.status == AgentStatus.ERROR
        assert "Test error" in result.result

        # Get event bus instance directly
        bus = await EventBus.get_instance()

        # Verify error events were published
        agent_events = bus.get_events(caller._id)
        result_events = [e for e in agent_events if e.type == EventType.AGENT_RESULT]

        # The error should be in the agent result content
        assert len(result_events) > 0, "No result events were published"
        assert any("Test error" in e.content for e in result_events)

        # Verify callgraph error recording
        node = graph.graph.get_node(sub_agent._id)
        assert node is not None, "Node not found in callgraph"
        assert not node.success

    @pytest.mark.asyncio
    async def test_agent_cancellation(self, parent_agent):
        """Test proper handling of agent cancellation."""
        caller = parent_agent

        # Get event bus instance early
        bus = await EventBus.get_instance()

        # Register test agent in registry
        agent_registry["test_agent"] = SimpleTestAgent

        # Create a sub-agent that will sleep
        class SlowAgent(SimpleTestAgent):
            async def execute(self):
                # Short sleep that we can cancel
                await asyncio.sleep(0.5)
                return AgentResult(
                    agent_name=self.AGENT_NAME,
                    status=AgentStatus.SUCCESS,
                    result="Should not complete",
                    metrics=self._metrics
                )

        # Create the agent instance
        sub_agent = SlowAgent(
            parent=caller,
            workdir=caller._workdir,
            logdir=caller._logdir
        )

        # Get callgraph instance directly
        graph = await CallGraphManager.get_instance()

        # Create task with agent content
        agent_content = ToolCallContent(
            call_id="test_call",
            tool_name="test_agent",
            tool_args={
                "workdir": str(caller._workdir),
                "logdir": str(caller._logdir)
            },
            call_type=FCI.CONSTRAINED
        )

        # Set up the agent node
        await graph.start_agent(
            agent_name=sub_agent.AGENT_NAME,
            node_id=sub_agent._id,
            args=sub_agent.model_dump()
        )

        # Start agent execution
        task = asyncio.create_task(execute_agent_call(sub_agent, caller, agent_content))
        await asyncio.sleep(0.1)  # Let agent start running

        # Register task with callgraph
        await graph.register_agent_task(sub_agent._id, task)

        # Cancel and get result
        task.cancel()
        try:
            result = await task
        except asyncio.CancelledError:
            result = AgentResult(
                agent_name=sub_agent.AGENT_NAME,
                status=AgentStatus.CANCELLED,
                result="Agent was cancelled",
                metrics=sub_agent._metrics
            )

        # Let any event publications complete
        await asyncio.sleep(0.1)

        assert result.status == AgentStatus.CANCELLED
        assert "cancelled" in result.result.lower()

        # Verify cancellation in callgraph
        node = graph.graph.get_node(sub_agent._id)
        assert node is not None, "Node not found in callgraph"
        assert not node.success

        # Check either error or result field has cancellation message
        cancellation_msg = (node.error or "").lower() + (node.result or "").lower()
        assert "cancelled" in cancellation_msg, "No cancellation message found in node"

        # Check event publication after allowing time for events
        agent_events = bus.get_events(caller._id)
        result_events = [e for e in agent_events if e.type == EventType.AGENT_RESULT]

        assert len(result_events) > 0, "No result events published"

        # The cancellation might be in result content or error field
        cancellation_found = False
        for event in result_events:
            event_content = event.content.lower()
            if "cancelled" in event_content:
                cancellation_found = True
                break

        assert cancellation_found, "No cancellation event found"
