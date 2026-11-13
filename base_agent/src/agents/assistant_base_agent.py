# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
import asyncio

from abc import abstractmethod
from uuid import uuid4
from typing import Set, Type, ClassVar, Any, Dict
from pathlib import Path
from datetime import datetime
from pydantic import PrivateAttr, TypeAdapter

from ..llm import create_completion, Message
from ..events import EventBus
from ..schemas import get_schema_representation, dumps, args_str_to_dict
from ..llm.api import get_tool_documentation
from ..tools.base_tool import handle_tool_call, tool_registry
from ..utils.stop_tokens import AGENT_STOP_TOKEN, TOOL_STOP_TOKEN
from ..callgraph.manager import CallGraphManager
from ..utils.file_views import (
    FileTreeOptions,
    create_filetree,
    get_file_edit_view,
    get_file_view,
)
from ..events.event_bus_utils import (
    get_open_file_set,
    get_file_content_size,
    get_latest_core_prompt_event,
    get_subagent_events,
)
from ..types.tool_types import ToolInterface
from ..types.agent_types import AgentInterface, AgentStatus, AgentMetrics, AgentResult, InheritanceFlags, InheritanceConfig
from ..types.event_types import EventType, Event, FileOperation, FileEvent
from ..types.llm_types import FCI, Model, StopReason, TextContent, ReasoningContent, ToolCallContent, ToolResultContent, ContentTypes
from ..types.common import ArgFormat

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Create an empty registry dictionary.
agent_registry: dict[str, type[AgentInterface]] = {}


class BaseAgent(AgentInterface):
    """
    Abstract base class for all agents.
    """

    # Required class-level attributes
    AGENT_NAME: ClassVar[str]
    AGENT_DESCRIPTION: ClassVar[str]
    SYSTEM_PROMPT: ClassVar[str]

    # Optional class-level configuration
    AVAILABLE_TOOLS: ClassVar[Set[Type[ToolInterface]]] = set()
    AVAILABLE_AGENTS: ClassVar[Set[Type[AgentInterface]]] = set()

    HAS_FILEVIEW: ClassVar[bool] = True
    MODEL: ClassVar[Model] = Model.SONNET_55
    TEMPERATURE: ClassVar[float] = 0.666
    MAX_ITERATIONS: ClassVar[int] = 500
    INHERITANCE: ClassVar[InheritanceConfig] = InheritanceConfig()

    # Private agent attributes (not arguments)
    _workdir: Path = PrivateAttr(default=Path.home() / "workdir")
    _logdir: Path = PrivateAttr(default=Path.home() / "workdir" / "agent_outputs")
    _debug_mode: bool = PrivateAttr(default=False)  # TODO: make this a global configuration
    _local_state: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _parent_id: str | None = PrivateAttr(default=None)
    _id: str = PrivateAttr(default_factory=lambda: f"agent_{uuid4().hex[:8]}")
    _available_agents: Set[Type[AgentInterface]] = PrivateAttr(default_factory=set)
    _available_agent_names: Set[str] = PrivateAttr(default_factory=set)
    _available_tools: Set[Type[ToolInterface]] = PrivateAttr(default_factory=set)
    _available_tool_names: Set[str] = PrivateAttr(default_factory=set)
    _metrics: AgentMetrics = PrivateAttr(default_factory=lambda: AgentMetrics(start_time=datetime.now()))

    # Fields from subagent classes will go here to represent runtime arguments...

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        parent: 'BaseAgent' | None = None,
        workdir: Path | None = None,
        logdir: Path | None = None,
        debug_mode: bool = False,
        **data
    ):
        super().__init__(**data)

        if parent:
            self._parent_id = parent._id

        # Initialise the workdir
        if workdir:
            self._workdir = Path(workdir)
        elif parent:
            self._workdir = parent._workdir
        self._workdir.mkdir(parents=True, exist_ok=True)

        # Initialise the log directory
        if logdir:
            self._logdir = Path(logdir)
        elif parent:
            self._logdir = parent._logdir
        self._logdir.mkdir(parents=True, exist_ok=True)

        self._debug_mode = debug_mode

        self._available_agents = self.AVAILABLE_AGENTS
        self._available_agent_names = {a.AGENT_NAME for a in self._available_agents}

        # Setup this agent's tools
        from ..tools.base_agent_tools import ReturnResult, ExitAgent, RegenerateContext, Complete
        from ..tools.answer_submission import SubmitAnswer

        self._available_tools = self.AVAILABLE_TOOLS | {ExitAgent, RegenerateContext, ReturnResult, Complete}
        if self.AGENT_NAME == "main":
            self._available_tools.add(SubmitAnswer)
            self._available_tools.remove(ReturnResult)
            # TODO: remove exit agent here too?

        self._available_tool_names = {t.TOOL_NAME for t in self._available_tools}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Skip registering the BaseAgent class itself.
        if cls.__name__ != "BaseAgent":
            agent_registry[cls.AGENT_NAME] = cls

    @abstractmethod
    async def construct_core_prompt(self) -> str:
        """Composes and returns the agent's 'core prompt'"""
        pass

    @classmethod
    @abstractmethod
    def generate_examples(cls) -> list[tuple["BaseAgent", AgentResult]]:
        """Generate example uses of the tool with their expected outputs"""
        pass

    @classmethod
    async def args_str_to_dict(
        cls, args_str: str, arg_format: ArgFormat = ArgFormat.XML
    ) -> tuple[dict | None, str | None]:
        """Allows sub-agents to override the argument parsing"""
        args_dict, parse_warnings = await args_str_to_dict(
            args_str, guide_obj=cls, arg_format=arg_format, root_tag="TOOL_ARGS"
        )
        return args_dict, parse_warnings

    async def construct_system_prompt(self) -> str:
        """Constructs the system prompt"""
        event_bus = await EventBus.get_instance()

        parts: list[str] = [self.SYSTEM_PROMPT]

        parts.append(
            f"Very important instruction:\nOnce you are satisfied that the task is complete and whatever answer or response has been submitted, you MUST generate call your `complete` tool to exit."
        )

        # For constrained tool generation,
        if self.MODEL.fci == FCI.CONSTRAINED:
            parts.append(get_tool_documentation(
                available_tools=list(self._available_tools | self._available_agents),
                arg_format=self.MODEL.arg_format,
                plain=True
            ))

        sys_prompt = "\n\n".join(parts)

        await event_bus.publish(
            Event(type=EventType.SYSTEM_PROMPT_UPDATE, content=sys_prompt), self._id
        )

        # If debugging, save the system prompt to disk
        if self._debug_mode:
            all_events = event_bus.get_events(self._id)
            system_prompts = [
                e for e in all_events if e.type == EventType.SYSTEM_PROMPT_UPDATE
            ]
            sys_path = (
                self._logdir
                / "contexts"
                / f"{self._id}_system_prompt_{len(system_prompts):01d}.txt"
            )
            sys_path.parent.mkdir(exist_ok=True, parents=True)
            sys_path.write_text(sys_prompt)

        return sys_prompt

    def get_filetree_for_core_prompt(self, path: Path | None = None) -> str:
        options = FileTreeOptions(
            collapse_threshold=20,
            max_chars=5000,
            show_hidden=True,
            show_mtime=True,
            min_dir_level=1,
            show_full_path=True,
        )
        workdir = path if path else self._workdir
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_tree = create_filetree(workdir, options)
        tree_prompt = f"\nYour work directory has been set as {workdir}"
        tree_prompt += f"\nHere are the files in your work directory, as of {now}:\n"
        tree_prompt += f"<FILETREE>\n{file_tree}</FILE_TREE>\n"
        tree_prompt += f"\nNote that these files may change over time, and you should use your view directory tool to get an up-to-date view into a particular directory.\n"
        return tree_prompt

    async def get_open_files_for_core_prompt(self) -> str:
        """Get concatenated file views for open files"""
        file_prompt = ""
        open_files = await get_open_file_set(self._id)

        if InheritanceFlags.OPEN_FILES in self.INHERITANCE.flags and self._parent_id:
            open_files |= await get_open_file_set(self._parent_id)

        if len(open_files) > 0:
            file_warnings = []
            file_prompt += (
                f"\nHere are the files you have opened for viewing:\n<FILE_VIEWER>\n"
            )
            for open_event in open_files:
                show_line_numbers = False
                if open_event is not None:
                    show_line_numbers = open_event.metadata.get(
                        "show_line_numbers", False
                    )
                file_view, warning = await get_file_view(
                    open_event.path,
                    show_line_numbers,
                    show_diagnostics=False,  # Temporarily disabled
                )
                if warning:
                    file_warnings.append(warning)
                    continue
                if file_view:
                    file_prompt += file_view

            file_prompt += "</FILE_VIEWER>\n"
            file_prompt += "Note that the file contents, location and other properties shown above may be out of date depending on operations you or other processes take following this message. Use your file and directory viewing tools to get up-to-date representations of content.\n"
            if len(file_warnings):
                warnings = "\n".join(file_warnings)
                file_prompt += f"The following warnings arose when viewing the files:\n{warnings}\n"

        return file_prompt

    async def compose_core_prompt(self) -> str:
        """Composes the core prompt"""
        core_prompt = await self.construct_core_prompt()

        if self.HAS_FILEVIEW:
            core_prompt += self.get_filetree_for_core_prompt()
            core_prompt += await self.get_open_files_for_core_prompt()

        # Inherit from parent
        if (
            InheritanceFlags.CORE_DIALOGUE in self.INHERITANCE.flags
            and self._parent_id
        ):
            parent_core = await get_latest_core_prompt_event(self._parent_id)
            parent_prefill = await self.get_assistant_prefill_messages(
                parent_core, for_agent_id=self._parent_id,
            )
            core_prompt += f"\nThe prompt and dialogue history of your caller was as follows:\nBegin Caller Prompt {'='*70}\n"
            core_prompt += parent_core.content if parent_core else ""
            core_prompt += "\n".join([str(msg) for msg in parent_prefill])
            # for msg in parent_prefill:
            #     core_prompt += f"\n{msg.role}:\n{msg.content}"
            core_prompt += f"\nEnd Caller Prompt {'='*70}\nThis marks the end of the prompt and dialogue history of your calling agent."

        core_prompt += f"\nThe time at which this prompt was written was {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"

        event_bus = await EventBus.get_instance()
        await event_bus.publish(
            Event(type=EventType.CORE_PROMPT_UPDATE, content=core_prompt),
            self._id,
        )

        # Save the core prompt to disk for debugging
        if self._debug_mode:
            all_events = event_bus.get_events(self._id)
            system_prompts = [
                e for e in all_events if e.type == EventType.CORE_PROMPT_UPDATE
            ]
            core_path = (
                self._logdir
                / "contexts"
                / f"{self._id}_core_prompt_{len(system_prompts):05d}.txt"
            )
            core_path.parent.mkdir(exist_ok=True, parents=True)
            core_path.write_text(core_prompt)

        self._local_state["should_regenerate"] = False

        return core_prompt

    @classmethod
    def to_prompt_format(cls, arg_format: ArgFormat = ArgFormat.XML) -> str:
        """Convert the agent definition to string format"""
        # TODO: allow just the examples to be returned, in plain format
        examples = cls.generate_examples()

        examples_str = []
        for agent_instance, example_output in examples:
            instance_data = dict(
                agent_name=cls.AGENT_NAME, args=agent_instance.model_dump()
            )

            examples_str.append(
                f"""<EXAMPLE>
<AGENT_CALL>
<AGENT_NAME>{cls.AGENT_NAME}</AGENT_NAME>
<AGENT_ARGS>
{dumps(instance_data, arg_format, indent=2)}
</AGENT_ARGS>
{AGENT_STOP_TOKEN}
{example_output}
</EXAMPLE>"""
            )
        examples_joined = "\n\n".join(examples_str)

        tools_available = ", ".join(t.TOOL_NAME for t in cls.AVAILABLE_TOOLS)
        agents_available = ", ".join(f.AGENT_NAME for f in cls.AVAILABLE_AGENTS)

        result = f"""\n## `{cls.AGENT_NAME}` Agent Documentation

{cls.AGENT_DESCRIPTION}

"""
        if tools_available != "":
            result += f"Here are the tools that the `{cls.AGENT_NAME}` agent has available to it: {tools_available}\n"

        if agents_available != "":
            result += f"Here are the sub-agents that the `{cls.AGENT_NAME}` agent has available to it: {agents_available}\n"

        result += f"""
<AGENT_ARGS_SCHEMA>
{get_schema_representation(cls, arg_format=arg_format, root_tag="AGENT_ARGS")}
</AGENT_ARGS_SCHEMA>

{examples_joined}

This concludes the {cls.AGENT_NAME} agent documentation.
"""
        return result

    @classmethod
    def to_plain_prompt_format(cls, arg_format: ArgFormat = ArgFormat.XML) -> str:
        """Convert the agent definition to a formatted string for the constrained tool use prompt"""
        examples = cls.generate_examples()

        examples_str = []
        for i, (agent_instance, example_output) in enumerate(examples):
            examples_str.append(
                f"""### `{cls.AGENT_NAME}` parameter example {i}

When the arguments are set to:
{dumps(agent_instance.model_dump(), arg_format, indent=2)}

The output might look like:
{example_output}
"""
            )
        examples_joined = "\n\n".join(examples_str)

        tools_available = ", ".join(t.TOOL_NAME for t in cls.AVAILABLE_TOOLS)
        agents_available = ", ".join(f.AGENT_NAME for f in cls.AVAILABLE_AGENTS)

        result = f"""\n## `{cls.AGENT_NAME}` Agent Documentation

{cls.AGENT_DESCRIPTION}

"""
        if tools_available != "":
            result += f"Here are the tools that the `{cls.AGENT_NAME}` agent has available to it: {tools_available}\n"

        if agents_available != "":
            result += f"Here are the sub-agents that the `{cls.AGENT_NAME}` agent has available to it: {agents_available}\n"

        result += f"""
The agent's arguments are as follows:

{get_schema_representation(cls, arg_format=arg_format, root_tag="AGENT_ARGS")}

Here are some examples of how the arguments might be set and the associated results:

{examples_joined}

This concludes the {cls.AGENT_NAME} agent documentation.
"""
        return result

    async def get_assistant_prefill_messages(
        self,
        last_core_prompt_event: Event | None = None,
        enable_overseer_notifications: bool = True,
        for_agent_id: str | None = None,
        prefill_message_types: set[EventType] | None = None
    ) -> list[Message]:
        """
        Composes a 'view' on the event bus, adhering to context inheritance
        rules and tool calling interfaces to compose the assistant's context.
        """
        messages: list[Message] = []
        current_message_content: list[ContentTypes] = []
        current_role: str | None = "assistant"

        def ensure_role(role: str):
            """
            This 'commits' any accumulated message content if the current role
            differs from the message role that the event should be associated
            with.
            """
            nonlocal current_message_content, current_role
            if current_role != role and current_role and current_message_content:
                messages.append(
                    Message(role=current_role, content=current_message_content)
                )
                current_message_content = []
            current_role = role

        # Used to decide what file events to show in full and which to
        # collapse; knowing that a newer view of the open files will be shown
        # in the file viewer at the end of the core prompt.
        last_core_regeneration_time = (
            last_core_prompt_event.timestamp if last_core_prompt_event else datetime.min
        )

        # The types of messages we deem 'interesting' to the agent, and which
        # we'll include in the prefill
        if prefill_message_types is None:
            prefill_message_types = {
                EventType.ASSISTANT_MESSAGE,
                # EventType.ASSISTANT_REASONING,
                EventType.TOOL_CALL,
                EventType.TOOL_RESULT,
                EventType.AGENT_CALL,
                EventType.AGENT_RESULT,
                EventType.FILE_EVENT,
                EventType.OVERSEER_NOTIFICATION,
                EventType.APPLICATION_ERROR,
                EventType.APPLICATION_WARNING,
                EventType.EXTERNAL_MESSAGE,
            }

        target_id = for_agent_id if for_agent_id else self._id
        events = await get_subagent_events(target_id, prefill_message_types)

        for event in events:
            # Do special event type handling first; starting with file events:
            if event.type == EventType.FILE_EVENT:
                assert isinstance(event, FileEvent)

                if self.MODEL.fci == FCI.CONSTRAINED:
                    ensure_role("assistant")

                match event.operation:
                    case FileOperation.OPEN:
                        if event.timestamp > last_core_regeneration_time:
                            file_view, errors = await get_file_view(
                                event.path,
                                event.metadata.get("show_line_numbers", False),
                                show_diagnostics=False,
                            )
                            if file_view:
                                current_message_content.append(
                                    TextContent(text=file_view)
                                )
                            else:
                                file_msg = f"\n<FILE>\n<PATH>{event.path}</PATH>\n"
                                file_msg += f"<ERROR>\nThere was an error getting the file contents: {errors}<ERROR>\n"
                                file_msg += "</FILE>\n"
                                current_message_content.append(TextContent(text=file_msg))
                        else:
                            file_msg = f"\n<FILE>\n<PATH>{event.path}</PATH>\n"
                            file_msg += f"<NOTICE>\nThe file was opened at this point. If still open, you should be able to see its content, which may be updated since when the file was opened at this time, in the FILE_VIEWER at the top of your context.\n</NOTICE>\n</FILE>\n"
                            current_message_content.append(TextContent(text=file_msg))
                    case FileOperation.EDIT:
                        if event.timestamp > last_core_regeneration_time:
                            edit_view = await get_file_edit_view(
                                file=Path(event.path),
                                edit_diff=event.content,
                                show_diagnostics=False,
                            )
                            if edit_view:
                                current_message_content.append(TextContent(text=edit_view))
                            # we don't handle the case when the edit_view is
                            # none, since the edit should be fairly clear
                            # inline from the previous tool call.
                        else:
                            file_msg = f"\n<FILE>\n<PATH>{event.path}</PATH>\n"
                            file_msg += f"<NOTICE>\nThe file was edited at this point, and your context window was consolidated since. If the file is still open, you should be able to see its content in the FILE_VIEWER at the top of your context.\n</NOTICE>\n</FILE>\n"
                            current_message_content.append(TextContent(text=file_msg))
                    case FileOperation.CLOSE:
                        file_msg = f"\n<FILE>\n<PATH>{event.path}</PATH>\n"
                        file_msg += (
                            "<NOTICE>\nThe file was closed at this point.\n</NOTICE>"
                        )
                        file_msg += "\n</FILE>\n"
                        current_message_content.append(TextContent(text=file_msg))

            elif (
                event.type == EventType.OVERSEER_NOTIFICATION
                and enable_overseer_notifications
            ):
                if self.MODEL.fci == FCI.CONSTRAINED:
                    ensure_role("user")
                current_message_content.append(TextContent(text=f"""
<OVERSEER_NOTIFICATION - {event.timestamp.strftime("%H:%M:%S")}>
{event.content}
</OVERSEER_NOTIFICATION>
"""))

            elif event.type == EventType.APPLICATION_ERROR:
                if self.MODEL.fci == FCI.CONSTRAINED:
                    ensure_role("user")
                current_message_content.append(TextContent(text=f"""
<APPLICATION_ERROR - {event.timestamp.strftime("%H:%M:%S")}>
{event.content}
</APPLICATION_ERROR>
"""))

            elif event.type == EventType.APPLICATION_WARNING:
                if self.MODEL.fci == FCI.CONSTRAINED:
                    ensure_role("user")
                current_message_content.append(TextContent(text=f"""
<APPLICATION_WARNING - {event.timestamp.strftime("%H:%M:%S")}>
{event.content}
</APPLICATION_WARNING>
"""))
            elif event.type == EventType.EXTERNAL_MESSAGE:
                # NOTE: unused. During benchmarking and self-improvement, the user has no input
                if self.MODEL.fci == FCI.CONSTRAINED:
                    ensure_role("user")
                current_message_content.append(TextContent(text=f"""
<EXTERNAL_USER_MESSAGE - {event.timestamp.strftime("%H:%M:%S")}>
{event.content}
</EXTERNAL_USER_MESSAGE>
"""))

            elif event.type in {EventType.TOOL_CALL, EventType.AGENT_CALL}:
                # If native tool calling, or constrained tool calling, this
                # goes into the assistant message
                if self.MODEL.fci == FCI.CONSTRAINED:
                    ensure_role("assistant")
                    current_message_content.append(ToolCallContent(
                        call_id=event.metadata["call_id"],
                        tool_name=event.metadata["name"],
                        tool_args=event.metadata["args"],
                        call_type=event.metadata["call_type"],
                    ))
                else:
                    # In unconstrained mode, the tool call is just a normal generation
                    # which happens to be parsed
                    current_message_content.append(TextContent(
                        text=event.content,
                    ))

            elif event.type in {EventType.TOOL_RESULT, EventType.AGENT_RESULT}:
                if self.MODEL.fci == FCI.CONSTRAINED:
                    ensure_role("user")
                    current_message_content.append(ToolResultContent(
                        call_id=event.metadata["call_id"],
                        content=event.content,
                        tool_name=event.metadata["name"]
                    ))
                else:
                    current_message_content.append(TextContent(text=event.content))

            elif event.type in prefill_message_types:
                # Catch-all for all other prefill event types
                if self.MODEL.fci == FCI.CONSTRAINED:
                    ensure_role("assistant")
                current_message_content.append(TextContent(text=event.content))

        if len(current_message_content) > 0 and current_role is not None:
            messages.append(Message(role=current_role, content=current_message_content))

        return messages

    def _update_metrics(self, completion) -> None:
        """Update metrics from a completion."""
        self._metrics.token_usage += completion.usage
        self._metrics.cost += completion.calculate_cost()

    async def _handle_tool_call(self, tool_content: ToolCallContent):
        """Handle the tool call"""
        if tool_content.tool_name not in [t.TOOL_NAME for t in self._available_tools]:
            event_bus = await EventBus.get_instance()
            await event_bus.publish(
                Event(
                    type=EventType.APPLICATION_ERROR,
                    content=f"Tool {tool_content.tool_name} not available in your current tool set. Please invoke an appropriate sub-agent that does have this tool available to call it.",
                ),
                self._id,
            )
            return

        self._metrics.tool_calls += 1
        try:
            async with asyncio.timeout(600):  # 10 min safety measure
                await handle_tool_call(tool_content, self)
        except Exception as e:
            event_bus = await EventBus.get_instance()
            await event_bus.publish(
                Event(
                    type=EventType.APPLICATION_ERROR,
                    content=f"Tool execution failed: {e}",
                ),
                self._id,
            )

    async def _handle_agent_call(self, agent_content: ToolCallContent):
        """Handle agent call"""
        from .agent_calling import execute_agent_call, handle_parse_errors

        # Handle any parse errors
        if agent_content.parse_errors:
            await handle_parse_errors(agent_content, self)
            return

        if agent_content.tool_name not in [t.AGENT_NAME for t in self._available_agents]:
            event_bus = await EventBus.get_instance()
            await event_bus.publish(
                Event(
                    type=EventType.APPLICATION_ERROR,
                    content=f"Tool {agent_content.tool_name} not available in your current agent set. Please carefully read the list of sub-agents (if any) available to you.",
                ),
                self._id,
            )
            return

        self._metrics.agent_calls += 1
        try:
            agent_cls = agent_registry.get(agent_content.tool_name)

            validated_agent = TypeAdapter(agent_cls).validate_python(
                agent_content.tool_args | {'parent': self, 'workdir': self._workdir, 'logdir': self._logdir, 'debug_mode': self._debug_mode}
            )

            await execute_agent_call(validated_agent, self, agent_content)

        except Exception as e:
            event_bus = await EventBus.get_instance()
            await event_bus.publish(
                Event(
                    type=EventType.APPLICATION_ERROR,
                    content=f"Agent execution failed: {e}",
                ),
                self._id,
            )

    async def execute(self) -> AgentResult:
        """
        Executes this agent with parameter values provided in the fields

        Returns:
            The result of the reasoning run, as an AgentResult
        """
        callgraph = await CallGraphManager.get_instance()
        event_bus = await EventBus.get_instance()

        status = AgentStatus.RUNNING

        try:
            # First compose the system prompt (adds the SYSTEM_PROMPT_UPDATE event to event bus)
            system_prompt = await self.construct_system_prompt()

            for iteration in range(self.MAX_ITERATIONS):

                file_content_size = await get_file_content_size(self._id)
                core_event = await get_latest_core_prompt_event(self._id)
                # Note: should_regenerate is reset at the end of compose_core_prompt
                regen_core = False
                regen_core |= file_content_size > int(4e5)  # approx 100k context; assuming ~4 bytes per token
                regen_core |= self._local_state.get("should_regenerate", False)

                if regen_core or core_event is None:
                    core_prompt = await self.compose_core_prompt()
                    core_event = await get_latest_core_prompt_event(self._id)
                else:
                    core_prompt = core_event.content

                assistent_prefill_messages = await self.get_assistant_prefill_messages(
                    core_event,
                )

                messages = [
                    Message(role="system", content=[TextContent(text=system_prompt)]),
                    Message(role="user", content=[TextContent(text=core_prompt)]),
                    *assistent_prefill_messages,
                ]

                # Debug logging of messages -------------
                if self._debug_mode:
                    ctx_path = (
                        self._logdir
                        / "contexts"
                        / f"{self._id}_{iteration:05d}.txt"
                    )
                    ctx_path.parent.mkdir(exist_ok=True, parents=True)
                    # skip sys and core
                    msg_string = "\n\n".join(
                        # [f"{m.role}:\n{m.content}" for m in messages[2:]]
                        [str(m) for m in messages[:]]
                    )
                    if msg_string:
                        ctx_path.write_text(msg_string)
                # Debug logging of messages -------------

                logger.info(f"Awaiting completion for iteration {iteration} ({len(messages)} messages, {sum(len(m.content) for m in messages)} content blocks)...")
                completion = await create_completion(
                    messages=messages,
                    model=self.MODEL,
                    temperature=self.TEMPERATURE,
                    stop=[],
                    available_tools=list(self._available_tools | self._available_agents),
                    max_continuations=5,
                )
                logger.info(f"Completion received for iteration {iteration}.")

                self._update_metrics(completion)

                await callgraph.track_tokens(
                    node_id=self._id,
                    token_count=completion.usage.total_tokens,
                    num_cached_tokens=completion.usage.cached_prompt_tokens,
                    cost=completion.calculate_cost(),
                )


                num_tools = 0
                for block in completion.content:
                    if isinstance(block, TextContent):
                        if block.text.rstrip() != "":
                            await event_bus.publish(
                                Event(
                                    type=EventType.ASSISTANT_MESSAGE,
                                    content=block.text.rstrip(),
                                    metadata=dict(completion=completion)
                                ),
                                self._id
                            )
                        # else:
                        #     # if len(completion.content) == 1:
                        #     if num_tools == 0:
                        #         await event_bus.publish(
                        #             Event(
                        #                 type=EventType.APPLICATION_ERROR,
                        #                 content=f"Your last response was an empty string. Please closely read your tool use instructions again. If you intend to exit, call your `early_exit` or `complete` tool.",
                        #             ),
                        #             self._id,
                        #         )
                    elif isinstance(block, ReasoningContent):
                        if block.text.rstrip() != "":
                            await event_bus.publish(
                                Event(
                                    type=EventType.ASSISTANT_REASONING,
                                    content=block.text.rstrip(),
                                    metadata=dict(completion=completion)
                                ),
                                self._id
                            )
                    elif isinstance(block, ToolCallContent):
                        num_tools += 1
                        # Determine whether this is an agent call or a tool call
                        if block.tool_name in tool_registry:
                            await self._handle_tool_call(block)
                        elif block.tool_name in agent_registry:
                            await self._handle_agent_call(block)

                if self._local_state.get("exec_complete", False):
                    # Handle normal completion
                    status = AgentStatus.SUCCESS
                    break
                elif self._local_state.get("needs_exit", False):
                    # Handle early exit
                    status = AgentStatus.INCOMPLETE
                    break
                elif completion.stop_reason == StopReason.COMPLETE and num_tools == 0:
                    error_msg = f"You completed with a normal stop token and no tool calls. If this was unintentional, closely read your tool use instructions again. If you wish to exit the agent loop, you MUST do so by invoking the `complete` tool."
                    if self.MODEL.fci == FCI.UNCONSTRAINED:
                        error_msg += f"Remember, to invoke a tool you must directly generate <TOOL_CALL>...{TOOL_STOP_TOKEN}, and to invoke an agent you must use <AGENT_CALL>...{AGENT_STOP_TOKEN} in your message."
                    await event_bus.publish(
                        Event(
                            type=EventType.APPLICATION_ERROR,
                            content=error_msg,
                        ),
                        self._id,
                    )
                elif completion.stop_reason in {StopReason.LENGTH, StopReason.ERROR}:
                    await event_bus.publish(
                        Event(
                            type=EventType.APPLICATION_ERROR,
                            content=f"Agent {self._id} completed abnormally due to stop reason: {completion.stop_reason.value}",
                        ),
                        self._id,
                    )
                    # status = AgentStatus.INCOMPLETE
                    # break

            self._metrics.end_time = datetime.now()

            result = self._local_state.get(
                "return_value", "No result value returned."
            )
            early_exit_reason = self._local_state.get("exit_reason")

            return AgentResult(
                agent_name=self.AGENT_NAME,
                status=status,
                result=result,
                warnings=early_exit_reason,
                metrics=self._metrics,
            )

        except Exception as e:
            self._metrics.end_time = datetime.now()
            logger.info(f"Agent failed: {e}")
            raise e
