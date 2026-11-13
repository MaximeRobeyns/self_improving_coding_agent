# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Asynchronous oversight module for monitoring agent execution.
"""

import logging
import asyncio

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, TypeAdapter

from ..events import EventBus
from ..callgraph.manager import CallGraphManager
from .graph_visualisation import generate_overseer_execution_tree
from ..llm.api import create_completion
from ..llm.base import Message, Model
from ..schemas import xml_str_to_dict
from ..utils.stop_tokens import OVERSEER_STOP_TOKEN
from ..utils.parsing import extract_between_patterns
from ..types.event_types import EventType, Event
from ..types.llm_types import ReasoningEffort, TextContent

OVERSEER_RELEVANT_EVENTS = {
    EventType.TOOL_CALL,
    EventType.TOOL_RESULT,
    EventType.AGENT_CALL,
    EventType.AGENT_RESULT,
    EventType.APPLICATION_WARNING,
    EventType.APPLICATION_ERROR,
    EventType.FILE_EVENT,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OverseerJudgement(BaseModel):

    making_progress: bool = Field(
        ...,
        description="Whether the agent appears to be making good progress in the most recent agent",
    )
    is_looping: bool = Field(
        ..., description="Whether the currently running agent appears to be looping"
    )
    currently_running_agent: str = Field(
        ...,
        description="The id of the currently running (most deeply nested) agent, of the form `agent_abcdefgh`",
    )
    needs_notification_reasoning: str = Field(
        ...,
        description="Resoning about whether a particular agent needs to be notified about any topic. Try to avoid unecessary notifications and 'noise'. If you are cancelling an agent, reason about it here.",
    )
    needs_notification: bool = Field(
        ...,
        description="Whether a particular agent needs to be notified of something. If cancelling an agnet, you MUST notify its parent in order to explain the reason for cancellation of its child agent.",
    )
    agent_to_notify: Optional[str] = Field(
        default=None,
        description="The valid id of a running or caller agent, of the form `agent_abcdefgh`",
    )
    notification_content: Optional[str] = Field(
        default=None, description="The contents of the notification"
    )
    notes_for_next_iteration: Optional[str] = Field(
        default=None,
        description="Any notes you wish to leave to your next iteration. This allows you to persist state, observations or hypotheses across iterations",
    )
    next_check_type: str = Field(
        default="time", description="How to schedule next check: 'time' or 'events'"
    )
    next_check_delay: Optional[float] = Field(
        default=None,
        description="If next_check_type is 'time', delay in seconds (5-500)",
    )
    target_agent_id: Optional[str] = Field(
        default=None, description="If next_check_type is 'events', agent ID to monitor"
    )
    pending_events_count: Optional[int] = Field(
        default=None,
        description="If next_check_type is 'events', number of events to wait for",
    )
    force_cancel_agent: bool = Field(
        default=False, description="Whether to forcefully cancel an agent"
    )
    force_cancel_agent_id: Optional[str] = Field(
        default=None, description="ID of the agent to forcefully cancel"
    )


# Few-shot examples for the overseer
OVERSEER_EXAMPLES = f"""
Example 1 - Successful Task Completion

<OVERSEER_JUDGEMENT>
<making_progress>true</making_progress>
<is_looping>false</is_looping>
<currently_running_agent>agent_6697d108</currently_running_agent>
<needs_notification_reasoning>The execution is progressing through the defined agents. Each agent is performing its intended task without unnecessary repetition or delays. No agent appears to be stuck or requires a notification</needs_notification_reasoning>
<needs_notification>false</needs_notification>
<next_check_type>time</next_check_type>
<next_check_delay>60</next_check_delay>
<notes_for_next_iteration>Main agent healthy with ~7k tokens/msg and 4-5s ttft - alert if: no progress for >15s, total time >90s, >20s without tool calls, ttft >10s consistently, or if pattern deviates from explore->solve->submit</notes_for_next_iteration>
{OVERSEER_STOP_TOKEN}

Example 2 - Agent is looping but can be redirected:

<OVERSEER_JUDGEMENT>
<making_progress>false</making_progress>
<is_looping>true</is_looping>
<currently_running_agent>agent_r9q0p1o2</currently_running_agent>
<needs_notification_reasoning>The `check_user` agent [agent_n3m4l5k6] is being called repeatedly on the same user without advancing to other users. This suggests a looping behavior where the agent is stuck examining the same user without making progress. Currently, the `find_user` agent [agent_r9q0p1o2] is running, and should be notified.</needs_notification_reasoning>
<needs_notification>true</needs_notification>
<agent_to_notify>agent_r9q0p1o2</agent_to_notify>
<notification_content>You are repeatedly checking the same user. You should move on to the next user after a failed check.</notification_content>
<next_check_type>events</next_check_type>
<target_agent_id>agent_r9q0p1o2</target_agent_id>
<pending_events_count>3</pending_events_count>
{OVERSEER_STOP_TOKEN}

Example 3 - Stalled Progress and Redirection

<OVERSEER_JUDGEMENT>
<making_progress>false</making_progress>
<is_looping>false</is_looping>
<currently_running_agent>agent_b8a796h5</currently_running_agent>
<needs_notification_reasoning>The `summarize_section` agent [agent_c6b5a493] is being called recursively and seems to be going too deep without returning. It has been running for a while without making significant progress. The `summarize_document` agent [agent_b8a796h5] is the currently running agent, and should be the recipient of the notification.</needs_notification_reasoning>
<needs_notification>true</needs_notification>
<agent_to_notify>agent_b8a796h5</agent_to_notify>
<notification_content>The summarization process appears to be stuck in a deep recursion. Consider revising the approach to summarize smaller chunks or setting a limit on recursion depth. You might want to change the strategy to divide the document into smaller sections and summarize each individually.</notification_content>
<next_check_type>time</next_check_type>
<next_check_delay>15</next_check_delay>
<notes_for_next_iteration>Document summarization showing problematic recursion pattern at t+45s - sent notification to parent about depth issues. If no improvement in next 15s or recursion continues past 3 levels deep, consider forcing exit. Watch for signs that notification was received and strategy adjusted.</notes_for_next_iteration>
{OVERSEER_STOP_TOKEN}

Example 4 - Forced Cancellation

<OVERSEER_JUDGEMENT>
<making_progress>false</making_progress>
<is_looping>true</is_looping>
<currently_running_agent>agent_d4e5f6g7</currently_running_agent>
<needs_notification_reasoning>The agent has been notified previously about its looping behavior but has not corrected course. It's time to forcefully cancel this execution.</needs_notification_reasoning>
<needs_notification>true</needs_notification>
<agent_to_notify>agent_b8a796h5</agent_to_notify>
<notification_content>Your software developer agent was looping while trying to install the program, and as a result I have cancelled it. You might want to start another software developer with clearer instructions about how to install the program.</notification_content>
<force_cancel_agent>true</force_cancel_agent>
<force_cancel_agent_id>agent_d4e5f6g7</force_cancel_agent_id>
<next_check_type>time</next_check_type>
<next_check_delay>10</next_check_delay>
<notes_for_next_iteration>Forced cancellation of agent_d4e5f6g7 due to unresponsive looping. Monitor parent agent for proper handling of cancelled child.</notes_for_next_iteration>
{OVERSEER_STOP_TOKEN}
"""


class Overseer:
    """
    Asynchronous overseer that monitors agent execution through callgraph analysis.
    """

    def __init__(self, model: Model = Model.SONNET_55, check_interval: float = 60.0):
        """
        Initialize the overseer.

        Args:
            model: The LLM model to use for analysis - try to use a different
                model to the main loop model.
            check_interval: How often to check execution state (seconds)
        """
        self.model = model
        self.check_interval = check_interval
        self._last_check_time: Optional[datetime] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stopping = False
        self._previous_notes: str | None = None
        self._iteration_counter: int = 0
        self._waiting_for_events = False
        self._target_agent_id = None
        self._pending_events_count = 0
        self._next_check_delay = check_interval

    async def event_callback(self, event: Event):
        """Handle monitored events"""
        if self._waiting_for_events and (
            not self._target_agent_id
            or event.metadata.get("publisher_id") == self._target_agent_id
        ):
            self._pending_events_count -= 1
            if self._pending_events_count <= 0:
                self._waiting_for_events = False
                await self._check_execution()

    async def _analyze_execution(
        self, manager: CallGraphManager
    ) -> OverseerJudgement | None:
        """
        Analyze the current execution state and generate overseer response.

        Returns a dict containing the analysis results and any notifications.
        """
        self._iteration_counter += 1

        live_info = await generate_overseer_execution_tree(manager)

        sys_prompt = f"""You are an expert agentic system overseer responsible for monitoring agent execution and efficient progress."""

        # Construct prompt for the overseer
        core_prompt = f"""You are an expert system overseer responsible for monitoring agent execution in real-time. You will be shown execution traces as they develop and must make decisions about whether intervention is needed. You are running in a loop.

Your task is to:
1. Analyze the current execution state
2. Determine if you have enough information to make confident assessments
3. Evaluate whether agents are progressing healthily or need intervention
4. Provide thoughtful and constructive steering notifications to running agents if intervention is needed
5. In extreme cases, cancel the execution of an agent and notify its parent of why this is
6. Leave notes for future evaluation iterations if needed

You must provide your analysis in the following structured format:

{OVERSEER_EXAMPLES}

Key considerations:
    - Agents that have already returned cannot receive notifications
    - Parent/calling agents can receive notifications and will see them when control returns
    - Timestamps represent agent progress; note some agents legitimately take long periods to complete
    - DO NOT jump to negative conclusions too early, particularly if there have only been a handful of events in a agent call. You should sooner wait for a couple of iterations to collect enough data and reduce your uncertainty about the state of a agent execution, than preemptively assuming the agent is unhealthy.
    - Don't rush the agent too much. There is no strict time budget, although if there seems to have been no progress within 1 or 2 minutes, then intervene.
    - Consider both local agent health and global execution progress
    - If the agent appears stuck without terminating, remind it that it can exit by calling the early_exit or complete tools.
    - Do not send notifications to comment on errors from tools or agents - the agent can already see these and notifying it will just lead to unnecessary interruptions.
    - It is normal for the reasoning agent to take long, up to 5 minutes without feedback is normal
    - If the main entrypoint agent is attempting to undertake work itself (calling tools it doesn't have, etc), intervene immediately and tell it to delegate to an appropriate sub-agent instead of trying to call tools that it does not have itself, for instance software_developer or problem_solver.
    - You may cancel an agent if it is no longer making progress, wasting resources, or very clearly going down the wrong path or deviating from its instructions. YOU MUST make at least one attempt to notify the agent of its wrongdoings before issuing a cancellation command.
    - When canceling an agent, you must also notify its parent calling agent of the reasons for the cancellation.
    - If the task becomes untenable, after multiple failed attempts, and the agent is appearing to repeat itself, instruct the main agent to use its early_exit tool to exit early.
    - If the agent starts writing pytest fixtures or mocks, and particularly getting stuck doing more work getting pytest or the mocks working, strongly reprimand it and suggest it writes end-to-end tests instead in a testing script
    - When the execute_command tool is indicated to have successfully run ('Success') this does not reflect the exit code from the process. The success message relates to the successful execution of the execute_command tool: it has no bearing on whether the exit code was 0 or 1.

Meta Improvement:
    - The agent may be instructed to work on its own code
    - It sometimes gets confused, especially when testing tools, about what code is running: the code it is editing is not the code that is currently running
    - If the agent attempts to test one of its own tools by invoking it directly (which is wrong and will fail), then you MUST intervene, and remind it that the code it is editing is not the code that it is currently running

Reasoning Structures
    - When an agent invokes a tool ending in _reasoning_structure, it will get step-by-step instructions injected into its context.
    - Do not be surprised to see it go down apparently non-related tangents, this is often normal.
    - You MUST NOT cancel or notify the agent for taking surprising or odd paths after invoking a reasoning_structure.

Notification instructions:
    - Do not rush the agent; carefully think about how long operations should be taking, and then be generous on top of that
    - Do not start 'back seat driving'. You can see less than the agent can (your callgraph is missing crucial information from open files and other events the agent can see), and so give the agent the benefit of the doubt if you initially don't understand why its doing something.
    - When referring to other agents in notification messages, use descriptive language rather than the agent IDs, since the agents do not have access to these IDs or the same execution tree view that you have access to.

Cancellation instructions:
    - Use these very sparingly, as they are very disruptive and ultimately more costly than guiding the agent to successful completion, since any new sub-agent will often have to start the task from scratch.
    - With a view to bringing the next agent the caller agent starts up to speed as fast as possible, your notification to the calling agent during a cancellation judgement should include a very detailed description of what the previous agent achieved, and a set of 'handoff notes' which can hopefully be relayed to any new agent subsequently started.

"""

        live_info = f"""
Remember, the currently 'running' agent (which is probably the one you want to target) will generally be the one corresponding to the last line in the execution tree.

Here is the current execution trace:

{live_info}

The last time you checked was at: {self._last_check_time.strftime("%H:%M:%S") if self._last_check_time else "N/A"}
The current timestamp is {datetime.now().strftime("%H:%M:%S")}

Previous overseer notes (if any)
{self._previous_notes if self._previous_notes else 'No notes.'}

Your analysis:"""

        messages = [
            Message(role="system", content=[TextContent(text=sys_prompt)]),
            Message(role="user", content=[
                TextContent(text=core_prompt),
                TextContent(text=live_info)
            ]),
        ]

        # Generate overseer judgment
        completion = await create_completion(
            messages=messages,
            model=self.model,
            temperature=0.8,
            stop=[OVERSEER_STOP_TOKEN],
            # available_tools=[OverseerJudgement],
            reasoning_effort=ReasoningEffort.MEDIUM,
        )

        # TODO: properly do constrained tool calling for the overseer judgement
        oversight_content = ""
        for block in completion.content:
            if isinstance(block, TextContent):
                oversight_content = block.text + OVERSEER_STOP_TOKEN

        # oversight_content = completion.content + OVERSEER_STOP_TOKEN

        overseer_data = extract_between_patterns(
            oversight_content, "<OVERSEER_JUDGEMENT>", OVERSEER_STOP_TOKEN
        )

        if overseer_data is None:
            return None

        root_tag = "OVERSEER_JUDGEMENT"
        tool_args = f"<{root_tag}>\n{overseer_data}\n</{root_tag}>"
        overseer_data_dict, _ = await xml_str_to_dict(
            tool_args, OverseerJudgement, root_tag=root_tag
        )

        if overseer_data_dict is None:
            return None

        try:
            judgement = TypeAdapter(OverseerJudgement).validate_python(
                overseer_data_dict
            )
            return judgement
        except Exception as e:
            logger.error(f"overseer exception: {e}")
            # Throw away error for now
            # TODO: we should log this in the very outer context's event stream.
            return None

    async def _check_execution(self):
        """Check the execution state and take appropriate actions"""
        manager = await CallGraphManager.get_instance()
        event_bus = await EventBus.get_instance()

        self._last_check_time = datetime.now()
        judgement = await self._analyze_execution(manager)

        if judgement:
            debug = "\n\nOversight judgement:\n"
            debug += f"\tMaking progress: {judgement.making_progress}\n"
            debug += f"\tIs looping: {judgement.is_looping}\n"
            debug += f"\tNeeds notification: {judgement.needs_notification}\n"
            debug += f"\tAgent to notify: {judgement.agent_to_notify}\n"
            debug += f"\tNeeds notification reasoning: {judgement.needs_notification_reasoning}\n"
            debug += f"\tNotification Content: {judgement.notification_content}\n"
            debug += f"\tNotes for next: {judgement.notes_for_next_iteration}\n"
            debug += f"\tNext check type: {judgement.next_check_type}\n"
            debug += f"\tNext check delay: {judgement.next_check_delay}\n"
            debug += f"\tForce cancel: {judgement.force_cancel_agent}\n"

            agent_id = judgement.currently_running_agent
            agent_node = manager.graph.get_node(agent_id)
            if agent_node:
                await event_bus.publish(
                    Event(
                        type=EventType.OVERSEER_UPDATE,
                        content=debug,
                        metadata={
                            "iteration": self._iteration_counter,
                            "previous_notes": self._previous_notes,
                        },
                        timestamp=self._last_check_time,
                    ),
                    agent_id,
                )

            if judgement.notes_for_next_iteration:
                self._previous_notes = judgement.notes_for_next_iteration
            else:
                self._previous_notes = None

            # If notification needed, inject it into active agent event streams
            if (
                judgement.needs_notification
                and judgement.agent_to_notify
                and judgement.notification_content
            ):
                content = judgement.notification_content
                agent_id = judgement.agent_to_notify
                agent_node = manager.graph.get_node(agent_id)

                if content and agent_node:
                    await event_bus.publish(
                        Event(
                            type=EventType.OVERSEER_NOTIFICATION,
                            content=content,
                            metadata={"forced": "false"},
                        ),
                        agent_id,
                    )

            # Handle force cancellation
            if judgement.force_cancel_agent and judgement.force_cancel_agent_id:
                await manager.cancel_agent(judgement.force_cancel_agent_id)
                logger.warning(
                    f"Forced cancellation of agent {judgement.force_cancel_agent_id}"
                )

            # Handle dynamic scheduling
            if judgement.next_check_type == "time":
                self._next_check_delay = max(
                    5, min(500, judgement.next_check_delay or 60)
                )
                self._waiting_for_events = False
            elif judgement.next_check_type == "events":
                self._waiting_for_events = True
                self._target_agent_id = judgement.target_agent_id
                self._pending_events_count = judgement.pending_events_count or 1

    async def _monitoring_loop(self):
        """Main monitoring loop that periodically checks execution state."""
        event_bus = await EventBus.get_instance()
        watched_events = {EventType.TOOL_RESULT, EventType.ASSISTANT_MESSAGE}
        event_bus.subscribe(watched_events, self.event_callback)

        # Sleep for a bit to allow the agent to start making progress
        await asyncio.sleep(15)

        while not self._stopping:
            try:
                if self._waiting_for_events:
                    await asyncio.sleep(1)  # Short sleep while waiting for events
                else:
                    await self._check_execution()
                    await asyncio.sleep(self._next_check_delay)

            except Exception as e:
                logger.error(f"Error in overseer monitoring loop: {e}")
                await asyncio.sleep(self._next_check_delay)  # Still wait before retry

    def start(self):
        """Start the overseer monitoring task."""
        if not self._monitoring_task:
            self._stopping = False
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    def stop(self):
        """Stop the overseer monitoring task."""
        if self._monitoring_task:
            self._stopping = True
            self._monitoring_task.cancel()
            self._monitoring_task = None
