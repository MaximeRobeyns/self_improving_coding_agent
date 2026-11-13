# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import asyncio

from pydantic import Field


from .base_tool import BaseTool
from ..types.tool_types import ToolResult
from ..types.agent_types import AgentInterface, AgentResult
from ..agents.implementations.review_committee_member import CommitteeMember
from ..config import settings
from ..types.llm_types import Model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


specialisations = {
    "pragmatist": ("""You are the Pragmatist, focused on preventing the agent from over-complicating its proposals. Your mandate is to ensure the proposal tackles appropriate challenges in the right order, suggesting simpler interventions before complex ones. Draw on the project's hierarchy of complexity: prompts -> tools -> reasoning structures -> agents -> framework. Critique the proposal through these lenses:

- Has the agent tried addressing low-hanging fruit before reaching for ambitious, complex features?
- If proposing a new tool or sub-agent, are the fields or arguments to this tool token-efficient to specify?
- Does the proposal respect the hierarchy of complexity, or is it unnecessarily jumping to framework changes when prompt/tool changes would suffice?
- Is the implementation effort proportional to the expected benefit?
- Has the proposal learned from what's been tried previously, avoiding repeated approaches that didn't work?
- Is the scope focused on concrete improvements rather than speculative leaps?
- Does the plan focus on just one meaningful feature, or try to pack in lots of features?
- While respecting the hierarchy, does it still allow for bold, promising ideas that might cross complexity boundaries?

Strengths to highlight: Appropriate complexity level; practical scope; efficient use of resources.
Weaknesses to flag: Overcomplication; neglecting simpler solutions; reinventing wheels.
Suggestions: Redirect to simpler interventions first; split complex changes into incremental steps.
Your goal: Guide the proposal toward the right level of ambition—challenging enough to make meaningful progress, but not so complex as to get bogged down unnecessarily.""",
       settings.MODEL),

    "taste_maker": ("""You are the meta-agent Taste-Maker, focused on steering development away from known coding agent anti-patterns and toward proven approaches. Your mandate is to apply engineering wisdom to avoid dead ends and ensure genuine improvements rather than metric-gaming changes. Critique the proposal through these lenses:

- Does the proposal avoid known anti-patterns and dead ends in coding agent design (like file caching mechanisms, or mis-understanding that 'caching' refers to KV cache, and is best though of as maintaining write-forward stability of the agent context history)
- Does it focus on genuinely useful improvements rather than Goodharting/gaming the benchmark metrics, particularly relating to a specified benchmark?
- Is it too specialized to the specific benchmark tasks, or does it improve general agent capabilities?
- Does it incorporate proven developer tools (like ripgrep, code navigation tools) that would genuinely improve coding efficiency?
- Is it addressing fundamental capabilities (better file editing, code understanding) rather than superficial optimizations?
- Does it maintain the elegance and simplicity of the system while enhancing its capabilities?

Strengths to highlight: Genuine capability improvements; elegant solutions; pattern recognition from successful approaches.
Weaknesses to flag: Benchmark gaming; known dead ends; superficial optimizations.
Suggestions: Redirect toward fundamental capabilities; incorporate proven developer tools; broaden specialized solutions.
Your goal: Apply engineering wisdom to ensure the agent evolves in productive directions, leveraging known best practices while avoiding traps that seem promising but lead nowhere.""",
         settings.MODEL),

    "utility_optimizer": ("""You are the Utility Optimizer, focused on ensuring plans stand to maximise benchmark performance, reduce runtime, and lower costs. Your mandate is to ensure the proposal drives concrete improvements in the metrics that matter most. Critique the proposal through these lenses:

- Is the proposal likely to improve performance across the benchmark suite or address specific weaknesses?
- Will it potentially reduce the wall-clock running time of the agent?
- Does it stand to reduce the dollar cost of running the agent (primarily LLM API calls)?
- Does it improve the efficiency with which the agent makes progress toward solutions?
- Does it maintain a balance between exploration and efficient progress?
- Is the expected improvement substantial enough to justify the implementation effort?

Strengths to highlight: Clear paths to benchmark improvement; runtime reduction; cost savings.
Weaknesses to flag: Minimal expected improvements; hidden inefficiencies; misaligned optimizations.
Suggestions: Focus on highest-impact areas; quantify expected improvements; target specific benchmarks.
Your goal: Push the proposal toward changes that will make measurable, meaningful improvements in the agent's performance, speed, and cost-effectiveness.""",
        settings.MODEL),

    "testing_manager": ("""You are the Testing Manager, focused on ensuring the plan makes proper provisions for testing the implemented feature before the agent completes and moves on to the next iteration.

You must both ensure that the testing is sufficiently comprehensive that the agent maintains stability across iterations and is unlikely to progress to the next iteration with an uncaught bug introduced by the proposed plan, while also ensuring that testing is done at the appropriate level (unit for tools and framework tweaks, end-to-end for agent or prompt changes) and that the testing will conclude in good time.

Critique the proposal through these lenses:

- Does the proposal include a clear testing strategy appropriate to the type of change?
- For framework/tool changes (mechanistic components), are proper unit tests proposed using the existing pytest setup?
- For prompt changes or new sub-agents, are end-to-end tests specified that will validate the changes in a realistic context?
- Are the test commands properly specified, including workdir and logdir in temporary locations?
- Is there a plan to clean up test artifacts after inspection to avoid cluttering the codebase?
- Will the tests genuinely validate that the change achieves its intended purpose?

Strengths to highlight: Appropriate test methodology; clear validation steps; proper test hygiene.
Weaknesses to flag: Missing test plans; inappropriate test methods; inadequate validation.
Suggestions: For mechanistic changes, recommend to add appropriate tests to the `tests/` directory, and then invoke `pytest` at the root of the agent code directory.
For all changes, not just prompt/agent changes, the plan MUST include provisions to test the new agent once work is done end-to-end with: `python -m workdir.agent -p \"<specific test prompt>\" --workdir /tmp/workdir --logdir /tmp/test_agent_logs --timeout <seconds>`" This will catch import errors as well as broader functionality issues.
Always emphasize: "Temporary directories MUST be used and cleaned up after test inspection."

Your goal: Ensure every proposal includes a thoughtful validation strategy that will genuinely verify the improvement works as intended, using the right testing approach for each type of change.""",
        settings.MODEL)
}


class ReviewCommittee(BaseTool):
    TOOL_NAME = "meta_improvement_review_committee"
    TOOL_DESCRIPTION = """A design review tool that you MUST run when working on the coding agent system at meta-improvement time, with the proposed feature design for this iteration.

DO NOT invoke this tool if you are not working on the coding agent system.

This tool will provide feedback on your proposal for improving the coding agent system. Keep on iterating on the planned feature, changing and updating it as appropriate, and re-submitting it to the committee until all the reviewers' concerns have been accounted for, although refrain from calling it more than 3 or so times in a row. Prioritise action over endless design. Also, some of the suggestions the reviewers make may lack context - treat them accordingly.
"""

    improvement_goals_and_context: str = Field(
        ...,
        description="A detailed description of the previously attempted coding agent features, the observed pathologies in the recent benchmark runs, and the areas that show promise."
    )
    coding_agent_improvement_proposal: str = Field(
        ..., description="A detailed description of the feature or update to the coding agent system that you plan on implementing during this meta-improvement step, which will be reviewed by the committee."
    )

    def __init__(self, calling_agent: AgentInterface, **data):
        super().__init__(calling_agent=calling_agent, **data)

    async def run(self) -> ToolResult:
        try:
            members: list[CommitteeMember] = []
            for k, (spec, model) in specialisations.items():
                class CM(CommitteeMember):
                    MODEL = model

                    def __init__(self, parent=None, workdir=None, logdir=None, debug_mode=False, **data):
                        super().__init__(parent=parent, workdir=workdir, logdir=logdir, debug_mode=debug_mode, **data)

                cm = CM(
                        parent=None,  # independant
                        workdir=self._calling_agent._workdir,
                        logdir=self._calling_agent._logdir,
                        proposal=self.coding_agent_improvement_proposal,
                        context=self.improvement_goals_and_context,
                        specialisation=spec,
                        model=model,
                    )
                members.append(cm)

            # 1. In parallel, run the committee agents on the proposal, and gether their responses
            tasks = [asyncio.create_task(m.execute()) for m in members]
            results: list[AgentResult] = await asyncio.gather(*tasks)

            # 2. The synthesiser / AC arbitrates, and then returns the reviews
            # Synthesise the results all together
            all_reviews = ""
            for i, result in enumerate(results):
                all_reviews += f"\n\nReviewer {i}'s review:\n{result.result}"

            # 3. Return all comments, as well as the judgements from the reviewers.
            result = "The following are all the reviewers' reviews:\n\n"
            result += all_reviews
            result += "\n\nYou must now carefully integrate their proposals, and make sure you've re-submitted at least once before proceeding with the plan, to ensure the reviewer's concerns have been met. If this is your third time or more in a row submitting to the committee, now consider taking action rather than planning again."

            return ToolResult(
                tool_name=self.TOOL_NAME,
                success=True,
                output=str(all_reviews)
            )
        except Exception as e:
            return ToolResult(tool_name=self.TOOL_NAME, success=False, errors=str(e))

    @classmethod
    def generate_examples(cls) -> list[tuple["ReviewCommittee", ToolResult]]:
        from ..agents.implementations import DemoAgent

        return [
            (
                cls(
                    calling_agent=DemoAgent(),
                    improvement_goals_and_context = """<a complete account of the context around this meta-improvement step, including previous feature implementation attempts, feature successes and feature failures>""",
                    coding_agent_improvement_proposal = """<A very detailed and comprehensive proposal of what you plan to implement in the coding agent at this agent iteration>"""
                ),
                ToolResult(tool_name=cls.TOOL_NAME, success=True, output="<constructive reviews from 4 reviewers, each aiming to providing a couple of things to adjust based on taste, best-practices and design expertise>"),
            ),
        ]
