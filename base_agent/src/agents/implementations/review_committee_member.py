# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from pydantic import Field


from ..base_agent import BaseAgent
from ...config import settings
from ...tools.directory_tools import ViewDirectory
from ...tools.file_tools import OpenFile, CloseFile
from ...tools.ripgrep_tool import RipGrepTool
from ...utils.metrics import make_random_agent_metrics
from ...types.agent_types import AgentStatus, AgentResult
from ...types.llm_types import Model
from .reasoner import ReasoningAgent


class CommitteeMember(BaseAgent):
    """
    A simple review committee agent, with read-only access to the project.
    """

    AGENT_NAME = "meta_agent_design_reviewer"

    AGENT_DESCRIPTION = """A meta-agent design review committee member. Called from the committee_design tool."""

    SYSTEM_PROMPT = """You are a member of a Meta-Agent design review committee, tasked with evaluating a coding agent's design proposal, about how to improve a coding agent system, before it begins work on the implementation. Your role is to provide a detailed, constructive and reasonable critique that ensures the proposed design avoids commonly identified pathologies in coding agent design, and is robust, practical, and aligned with the goals of the self-improving coding agent.

Approach the review with a critical yet collaborative mindset, drawing on established engineering principles such as simplicity (delete unnecessary parts), conceptual integrity (a cohesive whole), and testability.

You must ensure that the design is grounded in making the coding agent system better at writing softare, advocating for things like
- improving the mechanics of writing the code files: more efficient file editing strategies and tools
- building reasoning and organisational structures which guide the agent to generate better code
- things which improve the speed with which the agent is able to complete code tasks
- features which improve the quality of the written code: such as improving the generated code's formatting and structure, utillities for robust and efficient testing, or which enhance the maintainability of the code

Focus on the following desiderata:
- Clarity: Is the proposal understandable and well-articulated?
- Feasibility: Can it be realistically implemented given constraints?
- Robustness: Does it handle real-world challenges (e.g., edge cases, failures)?
- Quality: Does it reflect good design and testing practices for long-term value?
- Grounding: Is it supported by executable feedback (e.g., tests) to verify its claims?

Provide a structured evaluation: identify strengths, flag weaknesses, and suggest actionable improvements. Avoid vague or frivolous feedback-every critique should tie back to the project's success. Your specialized role will guide your focus, but always consider the proposal as a whole."""

    # Available tools
    # NOTE: ExitAgent and ReturnResult are automatically included
    # We limit ourselves to 'read only' tools.
    AVAILABLE_TOOLS = {
        ViewDirectory,
        OpenFile,
        CloseFile,
        RipGrepTool,
    }

    # Available agents
    # AVAILABLE_AGENTS = {ReasoningAgent}
    AVAILABLE_AGENTS = set()

    HAS_FILEVIEW = True

    MODEL = settings.MODEL
    TEMPERATURE = 0.666

    # Agent parameters
    proposal: str = Field(
        ...,
        description="The full proposal to review",
    )
    context: str = Field(
        ...,
        description="The motivation and context for understanding the plan",
    )
    specialisation: str = Field(
        ..., description="The specialisation of this committee member"
    )
    model: Model = Field(default_factory=lambda: settings.MODEL)

    def __init__(
        self,
        parent: BaseAgent | None = None,
        workdir: Path | None = None,
        logdir: Path | None = None,
        debug_mode: bool = False,
        **data,
    ):
        super().__init__(
            parent=parent, workdir=workdir, logdir=logdir, debug_mode=debug_mode, **data
        )

    async def construct_core_prompt(self) -> str:
        """Construct the core prompt for the committee member."""

        prompt = f"""{self.specialisation}

Here is the agent's self-provided goals and context surrounding the plan
<goals_and_context>
{self.context}
</goals_and_context>

Here is the design proposal you have been asked to review:

<proposal>
{self.proposal}
</proposal>

You should read the README.md file first to get the full context of this self-improving coding agent project.
You should then view the agent_change_log.md to get an idea of what (if anything) has already been tried by the coding agent as it attempts to improve itself, as measured by the benchmark performance.
You can also quickly view any other code files that you need to get context on the proposal.

Then, craft your review. Don't spend too long opening other files and doing research. Move swiftly. Note that you MUST provide your full review in the return_result tool since this is how it is communicated back. Anything not put in the return_result tool will not be seen by the agent.

DO NOT attempt the task yourself, and avoid calling tools unless you absolutely need to. Then, simply provide your review in the return_result tool and complete.
"""

        return prompt

    @classmethod
    def generate_examples(cls) -> list[tuple["CommitteeMember", AgentResult]]:
        """Generate example uses of the tool with their expected outputs.

        Note that the committee member is deterministically invoked (for now)
        so these examples won't be used.
        """
        examples = []
        return examples
