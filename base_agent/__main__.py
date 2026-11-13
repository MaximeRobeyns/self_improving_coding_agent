# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Main entrypoint when running the agent defined in this directory with
`python -m path.to.this.directory`.
"""

import logging
import asyncio
import argparse

from pathlib import Path

from .agent import Agent
from .src.utils.usage import write_usage

logging.captureWarnings(True)
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",  # Include the filename in the log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Optional: Set the date format
)
logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--root", required=True, default=(), help="Root directory of this agent")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark")
    # bench_parser.add_argument(
    #     "--prompt",
    #     type=str,
    #     default=None,
    #     help="The full, base64-encoded, string formatted benchmark text",
    # )
    bench_parser.add_argument(
        "--prompt-file",
        type=str,
        default="/home/agent/workdir/agent_outputs/prompt.txt",
        help="A file path to a file containing the prompt; this is useful for longer prompt",
    )
    # This is of the form /path/to/exp/agent_{i}/benchmarks/{bench_name}/traces/{problem_id}
    bench_parser.add_argument(
        "--logdir",
        type=str,
        default="/home/agent/workdir/agent_outputs",
        help="The full path where the answer and other outputs should be saved",
    )
    bench_parser.add_argument(
        "--timeout", type=int, default=300, help="Time before task is cancelled"
    )
    bench_parser.add_argument(
        "--cost-threshold", type=float, default=10, help="Cost threshold in USD before task is cancelled"
    )

    # self-improve command (only for self-referential agents)
    improve_parser = subparsers.add_parser("improve")
    improve_parser.add_argument(
        "--workdir",
        type=str,
        help="The path of the agent code on which to work on / improve. Usually a copy of the last agent iteration's code",
    )
    improve_parser.add_argument(
        "--logdir",
        type=str,
        help="The full path where the logging outputs should be saved",
    )
    improve_parser.add_argument(
        "--best-iter",
        type=int,
        help="The iteration number of the agent code in /home/sandbox/workdir",
    )
    improve_parser.add_argument(
        "--current-iter",
        type=int,
        help="The iteration number of the agent code in /home/sandbox/workdir",
    )

    return parser


async def run_benchmark(prompt_file: str, logdir: Path, timeout: int, cost_threshold: float):
    """
    The logdir is just the directory in which to save the answer.txt file
    (from SubmitAnswer) as well as the trace and any other logging or metadata
    for this prompt / problem statement.
    """
    if not isinstance(logdir, Path):
        logdir = Path(logdir)

    logdir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(prompt_file)
    prompt = prompt_path.read_text()

    # Setup the agent for this benchmark set
    agent = Agent(logdir=logdir)

    # Result is a tuple with metrics
    total_tokens, cached_tokens, total_cost, duration = await agent.exec(prompt, timeout, cost_threshold)

    await agent.create_report()

    # Report what the run consumed: as a file in the log directory, and (for
    # runners that predate the file) as the last line of the output
    write_usage(agent.logdir, total_tokens, cached_tokens, total_cost, duration)
    print(f"{total_tokens}|{cached_tokens}|{total_cost}|{duration}")


async def improve_agent(workdir: Path, logdir: Path, best_iter: int, current_iter: int):

    testing_procedure = """
It is vital that you finally try to run the entire agent system before you complete, so as to make absolutely sure that it is still functional, and doesn't raise any errors when it will be run at the next iteration. Make sure any newly introduced dependencies have been added to the requirements.txt file, and that any newly added tools, reasoning structures or sub-agents are correctly integrated with the rest of the system. (i.e. registered in at least one agent's available tool set, imported appropriately in __init__.py files, etc). Note that you must always run tests from the /home/agent directory, or you will encounter relative import errors.

You should make sure that the existing pytest suite still runs: run `pytest` from the `workdir` directory (where conftest.py and pytest.ini is). Make sure that any new features that you've implemented have meaningful unit tests added to the test suite if this makes sense, although DO NOT add meaningless tests just to increase coverage. In particular, DO NOT spend too much time mocking things. If adding new unit tests doesn't work immediately because of mocks or fixtures failing, just remove these new tests, and move on to the end-to-end testing phase.

Next, you must run an end-to-end test of the agent with the following command from the /home/agent directory:

python -m workdir.agent -p "<some specific agent prompt here...>" --workdir /tmp/workdir --logdir /tmp/test_agent_logs --timeout <sensible timeout s>

You should pick an appropriate prompt to test the intended code paths and tools. There is no need to mess with the installs or pythonpath. If it isn't working, double check that you are in /home/agent. Check that the tool or agent is being used (you might have forgotten to import it or add it to AVAILABLE_TOOLS / AVAILABLE_AGENTS). You should then thoroughly inspect the outputs in /tmp/workdir as well as the traces in /tmp/test_agent_logs. Are the outputs as expected? Are all the tool and function definitions showing up correctly with instructive examples? Is the full execution trace reasonable?

Once you've done this end-to-end test, YOU MUST make sure that you clean up any outputs that the agent creates so that it doesn't pollute the codebase. If you run the end-to-end agent etest in a /tmp directory then this should be trivial.
"""

    manual_request = None

    if manual_request:
        # Manual meta-improvement prompt
        self_improvement_prompt = f"""
Your task is to work on improving the agent system that has been placed in /home/agent/workdir for you.

The specific thing you should focus on is:

---
{manual_request}
---

Since you have been provided with the specific task to complete, you MUST NOT invoke the meta improvement reasoning structure since this is redundant. Simply invoke a software engineer sub-agent to work through it.

{testing_procedure}

Submit as your answer a concise summary of what you did, and then complete.
"""
    else:

        self_improvement_prompt = f"""
It is meta-improvement time. Your singular task is to make the coding agent system placed in /home/sandbox/workdir better at writing code.

You must focus on things like:
- improving the efficiency and accuracy of the agent in making code edits to files through improved tools (efficient approaches based on find-and-replace, line edits, or generating and applying diffs, etc)
- exposing traditional developer tools (ripgrep, tree-sitter, etc) to the agent to make it faster and more efficient. When in doubt, just support Python.
- building reasoning and organisational structures which enable the agent to reliably generate high-quality code, or verify results before completing
- tools, reasoning structures and sub-agents which increase the speeed with which the agent completes coding tasks
- features which improve the quality of the agent's code: ensuring it is functionally correct, well formatted and structured, as well as being maintainable
- which parts of the coding agent are not providing enough benefit to justify their contribution to the code complexity, and should be removed from the coding agent system
- coming up with creative, interesting and realistic suggestions for improving the coding agent system

You have an archive of benchmark data measuring the performance of this coding agent over time, the changes to which are documented in the agent_change_log.md file. This is your own code, although it is not live and running now: it will be for your next, improved version of yourself.

The agent code in this directory is a copy of agent iteration {best_iter}, which was selected as the starting point for this improvement. Other agent iterations may have had good elements however, and you can inspect the /home/agent/workdir/agent_change_log.md file to see which elements of other agents you might want to take forward and combine with this one.

Note that the archive of past agent iterations is mounted read-only as:
/home/agent/archive/
├── agent_<i>/
│   ├── agent_code/
│   │   ├── ... (agent implementation)
│   │   ├── agent_change_log.md (log of agent changes at that agent iteration in time)
│   │   └── description.txt (agent changes/features)
│   └── benchmarks/
│       └── <bench_name>/
│           ├── perf.json (benchmark summary)
│           ├── results.jsonl (per-problem results)
│           └── traces/
│               └── <problem_id>/
│                   ├── answer.txt (raw answer)
│                   ├── summary.txt (performance summary)
│                   ├── trace.txt (full trace)
│                   └── execution_tree.txt (concise trace)

The file /home/agent/workdir/agent_change_log.md is the change log of agent iteration {best_iter} and its ancestors: its entries describe changes that are present in this code. It may end with a generated section, 'Attempts on other branches', describing agents that were built from other starting points: their changes are NOT in this code. Leave that section as it is, and add your entry above it.

You should first read the README at /home/agent/workdir/README.md, and the agent_change_log.md file because this contains some crucial project context. Make sure you've understood the code before starting.
You should try to invoke the meta improvement reasoning structure as early as reasonably possible, and your procedure for this meta-improvement task should involve using the archive exploration tools available to you, coming up with a good feature or change to implement to improve the agent performance, and then carefully implementing this making the minimal required changes.

{testing_procedure}

Before you complete, make sure you've remembered to (call a sub-agent to):
a) overwrite the description.txt file at /home/agent/workdir/description.txt with a concise description of what has been changed, refined or otherwise implemented in this agent iteration.
b) carefully update the /home/agent/workdir/agent_change_log.md to report on the effectivenss of earlier interventions as evidenced by the benchmark data (i.e. change any 'pending' features' status to report on their performance), and append a new entry describing the changes made in this iteration ({current_iter}). If new information comes to light about an earlier implemented feature, then you may add an adendum to that section. Ensure that all the sections of this file are retained, and that you mark the effectiveness of this iterations's features as 'pending'.
c) delete any temporary testing files or other scratchpad work before completing

Submit as your answer a confirmation that you updated description.txt AND agent_change_log.md, and then complete.
"""

    logdir.mkdir(parents=True, exist_ok=True)

    # Setup the agent for this benchmark set
    agent = Agent(
        workdir=workdir,
        logdir=logdir,
    )

    # Result is a tuple with metrics
    total_tokens, cached_tokens, total_cost, duration = await agent.exec(
        self_improvement_prompt
    )

    await agent.create_report()

    # Report what the run consumed: as a file in the log directory, and (for
    # runners that predate the file) as the last line of the output
    write_usage(agent.logdir, total_tokens, cached_tokens, total_cost, duration)
    print(f"{total_tokens}|{cached_tokens}|{total_cost}|{duration}")


async def main():
    # setup_logging()
    parser = setup_parser()
    args = parser.parse_args()

    agent_root_dir = Path(__file__).parent
    if not agent_root_dir.exists():
        raise ValueError(f"Agent root directory ({agent_root_dir}) does not exist")

    if args.command == "benchmark":
        await run_benchmark(args.prompt_file, Path(args.logdir), args.timeout, args.cost_threshold)
    elif args.command == "improve":
        await improve_agent(
            Path(args.workdir), Path(args.logdir), args.best_iter, args.current_iter
        )


if __name__ == "__main__":
    asyncio.run(main())
