# Self-Improving Coding Agent runner
"""
An agent runner that handles benchmark evaluation and self-improvement cycles.

    python runner.py --help

The implementation is in the `orchestrator` package:

    cli         the command line and its settings
    loop        the self-improvement loop
    evaluation  running benchmark problems for an agent
    creation    making a new agent, and validating it
    state       the experiment directory and its metadata
    containers  docker
"""

import asyncio
import logging

from orchestrator.cli import main

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    asyncio.run(main())
