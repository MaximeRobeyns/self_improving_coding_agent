# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Type
from collections import OrderedDict

from .base import BaseBenchmark
from .gpqa import GPQABenchmark
from .aime import AIMEBenchmark
from .drop import DROPBenchmark
from .math import MATHBenchmark
from .gsm8k import GSM8KBenchmark
from .gsm_ic import GSMICBenchmark
from .refute import RefuteBenchmark
from .arc_agi import ARCAGIBenchmark
from .humaneval import HumanEvalBenchmark
from .file_editing import FileEditingBenchmark
from .aiq_benchmark import AIQBenchmark
from .livecodebench import LiveCodeBenchmark
from .symbol_location import SymbolLocationBenchmark
from .swebench_verified import SWEBenchBenchmark
from .aiq_project_benchmarks import (
    LinalgAIQBenchmark,
    CSVParsingAIQBenchmark,
    MessagingAppAIQBenchmark,
    DistKVStoreAIQBenchmark,
)

# All available benchmarks, by name. Which of them an experiment runs is a
# runner setting (`benchmarks`, by default just gsm8k).
# Important, append new benchmarks to the end of this
benchmark_registry: OrderedDict[str, Type[BaseBenchmark]] = OrderedDict(
    [
        (GSM8KBenchmark.name, GSM8KBenchmark),
        (DROPBenchmark.name, DROPBenchmark),
        (ARCAGIBenchmark.name, ARCAGIBenchmark),
        (MATHBenchmark.name, MATHBenchmark),
        (GSMICBenchmark.name, GSMICBenchmark),
        (FileEditingBenchmark.name, FileEditingBenchmark),
        (SWEBenchBenchmark.name, SWEBenchBenchmark),
        (HumanEvalBenchmark.name, HumanEvalBenchmark),
        (AIMEBenchmark.name, AIMEBenchmark),
        (GPQABenchmark.name, GPQABenchmark),
        (LiveCodeBenchmark.name, LiveCodeBenchmark),
        (SymbolLocationBenchmark.name, SymbolLocationBenchmark),
        (RefuteBenchmark.name, RefuteBenchmark),
        (AIQBenchmark.name, AIQBenchmark),
        (LinalgAIQBenchmark.name, LinalgAIQBenchmark),
        (CSVParsingAIQBenchmark.name, CSVParsingAIQBenchmark),
        (MessagingAppAIQBenchmark.name, MessagingAppAIQBenchmark),
        (DistKVStoreAIQBenchmark.name, DistKVStoreAIQBenchmark),
    ]
)
