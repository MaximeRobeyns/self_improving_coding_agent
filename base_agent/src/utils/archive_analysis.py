# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, Optional, Union, List, Any, Literal, Tuple
from dataclasses import dataclass
from tabulate import tabulate

from .lineage import STATUS_OK, read_lineage

logger = logging.getLogger(__name__)

ScoreType = Literal["mean_score", "utility_score"]


def result_id(problem_id: str, repeat: int = 0) -> str:
    """The id under which a run of a problem is stored. The first run keeps
    the problem id itself, so that single runs look as they always did."""
    return problem_id if repeat == 0 else f"{problem_id}#{repeat}"


def base_problem_id(stored_id: str, repeat: int = 0) -> str:
    """The id of the problem that a stored result is a run of."""
    suffix = f"#{repeat}"
    if repeat and stored_id.endswith(suffix):
        return stored_id[: -len(suffix)]
    return stored_id


def per_problem_scores(scores_df: pd.DataFrame, score_key: str = "score") -> pd.DataFrame:
    """One row per (iteration, benchmark, problem), averaging repeated runs.

    Repeated runs of a problem are not independent observations of an agent's
    score, so statistics over problems must be computed on these means.
    """
    keys = ["iteration", "benchmark", "problem_id"]
    if scores_df.empty or not set(keys) <= set(scores_df.columns):
        return scores_df
    return scores_df.groupby(keys, sort=False)[score_key].mean().reset_index()


def read_jsonl(file_path: Path) -> List[Dict]:
    """Read a JSONL file and return list of records."""
    records = []
    with open(file_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def read_perf_json(file_path: Path) -> Dict:
    """Read perf.json file."""
    with open(file_path) as f:
        return json.load(f)


@dataclass
class ProblemResult:
    """Represents the results for a single problem within a benchmark."""

    problem_id: str
    score: float
    tokens_used: int
    num_cached_tokens: int
    cost_estimate: float
    wall_time: float
    timed_out: bool = False
    summary: Optional[str] = None
    trace: Optional[str] = None


@dataclass
class BenchmarkScore:
    """Represents the performance metrics for a single benchmark."""

    avg_score: float
    tokens: int
    num_cached_tokens: int
    cost: float
    time: float
    timed_out: bool = False
    problem_results: Optional[Dict[str, ProblemResult]] = None

    def utility(
        self,
        cost_limit: float = 3.0,  # Maximum allowed cost in dollars
        time_limit: float = 5 * 60,  # Maximum allowed time in seconds
    ) -> float:
        """Calculate utility score, based on performance, cost and time.

        Args:
            cost_limit: Maximum allowed cost in dollars
            time_limit: Maximum allowed time in seconds

        Returns:
            Combined utility score from 0 to 1
        """

        # Check if task was successful and within resource limits
        if (self.avg_score > 0.0 and  # Any positive score counts as partial success
            self.cost <= cost_limit and
            self.time <= time_limit and
            not self.timed_out):
            return self.avg_score  # Return the actual score for partial credit
        else:
            return 0.0  # Failure if any constraint is violated


@dataclass
class AgentIteration:
    """Represents a single agent iteration with its benchmark scores."""

    iteration: int
    benchmarks: Dict[str, BenchmarkScore]
    description: Optional[str] = None


class ArchiveAnalyzer:
    """Analyzes the performance archive across multiple agent iterations."""

    def __init__(self, archive_root: Union[str, Path]):
        self.archive_root = Path(archive_root)

    def _read_problem_trace(self, trace_dir: Path) -> tuple[str | None, str | None]:
        """Read the trace and summary from a problem trace directory."""
        try:
            trace = (
                (trace_dir / "execution_tree.txt").read_text()
                if (trace_dir / "execution_tree.txt").exists()
                else None
            )
            summary = (
                (trace_dir / "summary.txt").read_text()
                if (trace_dir / "summary.txt").exists()
                else None
            )
            return trace, summary
        except Exception as e:
            logger.debug(f"Error reading trace files in {trace_dir}: {str(e)}")
            return None, None

    def _parse_results_file(self, results_path: Path) -> Dict[str, ProblemResult]:
        """Parse a results.jsonl file into ProblemResult objects."""
        problem_results = {}
        try:
            with results_path.open() as f:
                for line in f:
                    data = json.loads(line)
                    problem_id = data["problem_id"]

                    # Skip entries with null scores
                    if data["score"] is None:
                        continue

                    problem_results[problem_id] = ProblemResult(
                        problem_id=problem_id,
                        score=data["score"],
                        tokens_used=data["tokens_used"],
                        num_cached_tokens=data["num_cached_tokens"],
                        cost_estimate=data["cost_estimate"],
                        wall_time=data["wall_time"],
                        timed_out=data.get("timed_out", False)
                    )
        except Exception as e:
            logger.debug(f"Error parsing results file {results_path}: {str(e)}")

        return problem_results

    def _parse_perf_file(
        self, perf_path: Path, results_path: Optional[Path] = None
    ) -> Optional[BenchmarkScore]:
        """Parse a perf.json file into a BenchmarkScore object."""
        try:
            with perf_path.open() as f:
                data = json.load(f)
                problem_results = None
                if results_path and results_path.exists():
                    problem_results = self._parse_results_file(results_path)

                return BenchmarkScore(
                    avg_score=data["avg_score"],
                    tokens=data["tokens"],
                    num_cached_tokens=data["num_cached_tokens"],
                    cost=data["cost"],
                    time=data["time"],
                    problem_results=problem_results,
                )
        except Exception as e:
            logger.debug(f"Error parsing {perf_path}: {str(e)}")
            return None

    def _get_agent_description(self, agent_dir: Path) -> Optional[str]:
        """Get the agent description from agent_code/description.txt if it exists."""
        desc_path = agent_dir / "agent_code" / "description.txt"
        try:
            return desc_path.read_text() if desc_path.exists() else None
        except Exception as e:
            logger.debug(f"Error reading description file: {str(e)}")
            return None

    def _get_agent_benchmarks(self, agent_dir: Path) -> Dict[str, BenchmarkScore]:
        """Get all benchmark scores for a single agent iteration."""
        benchmarks = {}
        benchmark_dir = agent_dir / "benchmarks"

        if benchmark_dir.exists():
            for bench_dir in benchmark_dir.iterdir():
                if bench_dir.is_dir():
                    perf_path = bench_dir / "perf.json"
                    results_path = bench_dir / "results.jsonl"
                    score = self._parse_perf_file(perf_path, results_path)
                    if score:
                        # Add trace information to problem results
                        if score.problem_results:
                            traces_dir = bench_dir / "traces"
                            if traces_dir.exists():
                                for problem_id, result in score.problem_results.items():
                                    problem_dir = traces_dir / problem_id
                                    trace, summary = self._read_problem_trace(
                                        problem_dir
                                    )
                                    result.trace = trace
                                    result.summary = summary

                        benchmarks[bench_dir.name] = score

        return benchmarks

    def _collect_iterations(self) -> List[AgentIteration]:
        """Collect all agent iterations from the archive."""
        run_dir = self.archive_root
        if not run_dir.exists():
            raise ValueError(f"Run directory not found: {run_dir}")

        iterations: List[AgentIteration] = []

        for agent_dir in run_dir.glob("agent_*"):
            try:
                agent_num = int(agent_dir.name.split("_")[1])
                benchmarks = self._get_agent_benchmarks(agent_dir)
                description = self._get_agent_description(agent_dir)
                if benchmarks:
                    iterations.append(
                        AgentIteration(agent_num, benchmarks, description)
                    )
            except ValueError as e:
                logger.error(f"Error parsing agent directory {agent_dir}: {str(e)}")
                continue

        # Sort by iteration number
        iterations.sort(key=lambda x: x.iteration)
        return iterations

    def analyze_run(self) -> pd.DataFrame:
        """Get performance metrics for all iterations."""
        iterations = self._collect_iterations()

        data = []
        for iter_data in iterations:
            row: dict[str, Any] = {"iteration": iter_data.iteration}

            total_tokens = total_cached = total_cost = total_time = 0
            total_problems = 1
            benchmark_scores = []

            for bench_name, score in iter_data.benchmarks.items():
                row[f"{bench_name}_score"] = score.avg_score

                # Get number of actual problems run in this benchmark
                num_problems = (len(score.problem_results)
                              if score.problem_results is not None
                              else 1)  # Fallback to 1 if no problem results
                num_problems = max(1, num_problems)

                # Calculate per-problem averages
                row[f"{bench_name}_avg_tokens"] = score.tokens / num_problems
                row[f"{bench_name}_avg_cached"] = score.num_cached_tokens / num_problems
                row[f"{bench_name}_avg_cost"] = score.cost / num_problems
                row[f"{bench_name}_avg_time"] = score.time / num_problems

                # Add to totals for overall averages
                total_tokens += score.tokens
                total_cached += score.num_cached_tokens
                total_cost += score.cost
                total_time += score.time

                # Get total number of problems across all benchmarks
                total_problems = sum(
                    len(b.problem_results) if b.problem_results is not None else 1
                    for b in iter_data.benchmarks.values()
                )
                total_problems = max(1, total_problems)

                benchmark_scores.append(score.utility())

            # Calculate overall averages per problem
            row.update({
                "avg_tokens": total_tokens / total_problems,
                "avg_cached": total_cached / total_problems,
                "avg_cost": total_cost / total_problems,
                "avg_time": total_time / total_problems,
                "utility_score": sum(benchmark_scores) / len(benchmark_scores),
            })

            data.append(row)

        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index("iteration", inplace=True, drop=False)
        return df

    def get_lineage(self) -> Dict[int, Dict[str, Any]]:
        """The lineage record of every agent in the archive, by iteration.

        Each record has at least `id`, `parent_id` (the agent whose code this
        one started from), `meta_agent_id` (the agent that made the changes),
        `status` and `generation`. Agents without a stored record (the root
        agent of older archives, say) get one with unknown ancestry.
        """
        lineage: Dict[int, Dict[str, Any]] = {}
        for agent_dir in self.archive_root.glob("agent_*"):
            if not agent_dir.is_dir():
                continue
            try:
                iteration = int(agent_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            record: Dict[str, Any] = {
                "id": iteration,
                "parent_id": None,
                "meta_agent_id": None,
                "status": STATUS_OK,
                "generation": 0 if iteration == 0 else None,
            }
            record.update(read_lineage(agent_dir) or {})
            lineage[iteration] = record
        return dict(sorted(lineage.items()))

    def get_children(self, iteration: int) -> List[int]:
        """Iterations whose code started from the given iteration."""
        return [
            i
            for i, record in self.get_lineage().items()
            if record["parent_id"] == iteration
        ]

    def get_ancestors(self, iteration: int) -> List[int]:
        """The chain of parents of an iteration, nearest first, up to the root."""
        lineage = self.get_lineage()
        ancestors: List[int] = []
        seen = {iteration}
        current = lineage.get(iteration, {}).get("parent_id")
        while current is not None and current in lineage and current not in seen:
            ancestors.append(current)
            seen.add(current)
            current = lineage[current]["parent_id"]
        return ancestors

    def paired_delta(
        self,
        iteration: int,
        baseline: int,
        score_type: ScoreType = "mean_score",
    ) -> Optional["PairedDelta"]:
        """How much better `iteration` scores than `baseline` on the problems
        both were run on, with a confidence interval. See `paired_delta`."""
        scores_df, _ = self.get_problem_scores_by_iteration()
        return paired_delta(scores_df, iteration, baseline, score_type=score_type)

    def get_best_agent_iteration(self, exclude: set[int] | None = None) -> int:
        """The iteration with the best mean benchmark score, ignoring any
        iterations in `exclude`."""
        df = self.analyze_run()
        if exclude:
            df = df[~df["iteration"].isin(exclude)].copy()

        # Get all benchmark score columns (excluding utility_score)
        score_cols = [
            col
            for col in df.columns
            if col.endswith("_score") and col != "utility_score"
        ]

        # Calculate mean performance across all benchmarks for each iteration
        df["mean_benchmark_score"] = df[score_cols].mean(axis=1)

        # Find the best performing iteration
        best_iteration = df.loc[df["mean_benchmark_score"].idxmax()]

        return int(best_iteration["iteration"])

    def analyze_run_formatted(self) -> str:
        """Generate formatted string summary of agent performance."""
        df = self.analyze_run()
        if df.empty:
            return "No data available"

        # Extract score columns and resource columns
        score_cols = [
            col
            for col in df.columns
            if col.endswith("_score") and col != "utility_score"
        ]

        # Create summary table
        summary_data = []
        lineage = self.get_lineage()
        scores_df, _ = self.get_problem_scores_by_iteration()
        headers = (
            ["Iteration", "Parent", "vs Parent"]
            + [col.replace("_score", "") for col in score_cols]
            + ["Utility", "Avg Cost", "Avg Time", "Avg Tokens", "Avg Cached Tokens"]
        )

        for _, row in df.iterrows():
            iteration = int(row["iteration"])
            parent_id = lineage.get(iteration, {}).get("parent_id")
            delta = (
                None
                if parent_id is None
                else paired_delta(scores_df, iteration, parent_id)
            )
            summary_row = [
                iteration,
                "-" if parent_id is None else parent_id,
                "-" if delta is None else str(delta),
            ]
            summary_row.extend([f"{row[col]:.3f}" for col in score_cols])
            summary_row.extend(
                [
                    f"{row['utility_score']:.3f}",
                    f"${row['avg_cost']:.2f}",
                    f"{row['avg_time']:.1f}s",
                    f"{row['avg_tokens']:.1f}",
                    f"{row['avg_cached']:.1f}",
                ]
            )
            summary_data.append(summary_row)

        table = tabulate(
            summary_data,
            headers=headers,
            tablefmt="pipe",
            floatfmt=".3f",
            disable_numparse=True,
        )
        return f"{table}\n\n{PAIRED_DELTA_LEGEND.format(column='vs Parent')}"

    def get_agent_description(self, iteration: int) -> str:
        """Get formatted description for a specific agent iteration."""
        agent_dir = self.archive_root / f"agent_{iteration}"
        description = self._get_agent_description(agent_dir)
        return f"Agent {iteration} Description:\n{description or 'No description available'}"

    def get_worst_performing_problems(
        self, iteration: int, n: int = 5, benchmark: Optional[str] = None
    ) -> pd.DataFrame:
        """Identify the n worst performing problems across all benchmarks or a specific benchmark for a specific iteration.

        Args:
            iteration: The agent iteration number to analyze
            n: Number of worst performing problems to return
            benchmark: Optional benchmark name to filter by. If provided, only problems from this benchmark are considered.
                      If None, problems across all benchmarks are considered.

        Returns:
            DataFrame containing the n worst performing problems, sorted by score ascending
        """
        agent_dir = self.archive_root / f"agent_{iteration}"
        if not agent_dir.exists():
            raise ValueError(f"Agent directory not found: {agent_dir}")

        problem_data = []
        benchmarks = self._get_agent_benchmarks(agent_dir)

        for bench_name, score in benchmarks.items():
            # Skip if benchmark filter is provided and doesn't match
            if benchmark and bench_name != benchmark:
                continue

            if score.problem_results:
                for problem_id, result in score.problem_results.items():
                    problem_data.append(
                        {
                            "benchmark": bench_name,
                            "problem_id": problem_id,
                            "score": result.score,
                            "tokens": result.tokens_used,
                            "num_cached_tokens": result.num_cached_tokens,
                            "cost": result.cost_estimate,
                            "time": result.wall_time,
                            "timed_out": result.timed_out,
                            "summary": result.summary or "No summary available",
                            "trace": result.trace or "No trace available",
                        }
                    )

        df = pd.DataFrame(problem_data)
        if df.empty:
            if benchmark:
                raise ValueError(
                    f"No problem data found for benchmark '{benchmark}' in iteration {iteration}"
                )
            else:
                raise ValueError(f"No problem data found for iteration {iteration}")

        return df.nsmallest(n, "score")

    def get_worst_performing_problems_formatted(
        self, iteration: int, n: int = 5, benchmark: Optional[str] = None
    ) -> str:
        """Get formatted summary of worst performing problems.

        Args:
            iteration: The agent iteration number to analyze
            n: Number of worst performing problems to return
            benchmark: Optional benchmark name to filter by. If provided, only problems from this benchmark are considered.
                      If None, problems across all benchmarks are considered.

        Returns:
            Formatted string containing the analysis of the worst performing problems
        """
        try:
            df = self.get_worst_performing_problems(iteration, n, benchmark)
        except ValueError as e:
            return str(e)

        benchmark_scope = f"benchmark '{benchmark}'" if benchmark else "all benchmarks"
        output = [
            f"Worst {n} performing problems for iteration {iteration} from {benchmark_scope}:"
        ]

        for _, row in df.iterrows():
            timeout_status = " (TIMED OUT)" if row['timed_out'] else ""
            output.extend(
                [
                    f"\nBenchmark: {row['benchmark']} - Problem {row['problem_id']}{timeout_status}",
                    f"Score: {row['score']:.3f}",
                    f"Resources: {row['tokens']} tokens (of which cached {row['num_cached_tokens']}), ${row['cost']:.3f}, {row['time']:.1f}s",
                    f"Execution Tree:\n{row['trace']}",
                    f"Summary:\n{row['summary']}",
                ]
            )

        return "\n".join(output)

    def get_best_performing_problems(
        self, iteration: int, n: int = 5, benchmark: Optional[str] = None
    ) -> pd.DataFrame:
        """Identify the n best performing problems across all benchmarks or a specific benchmark for a specific iteration.

        Args:
            iteration: The agent iteration number to analyze
            n: Number of best performing problems to return
            benchmark: Optional benchmark name to filter by. If provided, only problems from this benchmark are considered.
                      If None, problems across all benchmarks are considered.

        Returns:
            DataFrame containing the n best performing problems, sorted by score descending
        """
        agent_dir = self.archive_root / f"agent_{iteration}"
        if not agent_dir.exists():
            raise ValueError(f"Agent directory not found: {agent_dir}")

        problem_data = []
        benchmarks = self._get_agent_benchmarks(agent_dir)

        # If benchmark specified, validate it exists
        if benchmark and benchmark not in benchmarks:
            raise ValueError(
                f"Benchmark '{benchmark}' not found in iteration {iteration}"
            )

        for bench_name, score in benchmarks.items():
            # Skip if benchmark filter is provided and doesn't match
            if benchmark and bench_name != benchmark:
                continue

            if score.problem_results:
                for problem_id, result in score.problem_results.items():
                    problem_data.append(
                        {
                            "benchmark": bench_name,
                            "problem_id": problem_id,
                            "score": result.score,
                            "tokens": result.tokens_used,
                            "num_cached_tokens": result.num_cached_tokens,
                            "cost": result.cost_estimate,
                            "time": result.wall_time,
                            "timed_out": result.timed_out,
                            "summary": result.summary or "No summary available",
                            "trace": result.trace or "No trace available",
                        }
                    )

        df = pd.DataFrame(problem_data)
        if df.empty:
            if benchmark:
                raise ValueError(
                    f"No problem data found for benchmark '{benchmark}' in iteration {iteration}"
                )
            else:
                raise ValueError(f"No problem data found for iteration {iteration}")

        return df.nlargest(n, "score")

    def get_best_performing_problems_formatted(
        self, iteration: int, n: int = 5, benchmark: Optional[str] = None
    ) -> str:
        """Get formatted summary of best performing problems.

        Args:
            iteration: The agent iteration number to analyze
            n: Number of best performing problems to return
            benchmark: Optional benchmark name to filter by. If provided, only problems from this benchmark are considered.
                      If None, problems across all benchmarks are considered.

        Returns:
            Formatted string containing the analysis of the best performing problems
        """
        try:
            df = self.get_best_performing_problems(iteration, n, benchmark)
        except ValueError as e:
            return str(e)

        benchmark_scope = f"benchmark '{benchmark}'" if benchmark else "all benchmarks"
        output = [
            f"Best {n} performing problems for iteration {iteration} from {benchmark_scope}:"
        ]

        for _, row in df.iterrows():
            # We don't expect it to, but represent timeouts nonetheless
            timeout_status = " (TIMED OUT)" if row['timed_out'] else ""
            output.extend(
                [
                    f"\nBenchmark: {row['benchmark']} - Problem {row['problem_id']}{timeout_status}",
                    f"Score: {row['score']:.3f}",
                    f"Resources: {row['tokens']} tokens (of which cached {row['num_cached_tokens']}), ${row['cost']:.3f}, {row['time']:.1f}s",
                    f"Execution Tree:\n{row['trace']}",
                    f"Summary: {row['summary']}",
                ]
            )

        return "\n".join(output)

    def compare_iterations_detailed(self, iterations: List[int]) -> pd.DataFrame:
        """Compare specific agent iterations side by side with detailed metrics."""
        df = self.analyze_run()
        comparison_df = df[df["iteration"].isin(iterations)].copy()

        if comparison_df.empty:
            return pd.DataFrame()

        # Pivot the data to have iterations as columns
        result = pd.DataFrame()
        for col in comparison_df.columns:
            if col != "iteration":
                for iter_num in iterations:
                    if iter_num in comparison_df.index:
                        result.loc[col, iter_num] = comparison_df.loc[iter_num, col]

        return result

    def compare_iterations_formatted(self, iterations: List[int]) -> str:
        """Get formatted comparison of iterations."""
        df = self.compare_iterations_detailed(iterations)
        if df.empty:
            return "No data available for comparison"

        output = ["Iteration Comparison:"]

        # Add descriptions
        for iteration in iterations:
            output.append(f"\nIteration {iteration}:")
            output.append(self.get_agent_description(iteration))

        # Add performance comparison
        output.append("\nPerformance Metrics:")
        comparison_data = []

        # Prepare data for tabulate
        for metric in df.index:
            if metric != "description":
                # Format the metric name for display
                metric_name = str(metric).replace("_", " ").title()
                row = [metric_name]

                # Add values for each iteration
                for iteration in iterations:
                    if iteration in df.columns:
                        value = df.loc[metric, iteration]
                        if pd.isna(value):
                            row.append("N/A")
                        elif isinstance(value, float):
                            row.append(f"{value:.3f}")
                        else:
                            row.append(str(value))
                    else:
                        row.append("N/A")

                comparison_data.append(row)

        headers = ["Metric"] + [f"Iteration {i}" for i in iterations]
        output.append(tabulate(comparison_data, headers=headers, tablefmt="pipe"))

        # Add paired differences between each pair of iterations
        scores_df, _ = self.get_problem_scores_by_iteration()
        paired_lines = []
        for n, baseline in enumerate(iterations):
            for iteration in iterations[n + 1 :]:
                delta = paired_delta(scores_df, iteration, baseline)
                if delta is not None:
                    paired_lines.append(
                        f"Iteration {iteration} vs {baseline}: {delta} ({delta.n_problems} shared problems)"
                    )
        if paired_lines:
            output.append("\nPaired Score Differences:")
            output.extend(paired_lines)
            output.append(PAIRED_DELTA_LEGEND.format(column="Each line"))

        return "\n".join(output)

    def get_improvement_trend(self) -> pd.DataFrame:
        """Analyze the trend of improvement across iterations."""
        df = self.analyze_run()

        if df.empty:
            return df

        # Get all benchmark score columns
        score_cols = [
            col
            for col in df.columns
            if col.endswith("_score")  # and col != "utility_score"
        ]

        # Calculate mean score across all benchmarks
        df["mean_score"] = df[score_cols].mean(axis=1)

        # Calculate rolling averages
        df["rolling_avg_3"] = df["mean_score"].rolling(window=3, min_periods=1).mean()
        df["rolling_avg_5"] = df["mean_score"].rolling(window=5, min_periods=1).mean()

        # Select relevant columns
        trend_cols = [
            "iteration",
            "mean_score",
            "rolling_avg_3",
            "rolling_avg_5",
        ] + score_cols
        return df[trend_cols]

    def get_improvement_trend_formatted(self) -> str:
        """Get formatted improvement trend analysis."""
        df = self.get_improvement_trend()
        if df.empty:
            return "No trend data available"

        output = ["Performance Trend Analysis:"]

        # Overall metrics
        first_score = df.iloc[0]["mean_score"]
        last_score = df.iloc[-1]["mean_score"]
        best_score = df["mean_score"].max()
        best_iter = df.loc[df["mean_score"] == best_score, "iteration"].iloc[0]

        output.extend(
            [
                f"\nStarting Score: {first_score:.3f}",
                f"Final Score: {last_score:.3f}",
                f"Best Score: {best_score:.3f} (iteration {best_iter})",
                f"Total Improvement: {((last_score - first_score) / first_score * 100):.1f}%",
            ]
        )

        # Trend table
        trend_data = []
        headers = ["Iteration", "Score", "3-iter Avg", "5-iter Avg"]

        for _, row in df.iterrows():
            trend_data.append(
                [
                    row["iteration"],
                    f"{row['mean_score']:.3f}",
                    f"{row['rolling_avg_3']:.3f}",
                    f"{row['rolling_avg_5']:.3f}",
                ]
            )

        output.append("\nDetailed Trend:")
        output.append(tabulate(trend_data, headers=headers, tablefmt="pipe"))

        return "\n".join(output)

    def get_problem_scores_by_iteration(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract individual problem scores and benchmark summaries for each iteration."""
        all_scores = []
        benchmark_summaries = []

        root_path = Path(self.archive_root).exists()

        # Iterate through agent directories
        for agent_dir in sorted(Path(self.archive_root).glob("agent_*")):
            try:
                iteration = int(agent_dir.name.split("_")[1])
                benchmark_dir = agent_dir / "benchmarks"

                if not benchmark_dir.exists():
                    continue

                # Iterate through benchmark directories
                for bench_dir in benchmark_dir.iterdir():
                    if not bench_dir.is_dir():
                        continue

                    # Read benchmark summary from perf.json
                    perf_file = bench_dir / "perf.json"
                    if perf_file.exists():
                        perf_data = read_perf_json(perf_file)

                        # Create BenchmarkScore object to calculate utility if needed
                        benchmark_score = BenchmarkScore(
                            avg_score=perf_data["avg_score"],
                            tokens=perf_data["tokens"],
                            num_cached_tokens=perf_data["num_cached_tokens"],
                            cost=perf_data["cost"],
                            time=perf_data["time"],
                            timed_out=perf_data.get("timed_out", False)
                        )

                        benchmark_summaries.append(
                            {
                                "iteration": iteration,
                                "benchmark": bench_dir.name,
                                "avg_score": perf_data["avg_score"],
                                "utility_score": benchmark_score.utility(),
                                "total_tokens": perf_data["tokens"],
                                "total_cached_tokens": perf_data["num_cached_tokens"],
                                "total_cost": perf_data["cost"],
                                "total_time": perf_data["time"],
                                "timed_out": perf_data.get("timed_out", False)
                            }
                        )

                    # Read individual problem results
                    results_file = bench_dir / "results.jsonl"
                    if results_file.exists():
                        results = read_jsonl(results_file)
                        for result in results:
                            # Include all results, treating null scores as 0
                            score = result.get("score", 0)  # Default to 0 if missing
                            if score is None:
                                score = 0

                            # For null runs, assume maximum cost/time impact for utility
                            timed_out = result.get("timed_out", False)
                            tokens = result.get("tokens_used", 0) or 0
                            cached = result.get("num_cached_tokens", 0) or 0
                            cost = result.get("cost_estimate", 10.0) or 10.0  # Max cost penalty
                            time = result.get("wall_time", 300.0) or 300.0  # Max time penalty

                            problem_utility = BenchmarkScore(
                                avg_score=score,
                                tokens=tokens,
                                num_cached_tokens=cached,
                                cost=cost,
                                time=time,
                                timed_out=timed_out,
                            ).utility()

                            repeat = result.get("repeat", 0) or 0
                            score_data = {
                                "iteration": iteration,
                                "benchmark": bench_dir.name,
                                "problem_id": base_problem_id(
                                    result["problem_id"], repeat
                                ),
                                "repeat": repeat,
                                "score": score,
                                "utility": problem_utility,
                                "tokens_used": tokens,
                                "num_cached_tokens": cached,
                                "cost_estimate": cost,
                                "wall_time": time,
                                "timed_out": timed_out,
                            }
                            all_scores.append(score_data)

            except Exception as e:
                logger.error(f"Error processing {agent_dir}: {e}")
                continue

        scores_df = pd.DataFrame(all_scores)
        summaries_df = pd.DataFrame(benchmark_summaries)

        if scores_df.empty or summaries_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        return scores_df, summaries_df


PAIRED_DELTA_LEGEND = (
    "{column}: score difference on the problems both agents were run on, "
    "with its 95% confidence interval. * marks a difference that is "
    "distinguishable from run-to-run noise (the interval excludes 0)."
)


@dataclass(frozen=True)
class PairedDelta:
    """The score difference between two agents on their shared problems."""

    mean: float
    ci_lower: float
    ci_upper: float
    n_problems: int  # problems that both agents were run on
    n_benchmarks: int

    @property
    def significant(self) -> bool:
        """Whether the confidence interval excludes zero."""
        if self.n_problems < 2:
            return False  # a single problem says nothing about variability
        return self.ci_lower > 0 or self.ci_upper < 0

    def __str__(self) -> str:
        text = f"{self.mean:+.3f} [{self.ci_lower:+.3f}, {self.ci_upper:+.3f}]"
        return f"{text} *" if self.significant else text


def paired_delta(
    scores_df: pd.DataFrame,
    iteration: int,
    baseline: int,
    score_type: ScoreType = "mean_score",
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Optional[PairedDelta]:
    """Score of `iteration` minus score of `baseline`, on shared problems.

    Agents are evaluated on the same problems, and problems differ a lot in
    difficulty. Comparing the two agents problem by problem removes that
    shared variation, which gives a much tighter interval than comparing
    their two independent confidence intervals.

    The difference is the mean over benchmarks of the mean per-problem
    difference (matching how the overall score weights benchmarks), and the
    interval is a percentile bootstrap that resamples problems within each
    benchmark. Repeated runs of a problem are averaged first.

    Args:
        scores_df: Problem scores, from `get_problem_scores_by_iteration`
        iteration: The agent being assessed
        baseline: The agent it is compared against
        score_type: Whether to compare raw scores or utility
        n_resamples: Number of bootstrap resamples
        confidence: Coverage of the interval
        seed: Seed of the bootstrap, making the result reproducible

    Returns:
        None if the two agents have no problems in common.
    """
    if scores_df.empty:
        return None
    score_key = "score" if score_type == "mean_score" else "utility"

    selected = scores_df[scores_df["iteration"].isin([iteration, baseline])]
    per_problem = (
        selected.groupby(["benchmark", "problem_id", "iteration"])[score_key]
        .mean()
        .unstack("iteration")
    )
    if iteration not in per_problem.columns or baseline not in per_problem.columns:
        return None
    differences = (per_problem[iteration] - per_problem[baseline]).dropna()
    if differences.empty:
        return None

    rng = np.random.default_rng(seed)
    by_benchmark = [
        d.to_numpy(dtype=float) for _, d in differences.groupby(level="benchmark")
    ]
    mean = 0.0
    resampled = np.zeros(n_resamples)
    for values in by_benchmark:
        mean += values.mean()
        picks = rng.integers(0, len(values), size=(n_resamples, len(values)))
        resampled += values[picks].mean(axis=1)
    mean /= len(by_benchmark)
    resampled /= len(by_benchmark)

    tail = (1.0 - confidence) / 2.0
    ci_lower, ci_upper = np.quantile(resampled, [tail, 1.0 - tail])
    return PairedDelta(
        mean=float(mean),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_problems=len(differences),
        n_benchmarks=len(by_benchmark),
    )


def compute_statistics(
    scores_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    score_type: ScoreType = "mean_score",
) -> pd.DataFrame:
    """Compute statistics using benchmark summaries and individual scores."""
    if scores_df.empty or summaries_df.empty:
        return pd.DataFrame()

    # Calculate aggregate statistics per iteration using perf.json summaries
    stats = (
        summaries_df.groupby("iteration")
        .agg(
            {
                "avg_score": "mean",
                "utility_score": "mean",
                "total_tokens": "sum",
                "total_cached_tokens": "sum",
                "total_cost": "sum",
                "total_time": "sum",
            }
        )
        .reset_index()
    )

    # Calculate score standard deviation and count from individual results
    score_key = "score" if score_type == "mean_score" else "utility"
    score_stats = (
        per_problem_scores(scores_df, score_key)
        .groupby("iteration")[score_key]
        .agg(["std", "count"])
        .reset_index()
    )

    # Merge the statistics
    stats = stats.merge(score_stats, on="iteration", how="left")

    # Select the appropriate score column for final metrics
    score_col = "avg_score" if score_type == "mean_score" else "utility_score"

    # Rename columns for clarity
    column_mapping = {
        "iteration": "iteration",
        score_col: "target_score",  # This will be either avg_score or utility_score
        "total_tokens": "total_tokens",
        "total_cached_tokens": "total_cached_tokens",
        "total_cost": "total_cost",
        "total_time": "total_time",
        "std": "std_score",
        "count": "num_problems"
    }

    # Rename and select only the columns we want
    stats = stats.rename(columns=column_mapping)[column_mapping.values()]

    # Calculate cumulative maximum
    stats["cummax_score"] = stats["target_score"].cummax()

    # Compute 95% confidence intervals
    z_score = 1.96  # For 95% confidence
    stats["ci_lower"] = stats["target_score"] - z_score * (
        stats["std_score"] / np.sqrt(stats["num_problems"])
    )
    stats["ci_upper"] = stats["target_score"] + z_score * (
        stats["std_score"] / np.sqrt(stats["num_problems"])
    )

    return stats


if __name__ == "__main__":
    analyzer = ArchiveAnalyzer("results/run_1")

    # Get formatted overview
    print(analyzer.analyze_run_formatted())

    # Get specific agent description
    print(analyzer.get_agent_description(1))

    # Get improvement analysis
    print(analyzer.get_improvement_trend_formatted())

    # # Compare specific iterations
    print(analyzer.compare_iterations_formatted([1, 2, 3]))

    scores_df, summaries_df = analyzer.get_problem_scores_by_iteration()

    # Compute statistics including confidence intervals
    stats = compute_statistics(scores_df, summaries_df, score_type="utility_score")

    best_idx = stats["target_score"].idxmax()
    best_stats = stats.loc[best_idx]

    best_lower_bound = best_stats["ci_lower"]
    # print(best_lower_bound)
