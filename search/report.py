"""
Report on self-improvement experiments.

    python -m search.report results/run_1
    python -m search.report results/run_1 results/run_2 results/run_3

For one experiment: the tree of agents, progress per generation (scores and
cost) and how each parent was selected. For several experiments: their
best-score curves side by side, and how much they differ from run to run.

Options: --dot FILE writes the agent tree in Graphviz format, --csv FILE the
best-score curves, and --json prints everything as JSON.
"""

import sys
import csv
import json
import argparse
import statistics

from pathlib import Path
from dataclasses import asdict, dataclass

from tabulate import tabulate

from base_agent.src.utils.archive_analysis import (
    ArchiveAnalyzer,
    PairedDelta,
    compute_statistics,
    paired_delta,
)
from search.evaluation import holdout_dir


@dataclass
class AgentRow:
    """What is known about one agent of an experiment."""

    id: int
    parent_id: int | None = None
    generation: int | None = None
    status: str = "ok"
    pruned: bool = False  # dropped after the first evaluation stage
    meta_agent_id: int | None = None
    meta_model: str | None = None
    score: float | None = None  # on the problems in the archive
    holdout_score: float | None = None  # on the held-out problems
    vs_parent: PairedDelta | None = None
    eval_cost: float = 0.0  # benchmarks, including held-out ones
    meta_cost: float = 0.0  # the meta-improvement step that created it
    selection: dict | None = None

    @property
    def usable(self) -> bool:
        """Whether its score counts: created successfully, evaluated in full."""
        return self.status == "ok" and not self.pruned and self.score is not None


def _scores_and_costs(root: Path) -> tuple[dict[int, float], dict[int, float]]:
    """Overall score and total benchmark cost of each agent in an archive."""
    if not root.is_dir():
        return {}, {}
    scores_df, summaries_df = ArchiveAnalyzer(root).get_problem_scores_by_iteration()
    stats = compute_statistics(scores_df, summaries_df)
    if stats.empty:
        return {}, {}
    stats = stats.set_index("iteration")
    return (
        {int(i): float(v) for i, v in stats["target_score"].items()},
        {int(i): float(v) for i, v in stats["total_cost"].items()},
    )


def collect(exp_dir: Path) -> list[AgentRow]:
    """Gather the agents of an experiment, in order of id."""
    exp_dir = Path(exp_dir)
    analyzer = ArchiveAnalyzer(exp_dir)
    scores, costs = _scores_and_costs(exp_dir)
    holdout_scores, holdout_costs = _scores_and_costs(holdout_dir(exp_dir))
    scores_df, _ = analyzer.get_problem_scores_by_iteration()

    rows = []
    for agent_id, record in analyzer.get_lineage().items():
        parent_id = record.get("parent_id")
        usage = record.get("usage") or {}
        rows.append(
            AgentRow(
                id=agent_id,
                parent_id=parent_id,
                generation=record.get("generation"),
                status=record.get("status", "ok"),
                pruned=(record.get("evaluation") or {}).get("promoted") is False,
                meta_agent_id=record.get("meta_agent_id"),
                meta_model=record.get("meta_model"),
                score=scores.get(agent_id),
                holdout_score=holdout_scores.get(agent_id),
                vs_parent=(
                    None
                    if parent_id is None
                    else paired_delta(scores_df, agent_id, parent_id)
                ),
                eval_cost=costs.get(agent_id, 0.0) + holdout_costs.get(agent_id, 0.0),
                meta_cost=float(usage.get("cost") or 0.0),
                selection=record.get("selection"),
            )
        )
    return rows


def best_agent(rows: list[AgentRow]) -> AgentRow | None:
    usable = [row for row in rows if row.usable]
    return max(usable, key=lambda row: (row.score, -row.id)) if usable else None


# Agent tree -------------------------------------------------------------------


def _describe(row: AgentRow, best_id: int | None) -> str:
    parts = [f"agent {row.id}"]
    if row.status != "ok":
        parts.append(f"[{row.status}]")
    elif row.score is None:
        parts.append("[not evaluated]")
    else:
        parts.append(f"{row.score:.3f}")
        if row.holdout_score is not None:
            parts.append(f"(holdout {row.holdout_score:.3f})")
        if row.vs_parent is not None:
            parts.append(f"vs parent {row.vs_parent}")
        if row.pruned:
            parts.append("[dropped after first stage]")
    if row.id == best_id:
        parts.append("<- best")
    return "  ".join(parts)


def render_tree(rows: list[AgentRow]) -> str:
    """The agents as an indented tree, each under the agent it was built from."""
    ids = {row.id for row in rows}
    children: dict[int | None, list[AgentRow]] = {}
    for row in rows:
        parent = row.parent_id if row.parent_id in ids else None
        children.setdefault(parent, []).append(row)
    best = best_agent(rows)
    best_id = best.id if best else None

    lines: list[str] = []

    def walk(row: AgentRow, prefix: str, connector: str, child_prefix: str) -> None:
        lines.append(f"{prefix}{connector}{_describe(row, best_id)}")
        kids = children.get(row.id, [])
        for n, kid in enumerate(kids):
            last = n == len(kids) - 1
            walk(
                kid,
                prefix + child_prefix,
                "└── " if last else "├── ",
                "    " if last else "│   ",
            )

    for root in children.get(None, []):
        walk(root, "", "", "")
    return "\n".join(lines)


def to_dot(rows: list[AgentRow]) -> str:
    """The agent tree in Graphviz format."""
    best = best_agent(rows)
    lines = ["digraph agents {", "  rankdir=TB;", "  node [shape=box, fontname=Helvetica];"]
    for row in rows:
        label = f"agent {row.id}"
        if row.status != "ok":
            label += f"\\n{row.status}"
        elif row.score is not None:
            label += f"\\n{row.score:.3f}"
            if row.holdout_score is not None:
                label += f" / holdout {row.holdout_score:.3f}"
        attrs = [f'label="{label}"']
        if row.status != "ok" or row.pruned:
            attrs.append('style=dashed, color=gray50, fontcolor=gray50')
        elif best is not None and row.id == best.id:
            attrs.append("penwidth=3")
        lines.append(f"  a{row.id} [{', '.join(attrs)}];")
    ids = {row.id for row in rows}
    for row in rows:
        if row.parent_id in ids:
            label = ""
            if row.vs_parent is not None:
                label = f' [label="{row.vs_parent.mean:+.3f}{"*" if row.vs_parent.significant else ""}"]'
            lines.append(f"  a{row.parent_id} -> a{row.id}{label};")
    lines.append("}")
    return "\n".join(lines) + "\n"


# Progress per generation ------------------------------------------------------


def generations(rows: list[AgentRow]) -> list[dict]:
    """Progress of the search, one entry per generation.

    Agents of unknown generation (from before lineage tracking) each count as
    a generation of their own, in order of id: that is how they were made.
    """
    by_generation: dict[int, list[AgentRow]] = {}
    for row in rows:
        generation = row.generation if row.generation is not None else row.id
        by_generation.setdefault(generation, []).append(row)

    entries = []
    best: AgentRow | None = None
    cumulative_cost = 0.0
    for generation in sorted(by_generation):
        agents = by_generation[generation]
        usable = [row for row in agents if row.usable]
        for row in usable:
            if best is None or row.score > best.score:
                best = row
        eval_cost = sum(row.eval_cost for row in agents)
        meta_cost = sum(row.meta_cost for row in agents)
        cumulative_cost += eval_cost + meta_cost
        entries.append(
            {
                "generation": generation,
                "agents": [row.id for row in agents],
                "failed": sum(row.status != "ok" for row in agents),
                "dropped": sum(row.pruned for row in agents),
                "best_score": max((row.score for row in usable), default=None),
                "best_so_far": best.score if best else None,
                "best_agent": best.id if best else None,
                "best_holdout_score": best.holdout_score if best else None,
                "eval_cost": eval_cost,
                "meta_cost": meta_cost,
                "cumulative_cost": cumulative_cost,
            }
        )
    return entries


def _fmt(value, pattern="{:.3f}") -> str:
    return "-" if value is None else pattern.format(value)


def render_generations(entries: list[dict]) -> str:
    has_holdout = any(e["best_holdout_score"] is not None for e in entries)
    headers = ["Gen", "Agents", "Failed", "Dropped", "Best", "Best so far", "By agent"]
    if has_holdout:
        headers.append("Its holdout")
    headers += ["Eval cost", "Meta cost", "Total cost"]
    table = []
    for e in entries:
        line = [
            e["generation"],
            ",".join(map(str, e["agents"])),
            e["failed"],
            e["dropped"],
            _fmt(e["best_score"]),
            _fmt(e["best_so_far"]),
            _fmt(e["best_agent"], "{}"),
        ]
        if has_holdout:
            line.append(_fmt(e["best_holdout_score"]))
        line += [
            f"${e['eval_cost']:.2f}",
            f"${e['meta_cost']:.2f}",
            f"${e['cumulative_cost']:.2f}",
        ]
        table.append(line)
    return tabulate(table, headers=headers, tablefmt="pipe", disable_numparse=True)


# Selection log ----------------------------------------------------------------


def render_selections(rows: list[AgentRow]) -> str:
    table = []
    for row in rows:
        selection = row.selection
        if not selection:
            continue
        weights = selection.get("weights") or {}
        probability = weights.get(str(row.parent_id))
        candidates = ", ".join(
            f"{agent}: {weight:.2f}" for agent, weight in weights.items()
        )
        table.append(
            [
                row.id,
                _fmt(selection.get("generation"), "{}"),
                selection.get("strategy", "-"),
                _fmt(selection.get("seed"), "{}"),
                _fmt(row.parent_id, "{}"),
                _fmt(probability, "{:.2f}"),
                candidates or "-",
                _fmt(row.meta_agent_id, "{}") + (f" ({row.meta_model})" if row.meta_model else ""),
            ]
        )
    if not table:
        return "No selections recorded."
    headers = ["Agent", "Gen", "Strategy", "Seed", "Parent", "P(parent)", "Candidates", "Meta agent"]
    return tabulate(table, headers=headers, tablefmt="pipe", disable_numparse=True)


# Holdout ----------------------------------------------------------------------


def render_holdout(rows: list[AgentRow]) -> str | None:
    """Do gains on the archive problems carry over to the held-out ones?"""
    both = [row for row in rows if row.usable and row.holdout_score is not None]
    if not both:
        return None
    baseline = both[0]
    table = [
        [
            row.id,
            f"{row.score:.3f}",
            f"{row.holdout_score:.3f}",
            f"{row.score - baseline.score:+.3f}",
            f"{row.holdout_score - baseline.holdout_score:+.3f}",
        ]
        for row in both
    ]
    headers = [
        "Agent",
        "Archive",
        "Holdout",
        f"Archive gain vs agent {baseline.id}",
        f"Holdout gain vs agent {baseline.id}",
    ]
    note = (
        "Agents see the archive problems but never the held-out ones. A gain on the "
        "archive that does not show on the holdout is overfitting to those problems."
    )
    return f"{tabulate(table, headers=headers, tablefmt='pipe', disable_numparse=True)}\n\n{note}"


# One experiment ---------------------------------------------------------------


def render_experiment(exp_dir: Path) -> str:
    rows = collect(exp_dir)
    if not rows:
        return f"No agents found in {exp_dir}"
    sections = [
        f"# {exp_dir}",
        "## Agent tree\n\n"
        "Score on the archive problems; 'vs parent' is the paired difference with its "
        "95% confidence interval (* excludes 0).\n\n" + render_tree(rows),
        "## Progress per generation\n\n" + render_generations(generations(rows)),
    ]
    holdout = render_holdout(rows)
    if holdout:
        sections.append("## Archive vs holdout\n\n" + holdout)
    sections.append("## Parent selection\n\n" + render_selections(rows))
    return "\n\n".join(sections)


# Several experiments ----------------------------------------------------------


def best_so_far_curve(rows: list[AgentRow]) -> dict[int, float]:
    """Best score so far, by generation (only generations with a score)."""
    return {
        e["generation"]: e["best_so_far"]
        for e in generations(rows)
        if e["best_so_far"] is not None
    }


def compare_runs(exp_dirs: list[Path]) -> dict:
    """Best-score curves of several experiments, and their spread.

    A run that has not reached a generation carries its last value forward
    for the mean and standard deviation of that generation.
    """
    curves = {str(d): best_so_far_curve(collect(d)) for d in exp_dirs}
    all_generations = sorted({g for curve in curves.values() for g in curve})
    table = []
    for generation in all_generations:
        values = []
        for curve in curves.values():
            reached = [g for g in curve if g <= generation]
            values.append(curve[max(reached)] if reached else None)
        known = [v for v in values if v is not None]
        table.append(
            {
                "generation": generation,
                "scores": values,
                "mean": statistics.fmean(known) if known else None,
                "std": statistics.stdev(known) if len(known) > 1 else None,
            }
        )
    finals = [curve[max(curve)] for curve in curves.values() if curve]
    return {
        "runs": list(curves),
        "generations": table,
        "final_mean": statistics.fmean(finals) if finals else None,
        "final_std": statistics.stdev(finals) if len(finals) > 1 else None,
    }


def render_comparison(comparison: dict) -> str:
    names = [Path(run).name for run in comparison["runs"]]
    table = [
        [row["generation"], *[_fmt(v) for v in row["scores"]], _fmt(row["mean"]), _fmt(row["std"])]
        for row in comparison["generations"]
    ]
    text = tabulate(
        table, headers=["Gen", *names, "Mean", "Std"], tablefmt="pipe", disable_numparse=True
    )
    summary = (
        f"Final best score over {len(names)} runs: mean {_fmt(comparison['final_mean'])}, "
        f"standard deviation {_fmt(comparison['final_std'])}"
    )
    return f"# Best score so far, by generation\n\n{text}\n\n{summary}"


def write_curves_csv(comparison: dict, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", *comparison["runs"], "mean", "std"])
        for row in comparison["generations"]:
            writer.writerow([row["generation"], *row["scores"], row["mean"], row["std"]])


# Entry point ------------------------------------------------------------------


def _row_dict(row: AgentRow) -> dict:
    data = asdict(row)
    data["vs_parent"] = asdict(row.vs_parent) if row.vs_parent else None
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m search.report", description=__doc__.strip().splitlines()[0]
    )
    parser.add_argument("experiments", nargs="+", type=Path, help="e.g. results/run_1")
    parser.add_argument("--dot", type=Path, help="Write the agent tree as Graphviz (one experiment)")
    parser.add_argument("--csv", type=Path, help="Write the best-score curves as CSV")
    parser.add_argument("--json", action="store_true", help="Print the report data as JSON")
    args = parser.parse_args(argv)

    missing = [str(d) for d in args.experiments if not d.is_dir()]
    if missing:
        parser.error(f"not a directory: {', '.join(missing)}")
    if args.dot and len(args.experiments) != 1:
        parser.error("--dot takes exactly one experiment")

    comparison = compare_runs(args.experiments)
    if args.dot:
        args.dot.write_text(to_dot(collect(args.experiments[0])))
    if args.csv:
        write_curves_csv(comparison, args.csv)

    if args.json:
        data = {
            "experiments": {
                str(d): {
                    "agents": [_row_dict(row) for row in collect(d)],
                    "generations": generations(collect(d)),
                }
                for d in args.experiments
            },
            "comparison": comparison,
        }
        print(json.dumps(data, indent=2))
    elif len(args.experiments) == 1:
        print(render_experiment(args.experiments[0]))
    else:
        print(render_comparison(comparison))
    return 0


if __name__ == "__main__":
    sys.exit(main())
