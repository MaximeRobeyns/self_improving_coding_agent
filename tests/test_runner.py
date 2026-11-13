"""Tests for the host-side runner. Run from the repo root: pytest tests"""
import shutil
import asyncio

import json
from pathlib import Path
import pytest

try:
    import runner
except ImportError as e:  # e.g. the SWE-bench harness is not installed from source
    pytest.skip(f"runner.py is not importable here: {e}", allow_module_level=True)


# get_container_host_port ------------------------------------------------------


@pytest.mark.parametrize(
    "docker_output, expected",
    [
        ((True, "127.0.0.1:49153", ""), 49153),
        # Docker prints one line per address family
        ((True, "0.0.0.0:32768\n[::]:32768", ""), 32768),
        ((False, "", "Error: No such container"), None),
        ((True, "", ""), None),
        ((True, "garbage", ""), None),
    ],
)
def test_get_container_host_port(monkeypatch, docker_output, expected):
    calls = []

    async def fake_docker(*args):
        calls.append(args)
        return docker_output

    monkeypatch.setattr(runner, "run_docker_command", fake_docker)

    port = asyncio.run(runner.get_container_host_port("some_container"))

    assert port == expected
    assert calls == [("docker", "port", "some_container", "8080/tcp")]


def test_monitor_port_mapping_lets_docker_pick_the_host_port():
    # No fixed host port: concurrent containers must not collide
    assert runner.MONITOR_PORT_MAPPING == "127.0.0.1::8080"


# Fixtures --------------------------------------------------------------------


class FakeDocker:
    """Replaces run_docker_command; fails the sub-commands named in `fail`."""

    def __init__(self, fail: set[str] = frozenset(), fail_commands: dict | None = None):
        self.fail = fail
        # {text: n}: commands containing the text fail, the first n times
        # they are run (always, if n is None)
        self.fail_commands = dict(fail_commands or {})
        self.calls: list[tuple] = []

    async def __call__(self, *args):
        self.calls.append(args)
        subcommand = args[1]
        if subcommand in self.fail:
            return False, "", f"{subcommand} went wrong"
        text = " ".join(map(str, args))
        for marker, remaining in self.fail_commands.items():
            if marker in text and remaining != 0:
                if remaining is not None:
                    self.fail_commands[marker] = remaining - 1
                return False, "", f"{marker} went wrong"
        if subcommand == "exec" and "cat" in args:
            return True, "4", ""  # the answer of the validation gate's smoke run
        if subcommand == "inspect":
            return True, '"running"', ""
        if subcommand == "port":
            return True, "127.0.0.1:49153", ""
        if subcommand == "exec" and ("improve" in args or "benchmark" in args):
            return True, "some agent output\n1000|500|0.25|12.5", ""
        return True, "", ""

    def calls_to(self, subcommand: str) -> list[tuple]:
        return [c for c in self.calls if c[1] == subcommand]

    def meta_run(self) -> tuple:
        """The `docker run` of the container that the meta agent works in"""
        (cmd,) = [
            c for c in self.calls_to("run") if any(":/home/agent/workdir:rw" in a for a in c)
        ]
        return cmd

    def gate_runs(self) -> list[tuple]:
        """The `docker run`s of validation containers"""
        return [c for c in self.calls_to("run") if any(":/mnt/agent_code:ro" in a for a in c)]

    def commands_containing(self, text: str) -> list[tuple]:
        return [c for c in self.calls if text in " ".join(map(str, c))]


def make_agent(exp_dir, agent_id, parent_id=None, status="ok", description=None):
    """An agent directory with code and (unless parent_id is ...) a lineage record."""
    code_dir = exp_dir / f"agent_{agent_id}" / "agent_code"
    code_dir.mkdir(parents=True)
    (code_dir / "agent_change_log.md").write_text(f"# log of agent {agent_id}\n")
    (code_dir / "description.txt").write_text(description or f"Agent {agent_id}")
    if parent_id is not ...:
        runner.write_lineage(
            exp_dir / f"agent_{agent_id}", id=agent_id, parent_id=parent_id, status=status
        )
    return exp_dir / f"agent_{agent_id}"


def add_results(exp_dir, agent_id, scores):
    """Give an agent benchmark results, so that it is a parent candidate."""
    bench_dir = exp_dir / f"agent_{agent_id}" / "benchmarks" / "gsm8k"
    bench_dir.mkdir(parents=True)
    results = [
        {
            "problem_id": f"p{n}",
            "score": float(score),
            "tokens_used": 100,
            "num_cached_tokens": 0,
            "cost_estimate": 0.01,
            "wall_time": 1.0,
            "timed_out": False,
            "cost_threshold_exceeded": False,
        }
        for n, score in enumerate(scores)
    ]
    (bench_dir / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in results)
    )
    perf = {
        "avg_score": sum(scores) / len(scores),
        "tokens": 100 * len(scores),
        "num_cached_tokens": 0,
        "cost": 0.01 * len(scores),
        "time": 1.0 * len(scores),
    }
    (bench_dir / "perf.json").write_text(json.dumps(perf))


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    """A fresh experiment (results/run_1) holding only agent_0, as the cwd."""
    monkeypatch.chdir(tmp_path)
    exp_dir = tmp_path / "results" / "run_1"
    exp_dir.mkdir(parents=True)
    runner.load_metadata(exp_dir)
    make_agent(exp_dir, 0)

    async def fake_summary(*args, **kwargs):
        return "summary"

    monkeypatch.setattr(runner, "generate_contextual_summary", fake_summary)
    # Benchmark jobs stagger their start by a random delay
    monkeypatch.setattr(runner.random, "random", lambda: 0.0)
    return exp_dir


@pytest.fixture
def docker(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(runner, "run_docker_command", fake)
    return fake


@pytest.fixture
def scored_experiment(experiment):
    """agent_0 is weak, agent_1 strong, agent_2 (the latest) regressed."""
    make_agent(experiment, 1, parent_id=0)
    make_agent(experiment, 2, parent_id=1)
    add_results(experiment, 0, [0, 0, 0, 1])
    add_results(experiment, 1, [1, 1, 1, 1])
    add_results(experiment, 2, [1, 0, 0, 0])
    return experiment


def child(agent_id=1, parent_id=0, meta_agent_id=0, **extra):
    return {"id": agent_id, "parent_id": parent_id, "meta_agent_id": meta_agent_id, **extra}


def create(exp_dir, planned=None):
    planned = planned or child()
    asyncio.run(runner.run_meta_agent_benchmark(1, exp_dir, planned))
    return runner.read_lineage(exp_dir / f"agent_{planned['id']}")


# run_meta_agent_benchmark -----------------------------------------------------


def test_created_agent_is_recorded(experiment, docker):
    record = create(experiment, child(selection={"strategy": "greedy_ci"}))

    assert record["status"] == "ok"
    assert (record["id"], record["parent_id"], record["meta_agent_id"]) == (1, 0, 0)
    assert record["generation"] == 1
    assert record["selection"] == {"strategy": "greedy_ci"}
    # What the meta-improvement step used, as reported by the meta agent
    assert record["usage"] == {"tokens": 1000, "cached_tokens": 500, "cost": 0.25, "time": 12.5}
    assert "failure_reason" not in record
    assert runner.get_failed_iterations(experiment) == set()
    # It starts from the parent's code
    assert (experiment / "agent_1/agent_code/description.txt").read_text() == "Agent 0"
    # The logs of how an agent was made are kept with that agent
    assert (experiment / "agent_1/meta_improvement_logs/summary.txt").read_text() == "summary"


def test_meta_agent_is_told_which_agents_it_works_on(scored_experiment, docker):
    create(scored_experiment, child(agent_id=3, parent_id=2, meta_agent_id=1))

    (exec_cmd,) = [c for c in docker.calls_to("exec") if "improve" in c]
    assert exec_cmd[exec_cmd.index("-m") + 1] == "archive.agent_1.agent_code"
    # The code in the workdir is the parent's; the agent being made is the child
    assert exec_cmd[exec_cmd.index("--best-iter") + 1] == "2"
    assert exec_cmd[exec_cmd.index("--current-iter") + 1] == "3"


def test_container_uses_a_docker_assigned_port(experiment, docker):
    create(experiment)

    run_cmd = docker.meta_run()
    assert run_cmd[run_cmd.index("-p") + 1] == "127.0.0.1::8080"
    assert "8080:8080" not in run_cmd


def test_new_agent_gets_an_image_built_from_its_own_code(experiment, docker):
    create(experiment)

    (build_cmd,) = docker.calls_to("buildx")
    assert build_cmd[build_cmd.index("-t") + 1] == "sica_sandbox:run1_agent1"
    context = build_cmd[build_cmd.index("--build-context") + 1]
    assert context == f"base_agent={experiment / 'agent_1' / 'agent_code'}"


def test_meta_agent_runs_in_its_own_image(experiment, docker):
    create(experiment, child(meta_agent_id=0))
    run_cmd = docker.meta_run()
    assert "sica_sandbox:run1_agent0" in run_cmd


def test_agents_without_an_image_run_in_the_base_image(experiment, monkeypatch):
    docker = FakeDocker(fail={"image"})  # `docker image inspect` finds nothing
    monkeypatch.setattr(runner, "run_docker_command", docker)

    create(experiment)

    run_cmd = docker.meta_run()
    assert run_cmd[run_cmd.index("tail") - 1] == "sica_sandbox"


@pytest.mark.parametrize(
    "failing, reason",
    [
        ("run", "Could not start container"),
        ("exec", "exec went wrong"),
        ("buildx", "Image build failed"),
    ],
)
def test_failures_mark_the_new_agent_failed(experiment, monkeypatch, failing, reason):
    docker = FakeDocker(fail={failing})
    monkeypatch.setattr(runner, "run_docker_command", docker)

    record = create(experiment)

    assert record["status"] == "failed"
    assert reason in record["failure_reason"]
    assert runner.get_failed_iterations(experiment) == {1}
    # The container is still cleaned up
    assert docker.calls_to("rm")


def test_exception_marks_failed_and_propagates(experiment, docker, monkeypatch):
    async def exploding_wait(container_name, timeout=30):
        raise TimeoutError("container never became ready")

    monkeypatch.setattr(runner, "wait_for_container_ready", exploding_wait)

    with pytest.raises(TimeoutError):
        create(experiment)

    record = runner.read_lineage(experiment / "agent_1")
    assert record["status"] == "failed"
    assert "TimeoutError" in record["failure_reason"]


def test_interruption_leaves_the_agent_pending(experiment, docker, monkeypatch):
    async def interrupted_wait(container_name, timeout=30):
        raise asyncio.CancelledError()

    monkeypatch.setattr(runner, "wait_for_container_ready", interrupted_wait)

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await runner.run_meta_agent_benchmark(1, experiment, child())

    asyncio.run(run())

    # Not 'failed': it is to be created afresh when the experiment resumes
    assert runner.read_lineage(experiment / "agent_1")["status"] == "pending"
    assert docker.calls_to("rm")


def test_interrupted_attempt_is_set_aside_and_redone(experiment, docker):
    leftover = make_agent(experiment, 1, parent_id=0, status="pending")
    (leftover / "agent_code" / "half_written.py").write_text("x")

    record = create(experiment)

    assert record["status"] == "ok"
    assert not (experiment / "agent_1/agent_code/half_written.py").exists()
    # Nothing is deleted
    (kept,) = (experiment / "_interrupted").iterdir()
    assert (kept / "agent_code" / "half_written.py").exists()


@pytest.mark.parametrize("status", ["ok", "failed"])
def test_finished_agent_is_never_overwritten(experiment, docker, status):
    make_agent(experiment, 1, parent_id=0, status=status, description="keep me")

    with pytest.raises(ValueError, match="already been created"):
        create(experiment)

    assert (experiment / "agent_1/agent_code/description.txt").read_text() == "keep me"
    assert docker.calls == []


def test_change_log_is_the_parents_plus_other_branches(scored_experiment, docker):
    # 0 -> 1 -> 2, and the new agent 3 branches off agent 1
    create(scored_experiment, child(agent_id=3, parent_id=1, meta_agent_id=1))

    log = (scored_experiment / "agent_3/agent_code/agent_change_log.md").read_text()
    assert log.startswith("# log of agent 1\n")
    assert "### Agent 2 (built from agent 1): vs parent: -0.750" in log
    assert "Agent 1 (built" not in log  # its parent is not another branch


# plan_generation --------------------------------------------------------------


def test_plan_defaults_to_one_greedy_child(scored_experiment):
    plan = runner.plan_generation(scored_experiment, generation=2, n_children=1)

    assert plan == [
        {
            "id": 3,
            # Rolls back to the best agent rather than building on the regression
            "parent_id": 1,
            "meta_agent_id": 1,
            "meta_model": None,
            "selection": {
                "strategy": "greedy_ci",
                "seed": 0,
                "generation": 2,
                "n_children": 1,
                "index": 0,
                "weights": {"0": 0.0, "1": 1.0, "2": 0.0},
            },
        }
    ]


@pytest.mark.parametrize("strategy", ["score_child_prop", "thompson"])
def test_plan_of_several_children_can_be_replayed(scored_experiment, strategy):
    from search.selection import ArchiveView, get_selector, selection_rng

    plan = runner.plan_generation(
        scored_experiment, generation=5, n_children=4, selection=strategy, seed=99
    )

    assert [c["id"] for c in plan] == [3, 4, 5, 6]
    assert [c["selection"]["index"] for c in plan] == [0, 1, 2, 3]
    view = ArchiveView.from_analyzer(runner.ArchiveAnalyzer(scored_experiment), 2)
    for planned in plan:
        record = planned["selection"]
        replayed = get_selector(record["strategy"]).select(
            view, record["n_children"], selection_rng(record["seed"], record["generation"])
        )
        assert planned["parent_id"] == replayed.parents[record["index"]]
        assert record["weights"] == {str(i): w for i, w in replayed.weights.items()}


def test_plan_never_uses_a_failed_agent(scored_experiment):
    runner.write_lineage(scored_experiment / "agent_1", status="failed")

    for seed in range(20):
        plan = runner.plan_generation(
            scored_experiment, 2, 3, selection="score_child_prop", seed=seed
        )
        for planned in plan:
            assert planned["parent_id"] in (0, 2)
            assert planned["meta_agent_id"] != 1
            assert "1" not in planned["selection"]["weights"]


def test_plan_for_an_archive_without_results(experiment):
    plan = runner.plan_generation(experiment, generation=0, n_children=2)
    assert [(c["id"], c["parent_id"], c["meta_agent_id"]) for c in plan] == [
        (1, 0, 0),
        (2, 0, 0),
    ]


# run_search -------------------------------------------------------------------


class FakeSteps:
    """Replaces the two expensive steps of the loop, recording what happens."""

    def __init__(self, monkeypatch, exp_dir, fail=(), raise_for=(), stop_at=None):
        self.exp_dir = exp_dir
        self.fail, self.raise_for, self.stop_at = set(fail), set(raise_for), stop_at
        self.events: list[tuple] = []
        self.planned: list[dict] = []
        monkeypatch.setattr(runner, "run_benchmarks_with_job_queue", self.benchmarks)
        monkeypatch.setattr(runner, "run_meta_agent_benchmark", self.create)
        monkeypatch.setattr(runner, "run_docker_command", FakeDocker(fail={"image"}))

    async def benchmarks(self, exp_dir, agent_dir, benchmarks, workers, **kwargs):
        self.events.append(("evaluate", int(agent_dir.name.split("_")[1])))

    async def create(self, exp_id, exp_dir, planned, gate=None):
        self.planned.append({k: v for k, v in planned.items() if k != "selection"})
        agent_id = planned["id"]
        agent_dir = exp_dir / f"agent_{agent_id}"
        if agent_dir.exists():  # the real step sets an interrupted attempt aside
            assert runner.read_lineage(agent_dir)["status"] == "pending"
            shutil.rmtree(agent_dir)
        if agent_id == self.stop_at:
            self.stop_at = None
            make_agent(exp_dir, agent_id, planned["parent_id"], status="pending")
            raise KeyboardInterrupt()
        self.events.append(("create", agent_id, planned["parent_id"]))
        if agent_id in self.raise_for:
            make_agent(exp_dir, agent_id, planned["parent_id"], status="failed")
            raise RuntimeError(f"agent {agent_id} blew up")
        status = "failed" if agent_id in self.fail else "ok"
        make_agent(exp_dir, agent_id, planned["parent_id"], status=status)


def search(exp_dir, iterations, **kwargs):
    asyncio.run(runner.run_search(1, exp_dir, iterations, workers=2, **kwargs))


def test_one_child_per_generation_is_the_original_loop(experiment, monkeypatch):
    steps = FakeSteps(monkeypatch, experiment)

    search(experiment, iterations=3)

    assert steps.events == [
        ("evaluate", 0),
        ("create", 1, 0),
        ("evaluate", 1),
        ("create", 2, 0),
        ("evaluate", 2),
        ("create", 3, 0),
    ]
    metadata = runner.load_metadata(experiment)
    assert metadata["generation"] == 3
    assert metadata["pending_agents"] == [3]  # created, but not evaluated yet
    assert metadata["planned_children"] is None


def test_several_children_per_generation(experiment, monkeypatch):
    steps = FakeSteps(monkeypatch, experiment)

    search(experiment, iterations=2, n_children=3)

    assert steps.events[0] == ("evaluate", 0)
    assert sorted(steps.events[1:4]) == [("create", 1, 0), ("create", 2, 0), ("create", 3, 0)]
    assert steps.events[4:7] == [("evaluate", 1), ("evaluate", 2), ("evaluate", 3)]
    assert sorted(e[1] for e in steps.events[7:]) == [4, 5, 6]
    assert runner.load_metadata(experiment)["pending_agents"] == [4, 5, 6]


def test_failed_children_are_not_evaluated(experiment, monkeypatch):
    steps = FakeSteps(monkeypatch, experiment, fail={2})

    search(experiment, iterations=2, n_children=3)

    assert [e for e in steps.events if e[0] == "evaluate"] == [
        ("evaluate", 0),
        ("evaluate", 1),
        ("evaluate", 3),
    ]


def test_one_child_raising_does_not_stop_the_others(experiment, monkeypatch):
    steps = FakeSteps(monkeypatch, experiment, raise_for={2})

    search(experiment, iterations=2, n_children=3)

    assert runner.load_metadata(experiment)["generation"] == 2
    assert ("evaluate", 2) not in steps.events
    assert ("evaluate", 3) in steps.events


def test_search_stops_when_every_child_raises(experiment, monkeypatch):
    FakeSteps(monkeypatch, experiment, raise_for={1, 2})

    with pytest.raises(RuntimeError, match="blew up"):
        search(experiment, iterations=2, n_children=2)

    assert runner.load_metadata(experiment)["generation"] == 0


def test_resuming_finishes_the_planned_generation(experiment, monkeypatch):
    steps = FakeSteps(monkeypatch, experiment, stop_at=2)
    with pytest.raises(KeyboardInterrupt):
        search(experiment, iterations=1, n_children=3, selection="thompson", seed=5)
    plan = runner.load_metadata(experiment)["planned_children"]
    steps.events.clear()

    search(experiment, iterations=1, n_children=3, selection="thompson", seed=5)

    # agent_0 is not evaluated again, agent 1 is not created again, and the
    # agents that were not finished are created as they were planned
    assert sorted(steps.events) == [
        ("create", 2, plan[1]["parent_id"]),
        ("create", 3, plan[2]["parent_id"]),
    ]
    assert runner.load_metadata(experiment)["planned_children"] is None
    metadata = runner.load_metadata(experiment)
    assert metadata["generation"] == 1
    assert metadata["pending_agents"] == [1, 2, 3]


def test_two_generations_with_real_agent_creation(experiment, docker, monkeypatch):
    """Everything but docker and the benchmarks themselves is real."""
    scores = {0: [0, 1, 0, 1], 1: [1, 1, 0, 1], 2: [0, 0, 0, 1]}

    async def benchmarks(exp_dir, agent_dir, benchmarks, workers, **kwargs):
        agent_id = int(agent_dir.name.split("_")[1])
        add_results(exp_dir, agent_id, scores[agent_id])

    monkeypatch.setattr(runner, "run_benchmarks_with_job_queue", benchmarks)

    search(experiment, iterations=2, n_children=2, selection="score_child_prop", seed=3)

    lineage = runner.ArchiveAnalyzer(experiment).get_lineage()
    assert sorted(lineage) == [0, 1, 2, 3, 4]
    assert all(record["status"] == "ok" for record in lineage.values())
    assert [lineage[i]["parent_id"] for i in (1, 2)] == [0, 0]
    assert [lineage[i]["generation"] for i in (1, 2)] == [1, 1]
    for agent_id in (3, 4):
        record = lineage[agent_id]
        assert record["parent_id"] in (0, 1, 2)
        assert record["generation"] == lineage[record["parent_id"]]["generation"] + 1
        assert record["meta_agent_id"] == 1  # the best agent so far
        assert record["selection"]["generation"] == 1
        # Branches other than its own are described in its change log
        log = (experiment / f"agent_{agent_id}/agent_code/agent_change_log.md").read_text()
        siblings_of_parent = {1, 2} - {record["parent_id"]}
        for other in siblings_of_parent:
            assert f"### Agent {other} (built from agent 0): vs parent:" in log
    # Each new agent got its own image
    tags = [c[c.index("-t") + 1] for c in docker.calls_to("buildx")]
    assert sorted(tags) == [f"sica_sandbox:run1_agent{i}" for i in (1, 2, 3, 4)]
    assert runner.load_metadata(experiment)["pending_agents"] == [3, 4]


def test_legacy_experiment_is_resumed_where_it_was(experiment, monkeypatch):
    # As left by the runner before generations: agents 0-2 exist without
    # lineage records, agent 2 was being evaluated, and agent 3 is the remains
    # of a meta-improvement step that was interrupted earlier.
    shutil.rmtree(experiment / "agent_0")
    for agent_id in (0, 1, 2):
        make_agent(experiment, agent_id, parent_id=...)
    add_results(experiment, 0, [0, 1])
    add_results(experiment, 1, [1, 1])
    make_agent(experiment, 3, parent_id=...)
    (experiment / "metadata.json").write_text(json.dumps({"agent_iteration": 2}))
    steps = FakeSteps(monkeypatch, experiment)

    search(experiment, iterations=3)

    assert steps.events == [("evaluate", 2), ("create", 3, 1)]
    # Only the remains of the interrupted step were moved out of the way
    (kept,) = (experiment / "_interrupted").iterdir()
    assert kept.name.startswith("agent_3_")
    assert runner.read_lineage(experiment / "agent_2")["status"] == "ok"


# resolve_selection_seed -------------------------------------------------------


@pytest.fixture
def exp_dir(tmp_path):
    runner.load_metadata(tmp_path)  # creates metadata.json
    return tmp_path


def test_new_experiment_gets_a_random_seed_that_is_kept(exp_dir):
    seed = runner.resolve_selection_seed(exp_dir)

    assert isinstance(seed, int)
    assert runner.load_metadata(exp_dir)["selection_seed"] == seed
    # Resuming the experiment keeps it
    assert runner.resolve_selection_seed(exp_dir) == seed


def test_explicit_seed_is_used_and_stored(exp_dir):
    assert runner.resolve_selection_seed(exp_dir, 42) == 42
    assert runner.load_metadata(exp_dir)["selection_seed"] == 42
    assert runner.resolve_selection_seed(exp_dir) == 42
    # Seed 0 is a seed like any other
    assert runner.resolve_selection_seed(exp_dir, 0) == 0
    assert runner.resolve_selection_seed(exp_dir) == 0


def test_selection_cli_arguments():
    parser = runner.setup_argparse()
    assert parser.parse_args([]).selection == "greedy_ci"
    assert parser.parse_args([]).seed is None
    assert parser.parse_args([]).children == 1
    args = parser.parse_args(["--selection", "thompson", "--seed", "3", "--children", "4"])
    assert (args.selection, args.seed, args.children) == ("thompson", 3, 4)
    with pytest.raises(SystemExit):
        parser.parse_args(["--selection", "nope"])


# Evaluation: repeats, holdout, stages -------------------------------------------

from base_agent.src.benchmarks.base import BaseBenchmark, Problem
from search.evaluation import EvaluationConfig, split_problems, stage1_problems

N_FAKE_PROBLEMS = 10


class FakeBenchmark(BaseBenchmark):
    name = "fake"

    @property
    def problems(self):
        return [Problem(f"p{i}", f"What is {i}?", str(i), None) for i in range(N_FAKE_PROBLEMS)]

    async def score_problem(self, problem, agent_workdir, agent_answer_dir, container_name):
        return 1.0, None, None


@pytest.fixture
def fake_benchmark(monkeypatch):
    monkeypatch.setattr(runner, "benchmark_registry", {"fake": FakeBenchmark})
    return FakeBenchmark


def read_results(agent_results_dir):
    path = agent_results_dir / "benchmarks" / "fake" / "results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def run_queue(exp_dir, agent_id=0, **kwargs):
    asyncio.run(
        runner.run_benchmarks_with_job_queue(
            exp_dir, exp_dir / f"agent_{agent_id}", [FakeBenchmark], 3, **kwargs
        )
    )


def test_each_problem_is_run_repeatedly(experiment, docker):
    run_queue(experiment, repeats=3)

    results = read_results(experiment / "agent_0")
    assert len(results) == 3 * N_FAKE_PROBLEMS
    assert sorted(r["problem_id"] for r in results if r["problem_id"].startswith("p0")) == [
        "p0",
        "p0#1",
        "p0#2",
    ]
    assert {r["problem_id"]: r["repeat"] for r in results}["p0#2"] == 2
    assert all(r["score"] == 1.0 for r in results)
    # Every run keeps its own trace
    traces = experiment / "agent_0" / "benchmarks" / "fake" / "traces"
    assert (traces / "p0" / "summary.txt").exists()
    assert (traces / "p0#2" / "summary.txt").exists()
    progress = runner.load_metadata(experiment)["benchmark_progress"]["fake"]
    assert (progress["total"], progress["completed"]) == (30, 30)


def test_single_runs_are_stored_as_before(experiment, docker):
    run_queue(experiment)

    results = read_results(experiment / "agent_0")
    assert [r["problem_id"] for r in results] == [f"p{i}" for i in range(N_FAKE_PROBLEMS)]
    assert all(r["repeat"] == 0 for r in results)


def test_more_repeats_only_run_what_is_missing(experiment, docker):
    run_queue(experiment, repeats=1)
    n_runs = len([c for c in docker.calls_to("exec") if "benchmark" in c])

    run_queue(experiment, repeats=2)

    assert len([c for c in docker.calls_to("exec") if "benchmark" in c]) == 2 * n_runs
    assert len(read_results(experiment / "agent_0")) == 2 * N_FAKE_PROBLEMS


def test_only_selected_problems_are_run(experiment, docker):
    run_queue(experiment, select_problems=lambda b: b.problems[:4])
    assert [r["problem_id"] for r in read_results(experiment / "agent_0")] == [
        "p0",
        "p1",
        "p2",
        "p3",
    ]


def test_statistics_count_each_problem_once(tmp_path):
    bench_dir = tmp_path / "agent_0" / "benchmarks" / "fake"
    bench_dir.mkdir(parents=True)
    rows = [
        {"problem_id": "a", "repeat": 0, "score": 1.0},
        {"problem_id": "a#1", "repeat": 1, "score": 1.0},
        {"problem_id": "a#2", "repeat": 2, "score": 1.0},
        {"problem_id": "b", "repeat": 0, "score": 0.0},  # only run once so far
    ]
    (bench_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    asyncio.run(runner.generate_benchmark_statistics(tmp_path / "agent_0", "fake"))

    perf = json.loads((bench_dir / "perf.json").read_text())
    assert perf["avg_score"] == 0.5  # not 3/4


def test_holdout_results_stay_out_of_the_archive(experiment, docker):
    holdout = runner.holdout_dir(experiment)

    run_queue(experiment, results_exp_dir=holdout, select_problems=lambda b: b.problems[:2])

    assert holdout == experiment.parent / "holdout" / "run_1"
    assert [r["problem_id"] for r in read_results(holdout / "agent_0")] == ["p0", "p1"]
    assert (holdout / "agent_0" / "benchmarks" / "fake" / "perf.json").exists()
    assert (holdout / "agent_0" / "benchmarks" / "fake" / "traces" / "p0" / "summary.txt").exists()
    # Nothing about these problems is in the experiment directory, which is
    # what agents get to see as their archive
    assert not (experiment / "agent_0" / "benchmarks").exists()
    assert not (experiment / "problem_data").exists()
    assert "benchmark_progress" not in runner.load_metadata(experiment)
    # The container can write to the holdout directory
    run_cmd = docker.calls_to("run")[0]
    assert any(arg.startswith(f"{holdout}:") and arg.endswith(":rw") for arg in run_cmd)
    # And the holdout results can be analysed like an archive
    analyzer = runner.ArchiveAnalyzer(holdout)
    assert analyzer.analyze_run().loc[0, "fake_score"] == 1.0


class FakeQueue:
    """Replaces run_benchmarks_with_job_queue: 'runs' the selected problems,
    giving each agent the scores it is set up to get."""

    def __init__(self, monkeypatch, scores_by_agent):
        self.scores_by_agent = scores_by_agent
        self.calls: list[dict] = []
        monkeypatch.setattr(runner, "run_benchmarks_with_job_queue", self)
        monkeypatch.setattr(runner, "run_docker_command", FakeDocker(fail={"image"}))

    async def __call__(
        self, exp_dir, agent_dir, benchmarks, workers,
        image=None, results_exp_dir=None, select_problems=None, repeats=1,
    ):
        agent_id = int(agent_dir.name.split("_")[1])
        benchmark = benchmarks[0](seed=1, subset_size=18)
        selected = select_problems(benchmark) if select_problems else benchmark.problems
        results_root = results_exp_dir or exp_dir
        self.calls.append(
            {
                "agent": agent_id,
                "holdout": results_root != exp_dir,
                "problems": [p.problem_id for p in selected],
                "repeats": repeats,
            }
        )
        bench_dir = results_root / agent_dir.name / "benchmarks" / "fake"
        bench_dir.mkdir(parents=True, exist_ok=True)
        path = bench_dir / "results.jsonl"
        rows = {}
        if path.exists():
            rows = {r["problem_id"]: r for r in map(json.loads, path.read_text().splitlines())}
        for problem in selected:
            rows[problem.problem_id] = {
                "problem_id": problem.problem_id,
                "repeat": 0,
                "score": float(self.scores_by_agent[agent_id]),
                "tokens_used": 1,
                "num_cached_tokens": 0,
                "cost_estimate": 0.01,
                "wall_time": 1.0,
                "timed_out": False,
            }
        path.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
        perf = {"avg_score": float(self.scores_by_agent[agent_id]), "tokens": 1,
                "num_cached_tokens": 0, "cost": 0.01, "time": 1.0}
        (bench_dir / "perf.json").write_text(json.dumps(perf))


def evaluate(exp_dir, agent_id, config):
    asyncio.run(runner.evaluate_agent(1, exp_dir, agent_id, 2, config))


def fake_problem_ids(problem_list):
    return [p.problem_id for p in problem_list]


def test_default_evaluation_is_one_full_run(experiment, fake_benchmark, monkeypatch):
    queue = FakeQueue(monkeypatch, {0: 1.0})

    evaluate(experiment, 0, EvaluationConfig())

    assert queue.calls == [
        {
            "agent": 0,
            "holdout": False,
            "problems": [f"p{i}" for i in range(N_FAKE_PROBLEMS)],
            "repeats": 1,
        }
    ]
    assert "evaluation" not in runner.read_lineage(experiment / "agent_0")


def test_holdout_problems_are_run_separately(experiment, fake_benchmark, monkeypatch):
    queue = FakeQueue(monkeypatch, {0: 1.0})
    config = EvaluationConfig(holdout_fraction=0.3, repeats=2, split_seed=7)

    evaluate(experiment, 0, config)

    dev, holdout = split_problems(FakeBenchmark().problems, "fake", 0.3, 7)
    assert len(holdout) == 3
    assert queue.calls == [
        {"agent": 0, "holdout": False, "problems": fake_problem_ids(dev), "repeats": 2},
        {"agent": 0, "holdout": True, "problems": fake_problem_ids(holdout), "repeats": 2},
    ]
    # No held-out problem is in the archive
    archived = {r["problem_id"] for r in read_results(experiment / "agent_0")}
    assert archived.isdisjoint(fake_problem_ids(holdout))


@pytest.fixture
def parent_and_children(experiment, fake_benchmark, monkeypatch):
    """agent_0 (solves everything) has been evaluated; agent_1 is as good,
    agent_2 solves nothing."""
    make_agent(experiment, 1, parent_id=0)
    make_agent(experiment, 2, parent_id=0)
    queue = FakeQueue(monkeypatch, {0: 1.0, 1: 1.0, 2: 0.0})
    config = EvaluationConfig(stage1_size=4, holdout_fraction=0.2, split_seed=7)
    evaluate(experiment, 0, config)
    queue.calls.clear()
    return experiment, queue, config


def test_root_agent_is_always_evaluated_in_full(experiment, fake_benchmark, monkeypatch):
    queue = FakeQueue(monkeypatch, {0: 1.0})

    evaluate(experiment, 0, EvaluationConfig(stage1_size=4))

    assert [len(c["problems"]) for c in queue.calls] == [N_FAKE_PROBLEMS]


def test_good_agent_is_promoted_to_the_full_evaluation(parent_and_children):
    experiment, queue, config = parent_and_children

    evaluate(experiment, 1, config)

    dev, holdout = split_problems(FakeBenchmark().problems, "fake", 0.2, 7)
    stage1 = stage1_problems(dev, "fake", 4, 7)
    assert queue.calls == [
        {"agent": 1, "holdout": False, "problems": fake_problem_ids(stage1), "repeats": 1},
        {"agent": 1, "holdout": False, "problems": fake_problem_ids(dev), "repeats": 1},
        {"agent": 1, "holdout": True, "problems": fake_problem_ids(holdout), "repeats": 1},
    ]
    evaluation = runner.read_lineage(experiment / "agent_1")["evaluation"]
    assert evaluation["promoted"] is True
    assert evaluation["stage1_delta"] == {
        "mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n_problems": 4,
    }
    assert runner.get_pruned_iterations(experiment) == set()


def test_clearly_worse_agent_is_dropped_after_the_first_stage(parent_and_children):
    experiment, queue, config = parent_and_children

    evaluate(experiment, 2, config)

    # Four problems instead of ten, and no held-out problems at all
    assert [(c["holdout"], len(c["problems"])) for c in queue.calls] == [(False, 4)]
    evaluation = runner.read_lineage(experiment / "agent_2")["evaluation"]
    assert evaluation["promoted"] is False
    assert evaluation["stage1_delta"]["mean"] == -1.0
    assert runner.get_pruned_iterations(experiment) == {2}
    assert not (runner.holdout_dir(experiment) / "agent_2").exists()


def test_dropped_agent_is_not_built_upon(parent_and_children):
    experiment, queue, config = parent_and_children
    evaluate(experiment, 1, config)
    evaluate(experiment, 2, config)

    for seed in range(20):
        plan = runner.plan_generation(experiment, 1, 3, selection="thompson", seed=seed)
        assert all(c["parent_id"] in (0, 1) for c in plan)
        assert all("2" not in c["selection"]["weights"] for c in plan)


def test_promotion_is_decided_once(parent_and_children):
    experiment, queue, config = parent_and_children
    evaluate(experiment, 1, config)
    queue.calls.clear()

    # Resuming an evaluation that was interrupted after the decision
    evaluate(experiment, 1, config)
    assert [(c["holdout"], len(c["problems"])) for c in queue.calls] == [(False, 8), (True, 2)]

    queue.calls.clear()
    evaluate(experiment, 2, config)
    queue.calls.clear()
    evaluate(experiment, 2, config)
    assert queue.calls == []  # still dropped


def test_threshold_controls_how_readily_agents_are_dropped(parent_and_children):
    experiment, queue, config = parent_and_children
    lenient = EvaluationConfig(**{**config.to_dict(), "promote_threshold": -1.0})

    evaluate(experiment, 2, lenient)

    assert runner.read_lineage(experiment / "agent_2")["evaluation"]["promoted"] is True
    assert len(queue.calls) == 3


def test_search_passes_the_evaluation_config_on(experiment, fake_benchmark, monkeypatch):
    queue = FakeQueue(monkeypatch, {0: 1.0})

    asyncio.run(
        runner.run_search(
            1, experiment, 0, 2, config=EvaluationConfig(repeats=3)
        )
    )
    assert queue.calls == []  # no generations to run

    asyncio.run(
        runner.evaluate_pending_agents(1, experiment, 2, EvaluationConfig(repeats=3))
    )
    assert [c["repeats"] for c in queue.calls] == [3]
    assert runner.load_metadata(experiment)["pending_agents"] == []


# resolve_evaluation_config ----------------------------------------------------


def test_evaluation_config_defaults_and_is_stored(exp_dir):
    config = runner.resolve_evaluation_config(exp_dir, seed=11)

    assert config == EvaluationConfig(split_seed=11)
    assert runner.load_metadata(exp_dir)["evaluation"] == config.to_dict()


def test_evaluation_config_is_kept_when_resuming(exp_dir):
    runner.resolve_evaluation_config(
        exp_dir, seed=11, repeats=3, holdout_fraction=0.25, stage1_size=6
    )

    # Resumed without options, and with another selection seed
    config = runner.resolve_evaluation_config(exp_dir, seed=99)

    assert config == EvaluationConfig(
        repeats=3, holdout_fraction=0.25, stage1_size=6, split_seed=11
    )


def test_evaluation_settings_other_than_the_split_can_change(exp_dir):
    runner.resolve_evaluation_config(exp_dir, seed=11, repeats=1)
    (exp_dir / "agent_0" / "benchmarks").mkdir(parents=True)

    config = runner.resolve_evaluation_config(
        exp_dir, seed=11, repeats=4, stage1_size=5, promote_threshold=-0.2
    )

    assert (config.repeats, config.stage1_size, config.promote_threshold) == (4, 5, -0.2)
    assert runner.load_metadata(exp_dir)["evaluation"]["repeats"] == 4


def test_holdout_split_cannot_change_once_agents_are_evaluated(exp_dir):
    runner.resolve_evaluation_config(exp_dir, seed=11, holdout_fraction=0.25)
    assert runner.resolve_evaluation_config(exp_dir, 11, holdout_fraction=0.5).holdout_fraction == 0.5
    (exp_dir / "agent_0" / "benchmarks").mkdir(parents=True)

    with pytest.raises(ValueError, match="cannot be changed"):
        runner.resolve_evaluation_config(exp_dir, seed=11, holdout_fraction=0.25)
    # Nor can one be introduced in an experiment that was started without
    assert runner.resolve_evaluation_config(exp_dir, seed=11).holdout_fraction == 0.5


def test_legacy_experiment_cannot_get_a_holdout_split(exp_dir):
    (exp_dir / "agent_0" / "benchmarks").mkdir(parents=True)
    with pytest.raises(ValueError, match="cannot be changed"):
        runner.resolve_evaluation_config(exp_dir, seed=11, holdout_fraction=0.3)
    assert runner.resolve_evaluation_config(exp_dir, seed=11, repeats=2).repeats == 2


def test_invalid_evaluation_settings_are_rejected(exp_dir):
    with pytest.raises(ValueError):
        runner.resolve_evaluation_config(exp_dir, seed=11, repeats=0)


def test_evaluation_cli_arguments():
    parser = runner.setup_argparse()
    args = parser.parse_args([])
    assert (args.repeats, args.holdout_fraction, args.stage1_size, args.promote_threshold) == (
        None, None, None, None,
    )
    args = parser.parse_args(
        ["--repeats", "3", "--holdout-fraction", "0.3", "--stage1-size", "8",
         "--promote-threshold", "-0.1"]
    )
    assert (args.repeats, args.holdout_fraction, args.stage1_size, args.promote_threshold) == (
        3, 0.3, 8, -0.1,
    )


# Meta agent and meta model ----------------------------------------------------


def meta_agents(exp_dir, **kwargs):
    plan = runner.plan_generation(
        exp_dir, generation=2, n_children=6, selection="score_child_prop", seed=4, **kwargs
    )
    return [(c["parent_id"], c["meta_agent_id"]) for c in plan]


def test_meta_agent_is_the_best_agent_by_default(scored_experiment):
    pairs = meta_agents(scored_experiment)
    assert len({parent for parent, _ in pairs}) > 1  # parents vary
    assert {meta for _, meta in pairs} == {1}


def test_meta_agent_can_be_each_childs_parent(scored_experiment):
    pairs = meta_agents(scored_experiment, meta_agent="parent")
    assert len({parent for parent, _ in pairs}) > 1
    assert all(parent == meta for parent, meta in pairs)


def test_meta_agent_can_be_the_initial_agent(scored_experiment):
    assert {meta for _, meta in meta_agents(scored_experiment, meta_agent="seed")} == {0}


def test_meta_agent_can_be_a_fixed_agent(scored_experiment):
    assert {meta for _, meta in meta_agents(scored_experiment, meta_agent=2)} == {2}


def test_fixed_meta_agent_has_to_be_usable(scored_experiment):
    with pytest.raises(ValueError, match="does not exist"):
        runner.plan_generation(scored_experiment, 2, 1, meta_agent=9)

    runner.write_lineage(scored_experiment / "agent_2", status="failed")
    with pytest.raises(ValueError, match="not created successfully"):
        runner.plan_generation(scored_experiment, 2, 1, meta_agent=2)

    with pytest.raises(ValueError, match="Unknown meta agent"):
        runner.plan_generation(scored_experiment, 2, 1, meta_agent="strongest")


def test_meta_model_is_planned_and_recorded(scored_experiment, docker):
    (planned,) = runner.plan_generation(
        scored_experiment, 2, 1, meta_agent="seed", meta_model="GPT4O"
    )
    assert planned["meta_model"] == "GPT4O"

    record = create(scored_experiment, planned)

    assert (record["meta_agent_id"], record["meta_model"]) == (0, "GPT4O")
    (exec_cmd,) = [c for c in docker.calls_to("exec") if "improve" in c]
    # The override is an option of `docker exec`, so it comes before the container
    assert exec_cmd[:4] == ("docker", "exec", "-e", "AGENT_MODEL=GPT4O")
    assert exec_cmd[exec_cmd.index("-m") + 1] == "archive.agent_0.agent_code"


def test_no_model_override_by_default(experiment, docker):
    record = create(experiment)

    assert record["meta_model"] is None
    (exec_cmd,) = [c for c in docker.calls_to("exec") if "improve" in c]
    assert "-e" not in exec_cmd


def test_meta_model_does_not_reach_the_benchmarks(experiment, docker):
    run_queue(experiment, select_problems=lambda b: b.problems[:1])
    (exec_cmd,) = [c for c in docker.calls_to("exec") if "benchmark" in c]
    assert not any("AGENT_MODEL" in str(arg) for arg in exec_cmd)


def test_search_plans_with_the_given_meta_agent_and_model(experiment, monkeypatch):
    steps = FakeSteps(monkeypatch, experiment)

    asyncio.run(
        runner.run_search(
            1, experiment, 1, 2, n_children=2, meta_agent="parent", meta_model="GPT4O"
        )
    )

    assert steps.planned == [
        {"id": 1, "parent_id": 0, "meta_agent_id": 0, "meta_model": "GPT4O"},
        {"id": 2, "parent_id": 0, "meta_agent_id": 0, "meta_model": "GPT4O"},
    ]


def test_meta_cli_arguments():
    parser = runner.setup_argparse()
    args = parser.parse_args([])
    assert (args.meta_agent, args.meta_model) == ("best", None)
    assert parser.parse_args(["--meta-agent", "parent"]).meta_agent == "parent"
    assert parser.parse_args(["--meta-agent", "seed"]).meta_agent == "seed"
    assert parser.parse_args(["--meta-agent", "3"]).meta_agent == 3
    assert parser.parse_args(["--meta-model", "GPT4O"]).meta_model == "GPT4O"
    for bad in (["--meta-agent", "strongest"], ["--meta-agent", "-1"], ["--meta-model", "nope"]):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


# Validation gate ----------------------------------------------------------------

from search.gate import GateConfig


def test_new_agent_is_validated_in_its_own_image(experiment, docker):
    record = create(experiment)

    assert record["status"] == "ok"
    assert record["gate"]["passed"] is True
    assert [c["name"] for c in record["gate"]["checks"]] == ["import", "tests", "smoke"]
    assert record["gate"]["fix_attempts"] == 0
    assert record["tests_passed"] is True
    (gate_run,) = docker.gate_runs()
    assert "sica_sandbox:run1_agent1" in gate_run
    # The agent's code is mounted read-only, and copied before anything runs
    assert f"{experiment / 'agent_1' / 'agent_code'}:/mnt/agent_code:ro" in gate_run
    assert docker.commands_containing("cp -r /mnt/agent_code /home/agent/agent_code")
    assert not list((experiment / "agent_1" / "agent_code").glob("**/.pytest_cache"))
    # Both containers are removed
    assert len(docker.calls_to("rm")) == 2


@pytest.mark.parametrize(
    "broken, check",
    [
        ("import agent_code.agent", "import"),
        ("&& pytest", "tests"),
        ("agent_code benchmark", "smoke"),
    ],
)
def test_agent_that_cannot_run_is_marked_invalid(experiment, monkeypatch, broken, check):
    runner.write_lineage(experiment / "agent_0", tests_passed=True)
    docker = FakeDocker(fail_commands={broken: None})
    monkeypatch.setattr(runner, "run_docker_command", docker)

    record = create(experiment)

    assert record["status"] == "invalid"
    assert f"Validation gate failed at '{check}'" in record["failure_reason"]
    assert record["gate"]["passed"] is False
    assert record["gate"]["checks"][-1]["name"] == check
    assert "went wrong" in record["gate"]["checks"][-1]["output"]
    # It is not evaluated, and never built upon
    assert runner.get_failed_iterations(experiment) == {1}
    # What it cost to make is still accounted for
    assert record["usage"]["cost"] == 0.25


def test_invalid_agent_is_final(experiment, monkeypatch):
    runner.write_lineage(experiment / "agent_0", tests_passed=True)
    docker = FakeDocker(fail_commands={"import agent_code.agent": None})
    monkeypatch.setattr(runner, "run_docker_command", docker)
    create(experiment)

    with pytest.raises(ValueError, match="already been created"):
        create(experiment)


def test_failing_tests_inherited_from_the_parent_do_not_invalidate(experiment, monkeypatch):
    runner.write_lineage(experiment / "agent_0", tests_passed=False)
    docker = FakeDocker(fail_commands={"&& pytest": None})
    monkeypatch.setattr(runner, "run_docker_command", docker)

    record = create(experiment)

    assert record["status"] == "ok"
    assert record["tests_passed"] is False  # the standard for its own children
    tests_check = record["gate"]["checks"][1]
    assert tests_check["passed"] is True
    assert "already did for the parent" in tests_check["note"]
    assert len(docker.gate_runs()) == 1  # the parent's result was on record


def test_parents_tests_are_run_when_their_result_is_unknown(experiment, monkeypatch):
    # e.g. a parent from before the gate existed
    assert "tests_passed" not in runner.read_lineage(experiment / "agent_0")
    docker = FakeDocker(fail_commands={"&& pytest": None})
    monkeypatch.setattr(runner, "run_docker_command", docker)

    record = create(experiment)

    assert record["status"] == "ok"
    child_run, parent_run = docker.gate_runs()
    assert f"{experiment / 'agent_0' / 'agent_code'}:/mnt/agent_code:ro" in parent_run
    assert "sica_sandbox:run1_agent0" in parent_run  # in the parent's own image
    # Recorded, so that it is not run again for the next child
    assert runner.read_lineage(experiment / "agent_0")["tests_passed"] is False


def test_meta_agent_gets_to_fix_an_agent_that_fails_the_gate(experiment, monkeypatch):
    runner.write_lineage(experiment / "agent_0", tests_passed=True)
    docker = FakeDocker(fail_commands={"&& pytest": 1})  # fails once, then passes
    monkeypatch.setattr(runner, "run_docker_command", docker)

    record = create_with_gate(experiment, GateConfig(retries=2))

    assert record["status"] == "ok"
    assert record["gate"] == {**record["gate"], "passed": True, "fix_attempts": 1}
    (fix_cmd,) = docker.commands_containing("archive.agent_0.agent_code.agent")
    prompt = fix_cmd[fix_cmd.index("--prompt") + 1]
    assert "'tests'" in prompt and "pytest went wrong" in prompt
    assert fix_cmd[fix_cmd.index("--workdir") + 1] == "/home/agent/workdir"
    assert fix_cmd[fix_cmd.index("--logdir") + 1] == "/home/agent/meta_improvement_logs_1/gate_fix_1"
    # The fixed code is built and validated afresh
    assert len(docker.calls_to("buildx")) == 2
    assert len(docker.gate_runs()) == 2


def test_fix_attempts_are_limited(experiment, monkeypatch):
    runner.write_lineage(experiment / "agent_0", tests_passed=True)
    docker = FakeDocker(fail_commands={"&& pytest": None})
    monkeypatch.setattr(runner, "run_docker_command", docker)

    record = create_with_gate(experiment, GateConfig(retries=2))

    assert record["status"] == "invalid"
    assert record["gate"]["fix_attempts"] == 2
    assert len(docker.commands_containing("archive.agent_0.agent_code.agent")) == 2
    assert len(docker.gate_runs()) == 3


def test_fix_uses_the_meta_model(experiment, monkeypatch):
    runner.write_lineage(experiment / "agent_0", tests_passed=True)
    docker = FakeDocker(fail_commands={"&& pytest": 1})
    monkeypatch.setattr(runner, "run_docker_command", docker)

    create_with_gate(experiment, GateConfig(retries=1), child(meta_model="GPT4O"))

    (fix_cmd,) = docker.commands_containing("archive.agent_0.agent_code.agent")
    assert fix_cmd[:4] == ("docker", "exec", "-e", "AGENT_MODEL=GPT4O")


def test_gate_can_be_switched_off(experiment, docker):
    record = create_with_gate(experiment, GateConfig(checks=()))

    assert record["status"] == "ok"
    assert "gate" not in record
    assert docker.gate_runs() == []


def test_gate_container_failure_marks_the_agent_failed(experiment, monkeypatch):
    docker = FakeDocker(fail_commands={"cp -r /mnt/agent_code": None})
    monkeypatch.setattr(runner, "run_docker_command", docker)

    with pytest.raises(RuntimeError, match="Could not copy the agent code"):
        create(experiment)

    # Not 'invalid': the agent was never actually checked
    assert runner.read_lineage(experiment / "agent_1")["status"] == "failed"
    assert len(docker.calls_to("rm")) == 2


def test_gate_command_timeout_is_a_failure(experiment, docker, monkeypatch):
    async def run():
        async def slow(*args):
            if "exec" in args and "sleepy" in args:
                await asyncio.sleep(5)
            if "inspect" in args:
                return True, '"running"', ""
            return True, "", ""

        monkeypatch.setattr(runner, "run_docker_command", slow)
        async with runner.gate_container(1, 0, "image", experiment / "agent_0" / "agent_code") as exec:
            return await exec("sleepy", timeout=0.05)

    assert asyncio.run(run()) == (False, "", "Timed out after 0.05 seconds")


def create_with_gate(exp_dir, gate, planned=None):
    planned = planned or child()
    asyncio.run(runner.run_meta_agent_benchmark(1, exp_dir, planned, gate))
    return runner.read_lineage(exp_dir / f"agent_{planned['id']}")


# The initial agent


def check_initial(exp_dir, gate=GateConfig()):
    asyncio.run(runner.check_initial_agent(1, exp_dir, gate))
    return runner.read_lineage(exp_dir / "agent_0")


def test_initial_agent_is_checked_once(experiment, docker):
    record = check_initial(experiment)

    assert record["gate"]["passed"] is True
    assert record["tests_passed"] is True
    (gate_run,) = docker.gate_runs()
    assert f"{experiment / 'agent_0' / 'agent_code'}:/mnt/agent_code:ro" in gate_run

    check_initial(experiment)
    assert len(docker.gate_runs()) == 1


@pytest.mark.parametrize("broken", ["import agent_code.agent", "agent_code benchmark"])
def test_search_does_not_start_with_a_broken_setup(experiment, monkeypatch, broken):
    steps = FakeSteps(monkeypatch, experiment)
    monkeypatch.setattr(
        runner, "run_docker_command", FakeDocker(fail_commands={broken: None})
    )

    with pytest.raises(RuntimeError, match="initial agent does not pass the validation gate"):
        search(experiment, iterations=2)

    assert steps.events == []  # nothing was spent
    assert "gate" not in runner.read_lineage(experiment / "agent_0")


def test_failing_tests_of_the_initial_agent_set_the_standard(experiment, monkeypatch):
    monkeypatch.setattr(
        runner, "run_docker_command", FakeDocker(fail_commands={"&& pytest": None})
    )

    record = check_initial(experiment)

    assert record["gate"]["passed"] is True
    assert record["tests_passed"] is False


def test_initial_agent_is_not_checked_when_the_gate_is_off(experiment, docker):
    check_initial(experiment, GateConfig(checks=()))
    assert docker.calls == []


def test_search_passes_the_gate_config_on(experiment, monkeypatch):
    seen = []
    steps = FakeSteps(monkeypatch, experiment)

    async def create_recording(exp_id, exp_dir, planned, gate=None):
        seen.append(gate)
        await steps.create(exp_id, exp_dir, planned)

    monkeypatch.setattr(runner, "run_meta_agent_benchmark", create_recording)
    gate = GateConfig(checks=("import",), retries=3)

    asyncio.run(runner.run_search(1, experiment, 1, 2, gate=gate))

    assert seen == [gate]
    assert runner.read_lineage(experiment / "agent_0")["gate"]["checks"] == [
        {"name": "import", "passed": True, "command": "python -c 'import agent_code.agent'"}
    ]


def test_invalid_agents_are_not_evaluated(experiment, monkeypatch):
    steps = FakeSteps(monkeypatch, experiment)

    async def create(exp_id, exp_dir, planned, gate=None):
        status = "invalid" if planned["id"] == 2 else "ok"
        make_agent(exp_dir, planned["id"], planned["parent_id"], status=status)

    monkeypatch.setattr(runner, "run_meta_agent_benchmark", create)

    search(experiment, iterations=2, n_children=2)

    assert [e for e in steps.events if e[0] == "evaluate"] == [("evaluate", 0), ("evaluate", 1)]
    # And on resumption it counts as created: it is not made again
    assert runner.load_metadata(experiment)["generation"] == 2


def test_gate_cli_arguments():
    parser = runner.setup_argparse()
    args = parser.parse_args([])
    assert (args.gate_checks, args.gate_retries, args.gate_model) == ("import,tests,smoke", 0, None)
    args = parser.parse_args(
        ["--gate-checks", "import,smoke", "--gate-retries", "2", "--gate-model", "GPT4O_MINI"]
    )
    assert GateConfig.parse_checks(args.gate_checks) == ("import", "smoke")
    assert (args.gate_retries, args.gate_model) == (2, "GPT4O_MINI")
    assert GateConfig.parse_checks(parser.parse_args(["--gate-checks", "none"]).gate_checks) == ()


# Credentials ------------------------------------------------------------------

REPO_ROOT = Path(runner.__file__).parent
SECRET = "sk-test-secret-value"


@pytest.fixture
def api_keys(monkeypatch):
    for name in runner.SANDBOX_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("VERTEX_REGION", "europe-west1")
    monkeypatch.setenv("OPENAI_API_KEY", "")  # set, but empty


def test_secrets_are_passed_by_name_only(experiment, api_keys):
    args = runner.sandbox_secret_args()

    assert args == ["-e", "ANTHROPIC_API_KEY", "-e", "VERTEX_REGION"]
    assert SECRET not in " ".join(args)


def test_google_credentials_are_mounted_read_only(experiment, api_keys):
    credentials = Path("sandbox/GOOGLE_APPLICATION_CREDENTIALS.json")
    credentials.parent.mkdir()
    credentials.write_text("{}")

    args = runner.sandbox_secret_args()

    assert args[-4:] == [
        "-v",
        f"{credentials.absolute()}:/tmp/GOOGLE_APPLICATION_CREDENTIALS.json:ro",
        "-e",
        "GOOGLE_APPLICATION_CREDENTIALS=/tmp/GOOGLE_APPLICATION_CREDENTIALS.json",
    ]


def test_every_container_gets_the_credentials_and_no_command_holds_a_value(
    experiment, docker, api_keys, fake_benchmark
):
    create(experiment)  # meta-improvement, image build, validation gate
    run_queue(experiment, select_problems=lambda b: b.problems[:1])  # a benchmark

    runs = docker.calls_to("run")
    assert len(runs) == 3
    for run_cmd in runs:
        position = run_cmd.index("ANTHROPIC_API_KEY")
        assert run_cmd[position - 1] == "-e"
        # Options have to come before the image name
        assert position < run_cmd.index("tail") - 1
    for call in docker.calls:
        assert SECRET not in " ".join(map(str, call))


def test_image_build_takes_no_credentials(experiment, docker, api_keys):
    create(experiment)

    (build_cmd,) = docker.calls_to("buildx")
    build_args = [build_cmd[i + 1] for i, arg in enumerate(build_cmd) if arg == "--build-arg"]
    assert len(build_args) == 1 and build_args[0].startswith("TARGET_ARCH=")


def test_no_credentials_in_the_image_definition():
    dockerfile = (REPO_ROOT / "sandbox" / "Dockerfile").read_text()
    instructions = [
        line for line in dockerfile.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    for line in instructions:
        assert "API_KEY" not in line
        assert "VERTEX_PROJECT_ID" not in line
        assert "GOOGLE_APPLICATION_CREDENTIALS" not in line

    makefile = (REPO_ROOT / "Makefile").read_text()
    for line in makefile.splitlines():
        if "--build-arg" in line:
            assert "TARGET_ARCH" in line
    # Each container that the Makefile starts is given the credentials
    assert makefile.count("docker run") == makefile.count("$(SANDBOX_SECRETS)") == 3
