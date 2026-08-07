"""Finalist / DES selection correctness for the native planner redesign."""

from __future__ import annotations

import itertools
import random
from typing import Any
from unittest.mock import MagicMock

import pytest

from tensortorrent.compile.specialize import (
    _prefetch_variants,
    _recompute_winner_metadata,
    _scores_near_equal,
    _select_des_winner,
    _select_finalist_by_simulation,
)
from tensortorrent.config import CompileConfig, Objective
from tensortorrent.ir.resource_graph import ResourceDecision
from tensortorrent.native import native_available
from tensortorrent.planner.maximal import ExecutionPlan, Placement
from tensortorrent.runtime.simulator.discrete_event import SimulationResult

# Pure Python helpers (_select_des_winner, _prefetch_variants, …) always run.
# Only tests that call into the Rust extension are gated.
requires_native = pytest.mark.skipif(not native_available(), reason="native extension required")


def _plan(
    *,
    rank: int,
    devices: tuple[str, ...],
    placements: list[Placement],
    latency: float = 0.1,
    prefetch: int = 1,
    signature: str = "",
) -> ExecutionPlan:
    return ExecutionPlan(
        graph_name="g",
        fingerprint="fp",
        objective="latency",
        placements=placements,
        decisions=[ResourceDecision(resource=d, selected=True, reason=f"analytic {d}") for d in devices],
        devices_used=devices,
        communication_backend="host",
        predicted_latency_s=latency,
        predicted_peak_bytes={d: 100 for d in devices},
        predicted_throughput_per_s=10.0,
        prefetch_distance=prefetch,
        strategy="single_gpu" if len(devices) == 1 else "multi_gpu",
        search_statistics={
            "search_rank": rank,
            "analytic_rank": rank,
            "finalist_rank": rank,
            "placement_signature": signature
            or "|".join(f"{p.region_id}:{p.device}:{p.backend_id}:{p.kernel_id}:{p.dtype}" for p in placements),
            "planner_engine": "rust",
            "host_staged_transfer_count": 1,
        },
        notes=[f"analytic_rank={rank}"],
    )


def _placement(region: str, device: str, kernel: str = "k") -> Placement:
    return Placement(
        region_id=region,
        device=device,
        backend_id="mock",
        dtype="float32",
        kernel_id=kernel,
        estimated_latency_s=0.01,
    )


def _sim(*, makespan: float, peak: dict[str, int] | None = None) -> SimulationResult:
    return SimulationResult(
        makespan_s=makespan,
        peak_bytes=peak or {"vram": 100},
        timeline=[],
        exposed_transfer_latency_s=0.0,
        resource_busy_s={"gpu": makespan},
        initiation_interval_s=makespan,
    )


def _stub_schedule_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tensortorrent.runtime.residency.attach_residency_to_plan",
        lambda plan, program: MagicMock(),
    )
    for name in (
        "schedule_matches_plan",
        "validate_schedule",
        "validate_schedule_resources",
        "validate_schedule_tensor_sizes",
    ):
        monkeypatch.setattr(f"tensortorrent.runtime.schedule.{name}", lambda *a, **k: [])


def _patch_batch_sim(monkeypatch: pytest.MonkeyPatch, fake_sim: Any) -> None:
    """Patch DES batch API used by specialize (returns outcomes + Rust-style stats)."""

    def wrapped(schedules: list[Any], machine: Any, workers: int = 0) -> tuple[list[Any], dict[str, Any]]:
        outs = fake_sim(schedules, machine, workers=workers)
        n = len(schedules)
        # Match Rust explicit-worker semantics; auto (0) stays serial in fakes
        # (no real instruction-work signal on mock schedules).
        parallel = int(workers) > 1 and n > 1
        used = 1
        if parallel:
            used = max(1, min(int(workers), n))
        return outs, {
            "schedule_count": n,
            "simulator_workers_requested": int(workers),
            "simulator_workers_available": used if parallel else 1,
            "simulator_workers_used": used,
            "parallel_simulation_used": parallel,
        }

    monkeypatch.setattr(
        "tensortorrent.runtime.simulator.discrete_event.simulate_schedules_with_stats",
        wrapped,
    )


def test_prefetch_variants_include_zero_when_streaming() -> None:
    assert 0 in _prefetch_variants(2, streaming=True)
    # Primary analytic estimate first, then exploratory alternatives.
    assert _prefetch_variants(2, streaming=True) == [2, 0, 1, 3]
    assert _prefetch_variants(5, streaming=True) == [5, 0, 1, 4, 6]
    # Estimate 0 = hard disable: do not explore positives.
    assert _prefetch_variants(0, streaming=True) == [0]


def test_des_steady_state_hoist_drops_parameter_h2d_and_evict() -> None:
    from tensortorrent.ir.graph import OpCode
    from tensortorrent.runtime.schedule import (
        ExecutableSchedule,
        PlanInstruction,
        hoist_resident_parameter_transfers,
    )

    sched = ExecutableSchedule(
        graph_name="g",
        fingerprint="f",
        instructions=(
            PlanInstruction(
                opcode=OpCode.TRANSFER,
                name="transfer::state::r0->gpu",
                resource="copy_engine",
                source="cpu",
                destination="gpu",
                nbytes=1000,
                attributes={"kind": "parameter_host_to_device"},
            ),
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="compute::r0",
                resource="gpu",
                depends_on=("transfer::state::r0->gpu",),
                predicted_duration_s=0.01,
            ),
            PlanInstruction(
                opcode=OpCode.EVICT,
                name="evict::state::r0",
                resource="gpu",
                depends_on=("compute::r0",),
                attributes={"kind": "parameter_evict"},
            ),
        ),
    )
    des = hoist_resident_parameter_transfers(sched, drop_parameter_evicts=True)
    names = {i.name for i in des.instructions}
    assert "transfer::state::r0->gpu" not in names
    assert "evict::state::r0" not in names
    assert "compute::r0" in names
    compute = next(i for i in des.instructions if i.name == "compute::r0")
    assert compute.depends_on == ()
    assert _prefetch_variants(0, streaming=True) == [0]
    assert _prefetch_variants(5, streaming=False) == [0]


def test_fair_des_budget_gives_each_finalist_a_primary_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Candidate 0's many prefetch variants must not starve candidate 1."""
    builds: list[tuple[int, int]] = []
    plan_a = _plan(
        rank=0,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0", "a")],
        prefetch=4,
        signature="a",
    )
    plan_b = _plan(
        rank=1,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0", "b")],
        prefetch=4,
        signature="b",
    )
    _stub_schedule_validators(monkeypatch)

    def fake_build(plan: ExecutionPlan, residency: Any, **kwargs: Any) -> Any:
        prefetch = int(kwargs.get("prefetch_distance") or 0)
        rank = int(plan.search_statistics["search_rank"])
        builds.append((rank, prefetch))
        return MagicMock(name=f"sched-{rank}-{prefetch}", instructions=[])

    monkeypatch.setattr("tensortorrent.runtime.schedule.build_executable_schedule", fake_build)

    def fake_sim(schedules: list[Any], machine: Any, workers: int = 0) -> list[Any]:
        out = []
        for sched in schedules:
            name = getattr(sched, "_mock_name", "") or str(sched)
            out.append(_sim(makespan=0.05 if "sched-1-" in name else 0.20))
        return out

    _patch_batch_sim(monkeypatch, fake_sim)
    monkeypatch.setattr("tensortorrent.planner.maximal._eligible_compute", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.planner.maximal._decide_resources", lambda *a, **k: [])
    monkeypatch.setattr(
        "tensortorrent.backends.communication.select_communication_backend",
        lambda devices: MagicMock(backend_id="host"),
    )

    machine = MagicMock()
    machine.memory = {}
    machine.compute = {}
    cfg = CompileConfig(planner_des_candidates=2, planner_workers=1)
    win, _, _, _, stats = _select_finalist_by_simulation(
        [plan_a, plan_b],
        program=None,
        streaming=True,
        activation_budget_bytes=None,
        machine=machine,
        config=cfg,
    )
    assert {r for r, _ in builds} == {0, 1}, builds
    # Breadth-first: first two builds are round-0 (analytic primary) for each finalist.
    assert {builds[0][0], builds[1][0]} == {0, 1}
    assert builds[0][1] == builds[1][1] == 4  # estimated primary before exploratories
    assert stats["winning_analytic_rank"] == 1
    assert win.search_statistics["placement_signature"] == "b"


def test_des_same_subset_overturns_analytic_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    plan_a = _plan(
        rank=0,
        devices=("gpu0", "gpu1"),
        placements=[_placement("r0", "gpu0", "fast"), _placement("r1", "gpu0", "fast")],
        latency=0.10,
        signature="colocated",
    )
    plan_b = _plan(
        rank=1,
        devices=("gpu0", "gpu1"),
        placements=[_placement("r0", "gpu0", "split"), _placement("r1", "gpu1", "split")],
        latency=0.11,
        signature="split",
    )
    _stub_schedule_validators(monkeypatch)
    monkeypatch.setattr(
        "tensortorrent.runtime.schedule.build_executable_schedule",
        lambda plan, *a, **k: MagicMock(instructions=[], _sig=plan.search_statistics["placement_signature"]),
    )

    def fake_sim(schedules: list[Any], machine: Any, workers: int = 0) -> list[Any]:
        return [_sim(makespan=0.05 if getattr(s, "_sig", "") == "split" else 0.20) for s in schedules]

    _patch_batch_sim(monkeypatch, fake_sim)
    monkeypatch.setattr("tensortorrent.planner.maximal._eligible_compute", lambda *a, **k: [])
    monkeypatch.setattr(
        "tensortorrent.planner.maximal._decide_resources",
        lambda *a, **k: [
            ResourceDecision(resource="gpu0", selected=True, reason="des"),
            ResourceDecision(resource="gpu1", selected=True, reason="des"),
            ResourceDecision(resource="cpu_0", selected=False, reason="des exclude"),
        ],
    )
    monkeypatch.setattr(
        "tensortorrent.backends.communication.select_communication_backend",
        lambda devices: MagicMock(backend_id="nccl"),
    )

    machine = MagicMock()
    machine.memory = {}
    machine.compute = {}
    cfg = CompileConfig(planner_des_candidates=4, planner_workers=1, objective=Objective.LATENCY)
    win, _, _, _, stats = _select_finalist_by_simulation(
        [plan_a, plan_b],
        program=None,
        streaming=False,
        activation_budget_bytes=None,
        machine=machine,
        config=cfg,
    )
    assert win.search_statistics["placement_signature"] == "split"
    assert stats["winning_analytic_rank"] == 1
    assert stats.get("simulator_changed_winner")
    assert {d.resource for d in win.decisions if d.selected} == {"gpu0", "gpu1"}


def test_winner_metadata_recomputed_for_des_winner() -> None:
    analytic = _plan(
        rank=0,
        devices=("cpu_0", "gpu0"),
        placements=[_placement("r0", "cpu_0"), _placement("r1", "gpu0")],
        signature="cpu+gpu",
    )
    des_win = _plan(
        rank=1,
        devices=("gpu0", "gpu1"),
        placements=[_placement("r0", "gpu0"), _placement("r1", "gpu1")],
        signature="2gpu",
    )
    des_win.decisions = list(analytic.decisions)
    machine = MagicMock()
    machine.memory = {}
    machine.compute = {}
    from tensortorrent.planner import maximal as maximal_mod

    decided = [
        ResourceDecision(resource="gpu0", selected=True, reason="winner"),
        ResourceDecision(resource="gpu1", selected=True, reason="winner"),
        ResourceDecision(resource="cpu_0", selected=False, reason="not used"),
    ]
    original_eligible = maximal_mod._eligible_compute
    original_decide = maximal_mod._decide_resources
    maximal_mod._eligible_compute = lambda m, c: []  # type: ignore[assignment]
    maximal_mod._decide_resources = lambda *a, **k: decided  # type: ignore[assignment]
    try:
        _recompute_winner_metadata(
            des_win,
            [analytic, des_win],
            machine=machine,
            config=CompileConfig(),
            prefetch=0,
            sim=_sim(makespan=0.07),
        )
    finally:
        maximal_mod._eligible_compute = original_eligible
        maximal_mod._decide_resources = original_decide

    assert des_win.devices_used == ("gpu0", "gpu1")
    assert des_win.prefetch_distance == 0
    assert {d.resource for d in des_win.decisions if d.selected} == {"gpu0", "gpu1"}
    assert not any(d.resource == "cpu_0" and d.selected for d in des_win.decisions)


def test_pageable_used_when_pinned_des_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(
        rank=0,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0")],
        prefetch=1,
        signature="p",
    )
    _stub_schedule_validators(monkeypatch)

    def fake_build(*args: Any, **kwargs: Any) -> Any:
        pageable = bool(kwargs.get("force_pageable_host_staging"))
        return MagicMock(instructions=[], pageable=pageable)

    monkeypatch.setattr("tensortorrent.runtime.schedule.build_executable_schedule", fake_build)
    calls = {"n": 0}

    def fake_sim(schedules: list[Any], machine: Any, workers: int = 0) -> list[Any]:
        calls["n"] += 1
        out = []
        for sched in schedules:
            if getattr(sched, "pageable", False):
                out.append(_sim(makespan=0.08, peak={"host_ram": 10}))
            else:
                out.append({"status": "infeasible", "error": "pinned host memory exceeded"})
        return out

    _patch_batch_sim(monkeypatch, fake_sim)
    monkeypatch.setattr("tensortorrent.planner.maximal._eligible_compute", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.planner.maximal._decide_resources", lambda *a, **k: [])
    monkeypatch.setattr(
        "tensortorrent.backends.communication.select_communication_backend",
        lambda devices: MagicMock(backend_id="host"),
    )

    machine = MagicMock()
    machine.memory = {}
    machine.compute = {}
    cfg = CompileConfig(
        planner_des_candidates=2,
        planner_workers=1,
        allow_host_staged_transfers=True,
    )
    win, _, _, _, stats = _select_finalist_by_simulation(
        [plan],
        program=None,
        streaming=True,
        activation_budget_bytes=None,
        machine=machine,
        config=cfg,
    )
    assert calls["n"] >= 2
    assert stats["schedule_variants_simulated"] >= 2
    assert any("pageable" in n for n in win.notes)


def test_memory_objective_can_prefer_prefetch_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(
        rank=0,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0")],
        prefetch=2,
        signature="p",
    )
    _stub_schedule_validators(monkeypatch)

    def fake_build(plan: ExecutionPlan, residency: Any, **kwargs: Any) -> Any:
        pref = int(kwargs.get("prefetch_distance") or 0)
        return MagicMock(instructions=[], pref=pref)

    monkeypatch.setattr("tensortorrent.runtime.schedule.build_executable_schedule", fake_build)

    def fake_sim(schedules: list[Any], machine: Any, workers: int = 0) -> list[Any]:
        out = []
        for sched in schedules:
            pref = int(getattr(sched, "pref", 1))
            # Higher prefetch → higher peak memory.
            out.append(_sim(makespan=0.1 + 0.001 * pref, peak={"vram": 1000 + 500 * pref}))
        return out

    _patch_batch_sim(monkeypatch, fake_sim)
    monkeypatch.setattr("tensortorrent.planner.maximal._eligible_compute", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.planner.maximal._decide_resources", lambda *a, **k: [])
    monkeypatch.setattr(
        "tensortorrent.backends.communication.select_communication_backend",
        lambda devices: MagicMock(backend_id="host"),
    )
    machine = MagicMock()
    machine.memory = {"vram": MagicMock(allocatable_bytes=10_000, memory_class=MagicMock(value="device_vram"))}
    machine.compute = {}
    cfg = CompileConfig(
        planner_des_candidates=4,
        planner_workers=1,
        objective=Objective.MEMORY,
    )
    _, _, _, pref, _ = _select_finalist_by_simulation(
        [plan],
        program=None,
        streaming=True,
        activation_budget_bytes=None,
        machine=machine,
        config=cfg,
    )
    assert pref == 0


def test_compile_only_des_winner_placements(monkeypatch: pytest.MonkeyPatch) -> None:
    """specialize_for_machine compiles only the DES-selected plan's placements."""
    import importlib

    from tensortorrent.compile import specialize as spec
    from tensortorrent.compile.concurrency import ConcurrencyDecision

    fit_mod = importlib.import_module("tensortorrent.compile.fit")

    plan_a = _plan(
        rank=0,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0", "a")],
        signature="a",
    )
    plan_b = _plan(
        rank=1,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0", "b")],
        signature="b",
    )
    plan_a.finalist_plans = [plan_a, plan_b]
    compile_calls: list[str] = []

    monkeypatch.setattr(spec, "plan_execution", lambda *a, **k: plan_a)
    monkeypatch.setattr(fit_mod, "needs_parameter_streaming", lambda *a, **k: False)
    monkeypatch.setattr(
        "tensortorrent.planner.local_search.refine_prefetch_distance",
        lambda plan, **k: plan,
    )

    def fake_select(finalists, **kwargs):  # type: ignore[no-untyped-def]
        win = finalists[1]
        return (
            win,
            MagicMock(instructions=[], transfer_ops=lambda: []),
            _sim(makespan=0.05),
            0,
            {
                "batch_simulation_s": 0.0,
                "schedule_build_s": 0.0,
                "finalist_selection_s": 0.0,
            },
        )

    monkeypatch.setattr(spec, "_select_finalist_by_simulation", fake_select)
    monkeypatch.setattr(
        "tensortorrent.runtime.residency.attach_residency_to_plan",
        lambda plan, program: MagicMock(as_dict=lambda: {}, transfers=[]),
    )

    def fake_compile(plan, **kwargs):  # type: ignore[no-untyped-def]
        compile_calls.append("|".join(str(p.kernel_id) for p in plan.placements))
        return [], {}

    monkeypatch.setattr(spec, "_compile_plan_placements", fake_compile)
    monkeypatch.setattr("tensortorrent.planner.collectives.plan_collectives", lambda *a, **k: [])
    monkeypatch.setattr(
        "tensortorrent.planner.cost.calibration.runtime_predicted_makespan_s",
        lambda m, **k: float(m),
    )
    monkeypatch.setattr(spec, "_planning_storage_bandwidth", lambda machine: None)
    monkeypatch.setattr(
        spec,
        "_decide_concurrency",
        lambda *a, **k: ConcurrencyDecision(enabled=False, workers=1, reason="test"),
    )

    portable = MagicMock()
    portable.ir = MagicMock()
    portable.ir.name = "g"
    portable.fingerprint = "fp"
    portable.program = None  # skip region-coverage / passthrough
    portable.metadata = {}
    machine = MagicMock()
    machine.fingerprint = "m"
    machine.compute = {}
    machine.memory = {}
    machine.links = {}

    art = spec.specialize_for_machine(
        portable,
        config=CompileConfig(measure_regions=False, use_torch_compile=False),
        machine=machine,
        compile_regions=True,
    )
    assert art.plan is plan_b
    assert compile_calls == ["b"]
    assert "a" not in compile_calls[0]


def test_positive_prefetch_wins_latency_when_overlap_helps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Useful positive prefetch must beat prefetch=0 on latency when DES says so."""
    plan = _plan(
        rank=0,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0")],
        prefetch=2,
        signature="p",
    )
    _stub_schedule_validators(monkeypatch)

    def fake_build(plan: ExecutionPlan, residency: Any, **kwargs: Any) -> Any:
        pref = int(kwargs.get("prefetch_distance") or 0)
        return MagicMock(instructions=[], pref=pref)

    monkeypatch.setattr("tensortorrent.runtime.schedule.build_executable_schedule", fake_build)

    def fake_sim(schedules: list[Any], machine: Any, workers: int = 0) -> list[Any]:
        out = []
        for sched in schedules:
            pref = int(getattr(sched, "pref", 1))
            # Prefetch=2 hides transfer; 0 exposes it.
            out.append(_sim(makespan=0.05 if pref == 2 else 0.40))
        return out

    _patch_batch_sim(monkeypatch, fake_sim)
    monkeypatch.setattr("tensortorrent.planner.maximal._eligible_compute", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.planner.maximal._decide_resources", lambda *a, **k: [])
    monkeypatch.setattr(
        "tensortorrent.backends.communication.select_communication_backend",
        lambda devices: MagicMock(backend_id="host"),
    )
    machine = MagicMock()
    machine.memory = {}
    machine.compute = {}
    cfg = CompileConfig(
        planner_des_candidates=4,
        planner_workers=1,
        objective=Objective.LATENCY,
        prefetch_distance=2,
    )
    _, _, _, pref, _ = _select_finalist_by_simulation(
        [plan],
        program=None,
        streaming=True,
        activation_budget_bytes=None,
        machine=machine,
        config=cfg,
    )
    assert pref == 2


def test_latency_prefetch_zero_wins_when_clearly_better(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefetch=0 remains selectable for latency when DES makespan wins by a margin."""
    plan = _plan(
        rank=0,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0")],
        prefetch=2,
        signature="p",
    )
    _stub_schedule_validators(monkeypatch)

    def fake_build(plan: ExecutionPlan, residency: Any, **kwargs: Any) -> Any:
        pref = int(kwargs.get("prefetch_distance") or 0)
        return MagicMock(instructions=[], pref=pref)

    monkeypatch.setattr("tensortorrent.runtime.schedule.build_executable_schedule", fake_build)

    def fake_sim(schedules: list[Any], machine: Any, workers: int = 0) -> list[Any]:
        out = []
        for sched in schedules:
            pref = int(getattr(sched, "pref", 1))
            # Prefetch=0 dramatically faster (>> margin).
            out.append(_sim(makespan=0.01 if pref == 0 else 0.50))
        return out

    _patch_batch_sim(monkeypatch, fake_sim)
    monkeypatch.setattr("tensortorrent.planner.maximal._eligible_compute", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.planner.maximal._decide_resources", lambda *a, **k: [])
    monkeypatch.setattr(
        "tensortorrent.backends.communication.select_communication_backend",
        lambda devices: MagicMock(backend_id="host"),
    )
    machine = MagicMock()
    machine.memory = {}
    machine.compute = {}
    cfg = CompileConfig(
        planner_des_candidates=4,
        planner_workers=1,
        objective=Objective.LATENCY,
        prefetch_distance=2,
    )
    _, _, _, pref, _ = _select_finalist_by_simulation(
        [plan],
        program=None,
        streaming=True,
        activation_budget_bytes=None,
        machine=machine,
        config=cfg,
    )
    assert pref == 0


@requires_native
def test_batch_des_error_isolation() -> None:
    from tensortorrent.ir.graph import OpCode
    from tensortorrent.ir.resource_graph import (
        ComputeClass,
        ComputeResource,
        MemoryClass,
        MemoryResource,
        ResourceGraph,
        ResourceId,
        ResourceKind,
    )
    from tensortorrent.runtime.schedule import ExecutableSchedule, PlanInstruction
    from tensortorrent.runtime.simulator.discrete_event import simulate_schedules

    machine = ResourceGraph(fingerprint="iso", backends_present=("cpu",))
    machine.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "host_ram"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=1 << 20,
            allocatable_bytes=64,
            attached_compute=("cpu0",),
        )
    )
    machine.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "cpu0"),
            compute_class=ComputeClass.CPU_NUMA_POOL,
            backend_id="cpu",
            vendor="test",
            model="cpu0",
            memory_affinity=("host_ram",),
            supported_dtypes=("float32",),
        )
    )
    ok = ExecutableSchedule(
        graph_name="g",
        fingerprint="fp",
        instructions=(
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="c0",
                resource="cpu0",
                outputs=("o0",),
                nbytes=8,
                predicted_duration_s=0.01,
            ),
        ),
    )
    bad = ExecutableSchedule(
        graph_name="g",
        fingerprint="fp",
        instructions=(
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="c1",
                resource="cpu0",
                outputs=("o1",),
                nbytes=1 << 30,
                predicted_duration_s=0.01,
            ),
        ),
    )
    batch = simulate_schedules([ok, bad, ok], machine, workers=1)
    assert isinstance(batch[0], SimulationResult)
    assert isinstance(batch[2], SimulationResult)
    assert isinstance(batch[1], dict) or (isinstance(batch[1], SimulationResult) and batch[1].makespan_s >= 0)


@requires_native
def test_native_same_subset_multiple_finalists() -> None:
    from tensortorrent.native import require_native

    native = require_native()
    # Two devices, two regions, rich candidate pools → multiple terminals.
    problem = {
        "config": {
            "objective": "latency",
            "beam_width": 16,
            "candidates_per_device": 2,
            "local_search_iters": 1,
            "planner_workers": 1,
            "allow_parallel_subsets": False,
            "finalist_count": 6,
            "per_subset_finalists": 4,
            "allow_host_staged_transfers": True,
            "target_inflight_requests": 1,
        },
        "device_names": ["accel_0", "accel_1"],
        "capacities": [100_000, 100_000],
        "device_memory": ["vram_0", "vram_1"],
        "regions": [
            {
                "name": "r0",
                "depends_on": [],
                "output_bytes": 1000,
                "state_bytes": 0,
                "consumer_count": 1,
            },
            {
                "name": "r1",
                "depends_on": [0],
                "output_bytes": 4,
                "state_bytes": 0,
                "consumer_count": 0,
            },
        ],
        "order": [0, 1],
        "candidates": [
            [
                {
                    "device": 0,
                    "backend_id": "mock",
                    "kernel_id": "r0:a0",
                    "dtype": "float32",
                    "estimated_latency_s": 0.01,
                    "workspace_bytes": 0,
                    "measured": True,
                },
                {
                    "device": 1,
                    "backend_id": "mock",
                    "kernel_id": "r0:a1",
                    "dtype": "float32",
                    "estimated_latency_s": 0.03,
                    "workspace_bytes": 0,
                    "measured": True,
                },
            ],
            [
                {
                    "device": 0,
                    "backend_id": "mock",
                    "kernel_id": "r1:a0",
                    "dtype": "float32",
                    "estimated_latency_s": 0.02,
                    "workspace_bytes": 0,
                    "measured": True,
                },
                {
                    "device": 1,
                    "backend_id": "mock",
                    "kernel_id": "r1:a1",
                    "dtype": "float32",
                    "estimated_latency_s": 0.015,
                    "workspace_bytes": 0,
                    "measured": True,
                },
            ],
        ],
        "edge_bytes": [(0, 1, 1000)],
        "subsets": [{"device_indices": [0, 1]}],
        "machine": None,
    }
    # machine_from_py needs a real object; use ResourceGraph.
    from tensortorrent.ir.resource_graph import (
        ComputeClass,
        ComputeResource,
        LinkClass,
        MemoryClass,
        MemoryResource,
        ResourceGraph,
        ResourceId,
        ResourceKind,
        TransferLink,
    )

    machine = ResourceGraph(fingerprint="subset-multi", backends_present=("mock",))
    for i, name in enumerate(("accel_0", "accel_1")):
        machine.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, f"vram_{i}"),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=100_000,
                allocatable_bytes=100_000,
                attached_compute=(name,),
            )
        )
        machine.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, name),
                compute_class=ComputeClass.ACCELERATOR,
                backend_id="virtual",
                vendor="test",
                model=name,
                memory_affinity=(f"vram_{i}",),
                supported_dtypes=("float32",),
            )
        )
    machine.add_link(
        TransferLink(
            id=ResourceId(ResourceKind.LINK, "vram_0->vram_1"),
            link_class=LinkClass.PCIE,
            source="vram_0",
            destination="vram_1",
            bidirectional=True,
            measured=True,
            latency_s=0.0,
            bytes_per_s=1e6,
        )
    )
    problem["machine"] = machine
    out = native.plan_placements(problem)
    finalists = list(out.get("finalists") or [])
    assert len(finalists) >= 2, finalists
    sigs = {f.get("placement_signature") for f in finalists}
    assert len(sigs) >= 2
    for f in finalists:
        assert list(f.get("subset_devices") or []) == ["accel_0", "accel_1"]


def test_pageable_fallback_when_pinned_fills_des_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed pinned A still gets pageable recovery after cap fills; A pageable wins."""
    plan_a = _plan(
        rank=0,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0", "a")],
        prefetch=4,
        signature="a",
    )
    plan_b = _plan(
        rank=1,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0", "b")],
        prefetch=4,
        signature="b",
    )
    _stub_schedule_validators(monkeypatch)

    def fake_build(plan: ExecutionPlan, residency: Any, **kwargs: Any) -> Any:
        pageable = bool(kwargs.get("force_pageable_host_staging"))
        pref = int(kwargs.get("prefetch_distance") or 0)
        sig = plan.search_statistics["placement_signature"]
        return MagicMock(instructions=[], pageable=pageable, pref=pref, _sig=sig)

    monkeypatch.setattr("tensortorrent.runtime.schedule.build_executable_schedule", fake_build)

    def fake_sim(schedules: list[Any], machine: Any, workers: int = 0) -> list[Any]:
        out = []
        for sched in schedules:
            sig = getattr(sched, "_sig", "")
            pageable = bool(getattr(sched, "pageable", False))
            if sig == "a" and not pageable:
                out.append({"status": "infeasible", "error": "pinned host memory exceeded"})
            elif sig == "a" and pageable:
                out.append(_sim(makespan=0.02))  # fast pageable A
            else:
                out.append(_sim(makespan=0.15))  # slower feasible B
        return out

    _patch_batch_sim(monkeypatch, fake_sim)
    monkeypatch.setattr("tensortorrent.planner.maximal._eligible_compute", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.planner.maximal._decide_resources", lambda *a, **k: [])
    monkeypatch.setattr(
        "tensortorrent.backends.communication.select_communication_backend",
        lambda devices: MagicMock(backend_id="host"),
    )
    machine = MagicMock()
    machine.memory = {}
    machine.compute = {}
    # cap = 2*3 = 6 → BFS fills with pinned prefs; pageable must still recover.
    cfg = CompileConfig(
        planner_des_candidates=2,
        planner_workers=1,
        allow_host_staged_transfers=True,
        objective=Objective.LATENCY,
    )
    win, _, _, _, stats = _select_finalist_by_simulation(
        [plan_a, plan_b],
        program=None,
        streaming=True,
        activation_budget_bytes=None,
        machine=machine,
        config=cfg,
    )
    assert win.search_statistics["placement_signature"] == "a"
    assert any("pageable" in n for n in win.notes)
    assert stats["winning_analytic_rank"] == 0


def test_des_prefetch_zero_wins_without_artificial_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    """DES-faster prefetch=0 must win even when margin is small (no 5% penalty)."""
    plan = _plan(
        rank=0,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0")],
        prefetch=2,
        signature="p",
    )
    _stub_schedule_validators(monkeypatch)

    def fake_build(plan: ExecutionPlan, residency: Any, **kwargs: Any) -> Any:
        pref = int(kwargs.get("prefetch_distance") or 0)
        return MagicMock(instructions=[], pref=pref)

    monkeypatch.setattr("tensortorrent.runtime.schedule.build_executable_schedule", fake_build)

    def fake_sim(schedules: list[Any], machine: Any, workers: int = 0) -> list[Any]:
        out = []
        for sched in schedules:
            pref = int(getattr(sched, "pref", 1))
            # Old 5% penalty would make 0.100 → ~0.105 and lose to 0.104.
            out.append(_sim(makespan=0.100 if pref == 0 else 0.104))
        return out

    _patch_batch_sim(monkeypatch, fake_sim)
    monkeypatch.setattr("tensortorrent.planner.maximal._eligible_compute", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.planner.maximal._decide_resources", lambda *a, **k: [])
    monkeypatch.setattr(
        "tensortorrent.backends.communication.select_communication_backend",
        lambda devices: MagicMock(backend_id="host"),
    )
    machine = MagicMock()
    machine.memory = {}
    machine.compute = {}
    cfg = CompileConfig(
        planner_des_candidates=4,
        planner_workers=1,
        objective=Objective.LATENCY,
        prefetch_distance=2,
    )
    _, _, _, pref, _ = _select_finalist_by_simulation(
        [plan],
        program=None,
        streaming=True,
        activation_budget_bytes=None,
        machine=machine,
        config=cfg,
    )
    assert pref == 0


def test_des_winner_strategy_uses_canonical_vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
    """After DES changes the device set, strategy must be a documented catalog label."""
    from tensortorrent.ir.resource_graph import (
        ComputeClass,
        ComputeResource,
        ResourceId,
        ResourceKind,
    )
    from tensortorrent.planner.maximal import enumerate_plan_strategies

    analytic = _plan(
        rank=0,
        devices=("cpu_0",),
        placements=[_placement("r0", "cpu_0")],
        signature="cpu",
    )
    des_win = _plan(
        rank=1,
        devices=("gpu0", "gpu1"),
        placements=[_placement("r0", "gpu0"), _placement("r1", "gpu1")],
        signature="2gpu",
    )
    gpu0 = ComputeResource(
        id=ResourceId(ResourceKind.COMPUTE, "gpu0"),
        compute_class=ComputeClass.DISCRETE_GPU,
        backend_id="cuda",
        vendor="nvidia",
        model="gpu0",
        memory_affinity=("vram0",),
        supported_dtypes=("float32",),
    )
    gpu1 = ComputeResource(
        id=ResourceId(ResourceKind.COMPUTE, "gpu1"),
        compute_class=ComputeClass.DISCRETE_GPU,
        backend_id="cuda",
        vendor="nvidia",
        model="gpu1",
        memory_affinity=("vram1",),
        supported_dtypes=("float32",),
    )
    machine = MagicMock()
    machine.memory = {}
    machine.compute = {"gpu0": gpu0, "gpu1": gpu1, "cpu_0": MagicMock()}

    monkeypatch.setattr(
        "tensortorrent.planner.maximal._eligible_compute",
        lambda m, c: [gpu0, gpu1],
    )
    monkeypatch.setattr(
        "tensortorrent.planner.maximal._decide_resources",
        lambda *a, **k: [
            ResourceDecision(resource="gpu0", selected=True, reason="des"),
            ResourceDecision(resource="gpu1", selected=True, reason="des"),
        ],
    )
    monkeypatch.setattr(
        "tensortorrent.backends.communication.select_communication_backend",
        lambda devices: MagicMock(backend_id="nccl"),
    )
    _recompute_winner_metadata(
        des_win,
        [analytic, des_win],
        machine=machine,
        config=CompileConfig(),
        prefetch=1,
        sim=_sim(makespan=0.05),
    )
    assert des_win.strategy in enumerate_plan_strategies()
    assert des_win.strategy == "multi_gpu"
    assert des_win.strategy not in {"heterogeneous", "single_accelerator", "empty"}


def test_winning_analytic_rank_is_not_finalist_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """winning_analytic_rank must reflect analytical rank, not shortlist position."""
    # Diversity shortlist: finalist_rank 0 has analytic_rank 2; rank 1 has analytic 5.
    plan_a = _plan(
        rank=2,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0", "a")],
        signature="a",
    )
    plan_a.search_statistics["analytic_rank"] = 2
    plan_a.search_statistics["finalist_rank"] = 0
    plan_a.search_statistics["search_rank"] = 2
    plan_b = _plan(
        rank=5,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0", "b")],
        signature="b",
    )
    plan_b.search_statistics["analytic_rank"] = 5
    plan_b.search_statistics["finalist_rank"] = 1
    plan_b.search_statistics["search_rank"] = 5
    _stub_schedule_validators(monkeypatch)
    monkeypatch.setattr(
        "tensortorrent.runtime.schedule.build_executable_schedule",
        lambda plan, *a, **k: MagicMock(instructions=[], _sig=plan.search_statistics["placement_signature"]),
    )

    def fake_sim(schedules: list[Any], machine: Any, workers: int = 0) -> list[Any]:
        return [_sim(makespan=0.05 if getattr(s, "_sig", "") == "b" else 0.20) for s in schedules]

    _patch_batch_sim(monkeypatch, fake_sim)
    monkeypatch.setattr("tensortorrent.planner.maximal._eligible_compute", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.planner.maximal._decide_resources", lambda *a, **k: [])
    monkeypatch.setattr(
        "tensortorrent.backends.communication.select_communication_backend",
        lambda devices: MagicMock(backend_id="host"),
    )
    machine = MagicMock()
    machine.memory = {}
    machine.compute = {}
    cfg = CompileConfig(planner_des_candidates=4, planner_workers=1, objective=Objective.LATENCY)
    win, _, _, _, stats = _select_finalist_by_simulation(
        [plan_a, plan_b],
        program=None,
        streaming=False,
        activation_budget_bytes=None,
        machine=machine,
        config=cfg,
    )
    assert win.search_statistics["placement_signature"] == "b"
    assert stats["winning_analytic_rank"] == 5
    assert stats["winning_finalist_rank"] == 1
    assert stats["winning_analytic_rank"] != stats["winning_finalist_rank"]


@requires_native
def test_native_planner_workers_reporting_serial_vs_requested() -> None:
    from tensortorrent.ir.resource_graph import (
        ComputeClass,
        ComputeResource,
        MemoryClass,
        MemoryResource,
        ResourceGraph,
        ResourceId,
        ResourceKind,
    )
    from tensortorrent.native import require_native

    native = require_native()
    problem = {
        "config": {
            "objective": "latency",
            "beam_width": 8,
            "candidates_per_device": 1,
            "local_search_iters": 0,
            "planner_workers": 4,
            "allow_parallel_subsets": True,
            "finalist_count": 2,
            "per_subset_finalists": 1,
            "allow_host_staged_transfers": True,
            "target_inflight_requests": 1,
        },
        "device_names": ["accel_0"],
        "capacities": [100_000],
        "device_memory": ["vram_0"],
        "regions": [
            {
                "name": "r0",
                "depends_on": [],
                "output_bytes": 64,
                "state_bytes": 0,
                "consumer_count": 0,
            }
        ],
        "order": [0],
        "candidates": [
            [
                {
                    "device": 0,
                    "backend_id": "mock",
                    "kernel_id": "r0:a0",
                    "dtype": "float32",
                    "estimated_latency_s": 0.01,
                    "workspace_bytes": 0,
                    "measured": True,
                }
            ]
        ],
        "edge_bytes": {},
        "subsets": [{"device_indices": [0]}],
    }
    machine = ResourceGraph(fingerprint="tiny-workers", backends_present=("mock",))
    machine.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "vram_0"),
            memory_class=MemoryClass.DEVICE_VRAM,
            capacity_bytes=100_000,
            allocatable_bytes=100_000,
            attached_compute=("accel_0",),
        )
    )
    machine.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "accel_0"),
            compute_class=ComputeClass.ACCELERATOR,
            backend_id="virtual",
            vendor="test",
            model="accel_0",
            memory_affinity=("vram_0",),
            supported_dtypes=("float32",),
        )
    )
    problem["machine"] = machine
    out = native.plan_placements(problem)
    stats = out["statistics"]
    assert stats["planner_workers_requested"] == 4
    assert stats["planner_workers_available"] == 4
    assert stats["planner_workers_used"] == 1
    assert stats.get("planner_pool_threads", 1) == 1
    assert not stats["parallel_search_used"]
    assert not stats["parallel_beam_used"]
    for f in out["finalists"]:
        assert "analytic_rank" in f
        assert "finalist_rank" in f
        assert f["search_rank"] == f["analytic_rank"]


def test_pageable_pressure_requires_host_pinned_signal() -> None:
    """Generic device/infeasible rejects must not burn pageable recovery slots."""
    from tensortorrent.compile.specialize import _host_or_pinned_pressure

    assert _host_or_pinned_pressure({"status": "infeasible", "error": "pinned host memory exceeded"})
    assert _host_or_pinned_pressure({"status": "rejected", "message": "host_ram budget"})
    assert not _host_or_pinned_pressure({"status": "infeasible", "error": "device vram exceeded"})
    assert not _host_or_pinned_pressure({"status": "infeasible", "error": "schedule memory peak"})
    assert not _host_or_pinned_pressure({"status": "infeasible"})


def test_all_des_variants_infeasible_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """No silent analytic#1 fallback when every DES outcome is rejected."""
    from tensortorrent.errors import SpecializationError

    plan = _plan(
        rank=0,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0")],
        prefetch=0,
        signature="p",
    )
    _stub_schedule_validators(monkeypatch)
    monkeypatch.setattr(
        "tensortorrent.runtime.schedule.build_executable_schedule",
        lambda *a, **k: MagicMock(instructions=[]),
    )

    def fake_sim(schedules: list[Any], machine: Any, workers: int = 0) -> list[Any]:
        return [{"status": "infeasible", "error": "device vram exceeded"} for _ in schedules]

    _patch_batch_sim(monkeypatch, fake_sim)
    monkeypatch.setattr("tensortorrent.planner.maximal._eligible_compute", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.planner.maximal._decide_resources", lambda *a, **k: [])
    monkeypatch.setattr(
        "tensortorrent.backends.communication.select_communication_backend",
        lambda devices: MagicMock(backend_id="host"),
    )
    machine = MagicMock()
    machine.memory = {}
    machine.compute = {}
    cfg = CompileConfig(planner_des_candidates=2, planner_workers=1, allow_host_staged_transfers=True)
    with pytest.raises(SpecializationError, match="All .* DES schedule variants infeasible"):
        _select_finalist_by_simulation(
            [plan],
            program=None,
            streaming=True,
            activation_budget_bytes=None,
            machine=machine,
            config=cfg,
        )


def test_near_equal_des_scores_use_tolerance_not_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real objective gaps win; only float-noise near-equals defer to secondary keys."""
    assert _scores_near_equal(0.1, 0.1 + 1e-15)
    assert not _scores_near_equal(0.100, 0.104)  # real gap still decides

    plan = _plan(
        rank=0,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0")],
        prefetch=2,
        signature="p",
    )
    _stub_schedule_validators(monkeypatch)

    def fake_build(plan: ExecutionPlan, residency: Any, **kwargs: Any) -> Any:
        pref = int(kwargs.get("prefetch_distance") or 0)
        return MagicMock(instructions=[], pref=pref)

    monkeypatch.setattr("tensortorrent.runtime.schedule.build_executable_schedule", fake_build)

    def fake_sim(schedules: list[Any], machine: Any, workers: int = 0) -> list[Any]:
        out = []
        for sched in schedules:
            pref = int(getattr(sched, "pref", 1))
            # Gap beats old 5% penalty band (0.100+5% ≈ 0.105) but is a real DES win.
            out.append(_sim(makespan=0.100 if pref == 0 else 0.104))
        return out

    _patch_batch_sim(monkeypatch, fake_sim)
    monkeypatch.setattr("tensortorrent.planner.maximal._eligible_compute", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.planner.maximal._decide_resources", lambda *a, **k: [])
    monkeypatch.setattr(
        "tensortorrent.backends.communication.select_communication_backend",
        lambda devices: MagicMock(backend_id="host"),
    )
    machine = MagicMock()
    machine.memory = {}
    machine.compute = {}
    cfg = CompileConfig(planner_des_candidates=4, planner_workers=1, objective=Objective.LATENCY)
    _, _, _, pref, stats = _select_finalist_by_simulation(
        [plan],
        program=None,
        streaming=True,
        activation_budget_bytes=None,
        machine=machine,
        config=cfg,
    )
    assert pref == 0
    assert stats["winning_simulated_rank"] == 0
    assert stats.get("parallel_simulation_used") is False
    assert stats.get("simulator_workers_used") == 1


def test_des_winner_nontransitive_near_equal_is_stable() -> None:
    """Pairwise ~= is non-transitive; two-stage select must still be permutation-stable.

    A ~= B, B ~= C, but A ≉ C (rel_tol=1e-9). Secondary ranks disagree so a
    cyclic fuzzy comparator would flip winners under shuffle.
    """
    # Scores around 1.0 with ~0.75e-9 steps: adjacent pairs near-equal, ends not.
    a = 1.00000000000
    b = 1.00000000075
    c = 1.00000000150
    assert _scores_near_equal(a, b)
    assert _scores_near_equal(b, c)
    assert not _scores_near_equal(a, c)

    def cand(score: float, analytic: int, finalist: int, idx: int) -> dict[str, Any]:
        return {
            "raw_score": score,
            "analytic_rank": analytic,
            "finalist_rank": finalist,
            "pref_distance": 0,
            "pref_tie": 0,
            "variant_idx": idx,
            "outcome": f"out-{idx}",
        }

    # Stage 1 best raw is A. Stage 2 tie set is {A,B} (C is outside A's tolerance).
    # B has better analytic_rank → winner is B. A fuzzy pairwise sort that also
    # treats B~=C could cycle when secondary ranks disagree across the chain.
    base = [
        cand(a, analytic=2, finalist=2, idx=0),
        cand(b, analytic=0, finalist=0, idx=1),
        cand(c, analytic=1, finalist=1, idx=2),
    ]
    winners: set[int] = set()
    for order in itertools.permutations(base):
        win, ranks = _select_des_winner(list(order))
        winners.add(int(win["variant_idx"]))
        assert ranks[0] == 0  # strict raw-score order: A first
    assert winners == {1}, f"winner must always be B (idx 1), got {winners}"

    rng = random.Random(0)
    for _ in range(64):
        shuffled = list(base)
        rng.shuffle(shuffled)
        win, _ = _select_des_winner(shuffled)
        assert int(win["variant_idx"]) == 1


def test_des_winner_exact_equal_uses_secondary_keys() -> None:
    """Exact equal DES scores → deterministic analytic/finalist/index tie-break."""
    cands = [
        {
            "raw_score": 0.1,
            "analytic_rank": 2,
            "finalist_rank": 1,
            "pref_distance": 0,
            "pref_tie": 0,
            "variant_idx": 0,
            "outcome": "a",
        },
        {
            "raw_score": 0.1,
            "analytic_rank": 0,
            "finalist_rank": 2,
            "pref_distance": 0,
            "pref_tie": 0,
            "variant_idx": 1,
            "outcome": "b",
        },
        {
            "raw_score": 0.1,
            "analytic_rank": 1,
            "finalist_rank": 0,
            "pref_distance": 0,
            "pref_tie": 0,
            "variant_idx": 2,
            "outcome": "c",
        },
    ]
    rng = random.Random(1)
    for _ in range(32):
        order = list(cands)
        rng.shuffle(order)
        win, ranks = _select_des_winner(order)
        assert int(win["variant_idx"]) == 1  # best analytic_rank
        assert ranks[1] == 0


def test_des_winner_meaningful_gap_ignores_analytic_preference() -> None:
    """96ms vs 100ms: lower DES score wins even if analytic rank prefers the slower plan."""
    cands = [
        {
            "raw_score": 0.100,
            "analytic_rank": 0,
            "finalist_rank": 0,
            "pref_distance": 0,
            "pref_tie": 0,
            "variant_idx": 0,
            "outcome": "slow-but-analytic-best",
        },
        {
            "raw_score": 0.096,
            "analytic_rank": 5,
            "finalist_rank": 5,
            "pref_distance": 9,
            "pref_tie": 9,
            "variant_idx": 1,
            "outcome": "fast",
        },
    ]
    for order in itertools.permutations(cands):
        win, ranks = _select_des_winner(list(order))
        assert int(win["variant_idx"]) == 1
        assert ranks[1] == 0
        assert ranks[0] == 1


def test_float_noise_defers_to_analytic_prefetch_tiebreak(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scores within rel/abs tolerance use secondary keys (closer to analytic prefetch)."""
    plan = _plan(
        rank=0,
        devices=("gpu0",),
        placements=[_placement("r0", "gpu0")],
        prefetch=2,
        signature="p",
    )
    _stub_schedule_validators(monkeypatch)

    def fake_build(plan: ExecutionPlan, residency: Any, **kwargs: Any) -> Any:
        pref = int(kwargs.get("prefetch_distance") or 0)
        return MagicMock(instructions=[], pref=pref)

    monkeypatch.setattr("tensortorrent.runtime.schedule.build_executable_schedule", fake_build)

    def fake_sim(schedules: list[Any], machine: Any, workers: int = 0) -> list[Any]:
        # Identical makespan → tolerance treats as equal → prefer analytic pref=2.
        return [_sim(makespan=0.1) for _ in schedules]

    _patch_batch_sim(monkeypatch, fake_sim)
    monkeypatch.setattr("tensortorrent.planner.maximal._eligible_compute", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.planner.maximal._decide_resources", lambda *a, **k: [])
    monkeypatch.setattr(
        "tensortorrent.backends.communication.select_communication_backend",
        lambda devices: MagicMock(backend_id="host"),
    )
    machine = MagicMock()
    machine.memory = {}
    machine.compute = {}
    cfg = CompileConfig(planner_des_candidates=4, planner_workers=1, objective=Objective.LATENCY)
    _, _, _, pref, _ = _select_finalist_by_simulation(
        [plan],
        program=None,
        streaming=True,
        activation_budget_bytes=None,
        machine=machine,
        config=cfg,
    )
    assert pref == 2


@requires_native
def test_batch_des_stats_authoritative_from_rust() -> None:
    from tensortorrent.ir.graph import OpCode
    from tensortorrent.ir.resource_graph import (
        ComputeClass,
        ComputeResource,
        MemoryClass,
        MemoryResource,
        ResourceGraph,
        ResourceId,
        ResourceKind,
    )
    from tensortorrent.runtime.schedule import ExecutableSchedule, PlanInstruction
    from tensortorrent.runtime.simulator.discrete_event import simulate_schedules_with_stats

    machine = ResourceGraph(fingerprint="des-stats", backends_present=("cpu",))
    machine.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "host_ram"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=1 << 20,
            allocatable_bytes=1 << 20,
            attached_compute=("cpu0",),
        )
    )
    machine.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "cpu0"),
            compute_class=ComputeClass.CPU_NUMA_POOL,
            backend_id="cpu",
            vendor="test",
            model="cpu0",
            memory_affinity=("host_ram",),
            supported_dtypes=("float32",),
        )
    )
    sched = ExecutableSchedule(
        graph_name="g",
        fingerprint="fp",
        instructions=(
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="c0",
                resource="cpu0",
                outputs=("o0",),
                nbytes=8,
                predicted_duration_s=0.01,
            ),
        ),
    )
    batch = [sched, sched, sched]
    _outs, serial_stats = simulate_schedules_with_stats(batch, machine, workers=1)
    assert serial_stats["parallel_simulation_used"] is False
    assert serial_stats["simulator_workers_used"] == 1
    # Tiny instruction work: auto must stay serial (pool overhead).
    _outs, auto_tiny = simulate_schedules_with_stats(batch, machine, workers=0)
    assert auto_tiny["parallel_simulation_used"] is False
    assert auto_tiny["simulator_workers_used"] == 1
    # Explicit workers still parallelize n>1.
    _outs, capped = simulate_schedules_with_stats(batch, machine, workers=2)
    assert capped["parallel_simulation_used"] is True
    assert capped["simulator_workers_used"] == 2
    assert capped["simulator_workers_requested"] == 2

    heavy_instr = tuple(
        PlanInstruction(
            opcode=OpCode.COMPUTE,
            name=f"c{i}",
            resource="cpu0",
            outputs=(f"o{i}",),
            nbytes=8,
            predicted_duration_s=0.001,
            depends_on=(f"c{i - 1}",) if i else (),
        )
        for i in range(40)
    )
    heavy = ExecutableSchedule(graph_name="g", fingerprint="fp-h", instructions=heavy_instr)
    heavy_batch = [heavy, heavy, heavy]
    _outs, auto_heavy = simulate_schedules_with_stats(heavy_batch, machine, workers=0)
    assert auto_heavy["parallel_simulation_used"] is True
    assert auto_heavy["simulator_workers_used"] >= 2
