"""Beyond-VRAM staging, Evict gating, and host-pin helpers."""

from __future__ import annotations

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
from tensortorrent.planner.maximal import ExecutionPlan, Placement
from tensortorrent.runtime.schedule import MemoryTier, build_executable_schedule
from tensortorrent.runtime.simulator.discrete_event import simulate_schedule


def _gpu_machine(*, pinned_alloc: int, numa_alloc: int, vram_alloc: int) -> ResourceGraph:
    machine = ResourceGraph(fingerprint="beyond-vram-stage")
    machine.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "pinned_host_0"),
            memory_class=MemoryClass.PINNED_HOST,
            capacity_bytes=pinned_alloc,
            allocatable_bytes=pinned_alloc,
        )
    )
    machine.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "numa_ram_0"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=numa_alloc,
            allocatable_bytes=numa_alloc,
        )
    )
    machine.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "mock_vram_0"),
            memory_class=MemoryClass.DEVICE_VRAM,
            capacity_bytes=vram_alloc,
            allocatable_bytes=vram_alloc,
        )
    )
    machine.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "mock_accel_0"),
            compute_class=ComputeClass.ACCELERATOR,
            backend_id="mock_accel",
            model="test",
            memory_affinity=("mock_vram_0",),
        )
    )
    return machine


def _stream_plan(*state_bytes: int) -> ExecutionPlan:
    placements = []
    for i, nbytes in enumerate(state_bytes):
        placements.append(
            Placement(
                region_id=f"region_{i}",
                device="mock_accel_0",
                backend_id="mock_accel",
                dtype="float32",
                kernel_id="k",
                estimated_latency_s=0.01,
                state_bytes=nbytes,
                output_bytes=64,
                depends_on=() if i == 0 else (f"region_{i - 1}",),
            )
        )
    return ExecutionPlan(
        graph_name="beyond",
        fingerprint="t",
        objective="latency",
        placements=placements,
        decisions=[],
        devices_used=("mock_accel_0",),
        communication_backend="none",
        predicted_latency_s=0.01 * len(placements),
    )


def _parameter_loads(schedule):
    return [
        i
        for i in schedule.instructions
        if i.opcode == OpCode.LOAD and i.attributes.get("kind") == "parameter_materialize"
    ]


def test_oversized_region_stages_on_numa_not_pinned() -> None:
    """Single Load larger than pinned allocatable must use NUMA staging."""
    machine = _gpu_machine(pinned_alloc=1 << 20, numa_alloc=64 << 20, vram_alloc=64 << 20)
    plan = _stream_plan(8 << 20)
    schedule = build_executable_schedule(plan, streaming=True, machine=machine, prefetch_distance=0)
    loads = _parameter_loads(schedule)
    assert len(loads) == 1
    assert loads[0].destination == "numa_ram_0"
    assert loads[0].memory_tier == MemoryTier.SYSTEM_RAM


def test_prefetch_destination_is_staging_host_not_device() -> None:
    machine = _gpu_machine(pinned_alloc=64 << 20, numa_alloc=64 << 20, vram_alloc=64 << 20)
    plan = _stream_plan(1024)
    schedule = build_executable_schedule(plan, streaming=True, machine=machine, prefetch_distance=1)
    prefetches = [i for i in schedule.instructions if i.opcode == OpCode.PREFETCH]
    assert prefetches
    assert prefetches[0].destination == "pinned_host_0"
    assert prefetches[0].destination != "mock_accel_0"


def test_force_pageable_host_staging_skips_pinned() -> None:
    machine = _gpu_machine(pinned_alloc=64 << 20, numa_alloc=64 << 20, vram_alloc=64 << 20)
    plan = _stream_plan(1024)
    schedule = build_executable_schedule(
        plan,
        streaming=True,
        machine=machine,
        prefetch_distance=0,
        force_pageable_host_staging=True,
    )
    loads = _parameter_loads(schedule)
    assert loads[0].destination == "numa_ram_0"
    assert "host_staging=pageable" in schedule.notes


def test_tiny_pinned_multi_region_des_accepts_with_numa_fallback() -> None:
    """Layers larger than pinned pool: NUMA staging + DES stays feasible."""
    pinned = 2 << 20
    layer = 8 << 20
    machine = _gpu_machine(pinned_alloc=pinned, numa_alloc=256 << 20, vram_alloc=32 << 20)
    plan = _stream_plan(layer, layer, layer)
    schedule = build_executable_schedule(plan, streaming=True, machine=machine, prefetch_distance=1)
    loads = _parameter_loads(schedule)
    assert all(i.destination == "numa_ram_0" for i in loads)
    result = simulate_schedule(schedule, machine)
    assert result.peak_bytes.get("pinned_host_0", 0) <= pinned
    assert result.peak_bytes.get("numa_ram_0", 0) <= 256 << 20


def test_sequential_regions_fit_pinned_when_each_fits() -> None:
    """Evict gates next Load → peak is one region; both may use pinned."""
    pinned = 4 << 20
    machine = _gpu_machine(pinned_alloc=pinned, numa_alloc=64 << 20, vram_alloc=64 << 20)
    plan = _stream_plan(3 << 20, 3 << 20)
    schedule = build_executable_schedule(plan, streaming=True, machine=machine, prefetch_distance=0)
    loads = _parameter_loads(schedule)
    assert all(i.destination == "pinned_host_0" for i in loads)
    staging_evicts = [i for i in schedule.instructions if i.opcode == OpCode.EVICT and i.attributes.get("staging")]
    assert staging_evicts
    assert all(i.memory_tier == MemoryTier.PINNED_RAM for i in staging_evicts)
    result = simulate_schedule(schedule, machine)
    assert result.peak_bytes.get("pinned_host_0", 0) <= pinned


def test_schedule_host_pin_helpers() -> None:
    from tensortorrent.runtime.provisioning import (
        schedule_needs_host_pin,
        schedule_uses_pinned_staging,
    )

    machine = _gpu_machine(pinned_alloc=64 << 20, numa_alloc=64 << 20, vram_alloc=64 << 20)
    pinned_sched = build_executable_schedule(_stream_plan(1024), streaming=True, machine=machine, prefetch_distance=0)
    pageable_sched = build_executable_schedule(
        _stream_plan(1024),
        streaming=True,
        machine=machine,
        prefetch_distance=0,
        force_pageable_host_staging=True,
    )
    resident_sched = build_executable_schedule(
        _stream_plan(1024), streaming=False, machine=machine, prefetch_distance=0
    )
    assert schedule_uses_pinned_staging(pinned_sched) is True
    assert schedule_uses_pinned_staging(pageable_sched) is False
    assert schedule_uses_pinned_staging(None) is False
    assert schedule_needs_host_pin(pinned_sched) is True
    assert schedule_needs_host_pin(pageable_sched) is True  # Transfer still H2D
    assert schedule_needs_host_pin(resident_sched) is True
    assert schedule_needs_host_pin(None) is False


def test_parameter_evict_gate_skips_stateless_regions() -> None:
    """ReLU-only placements must not clear the Transfer←Evict dependency chain."""
    from tensortorrent.runtime.schedule.build import _ParameterEvictGate

    gate = _ParameterEvictGate()
    gate.record(0, device="evict::state::region_0", staging=None)
    gate.skip(1)  # activation-only
    gate.record(2, device="evict::state::region_2", staging=None)
    assert gate.prior_deps(1) == ["evict::state::region_0"]
    assert gate.prior_deps(2) == ["evict::state::region_0"]
    assert gate.prior_deps(3) == ["evict::state::region_2"]
    assert gate.lead_gate(1) == "evict::state::region_0"


def test_resident_beyond_vram_transfers_wait_on_prior_device_evict() -> None:
    """Host-resident + VRAM stream: each H2D must wait on the previous device Evict."""
    machine = _gpu_machine(pinned_alloc=64 << 20, numa_alloc=256 << 20, vram_alloc=32 << 20)
    # Alternate weight regions with zero-state placeholders (like Linear/ReLU splits).
    placements = []
    for i in range(4):
        placements.append(
            Placement(
                region_id=f"linear_{i}",
                device="mock_accel_0",
                backend_id="mock_accel",
                dtype="float32",
                kernel_id="k",
                estimated_latency_s=0.01,
                state_bytes=8 << 20,
                output_bytes=64,
                depends_on=() if i == 0 else (f"act_{i - 1}",),
            )
        )
        placements.append(
            Placement(
                region_id=f"act_{i}",
                device="mock_accel_0",
                backend_id="mock_accel",
                dtype="float32",
                kernel_id="k",
                estimated_latency_s=0.001,
                state_bytes=0,
                output_bytes=64,
                depends_on=(f"linear_{i}",),
            )
        )
    plan = ExecutionPlan(
        graph_name="beyond",
        fingerprint="t",
        objective="latency",
        placements=placements,
        decisions=[],
        devices_used=("mock_accel_0",),
        communication_backend="none",
        predicted_latency_s=0.1,
    )
    schedule = build_executable_schedule(plan, streaming=False, machine=machine, prefetch_distance=0)
    xfers = [
        i
        for i in schedule.instructions
        if i.opcode == OpCode.TRANSFER and i.attributes.get("kind") == "parameter_host_to_device"
    ]
    # One Transfer per weight region (coalesced; not one op per tensor name).
    assert len(xfers) == 4
    assert {i.attributes.get("region_id") for i in xfers} == {f"linear_{i}" for i in range(4)}
    # First region's transfer may be ungated; every later weight Transfer must
    # depend on some prior device Evict.
    later = [i for i in xfers if "linear_0" not in (i.attributes.get("region_id") or i.name)]
    assert later
    for i in later:
        assert any(d.startswith("evict::state::") for d in i.depends_on), i
    result = simulate_schedule(schedule, machine)
    assert result.peak_bytes.get("mock_vram_0", 0) <= 32 << 20


def test_real_device_transfers_omit_mock_delay_attrs() -> None:
    from tensortorrent.runtime.schedule.build import _mock_delay_attrs

    assert _mock_delay_attrs("cuda_gpu_0", transfer=True, compute=True) == {}
    assert _mock_delay_attrs("mock_accel_0", transfer=True) == {"mock_transfer_delay_s": 0.08}
    assert _mock_delay_attrs("mock_accel_0", compute=True) == {"mock_compute_delay_s": 0.05}


def test_should_hoist_resident_parameters_respects_vram_budget() -> None:
    from tensortorrent.compile.fit import ACCELERATOR_REGION_STATE_FRACTION, should_hoist_resident_parameters
    from tensortorrent.config import CompileConfig

    assert should_hoist_resident_parameters(CompileConfig(vram_budget_bytes=None), state_bytes=1 << 30) is True
    assert should_hoist_resident_parameters(CompileConfig(vram_budget_bytes=2 << 30), state_bytes=1 << 30) is True
    # At full budget there is no activation headroom — do not hoist.
    assert should_hoist_resident_parameters(CompileConfig(vram_budget_bytes=1 << 30), state_bytes=1 << 30) is False
    assert should_hoist_resident_parameters(CompileConfig(vram_budget_bytes=1 << 30), state_bytes=2 << 30) is False
    assert (
        should_hoist_resident_parameters(
            CompileConfig(allow_training=True, vram_budget_bytes=4 << 30),
            state_bytes=1 << 20,
        )
        is False
    )
    budget = 10 << 20
    ok_state = int(budget * ACCELERATOR_REGION_STATE_FRACTION)
    assert should_hoist_resident_parameters(CompileConfig(vram_budget_bytes=budget), state_bytes=ok_state) is True
    assert should_hoist_resident_parameters(CompileConfig(vram_budget_bytes=budget), state_bytes=ok_state + 1) is False


def test_should_hoist_uses_machine_capacity_when_budget_unset() -> None:
    """Near-VRAM fits must stream (no hoist) when only machine capacity is known.

    Mirrors the 0.75× crossover failure mode: budget unset + state under physical
    VRAM but over ACCELERATOR_REGION_STATE_FRACTION → full residency OOMs on workspace.
    """
    from tensortorrent.compile.fit import ACCELERATOR_REGION_STATE_FRACTION, should_hoist_resident_parameters
    from tensortorrent.config import CompileConfig

    vram = 8 << 30
    machine = _gpu_machine(pinned_alloc=64 << 20, numa_alloc=64 << 20, vram_alloc=vram)
    cfg = CompileConfig(vram_budget_bytes=None, allow_gpu=True)
    under = int(vram * 0.50)
    over_headroom = int(vram * 0.75)
    limit = int(vram * ACCELERATOR_REGION_STATE_FRACTION)
    assert should_hoist_resident_parameters(cfg, state_bytes=under, machine=machine) is True
    assert should_hoist_resident_parameters(cfg, state_bytes=over_headroom, machine=machine) is False
    assert should_hoist_resident_parameters(cfg, state_bytes=limit, machine=machine) is True
    assert should_hoist_resident_parameters(cfg, state_bytes=limit + 1, machine=machine) is False


def test_should_hoist_uses_allocatable_not_raw_budget() -> None:
    """Explicit budget must not inflate hoist past discovered allocatable VRAM.

    Benchmarks often set vram_budget_bytes to physical total while discovery
    reports allocatable = total − display/driver headroom. Hoist and region
    budgets must share accelerator_vram_capacity_bytes (min of both).
    """
    from tensortorrent.compile.fit import (
        ACCELERATOR_REGION_STATE_FRACTION,
        accelerator_vram_capacity_bytes,
        should_hoist_resident_parameters,
    )
    from tensortorrent.config import CompileConfig

    allocatable = 7 << 30
    physical = 8 << 30
    machine = _gpu_machine(pinned_alloc=64 << 20, numa_alloc=64 << 20, vram_alloc=allocatable)
    cfg = CompileConfig(vram_budget_bytes=physical, allow_gpu=True)
    effective = accelerator_vram_capacity_bytes(cfg, machine)
    assert effective == allocatable
    # Between 0.70×allocatable and 0.70×physical: must refuse hoist (old bug → True).
    state = int(allocatable * ACCELERATOR_REGION_STATE_FRACTION) + (1 << 20)
    assert state <= int(physical * ACCELERATOR_REGION_STATE_FRACTION)
    assert should_hoist_resident_parameters(cfg, state_bytes=state, machine=machine) is False
    under = int(allocatable * 0.50)
    assert should_hoist_resident_parameters(cfg, state_bytes=under, machine=machine) is True


def test_specialize_rebuilds_pageable_after_pinned_des_reject(monkeypatch) -> None:
    """prefetch=0 + pinned DES reject → one pageable rebuild, then accept."""
    from tensortorrent.compile import specialize as specialize_mod
    from tensortorrent.errors import MemoryCapacityError
    from tensortorrent.runtime.simulator import discrete_event as de

    machine = _gpu_machine(pinned_alloc=64 << 20, numa_alloc=64 << 20, vram_alloc=64 << 20)
    plan = _stream_plan(1024)
    plan.prefetch_distance = 0
    calls: list[bool] = []

    def fake_simulate(schedule, _machine):
        pageable = "host_staging=pageable" in list(schedule.notes)
        calls.append(pageable)
        if not pageable:
            raise MemoryCapacityError(
                "schedule infeasible: memory pinned_host_0 resident=999 allocatable=1 at inst load::region_0"
            )
        return type("R", (), {"peak_bytes": {}, "makespan_s": 0.0, "simulated": True})()

    # Local import inside _schedule_and_simulate → patch source module.
    monkeypatch.setattr(de, "simulate_schedule", fake_simulate)
    sched, _sim, prefetch = specialize_mod._schedule_and_simulate(
        plan,
        None,
        streaming=True,
        program=None,
        activation_budget_bytes=None,
        machine=machine,
    )
    assert prefetch == 0
    assert calls == [False, True]
    assert "host_staging=pageable_after_pinned_pressure" in plan.notes
    assert "host_staging=pageable" in sched.notes


def test_resident_beyond_pinned_pool_skips_full_model_pin() -> None:
    """Beyond-VRAM resident stores must not lock the whole model into pinned_host."""
    from tensortorrent.runtime.provisioning import (
        pinned_host_allocatable_bytes,
        should_pin_parameter_store,
    )

    machine = _gpu_machine(pinned_alloc=64 << 20, numa_alloc=256 << 20, vram_alloc=32 << 20)
    assert pinned_host_allocatable_bytes(machine) == 64 << 20
    resident = build_executable_schedule(_stream_plan(1024), streaming=False, machine=machine, prefetch_distance=0)
    # Schedule still wants H2D, but full-model pin is refused when state > pool.
    assert should_pin_parameter_store(resident, state_bytes=8 << 20, machine=machine, streaming=False) is True
    assert should_pin_parameter_store(resident, state_bytes=128 << 20, machine=machine, streaming=False) is False
    # Streaming pins per-acquire (region sized); size gate does not apply the same way.
    pinned_sched = build_executable_schedule(_stream_plan(1024), streaming=True, machine=machine, prefetch_distance=0)
    assert should_pin_parameter_store(pinned_sched, state_bytes=128 << 20, machine=machine, streaming=True) is True


def test_batch_des_pageable_recovery_on_native_infeasible_memory(monkeypatch) -> None:
    """Batch DES ``infeasible_memory`` on ``pinned_host_*`` rebuilds pageable staging."""
    from unittest.mock import MagicMock

    from tensortorrent.compile.specialize import _select_finalist_by_simulation
    from tensortorrent.config import CompileConfig
    from tensortorrent.planner.maximal import ExecutionPlan, Placement
    from tensortorrent.runtime.simulator.discrete_event import SimulationResult

    plan = ExecutionPlan(
        graph_name="reg",
        fingerprint="t",
        objective="latency",
        placements=[
            Placement(
                region_id="r0",
                device="mock_accel_0",
                backend_id="mock_accel",
                dtype="float32",
                kernel_id="k",
                estimated_latency_s=0.01,
                state_bytes=1024,
                output_bytes=64,
            )
        ],
        decisions=[],
        devices_used=("mock_accel_0",),
        communication_backend="none",
        predicted_latency_s=0.01,
        prefetch_distance=1,
    )
    plan.search_statistics = {"analytic_rank": 0, "finalist_rank": 0}

    def fake_build(*args, **kwargs):
        pageable = bool(kwargs.get("force_pageable_host_staging"))
        return MagicMock(instructions=[], notes=["host_staging=pageable"] if pageable else [], pageable=pageable)

    monkeypatch.setattr("tensortorrent.runtime.schedule.build_executable_schedule", fake_build)
    monkeypatch.setattr("tensortorrent.runtime.schedule.schedule_matches_plan", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.runtime.schedule.validate_schedule", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.runtime.schedule.validate_schedule_resources", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.runtime.schedule.validate_schedule_tensor_sizes", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.runtime.residency.attach_residency_to_plan", lambda *a, **k: None)

    def fake_sim_stats(schedules, machine, workers=0):
        outs = []
        for sched in schedules:
            if getattr(sched, "pageable", False):
                outs.append(
                    SimulationResult(
                        makespan_s=0.05,
                        peak_bytes={"numa_ram_0": 1024},
                        timeline=[],
                        exposed_transfer_latency_s=0.0,
                        resource_busy_s={},
                        bytes_transferred=1024,
                        bytes_read=1024,
                        simulated=True,
                    )
                )
            else:
                outs.append(
                    {
                        "status": "infeasible_memory",
                        "memory": "pinned_host_0",
                        "resident_bytes": 999,
                        "allocatable_bytes": 1,
                        "instruction": "load::r0",
                    }
                )
        return outs, {"parallel_simulation_used": False, "simulator_workers_used": 1}

    monkeypatch.setattr(
        "tensortorrent.runtime.simulator.discrete_event.simulate_schedules_with_stats",
        fake_sim_stats,
    )
    monkeypatch.setattr("tensortorrent.planner.maximal._eligible_compute", lambda *a, **k: [])
    monkeypatch.setattr("tensortorrent.planner.maximal._decide_resources", lambda *a, **k: [])
    monkeypatch.setattr(
        "tensortorrent.backends.communication.select_communication_backend",
        lambda devices: MagicMock(backend_id="host"),
    )

    machine = _gpu_machine(pinned_alloc=1 << 20, numa_alloc=64 << 20, vram_alloc=64 << 20)
    cfg = CompileConfig(planner_des_candidates=2, planner_workers=1, allow_host_staged_transfers=True)
    win, sched, _sim, _pref, stats = _select_finalist_by_simulation(
        [plan],
        program=None,
        streaming=True,
        activation_budget_bytes=None,
        machine=machine,
        config=cfg,
    )
    assert stats["schedule_variants_simulated"] >= 2
    assert any("pageable" in n for n in win.notes)
    assert "host_staging=pageable" in list(sched.notes)
