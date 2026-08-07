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
    from tensortorrent.compile.fit import should_hoist_resident_parameters
    from tensortorrent.config import CompileConfig

    assert should_hoist_resident_parameters(CompileConfig(vram_budget_bytes=None), state_bytes=1 << 30) is True
    assert should_hoist_resident_parameters(CompileConfig(vram_budget_bytes=2 << 30), state_bytes=1 << 30) is True
    assert should_hoist_resident_parameters(CompileConfig(vram_budget_bytes=1 << 30), state_bytes=2 << 30) is False
    assert (
        should_hoist_resident_parameters(
            CompileConfig(allow_training=True, vram_budget_bytes=4 << 30),
            state_bytes=1 << 20,
        )
        is False
    )


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
