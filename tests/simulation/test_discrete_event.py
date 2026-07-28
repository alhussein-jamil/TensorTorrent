"""Simulator tests for concurrent multi-device schedules and byte-aware costing."""

from __future__ import annotations

import pytest

from streamcompiler.ir.resource_graph import (
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
from streamcompiler.planner.maximal import ExecutionPlan, Placement
from streamcompiler.simulator import simulate_plan


def _two_gpu_machine(*, bandwidth: float | None = 8e9) -> ResourceGraph:
    machine = ResourceGraph(fingerprint="sim")
    for i in range(2):
        machine.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, f"vram_{i}"),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=8 << 30,
                allocatable_bytes=8 << 30,
            )
        )
        machine.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, f"gpu_{i}"),
                compute_class=ComputeClass.DISCRETE_GPU,
                backend_id="cuda",
                model=f"g{i}",
                vendor="nvidia",
                memory_affinity=(f"vram_{i}",),
            )
        )
    machine.add_link(
        TransferLink(
            id=ResourceId(ResourceKind.LINK, "vram_0->vram_1"),
            link_class=LinkClass.NVLINK,
            source="vram_0",
            destination="vram_1",
            peer_to_peer=True,
            measured=bandwidth is not None,
            bytes_per_s=bandwidth,
            latency_s=1e-6,
        )
    )
    return machine


def test_multi_device_overlap_beats_serial() -> None:
    machine = _two_gpu_machine()
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("a", "gpu_0", "cuda", "float16", "k", 1.0, depends_on=()),
            Placement("b", "gpu_1", "cuda", "float16", "k", 1.0, depends_on=()),
        ],
        decisions=[],
        devices_used=("gpu_0", "gpu_1"),
        communication_backend="host_staged",
        predicted_latency_s=2.0,
        strategy="tensor_or_pipeline_multi_gpu",
    )
    result = simulate_plan(plan, machine)
    # Independent regions on two devices overlap → ~1s, not 2s.
    assert result.makespan_s == pytest.approx(1.0)
    assert result.makespan_s < 2.0


def test_peak_bytes_come_from_placement_working_sets() -> None:
    machine = _two_gpu_machine()
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement(
                "a",
                "gpu_0",
                "cuda",
                "float16",
                "k",
                0.1,
                output_bytes=4_000_000,
                state_bytes=1_000_000,
            ),
            Placement(
                "b",
                "gpu_1",
                "cuda",
                "float16",
                "k",
                0.1,
                output_bytes=2_000_000,
                state_bytes=500_000,
            ),
        ],
        decisions=[],
        devices_used=("gpu_0", "gpu_1"),
        communication_backend="host_staged",
        predicted_latency_s=0.1,
    )
    result = simulate_plan(plan, machine)
    assert result.peak_bytes["vram_0"] == 5_000_000
    assert result.peak_bytes["vram_1"] == 2_500_000
    # No more fabricated 1 MiB-per-region accounting.
    assert 1_048_576 not in result.peak_bytes.values()


def test_cross_device_transfer_scales_with_output_bytes() -> None:
    """A larger producer output must expose more transfer time on the same link."""
    machine = _two_gpu_machine(bandwidth=1e9)  # 1 GB/s
    small = ExecutionPlan(
        graph_name="t",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("a", "gpu_0", "cuda", "float16", "k", 0.01, output_bytes=1_000_000),
            Placement("b", "gpu_1", "cuda", "float16", "k", 0.01, depends_on=("a",), output_bytes=0),
        ],
        decisions=[],
        devices_used=("gpu_0", "gpu_1"),
        communication_backend="nccl",
        predicted_latency_s=0.0,
    )
    large = ExecutionPlan(
        graph_name="t",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("a", "gpu_0", "cuda", "float16", "k", 0.01, output_bytes=8_000_000),
            Placement("b", "gpu_1", "cuda", "float16", "k", 0.01, depends_on=("a",), output_bytes=0),
        ],
        decisions=[],
        devices_used=("gpu_0", "gpu_1"),
        communication_backend="nccl",
        predicted_latency_s=0.0,
    )
    small_result = simulate_plan(small, machine)
    large_result = simulate_plan(large, machine)
    assert large_result.exposed_transfer_latency_s > small_result.exposed_transfer_latency_s
    # 8 MB at 1 GB/s is about 8 ms; allow alpha and contention.
    assert large_result.exposed_transfer_latency_s == pytest.approx(0.008, rel=0.25)
    assert large_result.makespan_s > small_result.makespan_s


def test_same_device_dependency_exposes_no_transfer() -> None:
    machine = _two_gpu_machine()
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("a", "gpu_0", "cuda", "float16", "k", 0.1, output_bytes=8_000_000),
            Placement("b", "gpu_0", "cuda", "float16", "k", 0.1, depends_on=("a",)),
        ],
        decisions=[],
        devices_used=("gpu_0",),
        communication_backend="none",
        predicted_latency_s=0.2,
    )
    result = simulate_plan(plan, machine)
    assert result.exposed_transfer_latency_s == 0.0
    assert result.makespan_s == pytest.approx(0.2)


def test_simulator_releases_activations_after_last_consumer() -> None:
    """Peak must not keep every produced activation forever."""
    machine = _two_gpu_machine()
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("a", "gpu_0", "cuda", "float16", "k", 0.1, output_bytes=4_000_000, state_bytes=0),
            Placement("b", "gpu_0", "cuda", "float16", "k", 0.1, depends_on=("a",), output_bytes=1_000_000),
            Placement("c", "gpu_0", "cuda", "float16", "k", 0.1, depends_on=("b",), output_bytes=500_000),
        ],
        decisions=[],
        devices_used=("gpu_0",),
        communication_backend="none",
        predicted_latency_s=0.3,
    )
    result = simulate_plan(plan, machine)
    assert result.simulated is True
    assert result.release_events, "producer activations must be released"
    # a (4M) released after b; peak is at most a+b outputs briefly, not a+b+c forever.
    assert result.peak_bytes["vram_0"] <= 5_000_000
    assert result.peak_bytes["vram_0"] >= 4_000_000
    assert any(e.get("event") == "transfer" for e in result.transfer_events) is False


def test_cross_device_transfer_emits_transfer_events() -> None:
    machine = _two_gpu_machine(bandwidth=1e9)
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("a", "gpu_0", "cuda", "float16", "k", 0.01, output_bytes=2_000_000),
            Placement("b", "gpu_1", "cuda", "float16", "k", 0.01, depends_on=("a",), output_bytes=0),
        ],
        decisions=[],
        devices_used=("gpu_0", "gpu_1"),
        communication_backend="nccl",
        predicted_latency_s=0.0,
    )
    result = simulate_plan(plan, machine)
    assert len(result.transfer_events) == 1
    assert result.transfer_events[0]["nbytes"] == 2_000_000
    assert result.transfer_events[0]["simulated"] is True
    assert result.transfer_events[0]["contention_factor"] >= 1.0


def test_transfer_contention_factor_lengthens_hop() -> None:
    from streamcompiler.simulator.discrete_event import _schedule_transfer

    machine = _two_gpu_machine(bandwidth=1e9)
    producer = Placement("a", "gpu_0", "cuda", "float16", "k", 0.01, output_bytes=4_000_000)
    consumer = Placement("b", "gpu_1", "cuda", "float16", "k", 0.01, depends_on=("a",))
    link_free: dict[str, float] = {}
    hop1, meta1 = _schedule_transfer(
        machine, producer, consumer, ready_at=0.0, link_free_at=dict(link_free), contention_factor=1.0
    )
    hop2, meta2 = _schedule_transfer(
        machine, producer, consumer, ready_at=0.0, link_free_at=dict(link_free), contention_factor=2.0
    )
    assert hop2 == pytest.approx(2.0 * hop1)
    assert meta2["contention_factor"] == 2.0
    assert meta2["latency_s"] == pytest.approx(2.0 * meta1["latency_s"])


def test_cross_device_transfer_counts_destination_residency() -> None:
    """Transferred activations must land in destination memory until consumer ends."""
    machine = _two_gpu_machine(bandwidth=1e9)
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("a", "gpu_0", "cuda", "float16", "k", 0.01, output_bytes=4_000_000),
            Placement(
                "b",
                "gpu_1",
                "cuda",
                "float16",
                "k",
                0.01,
                depends_on=("a",),
                output_bytes=500_000,
                state_bytes=1_000_000,
            ),
        ],
        decisions=[],
        devices_used=("gpu_0", "gpu_1"),
        communication_backend="nccl",
        predicted_latency_s=0.0,
    )
    result = simulate_plan(plan, machine)
    # Landed transfer (4M) + consumer state (1M) + consumer output (0.5M) at end.
    assert result.peak_bytes["vram_1"] >= 5_000_000
    assert any(e.get("event") == "release" and e.get("memory") == "vram_1" for e in result.release_events)


def test_over_capacity_emits_eviction_pressure() -> None:
    """When live bytes exceed allocatable, simulator must mark eviction pressure."""
    machine = _two_gpu_machine()
    machine.memory["vram_0"].allocatable_bytes = 3_000_000
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("a", "gpu_0", "cuda", "float16", "k", 0.1, output_bytes=4_000_000),
        ],
        decisions=[],
        devices_used=("gpu_0",),
        communication_backend="none",
        predicted_latency_s=0.1,
    )
    result = simulate_plan(plan, machine)
    assert any(e.get("event") == "eviction_pressure" for e in result.timeline)
    assert result.simulated is True


def test_overlapping_shared_memory_state_stacks_in_peak() -> None:
    """Two concurrent devices on one memory pool must both count state in peak."""
    machine = ResourceGraph(fingerprint="shared-ram")
    machine.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "host_ram"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=32 << 30,
            allocatable_bytes=32 << 30,
        )
    )
    for i in (0, 1):
        machine.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, f"cpu_{i}"),
                compute_class=ComputeClass.CPU_NUMA_POOL,
                backend_id="cpu",
                model=f"cpu-{i}",
                vendor="cpu",
                memory_affinity=("host_ram",),
            )
        )
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="shared-ram",
        objective="latency",
        placements=[
            Placement("a", "cpu_0", "cpu", "float32", "k", 1.0, state_bytes=3_000_000),
            Placement("b", "cpu_1", "cpu", "float32", "k", 1.0, state_bytes=2_000_000),
        ],
        decisions=[],
        devices_used=("cpu_0", "cpu_1"),
        communication_backend="none",
        predicted_latency_s=1.0,
    )
    result = simulate_plan(plan, machine)
    assert result.makespan_s == pytest.approx(1.0)
    assert result.peak_bytes["host_ram"] == 5_000_000
    assert any(e.get("kind") == "region_state" for e in result.release_events)
