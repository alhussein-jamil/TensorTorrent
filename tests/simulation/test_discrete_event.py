"""Simulator tests for ExecutableSchedule DAGs (via simulate_plan wrapper)."""

from __future__ import annotations

import pytest

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
from tensortorrent.planner.cost.contention import concurrent_slowdown
from tensortorrent.planner.cost.transfer import TransferModel, transfer_time
from tensortorrent.planner.maximal import ExecutionPlan, Placement
from tensortorrent.runtime.simulator import simulate_plan


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
        strategy="multi_gpu",
    )
    result = simulate_plan(plan, machine)
    assert result.makespan_s == pytest.approx(1.0, rel=0.05)
    assert result.makespan_s < 2.0
    assert result.instruction_count >= 2


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
    # Load state + Compute output (state not double-counted).
    assert result.peak_bytes["vram_0"] == 5_000_000
    assert result.peak_bytes["vram_1"] == 2_500_000
    assert 1_048_576 not in result.peak_bytes.values()


def test_cross_device_transfer_scales_with_output_bytes() -> None:
    """Larger producer output must lengthen schedule transfer time on the same link."""
    machine = _two_gpu_machine(bandwidth=1e9)
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
    assert large_result.bytes_transferred > small_result.bytes_transferred
    assert large_result.makespan_s > small_result.makespan_s
    large_xfer = max((e["end_s"] - e["start_s"]) for e in large_result.transfer_events)
    assert large_xfer == pytest.approx(0.008, rel=0.25)


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
    assert result.bytes_transferred == 0
    assert result.transfer_events == []
    assert result.makespan_s == pytest.approx(0.2, rel=0.05)


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
    assert result.peak_bytes["vram_0"] <= 5_500_000
    assert result.peak_bytes["vram_0"] >= 4_000_000
    assert result.transfer_events == []


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
    """Contention model still lengthens analytic hops (schedule uses factor=1; cost model tested here)."""
    machine = _two_gpu_machine(bandwidth=1e9)
    link = machine.links["vram_0->vram_1"]
    model = TransferModel(
        source=link.source,
        destination=link.destination,
        alpha_s=float(link.latency_s or 0.0),
        beta_bytes_per_s=link.bytes_per_s,
        measured=True,
    )
    base = transfer_time(model, link.source, link.destination, 4_000_000)
    factor = concurrent_slowdown(active_compute=1, active_transfers=2, active_storage=0).transfer
    assert factor >= 1.0
    assert base * factor >= base


def test_cross_device_transfer_counts_destination_residency() -> None:
    """Transferred activations land in destination memory for the consumer."""
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
    assert result.peak_bytes["vram_1"] >= 5_000_000
    assert (
        any(e.get("event") in {"Release", "release"} for e in result.release_events)
        or result.peak_bytes["vram_1"] >= 5_000_000
    )


def test_over_capacity_is_infeasible() -> None:
    """Memory overflow is not a valid simulation — InfeasibleMemory / raise."""
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
    with pytest.raises(ValueError, match="infeasible"):
        simulate_plan(plan, machine)


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
    assert result.makespan_s == pytest.approx(1.0, rel=0.05)
    assert result.peak_bytes["host_ram"] == 5_000_000
