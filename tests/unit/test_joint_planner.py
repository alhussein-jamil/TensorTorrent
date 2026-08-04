"""Joint placement search covers transfer and memory costs."""

from __future__ import annotations

from tensortorrent.backends.base import KernelCandidate
from tensortorrent.config import CompileConfig, Objective
from tensortorrent.ir.graph import HeterogeneousGraph, Instruction, OpCode, TensorMeta
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
from tensortorrent.planner.search import search_placements


def _machine(*, capacities: tuple[int, int] = (10_000, 10_000), bandwidth: float = 1e6) -> ResourceGraph:
    graph = ResourceGraph(fingerprint="joint-search", backends_present=("mock_accel",))
    for index, capacity in enumerate(capacities):
        device = f"mock_accel_{index}"
        memory = f"mock_vram_{index}"
        graph.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, memory),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=capacity,
                allocatable_bytes=capacity,
                attached_compute=(device,),
            )
        )
        graph.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, device),
                compute_class=ComputeClass.ACCELERATOR,
                backend_id="mock_accel",
                vendor="mock",
                model=device,
                memory_affinity=(memory,),
                supported_dtypes=("float32",),
            )
        )
    graph.add_link(
        TransferLink(
            id=ResourceId(ResourceKind.LINK, "mock_vram_0->mock_vram_1"),
            link_class=LinkClass.PCIE,
            source="mock_vram_0",
            destination="mock_vram_1",
            bidirectional=True,
            measured=True,
            latency_s=0.0,
            bytes_per_s=bandwidth,
        )
    )
    return graph


def _chain(nbytes: int) -> HeterogeneousGraph:
    graph = HeterogeneousGraph(name="chain", outputs=("y",))
    graph.add_tensor(TensorMeta("x", (nbytes,), "uint8", size_bytes=nbytes, kind="input"))
    graph.add_tensor(TensorMeta("mid", (nbytes,), "uint8", size_bytes=nbytes, kind="activation"))
    graph.add_tensor(TensorMeta("y", (1,), "float32", size_bytes=4, kind="activation"))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r0", inputs=("x",), outputs=("mid",)))
    graph.add_instruction(
        Instruction(
            OpCode.COMPUTE,
            "r1",
            inputs=("mid",),
            outputs=("y",),
            attributes={"depends_on": ["r0"]},
        )
    )
    return graph


def _candidate(region: str, device: str, latency: float, *, workspace: int = 0) -> KernelCandidate:
    return KernelCandidate(
        region_id=region,
        device=device,
        backend_id="mock_accel",
        kernel_id=f"{region}:{device}",
        dtype="float32",
        estimated_latency_s=latency,
        workspace_bytes=workspace,
        attributes={"measured": True},
    )


def test_joint_search_avoids_fast_kernel_when_transfer_dominates() -> None:
    graph = _chain(1_000)
    machine = _machine(bandwidth=100.0)  # 10 seconds to cross devices
    candidates = {
        "r0": [_candidate("r0", "mock_accel_0", 0.01), _candidate("r0", "mock_accel_1", 0.03)],
        "r1": [_candidate("r1", "mock_accel_0", 0.02), _candidate("r1", "mock_accel_1", 0.001)],
    }
    result = search_placements(
        graph,
        machine,
        candidates,
        {"mock_accel_0", "mock_accel_1"},
        {"r0": (1_000, 0), "r1": (4, 0)},
        CompileConfig(
            objective=Objective.LATENCY,
            allow_cpu=False,
            planner_beam_width=16,
            planner_candidates_per_device=1,
        ),
    )
    assert result is not None
    assert [placement.device for placement in result.placements] == ["mock_accel_0", "mock_accel_0"]
    assert result.transfer_bytes == 0


def test_joint_search_rejects_device_whose_execution_peak_exceeds_capacity() -> None:
    graph = HeterogeneousGraph(name="memory", outputs=("y",))
    graph.add_tensor(TensorMeta("w", (200,), "float32", size_bytes=800, kind="parameter"))
    graph.add_tensor(TensorMeta("y", (75,), "float32", size_bytes=300, kind="activation"))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r0", inputs=("w",), outputs=("y",)))
    machine = _machine(capacities=(1_000, 2_000), bandwidth=1e9)
    candidates = {
        "r0": [
            _candidate("r0", "mock_accel_0", 0.001),
            _candidate("r0", "mock_accel_1", 0.01),
        ]
    }
    result = search_placements(
        graph,
        machine,
        candidates,
        {"mock_accel_0", "mock_accel_1"},
        {"r0": (300, 800)},
        CompileConfig(objective=Objective.LATENCY, allow_cpu=False),
    )
    assert result is not None
    assert result.placements[0].device == "mock_accel_1"
    assert result.peak_bytes["mock_accel_1"] == 1_100


def test_throughput_objective_uses_bottleneck_service_time() -> None:
    graph = _chain(10)
    machine = _machine(bandwidth=1e12)
    candidates = {
        "r0": [_candidate("r0", "mock_accel_0", 0.1), _candidate("r0", "mock_accel_1", 0.1)],
        "r1": [_candidate("r1", "mock_accel_0", 0.1), _candidate("r1", "mock_accel_1", 0.1)],
    }
    result = search_placements(
        graph,
        machine,
        candidates,
        {"mock_accel_0", "mock_accel_1"},
        {"r0": (10, 0), "r1": (4, 0)},
        CompileConfig(
            objective=Objective.THROUGHPUT,
            allow_cpu=False,
            target_inflight_requests=8,
            planner_beam_width=16,
        ),
    )
    assert result is not None
    assert len({placement.device for placement in result.placements}) == 2
    assert result.throughput_per_s >= 9.9
