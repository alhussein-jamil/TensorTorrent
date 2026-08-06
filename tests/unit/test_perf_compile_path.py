"""Regression coverage for compile-path perf work."""

from __future__ import annotations

import torch
import torch.nn as nn

from tensortorrent.backends.base import KernelCandidate
from tensortorrent.compile.measure import capture_region_inputs, measure_regions_on_devices
from tensortorrent.config import CompileConfig
from tensortorrent.hardware.discovery import discover_resource_graph
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
from tensortorrent.planner import search as search_mod
from tensortorrent.planner.search import search_placements


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(16, 16)
        self.b = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


def test_specialize_timing_present() -> None:
    import tensortorrent as tt

    model = _Tiny().eval()
    x = torch.randn(2, 16)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=CompileConfig(allow_gpu=False, use_torch_compile=False, measure_regions=True),
    )
    timing = compiled.specialized.profile.get("specialize_timing") or {}
    assert "total_s" in timing
    assert timing["total_s"] > 0
    assert "capture_s" in timing or "measure_s" in timing


def test_capture_time_regions_matches_keys() -> None:
    import tensortorrent as tt

    model = _Tiny().eval()
    x = torch.randn(2, 16)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=CompileConfig(allow_gpu=False, use_torch_compile=False, measure_regions=True),
    )
    program = compiled.program
    captured, times = capture_region_inputs(program, [x], time_regions=True)
    assert set(captured) == set(times)
    assert all(t >= 0 for t in times.values())


def test_measure_workers_serial_matches_parallel_keys() -> None:
    import tensortorrent as tt

    model = _Tiny().eval()
    x = torch.randn(2, 16)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=CompileConfig(allow_gpu=False, use_torch_compile=False, measure_regions=True),
    )
    program = compiled.program
    region_inputs = capture_region_inputs(program, [x])
    machine = discover_resource_graph()
    devices = [d for d in machine.compute.values() if str(d.backend_id) in {"cpu", "cpu_numa"}]
    if len(devices) < 1:
        return
    serial = measure_regions_on_devices(program, region_inputs, devices, iters=1, workers=1)
    parallel = measure_regions_on_devices(program, region_inputs, devices, iters=1, workers=0)
    assert set(serial.by_region) == set(parallel.by_region)
    for region_id in serial.by_region:
        assert set(serial.by_region[region_id]) == set(parallel.by_region[region_id])


def test_config_new_perf_knobs_roundtrip() -> None:
    cfg = CompileConfig(measure_workers=2, region_compile_workers=2, planner_parallel_subsets=True)
    restored = CompileConfig.from_json_dict(cfg.to_json_dict())
    assert restored.measure_workers == 2
    assert restored.region_compile_workers == 2
    assert restored.planner_parallel_subsets is True
    defaults = CompileConfig()
    assert defaults.region_compile_workers == 1
    assert defaults.planner_parallel_subsets is False


def test_incremental_local_search_matches_full_replay() -> None:
    """Incremental prefix reuse must pick the same placements as full replay."""
    graph = HeterogeneousGraph(name="chain", outputs=("y",))
    graph.add_tensor(TensorMeta("x", (1000,), "uint8", size_bytes=1000, kind="input"))
    graph.add_tensor(TensorMeta("mid", (1000,), "uint8", size_bytes=1000, kind="activation"))
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
    machine = ResourceGraph(fingerprint="inc-search", backends_present=("mock_accel",))
    for index in range(2):
        device = f"mock_accel_{index}"
        memory = f"mock_vram_{index}"
        machine.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, memory),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=10_000,
                allocatable_bytes=10_000,
                attached_compute=(device,),
            )
        )
        machine.add_compute(
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
    machine.add_link(
        TransferLink(
            id=ResourceId(ResourceKind.LINK, "mock_vram_0->mock_vram_1"),
            link_class=LinkClass.PCIE,
            source="mock_vram_0",
            destination="mock_vram_1",
            bidirectional=True,
            measured=True,
            latency_s=0.0,
            bytes_per_s=1e6,
        )
    )

    def cand(region: str, device: str, latency: float) -> KernelCandidate:
        return KernelCandidate(
            region_id=region,
            device=device,
            backend_id="mock_accel",
            kernel_id=f"{region}:{device}",
            dtype="float32",
            estimated_latency_s=latency,
            attributes={"measured": True},
        )

    candidates = {
        "r0": [cand("r0", "mock_accel_0", 0.01), cand("r0", "mock_accel_1", 0.03)],
        "r1": [cand("r1", "mock_accel_0", 0.02), cand("r1", "mock_accel_1", 0.015)],
    }
    byte_counts = {"r0": (1000, 0), "r1": (4, 0)}
    devices = {"mock_accel_0", "mock_accel_1"}
    cfg = CompileConfig(planner_beam_width=16, planner_local_search_iters=3, planner_candidates_per_device=2)

    incremental = search_placements(graph, machine, candidates, devices, byte_counts, cfg)
    assert incremental is not None

    orig = search_mod._incremental_evaluate_assignment

    def full_replay(
        assignment,
        *,
        change_index,
        alternate,
        prefix_states,
        order,
        dependencies,
        byte_counts,
        edge_bytes,
        machine,
        capacities,
        allow_host_staged,
    ):  # noqa: ANN001
        trial = list(assignment)
        trial[change_index] = alternate
        return search_mod._evaluate_assignment(
            trial,
            order=order,
            dependencies=dependencies,
            byte_counts=byte_counts,
            edge_bytes=edge_bytes,
            initial_consumers=dict(prefix_states[0].remaining_consumers),
            machine=machine,
            capacities=capacities,
            allow_host_staged=allow_host_staged,
        )

    search_mod._incremental_evaluate_assignment = full_replay  # type: ignore[assignment]
    try:
        full = search_placements(graph, machine, candidates, devices, byte_counts, cfg)
    finally:
        search_mod._incremental_evaluate_assignment = orig

    assert full is not None
    assert [(p.region_id, p.device, p.kernel_id) for p in incremental.placements] == [
        (p.region_id, p.device, p.kernel_id) for p in full.placements
    ]
    assert abs(incremental.latency_s - full.latency_s) < 1e-12
