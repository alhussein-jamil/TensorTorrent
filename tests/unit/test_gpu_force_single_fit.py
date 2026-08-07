"""Oversized models must stay partitioned so GPU streaming stays placeable."""

from __future__ import annotations

import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.compile.fit import exported_parameter_bytes, should_force_single_region
from tensortorrent.config import CompileConfig, Objective
from tensortorrent.hardware.discovery import discover_resource_graph
from tensortorrent.ir.graph import Instruction, OpCode
from tensortorrent.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
)
from tensortorrent.planner.maximal import _region_prior_work_units, _scaled_prior


class DeepMlp(nn.Module):
    """Wide MLP whose parameter footprint is easy to size against a fake VRAM cap."""

    def __init__(self, width: int = 512, depth: int = 24) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, 8))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fake_cpu_gpu_machine(*, vram_bytes: int) -> ResourceGraph:
    graph = ResourceGraph(fingerprint="test-gpu-fuse")
    graph.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "numa_ram_0"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=64 << 30,
            allocatable_bytes=64 << 30,
            attached_compute=("cpu_numa_0",),
        )
    )
    graph.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "cuda_vram_0"),
            memory_class=MemoryClass.DEVICE_VRAM,
            capacity_bytes=vram_bytes,
            allocatable_bytes=vram_bytes,
            attached_compute=("cuda_gpu_0",),
        )
    )
    graph.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "cpu_numa_0"),
            compute_class=ComputeClass.CPU_NUMA_POOL,
            backend_id="cpu",
            model="cpu",
            memory_affinity=("numa_ram_0",),
        )
    )
    graph.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "cuda_gpu_0"),
            compute_class=ComputeClass.DISCRETE_GPU,
            backend_id="cuda",
            model="fake-gpu",
            memory_affinity=("cuda_vram_0",),
        )
    )
    return graph


def test_exported_parameter_bytes_counts_state_dict() -> None:
    model = _Tiny().eval()
    exported = torch.export.export(model, (torch.randn(2, 16),))
    expected = sum(p.numel() * p.element_size() for p in model.parameters())
    assert exported_parameter_bytes(exported) == expected


def test_force_single_skipped_when_params_exceed_vram_budget() -> None:
    model = DeepMlp().eval()
    x = torch.randn(2, 512)
    exported = tt.capture_module(model, (x,))
    param_bytes = exported_parameter_bytes(exported)
    assert param_bytes > 0
    tiny_vram = max(1, param_bytes // 8)
    machine = _fake_cpu_gpu_machine(vram_bytes=tiny_vram)
    config = CompileConfig(
        max_concurrent_regions=1,
        allow_concurrent_regions=True,
        allow_gpu=True,
        vram_budget_bytes=tiny_vram,
        measure_regions=False,
        validate_numerics=False,
        use_torch_compile=False,
    )
    assert should_force_single_region(config, machine, parameter_bytes=param_bytes) is False


def test_force_single_kept_for_small_model_on_gpu_host() -> None:
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    x = torch.randn(2, 16)
    exported = tt.capture_module(model, (x,))
    machine = discover_resource_graph()
    config = CompileConfig(
        max_concurrent_regions=1,
        allow_concurrent_regions=True,
        measure_regions=False,
        validate_numerics=False,
        use_torch_compile=False,
    )
    assert should_force_single_region(config, machine, parameter_bytes=exported_parameter_bytes(exported)) is True


def test_ram_budget_that_fits_params_does_not_force_streaming_shards() -> None:
    """Resident path: ram_budget >= params must not invent a streaming region cap."""
    from tensortorrent.compile.fit import region_state_budget, streaming_region_budget

    model = _Tiny().eval()
    exported = torch.export.export(model, (torch.randn(2, 16),))
    param_bytes = exported_parameter_bytes(exported)
    machine = _fake_cpu_gpu_machine(vram_bytes=8 << 30)
    config = CompileConfig(
        ram_budget_bytes=param_bytes * 4,
        allow_nvme_streaming=True,
        allow_gpu=True,
        prefetch_distance=2,
        max_concurrent_regions=1,
        allow_concurrent_regions=True,
        measure_regions=False,
        validate_numerics=False,
        use_torch_compile=False,
    )
    assert streaming_region_budget(config, parameter_bytes=param_bytes) is None
    # Accelerator fraction still applies; streaming divisor must not tighten further.
    budget = region_state_budget(config, machine, parameter_bytes=param_bytes)
    accel_only = region_state_budget(
        CompileConfig(allow_gpu=True, ram_budget_bytes=None),
        machine,
        parameter_bytes=param_bytes,
    )
    assert budget == accel_only
    assert should_force_single_region(config, machine, parameter_bytes=param_bytes) is True


def test_region_prior_scales_with_node_count() -> None:
    """Unmeasured giant regions must not be priced like a tiny Linear sample."""
    small = Instruction(
        opcode=OpCode.COMPUTE,
        name="region_small",
        attributes={"node_count": 1},
    )
    huge = Instruction(
        opcode=OpCode.COMPUTE,
        name="region_huge",
        attributes={"node_count": 2820},
    )
    machine = discover_resource_graph()
    cpu = next(iter(machine.compute.values()))
    t_small = _scaled_prior(small, cpu, "float32", None)
    t_huge = _scaled_prior(huge, cpu, "float32", None)
    assert _region_prior_work_units(huge) == 2820.0
    assert t_huge > t_small * 100
    assert t_huge > 1e-3


def test_large_model_with_vram_cap_places_on_gpu_not_cpu_only() -> None:
    """Params fit RAM but exceed VRAM → keep partitions, place on CUDA (not cpu_only)."""
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA required")

    model = DeepMlp(width=256, depth=32).eval()
    x = torch.randn(2, 256)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # Resident host store (params fit RAM) but multi-region GPU streaming required.
    vram_cap = max(1, param_bytes // 8)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(
            objective=Objective.LATENCY,
            max_concurrent_regions=1,
            allow_concurrent_regions=True,
            allow_gpu=True,
            allow_cpu=False,
            allow_nvme_streaming=True,
            vram_budget_bytes=vram_cap,
            ram_budget_bytes=param_bytes * 2,
            measure_regions=False,
            validate_numerics=False,
            use_torch_compile=False,
            prefer_direct_path=False,
        ),
    )
    explain = compiled.explain()
    assert "cuda_gpu_0" in compiled.specialized.plan.devices_used
    assert compiled.specialized.plan.strategy != "cpu_only"
    assert "strategy: cpu_only" not in explain
    assert len(compiled.regions) >= 2
    assert compiled.specialized.plan.search_statistics.get("planner_engine") == "rust"
    compiled.close()
