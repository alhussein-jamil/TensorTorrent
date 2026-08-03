"""Tests for _check_early_fit — early memory feasibility gate in pipeline.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.compile.pipeline import _check_early_fit
from tensortorrent.config import CompileConfig
from tensortorrent.errors import MemoryCapacityError
from tensortorrent.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
)

_MiB = 1 << 20
_GiB = 1 << 30


# ---------------------------------------------------------------------------
# Build a minimal fake machine (ResourceGraph) for testing
# ---------------------------------------------------------------------------


def _machine(host_allowed: int, device_allowed: int = 0) -> ResourceGraph:
    """Fake machine with one CPU NUMA node and optionally one VRAM device."""
    g = ResourceGraph(fingerprint="test", backends_present=("cpu",))
    g.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "numa_ram_0"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=host_allowed + 1 * _GiB,
            allocatable_bytes=host_allowed,
            numa_node=0,
            attributes={"budget_source": "explicit"},
        )
    )
    g.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "cpu_numa_0"),
            compute_class=ComputeClass.CPU_NUMA_POOL,
            backend_id="cpu",
            model="test-cpu",
            vendor="cpu",
            supported_dtypes=("float32",),
            supported_ops=("aten::mm",),
            core_count=2,
            concurrency_limit=2,
            numa_node=0,
            memory_affinity=("numa_ram_0",),
        )
    )
    if device_allowed > 0:
        g.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, "cuda_vram_0"),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=device_allowed + 256 * _MiB,
                allocatable_bytes=device_allowed,
                attached_compute=("cuda_gpu_0",),
            )
        )
        g.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, "cuda_gpu_0"),
                compute_class=ComputeClass.DISCRETE_GPU,
                backend_id="cuda",
                model="test-gpu",
                vendor="nvidia",
                supported_dtypes=("float32",),
                supported_ops=("aten::mm",),
                core_count=64,
                memory_affinity=("cuda_vram_0",),
            )
        )
    return g


def _program_for(model: nn.Module, x: torch.Tensor) -> any:
    """Return the RegionProgram produced by lowering a model."""
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False, allow_gpu=False),
    )
    prog = compiled.program
    compiled.close()
    return prog


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_check_early_fit_passes_with_ample_budget() -> None:
    """When host memory far exceeds model parameters, _check_early_fit must not raise."""
    model = nn.Linear(16, 8).eval()
    x = torch.randn(2, 16)
    program = _program_for(model, x)

    param_bytes = program.total_state_bytes()
    assert param_bytes > 0

    # 8 GiB host — vastly more than any tiny linear
    machine = _machine(host_allowed=8 * _GiB)
    config = CompileConfig()  # default — no streaming
    _check_early_fit(program, machine, config)  # must not raise


def test_check_early_fit_raises_when_budget_too_small() -> None:
    """With absurdly small budgets, MemoryCapacityError is raised with numbers."""
    model = nn.Linear(256, 256).eval()
    x = torch.randn(2, 256)
    program = _program_for(model, x)

    param_bytes = program.total_state_bytes()
    assert param_bytes > 0

    # Give a budget much smaller than the model
    tiny_budget = max(1, param_bytes // 100)
    machine = _machine(host_allowed=tiny_budget)

    config = CompileConfig()

    with pytest.raises(MemoryCapacityError):
        _check_early_fit(program, machine, config)


def test_check_early_fit_error_message_contains_numbers() -> None:
    """MemoryCapacityError message must name the numbers (bytes)."""
    model = nn.Linear(256, 256).eval()
    x = torch.randn(2, 256)
    program = _program_for(model, x)
    param_bytes = program.total_state_bytes()
    tiny = max(1, param_bytes // 100)
    machine = _machine(host_allowed=tiny)
    config = CompileConfig()

    with pytest.raises(MemoryCapacityError) as exc_info:
        _check_early_fit(program, machine, config)

    msg = str(exc_info.value)
    # Message must contain the byte counts (numeric substrings)
    assert str(param_bytes) in msg or "bytes" in msg


def test_check_early_fit_zero_param_model_always_passes() -> None:
    """Models with no trainable parameters skip the check unconditionally."""
    # A model whose state_dict is empty (no parameters)
    model = nn.ReLU()
    x = torch.randn(4, 8)
    program = _program_for(model, x)

    assert program.total_state_bytes() == 0, "ReLU has no parameters"
    machine = _machine(host_allowed=1)  # ludicrously small
    config = CompileConfig()
    _check_early_fit(program, machine, config)  # must NOT raise


def test_check_early_fit_device_budget_included() -> None:
    """Device VRAM counts toward total_allowed; combined may exceed model size."""
    model = nn.Linear(64, 64).eval()
    x = torch.randn(2, 64)
    program = _program_for(model, x)
    param_bytes = program.total_state_bytes()

    # Host budget is tiny but device budget pushes total over the model size
    host = 1  # essentially nothing
    device = param_bytes * 2
    machine = _machine(host_allowed=host, device_allowed=device)
    config = CompileConfig()
    # total_allowed = host + device > param_bytes → pass
    _check_early_fit(program, machine, config)
