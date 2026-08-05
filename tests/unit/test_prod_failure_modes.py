"""Production failure-mode regressions for local inference readiness."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.compile.pipeline import _check_early_fit
from tensortorrent.config import CompileConfig
from tensortorrent.errors import DiskSpaceError, ExecutionCancelled, MemoryCapacityError
from tensortorrent.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
)
from tensortorrent.native import native_available
from tensortorrent.runtime.provisioning import _ensure_pack


def _tiny_machine(host_allowed: int) -> ResourceGraph:
    g = ResourceGraph(fingerprint="prod-fail", backends_present=("cpu",))
    g.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "numa_ram_0"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=host_allowed + (1 << 30),
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
    return g


def test_disk_space_precheck_raises_before_pack_write(tmp_path) -> None:
    """Pack write must refuse when free space is below needed bytes."""
    model = nn.Linear(32, 32).eval()
    x = torch.randn(2, 32)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(allow_gpu=False, use_torch_compile=False, measure_regions=False),
    )
    try:
        program = compiled.program
        portable = SimpleNamespace(name="m", packed_model_path="")
        config = CompileConfig(cache_dir=str(tmp_path))
        with (
            patch.object(type(program), "total_state_bytes", return_value=10 << 30),
            patch("tensortorrent.runtime.provisioning.shutil.disk_usage") as usage,
        ):
            usage.return_value = SimpleNamespace(total=100 << 30, used=99 << 30, free=1 << 20)
            with pytest.raises(DiskSpaceError) as exc_info:
                _ensure_pack(program, portable, config, artifact_dir=tmp_path)  # type: ignore[arg-type]
            assert exc_info.value.needed > exc_info.value.free
    finally:
        compiled.close()


def test_memory_capacity_gate_blocks_impossible_fit() -> None:
    model = nn.Linear(256, 256).eval()
    x = torch.randn(2, 256)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(allow_gpu=False, use_torch_compile=False, measure_regions=False),
    )
    try:
        program = compiled.program
        param_bytes = program.total_state_bytes()
        tiny = max(1, param_bytes // 100)
        with pytest.raises(MemoryCapacityError):
            _check_early_fit(program, _tiny_machine(tiny), CompileConfig())
    finally:
        compiled.close()


@pytest.mark.skipif(not native_available(), reason="native required")
def test_cancel_mid_forward_raises_and_allows_rerun() -> None:
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
    x = torch.randn(2, 16)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(
            allow_gpu=False,
            use_torch_compile=False,
            measure_regions=False,
            prefer_direct_path=False,
        ),
    )
    try:
        with torch.no_grad():
            expected = model(x)
        compiled(x)  # warm
        se = compiled.executor._schedule_executor

        def boom(*_a, **_k):
            raise ExecutionCancelled("Schedule execution cancelled")

        real = se._exec_compute
        se._exec_compute = boom  # type: ignore[method-assign]
        with pytest.raises(ExecutionCancelled):
            compiled(x)
        se._exec_compute = real  # type: ignore[method-assign]
        out = compiled(x)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
    finally:
        compiled.close()


def test_cpu_compile_matches_eager_parity() -> None:
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8)).eval()
    x = torch.randn(4, 64)
    with torch.no_grad():
        expected = model(x)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(allow_gpu=False, use_torch_compile=False, measure_regions=False),
    )
    try:
        out = compiled(x)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
        out2 = compiled(x)
        torch.testing.assert_close(out2, expected, atol=1e-4, rtol=1e-4)
    finally:
        compiled.close()


@pytest.mark.skipif(not native_available(), reason="native required")
def test_simulate_infeasible_surfaces_as_memory_capacity_error() -> None:
    """Native DES ValueError('…infeasible…') must become MemoryCapacityError."""
    from tensortorrent.hardware.discovery import discover_resource_graph
    from tensortorrent.runtime.simulator.discrete_event import simulate_schedule

    model = nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(allow_gpu=False, use_torch_compile=False, measure_regions=False),
    )
    try:
        schedule = compiled.specialized.schedule
        assert schedule is not None
        machine = discover_resource_graph()

        class _Boom:
            def simulate_schedule(self, *_a, **_k):
                raise ValueError("schedule infeasible: memory pinned_host_0 resident=9 allocatable=1")

        with (
            patch("tensortorrent.native.require_native", return_value=_Boom()),
            pytest.raises(MemoryCapacityError, match="infeasible"),
        ):
            simulate_schedule(schedule, machine)
    finally:
        compiled.close()
