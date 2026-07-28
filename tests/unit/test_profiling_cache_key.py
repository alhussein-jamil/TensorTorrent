"""Regression: measurement cache keys must not mix devices or shapes."""

from __future__ import annotations

from typing import Any

import torch

from streamcompiler.backends.base import BenchmarkResult, KernelCandidate, RegionSource
from streamcompiler.codegen.regions import Region, RegionProgram, ValueSpec
from streamcompiler.compile.measure import measure_regions_on_devices, profiling_cache_key
from streamcompiler.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    ResourceId,
    ResourceKind,
)


class _StubBackend:
    backend_id = "cpu"
    calls: list[str] = []

    def available(self) -> bool:
        return True

    def benchmark_region(
        self,
        source: RegionSource,
        candidate: KernelCandidate,
        example_inputs: tuple[Any, ...],
        *,
        iters: int = 3,
    ) -> BenchmarkResult:
        self.calls.append(candidate.device)
        # Distinct latency per device so a cache collision would poison results.
        latency = 0.01 if candidate.device.endswith("0") else 0.99
        return BenchmarkResult(
            candidate=candidate,
            latency_s=latency,
            memory_bytes=0,
            measured=True,
            notes="stub",
        )


def _program() -> RegionProgram:
    from torch.utils import _pytree as pytree

    region = Region(
        region_id="region_0",
        submodule="r0",
        inputs=("x",),
        outputs=("y",),
        multi_output=False,
        aten_ops=("aten::linear",),
        node_count=1,
        depends_on=(),
        state_inputs=(),
        output_bytes=32,
    )
    root = torch.nn.Module()
    root.add_module("r0", torch.nn.Identity())
    return RegionProgram(
        graph_name="cache",
        root=root,
        regions=(region,),
        values={
            "x": ValueSpec(name="x", shape=(2, 4), dtype="float32", nbytes=32, kind="input"),
            "y": ValueSpec(name="y", shape=(2, 4), dtype="float32", nbytes=32, kind="activation"),
        },
        user_inputs=("x",),
        state_bindings={},
        output_refs=(("value", "y"),),
        in_spec=pytree.tree_structure(((torch.zeros(2, 4),), {})),
        out_spec=pytree.tree_structure(torch.zeros(2, 4)),
        metadata={},
    )


def _device(name: str, *, threads: int = 4) -> ComputeResource:
    return ComputeResource(
        id=ResourceId(ResourceKind.COMPUTE, name),
        compute_class=ComputeClass.CPU_NUMA_POOL,
        backend_id="cpu",
        model=name,
        vendor="cpu",
        supported_dtypes=("float32",),
        supported_ops=("aten::linear",),
        core_count=threads,
        concurrency_limit=threads,
        attributes={"fingerprint": f"fp-{name}", "intraop_threads": threads},
    )


def test_profiling_cache_key_includes_device_shape_dtype_threads() -> None:
    source = RegionSource(
        region_id="r",
        module=torch.nn.Identity(),
        example_inputs=(torch.zeros(2, 3),),
        attributes={"dtype": "float32", "node_count": 1},
    )
    cand = KernelCandidate("r", "cpu_numa_0", "cpu", "cpu_fx", "float32")
    device = _device("cpu_numa_0", threads=8)
    key = profiling_cache_key(source, cand, device, example_inputs=source.example_inputs)
    assert "cpu_numa_0" in key
    assert "float32" in key
    assert "cpu_fx" in key
    assert "threads=8" in key or "8" in key
    other = profiling_cache_key(
        source,
        KernelCandidate("r", "cpu_numa_1", "cpu", "cpu_fx", "float32"),
        _device("cpu_numa_1", threads=8),
        example_inputs=source.example_inputs,
    )
    assert key != other


def test_measure_regions_does_not_reuse_latency_across_devices(monkeypatch: Any) -> None:
    stub = _StubBackend()
    stub.calls = []

    def fake_backend_by_id(backend_id: str) -> Any:
        return stub if backend_id == "cpu" else None

    monkeypatch.setattr("streamcompiler.backends.backend_by_id", fake_backend_by_id)
    program = _program()
    example = (torch.zeros(2, 4),)
    devices = [_device("cpu_numa_0"), _device("cpu_numa_1")]
    results = measure_regions_on_devices(program, {"region_0": example}, devices, iters=1)
    assert stub.calls == ["cpu_numa_0", "cpu_numa_1"]
    assert results.get("region_0", "cpu_numa_0") is not None
    assert results.get("region_0", "cpu_numa_1") is not None
    assert results.get("region_0", "cpu_numa_0").latency_s == 0.01
    assert results.get("region_0", "cpu_numa_1").latency_s == 0.99
