"""Regression: measurement cache keys must not mix devices or shapes."""

from __future__ import annotations

from typing import Any

import torch

from streamcompiler.backends.base import BenchmarkResult, KernelCandidate, RegionSource
from streamcompiler.compile.measure import measure_regions_on_devices, profiling_cache_key
from streamcompiler.compile.regions import Region, RegionProgram, ValueSpec
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

    def no_profiler(_backend_id: str) -> Any:
        raise NotImplementedError("force stub benchmark_region path")

    monkeypatch.setattr("streamcompiler.backends.backend_by_id", fake_backend_by_id)
    monkeypatch.setattr("streamcompiler.backends.profiler.profiler_for_backend", no_profiler)
    program = _program()
    example = (torch.zeros(2, 4),)
    devices = [_device("cpu_numa_0"), _device("cpu_numa_1")]
    results = measure_regions_on_devices(program, {"region_0": example}, devices, iters=1)
    assert stub.calls == ["cpu_numa_0", "cpu_numa_1"]
    assert results.get("region_0", "cpu_numa_0") is not None
    assert results.get("region_0", "cpu_numa_1") is not None
    assert results.get("region_0", "cpu_numa_0").latency_s == 0.01
    assert results.get("region_0", "cpu_numa_1").latency_s == 0.99


def test_measure_cache_preserves_simulated_flag(monkeypatch: Any) -> None:
    import importlib
    from dataclasses import replace

    measure_mod = importlib.import_module("streamcompiler.compile.measure")

    class _SimProfiler:
        backend_id = "mock_accel"
        calls = 0

        def profile_region(self, *args: Any, **kwargs: Any) -> Any:
            from streamcompiler.backends.profiler import ProfileRecord

            self.calls += 1
            return ProfileRecord(
                device_fingerprint="fp",
                region_graph_hash="h",
                shape=((2, 4),),
                dtype=("float32",),
                layout="contiguous",
                thread_configuration="1",
                backend_implementation="mock",
                warm_up_count=0,
                sample_count=1,
                median_s=0.123,
                dispersion_s=0.0,
                workspace_memory_bytes=0,
                measured=False,
                simulated=True,
                kind="region",
                notes=("simulated",),
            )

    profiler = _SimProfiler()

    class _Backend:
        def available(self) -> bool:
            return True

    monkeypatch.setattr(
        "streamcompiler.backends.backend_by_id",
        lambda backend_id: _Backend() if backend_id == "mock_accel" else None,
    )
    monkeypatch.setattr(
        "streamcompiler.backends.profiler.profiler_for_backend",
        lambda _bid: profiler,
    )
    monkeypatch.setattr(measure_mod, "profiling_cache_key", lambda *a, **k: "const-key")
    base = _program()
    r0 = base.regions[0]
    r1 = replace(r0, region_id="region_1", submodule="r1", outputs=("z",))
    root = torch.nn.Module()
    root.add_module("r0", torch.nn.Identity())
    root.add_module("r1", torch.nn.Identity())
    program = replace(
        base,
        root=root,
        regions=(r0, r1),
        values={
            **base.values,
            "z": ValueSpec(name="z", shape=(2, 4), dtype="float32", nbytes=32, kind="activation"),
        },
    )
    example = (torch.zeros(2, 4),)
    d0 = replace(_device("mock_accel_0"), backend_id="mock_accel")
    results = measure_regions_on_devices(
        program,
        {"region_0": example, "region_1": example},
        [d0],
        iters=1,
    )
    assert profiler.calls == 1
    m0 = results.get("region_0", "mock_accel_0")
    m1 = results.get("region_1", "mock_accel_0")
    assert m0 is not None and m1 is not None
    assert m0.measured is False and m0.simulated is True
    assert m1.measured is False and m1.simulated is True
    assert "cache hit" in m1.notes
