"""Heterogeneous execution path: schedule SoT, events, mapping, workers, profile."""

from __future__ import annotations

import sys
import time

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.backends import backend_id_for_resource
from streamcompiler.backends.mock_accel import make_mock_accel_graph
from streamcompiler.backends.rocm import RocmBackend
from streamcompiler.backends.sycl import SyclBackend
from streamcompiler.compile.measure import MeasurementSet, RegionMeasurement
from streamcompiler.config import CompileConfig, Objective
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode
from streamcompiler.ir.resource_graph import merge_graphs
from streamcompiler.planner.maximal import plan_execution
from streamcompiler.runtime.async_events import EventRegistry, make_event
from streamcompiler.runtime.process_workers import ProcessWorkerPool
from streamcompiler.runtime.profile_feedback import ProfileFeedback
from streamcompiler.runtime.tensor_store import StreamingParameterStore
from streamcompiler.storage.pack import load_pack_manifest, pack_state_dict


class _Parallel(nn.Module):
    """Two independent matmuls so CPU and mock-accel regions can overlap."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(8, 8)
        self.right = nn.Linear(8, 8)
        self.mix = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(torch.cat([torch.relu(self.left(x)), torch.relu(self.right(x))], dim=-1))


def test_training_backward_populates_input_and_param_grads() -> None:
    model = nn.Linear(4, 2)
    x = torch.randn(2, 4, requires_grad=True)
    compiled = sc.compile(
        model,
        (torch.randn(2, 4),),
        config=sc.CompileConfig(allow_training=True, use_torch_compile=False, measure_regions=False),
    )
    try:
        out = compiled(x)
        assert out.requires_grad
        out.sum().backward()
        assert x.grad is not None
        assert any(p.grad is not None for p in compiled.parameters())
    finally:
        compiled.close()


def test_resource_mapping_rocm_and_sycl_not_cpu() -> None:
    assert backend_id_for_resource("rocm_gpu_0") == "rocm"
    assert backend_id_for_resource("sycl_gpu_1") == "sycl"
    assert backend_id_for_resource("cuda_gpu_0") == "cuda"
    assert str(RocmBackend().resource_to_torch_device("rocm_gpu_0")) == "cuda:0"
    assert str(SyclBackend().resource_to_torch_device("sycl_gpu_1")) == "xpu:1"
    # Regression: naive ``"cuda" in name`` / missing ``xpu`` heuristics must not win.
    assert backend_id_for_resource("rocm_gpu_0") != "cpu"
    assert backend_id_for_resource("sycl_gpu_0") != "cpu"


def test_event_registry_record_then_wait_same_handle() -> None:
    registry = EventRegistry()
    event = make_event("record::t0", "cpu_numa_0")
    event.record()
    registry.store("record::t0", event)
    waited = registry.get("record::t0")
    assert waited is event
    waited.wait()
    assert waited.completed
    with pytest.raises(Exception, match="unknown RecordEvent"):
        registry.get("missing")


def _cpu_mock_machine():
    base = discover_resource_graph()
    return merge_graphs(base, make_mock_accel_graph(delay_hint_s=0.05))


def _split_measurements(region_ids: list[str], cpu: str, accel: str) -> MeasurementSet:
    ms = MeasurementSet()
    for i, rid in enumerate(region_ids):
        if i % 2 == 0:
            ms.add(RegionMeasurement(rid, cpu, "cpu", 0.001, True, notes="favor cpu"))
            ms.add(RegionMeasurement(rid, accel, "mock_accel", 1.0, True, notes="slow accel"))
        else:
            ms.add(RegionMeasurement(rid, cpu, "cpu", 1.0, True, notes="slow cpu"))
            ms.add(RegionMeasurement(rid, accel, "mock_accel", 0.001, True, notes="favor accel"))
    return ms


def test_hetero_schedule_cpu_plus_mock_accel_path() -> None:
    model = _Parallel().eval()
    x = torch.randn(2, 8)
    eager = model(x).detach()
    config = CompileConfig(
        use_torch_compile=False,
        measure_regions=False,
        allow_concurrent_regions=True,
        max_concurrent_regions=2,
        max_region_nodes=8,
        objective=Objective.LATENCY,
    )
    machine = _cpu_mock_machine()
    cpu = next(n for n, c in machine.compute.items() if c.backend_id == "cpu")
    accel = "mock_accel_0"
    # First compile on CPU-only to discover region ids, then compile with measurements.
    probe = sc.compile(model, (x,), config=config)
    try:
        region_ids = [r.region_id for r in probe._program.regions]
        assert len(region_ids) >= 2
        measurements = _split_measurements(region_ids, cpu, accel)
    finally:
        probe.close()

    compiled = sc.compile(
        model,
        (x,),
        config=config,
        machine=machine,
        measurements=measurements,
    )
    try:
        devices = {p.device for p in compiled.specialized.plan.placements}
        assert cpu in devices
        assert accel in devices
        schedule = compiled.specialized.schedule
        assert schedule is not None
        opcodes = [i.opcode for i in schedule.instructions]
        assert OpCode.TRANSFER in opcodes
        assert OpCode.RECORD_EVENT in opcodes
        assert OpCode.WAIT_EVENT in opcodes

        for binding in compiled.specialized.bindings.values():
            exe = binding.compiled.executable
            assert getattr(exe, "_needs_move", False) is False

        out = compiled(x)
        torch.testing.assert_close(out, eager, atol=1e-4, rtol=1e-4)

        # Force a second run through the schedule executor for transfer telemetry.
        flat, _report = compiled._executor.run(list(compiled._program.flatten_inputs((x,), {})))
        torch.testing.assert_close(flat[0], eager, atol=1e-4, rtol=1e-4)
        names = [e["name"] for e in compiled._executor._transfer_events]
        assert any(n.startswith("transfer::") for n in names)
        assert any(n.startswith("record::") for n in names)
        assert any(n.startswith("wait::") for n in names)

        from streamcompiler.backends.mock_accel import _DelayedRegion

        for rid, _binding in compiled.specialized.bindings.items():
            compiled._executor._callables[rid] = _DelayedRegion(compiled._executor._callables[rid], 0.08)
        _flat2, report2 = compiled._executor.run(list(compiled._program.flatten_inputs((x,), {})))
        seq = sum(e.duration_s for e in report2.events)
        assert len(report2.events) >= 2
        if report2.parallel_overlaps > 0 or report2.max_concurrent_regions > 1:
            assert report2.wall_time_s < seq * 0.9
        else:
            assert len(compiled._executor._transfer_events) >= 1
    finally:
        compiled.close()


def test_profile_feedback_changes_subsequent_plan() -> None:
    machine = _cpu_mock_machine()
    cpu = next(n for n, c in machine.compute.items() if c.backend_id == "cpu")
    accel = "mock_accel_0"
    ir = HeterogeneousGraph(name="fb")
    for i in range(2):
        ir.add_instruction(
            Instruction(
                opcode=OpCode.COMPUTE,
                name=f"region_{i}",
                attributes={"depends_on": () if i == 0 else ("region_0",)},
            )
        )
    ir.repeated_blocks = (("region_0",), ("region_1",))

    base = MeasurementSet()
    for rid in ("region_0", "region_1"):
        base.add(RegionMeasurement(rid, cpu, "cpu", 0.01, True))
        base.add(RegionMeasurement(rid, accel, "mock_accel", 0.02, True))
    plan1 = plan_execution(ir, machine, CompileConfig(objective=Objective.LATENCY), base)
    assert all(p.device == cpu for p in plan1.placements)

    fb = ProfileFeedback()
    fb.region_latency_s["region_0"] = 5.0
    fb.region_device["region_0"] = cpu
    fb.region_latency_s["region_1"] = 5.0
    fb.region_device["region_1"] = cpu
    fb.updates = 1
    merged = fb.merge_into_measurements(base)
    # Make accel attractive after feedback poison on CPU.
    merged.add(RegionMeasurement("region_0", accel, "mock_accel", 0.01, True))
    merged.add(RegionMeasurement("region_1", accel, "mock_accel", 0.01, True))
    plan2 = plan_execution(ir, machine, CompileConfig(objective=Objective.LATENCY), merged)
    assert any(p.device == accel for p in plan2.placements)
    assert plan1.devices_used != plan2.devices_used or plan1.placements != plan2.placements


def _add_slow(a: int, b: int) -> int:
    time.sleep(0.25)
    return a + b


def _boom() -> int:
    raise RuntimeError("worker boom")


def test_process_worker_pool_overlaps_and_survives_failure() -> None:
    pool = ProcessWorkerPool(max_workers=2)
    try:
        t0 = time.perf_counter()
        f1 = pool.submit(_add_slow, 1, 2)
        f2 = pool.submit(_add_slow, 3, 4)
        assert f1.result(timeout=30) == 3
        assert f2.result(timeout=30) == 7
        elapsed = time.perf_counter() - t0
        # Two 0.25s sleeps: sequential ≥0.5s; overlapped ~0.25–0.35s.
        assert elapsed < 0.45, f"submissions did not overlap: {elapsed:.3f}s"

        bad = pool.submit(_boom)
        with pytest.raises(Exception, match="worker boom"):
            bad.result(timeout=30)
        ok = pool.submit(_add_slow, 5, 6)
        assert ok.result(timeout=30) == 11
    finally:
        pool.shutdown()


def test_quantized_pack_loads_through_streaming_store(tmp_path) -> None:
    weight = torch.randn(32, 16)
    path = tmp_path / "q.pack"
    pack = pack_state_dict({"linear.weight": weight}, path, quantize=True)
    assert pack.metadata.get("quantize") is True
    manifest = load_pack_manifest(path)
    entry = manifest["tensors"][0]
    assert entry["compression"] == "int8_affine"
    assert entry["stored_dtype"] == "int8"
    store = StreamingParameterStore(path, {"w": "linear.weight"}, budget_bytes=1 << 20)
    try:
        loaded = store.acquire("w")
        assert loaded.shape == weight.shape
        assert loaded.dtype == torch.float32
        err = (loaded - weight).abs().max().item()
        assert err < 0.5
    finally:
        store.close()


def test_compile_process_workers_attach_pool_on_linux() -> None:
    if sys.platform != "linux":
        pytest.skip("fork process workers are Linux-only")
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    x = torch.randn(2, 8)
    compiled = sc.compile(
        model,
        (x,),
        config=sc.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_concurrent_regions=True,
            max_concurrent_regions=2,
            process_workers=2,
        ),
    )
    try:
        if compiled._executor.max_workers > 1 and len(compiled.regions) > 1:
            assert compiled._executor._process_pool is not None
        out = compiled(x)
        assert out.shape == (2, 4)
    finally:
        compiled.close()
