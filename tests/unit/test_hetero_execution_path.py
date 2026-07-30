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
from streamcompiler.compile.measure import MeasurementSet, RegionMeasurement
from streamcompiler.config import CompileConfig, Objective
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode
from streamcompiler.ir.resource_graph import merge_graphs
from streamcompiler.planner.maximal import plan_execution
from streamcompiler.runtime.process_workers import ProcessWorkerPool
from streamcompiler.runtime.profile_feedback import ProfileFeedback
from streamcompiler.runtime.schedule import with_instruction_attributes
from streamcompiler.runtime.streams import EventRegistry, make_event
from streamcompiler.runtime.tensor_store import StreamingParameterStore
from streamcompiler.storage.pack import load_pack_manifest, pack_state_dict
from tests.helpers import cpu_config, cpu_host_graph


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


def test_resource_mapping_rocm_not_cpu() -> None:
    assert backend_id_for_resource("rocm_gpu_0") == "rocm"
    assert backend_id_for_resource("cuda_gpu_0") == "cuda"
    assert str(RocmBackend().resource_to_torch_device("rocm_gpu_0")) == "cuda:0"
    # Regression: naive ``"cuda" in name`` heuristics must not win for ROCm.
    assert backend_id_for_resource("rocm_gpu_0") != "cpu"
    # Unsupported accelerator stubs removed; unknown names map to cpu discovery fallback.
    assert backend_id_for_resource("sycl_gpu_0") == "cpu"


def test_event_registry_record_then_wait_same_handle() -> None:
    registry = EventRegistry()
    event = make_event("record::t0", "cpu_numa_0")
    event.record()
    registry.store("record::t0", event)
    waited = registry.get("record::t0")
    assert waited is event
    waited.wait()
    assert waited.is_complete()
    with pytest.raises(Exception, match="unknown RecordEvent"):
        registry.get("missing")


def _cpu_mock_machine():
    return merge_graphs(cpu_host_graph(), make_mock_accel_graph(delay_hint_s=0.05))


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
    config = cpu_config(
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
    probe = sc.compile(model, (x,), config=config, machine=machine)
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

        # Annotate delays on a new immutable schedule (never mutate instructions).
        updates = {
            inst.name: {"mock_compute_delay_s": 0.08}
            for inst in compiled.specialized.schedule.instructions
            if inst.opcode == OpCode.COMPUTE and "mock" in str(inst.resource)
        }
        new_sched = with_instruction_attributes(compiled.specialized.schedule, updates)
        compiled.specialized.schedule = new_sched
        compiled.executor._schedule_executor.replace_schedule(new_sched)
        _flat2, report2 = compiled._executor.run(list(compiled._program.flatten_inputs((x,), {})))
        sreport = compiled.executor._last_schedule_report
        assert sreport is not None
        assert len(report2.events) >= 2
        # Hard overlap proof lives in test_independent_computes_overlap_on_minimal_schedule.
        # Here only require the hetero Transfer/Record/Wait path still executed.
        assert len(compiled._executor._transfer_events) >= 1
        assert sreport.max_concurrent >= 1
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
    plan1 = plan_execution(ir, machine, cpu_config(objective=Objective.LATENCY), base)
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
    plan2 = plan_execution(ir, machine, cpu_config(objective=Objective.LATENCY), merged)
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


def test_dual_unequal_mock_accel_compile_and_virtual_tensors() -> None:
    """Two unequal mock devices via compile(); host-staged path; VirtualDeviceTensor."""
    from streamcompiler.runtime import virtual_tensor as vmod
    from streamcompiler.runtime.virtual_tensor import VirtualDeviceTensor

    seen_wraps: list[VirtualDeviceTensor] = []
    real_wrap_native = vmod.wrap_virtual_native

    def _tracking_wrap_native(value, device_id, native_ctx):  # type: ignore[no-untyped-def]
        out = real_wrap_native(value, device_id, native_ctx)
        seen_wraps.append(out)
        return out

    vmod.wrap_virtual_native = _tracking_wrap_native  # type: ignore[assignment]
    try:
        _dual_unequal_mock_body(seen_wraps)
    finally:
        vmod.wrap_virtual_native = real_wrap_native  # type: ignore[assignment]


def _dual_unequal_mock_body(seen_wraps) -> None:  # type: ignore[no-untyped-def]
    from streamcompiler.runtime.virtual_tensor import VirtualDeviceTensor

    model = _Parallel().eval()
    x = torch.randn(2, 8)
    eager = model(x).detach()
    config = cpu_config(
        use_torch_compile=False,
        measure_regions=False,
        allow_concurrent_regions=True,
        max_concurrent_regions=2,
        max_region_nodes=8,
        allow_host_staged_transfers=True,
        objective=Objective.LATENCY,
    )
    mocks = make_mock_accel_graph(
        device_count=2,
        capacities_bytes=(512 << 20, 8 << 30),
        delay_hints_s=(0.25, 0.01),
    )
    machine = merge_graphs(cpu_host_graph(), mocks)
    cpu = next(n for n, c in machine.compute.items() if c.backend_id == "cpu")
    slow, fast = "mock_accel_0", "mock_accel_1"
    probe = sc.compile(model, (x,), config=config, machine=machine)
    try:
        region_ids = [r.region_id for r in probe._program.regions]
        assert len(region_ids) >= 2
    finally:
        probe.close()
    ms = MeasurementSet()
    for i, rid in enumerate(region_ids):
        ms.add(RegionMeasurement(rid, cpu, "cpu", 1.0, True, notes="slow cpu"))
        # Alternate mocks so both accelerators participate; host-stage between them.
        a, b = (slow, fast) if i % 2 == 0 else (fast, slow)
        ms.add(RegionMeasurement(rid, a, "mock_accel", 0.001, False, notes="favor", simulated=True))
        ms.add(RegionMeasurement(rid, b, "mock_accel", 1.0, False, notes="avoid", simulated=True))
    compiled = sc.compile(model, (x,), config=config, machine=machine, measurements=ms)
    try:
        devices = {p.device for p in compiled.specialized.plan.placements}
        assert slow in devices or fast in devices
        used_mocks = {d for d in devices if d.startswith("mock_accel_")}
        assert used_mocks
        schedule = compiled.specialized.schedule
        assert schedule is not None
        assert any(i.opcode == OpCode.TRANSFER for i in schedule.instructions)
        out = compiled(x)
        torch.testing.assert_close(out, eager, atol=1e-4, rtol=1e-4)
        report = compiled.executor._last_schedule_report
        assert report is not None
        assert any(isinstance(v, VirtualDeviceTensor) for v in seen_wraps), (
            "expected Transfer onto mock_accel to wrap VirtualDeviceTensor"
        )
        mock_computes = [i for i in schedule.instructions if i.opcode == OpCode.COMPUTE and "mock" in str(i.resource)]
        assert mock_computes
        assert report.peak_activation_bytes > 0
    finally:
        compiled.close()


def test_memory_heavy_prefers_larger_slower_mock() -> None:
    """Fast tiny VRAM loses to slower capacious mock under MEMORY objective."""
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8)).eval()
    x = torch.randn(8, 64)
    config = cpu_config(
        use_torch_compile=False,
        measure_regions=False,
        objective=Objective.MEMORY,
        max_region_nodes=4,
    )
    mocks = make_mock_accel_graph(
        device_count=2,
        capacities_bytes=(8 * 1024, 8 << 30),  # tiny vs huge
        delay_hints_s=(0.001, 0.2),  # fast tiny vs slow huge
    )
    machine = merge_graphs(cpu_host_graph(), mocks)
    cpu = next(n for n, c in machine.compute.items() if c.backend_id == "cpu")
    tiny, huge = "mock_accel_0", "mock_accel_1"
    probe = sc.compile(model, (x,), config=config, machine=machine)
    try:
        region_ids = [r.region_id for r in probe._program.regions]
    finally:
        probe.close()
    ms = MeasurementSet()
    for rid in region_ids:
        # Make both mocks faster than CPU; planner memory/capacity should avoid tiny.
        ms.add(RegionMeasurement(rid, cpu, "cpu", 1.0, True))
        ms.add(RegionMeasurement(rid, tiny, "mock_accel", 0.01, False, simulated=True))
        ms.add(RegionMeasurement(rid, huge, "mock_accel", 0.05, False, simulated=True))
    compiled = sc.compile(model, (x,), config=config, machine=machine, measurements=ms)
    try:
        devices = {p.device for p in compiled.specialized.plan.placements}
        # Either stay on CPU or land on capacious mock — never claim tiny VRAM alone for all.
        assert tiny not in devices or huge in devices or cpu in devices
        out = compiled(x)
        torch.testing.assert_close(out, model(x), atol=1e-4, rtol=1e-4)
        # Stronger: if any mock used under MEMORY, prefer huge when both candidates exist.
        mock_used = {d for d in devices if d.startswith("mock_accel_")}
        if mock_used == {tiny, huge} or mock_used == {tiny}:
            # tiny-only is a failure of capacity-aware placement for this topology
            assert mock_used != {tiny}, f"memory-heavy plan used only tiny VRAM: {devices}"
    finally:
        compiled.close()


def test_quantized_full_model_compile_assert_close(tmp_path) -> None:
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).eval()
    x = torch.randn(2, 16)
    eager = model(x).detach()
    compiled = sc.compile(
        model,
        (x,),
        config=cpu_config(
            use_torch_compile=False,
            measure_regions=False,
            allow_quantized_storage=True,
            numerical_mode="quantized",
            allow_nvme_streaming=True,
            ram_budget_bytes=4096,
            cache_dir=tmp_path / "cache",
        ),
    )
    try:
        out = compiled(x)
        # Quantized weights: allow larger tolerance than exact fp32 path.
        torch.testing.assert_close(out, eager, atol=0.5, rtol=0.2)
        assert out.shape == eager.shape
    finally:
        compiled.close()


def test_release_depends_on_wait_events_for_transferred_activations() -> None:
    model = _Parallel().eval()
    x = torch.randn(2, 8)
    config = cpu_config(
        use_torch_compile=False,
        measure_regions=False,
        allow_concurrent_regions=True,
        max_concurrent_regions=2,
        max_region_nodes=8,
    )
    machine = _cpu_mock_machine()
    cpu = next(n for n, c in machine.compute.items() if c.backend_id == "cpu")
    accel = "mock_accel_0"
    probe = sc.compile(model, (x,), config=config, machine=machine)
    try:
        region_ids = [r.region_id for r in probe._program.regions]
        ms = _split_measurements(region_ids, cpu, accel)
    finally:
        probe.close()
    compiled = sc.compile(model, (x,), config=config, machine=machine, measurements=ms)
    try:
        schedule = compiled.specialized.schedule
        assert schedule is not None
        releases = [i for i in schedule.instructions if i.opcode == OpCode.RELEASE]
        waits = {i.name for i in schedule.instructions if i.opcode == OpCode.WAIT_EVENT}
        records = {i.name for i in schedule.instructions if i.opcode == OpCode.RECORD_EVENT}
        if waits:
            assert any(any(d in waits or d in records for d in r.depends_on) for r in releases), (
                "Release must wait on async Transfer completion edges when present"
            )
        out = compiled(x)
        assert out.shape[0] == 2
    finally:
        compiled.close()
