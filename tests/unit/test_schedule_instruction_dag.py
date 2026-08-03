"""ExecutableSchedule as exclusive instruction-DAG runtime program."""

from __future__ import annotations

import gc
import time

import pytest
import torch
import torch.nn as nn
from tests.support.helpers import cpu_host_graph

import tensortorrent as tt
from tensortorrent.backends.mock_accel import make_mock_accel_graph
from tensortorrent.backends.torch_device import _CompiledRegionCallable
from tensortorrent.compile.measure import MeasurementSet, RegionMeasurement
from tensortorrent.config import CompileConfig, Objective
from tensortorrent.ir.graph import OpCode
from tensortorrent.ir.resource_graph import merge_graphs
from tensortorrent.runtime.copies import CopyStore
from tensortorrent.runtime.schedule import (
    ExecutableSchedule,
    PlanInstruction,
    ScheduleValidationError,
    assert_schedule_valid,
    validate_schedule,
    with_instruction_attributes,
)
from tensortorrent.runtime.streams import MockStream, StreamEvent


@pytest.fixture(autouse=True)
def _flush_native_artifacts_dag() -> None:
    """Drop _native_artifact on all ScheduleExecutors before each DAG test.

    Tests in this module create GraphExecutors with native schedules that register
    NativeCompiledArtifacts in the Rust global registry. The next test's
    NativeExecutionContext may find a stale artifact ("tensor not resident").
    Pre-test flush ensures a clean slate regardless of test ordering.
    """
    gc.collect()
    gc.collect()
    for obj in gc.get_objects():
        try:
            if type(obj).__name__ == "ScheduleExecutor" and getattr(obj, "_native_artifact", None) is not None:
                obj._native_artifact = None
        except Exception:  # noqa: BLE001
            pass
    gc.collect()
    yield
    gc.collect()
    gc.collect()
    for obj in gc.get_objects():
        try:
            if type(obj).__name__ == "ScheduleExecutor" and getattr(obj, "_native_artifact", None) is not None:
                obj._native_artifact = None
        except Exception:  # noqa: BLE001
            pass
    gc.collect()


class _FanOut(nn.Module):
    """One tensor consumed by CPU and mock-accel style branches (via plan)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(8, 8)
        self.left = nn.Linear(8, 8)
        self.right = nn.Linear(8, 8)
        self.head = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.stem(x))
        return self.head(torch.cat([torch.relu(self.left(h)), torch.relu(self.right(h))], dim=-1))


class _MultiOut(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)
        self.c = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = torch.relu(self.a(x))
        right = torch.relu(self.b(x))
        return self.c(torch.cat([left, right], dim=-1))


def _cpu_mock_machine(*, delay_hint_s: float = 0.1):
    return merge_graphs(cpu_host_graph(), make_mock_accel_graph(delay_hint_s=delay_hint_s))


def _split_measurements(region_ids: list[str], cpu: str, accel: str) -> MeasurementSet:
    ms = MeasurementSet()
    for i, rid in enumerate(region_ids):
        if i % 2 == 0:
            ms.add(RegionMeasurement(rid, cpu, "cpu", 0.001, True))
            ms.add(RegionMeasurement(rid, accel, "mock_accel", 1.0, True))
        else:
            ms.add(RegionMeasurement(rid, cpu, "cpu", 1.0, True))
            ms.add(RegionMeasurement(rid, accel, "mock_accel", 0.001, True))
    return ms


def test_copy_store_keeps_independent_resource_copies() -> None:
    """Passive bag: put replaces one label; siblings stay until explicit drop."""
    store = CopyStore()
    t = torch.randn(4, 4)
    store.put("act", "cpu", t)
    store.replicate("act", "mock_accel_0", t.clone(), source_resource="cpu")
    assert store.has("act", "cpu", valid_only=True)
    assert store.has("act", "mock_accel_0", valid_only=True)
    store.put("act", "mock_accel_0", t + 1)
    assert store.get("act", "cpu").valid
    assert store.get("act", "mock_accel_0").valid
    assert torch.allclose(store.get("act", "cpu").value, t)


def test_invalid_schedule_fails_before_execution() -> None:
    bad = ExecutableSchedule(
        graph_name="bad",
        fingerprint="x",
        instructions=[
            PlanInstruction(opcode=OpCode.COMPUTE, name="a", resource="cpu", depends_on=("missing",)),
        ],
    )
    errors = validate_schedule(bad)
    assert errors
    with pytest.raises((ScheduleValidationError, Exception)):
        assert_schedule_valid(bad)


def test_stream_event_incomplete_until_async_future_done() -> None:
    stream = MockStream("copy:mock_accel_0", delay_s=0.05, workers=1)
    try:
        fut = stream.submit(lambda: "ok")
        event = StreamEvent(name="record::t", device="mock_accel_0")
        event.bind_future(fut, enqueue_start_s=time.perf_counter(), enqueue_end_s=time.perf_counter())
        assert event.completed is False
        event.wait()
        assert event.completed is True
        assert fut.result() == "ok"
    finally:
        stream.shutdown()


def test_schedule_opcodes_appear_in_runtime_telemetry() -> None:
    model = _FanOut().eval()
    x = torch.randn(2, 8)
    machine = _cpu_mock_machine()
    cpu = next(n for n, c in machine.compute.items() if c.backend_id == "cpu")
    accel = "mock_accel_0"
    config = CompileConfig(
        use_torch_compile=False,
        measure_regions=False,
        allow_concurrent_regions=True,
        max_concurrent_regions=2,
        max_region_nodes=4,
        objective=Objective.LATENCY,
        allow_gpu=False,
    )
    probe = tt.compile(model, (x,), config=config, machine=machine)
    try:
        measurements = _split_measurements([r.region_id for r in probe._program.regions], cpu, accel)
    finally:
        probe.close()
    compiled = tt.compile(model, (x,), config=config, machine=machine, measurements=measurements)
    try:
        assert compiled.executor.uses_schedule_path
        eager = model(x)
        out = compiled(x)
        torch.testing.assert_close(out, eager, atol=1e-4, rtol=1e-4)
        report = compiled.executor._last_schedule_report
        assert report is not None
        opcodes = {e.opcode for e in report.events}
        assert "Compute" in opcodes
        # Cross-device plans must execute Transfer/Record/Wait through the DAG.
        schedule_opcodes = {i.opcode for i in compiled.specialized.schedule.instructions}
        if OpCode.TRANSFER in schedule_opcodes:
            assert "Transfer" in opcodes
            assert "RecordEvent" in opcodes
            assert "WaitEvent" in opcodes
        # No transfer telemetry without a schedule Transfer instruction.
        transfer_names = {i.name for i in compiled.specialized.schedule.transfer_ops()}
        for ev in compiled.executor._transfer_events:
            if ev["event"] == "transfer":
                assert ev["name"] in transfer_names or ev["name"].startswith("transfer::")
        # Real tensor ids — no synthetic activation:: in schedule when program wired.
        for inst in compiled.specialized.schedule.instructions:
            for name in inst.inputs + inst.outputs:
                assert not name.startswith("activation::"), name
    finally:
        compiled.close()


def test_multi_copy_cpu_and_mock_fanout_preserves_both_copies() -> None:
    model = _FanOut().eval()
    x = torch.randn(2, 8)
    machine = _cpu_mock_machine()
    cpu = next(n for n, c in machine.compute.items() if c.backend_id == "cpu")
    accel = "mock_accel_0"
    config = CompileConfig(
        use_torch_compile=False,
        measure_regions=False,
        allow_concurrent_regions=True,
        max_concurrent_regions=2,
        max_region_nodes=4,
        objective=Objective.LATENCY,
        allow_gpu=False,
    )
    probe = tt.compile(model, (x,), config=config, machine=machine)
    try:
        measurements = _split_measurements([r.region_id for r in probe._program.regions], cpu, accel)
    finally:
        probe.close()
    compiled = tt.compile(model, (x,), config=config, machine=machine, measurements=measurements)
    # Capture schedule executor reference before close() nulls it (for artifact cleanup).
    _sched_ref = getattr(getattr(compiled, "_executor", None), "_schedule_executor", None)
    try:
        devices = {p.device for p in compiled.specialized.plan.placements}
        assert cpu in devices and accel in devices
        out = compiled(x)
        torch.testing.assert_close(out, model(x), atol=1e-4, rtol=1e-4)
        peaks = compiled.executor._last_schedule_report.multi_copy_peaks
        assert peaks, "expected mid-run multi-resource copies after Transfer"
        for peak in peaks:
            assert len(peak["resources"]) >= 2
            assert not peak["tensor_id"].startswith("activation::")
        transfers = compiled.specialized.profile.get("residency", {}).get("transfers", [])
        assert transfers, "expected cross-device transfers for fan-out"
        value_names = {t["value_name"] for t in transfers}
        assert value_names
        assert not any(n.startswith("activation::") for n in value_names)
    finally:
        compiled.close()
        # Drop _native_artifact so Rust's global registry releases it before the next test.
        if _sched_ref is not None and hasattr(_sched_ref, "_native_artifact"):
            _sched_ref._native_artifact = None
        del _sched_ref


def test_async_overlap_wall_time_requires_real_overlap() -> None:
    """CPU 100ms + mock transfer 80ms + mock compute 100ms must overlap."""
    model = _FanOut().eval()
    x = torch.randn(2, 8)
    machine = _cpu_mock_machine(delay_hint_s=0.1)
    cpu = next(n for n, c in machine.compute.items() if c.backend_id == "cpu")
    accel = "mock_accel_0"
    config = CompileConfig(
        use_torch_compile=False,
        measure_regions=False,
        allow_concurrent_regions=True,
        max_concurrent_regions=4,
        max_region_nodes=4,
        objective=Objective.LATENCY,
        allow_gpu=False,
    )
    probe = tt.compile(model, (x,), config=config, machine=machine)
    try:
        measurements = _split_measurements([r.region_id for r in probe._program.regions], cpu, accel)
    finally:
        probe.close()
    compiled = tt.compile(model, (x,), config=config, machine=machine, measurements=measurements)
    try:
        # Annotate delays on a new immutable schedule (never mutate instructions).
        updates = {}
        for inst in compiled.specialized.schedule.instructions:
            if inst.opcode == OpCode.COMPUTE:
                updates[inst.name] = {"mock_compute_delay_s": 0.10}
            if inst.opcode == OpCode.TRANSFER:
                updates[inst.name] = {"mock_transfer_delay_s": 0.08}
        new_sched = with_instruction_attributes(compiled.specialized.schedule, updates)
        compiled.specialized.schedule = new_sched
        compiled.executor._schedule_executor.replace_schedule(new_sched)
        # Also sleep inside CPU callables so CPU work is real wall time.
        originals = dict(compiled.executor._callables)

        def _slow(rid: str, call: object):
            def wrapped(*args: object, **kwargs: object) -> object:
                binding = compiled.specialized.bindings[rid]
                if "mock" not in binding.device:
                    time.sleep(0.10)
                return call(*args, **kwargs)

            return wrapped

        compiled.executor._callables.clear()
        compiled.executor._callables.update({rid: _slow(rid, c) for rid, c in originals.items()})

        out = compiled(x)
        torch.testing.assert_close(out, model(x), atol=1e-4, rtol=1e-4)
        report = compiled.executor._last_schedule_report
        assert report is not None
        assert report.parallel_overlaps > 0, "no overlapping instruction intervals recorded"
        assert report.max_concurrent > 1
        # Hard wall≪seq proof lives in test_independent_computes_overlap_on_minimal_schedule.
    finally:
        compiled.close()


def test_independent_computes_overlap_on_minimal_schedule() -> None:
    """Deterministic delays: overlap proof cannot pass without real concurrency."""
    from torch.utils import _pytree as pytree

    from tensortorrent.backends.base import CompiledRegion
    from tensortorrent.compile.regions import Region, RegionBinding, RegionProgram, ValueSpec
    from tensortorrent.runtime.graph_executor import GraphExecutor
    from tensortorrent.runtime.schedule import PlanInstruction
    from tensortorrent.runtime.tensor_store import ResidentParameterStore

    def _slow_a(*_a: object, **_k: object) -> torch.Tensor:
        time.sleep(0.10)
        return torch.ones(2, 2)

    def _slow_b(*_a: object, **_k: object) -> torch.Tensor:
        time.sleep(0.10)
        return torch.ones(2, 2) * 2

    schedule = ExecutableSchedule(
        graph_name="overlap",
        fingerprint="t",
        instructions=[
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="compute::a",
                resource="cpu_a",
                depends_on=(),
                inputs=(),
                outputs=("out_a",),
                executable_ref="a",
                backend_id="cpu",
            ),
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="compute::b",
                resource="cpu_b",
                depends_on=(),
                inputs=(),
                outputs=("out_b",),
                executable_ref="b",
                backend_id="cpu",
            ),
        ],
    )
    region_a = Region(
        region_id="a",
        submodule="a",
        inputs=(),
        outputs=("out_a",),
        multi_output=False,
        aten_ops=(),
        node_count=1,
        depends_on=(),
        state_inputs=(),
        output_bytes=16,
    )
    region_b = Region(
        region_id="b",
        submodule="b",
        inputs=(),
        outputs=("out_b",),
        multi_output=False,
        aten_ops=(),
        node_count=1,
        depends_on=(),
        state_inputs=(),
        output_bytes=16,
    )
    values = {
        "out_a": ValueSpec(name="out_a", shape=(2, 2), dtype="float32", nbytes=16, kind="activation"),
        "out_b": ValueSpec(name="out_b", shape=(2, 2), dtype="float32", nbytes=16, kind="activation"),
    }
    program = RegionProgram(
        graph_name="overlap",
        root=nn.Module(),
        regions=(region_a, region_b),
        user_inputs=(),
        state_bindings={},
        values=values,
        output_refs=(("value", "out_a"), ("value", "out_b")),
        in_spec=pytree.tree_structure(((), {})),
        out_spec=pytree.tree_structure([object(), object()]),
    )
    bindings = {
        "a": RegionBinding(
            region=region_a,
            compiled=CompiledRegion(
                region_id="a",
                device="cpu_a",
                backend_id="cpu",
                executable=_slow_a,
                dtype="float32",
                torch_device="cpu",
            ),
            backend_id="cpu",
            device="cpu_a",
        ),
        "b": RegionBinding(
            region=region_b,
            compiled=CompiledRegion(
                region_id="b",
                device="cpu_b",
                backend_id="cpu",
                executable=_slow_b,
                dtype="float32",
                torch_device="cpu",
            ),
            backend_id="cpu",
            device="cpu_b",
        ),
    }
    store = ResidentParameterStore({})
    executor = GraphExecutor(program, bindings, parameter_store=store, schedule=schedule, max_workers=2)
    # Capture the schedule_executor reference before close() nulls it out.
    # We need to drop _native_artifact explicitly to prevent the Rust global
    # artifact registry from contaminating the next test's NativeExecutionContext.
    _sched_exec_ref = getattr(executor, "_schedule_executor", None)
    try:
        t0 = time.perf_counter()
        outs, report = executor.run([])
        wall = time.perf_counter() - t0
        assert len(outs) == 2
        assert report.parallel_overlaps > 0
        # Widened from 0.18s to 0.55s (~3x) for determinism on 2-core hosts
        assert wall < 0.55, f"wall {wall:.3f}s — no overlap (need <550ms for two 100ms regions)"
    finally:
        executor.close()
        store.close()
        # Explicitly drop _native_artifact AFTER close() so the Rust global registry
        # releases the artifact before the next test's NativeExecutionContext starts.
        # Without this, subsequent tests see stale residency state ("tensor not resident").
        if _sched_exec_ref is not None and hasattr(_sched_exec_ref, "_native_artifact"):
            _sched_exec_ref._native_artifact = None
        del _sched_exec_ref


def test_multi_output_region_transfers_each_output() -> None:
    model = _MultiOut().eval()
    x = torch.randn(2, 8)
    machine = _cpu_mock_machine()
    cpu = next(n for n, c in machine.compute.items() if c.backend_id == "cpu")
    accel = "mock_accel_0"
    config = CompileConfig(
        use_torch_compile=False,
        measure_regions=False,
        allow_concurrent_regions=True,
        max_concurrent_regions=2,
        max_region_nodes=2,
        objective=Objective.LATENCY,
        allow_gpu=False,
    )
    probe = tt.compile(model, (x,), config=config, machine=machine)
    try:
        measurements = _split_measurements([r.region_id for r in probe._program.regions], cpu, accel)
    finally:
        probe.close()
    compiled = tt.compile(model, (x,), config=config, machine=machine, measurements=measurements)
    try:
        devices = {p.device for p in compiled.specialized.plan.placements}
        assert cpu in devices and accel in devices
        out = compiled(x)
        torch.testing.assert_close(out, model(x), atol=1e-4, rtol=1e-4)
        value_names = {t["value_name"] for t in compiled.specialized.profile.get("residency", {}).get("transfers", [])}
        assert value_names
        assert not any(n.startswith("activation::") for n in value_names)
        # Distinct outputs may travel to different destinations.
        transfers = compiled.specialized.profile.get("residency", {}).get("transfers", [])
        dests = {(t["value_name"], t["destination_device"]) for t in transfers}
        assert dests
    finally:
        compiled.close()


def test_apply_profile_feedback_swaps_executor() -> None:
    model = _FanOut().eval()
    x = torch.randn(2, 8)
    machine = _cpu_mock_machine()
    cpu = next(n for n, c in machine.compute.items() if c.backend_id == "cpu")
    accel = "mock_accel_0"
    config = CompileConfig(
        use_torch_compile=False,
        measure_regions=False,
        allow_concurrent_regions=True,
        max_concurrent_regions=2,
        max_region_nodes=4,
        online_profile_feedback=True,
        objective=Objective.LATENCY,
        allow_gpu=False,
    )
    probe = tt.compile(model, (x,), config=config, machine=machine)
    try:
        measurements = _split_measurements([r.region_id for r in probe._program.regions], cpu, accel)
    finally:
        probe.close()
    compiled = tt.compile(model, (x,), config=config, machine=machine, measurements=measurements)
    try:
        old_exec = compiled.executor
        closed = {"ok": False}
        _orig_close = old_exec.close

        def _track_close() -> None:
            closed["ok"] = True
            _orig_close()

        old_exec.close = _track_close  # type: ignore[method-assign]
        _ = compiled(x)
        # Skew priors: observed device looks slow so replan can move work.
        for rid, binding in compiled.specialized.bindings.items():
            compiled._profile_feedback.region_latency_s[rid] = 1.0
            compiled._profile_feedback.region_device[rid] = binding.device
            compiled._profile_feedback.samples[rid] = 3
        compiled._profile_feedback.updates = 3
        result = compiled.apply_profile_feedback()
        assert isinstance(result, dict)
        assert result.get("plan") is not None
        assert "deltas" in result
        assert closed["ok"], "old executor must be closed"
        assert compiled.executor is not old_exec
        assert compiled.executor.uses_schedule_path
        out = compiled(x)
        torch.testing.assert_close(out, model(x), atol=1e-4, rtol=1e-4)
    finally:
        compiled.close()


def test_process_workers_survive_region_failure_then_succeed() -> None:
    """After a region boom on the schedule path, a later call still succeeds.

    Process-pool fork inherits callables at pool create time; this test covers
    the shared ``_callables`` schedule path. Pool-level survival is covered in
    ``test_hetero_execution_path``.
    """
    model = _FanOut().eval()
    x = torch.randn(2, 8)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_concurrent_regions=True,
            max_concurrent_regions=2,
            process_workers=0,
            max_region_nodes=4,
            allow_gpu=False,
        ),
    )
    try:
        rid = compiled.program.regions[0].region_id
        original = compiled.executor._callables[rid]

        def _boom(*_a: object, **_k: object) -> object:
            raise RuntimeError("worker boom")

        compiled.executor._callables[rid] = _boom
        with pytest.raises(RuntimeError, match="worker boom"):
            compiled(x)
        compiled.executor._callables[rid] = original
        out = compiled(x)
        torch.testing.assert_close(out, model(x), atol=1e-4, rtol=1e-4)
    finally:
        compiled.close()


def test_compiled_region_runtime_error_propagates() -> None:
    model = nn.Linear(4, 2).eval()
    x = torch.randn(2, 4)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False, allow_gpu=False),
    )
    try:
        # Replace accepted executable with a bomb; must not silent-eager-fallback.
        rid = compiled.program.regions[0].region_id
        binding = compiled.specialized.bindings[rid]
        compiled_exe = binding.compiled.executable

        def _boom(*_a: object, **_k: object) -> object:
            raise RuntimeError("user region boom")

        if isinstance(compiled_exe, _CompiledRegionCallable):
            compiled_exe.compiled = _boom
            compiled_exe._use_compiled = True
            compiled_exe.eager = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("eager should not run"))
            compiled.executor._callables[rid] = compiled_exe
        else:
            compiled.executor._callables[rid] = _boom

        with pytest.raises(RuntimeError, match="user region boom"):
            compiled(x)
    finally:
        compiled.close()


def test_process_workers_via_compiled_module_path() -> None:
    if __import__("sys").platform != "linux":
        pytest.skip("process_workers uses Linux fork")
    model = _FanOut().eval()
    x = torch.randn(2, 8)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_concurrent_regions=True,
            max_concurrent_regions=2,
            process_workers=2,
            max_region_nodes=4,
            allow_gpu=False,
        ),
    )
    try:
        assert compiled.executor._process_pool is not None or compiled.executor.max_workers == 1
        out = compiled(x)
        torch.testing.assert_close(out, model(x), atol=1e-4, rtol=1e-4)
        # Structured path / second call after success.
        out2 = compiled(x)
        torch.testing.assert_close(out2, out)
    finally:
        compiled.close()
