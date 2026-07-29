"""NativeCompiledArtifact persistence and path-oracle tests."""

from __future__ import annotations

import threading

import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.native import require_native
from streamcompiler.runtime.schedule import ExecutableSchedule, MemoryTier, PlanInstruction
from streamcompiler.ir.graph import OpCode
from streamcompiler.testing import (
    assert_native_extension_loaded,
    assert_native_runtime_used,
    assert_no_hot_path_schedule_conversion,
    assert_no_python_fallback,
    assert_scheduler_entered,
    reset_native_counters,
    snapshot_native_counters,
)


def test_native_artifact_created_once_and_reused() -> None:
    assert_native_extension_loaded()
    reset_native_counters()
    model = nn.Linear(8, 8).eval()
    x = torch.randn(2, 8)
    compiled = sc.compile(model, example_inputs=(x,), devices="cpu")
    try:
        se = compiled.executor._schedule_executor
        assert se is not None
        artifact = se._native_artifact
        assert artifact is not None
        aid = int(artifact.artifact_id)
        before = snapshot_native_counters()
        with torch.inference_mode():
            for _ in range(5):
                out = compiled(x)
        after = snapshot_native_counters()
        torch.testing.assert_close(out, model(x))
        assert int(artifact.artifact_id) == aid
        assert int(artifact.execute_count) >= 5
        assert artifact.is_unmutated()
        assert_scheduler_entered(before, after, min_enters=5)
        assert_no_hot_path_schedule_conversion(before, after, max_conversions=0)
        assert_no_python_fallback(before, after)
        report = compiled.executor._last_schedule_report
        assert report is not None
        assert_native_runtime_used(report.parameter_store)
    finally:
        compiled.close()


def test_native_artifact_serialization_stable_across_runs() -> None:
    native = require_native()
    schedule = ExecutableSchedule(
        graph_name="g",
        fingerprint="fp",
        instructions=(
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="compute::a",
                resource="cpu",
                outputs=("y",),
                nbytes=8,
                memory_tier=MemoryTier.SYSTEM_RAM,
                executable_ref="a",
            ),
        ),
    )
    art = native.NativeCompiledArtifact.from_schedule(schedule)
    before = bytes(art.serialized_fingerprint())
    art.execute(dry_run=True)
    after = bytes(art.serialized_fingerprint())
    assert before == after
    assert art.is_unmutated()


def test_failed_forward_does_not_mutate_artifact() -> None:
    model = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    compiled = sc.compile(model, example_inputs=(x,), devices="cpu", config=sc.CompileConfig(use_torch_compile=False))
    try:
        se = compiled.executor._schedule_executor
        assert se is not None
        artifact = se._native_artifact
        assert artifact is not None
        before = bytes(artifact.serialized_fingerprint())
        rid = next(iter(se._callables))
        original = se._callables[rid]

        def boom(*_a: object, **_k: object) -> object:
            raise RuntimeError("intentional boom")

        se._callables[rid] = boom
        try:
            compiled(x)
            raise AssertionError("expected boom")
        except RuntimeError as exc:
            assert "intentional boom" in str(exc)
        assert bytes(artifact.serialized_fingerprint()) == before
        assert artifact.is_unmutated()
        se._callables[rid] = original
        out = compiled(x)
        torch.testing.assert_close(out, model(x))
    finally:
        compiled.close()


def test_concurrent_forwards_use_independent_contexts() -> None:
    model = nn.Linear(8, 8).eval()
    x = torch.randn(2, 8)
    cfg = sc.CompileConfig(use_torch_compile=False)
    # Specialize sequentially — torch.export is not thread-safe here.
    compileds = [sc.compile(model, example_inputs=(x,), devices="cpu", config=cfg) for _ in range(4)]
    try:
        ids = {int(c.executor._schedule_executor._native_artifact.artifact_id) for c in compileds}
        assert len(ids) == 4
        errors: list[BaseException] = []
        outputs: list[torch.Tensor] = []
        lock = threading.Lock()

        def worker(compiled: sc.CompiledModule) -> None:
            try:
                out = compiled(x)
                with lock:
                    outputs.append(out)
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(c,)) for c in compileds]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert len(outputs) == 4
        for out in outputs:
            torch.testing.assert_close(out, model(x))
        for c in compileds:
            art = c.executor._schedule_executor._native_artifact
            assert art.is_unmutated()
            assert int(art.execute_count) >= 1
    finally:
        for c in compileds:
            c.close()


def test_max_concurrency_interval_sweep() -> None:
    from streamcompiler.runtime.schedule_executor import max_concurrency_from_intervals

    assert max_concurrency_from_intervals([(0.0, 2.0), (1.0, 3.0), (3.0, 4.0)]) == 2
    assert max_concurrency_from_intervals([(0.0, 1.0), (1.0, 2.0)]) == 1
    assert max_concurrency_from_intervals([]) == 0
