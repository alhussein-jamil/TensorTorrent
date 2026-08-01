"""Focused regressions for the final production-hardening pass."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from streamcompiler.artifact_io import (
    atomic_replace_directory,
    atomic_write_text,
    verify_integrity_manifest,
    write_integrity_manifest,
)
from streamcompiler.backends import backend_id_for_resource
from streamcompiler.backends.base import (
    BenchmarkResult,
    CompiledRegion,
    ExecutionBackend,
    KernelCandidate,
    TransferCapability,
)
from streamcompiler.config import CompileConfig
from streamcompiler.errors import ExecutionCancelled, RuntimePlanError, StreamCompilerError
from streamcompiler.ir.resource_graph import ComputeResource, ResourceGraph
from streamcompiler.serve import InferenceService, ServiceConfig
from streamcompiler.serve.model_manager import ModelManager


def test_compile_config_rejects_invalid_production_values() -> None:
    with pytest.raises(ValueError, match="prefetch_distance"):
        CompileConfig(prefetch_distance=-1)
    with pytest.raises(ValueError, match="ram_budget_bytes"):
        CompileConfig(ram_budget_bytes=0)
    with pytest.raises(ValueError, match="objective_weights"):
        CompileConfig(objective_weights={"latency": 0.0, "memory": 0.0, "throughput": 0.0})
    with pytest.raises(ValueError, match="profile_level"):
        CompileConfig(profile_level="unknown")
    # allow_gpu=False is the CPU-only switch; integrated GPUs coerce off.
    cpu_only = CompileConfig(allow_gpu=False, allow_integrated_gpu=True)
    assert cpu_only.allow_gpu is False
    assert cpu_only.allow_integrated_gpu is False


def test_compile_config_normalizes_weights_and_cache_dir(tmp_path: Path) -> None:
    cfg = CompileConfig(objective_weights={"latency": 2}, cache_dir=tmp_path / "cache")
    assert cfg.objective_weights == {"latency": 2.0, "memory": 0.0, "throughput": 0.0}
    assert cfg.cache_dir == tmp_path / "cache"


def test_xpu_resource_routing_is_explicit() -> None:
    assert backend_id_for_resource("xpu_gpu_0") == "xpu"
    assert backend_id_for_resource("intel_gpu_2") == "xpu"


class _PluginBackend(ExecutionBackend):
    backend_id = "plugin_test"

    def available(self) -> bool:
        return True

    def discover_devices(self) -> ResourceGraph:
        return ResourceGraph(fingerprint="plugin")

    def supported_ops(self, device: ComputeResource) -> tuple[str, ...]:
        return ()

    def supported_dtypes(self, device: ComputeResource) -> tuple[str, ...]:
        return ()

    def enumerate_kernels(self, region: Any, device: ComputeResource) -> list[KernelCandidate]:
        return []

    def benchmark(self, candidate: KernelCandidate) -> BenchmarkResult:
        return BenchmarkResult(candidate, 0.0, 0, True)

    def compile(self, region: Any, candidate: KernelCandidate) -> CompiledRegion:
        return CompiledRegion("r", "cpu", self.backend_id, lambda *args: args, "float32")

    def execute(self, executable: CompiledRegion, inputs: Any) -> tuple[Any, ...]:
        return tuple(inputs)

    def transfer_capabilities(self, source: Any, destination: Any) -> TransferCapability:
        return TransferCapability(str(source), str(destination), "shared")


class _EntryPoint:
    name = "plugin-test"
    value = "tests:_PluginBackend"

    @staticmethod
    def load() -> Any:
        return _PluginBackend


def test_backend_plugin_discovery_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    import streamcompiler.backends.registry as registry

    monkeypatch.delenv("STREAMCOMPILER_DISABLE_BACKEND_PLUGINS", raising=False)
    monkeypatch.setattr(registry, "_entry_points", lambda: [_EntryPoint()])
    found = registry.plugin_backends(refresh=True)
    assert [backend.backend_id for backend in found] == ["plugin_test"]
    assert registry.plugin_errors() == {}


def test_artifact_integrity_detects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    payload = root / "portable.json"
    atomic_write_text(payload, json.dumps({"ok": True}))
    write_integrity_manifest(root, [payload])
    assert verify_integrity_manifest(root, required=True) is not None
    payload.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimePlanError, match="mismatch"):
        verify_integrity_manifest(root, required=True)


def test_atomic_directory_publish_preserves_previous_bundle_on_failure(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")

    def fail(stage: Path) -> None:
        (stage / "new.txt").write_text("new", encoding="utf-8")
        raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        atomic_replace_directory(destination, fail)
    assert (destination / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (destination / "new.txt").exists()


class _FakeCancelToken:
    def __init__(self) -> None:
        self.cancelled = threading.Event()

    def cancel(self) -> None:
        self.cancelled.set()


class _FakeNative:
    NativeCancelToken = _FakeCancelToken


class _SlowModule:
    def __init__(self) -> None:
        self.closed = False
        self.started = threading.Event()

    def _forward_with_cancel_token(self, token: _FakeCancelToken, value: int) -> int:
        self.started.set()
        while not token.cancelled.wait(0.005):
            pass
        raise ExecutionCancelled("cancelled")

    def request_cancel(self) -> None:
        raise AssertionError("request-scoped token should be used")

    def close(self) -> None:
        self.closed = True


class _FastModule:
    def __init__(self) -> None:
        self.closed = 0

    def _forward_with_cancel_token(self, token: _FakeCancelToken, value: int) -> int:
        del token
        return value + 1

    def close(self) -> None:
        self.closed += 1


def test_service_uses_request_scoped_timeout_token(monkeypatch: pytest.MonkeyPatch) -> None:
    import streamcompiler.native as native_module

    monkeypatch.setattr(native_module, "require_native", lambda: _FakeNative())
    service = InferenceService(
        config=ServiceConfig(
            max_queue_depth=4,
            default_timeout_s=0.05,
            worker_threads=2,
            cancellation_grace_s=0.2,
        )
    )
    service.start()
    module = _SlowModule()
    service.models.load("slow", module)  # type: ignore[arg-type]
    try:
        with pytest.raises(ExecutionCancelled, match="timed out"):
            service.infer("slow", 1)
        assert module.started.is_set()
        deadline = time.time() + 1.0
        while service.health()["active_requests"] and time.time() < deadline:
            time.sleep(0.01)
        assert service.health()["active_requests"] == 0
        assert service.models.get("slow").in_flight == 0
        assert "streamcompiler_timeouts_total 1" in service.metrics_prometheus()
    finally:
        service.stop()


def test_model_manager_releases_exact_replaced_generation() -> None:
    manager = ModelManager()
    old = _FastModule()
    manager.load("m", old)  # type: ignore[arg-type]
    old_slot = manager.acquire("m")

    replacement = _FastModule()
    # Non-blocking publish: replace returns while old generation is still leased.
    manager.load("m", replacement)  # type: ignore[arg-type]
    assert old_slot.retired is True
    assert old.closed == 0
    assert replacement.closed == 0
    manager.release_slot(old_slot)
    assert old_slot.in_flight == 0
    assert old.closed == 1
    assert replacement.closed == 0
    assert manager.get("m").in_flight == 0
    manager.shutdown()
    assert replacement.closed == 1


def test_model_manager_warm_only_marks_current_generation() -> None:
    manager = ModelManager()

    class _SwapOnCall:
        def __init__(self) -> None:
            self.closed = 0

        def __call__(self, *args: object, **kwargs: object) -> int:
            # Replace the generation while a warm of *this* slot is in progress.
            manager.load("m", _FastModule())  # type: ignore[arg-type]
            return 0

        def close(self) -> None:
            self.closed += 1

    manager.load("m", _SwapOnCall())  # type: ignore[arg-type]
    manager.warm("m", 1)
    # Warm finished for the retired generation — current must stay unwarmed.
    assert manager.get("m").warm is False
    manager.shutdown()


def test_model_manager_backpressure_survives_replace() -> None:
    manager = ModelManager()
    first = _FastModule()
    manager.load("m", first, concurrency_limit=1)  # type: ignore[arg-type]
    leased = manager.acquire("m")
    with pytest.raises(StreamCompilerError, match="backpressure"):
        manager.acquire("m")
    second = _FastModule()
    manager.load("m", second, concurrency_limit=1)  # type: ignore[arg-type]
    manager.release_slot(leased)
    assert first.closed == 1
    assert second.closed == 0
    nxt = manager.acquire("m")
    assert nxt.module is second
    manager.release_slot(nxt)
    manager.shutdown()


def test_model_manager_double_release_is_safe() -> None:
    manager = ModelManager()
    first = _FastModule()
    manager.load("m", first)  # type: ignore[arg-type]
    leased = manager.acquire("m")
    manager.load("m", _FastModule())  # type: ignore[arg-type]
    manager.release_slot(leased)
    manager.release_slot(leased)
    assert leased.in_flight == 0
    assert first.closed == 1
    manager.shutdown()


def test_model_manager_concurrent_replace_and_acquire() -> None:
    manager = ModelManager()
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(40):
                manager.load("m", _FastModule())  # type: ignore[arg-type]
                try:
                    slot = manager.acquire("m")
                except StreamCompilerError as exc:
                    if "backpressure" not in str(exc) and "not loaded" not in str(exc):
                        raise
                    continue
                manager.release_slot(slot)
        except BaseException as exc:  # noqa: BLE001 - collect for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not errors
    manager.shutdown()


def test_xpu_discovery_is_treated_as_gpu_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    import streamcompiler.backends.xpu as xpu_module
    from streamcompiler.ir.resource_graph import ComputeClass, MemoryClass

    class Properties:
        name = "Intel Test GPU"
        total_memory = 8 << 30
        architecture = "test-xe"
        max_compute_units = 64
        copy_engines = 2

    class FakeXpu:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_properties(index: int) -> Properties:
            assert index == 0
            return Properties()

    monkeypatch.setattr(xpu_module, "_xpu_module", lambda: FakeXpu())
    monkeypatch.setattr(xpu_module.XpuBackend, "_probe_dtypes", lambda self, index: ("float32", "bfloat16"))
    graph = xpu_module.XpuBackend().discover_devices()
    device = graph.compute["xpu_gpu_0"]
    memory = graph.memory["xpu_vram_0"]
    assert device.compute_class == ComputeClass.DISCRETE_GPU
    assert device.backend_id == "xpu"
    assert memory.memory_class == MemoryClass.DEVICE_VRAM
    assert memory.allocatable_bytes == int((8 << 30) * 0.9)


def test_service_config_rejects_unusable_zero_queue() -> None:
    with pytest.raises(ValueError, match="max_queue_depth"):
        ServiceConfig(max_queue_depth=0)


def test_service_rejects_duplicate_active_request_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    import streamcompiler.native as native_module

    monkeypatch.setattr(native_module, "require_native", lambda: _FakeNative())
    service = InferenceService(
        config=ServiceConfig(max_queue_depth=4, default_timeout_s=1.0, worker_threads=2, cancellation_grace_s=0.2)
    )
    service.start()
    module = _SlowModule()
    service.models.load("slow", module)  # type: ignore[arg-type]
    first_error: list[BaseException] = []

    def first_request() -> None:
        try:
            service.infer("slow", 1, request_id="same", timeout_s=0.5)
        except BaseException as exc:  # noqa: BLE001 - captured for assertion
            first_error.append(exc)

    thread = threading.Thread(target=first_request)
    thread.start()
    assert module.started.wait(1.0)
    try:
        with pytest.raises(StreamCompilerError, match="duplicate active request_id"):
            service.infer("slow", 2, request_id="same", timeout_s=0.1)
        assert service.cancel("same") is True
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert first_error and isinstance(first_error[0], ExecutionCancelled)
    finally:
        service.stop()


def test_artifact_integrity_rejects_unexpected_files(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    payload = root / "portable.json"
    atomic_write_text(payload, "{}")
    write_integrity_manifest(root, [payload])
    (root / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(RuntimePlanError, match="unmanifested"):
        verify_integrity_manifest(root, required=True)


def test_artifact_integrity_rejects_symlink_entries(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"payload")
    link = root / "alias.bin"
    try:
        link.symlink_to(payload.name)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(RuntimePlanError, match="symlink"):
        write_integrity_manifest(root, [payload, link])


def test_cuda_and_rocm_backends_do_not_claim_same_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from streamcompiler.backends.cuda import CudaBackend
    from streamcompiler.backends.rocm import RocmBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    monkeypatch.setattr(torch.version, "cuda", None, raising=False)
    assert RocmBackend().available() is True
    assert CudaBackend().available() is False

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.version, "cuda", "12.4", raising=False)
    assert CudaBackend().available() is True
    assert RocmBackend().available() is False


def test_device_selection_does_not_mutate_caller_config() -> None:
    from streamcompiler.frontend.export import _apply_device_selection

    original = CompileConfig(allow_cpu=True, allow_gpu=True, allow_integrated_gpu=True)
    cpu_only = _apply_device_selection(original, "cpu")
    assert cpu_only.allow_cpu is True
    assert cpu_only.allow_gpu is False
    assert cpu_only.allow_integrated_gpu is False
    assert original.allow_cpu is True
    assert original.allow_gpu is True
    assert original.allow_integrated_gpu is True

    gpu_only = _apply_device_selection(original, "gpu")
    assert gpu_only.allow_cpu is False
    assert gpu_only.allow_gpu is True
    assert original.allow_cpu is True


def test_compile_config_rejects_wrong_scalar_types() -> None:
    with pytest.raises(TypeError, match="max_region_nodes"):
        CompileConfig(max_region_nodes=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="allow_gpu"):
        CompileConfig(allow_gpu="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ram_budget_bytes"):
        CompileConfig(ram_budget_bytes=1.5)  # type: ignore[arg-type]


def test_compile_config_accepts_string_objective_and_serializes() -> None:
    cfg = CompileConfig(objective="memory")  # type: ignore[arg-type]
    assert cfg.objective.value == "memory"
    restored = CompileConfig.from_json_dict(cfg.to_json_dict())
    assert restored.objective.value == "memory"


def test_compile_config_json_rejects_ambiguous_scalar_types() -> None:
    with pytest.raises(TypeError, match="max_region_nodes"):
        CompileConfig.from_json_dict({"max_region_nodes": "8"})
    with pytest.raises(TypeError, match="allow_gpu"):
        CompileConfig.from_json_dict({"allow_gpu": 1})
    with pytest.raises(TypeError, match="ram_budget_bytes"):
        CompileConfig.from_json_dict({"ram_budget_bytes": 1.5})
    with pytest.raises(TypeError, match="atol"):
        CompileConfig.from_json_dict({"atol": "1e-5"})


def test_executor_generation_manager_defers_retired_close_until_final_lease() -> None:
    from streamcompiler.runtime.module import _ExecutorGenerationManager

    closed: list[str] = []
    first = object()
    second = object()
    manager = _ExecutorGenerationManager(first, lambda value: closed.append("first" if value is first else "second"))

    lease = manager.acquire()
    assert lease is first
    manager.swap(second)
    assert closed == []
    assert manager.acquire() is second
    manager.release(second)
    manager.release(first)
    assert closed == ["first"]
    manager.close()
    assert closed == ["first", "second"]


def test_service_rejects_ambiguous_or_nonfinite_limits() -> None:
    with pytest.raises(TypeError, match="max_queue_depth"):
        ServiceConfig(max_queue_depth=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="worker_threads"):
        ServiceConfig(worker_threads=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="default_timeout_s"):
        ServiceConfig(default_timeout_s=float("nan"))
    with pytest.raises(ValueError, match="cancellation_grace_s"):
        ServiceConfig(cancellation_grace_s=float("inf"))


def test_atomic_directory_publish_serializes_concurrent_writers(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    first_started = threading.Event()
    allow_first = threading.Event()
    errors: list[BaseException] = []

    def first_writer(stage: Path) -> None:
        first_started.set()
        assert allow_first.wait(2.0)
        (stage / "generation.txt").write_text("first", encoding="utf-8")

    def second_writer(stage: Path) -> None:
        (stage / "generation.txt").write_text("second", encoding="utf-8")

    def publish(writer: Any) -> None:
        try:
            atomic_replace_directory(destination, writer)
        except BaseException as exc:  # noqa: BLE001 - captured for assertion
            errors.append(exc)

    first = threading.Thread(target=publish, args=(first_writer,))
    second = threading.Thread(target=publish, args=(second_writer,))
    first.start()
    assert first_started.wait(1.0)
    second.start()
    time.sleep(0.03)
    allow_first.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert (destination / "generation.txt").read_text(encoding="utf-8") == "second"


def test_profiler_factory_supports_all_builtin_accelerator_families() -> None:
    from streamcompiler.backends.profiler import CudaBackendProfiler, XpuBackendProfiler, profiler_for_backend

    cuda = profiler_for_backend("cuda", device_index=2)
    rocm = profiler_for_backend("rocm", device_index=3)
    xpu = profiler_for_backend("xpu", device_index=4)
    assert isinstance(cuda, CudaBackendProfiler) and cuda.backend_id == "cuda" and cuda.device_index == 2
    assert isinstance(rocm, CudaBackendProfiler) and rocm.backend_id == "rocm" and rocm.device_index == 3
    assert isinstance(xpu, XpuBackendProfiler) and xpu.device_index == 4


def test_cpu_profiler_profiles_regions_and_bounds_large_transfer_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import streamcompiler.backends.profiler as profiler_module

    profiler = profiler_module.CpuBackendProfiler()
    region = profiler.profile_region(
        lambda value: value + 1,
        (1,),
        device_fingerprint="cpu-test",
        region_graph_hash="region-test",
        warm_up=0,
        samples=1,
    )
    assert region.measured is True and region.sample_count == 1 and region.median_s >= 0

    monkeypatch.setattr(profiler_module, "_MAX_TRANSFER_PROFILE_BYTES", 32)
    transfer = profiler.profile_transfer(
        1024,
        source="host_a",
        destination="host_b",
        device_fingerprint="cpu-test",
        warm_up=0,
        samples=1,
    )
    assert "measured_bytes=32" in transfer.notes
    assert "requested_bytes=1024" in transfer.notes


def test_artifact_integrity_rejects_unmanifested_symlink_directories(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    payload = root / "portable.json"
    atomic_write_text(payload, "{}")
    write_integrity_manifest(root, [payload])
    try:
        (root / "linked-dir").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(RuntimePlanError, match="symlink"):
        verify_integrity_manifest(root, required=True)
