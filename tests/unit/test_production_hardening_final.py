"""Focused regressions for the final production-hardening pass."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from tensortorrent.artifact_io import (
    atomic_replace_directory,
    atomic_write_text,
    verify_integrity_manifest,
    write_integrity_manifest,
)
from tensortorrent.backends import backend_id_for_resource
from tensortorrent.backends.base import (
    BenchmarkResult,
    CompiledRegion,
    ExecutionBackend,
    KernelCandidate,
    TransferCapability,
)
from tensortorrent.config import CompileConfig
from tensortorrent.errors import ExecutionCancelled, RuntimePlanError, TensorTorrentError
from tensortorrent.ir.resource_graph import ComputeResource, ResourceGraph
from tensortorrent.serve import InferenceService, ServiceConfig
from tensortorrent.serve.model_manager import ModelManager


@pytest.mark.parametrize(
    "kwargs,message",
    (
        ({"nbytes": -1}, "nbytes must be >= 0"),
        ({"nbytes": True}, "nbytes must be an integer"),
        ({"predicted_duration_s": float("nan")}, "must be finite"),
        ({"sync_required": 1}, "sync_required must be a bool"),
        ({"inputs": (1,)}, "inputs must contain"),
    ),
)
def test_plan_instruction_rejects_malformed_runtime_metadata(kwargs: dict[str, object], message: str) -> None:
    from tensortorrent.ir.graph import OpCode
    from tensortorrent.runtime.schedule import PlanInstruction

    with pytest.raises((TypeError, ValueError), match=message):
        PlanInstruction(opcode=OpCode.COMPUTE, name="compute", resource="cpu", **kwargs)  # type: ignore[arg-type]


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


def test_tensor_move_rejects_unknown_resource_instead_of_relabeling() -> None:
    from tensortorrent.runtime.native_bridge import _move_tensor_to_resource

    with pytest.raises(RuntimePlanError, match="unknown non-host resource"):
        _move_tensor_to_resource(torch.ones(2), "mystery_accelerator_0")
    host = torch.ones(2)
    assert _move_tensor_to_resource(host, "pinned_host_0") is host


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
    import tensortorrent.backends.registry as registry

    monkeypatch.delenv("TENSORTORRENT_DISABLE_BACKEND_PLUGINS", raising=False)
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


def test_atomic_directory_publish_rejects_symlink_destination(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    destination = tmp_path / "artifact"
    destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimePlanError, match="destination cannot be a symlink"):
        atomic_replace_directory(destination, lambda stage: None)


class _UnlimitedLedger:
    inflight = 0

    def max_concurrent(self) -> int:
        return 1 << 30


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
        self.capacity_ledger = _UnlimitedLedger()

    def forward_with_cancel_token(self, token: _FakeCancelToken, value: int) -> int:
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
        self.capacity_ledger = _UnlimitedLedger()

    def forward_with_cancel_token(self, token: _FakeCancelToken, value: int) -> int:
        del token
        return value + 1

    def close(self) -> None:
        self.closed += 1


def test_service_uses_request_scoped_timeout_token(monkeypatch: pytest.MonkeyPatch) -> None:
    import tensortorrent.native as native_module

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
        # Widened from 1s to 5s for determinism on 2-core hosts (slow cancellation grace period)
        deadline = time.time() + 5.0
        while service.health()["active_requests"] and time.time() < deadline:
            time.sleep(0.01)
        assert service.health()["active_requests"] == 0
        assert service.models.get("slow").in_flight == 0
        assert "tensortorrent_timeouts_total 1" in service.metrics_prometheus()
    finally:
        service.stop()


def test_service_caps_caller_timeout_at_configured_maximum(monkeypatch: pytest.MonkeyPatch) -> None:
    import tensortorrent.native as native_module

    monkeypatch.setattr(native_module, "require_native", lambda: _FakeNative())
    service = InferenceService(
        config=ServiceConfig(
            default_timeout_s=0.02,
            max_request_timeout_s=0.02,
            worker_threads=1,
            cancellation_grace_s=0.2,
        )
    )
    service.start()
    service.models.load("slow", _SlowModule())  # type: ignore[arg-type]
    try:
        started = time.perf_counter()
        with pytest.raises(ExecutionCancelled, match="timed out after 0.02s"):
            service.infer("slow", 1, timeout_s=60.0)
        assert time.perf_counter() - started < 1.0
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
            self.capacity_ledger = _UnlimitedLedger()

        def __call__(self, *args: object, **kwargs: object) -> int:
            # Replace the generation while a warm of *this* slot is in progress.
            manager.load("m", _FastModule())  # type: ignore[arg-type]
            assert self.closed == 0
            return 0

        def close(self) -> None:
            self.closed += 1

    manager.load("m", _SwapOnCall())  # type: ignore[arg-type]
    manager.warm("m", 1)
    # Warm finished for the retired generation — current must stay unwarmed.
    assert manager.get("m").warm is False
    manager.shutdown()


@pytest.mark.parametrize("limit", (True, 0, 1.5))
def test_model_manager_rejects_invalid_concurrency_limit(limit: object) -> None:
    manager = ModelManager()
    with pytest.raises(ValueError, match="concurrency_limit"):
        manager.load("m", _FastModule(), concurrency_limit=limit)  # type: ignore[arg-type]


def test_model_manager_failed_warm_leaves_slot_unwarmed() -> None:
    manager = ModelManager()

    class _Boom:
        capacity_ledger = _UnlimitedLedger()

        def __call__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("warm failed")

        def close(self) -> None:
            return None

    manager.load("m", _Boom())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="warm failed"):
        manager.warm("m", 1)
    assert manager.get("m").warm is False
    manager.shutdown()


def test_model_manager_backpressure_survives_replace() -> None:
    manager = ModelManager()
    first = _FastModule()
    manager.load("m", first, concurrency_limit=1)  # type: ignore[arg-type]
    leased = manager.acquire("m")
    with pytest.raises(TensorTorrentError, match="backpressure"):
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


def test_model_manager_unload_timeout_never_closes_in_flight_generation() -> None:
    manager = ModelManager()
    module = _FastModule()
    manager.load("m", module)  # type: ignore[arg-type]
    leased = manager.acquire("m")

    manager.unload("m", drain_timeout_s=0.01)

    assert leased.retired is True
    assert leased.in_flight == 1
    assert module.closed == 0
    assert leased.version in manager._retired
    manager.release_slot(leased)
    assert module.closed == 1
    assert leased.version not in manager._retired


def test_model_manager_concurrent_replace_and_acquire() -> None:
    manager = ModelManager()
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(40):
                manager.load("m", _FastModule())  # type: ignore[arg-type]
                try:
                    slot = manager.acquire("m")
                except TensorTorrentError as exc:
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
    import tensortorrent.backends.xpu as xpu_module
    from tensortorrent.ir.resource_graph import ComputeClass, MemoryClass

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
    # NEW math: total*0.9 - headroom (display_active=True for XPU → headroom=768MiB)
    # total=8GiB, base=int(8GiB*0.9)=7730941132, headroom=768*1<<20=805306368
    # allowed = max(0, 7730941132 - 805306368) = 6925634764
    assert memory.allocatable_bytes == 6925634764


def test_service_config_rejects_unusable_zero_queue() -> None:
    with pytest.raises(ValueError, match="max_queue_depth"):
        ServiceConfig(max_queue_depth=0)


def test_service_rejects_duplicate_active_request_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    import tensortorrent.native as native_module

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
        with pytest.raises(TensorTorrentError, match="duplicate active request_id"):
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

    from tensortorrent.backends.cuda import CudaBackend
    from tensortorrent.backends.rocm import RocmBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    monkeypatch.setattr(torch.version, "cuda", None, raising=False)
    assert RocmBackend().available() is True
    assert CudaBackend().available() is False

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.version, "cuda", "12.4", raising=False)
    assert CudaBackend().available() is True
    assert RocmBackend().available() is False


def test_resource_classification_includes_xpu_without_host_substring_false_positives() -> None:
    from tensortorrent.runtime.resource_names import is_device_resource, is_host_resource
    from tensortorrent.runtime.schedule import MemoryTier, _tier_for_device
    from tensortorrent.runtime.schedule_executor import _tier_is_device

    assert is_device_resource("xpu_gpu_0")
    assert _tier_is_device("xpu_vram_0")
    assert _tier_for_device("xpu_gpu_0") == MemoryTier.DEVICE
    assert is_host_resource("cpu_numa_0")
    assert not is_host_resource("ghost_gpu_0")


def test_plugin_prefixed_resource_is_treated_as_device(monkeypatch: pytest.MonkeyPatch) -> None:
    import tensortorrent.backends as backends
    from tensortorrent.runtime.schedule import MemoryTier, _tier_for_device
    from tensortorrent.runtime.schedule_executor import _tier_is_device

    original = backends.backend_id_for_resource
    monkeypatch.setattr(
        backends,
        "backend_id_for_resource",
        lambda resource: "custom" if resource == "custom_accel_0" else original(resource),
    )
    assert _tier_for_device("custom_accel_0") == MemoryTier.DEVICE
    assert _tier_is_device("custom_accel_0")


def test_device_selection_does_not_mutate_caller_config() -> None:
    from tensortorrent.frontend.export import _apply_device_selection

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
    from tensortorrent.runtime.module import _ExecutorGenerationManager

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


def test_executor_generation_manager_survives_concurrent_swap() -> None:
    from tensortorrent.runtime.module import _ExecutorGenerationManager

    closed: list[object] = []
    manager = _ExecutorGenerationManager(object(), closed.append)
    errors: list[BaseException] = []

    def worker(worker_id: int) -> None:
        try:
            for step in range(80):
                if step % 8 == 0:
                    manager.swap(object())
                executor = manager.acquire()
                manager.release(executor)
        except BaseException as exc:  # noqa: BLE001 - collect for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not errors
    manager.close()
    assert closed  # at least the final generation closed


def test_service_rejects_ambiguous_or_nonfinite_limits() -> None:
    with pytest.raises(TypeError, match="max_queue_depth"):
        ServiceConfig(max_queue_depth=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="worker_threads"):
        ServiceConfig(worker_threads=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="default_timeout_s"):
        ServiceConfig(default_timeout_s=float("nan"))
    with pytest.raises(ValueError, match="cancellation_grace_s"):
        ServiceConfig(cancellation_grace_s=float("inf"))
    with pytest.raises(ValueError, match="default_timeout_s must be <="):
        ServiceConfig(default_timeout_s=2.0, max_request_timeout_s=1.0)
    with pytest.raises(ValueError, match="request_history_size"):
        ServiceConfig(request_history_size=0)


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
    from tensortorrent.backends.profiler import CudaBackendProfiler, XpuBackendProfiler, profiler_for_backend

    cuda = profiler_for_backend("cuda", device_index=2)
    rocm = profiler_for_backend("rocm", device_index=3)
    xpu = profiler_for_backend("xpu", device_index=4)
    assert isinstance(cuda, CudaBackendProfiler) and cuda.backend_id == "cuda" and cuda.device_index == 2
    assert isinstance(rocm, CudaBackendProfiler) and rocm.backend_id == "rocm" and rocm.device_index == 3
    assert isinstance(xpu, XpuBackendProfiler) and xpu.device_index == 4


def test_cpu_profiler_profiles_regions_and_bounds_large_transfer_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensortorrent.backends.profiler as profiler_module

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


def test_artifact_integrity_rejects_unmanifested_empty_directories(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"payload")
    write_integrity_manifest(root, [payload])
    (root / "unexpected").mkdir()

    with pytest.raises(RuntimePlanError, match="unmanifested directory"):
        verify_integrity_manifest(root, required=True)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unsupported")
def test_artifact_integrity_rejects_special_files_without_opening_them(tmp_path: Path) -> None:
    from tensortorrent.artifact_io import INTEGRITY_MANIFEST

    root = tmp_path / "artifact"
    root.mkdir()
    os.mkfifo(root / INTEGRITY_MANIFEST)
    with pytest.raises(RuntimePlanError, match="not a regular file"):
        verify_integrity_manifest(root, required=True)

    manifest_root = tmp_path / "artifact-with-fifo"
    manifest_root.mkdir()
    payload = manifest_root / "payload"
    payload.write_bytes(b"ok")
    write_integrity_manifest(manifest_root, [payload])
    os.mkfifo(manifest_root / "unexpected-fifo")
    with pytest.raises(RuntimePlanError, match="non-regular entry"):
        verify_integrity_manifest(manifest_root, required=True)


def test_artifact_integrity_rejects_oversized_manifest_before_json_parse(tmp_path: Path) -> None:
    from tensortorrent.artifact_io import INTEGRITY_MANIFEST, MAX_INTEGRITY_MANIFEST_BYTES

    root = tmp_path / "artifact"
    root.mkdir()
    manifest = root / INTEGRITY_MANIFEST
    with manifest.open("wb") as handle:
        handle.truncate(MAX_INTEGRITY_MANIFEST_BYTES + 1)
    with pytest.raises(RuntimePlanError, match="manifest too large"):
        verify_integrity_manifest(root, required=True)


def test_artifact_integrity_rejects_non_object_manifest(tmp_path: Path) -> None:
    from tensortorrent.artifact_io import INTEGRITY_MANIFEST

    root = tmp_path / "artifact"
    root.mkdir()
    (root / INTEGRITY_MANIFEST).write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimePlanError, match="JSON object"):
        verify_integrity_manifest(root, required=True)


def test_artifact_integrity_rejects_self_reference(tmp_path: Path) -> None:
    from tensortorrent.artifact_io import INTEGRITY_MANIFEST, INTEGRITY_SCHEMA, atomic_write_json

    root = tmp_path / "artifact"
    root.mkdir()
    atomic_write_json(
        root / INTEGRITY_MANIFEST,
        {
            "schema": INTEGRITY_SCHEMA,
            "files": {INTEGRITY_MANIFEST: {"size": 0, "sha256": "0" * 64}},
        },
    )
    with pytest.raises(RuntimePlanError, match="Unsafe path"):
        verify_integrity_manifest(root, required=True)


def test_artifact_integrity_rejects_non_utf8_manifest_paths(tmp_path: Path) -> None:
    from tensortorrent.artifact_io import INTEGRITY_MANIFEST, INTEGRITY_SCHEMA, atomic_write_json

    root = tmp_path / "artifact"
    root.mkdir()
    atomic_write_json(
        root / INTEGRITY_MANIFEST,
        {"schema": INTEGRITY_SCHEMA, "files": {"\ud800": {"size": 0, "sha256": "0" * 64}}},
    )
    with pytest.raises(RuntimePlanError, match="Unsafe path"):
        verify_integrity_manifest(root, required=True)


@pytest.mark.parametrize(
    "metadata, message",
    (
        ({"size": True, "sha256": "0" * 64}, "size metadata"),
        ({"size": 2, "sha256": "not-a-sha256"}, "checksum metadata"),
    ),
)
def test_artifact_integrity_rejects_malformed_metadata(tmp_path: Path, metadata: dict, message: str) -> None:
    from tensortorrent.artifact_io import INTEGRITY_MANIFEST, INTEGRITY_SCHEMA, atomic_write_json

    root = tmp_path / "artifact"
    root.mkdir()
    (root / "payload").write_bytes(b"ok")
    atomic_write_json(
        root / INTEGRITY_MANIFEST,
        {"schema": INTEGRITY_SCHEMA, "files": {"payload": metadata}},
    )
    with pytest.raises(RuntimePlanError, match=message):
        verify_integrity_manifest(root, required=True)
