"""Sticky cancel and serve-timeout isolation regressions."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.config import CompileConfig
from tensortorrent.errors import ExecutionCancelled
from tensortorrent.serve.app import InferenceService
from tensortorrent.serve.service_config import ServiceConfig


def test_sibling_completion_preserves_pending_cancel() -> None:
    """Completing forward must not clear a newer sticky cancel generation."""
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    x = torch.randn(2, 8)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=False,
            prefer_direct_path=True,
        ),
    )
    try:
        executor = compiled.executor
        assert getattr(executor, "direct_plan", None) is not None
        _ = compiled(x)
        start_gen = executor._cancel_generation
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def finishing_clear() -> None:
            try:
                with executor._cancel_lock:
                    gen = executor._cancel_generation
                barrier.wait(timeout=2)
                time.sleep(0.05)
                executor._clear_cancel_if_unchanged(gen)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def cancel_sibling() -> None:
            barrier.wait(timeout=2)
            compiled.request_cancel()

        t1 = threading.Thread(target=finishing_clear)
        t2 = threading.Thread(target=cancel_sibling)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not errors
        assert executor._cancel_generation > start_gen
        assert executor._cancel_requested is True
        with pytest.raises(ExecutionCancelled):
            compiled(x)
    finally:
        compiled.close()


def test_timeout_without_native_token_does_not_module_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve timeout without NativeCancelToken must not call module.request_cancel."""
    import tensortorrent.native as native_mod

    cancelled = {"module": 0}

    class _Module:
        def __call__(self, *_args, **_kwargs):  # noqa: ANN002
            time.sleep(0.25)
            return 1

        def request_cancel(self) -> None:
            cancelled["module"] += 1

        def close(self) -> None:
            return None

        capacity_ledger = SimpleNamespace(max_concurrent=lambda: 4)

    monkeypatch.setattr(
        native_mod,
        "require_native",
        lambda: (_ for _ in ()).throw(RuntimeError("native unavailable")),
    )
    service = InferenceService(
        config=ServiceConfig(
            max_queue_depth=4,
            default_timeout_s=0.05,
            worker_threads=2,
            cancellation_grace_s=0.05,
        )
    )
    service.start()
    try:
        service.models.load("m", _Module(), concurrency_limit=2)  # type: ignore[arg-type]
        with pytest.raises(ExecutionCancelled, match="timed out"):
            service.infer("m", 1, timeout_s=0.05)
        assert cancelled["module"] == 0
    finally:
        service.stop()
