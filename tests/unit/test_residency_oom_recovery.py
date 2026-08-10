"""Residency OOM recovery must fail closed on rebuild errors."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from tensortorrent.errors import RuntimePlanError
from tensortorrent.runtime.native_bridge import residency as residency_mod


def _executor(*, hoist_targets: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        parameter_store=SimpleNamespace(needs_prefetch=False),
        _persistent_param_lock=threading.Lock(),
        _persistent_param_cache=[("w", "w", torch.zeros(2), 8, None, {})],
        _hoist_resident_parameters=True,
        _partial_hoist_oom=False,
        _resident_parameter_targets={"w": ("cuda_0",)} if hoist_targets else {},
        _persistent_device_param_cache={},
        _persistent_parameter_ids={"w"},
        program=SimpleNamespace(state_bindings={"w": "w"}),
        schedule=object(),
        bindings={},
    )


def test_non_oom_residency_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = _executor()

    def boom(*_a, **_k):
        raise RuntimePlanError("mapping failed")

    monkeypatch.setattr(residency_mod, "_move_tensor_to_resource", boom)
    ctx = SimpleNamespace(
        enable_grad=False,
        native_residency=None,
        host_resource="host",
        publish_tensor=lambda *a, **k: None,
    )

    with pytest.raises(RuntimePlanError, match="mapping failed"):
        residency_mod._register_persistent_residency(executor, ctx)
    assert executor._hoist_resident_parameters is True


def test_oom_rebuild_failure_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = _executor()

    def oom(*_a, **_k):
        raise torch.OutOfMemoryError("CUDA out of memory")

    def bad_rebuild(_schedule):
        raise RuntimeError("rebuild exploded")

    monkeypatch.setattr(residency_mod, "_move_tensor_to_resource", oom)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    executor._install_native_artifact = bad_rebuild
    executor._recompute_schedule_caches = lambda _s: None
    ctx = SimpleNamespace(
        enable_grad=False,
        native_residency=None,
        host_resource="host",
        publish_tensor=lambda *a, **k: None,
    )

    with pytest.raises(RuntimePlanError, match="failed to rebuild"):
        residency_mod._register_persistent_residency(executor, ctx)


def test_oom_marks_partial_hoist_failed_not_permanent_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = _executor()
    executor._hoist_resident_parameters = True
    executor._partial_hoist_oom = False

    def oom(*_a, **_k):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(residency_mod, "_move_tensor_to_resource", oom)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    executor._install_native_artifact = lambda _s: None
    executor._recompute_schedule_caches = lambda _s: None
    ctx = SimpleNamespace(
        enable_grad=False,
        native_residency=None,
        host_resource="host",
        publish_tensor=lambda *a, **k: None,
    )
    residency_mod._register_persistent_residency(executor, ctx)
    assert executor._hoist_resident_parameters is True
    assert executor._partial_hoist_oom is True
    assert executor._persistent_parameter_ids == set()
