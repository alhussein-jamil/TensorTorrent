"""Regression: cancel must not silently restart; concurrent cancel isolation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.config import CompileConfig
from streamcompiler.errors import ExecutionCancelled, RuntimePlanError
from streamcompiler.native import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native required")


class _CancelArtifact:
    artifact_id = 99

    def __init__(self, counter: dict[str, int]) -> None:
        self._counter = counter

    def execute(self, **_kwargs):
        self._counter["n"] += 1
        raise RuntimeError("Schedule execution cancelled by token")

    def is_unmutated(self) -> bool:
        return True


def test_cancel_exception_does_not_fallback_and_rerun() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    x = torch.randn(2, 8)
    compiled = sc.compile(model, (x,), config=CompileConfig(use_torch_compile=False, measure_regions=False))
    try:
        compiled(x)
        se = compiled.executor._schedule_executor
        calls = {"n": 0}
        real = se._native_artifact
        se._native_artifact = _CancelArtifact(calls)
        with pytest.raises(ExecutionCancelled):
            compiled(x)
        assert calls["n"] == 1
        se._native_artifact = real
    finally:
        compiled.close()


def test_compute_cancel_raises_once() -> None:
    model = nn.Sequential(nn.Linear(8, 4)).eval()
    x = torch.randn(2, 8)
    compiled = sc.compile(model, (x,), config=CompileConfig(use_torch_compile=False, measure_regions=False))
    try:
        compiled(x)
        se = compiled.executor._schedule_executor
        calls = {"n": 0}

        def wrapped(*args, **kwargs):
            calls["n"] += 1
            raise ExecutionCancelled("Schedule execution cancelled")

        se._exec_compute = wrapped  # type: ignore[method-assign]
        with pytest.raises(ExecutionCancelled):
            compiled(x)
        assert calls["n"] == 1
    finally:
        compiled.close()


def test_virtual_backend_drop_joins_workers() -> None:
    from streamcompiler.native import require_native

    native = require_native()
    be = native.NativeVirtualBackend(compute_delay_s=0.01)
    ev = be.launch("compute0")
    assert be.query_event(ev) == "pending"
    be.wait_event(ev)
    del be


def test_closed_module_rejects_forward() -> None:
    model = nn.Linear(4, 2).eval()
    x = torch.randn(1, 4)
    compiled = sc.compile(model, (x,), config=CompileConfig(use_torch_compile=False, measure_regions=False))
    compiled.close()
    with pytest.raises(RuntimePlanError, match="closed"):
        compiled(x)


def test_request_cancel_does_not_poison_sibling_forward() -> None:
    """Idle cancel is sticky for one forward, then the module recovers."""
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(model, (x,), config=CompileConfig(use_torch_compile=False, measure_regions=False))
    try:
        with torch.no_grad():
            expected = model(x)
        compiled.request_cancel()
        with pytest.raises(ExecutionCancelled):
            compiled(x)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected)
    finally:
        compiled.close()
