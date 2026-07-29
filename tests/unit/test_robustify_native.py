"""Regression: cancel must not silently restart; concurrent cancel isolation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.config import CompileConfig
from streamcompiler.errors import ExecutionCancelled, RuntimePlanError
from streamcompiler.ir.graph import OpCode
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


class _Branching(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(16, 16)
        self.left = nn.Linear(16, 16)
        self.right = nn.Linear(16, 16)
        self.head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.stem(x))
        return self.head(torch.relu(self.left(h)) + torch.tanh(self.right(h)))


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


def test_activation_spill_temp_dir_cleaned_after_forward(monkeypatch) -> None:
    """Spill workspace must not leak temp dirs after a successful forward."""
    import streamcompiler.runtime.native_bridge as nb

    created: list[str] = []
    real_mkdtemp = nb.tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(nb.tempfile, "mkdtemp", tracking_mkdtemp)

    model = _Branching().eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(
            max_concurrent_regions=2,
            use_torch_compile=False,
            activation_budget_bytes=64,
            measure_regions=False,
        ),
    )
    try:
        schedule = compiled.specialized.schedule
        assert schedule is not None
        spills = [
            i
            for i in schedule.instructions
            if i.opcode == OpCode.EVICT and i.attributes.get("kind") == "activation_spill"
        ]
        assert spills, "expected spill ops under tiny budget"
        with torch.no_grad():
            torch.testing.assert_close(compiled(x), model(x), atol=1e-5, rtol=1e-5)
        assert created, "expected spill temp dir creation"
        for path in created:
            assert not Path(path).exists(), f"spill dir leaked: {path}"
    finally:
        compiled.close()


def test_release_keeps_opaque_handle_for_transfer_alias() -> None:
    """Release must not drop Python values still cited by Rust Transfer dests."""
    from streamcompiler.runtime.handles import NativeResidencyBridge

    bridge = NativeResidencyBridge.create()
    t = torch.randn(4)
    bridge.mirror_put("act", "cpu", t, nbytes=int(t.nbytes))
    bridge.mirror_alias("act", "cpu", "cpu_copy")
    bridge.release("act", "cpu")
    assert not bridge.session.has("act", "cpu")
    assert bridge.require_value("act", "cpu_copy") is t


def test_native_forward_does_not_construct_device_streams() -> None:
    """Production native path must not allocate Python DeviceStreams / sync pools."""
    model = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    compiled = sc.compile(model, (x,), config=CompileConfig(use_torch_compile=False, measure_regions=False))
    try:
        compiled(x)
        se = compiled.executor._schedule_executor
        assert se is not None
        assert se._streams is None
        assert se._sync_pool is None
        assert se._native_artifact is not None
    finally:
        compiled.close()


def test_native_forward_uses_value_bag_only_copystore() -> None:
    model = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    compiled = sc.compile(model, (x,), config=CompileConfig(use_torch_compile=False, measure_regions=False))
    try:
        compiled(x)
        se = compiled.executor._schedule_executor
        assert se is not None
        assert se.copies.value_bag_only is True
    finally:
        compiled.close()


def test_spill_bytes_to_tensor_keeps_backing_no_clone() -> None:
    from streamcompiler.runtime.activation_spill import spill_bytes_to_tensor

    raw = (torch.arange(8, dtype=torch.float32) * 0.5).numpy().tobytes()
    t = spill_bytes_to_tensor("float32", [2, 4], raw)
    assert t.shape == (2, 4)
    assert hasattr(t, "_sc_spill_buf")
    assert t[0, 0].item() == pytest.approx(0.0)
    assert t[0, 1].item() == pytest.approx(0.5)
