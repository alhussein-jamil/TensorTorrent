"""Shared NativeExecutionContext: no mid-forward restart; one residency store."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.config import CompileConfig
from tensortorrent.errors import RuntimePlanError
from tensortorrent.native import native_available, require_native

pytestmark = pytest.mark.skipif(not native_available(), reason="native required")


@pytest.fixture(autouse=True)
def _force_schedule_path_for_module(monkeypatch):
    """Pin this module to the schedule path for shared-context telemetry."""
    from tensortorrent.runtime import direct_path as _direct_path

    monkeypatch.setattr(_direct_path, "build_direct_plan", lambda _executor: None)
    yield


def test_region_path_uses_shared_execution_context() -> None:
    model = nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        out = compiled(x)
        torch.testing.assert_close(out, model(x), check_device=False)
        stats = compiled.last_report.parameter_store  # type: ignore[union-attr]
        assert stats.get("native_data_plane") is True
        assert stats.get("native_residency") is True
        assert stats.get("native_execution_id") is not None
        rs = stats.get("native_residency_stats") or {}
        assert rs.get("shared_execution_context") is True
        assert int(rs.get("execution_id")) == int(stats["native_execution_id"])
    finally:
        compiled.close()


def test_region_path_failure_does_not_restart_instruction_callback() -> None:
    """Static path selection: region failure must not re-run via instruction handler."""
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    x = torch.randn(2, 8)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        compiled(x)
        se = compiled.executor._schedule_executor
        calls = {"n": 0}

        class _BoomArtifact:
            artifact_id = 42

            def execute(self, **kwargs):
                calls["n"] += 1
                # Prove we are on the region path (execution_context wired).
                assert kwargs.get("execution_context") is not None or kwargs.get("region_callback") is not None
                raise RuntimeError("intentional region-path boom")

            def is_unmutated(self) -> bool:
                return True

        real = se._native_artifact
        se._native_artifact = _BoomArtifact()
        with pytest.raises(RuntimePlanError, match="intentional region-path boom|native schedule"):
            compiled(x)
        assert calls["n"] == 1, "must not restart through a second execute path"
        se._native_artifact = real
    finally:
        compiled.close()


def test_native_execution_context_api() -> None:
    native = require_native()
    ctx = native.NativeExecutionContext()
    assert int(ctx.execution_id) > 0
    session = native.NativeResidencySession.from_execution_context(ctx)
    assert session.execution_id == ctx.execution_id
    session.put("t0", "cpu", 7, 64, True)
    assert session.has("t0", "cpu")
    assert session.require("t0", "cpu") == 7
