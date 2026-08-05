"""Resident path: persistent residency, zero non-compute Python callbacks."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from tests.support.native import (
    assert_native_runtime_used,
    assert_no_hot_path_schedule_conversion,
    assert_zero_non_compute_callbacks,
    reset_native_counters,
    snapshot_native_counters,
)

import tensortorrent as tt
from tensortorrent.config import CompileConfig
from tensortorrent.native import native_available, require_native

pytestmark = pytest.mark.skipif(not native_available(), reason="native extension required")


@pytest.fixture(autouse=True)
def _force_schedule_path_for_module(monkeypatch):
    """Pin this module to the schedule path for native callback telemetry."""
    from tensortorrent.runtime import direct_path as _direct_path

    monkeypatch.setattr(_direct_path, "build_direct_plan", lambda _executor: None)
    yield


def test_resident_forward_zero_non_compute_callbacks() -> None:
    require_native()
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4)).eval()
    x = torch.randn(4, 32)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False, prefer_direct_path=False),
    )
    try:
        # Install native artifact before the hot-path counter window so the
        # one-time schedule_from_py of lazy install is not counted as churn.
        compiled.executor._schedule_executor._ensure_native_artifact()
        reset_native_counters()
        before = snapshot_native_counters()
        out = compiled(x)
        torch.testing.assert_close(out, model(x), check_device=False)
        after = snapshot_native_counters()
        stats = compiled.last_report.parameter_store
        assert_native_runtime_used(stats)
        assert stats.get("native_data_plane") is True
        assert stats.get("non_compute_python_io") is False
        assert_zero_non_compute_callbacks(before, after)
        assert_no_hot_path_schedule_conversion(before, after)
        assert after["compute_callbacks"] - before["compute_callbacks"] == 1
        assert after["native_scheduler_entries"] - before["native_scheduler_entries"] >= 1
    finally:
        compiled.close()


def test_resident_schedule_elides_fake_parameter_loads() -> None:
    """Resident packs register initial residency — no cosmetic Load ops."""
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 2)).eval()
    x = torch.randn(2, 16)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False, prefer_direct_path=False),
    )
    try:
        compiled(x)
        se = compiled.executor._schedule_executor
        stats = compiled.last_report.parameter_store
        assert stats.get("native_data_plane") is True
        loads = [
            i
            for i in se.schedule.instructions
            if i.opcode.value == "Load" and str(i.attributes.get("kind") or "") == "parameter_materialize"
        ]
        assert not loads, "resident path must not emit parameter_materialize Load"
        assert se.parameter_store.needs_prefetch is False
    finally:
        compiled.close()
