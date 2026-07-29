"""Resident path: persistent residency, zero non-compute Python callbacks."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.config import CompileConfig
from streamcompiler.native import native_available, require_native
from streamcompiler.testing import (
    assert_native_runtime_used,
    assert_no_hot_path_schedule_conversion,
    assert_zero_non_compute_callbacks,
    reset_native_counters,
    snapshot_native_counters,
)

pytestmark = pytest.mark.skipif(not native_available(), reason="native extension required")


def test_resident_forward_zero_non_compute_callbacks() -> None:
    require_native()
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4)).eval()
    x = torch.randn(4, 32)
    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        reset_native_counters()
        before = snapshot_native_counters()
        out = compiled(x)
        torch.testing.assert_close(out, model(x))
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


def test_load_events_are_schedule_native_not_prematerialized() -> None:
    """Load must appear as a real native schedule event, not a fake pre-run."""
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 2)).eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        compiled(x)
        se = compiled.executor._schedule_executor
        # Drive through schedule executor report via a fresh forward and inspect
        # instruction notes from the last ScheduleReport attached to stats.
        stats = compiled.last_report.parameter_store
        assert stats.get("native_data_plane") is True
        # Schedule always has a parameter Load for this model.
        loads = [i for i in se.schedule.instructions if i.opcode.value == "Load"]
        assert loads, "expected parameter Load in schedule"
        # Re-run capturing schedule report events from the native bridge path.
        from streamcompiler.runtime.native_bridge import run_schedule_native

        flat = [x]
        _outs, report = run_schedule_native(se, flat)
        load_events = [e for e in report.events if e.opcode == "Load"]
        assert load_events, "Load must execute at schedule position"
        assert all(e.notes in {"persistent_residency", "native_data_plane"} for e in load_events)
        assert all(e.notes != "prematerialized" for e in load_events)
    finally:
        compiled.close()
