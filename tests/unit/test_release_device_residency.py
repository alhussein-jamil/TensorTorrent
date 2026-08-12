"""ScheduleExecutor.release_device_residency policy."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from tensortorrent.runtime.schedule_executor import ScheduleExecutor


def _bare_executor(*, hoist: bool = True) -> ScheduleExecutor:
    """Minimal ScheduleExecutor without running native install in __init__."""
    ex = object.__new__(ScheduleExecutor)
    ex._closed = False
    ex._hoist_resident_parameters = hoist
    ex._partial_hoist_oom = False
    ex._persistent_parameter_ids = None
    ex._resident_parameter_targets = {"w": ("cuda_gpu_0",)}
    ex._persistent_device_param_cache = {("w", "cuda_gpu_0"): (0, object())}
    ex._persistent_param_cache = [("w",)]
    ex.schedule = SimpleNamespace(instructions=())
    ex._run_gate = SimpleNamespace(wait_idle=lambda: None)
    ex._install_native_artifact = MagicMock()
    ex._recompute_schedule_caches = MagicMock()
    return ex


def test_release_demote_hoist_disables_flag():
    ex = _bare_executor(hoist=True)
    assert ex.release_device_residency(demote_hoist=True) is True
    assert ex._hoist_resident_parameters is False
    assert ex._partial_hoist_oom is True
    assert ex._persistent_device_param_cache == {}
    assert ex._persistent_parameter_ids == set()
    ex._install_native_artifact.assert_called_once_with(ex.schedule)
    ex._recompute_schedule_caches.assert_called_once_with(ex.schedule)


def test_release_generation_local_keeps_hoist_flag():
    ex = _bare_executor(hoist=True)
    assert ex.release_device_residency(demote_hoist=False) is True
    assert ex._hoist_resident_parameters is True
    assert ex._partial_hoist_oom is True
    assert ex._persistent_device_param_cache == {}


def test_release_on_closed_is_noop():
    ex = _bare_executor()
    ex._closed = True
    assert ex.release_device_residency(demote_hoist=True) is False
    ex._install_native_artifact.assert_not_called()


def test_release_idempotent_when_already_demoted():
    ex = _bare_executor(hoist=False)
    ex._install_native_artifact.reset_mock()
    assert ex.release_device_residency(demote_hoist=True) is True
    ex._install_native_artifact.assert_not_called()
