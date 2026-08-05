"""Internal test-only helpers. Not part of the public API.

Provides a documented seam for tests that must observe schedule-path telemetry
(native artifact counters, ScheduleExecutor reports) even when the compiled
plan is eligible for the direct call. Users should never rely on this module.
"""

from __future__ import annotations

from typing import Any


def force_schedule_path(compiled: Any) -> None:
    """Force ``compiled`` to run every ``forward`` through the schedule path.

    Direct-path selection is automatic and correctness-gated at compile time;
    there is no user-facing knob. Some tests still need schedule-path bookkeeping
    (native artifact ``execute_count``, ``_last_schedule_report``, etc.) and use
    this helper to disable the direct plan on an already-compiled module.
    """
    executor = getattr(compiled, "_executor", None) or getattr(compiled, "executor", None)
    if executor is None:
        raise TypeError("force_schedule_path expected a CompiledModule with an executor")
    executor._direct_plan = None
