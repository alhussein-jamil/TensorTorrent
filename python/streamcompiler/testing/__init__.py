"""Test utilities that prove the native runtime path was actually entered."""

from __future__ import annotations

from typing import Any

from streamcompiler.native import native_available, require_native

__all__ = [
    "NativePathError",
    "assert_native_extension_loaded",
    "assert_native_runtime_used",
    "assert_no_hot_path_schedule_conversion",
    "assert_no_python_fallback",
    "assert_scheduler_entered",
    "assert_zero_non_compute_callbacks",
    "reset_native_counters",
    "snapshot_native_counters",
]


class NativePathError(AssertionError):
    """Raised when a test expected native execution but observed a fallback."""


def snapshot_native_counters() -> dict[str, int]:
    native = require_native()
    raw = native.debug_counters()
    return {k: int(v) for k, v in raw.items()}


def reset_native_counters() -> None:
    require_native().reset_debug_counters()


def assert_native_extension_loaded() -> None:
    if not native_available():
        raise NativePathError("native extension not loaded")
    native = require_native()
    assert bool(native.native_available())
    assert str(native.native_version())


def assert_native_runtime_used(stats: dict[str, Any] | None, *, require_artifact: bool = True) -> None:
    """Assert production stats prove native scheduler entry (not mere availability)."""
    if not stats:
        raise NativePathError("missing execution stats; native path not proven")
    if stats.get("native_runtime") is not True:
        raise NativePathError(f"native_runtime not set: {stats!r}")
    if stats.get("schedule_driven") is not True:
        raise NativePathError(f"schedule_driven not set: {stats!r}")
    if require_artifact and stats.get("native_artifact_reused") is not True:
        raise NativePathError(f"native artifact not reused on forward: {stats!r}")
    if require_artifact and stats.get("native_artifact_id") is None:
        raise NativePathError(f"native_artifact_id missing: {stats!r}")


def assert_no_python_fallback(before: dict[str, int], after: dict[str, int]) -> None:
    """Canary: deleted Python/legacy DAG counters must stay at zero."""
    delta = after.get("python_fallback_enters", 0) - before.get("python_fallback_enters", 0)
    if delta != 0:
        raise NativePathError(f"python fallback entered {delta} time(s) during window")
    legacy = after.get("legacy_fallback_entries", 0) - before.get("legacy_fallback_entries", 0)
    if legacy != 0:
        raise NativePathError(f"legacy_fallback_entries delta={legacy}")


def assert_zero_non_compute_callbacks(before: dict[str, int], after: dict[str, int]) -> None:
    delta = after.get("non_compute_python_callbacks", 0) - before.get("non_compute_python_callbacks", 0)
    if delta != 0:
        raise NativePathError(f"non_compute_python_callbacks delta={delta}, want 0; before={before} after={after}")


def assert_scheduler_entered(before: dict[str, int], after: dict[str, int], *, min_enters: int = 1) -> None:
    delta = after.get("scheduler_enters", 0) - before.get("scheduler_enters", 0)
    if delta < min_enters:
        raise NativePathError(f"scheduler_enters delta={delta}, want >={min_enters}; before={before} after={after}")


def assert_no_hot_path_schedule_conversion(
    before: dict[str, int], after: dict[str, int], *, max_conversions: int = 0
) -> None:
    """Forward path must not rebuild the Rust schedule from Python objects."""
    delta = after.get("schedule_from_py_calls", 0) - before.get("schedule_from_py_calls", 0)
    if delta > max_conversions:
        raise NativePathError(f"schedule_from_py called {delta} time(s) on hot path (max={max_conversions})")
