"""Native-path gate helpers (re-export)."""

from streamcompiler.testing import (
    NativePathError,
    assert_native_extension_loaded,
    assert_native_runtime_used,
    assert_no_hot_path_schedule_conversion,
    assert_no_python_fallback,
    assert_scheduler_entered,
    assert_zero_non_compute_callbacks,
    reset_native_counters,
    snapshot_native_counters,
)

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
