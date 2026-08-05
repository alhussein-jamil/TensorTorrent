"""Bridge: Rust schedules instructions; Python executes tensor-bearing ops."""

from __future__ import annotations

import tempfile

from tensortorrent.runtime.native_bridge.residency import (
    _alias_host_compute_resources,
    _configure_virtual_backends,
    _move_tensor_to_resource,
    _register_persistent_residency,
    _schedule_needs_parameter_load,
    _schedule_needs_spill_callbacks,
)
from tensortorrent.runtime.native_bridge.run import _reraise_pending, _run_schedule_native_body, run_schedule_native
from tensortorrent.runtime.native_bridge.spill import (
    _check_not_tmpfs,
    _merge_native_streaming_io_intervals,
    _resolve_spill_dir,
    _resolve_spill_root,
    _setup_native_spill,
)

__all__ = [
    "_alias_host_compute_resources",
    "_check_not_tmpfs",
    "_configure_virtual_backends",
    "_merge_native_streaming_io_intervals",
    "_move_tensor_to_resource",
    "_register_persistent_residency",
    "_reraise_pending",
    "_resolve_spill_dir",
    "_resolve_spill_root",
    "_run_schedule_native_body",
    "_schedule_needs_parameter_load",
    "_schedule_needs_spill_callbacks",
    "_setup_native_spill",
    "run_schedule_native",
    "tempfile",
]
