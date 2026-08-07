"""Bridge: Rust schedules instructions; Python executes tensor-bearing ops."""

from __future__ import annotations

import tempfile

from tensortorrent.runtime.native_bridge.residency import _move_tensor_to_resource
from tensortorrent.runtime.native_bridge.run import run_schedule_native
from tensortorrent.runtime.native_bridge.spill import (
    _check_not_tmpfs,
    _resolve_spill_dir,
    _resolve_spill_root,
)

__all__ = [
    "_check_not_tmpfs",
    "_move_tensor_to_resource",
    "_resolve_spill_dir",
    "_resolve_spill_root",
    "run_schedule_native",
    "tempfile",
]
