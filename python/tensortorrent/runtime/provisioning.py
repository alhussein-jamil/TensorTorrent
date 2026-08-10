"""Compatibility facade for runtime provisioning policy.

Provisioning is intentionally split by concern:
- :mod:`worker_policy` decides executor parallelism,
- :mod:`pinning` interprets schedule/pinned-host requirements,
- :mod:`parameter_provisioning` owns parameter-store and pack lifecycle.

Keep this module as the stable import surface for existing callers.
``torch`` / ``shutil`` stay importable here so older monkeypatches that target
``tensortorrent.runtime.provisioning.*`` keep working.
"""

import shutil

import torch

from tensortorrent.runtime.parameter_provisioning import _ensure_pack, build_parameter_store
from tensortorrent.runtime.pinning import (
    pinned_host_allocatable_bytes,
    schedule_needs_host_pin,
    schedule_uses_pinned_staging,
    should_pin_parameter_store,
)
from tensortorrent.runtime.worker_policy import intraop_threads, worker_count

__all__ = [
    "_ensure_pack",
    "build_parameter_store",
    "intraop_threads",
    "pinned_host_allocatable_bytes",
    "schedule_needs_host_pin",
    "schedule_uses_pinned_staging",
    "should_pin_parameter_store",
    "shutil",
    "torch",
    "worker_count",
]
