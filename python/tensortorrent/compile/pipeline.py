"""Two-stage compilation: portable artifact + machine specialization.

Public API is re-exported from focused submodules for backward compatibility.
"""

from __future__ import annotations

from tensortorrent.compile.artifacts import PortableArtifact, SpecializedArtifact, portable_compile_from_ir
from tensortorrent.compile.cache import _attach_storage_measurement, needs_respecialization
from tensortorrent.compile.concurrency import dependency_levels, measure_concurrency_benefit
from tensortorrent.compile.entry import (
    _check_early_fit,
    _choose_fusion_candidate,
    _example_flat_inputs,
    _lower_to_portable,
    _region_state_budget,
    _streaming_region_budget,
    _synchronize_bound_accelerators,
    _time_executor,
    compile_exported_program,
)
from tensortorrent.compile.measure import capture_region_inputs
from tensortorrent.compile.specialize import (
    _decide_concurrency,
    _passthrough_specialization,
    _plan_is_cpu_accelerator,
    _planning_storage_bandwidth,
    concurrency_budget,
    specialize_for_machine,
)

__all__ = [
    "PortableArtifact",
    "SpecializedArtifact",
    "_attach_storage_measurement",
    "_check_early_fit",
    "_choose_fusion_candidate",
    "_decide_concurrency",
    "_example_flat_inputs",
    "_lower_to_portable",
    "_passthrough_specialization",
    "_plan_is_cpu_accelerator",
    "_planning_storage_bandwidth",
    "_region_state_budget",
    "_streaming_region_budget",
    "_synchronize_bound_accelerators",
    "_time_executor",
    "compile_exported_program",
    "concurrency_budget",
    "capture_region_inputs",
    "dependency_levels",
    "measure_concurrency_benefit",
    "needs_respecialization",
    "portable_compile_from_ir",
    "specialize_for_machine",
]
