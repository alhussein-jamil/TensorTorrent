"""Two-stage compilation: portable artifact + machine specialization.

Re-exports the public compile surface and a few helpers tests still import.
"""

from __future__ import annotations

from tensortorrent.compile.artifacts import PortableArtifact, SpecializedArtifact, portable_compile_from_ir
from tensortorrent.compile.cache import needs_respecialization
from tensortorrent.compile.concurrency import dependency_levels, measure_concurrency_benefit
from tensortorrent.compile.entry import (
    _check_early_fit,
    _choose_fusion_candidate,
    _synchronize_bound_accelerators,
    _time_executor,
    compile_exported_program,
)
from tensortorrent.compile.fit import region_state_budget as _region_state_budget
from tensortorrent.compile.fit import streaming_region_budget as _streaming_region_budget
from tensortorrent.compile.measure import capture_region_inputs
from tensortorrent.compile.specialize import (
    _decide_concurrency,
    _plan_is_cpu_accelerator,
    concurrency_budget,
    specialize_for_machine,
)

__all__ = [
    "PortableArtifact",
    "SpecializedArtifact",
    "_check_early_fit",
    "_choose_fusion_candidate",
    "_decide_concurrency",
    "_plan_is_cpu_accelerator",
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
