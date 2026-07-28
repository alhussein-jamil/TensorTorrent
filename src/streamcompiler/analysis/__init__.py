"""Graph analysis passes: alias, liveness, repeated blocks."""

from streamcompiler.analysis.alias import AliasAnalysis, run_alias_analysis, storage_bytes_by_group
from streamcompiler.analysis.liveness import (
    LivenessAnalysis,
    peak_live_bytes,
    ranges_overlap,
    run_liveness_analysis,
)
from streamcompiler.analysis.repeated_blocks import detect_repeated_blocks

__all__ = [
    "AliasAnalysis",
    "LivenessAnalysis",
    "detect_repeated_blocks",
    "peak_live_bytes",
    "ranges_overlap",
    "run_alias_analysis",
    "run_liveness_analysis",
    "storage_bytes_by_group",
]
