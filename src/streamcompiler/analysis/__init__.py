"""Graph analysis passes: alias, liveness, repeated blocks."""

from streamcompiler.analysis.alias import AliasAnalysis, run_alias_analysis
from streamcompiler.analysis.liveness import LivenessAnalysis, run_liveness_analysis
from streamcompiler.analysis.repeated_blocks import detect_repeated_blocks

__all__ = [
    "AliasAnalysis",
    "LivenessAnalysis",
    "detect_repeated_blocks",
    "run_alias_analysis",
    "run_liveness_analysis",
]
