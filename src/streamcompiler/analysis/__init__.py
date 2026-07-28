"""Graph analysis passes: alias, liveness, repeated blocks, redundancy."""

from streamcompiler.analysis.alias import AliasAnalysis, run_alias_analysis
from streamcompiler.analysis.liveness import LivenessAnalysis, run_liveness_analysis
from streamcompiler.analysis.redundancy import RedundancyReport, eliminate_redundancy
from streamcompiler.analysis.repeated_blocks import detect_repeated_blocks

__all__ = [
    "AliasAnalysis",
    "LivenessAnalysis",
    "RedundancyReport",
    "detect_repeated_blocks",
    "eliminate_redundancy",
    "run_alias_analysis",
    "run_liveness_analysis",
]
