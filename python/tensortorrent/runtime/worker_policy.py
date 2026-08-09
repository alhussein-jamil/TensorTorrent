"""Worker-count and intra-op thread policy for compiled execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensortorrent.config import CompileConfig

if TYPE_CHECKING:
    from tensortorrent.compile.pipeline import SpecializedArtifact


def worker_count(specialized: SpecializedArtifact, config: CompileConfig) -> int:
    """Return the number of regions that may execute simultaneously."""
    if not config.allow_concurrent_regions:
        return 1
    if config.max_concurrent_regions > 0:
        return config.max_concurrent_regions
    decision = specialized.validation.get("concurrency")
    if isinstance(decision, dict):
        return max(1, int(decision.get("workers", 1)))
    return 1


def intraop_threads(specialized: SpecializedArtifact, config: CompileConfig) -> int:
    """Return intra-op threads per worker, or 0 to leave PyTorch unchanged."""
    if not config.allow_concurrent_regions:
        return 0
    decision = specialized.validation.get("concurrency")
    if isinstance(decision, dict) and decision.get("enabled"):
        return max(0, int(decision.get("intraop_threads", 0)))
    return 0
