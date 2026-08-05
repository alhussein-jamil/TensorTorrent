"""Default partition fallback when repeated blocks are not pre-specified."""

from __future__ import annotations

from tensortorrent.ir.graph import HeterogeneousGraph


def default_repeated_blocks(graph: HeterogeneousGraph) -> tuple[tuple[str, ...], ...]:
    """Return pre-set partitions or a single block containing all compute regions.

    This is a default partition fallback, not real repeated-block detection.
    """
    if graph.repeated_blocks:
        return graph.repeated_blocks
    names = [i.name for i in graph.compute_regions()]
    if not names:
        return ()
    return (tuple(names),)
