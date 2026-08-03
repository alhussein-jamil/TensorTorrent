"""Repeated block detection."""

from __future__ import annotations

from tensortorrent.ir.graph import HeterogeneousGraph


def detect_repeated_blocks(graph: HeterogeneousGraph) -> tuple[tuple[str, ...], ...]:
    if graph.repeated_blocks:
        return graph.repeated_blocks
    names = [i.name for i in graph.compute_regions()]
    if not names:
        return ()
    return (tuple(names),)
