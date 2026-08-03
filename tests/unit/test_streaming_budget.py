"""Compile-time helpers that size regions for streaming budgets."""

from __future__ import annotations

from tensortorrent.compile.pipeline import _streaming_region_budget
from tensortorrent.config import CompileConfig


def test_streaming_region_budget_scales_with_prefetch_distance() -> None:
    assert _streaming_region_budget(CompileConfig(ram_budget_bytes=None)) is None
    assert _streaming_region_budget(CompileConfig(ram_budget_bytes=1000, prefetch_distance=0)) == 1000
    assert _streaming_region_budget(CompileConfig(ram_budget_bytes=1000, prefetch_distance=1)) == 500
    assert _streaming_region_budget(CompileConfig(ram_budget_bytes=1000, prefetch_distance=3)) == 250
