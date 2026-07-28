"""Deterministic I/O ∩ compute interval math (no sleeps)."""

from __future__ import annotations

import pytest

from streamcompiler.runtime.tensor_store import intersect_interval_length, merge_intervals


def test_merge_intervals_collapses_overlaps() -> None:
    assert merge_intervals([(0.0, 1.0), (0.5, 1.5), (2.0, 2.5)]) == [(0.0, 1.5), (2.0, 2.5)]


def test_intersect_proves_overlap_length() -> None:
    io = [(0.0, 0.05), (0.10, 0.20)]
    compute = [(0.02, 0.12)]
    overlapped = intersect_interval_length(merge_intervals(io), merge_intervals(compute))
    # [0.02, 0.05] + [0.10, 0.12] = 0.05
    assert overlapped == pytest.approx(0.05)


def test_no_overlap_when_windows_disjoint() -> None:
    io = [(0.0, 0.01)]
    compute = [(0.02, 0.03)]
    assert intersect_interval_length(io, compute) == 0.0
