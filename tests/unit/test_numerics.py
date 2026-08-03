"""Numerical helper tests."""

from __future__ import annotations

import torch

from tensortorrent.validation.numerics import compare_tensors


def test_compare_tensors_identical() -> None:
    t = torch.randn(4, 4)
    report = compare_tensors(t, t.clone())
    assert report.passed
    assert report.max_abs_err == 0.0


def test_compare_tensors_detects_drift() -> None:
    a = torch.zeros(2, 2)
    b = torch.ones(2, 2)
    report = compare_tensors(a, b, atol=1e-6, rtol=1e-6)
    assert not report.passed
    assert report.max_abs_err == 1.0
