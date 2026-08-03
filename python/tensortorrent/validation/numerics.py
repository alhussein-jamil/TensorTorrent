"""Numerical comparison helpers against eager PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class NumericalReport:
    max_abs_err: float
    mean_abs_err: float
    passed: bool
    atol: float
    rtol: float


def compare_tensors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> NumericalReport:
    actual = actual.detach().cpu().float()
    expected = expected.detach().cpu().float()
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch {tuple(actual.shape)} vs {tuple(expected.shape)}")
    diff = (actual - expected).abs()
    max_err = float(diff.max()) if diff.numel() else 0.0
    mean_err = float(diff.mean()) if diff.numel() else 0.0
    passed = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
    return NumericalReport(max_err, mean_err, passed, atol, rtol)


def compare_module_outputs(
    compiled_out: Any,
    eager_out: Any,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> NumericalReport:
    if isinstance(compiled_out, (tuple, list)):
        reports = [compare_tensors(a, b, atol=atol, rtol=rtol) for a, b in zip(compiled_out, eager_out, strict=True)]
        return NumericalReport(
            max(r.max_abs_err for r in reports),
            sum(r.mean_abs_err for r in reports) / len(reports),
            all(r.passed for r in reports),
            atol,
            rtol,
        )
    return compare_tensors(compiled_out, eager_out, atol=atol, rtol=rtol)
