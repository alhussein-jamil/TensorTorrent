"""Dispatch overhead floor and GPU-absence honesty for Milestone 1 close-out."""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc


def test_micro_dispatch_overhead_stays_bounded() -> None:
    """Tiny models pay a fixed dispatch tax; keep it under a measured ceiling.

    Residual cost is report construction, the reentrancy lock, and input
    flatten/validate — not region kernels. Larger models approach eager parity.
    """
    model = nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    compiled = sc.compile(model, (x,), config=sc.CompileConfig(use_torch_compile=False))
    try:
        n = 1500
        with torch.inference_mode():
            for _ in range(50):
                model(x)
                compiled(x)
            t0 = time.perf_counter()
            for _ in range(n):
                model(x)
            eager = (time.perf_counter() - t0) / n
            t0 = time.perf_counter()
            for _ in range(n):
                compiled(x)
            compiled_s = (time.perf_counter() - t0) / n
        delta_us = (compiled_s - eager) * 1e6
        # Native schedule dispatch (artifact execute + one Compute region callback)
        # dominates on tiny models; keep a measured ceiling for regressions.
        assert delta_us < 1000.0, f"dispatch overhead {delta_us:.1f} µs exceeds 1000 µs floor"
        assert compiled.executor.uses_schedule_path
    finally:
        compiled.close()


def test_gpu_region_execution_is_explicitly_untested_without_cuda() -> None:
    """No CUDA here: backends must refuse, not fabricate, and stay labelled untested."""
    from streamcompiler.backends.base import KernelCandidate, RegionSource
    from streamcompiler.backends.cuda import CudaBackend
    from streamcompiler.errors import BackendError

    backend = CudaBackend()
    if backend.available():
        pytest.skip("CUDA present; this host can validate GPU regions elsewhere")
    assert backend.available() is False
    with pytest.raises(BackendError, match="not available"):
        backend.compile(
            RegionSource(region_id="x", module=torch.nn.Identity()),
            KernelCandidate("x", "cuda_gpu_0", "cuda", "k", "float16"),
        )
