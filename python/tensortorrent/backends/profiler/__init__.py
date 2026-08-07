"""Backend-neutral profiling interface.

CPU and CUDA profilers report measured timings when the runtime is available.
Virtual-accelerator profilers are always labelled simulated.
"""

from __future__ import annotations

from typing import Any

from tensortorrent.backends.profiler.base import (
    _MAX_TRANSFER_PROFILE_BYTES,
    BackendProfiler,
    ProfileRecord,
    _bounded_transfer_size,
)
from tensortorrent.backends.profiler.cpu import CpuBackendProfiler
from tensortorrent.backends.profiler.cuda import CudaBackendProfiler
from tensortorrent.backends.profiler.virtual import VirtualAccelBackendProfiler
from tensortorrent.backends.profiler.xpu import XpuBackendProfiler


def profiler_for_backend(backend_id: str, **kwargs: Any) -> BackendProfiler:
    if backend_id in {"cpu", "cpu_numa"}:
        return CpuBackendProfiler()
    if backend_id == "cuda":
        return CudaBackendProfiler(**kwargs, backend_id="cuda")
    if backend_id == "rocm":
        return CudaBackendProfiler(**kwargs, backend_id="rocm")
    if backend_id == "xpu":
        return XpuBackendProfiler(**kwargs)
    if backend_id == "mock_accel":
        return VirtualAccelBackendProfiler(**kwargs)
    raise NotImplementedError(
        f"BackendProfiler for {backend_id!r} is not implemented; "
        f"real accelerator profiling requires a validated backend profiler"
    )


__all__ = [
    "BackendProfiler",
    "CpuBackendProfiler",
    "CudaBackendProfiler",
    "ProfileRecord",
    "VirtualAccelBackendProfiler",
    "XpuBackendProfiler",
    "_MAX_TRANSFER_PROFILE_BYTES",
    "_bounded_transfer_size",
    "profiler_for_backend",
]
