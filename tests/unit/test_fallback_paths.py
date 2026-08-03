"""Failure and fallback path tests."""

from __future__ import annotations

import pytest
import torch

from tensortorrent.backends.communication import HostStagedComm, select_communication_backend
from tensortorrent.backends.cuda import CudaBackend


def test_cuda_compile_fails_explicitly_when_unavailable() -> None:
    backend = CudaBackend()
    if backend.available():
        pytest.skip("CUDA is available on this machine")
    from tensortorrent.backends.base import KernelCandidate, RegionSource
    from tensortorrent.errors import BackendError

    with pytest.raises(BackendError, match="not available"):
        backend.compile(
            RegionSource(region_id="x", module=torch.nn.Identity()),
            KernelCandidate("x", "cuda_gpu_0", "cuda", "k", "float16"),
        )


def test_cuda_execute_fails_explicitly_when_unavailable() -> None:
    backend = CudaBackend()
    if backend.available():
        pytest.skip("CUDA is available on this machine")
    from tensortorrent.backends.base import CompiledRegion
    from tensortorrent.errors import BackendError

    region = CompiledRegion(
        region_id="x",
        device="cuda_gpu_0",
        backend_id="cuda",
        executable=torch.nn.Identity(),
        dtype="float16",
    )
    with pytest.raises(BackendError, match="not available"):
        backend.execute(region, (torch.randn(2),))


def test_compiled_region_rejects_non_callable_executables() -> None:
    """A backend must never hand the runtime a status dictionary."""
    from tensortorrent.backends.base import CompiledRegion

    with pytest.raises(TypeError, match="must be callable"):
        CompiledRegion(
            region_id="x",
            device="cpu_numa_0",
            backend_id="cpu",
            executable={"status": "ok"},
            dtype="float32",
        )


def test_mixed_devices_select_host_staged_when_needed() -> None:
    backend = select_communication_backend(("cuda_gpu_0", "rocm_gpu_0"))
    # Mixed CUDA+ROCm: vendor-specific collectives do not span both → host/gloo.
    assert backend.backend_id in {"host_staged", "gloo"}
    caps = HostStagedComm().capabilities(("cuda_gpu_0", "rocm_gpu_0"))
    assert caps.available
    assert "allreduce" in caps.ops
