"""Failure and fallback path tests."""

from __future__ import annotations

import pytest

from streamcompiler.backends.cuda import CudaBackend
from streamcompiler.communication import HostStagedComm, select_communication_backend
from streamcompiler.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
)
from streamcompiler.runtime.executor import TieredAllocator


def test_cuda_compile_fails_explicitly_when_unavailable() -> None:
    backend = CudaBackend()
    if backend.available():
        pytest.skip("CUDA is available on this machine")
    from streamcompiler.backends.base import KernelCandidate
    from streamcompiler.ir.graph import Instruction, OpCode

    with pytest.raises(RuntimeError, match="not available"):
        backend.compile(
            Instruction(opcode=OpCode.COMPUTE, name="x"),
            KernelCandidate("x", "cuda_gpu_0", "cuda", "k", "float16"),
        )


def test_mixed_devices_select_host_staged_when_needed() -> None:
    backend = select_communication_backend(("cuda_gpu_0", "rocm_gpu_0"))
    # On this CPU-only host, NCCL/RCCL are unavailable → host-staged or gloo.
    assert backend.backend_id in {"host_staged", "gloo"}
    caps = HostStagedComm().capabilities(("cuda_gpu_0", "rocm_gpu_0"))
    assert caps.available
    assert "allreduce" in caps.ops


def test_allocator_raises_on_overcommit() -> None:
    g = ResourceGraph(fingerprint="t")
    g.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "tiny"),
            memory_class=MemoryClass.DEVICE_VRAM,
            capacity_bytes=1000,
            allocatable_bytes=1000,
        )
    )
    g.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "gpu"),
            compute_class=ComputeClass.DISCRETE_GPU,
            backend_id="cuda",
            model="x",
            memory_affinity=("tiny",),
        )
    )
    alloc = TieredAllocator(g)
    alloc.allocate("tiny", 800)
    with pytest.raises(Exception, match="exceed"):
        alloc.allocate("tiny", 300)
