"""ROCm / HIP execution backend (AMD GPUs)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from streamcompiler.backends.base import (
    BenchmarkResult,
    CompiledRegion,
    ExecutionBackend,
    KernelCandidate,
    TransferCapability,
)
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction
from streamcompiler.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    LinkClass,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
    TransferLink,
)


class RocmBackend(ExecutionBackend):
    backend_id = "rocm"

    def available(self) -> bool:
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            # ROCm builds often report hip devices via the torch.cuda API.
            name = torch.cuda.get_device_name(0).lower()
            return "amd" in name or "radeon" in name or "instinct" in name
        except Exception:  # noqa: BLE001
            return False

    def discover_devices(self) -> ResourceGraph:
        graph = ResourceGraph(fingerprint="", backends_present=())
        if not self.available():
            graph.attributes["rocm_status"] = "runtime_unavailable"
            return graph
        import torch

        graph = ResourceGraph(fingerprint="", backends_present=(self.backend_id,))
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            name = f"rocm_gpu_{index}"
            vram = f"rocm_vram_{index}"
            total = int(props.total_memory)
            graph.add_memory(
                MemoryResource(
                    id=ResourceId(ResourceKind.MEMORY, vram),
                    memory_class=MemoryClass.DEVICE_VRAM,
                    capacity_bytes=total,
                    allocatable_bytes=int(total * 0.9),
                    attached_compute=(name,),
                )
            )
            graph.add_compute(
                ComputeResource(
                    id=ResourceId(ResourceKind.COMPUTE, name),
                    compute_class=ComputeClass.DISCRETE_GPU,
                    backend_id=self.backend_id,
                    model=props.name,
                    architecture="gfx",
                    vendor="amd",
                    supported_dtypes=("float32", "float16", "bfloat16", "int8"),
                    supported_ops=("aten::mm", "aten::addmm", "aten::convolution", "aten::relu"),
                    memory_affinity=(vram,),
                    attributes={"index": index},
                )
            )
            graph.add_link(
                TransferLink(
                    id=ResourceId(ResourceKind.LINK, f"{name}->{vram}"),
                    link_class=LinkClass.SHARED_MEMORY,
                    source=name,
                    destination=vram,
                    bidirectional=True,
                    peer_to_peer=True,
                )
            )
        return graph

    def supported_ops(self, device: ComputeResource) -> tuple[str, ...]:
        return device.supported_ops

    def supported_dtypes(self, device: ComputeResource) -> tuple[str, ...]:
        return device.supported_dtypes

    def enumerate_kernels(
        self, region: Instruction | HeterogeneousGraph, device: ComputeResource
    ) -> list[KernelCandidate]:
        region_id = region.name if isinstance(region, Instruction) else region.name
        return [
            KernelCandidate(
                region_id=region_id,
                device=device.id.name,
                backend_id=self.backend_id,
                kernel_id=f"rocm_{dtype}",
                dtype=dtype,
            )
            for dtype in device.supported_dtypes
            if dtype in ("float32", "float16", "bfloat16")
        ]

    def benchmark(self, candidate: KernelCandidate) -> BenchmarkResult:
        return BenchmarkResult(
            candidate=candidate,
            latency_s=float("inf"),
            memory_bytes=0,
            measured=False,
            notes="rocm unavailable or not benchmarked on this machine",
        )

    def compile(
        self, region: Instruction | HeterogeneousGraph, candidate: KernelCandidate
    ) -> CompiledRegion:
        if not self.available():
            raise RuntimeError("ROCm backend not available on this machine")
        return CompiledRegion(
            region_id=candidate.region_id,
            device=candidate.device,
            backend_id=self.backend_id,
            executable={"kind": "rocm", "dtype": candidate.dtype},
            dtype=candidate.dtype,
        )

    def execute(self, executable: CompiledRegion, dependencies: Sequence[Any]) -> Any:
        if not self.available():
            raise RuntimeError("ROCm backend not available on this machine")
        return {"status": "ok", "backend": self.backend_id}

    def transfer_capabilities(
        self, source: ComputeResource | str, destination: ComputeResource | str
    ) -> TransferCapability:
        src = source if isinstance(source, str) else source.id.name
        dst = destination if isinstance(destination, str) else destination.id.name
        if not self.available():
            return TransferCapability(src, dst, kind="unsupported", notes="rocm unavailable")
        return TransferCapability(src, dst, kind="host_staged", notes="default to host staging unless RCCL/P2P validated")
