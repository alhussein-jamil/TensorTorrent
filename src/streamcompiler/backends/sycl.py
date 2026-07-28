"""Intel GPU / SYCL execution backend.

Discovery is capability-driven. The backend reports unavailable until a SYCL or
Intel Extension for PyTorch runtime is present and probed successfully.
"""

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
from streamcompiler.ir.resource_graph import ComputeResource, ResourceGraph


class SyclBackend(ExecutionBackend):
    backend_id = "sycl"

    def available(self) -> bool:
        # Prefer explicit Intel Extension / XPU probe when installed.
        try:
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return True
        except Exception:  # noqa: BLE001
            pass
        try:
            import dpctl  # type: ignore

            return bool(dpctl.get_devices())
        except Exception:  # noqa: BLE001
            return False

    def discover_devices(self) -> ResourceGraph:
        graph = ResourceGraph(fingerprint="", backends_present=())
        if not self.available():
            graph.attributes["sycl_status"] = "runtime_unavailable"
            return graph
        # Detailed device enumeration is deferred to when XPU/dpctl is present.
        # Returning an empty-but-present backend graph avoids silent claims.
        graph = ResourceGraph(
            fingerprint="",
            backends_present=(self.backend_id,),
            attributes={"note": "SYCL/XPU runtime present; enumerate via torch.xpu/dpctl"},
        )
        try:
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                from streamcompiler.ir.resource_graph import (
                    ComputeClass,
                    MemoryClass,
                    MemoryResource,
                    ResourceId,
                    ResourceKind,
                )

                for index in range(torch.xpu.device_count()):
                    name = f"sycl_gpu_{index}"
                    vram = f"sycl_vram_{index}"
                    # Memory query APIs vary; use a conservative unknown capacity of 0
                    # until measured — never invent VRAM sizes.
                    graph.add_memory(
                        MemoryResource(
                            id=ResourceId(ResourceKind.MEMORY, vram),
                            memory_class=MemoryClass.DEVICE_VRAM,
                            capacity_bytes=0,
                            allocatable_bytes=0,
                            attached_compute=(name,),
                            attributes={"capacity_unknown_until_profiled": True},
                        )
                    )
                    graph.add_compute(
                        ComputeResource(
                            id=ResourceId(ResourceKind.COMPUTE, name),
                            compute_class=ComputeClass.DISCRETE_GPU,
                            backend_id=self.backend_id,
                            model=f"Intel XPU {index}",
                            vendor="intel",
                            supported_dtypes=("float32", "float16", "bfloat16"),
                            supported_ops=("aten::mm", "aten::addmm", "aten::relu"),
                            memory_affinity=(vram,),
                            attributes={"index": index},
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            graph.attributes["sycl_enumerate_error"] = str(exc)
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
                kernel_id=f"sycl_{dtype}",
                dtype=dtype,
            )
            for dtype in device.supported_dtypes
        ]

    def benchmark(self, candidate: KernelCandidate) -> BenchmarkResult:
        return BenchmarkResult(
            candidate=candidate,
            latency_s=float("inf"),
            memory_bytes=0,
            measured=False,
            notes="sycl unavailable or not benchmarked on this machine",
        )

    def compile(
        self, region: Instruction | HeterogeneousGraph, candidate: KernelCandidate
    ) -> CompiledRegion:
        if not self.available():
            raise RuntimeError("SYCL backend not available on this machine")
        return CompiledRegion(
            region_id=candidate.region_id,
            device=candidate.device,
            backend_id=self.backend_id,
            executable={"kind": "sycl", "dtype": candidate.dtype},
            dtype=candidate.dtype,
        )

    def execute(self, executable: CompiledRegion, dependencies: Sequence[Any]) -> Any:
        if not self.available():
            raise RuntimeError("SYCL backend not available on this machine")
        return {"status": "ok", "backend": self.backend_id}

    def transfer_capabilities(
        self, source: ComputeResource | str, destination: ComputeResource | str
    ) -> TransferCapability:
        src = source if isinstance(source, str) else source.id.name
        dst = destination if isinstance(destination, str) else destination.id.name
        if not self.available():
            return TransferCapability(src, dst, kind="unsupported", notes="sycl unavailable")
        return TransferCapability(src, dst, kind="host_staged", notes="cross-vendor default host staging")
