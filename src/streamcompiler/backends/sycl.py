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
    RegionSource,
    TransferCapability,
    region_identifier,
)
from streamcompiler.backends.torch_device import (
    compile_region_for_torch_device,
    execute_region_on_torch_device,
)
from streamcompiler.errors import BackendError
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction
from streamcompiler.ir.resource_graph import ComputeResource, ResourceGraph


def _device_index(device_name: str) -> int:
    """Extract the vendor device ordinal from a resource name like ``gpu_cuda_1``."""
    tail = device_name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


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
        self, region: RegionSource | Instruction | HeterogeneousGraph, device: ComputeResource
    ) -> list[KernelCandidate]:
        region_id = region_identifier(region)
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

    def compile(self, region: RegionSource, candidate: KernelCandidate) -> CompiledRegion:
        if not self.available():
            raise BackendError(
                "sycl backend is not available on this machine; StreamCompiler will not fabricate a compiled region"
            )
        return compile_region_for_torch_device(
            region,
            candidate,
            backend_id=self.backend_id,
            torch_device=str(self.resource_to_torch_device(candidate.device)),
        )

    def execute(self, executable: CompiledRegion, inputs: Sequence[Any]) -> tuple[Any, ...]:
        if not self.available():
            raise BackendError("sycl backend is not available on this machine")
        return execute_region_on_torch_device(executable, inputs)

    def transfer_capabilities(
        self, source: ComputeResource | str, destination: ComputeResource | str
    ) -> TransferCapability:
        src = source if isinstance(source, str) else source.id.name
        dst = destination if isinstance(destination, str) else destination.id.name
        if not self.available():
            return TransferCapability(src, dst, kind="unsupported", notes="sycl unavailable")
        return TransferCapability(src, dst, kind="host_staged", notes="cross-vendor default host staging")

    def resource_to_torch_device(self, resource_id: str) -> Any:
        import torch

        return torch.device(f"xpu:{_device_index(resource_id)}")
