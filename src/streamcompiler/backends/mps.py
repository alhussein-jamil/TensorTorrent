"""Apple MPS execution backend."""

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
from streamcompiler.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
)


class MpsBackend(ExecutionBackend):
    backend_id = "mps"

    def available(self) -> bool:
        try:
            import torch

            return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        except Exception:  # noqa: BLE001
            return False

    def discover_devices(self) -> ResourceGraph:
        graph = ResourceGraph(fingerprint="", backends_present=())
        if not self.available():
            graph.attributes["mps_status"] = "runtime_unavailable"
            return graph
        import psutil

        graph = ResourceGraph(fingerprint="", backends_present=(self.backend_id,))
        # MPS uses unified memory; model as integrated GPU + shared memory.
        total = int(psutil.virtual_memory().total)
        mem_name = "mps_unified"
        gpu_name = "mps_gpu_0"
        graph.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, mem_name),
                memory_class=MemoryClass.UNIFIED_SHARED,
                capacity_bytes=total,
                allocatable_bytes=int(total * 0.5),
                attached_compute=(gpu_name,),
            )
        )
        graph.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, gpu_name),
                compute_class=ComputeClass.INTEGRATED_GPU,
                backend_id=self.backend_id,
                model="Apple MPS",
                architecture="metal",
                vendor="apple",
                supported_dtypes=("float32", "float16", "bfloat16"),
                supported_ops=("aten::mm", "aten::addmm", "aten::convolution", "aten::relu"),
                memory_affinity=(mem_name,),
            )
        )
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
                kernel_id=f"mps_{dtype}",
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
            notes="mps unavailable or not benchmarked on this machine",
        )

    def compile(self, region: RegionSource, candidate: KernelCandidate) -> CompiledRegion:
        if not self.available():
            raise BackendError(
                "mps backend is not available on this machine; StreamCompiler will not fabricate a compiled region"
            )
        return compile_region_for_torch_device(
            region,
            candidate,
            backend_id=self.backend_id,
            torch_device=str(self.resource_to_torch_device(candidate.device)),
        )

    def execute(self, executable: CompiledRegion, inputs: Sequence[Any]) -> tuple[Any, ...]:
        if not self.available():
            raise BackendError("mps backend is not available on this machine")
        return execute_region_on_torch_device(executable, inputs)

    def transfer_capabilities(
        self, source: ComputeResource | str, destination: ComputeResource | str
    ) -> TransferCapability:
        src = source if isinstance(source, str) else source.id.name
        dst = destination if isinstance(destination, str) else destination.id.name
        return TransferCapability(src, dst, kind="shared", notes="MPS unified memory")

    def resource_to_torch_device(self, resource_id: str) -> Any:
        import torch

        return torch.device("mps")
