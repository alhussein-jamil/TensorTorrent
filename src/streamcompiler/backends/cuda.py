"""CUDA execution backend.

Capabilities are queried from the CUDA runtime when present. Absence of CUDA on
the development machine is reported honestly — never treated as proof that GPU
execution works on production machines.
"""

from __future__ import annotations

import time
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
    LinkClass,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
    TransferLink,
)


def _device_index(device_name: str) -> int:
    """Extract the vendor device ordinal from a resource name like ``gpu_cuda_1``."""
    tail = device_name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


class CudaBackend(ExecutionBackend):
    backend_id = "cuda"

    def available(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False

    def discover_devices(self) -> ResourceGraph:
        graph = ResourceGraph(fingerprint="", backends_present=())
        if not self.available():
            graph.attributes["cuda_status"] = "runtime_unavailable"
            return graph
        import torch

        graph = ResourceGraph(fingerprint="", backends_present=(self.backend_id,))
        count = torch.cuda.device_count()
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            name = f"cuda_gpu_{index}"
            vram_name = f"cuda_vram_{index}"
            total_mem = int(props.total_memory)
            dtypes = self._probe_dtypes(index)
            graph.add_memory(
                MemoryResource(
                    id=ResourceId(ResourceKind.MEMORY, vram_name),
                    memory_class=MemoryClass.DEVICE_VRAM,
                    capacity_bytes=total_mem,
                    allocatable_bytes=int(total_mem * 0.9),
                    attached_compute=(name,),
                    attributes={"backend": self.backend_id, "index": index},
                )
            )
            graph.add_compute(
                ComputeResource(
                    id=ResourceId(ResourceKind.COMPUTE, name),
                    compute_class=ComputeClass.DISCRETE_GPU,
                    backend_id=self.backend_id,
                    model=props.name,
                    architecture=f"sm_{props.major}{props.minor}",
                    vendor="nvidia",
                    supported_dtypes=dtypes,
                    supported_ops=self._op_probe(),
                    compute_capability=f"{props.major}.{props.minor}",
                    core_count=int(getattr(props, "multi_processor_count", 0)),
                    copy_engines=2,
                    concurrency_limit=8,
                    memory_affinity=(vram_name,),
                    attributes={"index": index, "uuid": getattr(props, "uuid", None)},
                )
            )
            graph.add_link(
                TransferLink(
                    id=ResourceId(ResourceKind.LINK, f"{name}->{vram_name}"),
                    link_class=LinkClass.SHARED_MEMORY,
                    source=name,
                    destination=vram_name,
                    bidirectional=True,
                    peer_to_peer=True,
                    measured=False,
                )
            )

        # Peer-to-peer matrix — query, never assume.
        for i in range(count):
            for j in range(count):
                if i == j:
                    continue
                can_p2p = False
                try:
                    can_p2p = bool(torch.cuda.can_device_access_peer(i, j))
                except Exception:  # noqa: BLE001
                    can_p2p = False
                src = f"cuda_vram_{i}"
                dst = f"cuda_vram_{j}"
                graph.add_link(
                    TransferLink(
                        id=ResourceId(ResourceKind.LINK, f"{src}->{dst}"),
                        link_class=LinkClass.NVLINK if can_p2p else LinkClass.PCIE,
                        source=src,
                        destination=dst,
                        bidirectional=False,
                        peer_to_peer=can_p2p,
                        measured=False,
                        attributes={"access_peer": can_p2p},
                    )
                )
        return graph

    def _probe_dtypes(self, index: int) -> tuple[str, ...]:
        import torch

        found: list[str] = ["float32", "float16", "int8", "int32", "bool"]
        # Capability-query style probes; failures mean unsupported, not errors.
        try:
            torch.empty(1, device=f"cuda:{index}", dtype=torch.bfloat16)
            found.append("bfloat16")
        except Exception:  # noqa: BLE001
            pass
        try:
            if hasattr(torch, "float8_e4m3fn"):
                torch.empty(1, device=f"cuda:{index}", dtype=torch.float8_e4m3fn)
                found.append("float8_e4m3fn")
        except Exception:  # noqa: BLE001
            pass
        return tuple(found)

    def _op_probe(self) -> tuple[str, ...]:
        return (
            "aten::mm",
            "aten::addmm",
            "aten::bmm",
            "aten::convolution",
            "aten::relu",
            "aten::gelu",
            "aten::softmax",
            "aten::layer_norm",
            "aten::scaled_dot_product_attention",
            "aten::add",
            "aten::mul",
        )

    def supported_ops(self, device: ComputeResource) -> tuple[str, ...]:
        return device.supported_ops or self._op_probe()

    def supported_dtypes(self, device: ComputeResource) -> tuple[str, ...]:
        return device.supported_dtypes

    def enumerate_kernels(
        self, region: RegionSource | Instruction | HeterogeneousGraph, device: ComputeResource
    ) -> list[KernelCandidate]:
        region_id = region_identifier(region)
        preferred = [d for d in ("bfloat16", "float16", "float32") if d in device.supported_dtypes]
        return [
            KernelCandidate(
                region_id=region_id,
                device=device.id.name,
                backend_id=self.backend_id,
                kernel_id=f"cuda_inductor_{dtype}",
                dtype=dtype,
                attributes={"impl": "torch_inductor_or_eager"},
            )
            for dtype in preferred
        ]

    def benchmark(self, candidate: KernelCandidate) -> BenchmarkResult:
        if not self.available():
            return BenchmarkResult(
                candidate=candidate,
                latency_s=float("inf"),
                memory_bytes=0,
                measured=False,
                notes="cuda unavailable on this machine",
            )
        import torch

        index = int(candidate.device.rsplit("_", 1)[-1])
        device = torch.device(f"cuda:{index}")
        dtype = getattr(torch, candidate.dtype, torch.float16)
        n = 1024
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)
        torch.cuda.synchronize(device)
        for _ in range(5):
            torch.mm(a, b)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        iters = 20
        for _ in range(iters):
            torch.mm(a, b)
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) / iters
        return BenchmarkResult(
            candidate=candidate,
            latency_s=elapsed,
            memory_bytes=2 * n * n * a.element_size(),
            measured=True,
            notes=f"cuda matmul {n}x{n} {candidate.dtype}",
        )

    def compile(self, region: RegionSource, candidate: KernelCandidate) -> CompiledRegion:
        if not self.available():
            raise BackendError(
                "cuda backend is not available on this machine; StreamCompiler will not fabricate a compiled region"
            )
        return compile_region_for_torch_device(
            region,
            candidate,
            backend_id=self.backend_id,
            torch_device=f"cuda:{_device_index(candidate.device)}",
        )

    def execute(self, executable: CompiledRegion, inputs: Sequence[Any]) -> tuple[Any, ...]:
        if not self.available():
            raise BackendError("cuda backend is not available on this machine")
        return execute_region_on_torch_device(executable, inputs)

    def transfer_capabilities(
        self, source: ComputeResource | str, destination: ComputeResource | str
    ) -> TransferCapability:
        src = source if isinstance(source, str) else source.id.name
        dst = destination if isinstance(destination, str) else destination.id.name
        if not self.available():
            return TransferCapability(src, dst, kind="unsupported", notes="cuda unavailable")
        if src.startswith("cuda_vram_") and dst.startswith("cuda_vram_"):
            import torch

            i = int(src.rsplit("_", 1)[-1])
            j = int(dst.rsplit("_", 1)[-1])
            try:
                if torch.cuda.can_device_access_peer(i, j):
                    return TransferCapability(src, dst, kind="p2p", notes="cuda peer access")
            except Exception:  # noqa: BLE001
                pass
            return TransferCapability(src, dst, kind="host_staged", notes="no peer access; host staging required")
        return TransferCapability(src, dst, kind="dma", notes="cuda memcpy path")
