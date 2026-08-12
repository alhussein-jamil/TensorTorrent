"""ROCm / HIP execution backend for AMD GPUs exposed through PyTorch."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Sequence
from typing import Any

from tensortorrent.backends.base import (
    BenchmarkResult,
    CompiledRegion,
    ExecutionBackend,
    KernelCandidate,
    RegionSource,
    TransferCapability,
    region_identifier,
)
from tensortorrent.backends.torch_device import (
    benchmark_region_on_torch_device,
    compile_region_for_torch_device,
    execute_region_on_torch_device,
)
from tensortorrent.closed import TransferKind
from tensortorrent.errors import BackendError
from tensortorrent.hardware import budget as _budget
from tensortorrent.ir.graph import HeterogeneousGraph, Instruction
from tensortorrent.ir.resource_graph import (
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
    tail = device_name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


class RocmBackend(ExecutionBackend):
    backend_id = "rocm"

    def available(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available() and getattr(torch.version, "hip", None))
        except Exception:  # noqa: BLE001 - optional backend capability boundary
            return False

    @staticmethod
    def _probe_dtypes(index: int) -> tuple[str, ...]:
        import torch

        found: list[str] = []
        for name in ("float32", "float16", "bfloat16", "int8", "int32", "bool"):
            dtype = getattr(torch, name, None)
            if dtype is None:
                continue
            try:
                torch.empty(1, device=f"cuda:{index}", dtype=dtype)
            except Exception:  # noqa: BLE001
                continue
            found.append(name)
        return tuple(found)

    @staticmethod
    def _supported_ops() -> tuple[str, ...]:
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

    def discover_devices(self) -> ResourceGraph:
        graph = ResourceGraph(fingerprint="", backends_present=())
        if not self.available():
            graph.attributes["rocm_status"] = "runtime_unavailable"
            return graph
        import torch

        graph.backends_present = (self.backend_id,)
        count = int(torch.cuda.device_count())
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            name = f"rocm_gpu_{index}"
            vram = f"rocm_vram_{index}"
            total = int(props.total_memory)
            architecture = str(
                getattr(props, "gcnArchName", None)
                or getattr(props, "gcn_arch_name", None)
                or getattr(props, "architecture", None)
                or "gfx-unknown"
            )

            # Live free memory (mem_get_info works under ROCm via CUDA facade)
            free_mem: int | None = None
            if hasattr(torch.cuda, "mem_get_info"):
                try:
                    free_bytes, _total = torch.cuda.mem_get_info(index)
                    free_mem = int(free_bytes)
                except Exception:  # noqa: BLE001
                    free_mem = None

            # ROCm GPUs in compute clusters typically have no display attached
            display_active = True  # conservative default for AMD
            headroom = _budget.default_vram_headroom_bytes(display_active)
            dev_budget = _budget.resolve_device_memory_budget(
                total_bytes=total,
                free_bytes=free_mem,
                explicit=None,
                headroom_bytes=headroom,
            )

            graph.add_memory(
                MemoryResource(
                    id=ResourceId(ResourceKind.MEMORY, vram),
                    memory_class=MemoryClass.DEVICE_VRAM,
                    capacity_bytes=total,
                    allocatable_bytes=dev_budget.allowed_bytes,
                    attached_compute=(name,),
                    attributes={
                        "backend": self.backend_id,
                        "index": index,
                        "budget_source": dev_budget.source.kind,
                        "budget_detail": dev_budget.source.detail,
                        "budget_reserved_bytes": str(dev_budget.reserved_bytes),
                    },
                )
            )
            graph.add_compute(
                ComputeResource(
                    id=ResourceId(ResourceKind.COMPUTE, name),
                    compute_class=ComputeClass.DISCRETE_GPU,
                    backend_id=self.backend_id,
                    model=str(props.name),
                    architecture=architecture,
                    vendor="amd",
                    supported_dtypes=self._probe_dtypes(index),
                    supported_ops=self._supported_ops(),
                    core_count=int(getattr(props, "multi_processor_count", 0) or 0),
                    copy_engines=max(1, int(getattr(props, "copy_engines", 2) or 2)),
                    concurrency_limit=8,
                    memory_affinity=(vram,),
                    attributes={
                        "index": index,
                        "hip_version": str(getattr(torch.version, "hip", "")),
                        "budget_source": dev_budget.source.kind,
                        "budget_detail": dev_budget.source.detail,
                    },
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
                    measured=False,
                )
            )

        for source_index in range(count):
            for destination_index in range(count):
                if source_index == destination_index:
                    continue
                can_peer = False
                with contextlib.suppress(Exception):
                    can_peer = bool(torch.cuda.can_device_access_peer(source_index, destination_index))
                source = f"rocm_vram_{source_index}"
                destination = f"rocm_vram_{destination_index}"
                graph.add_link(
                    TransferLink(
                        id=ResourceId(ResourceKind.LINK, f"{source}->{destination}"),
                        link_class=LinkClass.INFINITY_FABRIC if can_peer else LinkClass.PCIE,
                        source=source,
                        destination=destination,
                        bidirectional=False,
                        peer_to_peer=can_peer,
                        measured=False,
                        attributes={"access_peer": can_peer},
                    )
                )
        return graph

    def supported_ops(self, device: ComputeResource) -> tuple[str, ...]:
        return device.supported_ops or self._supported_ops()

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
                kernel_id=f"rocm_fx_{dtype}",
                dtype=dtype,
                attributes={"impl": "torch_fx_subgraph"},
            )
            for dtype in ("bfloat16", "float16", "float32")
            if dtype in device.supported_dtypes
        ]

    def benchmark(self, candidate: KernelCandidate) -> BenchmarkResult:
        if not self.available():
            return BenchmarkResult(candidate, float("inf"), 0, False, "rocm unavailable")
        import torch

        device = self.resource_to_torch_device(candidate.device)
        dtype = getattr(torch, candidate.dtype, torch.float32)
        size = 1024
        try:
            left = torch.randn(size, size, device=device, dtype=dtype)
            right = torch.randn(size, size, device=device, dtype=dtype)
            torch.cuda.synchronize(device)
            for _ in range(5):
                torch.mm(left, right)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            iterations = 20
            for _ in range(iterations):
                torch.mm(left, right)
            torch.cuda.synchronize(device)
            return BenchmarkResult(
                candidate,
                (time.perf_counter() - started) / iterations,
                2 * size * size * left.element_size(),
                True,
                f"rocm matmul {size}x{size} {candidate.dtype}",
            )
        except Exception as exc:  # noqa: BLE001
            return BenchmarkResult(candidate, float("inf"), 0, False, f"rocm benchmark failed: {exc}")

    def benchmark_region(
        self,
        region: RegionSource,
        candidate: KernelCandidate,
        example_inputs: Sequence[Any],
        *,
        iters: int = 5,
    ) -> BenchmarkResult:
        if not self.available():
            return BenchmarkResult(candidate, float("inf"), 0, False, "rocm unavailable")
        import torch

        compiled = self.compile(region, candidate)
        device = self.resource_to_torch_device(candidate.device)
        placed = tuple(value.to(device) if isinstance(value, torch.Tensor) else value for value in example_inputs)
        return benchmark_region_on_torch_device(
            candidate,
            compiled,
            list(placed),
            iters=iters,
            synchronize=lambda: torch.cuda.synchronize(device),
        )

    def validate_basic_execution(self, device: ComputeResource) -> tuple[bool, str]:
        if not self.available():
            return False, "rocm unavailable"
        try:
            import torch

            index = int(device.attributes.get("index", _device_index(device.id.name)))
            torch_device = torch.device(f"cuda:{index}")
            left = torch.randn(32, 32, device=torch_device, dtype=torch.float32)
            right = torch.randn(32, 32, device=torch_device, dtype=torch.float32)
            torch.cuda.synchronize(torch_device)
            output = torch.mm(left, right)
            torch.cuda.synchronize(torch_device)
            if output.shape != (32, 32):
                return False, f"unexpected matmul shape {tuple(output.shape)}"
            return True, f"executed_matmul=rocm:{index}"
        except Exception as exc:  # noqa: BLE001
            return False, f"validation failed: {exc}"

    def compile(self, region: RegionSource, candidate: KernelCandidate) -> CompiledRegion:
        if not self.available():
            raise BackendError(
                "rocm backend is not available on this machine; TensorTorrent will not fabricate a compiled region"
            )
        return compile_region_for_torch_device(
            region,
            candidate,
            backend_id=self.backend_id,
            torch_device=str(self.resource_to_torch_device(candidate.device)),
        )

    def execute(self, executable: CompiledRegion, inputs: Sequence[Any]) -> tuple[Any, ...]:
        if not self.available():
            raise BackendError("rocm backend is not available on this machine")
        return execute_region_on_torch_device(executable, inputs)

    def transfer_capabilities(
        self, source: ComputeResource | str, destination: ComputeResource | str
    ) -> TransferCapability:
        src = source if isinstance(source, str) else source.id.name
        dst = destination if isinstance(destination, str) else destination.id.name
        if not self.available():
            return TransferCapability(src, dst, kind=TransferKind.UNSUPPORTED, notes="rocm unavailable")
        if src.startswith("rocm_vram_") and dst.startswith("rocm_vram_"):
            import torch

            source_index = _device_index(src)
            destination_index = _device_index(dst)
            try:
                if bool(torch.cuda.can_device_access_peer(source_index, destination_index)):
                    return TransferCapability(src, dst, kind=TransferKind.P2P, notes="rocm peer access")
            except Exception:  # noqa: BLE001
                pass
            return TransferCapability(src, dst, kind=TransferKind.HOST_STAGED, notes="rocm peer access unavailable")
        return TransferCapability(src, dst, kind=TransferKind.DMA, notes="rocm host/device copy path")

    def resource_to_torch_device(self, resource_id: str) -> Any:
        import torch

        # HIP devices intentionally use torch's cuda device facade on ROCm builds.
        return torch.device(f"cuda:{_device_index(resource_id)}")
