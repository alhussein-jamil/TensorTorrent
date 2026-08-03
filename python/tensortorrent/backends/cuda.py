"""CUDA execution backend for NVIDIA GPUs."""

from __future__ import annotations

import subprocess
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
    """Extract the vendor device ordinal from a resource name like ``gpu_cuda_1``."""
    tail = device_name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _safe_run(cmd: list[str], timeout: float = 5.0) -> str | None:
    """Run a subprocess command and return stdout, or None on any error."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def _probe_display_active(index: int) -> bool:
    """Return True if the GPU at *index* has an active display via nvidia-smi.

    Query: nvidia-smi --query-gpu=index,display_active --format=csv,noheader
    Falls back to True on any failure.
    """
    out = _safe_run(
        [
            "nvidia-smi",
            "--query-gpu=index,display_active",
            "--format=csv,noheader",
        ],
        timeout=5.0,
    )
    if out is None:
        return True  # conservative default
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            if int(parts[0]) == index:
                val = parts[1].lower()
                if val == "enabled":
                    return True
                return val != "disabled"
        except (ValueError, IndexError):
            continue
    return True  # not found → conservative


class CudaBackend(ExecutionBackend):
    backend_id = "cuda"

    def available(self) -> bool:
        try:
            import torch

            return bool(
                torch.cuda.is_available()
                and getattr(torch.version, "cuda", None)
                and not getattr(torch.version, "hip", None)
            )
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

            # Live free memory query (guarded)
            free_mem: int | None = None
            if hasattr(torch.cuda, "mem_get_info"):
                try:
                    free_bytes, _total = torch.cuda.mem_get_info(index)
                    free_mem = int(free_bytes)
                except Exception:  # noqa: BLE001
                    free_mem = None

            # Display activity probe via nvidia-smi
            display_active = _probe_display_active(index)

            # Headroom and allocatable via budget resolver
            headroom = _budget.default_vram_headroom_bytes(display_active)
            dev_budget = _budget.resolve_device_memory_budget(
                total_bytes=total_mem,
                free_bytes=free_mem,
                explicit=None,
                headroom_bytes=headroom,
            )
            # Shield the planner from transient caching-allocator readings.
            capacity_floor = _budget.vram_capacity_floor_bytes(total_mem, headroom)
            allocatable_bytes = max(dev_budget.allowed_bytes, capacity_floor)
            floor_applied = allocatable_bytes > dev_budget.allowed_bytes

            # Jetson / integrated GPU classification
            is_integrated = getattr(props, "is_integrated", False) or getattr(props, "integrated", False)
            compute_class = ComputeClass.INTEGRATED_GPU if is_integrated else ComputeClass.DISCRETE_GPU

            graph.add_memory(
                MemoryResource(
                    id=ResourceId(ResourceKind.MEMORY, vram_name),
                    memory_class=MemoryClass.DEVICE_VRAM,
                    capacity_bytes=total_mem,
                    allocatable_bytes=allocatable_bytes,
                    attached_compute=(name,),
                    attributes={
                        "backend": self.backend_id,
                        "index": index,
                        "budget_source": "capacity_floor" if floor_applied else dev_budget.source.kind,
                        "budget_detail": (
                            f"capacity_floor={capacity_floor} > live={dev_budget.allowed_bytes} "
                            f"({dev_budget.source.detail})"
                        ) if floor_applied else dev_budget.source.detail,
                        "budget_reserved_bytes": str(
                            max(0, total_mem - allocatable_bytes)
                            if floor_applied
                            else dev_budget.reserved_bytes
                        ),
                    },
                )
            )
            compute_attrs: dict[str, Any] = {
                "index": index,
                "uuid": getattr(props, "uuid", None),
                "display_active": display_active,
            }
            if is_integrated:
                compute_attrs["classified_integrated"] = "device-property"
            graph.add_compute(
                ComputeResource(
                    id=ResourceId(ResourceKind.COMPUTE, name),
                    compute_class=compute_class,
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
                    attributes=compute_attrs,
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
                kernel_id=f"cuda_fx_{dtype}",
                dtype=dtype,
                attributes={"impl": "torch_fx_subgraph"},
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

    def benchmark_region(
        self,
        region: RegionSource,
        candidate: KernelCandidate,
        example_inputs: Sequence[Any],
        *,
        iters: int = 5,
    ) -> BenchmarkResult:
        if not self.available():
            return BenchmarkResult(
                candidate=candidate,
                latency_s=float("inf"),
                memory_bytes=0,
                measured=False,
                notes="cuda unavailable on this machine",
            )
        import torch

        compiled = self.compile(region, candidate)
        index = _device_index(candidate.device)
        device = torch.device(f"cuda:{index}")
        placed = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in example_inputs)

        def _sync() -> None:
            torch.cuda.synchronize(device)

        return benchmark_region_on_torch_device(
            candidate,
            compiled,
            list(placed),
            iters=iters,
            synchronize=_sync,
        )

    def validate_basic_execution(self, device: ComputeResource) -> tuple[bool, str]:
        """Run a tiny synchronized matmul on the target CUDA device."""
        if not self.available():
            return False, "cuda unavailable"
        try:
            import torch

            index = int(device.attributes.get("index", _device_index(device.id.name)))
            torch_device = torch.device(f"cuda:{index}")
            a = torch.randn(64, 64, device=torch_device, dtype=torch.float32)
            b = torch.randn(64, 64, device=torch_device, dtype=torch.float32)
            torch.cuda.synchronize(torch_device)
            out = torch.mm(a, b)
            torch.cuda.synchronize(torch_device)
            if out.shape != (64, 64):
                return False, f"unexpected matmul shape {tuple(out.shape)}"
            ops = self.supported_ops(device)
            dtypes = self.supported_dtypes(device)
            return True, f"ops={len(ops)} dtypes={len(dtypes)} executed_matmul=cuda:{index}"
        except Exception as exc:  # noqa: BLE001
            return False, f"validation failed: {exc}"

    def compile(self, region: RegionSource, candidate: KernelCandidate) -> CompiledRegion:
        if not self.available():
            raise BackendError(
                "cuda backend is not available on this machine; TensorTorrent will not fabricate a compiled region"
            )
        return compile_region_for_torch_device(
            region,
            candidate,
            backend_id=self.backend_id,
            torch_device=str(self.resource_to_torch_device(candidate.device)),
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

    def resource_to_torch_device(self, resource_id: str) -> Any:
        import torch

        return torch.device(f"cuda:{_device_index(resource_id)}")
