"""Intel XPU execution backend through PyTorch's ``torch.xpu`` API.

The backend is capability-gated: importing TensorTorrent on a host without an
Intel accelerator remains harmless, and no capability is reported unless the
installed PyTorch build exposes and initializes ``torch.xpu`` successfully.
"""

from __future__ import annotations

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
from tensortorrent.closed import BudgetSourceKind, TransferKind
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


def _device_index(resource_id: str) -> int:
    tail = resource_id.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _xpu_module() -> Any | None:
    try:
        import torch

        xpu = getattr(torch, "xpu", None)
        if xpu is None or not callable(getattr(xpu, "is_available", None)):
            return None
        return xpu
    except Exception:  # noqa: BLE001 - optional backend discovery must be isolated
        return None


def _xpu_free_memory(index: int) -> int | None:
    """Return free XPU memory bytes via torch.xpu.mem_get_info when available."""
    xpu = _xpu_module()
    if xpu is None:
        return None
    mem_get_info = getattr(xpu, "mem_get_info", None)
    if not callable(mem_get_info):
        return None
    try:
        result = mem_get_info(index)
        if result and len(result) >= 1:
            return int(result[0])
    except Exception:  # noqa: BLE001
        pass
    return None


def _xpu_is_integrated(name: str, props: Any) -> tuple[bool, str]:
    """Heuristic classification of XPU as integrated or discrete.

    Returns (is_integrated, reason_string).
    """
    name_lower = name.lower()
    # Name-based heuristic: Intel integrated GPU families
    if any(kw in name_lower for kw in ("uhd", "iris", "integrated")):
        return True, "name-heuristic"
    # Strict: no FP64 support AND name lacks known discrete Arc/Max/Flex product lines
    has_fp64 = getattr(props, "has_fp64", None)
    if has_fp64 is False:
        discrete_keywords = ("arc", "max", "flex")
        if not any(kw in name_lower for kw in discrete_keywords):
            return True, "name-heuristic"
    return False, ""


class XpuBackend(ExecutionBackend):
    """Intel GPU/XPU backend backed by PyTorch.

    Real execution is only claimed when ``torch.xpu.is_available()`` succeeds.
    Discovery deliberately uses feature detection because PyTorch XPU builds
    expose slightly different property objects across releases.
    """

    backend_id = "xpu"

    def available(self) -> bool:
        xpu = _xpu_module()
        if xpu is None:
            return False
        try:
            return bool(xpu.is_available())
        except Exception:  # noqa: BLE001
            return False

    def _device_count(self) -> int:
        xpu = _xpu_module()
        if xpu is None:
            return 0
        fn = getattr(xpu, "device_count", None)
        if not callable(fn):
            return 1 if self.available() else 0
        try:
            return max(0, int(fn()))
        except Exception:  # noqa: BLE001
            return 0

    def _properties(self, index: int) -> Any | None:
        xpu = _xpu_module()
        if xpu is None:
            return None
        getter = getattr(xpu, "get_device_properties", None)
        if callable(getter):
            try:
                return getter(index)
            except Exception:  # noqa: BLE001
                return None
        return None

    def _name(self, index: int, props: Any | None) -> str:
        name = getattr(props, "name", None)
        if name:
            return str(name)
        xpu = _xpu_module()
        getter = getattr(xpu, "get_device_name", None) if xpu is not None else None
        if callable(getter):
            try:
                return str(getter(index))
            except Exception:  # noqa: BLE001
                pass
        return f"Intel XPU {index}"

    def _total_memory(self, index: int, props: Any | None) -> int:
        for attr in ("total_memory", "total_mem", "global_mem_size"):
            value = getattr(props, attr, None)
            if value is not None:
                try:
                    return max(0, int(value))
                except (TypeError, ValueError):
                    pass
        xpu = _xpu_module()
        memory_fn = getattr(xpu, "get_device_properties", None) if xpu is not None else None
        if callable(memory_fn):
            try:
                raw = getattr(memory_fn(index), "total_memory", 0)
                return max(0, int(raw or 0))
            except Exception:  # noqa: BLE001
                pass
        # Unknown capacity must not masquerade as a large valid device.
        return 0

    def _probe_dtypes(self, index: int) -> tuple[str, ...]:
        import torch

        found: list[str] = []
        for name in ("float32", "float16", "bfloat16", "int8", "int32", "bool"):
            dtype = getattr(torch, name, None)
            if dtype is None:
                continue
            try:
                torch.empty(1, device=f"xpu:{index}", dtype=dtype)
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
            "aten::add",
            "aten::mul",
        )

    def discover_devices(self) -> ResourceGraph:
        graph = ResourceGraph(fingerprint="", backends_present=())
        if not self.available():
            graph.attributes["xpu_status"] = "runtime_unavailable"
            return graph

        graph.backends_present = (self.backend_id,)
        for index in range(self._device_count()):
            props = self._properties(index)
            compute_name = f"xpu_gpu_{index}"
            memory_name = f"xpu_vram_{index}"
            total = self._total_memory(index, props)
            device_name = self._name(index, props)

            # Live free memory via torch.xpu.mem_get_info when available; else total-fallback
            free_mem = _xpu_free_memory(index) if total > 0 else None

            # Integrated classification heuristic
            is_integrated, integrated_reason = _xpu_is_integrated(device_name, props)
            compute_class = ComputeClass.INTEGRATED_GPU if is_integrated else ComputeClass.DISCRETE_GPU

            # Budget resolution
            display_active = True  # conservative for XPU (display may be attached)
            headroom = _budget.default_vram_headroom_bytes(display_active)
            if total > 0:
                dev_budget = _budget.resolve_device_memory_budget(
                    total_bytes=total,
                    free_bytes=free_mem,
                    explicit=None,
                    headroom_bytes=headroom,
                )
                allocatable = dev_budget.allowed_bytes
                budget_source = dev_budget.source.kind
                budget_detail = dev_budget.source.detail
                budget_reserved = str(dev_budget.reserved_bytes)
            else:
                # A zero capacity means the runtime could not query memory safely;
                # retain the device for diagnostics but make it infeasible to plan.
                allocatable = 0
                budget_source = BudgetSourceKind.TOTAL_FALLBACK
                budget_detail = "total=0; capacity unknown"
                budget_reserved = "0"

            mem_attrs: dict[str, Any] = {
                "backend": self.backend_id,
                "index": index,
                "capacity_known": total > 0,
                "budget_source": budget_source,
                "budget_detail": budget_detail,
                "budget_reserved_bytes": budget_reserved,
            }
            graph.add_memory(
                MemoryResource(
                    id=ResourceId(ResourceKind.MEMORY, memory_name),
                    memory_class=MemoryClass.DEVICE_VRAM,
                    capacity_bytes=total,
                    allocatable_bytes=allocatable,
                    attached_compute=(compute_name,),
                    attributes=mem_attrs,
                )
            )
            compute_attrs: dict[str, Any] = {
                "index": index,
                "budget_source": budget_source,
                "budget_detail": budget_detail,
            }
            if is_integrated:
                compute_attrs["classified_integrated"] = integrated_reason
            graph.add_compute(
                ComputeResource(
                    id=ResourceId(ResourceKind.COMPUTE, compute_name),
                    compute_class=compute_class,
                    backend_id=self.backend_id,
                    model=device_name,
                    architecture=str(getattr(props, "architecture", "xpu")),
                    vendor="intel",
                    supported_dtypes=self._probe_dtypes(index),
                    supported_ops=self._supported_ops(),
                    copy_engines=max(1, int(getattr(props, "copy_engines", 1) or 1)),
                    concurrency_limit=max(1, int(getattr(props, "max_compute_units", 1) or 1)),
                    memory_affinity=(memory_name,),
                    attributes=compute_attrs,
                )
            )
            graph.add_link(
                TransferLink(
                    id=ResourceId(ResourceKind.LINK, f"{compute_name}->{memory_name}"),
                    link_class=LinkClass.SHARED_MEMORY,
                    source=compute_name,
                    destination=memory_name,
                    bidirectional=True,
                    peer_to_peer=True,
                    measured=False,
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
        preferred = [dtype for dtype in ("bfloat16", "float16", "float32") if dtype in device.supported_dtypes]
        return [
            KernelCandidate(
                region_id=region_id,
                device=device.id.name,
                backend_id=self.backend_id,
                kernel_id=f"xpu_fx_{dtype}",
                dtype=dtype,
                attributes={"impl": "torch_fx_subgraph"},
            )
            for dtype in preferred
        ]

    def benchmark(self, candidate: KernelCandidate) -> BenchmarkResult:
        if not self.available():
            return BenchmarkResult(candidate, float("inf"), 0, False, "xpu unavailable")
        import torch

        device = self.resource_to_torch_device(candidate.device)
        dtype = getattr(torch, candidate.dtype, torch.float32)
        n = 512
        try:
            a = torch.randn(n, n, device=device, dtype=dtype)
            b = torch.randn(n, n, device=device, dtype=dtype)
            self._synchronize(device)
            for _ in range(3):
                torch.mm(a, b)
            self._synchronize(device)
            start = time.perf_counter()
            iterations = 10
            for _ in range(iterations):
                torch.mm(a, b)
            self._synchronize(device)
            latency = (time.perf_counter() - start) / iterations
            return BenchmarkResult(
                candidate,
                latency,
                2 * n * n * a.element_size(),
                True,
                f"xpu matmul {n}x{n} {candidate.dtype}",
            )
        except Exception as exc:  # noqa: BLE001
            return BenchmarkResult(candidate, float("inf"), 0, False, f"xpu benchmark failed: {exc}")

    def benchmark_region(
        self,
        region: RegionSource,
        candidate: KernelCandidate,
        example_inputs: Sequence[Any],
        *,
        iters: int = 5,
    ) -> BenchmarkResult:
        if not self.available():
            return BenchmarkResult(candidate, float("inf"), 0, False, "xpu unavailable")
        import torch

        compiled = self.compile(region, candidate)
        device = self.resource_to_torch_device(candidate.device)
        placed = tuple(value.to(device) if isinstance(value, torch.Tensor) else value for value in example_inputs)
        return benchmark_region_on_torch_device(
            candidate,
            compiled,
            list(placed),
            iters=iters,
            synchronize=lambda: self._synchronize(device),
        )

    @staticmethod
    def _synchronize(device: Any) -> None:
        xpu = _xpu_module()
        sync = getattr(xpu, "synchronize", None) if xpu is not None else None
        if callable(sync):
            try:
                sync(device)
            except TypeError:
                sync()

    def validate_basic_execution(self, device: ComputeResource) -> tuple[bool, str]:
        if not self.available():
            return False, "xpu unavailable"
        try:
            import torch

            torch_device = self.resource_to_torch_device(device.id.name)
            a = torch.randn(32, 32, device=torch_device, dtype=torch.float32)
            b = torch.randn(32, 32, device=torch_device, dtype=torch.float32)
            out = torch.mm(a, b)
            self._synchronize(torch_device)
            if out.shape != (32, 32):
                return False, f"unexpected matmul shape {tuple(out.shape)}"
            return True, f"executed_matmul={torch_device}"
        except Exception as exc:  # noqa: BLE001
            return False, f"validation failed: {exc}"

    def compile(self, region: RegionSource, candidate: KernelCandidate) -> CompiledRegion:
        if not self.available():
            raise BackendError("xpu backend is not available on this machine")
        return compile_region_for_torch_device(
            region,
            candidate,
            backend_id=self.backend_id,
            torch_device=str(self.resource_to_torch_device(candidate.device)),
        )

    def execute(self, executable: CompiledRegion, inputs: Sequence[Any]) -> tuple[Any, ...]:
        if not self.available():
            raise BackendError("xpu backend is not available on this machine")
        return execute_region_on_torch_device(executable, inputs)

    def transfer_capabilities(
        self, source: ComputeResource | str, destination: ComputeResource | str
    ) -> TransferCapability:
        src = source if isinstance(source, str) else source.id.name
        dst = destination if isinstance(destination, str) else destination.id.name
        if not self.available():
            return TransferCapability(src, dst, kind=TransferKind.UNSUPPORTED, notes="xpu unavailable")
        if src.startswith("xpu_vram_") and dst.startswith("xpu_vram_"):
            xpu = _xpu_module()
            peer = getattr(xpu, "can_device_access_peer", None) if xpu is not None else None
            if callable(peer):
                try:
                    if bool(peer(_device_index(src), _device_index(dst))):
                        return TransferCapability(src, dst, kind=TransferKind.P2P, notes="xpu peer access")
                except Exception:  # noqa: BLE001
                    pass
            return TransferCapability(src, dst, kind=TransferKind.HOST_STAGED, notes="xpu peer access not validated")
        return TransferCapability(src, dst, kind=TransferKind.DMA, notes="xpu host/device copy path")

    def resource_to_torch_device(self, resource_id: str) -> Any:
        import torch

        return torch.device(f"xpu:{_device_index(resource_id)}")
