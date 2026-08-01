"""Backend-neutral profiling interface.

CPU and CUDA profilers report measured timings when the runtime is available.
Virtual-accelerator profilers are always labelled simulated.
"""

from __future__ import annotations

import statistics
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

_MAX_TRANSFER_PROFILE_BYTES = 64 << 20


def _bounded_transfer_size(nbytes: int) -> tuple[int, float]:
    requested = max(0, int(nbytes))
    measured = max(1, min(requested or 1, _MAX_TRANSFER_PROFILE_BYTES))
    return measured, (float(requested) / float(measured) if requested > measured else 1.0)


@dataclass(frozen=True)
class ProfileRecord:
    device_fingerprint: str
    region_graph_hash: str
    shape: tuple[tuple[int, ...], ...]
    dtype: tuple[str, ...]
    layout: str
    thread_configuration: str
    backend_implementation: str
    warm_up_count: int
    sample_count: int
    median_s: float
    dispersion_s: float
    workspace_memory_bytes: int
    measured: bool
    simulated: bool
    kind: str
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "device_fingerprint": self.device_fingerprint,
            "region_graph_hash": self.region_graph_hash,
            "shape": [list(s) for s in self.shape],
            "dtype": list(self.dtype),
            "layout": self.layout,
            "thread_configuration": self.thread_configuration,
            "backend_implementation": self.backend_implementation,
            "warm_up_count": self.warm_up_count,
            "sample_count": self.sample_count,
            "median_s": self.median_s,
            "dispersion_s": self.dispersion_s,
            "workspace_memory_bytes": self.workspace_memory_bytes,
            "measured": self.measured,
            "simulated": self.simulated,
            "kind": self.kind,
            "notes": list(self.notes),
        }


class BackendProfiler(ABC):
    """Profile regions, transfers, overlap, and memory behavior for one backend."""

    backend_id: str

    @abstractmethod
    def profile_region(
        self,
        fn: Callable[..., Any],
        inputs: tuple[Any, ...],
        *,
        device_fingerprint: str,
        region_graph_hash: str,
        warm_up: int = 2,
        samples: int = 5,
    ) -> ProfileRecord: ...

    @abstractmethod
    def profile_transfer(
        self,
        nbytes: int,
        *,
        source: str,
        destination: str,
        device_fingerprint: str,
        warm_up: int = 1,
        samples: int = 5,
        transfer_fn: Callable[[], None] | None = None,
    ) -> ProfileRecord: ...

    @abstractmethod
    def profile_overlap(
        self,
        compute_fn: Callable[[], None],
        transfer_fn: Callable[[], None],
        *,
        device_fingerprint: str,
        warm_up: int = 1,
        samples: int = 3,
    ) -> ProfileRecord: ...

    @abstractmethod
    def profile_memory_behavior(
        self,
        alloc_fn: Callable[[], Any],
        free_fn: Callable[[Any], None],
        *,
        device_fingerprint: str,
        nbytes: int,
        samples: int = 3,
    ) -> ProfileRecord: ...


def _shapes_dtypes(inputs: tuple[Any, ...]) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    shapes: list[tuple[int, ...]] = []
    dtypes: list[str] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            shapes.append(tuple(value.shape))
            dtypes.append(str(value.dtype).replace("torch.", ""))
        else:
            shapes.append(())
            dtypes.append(type(value).__name__)
    return tuple(shapes), tuple(dtypes)


def _summarize(samples: list[float], *, warm_up: int, measured: bool, simulated: bool, **kwargs: Any) -> ProfileRecord:
    ordered = sorted(samples) if samples else [0.0]
    median = statistics.median(ordered)
    dispersion = statistics.pstdev(ordered) if len(ordered) >= 2 else 0.0
    return ProfileRecord(
        device_fingerprint=str(kwargs.get("device_fingerprint", "")),
        region_graph_hash=str(kwargs.get("region_graph_hash", "")),
        shape=tuple(kwargs.get("shape", ())),
        dtype=tuple(kwargs.get("dtype", ())),
        layout=str(kwargs.get("layout", "contiguous")),
        thread_configuration=str(kwargs.get("thread_configuration", f"intraop={torch.get_num_threads()}")),
        backend_implementation=str(kwargs.get("backend_implementation", "")),
        warm_up_count=warm_up,
        sample_count=len(samples),
        median_s=float(median),
        dispersion_s=float(dispersion),
        workspace_memory_bytes=int(kwargs.get("workspace_memory_bytes", 0) or 0),
        measured=measured,
        simulated=simulated,
        kind=str(kwargs.get("kind", "region")),
        notes=tuple(kwargs.get("notes", ()) or ()),
    )


class CpuBackendProfiler(BackendProfiler):
    backend_id = "cpu"

    def profile_region(
        self,
        fn: Callable[..., Any],
        inputs: tuple[Any, ...],
        *,
        device_fingerprint: str,
        region_graph_hash: str,
        warm_up: int = 2,
        samples: int = 5,
    ) -> ProfileRecord:
        shapes, dtypes = _shapes_dtypes(inputs)
        for _ in range(max(0, warm_up)):
            fn(*inputs)
        timings: list[float] = []
        for _ in range(max(1, samples)):
            t0 = time.perf_counter()
            fn(*inputs)
            timings.append(time.perf_counter() - t0)
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash=region_graph_hash,
            shape=shapes,
            dtype=dtypes,
            backend_implementation="cpu",
            kind="region",
        )

    def profile_transfer(
        self,
        nbytes: int,
        *,
        source: str,
        destination: str,
        device_fingerprint: str,
        warm_up: int = 1,
        samples: int = 5,
        transfer_fn: Callable[[], None] | None = None,
    ) -> ProfileRecord:
        measured_nbytes, scale = _bounded_transfer_size(nbytes)
        payload = torch.empty(measured_nbytes, dtype=torch.uint8)

        def _default() -> None:
            _ = payload.clone()

        fn = transfer_fn or _default
        for _ in range(max(0, warm_up)):
            fn()
        timings = []
        for _ in range(max(1, samples)):
            t0 = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - t0) * scale)
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash=f"transfer:{source}->{destination}:{nbytes}",
            shape=((nbytes,),),
            dtype=("uint8",),
            backend_implementation="cpu_memcpy",
            kind="transfer",
            notes=(
                f"source={source}",
                f"destination={destination}",
                f"measured_bytes={measured_nbytes}",
                f"requested_bytes={max(0, int(nbytes))}",
            ),
        )

    def profile_overlap(
        self,
        compute_fn: Callable[[], None],
        transfer_fn: Callable[[], None],
        *,
        device_fingerprint: str,
        warm_up: int = 1,
        samples: int = 3,
    ) -> ProfileRecord:
        from concurrent.futures import ThreadPoolExecutor

        for _ in range(max(0, warm_up)):
            compute_fn()
            transfer_fn()
        timings = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            for _ in range(max(1, samples)):
                t0 = time.perf_counter()
                f1 = pool.submit(compute_fn)
                f2 = pool.submit(transfer_fn)
                f1.result()
                f2.result()
                timings.append(time.perf_counter() - t0)
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash="overlap:cpu",
            backend_implementation="cpu",
            kind="overlap",
        )

    def profile_memory_behavior(
        self,
        alloc_fn: Callable[[], Any],
        free_fn: Callable[[Any], None],
        *,
        device_fingerprint: str,
        nbytes: int,
        samples: int = 3,
    ) -> ProfileRecord:
        timings = []
        for _ in range(max(1, samples)):
            t0 = time.perf_counter()
            handle = alloc_fn()
            free_fn(handle)
            timings.append(time.perf_counter() - t0)
        return _summarize(
            timings,
            warm_up=0,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash=f"memory:{nbytes}",
            backend_implementation="cpu",
            kind="memory",
            workspace_memory_bytes=nbytes,
        )


class VirtualAccelBackendProfiler(BackendProfiler):
    """Deterministic virtual-accelerator profiler — always labelled simulated."""

    backend_id = "mock_accel"

    def __init__(self, *, compute_delay_s: float = 0.05, transfer_delay_s: float = 0.08) -> None:
        self.compute_delay_s = float(compute_delay_s)
        self.transfer_delay_s = float(transfer_delay_s)

    def profile_region(
        self,
        fn: Callable[..., Any],
        inputs: tuple[Any, ...],
        *,
        device_fingerprint: str,
        region_graph_hash: str,
        warm_up: int = 2,
        samples: int = 5,
    ) -> ProfileRecord:
        shapes, dtypes = _shapes_dtypes(inputs)
        # Execute once for correctness; report configured virtual delay as the cost.
        if warm_up > 0:
            fn(*inputs)
        timings = [self.compute_delay_s for _ in range(max(1, samples))]
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=False,
            simulated=True,
            device_fingerprint=device_fingerprint,
            region_graph_hash=region_graph_hash,
            shape=shapes,
            dtype=dtypes,
            backend_implementation="mock_accel",
            kind="region",
            notes=("simulated virtual accelerator",),
        )

    def profile_transfer(
        self,
        nbytes: int,
        *,
        source: str,
        destination: str,
        device_fingerprint: str,
        warm_up: int = 1,
        samples: int = 5,
        transfer_fn: Callable[[], None] | None = None,
    ) -> ProfileRecord:
        timings = [self.transfer_delay_s for _ in range(max(1, samples))]
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=False,
            simulated=True,
            device_fingerprint=device_fingerprint,
            region_graph_hash=f"transfer:{source}->{destination}:{nbytes}",
            shape=((nbytes,),),
            dtype=("uint8",),
            backend_implementation="mock_accel_copy_engine",
            kind="transfer",
            notes=("simulated virtual accelerator", f"source={source}", f"destination={destination}"),
        )

    def profile_overlap(
        self,
        compute_fn: Callable[[], None],
        transfer_fn: Callable[[], None],
        *,
        device_fingerprint: str,
        warm_up: int = 1,
        samples: int = 3,
    ) -> ProfileRecord:
        # Independent virtual streams: overlap cost is max(compute, transfer).
        cost = max(self.compute_delay_s, self.transfer_delay_s)
        timings = [cost for _ in range(max(1, samples))]
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=False,
            simulated=True,
            device_fingerprint=device_fingerprint,
            region_graph_hash="overlap:mock_accel",
            backend_implementation="mock_accel",
            kind="overlap",
            notes=("simulated virtual accelerator",),
        )

    def profile_memory_behavior(
        self,
        alloc_fn: Callable[[], Any],
        free_fn: Callable[[Any], None],
        *,
        device_fingerprint: str,
        nbytes: int,
        samples: int = 3,
    ) -> ProfileRecord:
        timings = [1e-6 for _ in range(max(1, samples))]
        return _summarize(
            timings,
            warm_up=0,
            measured=False,
            simulated=True,
            device_fingerprint=device_fingerprint,
            region_graph_hash=f"memory:{nbytes}",
            backend_implementation="mock_accel",
            kind="memory",
            workspace_memory_bytes=nbytes,
            notes=("simulated virtual accelerator",),
        )


class CudaBackendProfiler(BackendProfiler):
    """Measured CUDA or ROCm profiler through PyTorch's ``torch.cuda`` API."""

    backend_id = "cuda"

    def __init__(self, device_index: int = 0, *, backend_id: str = "cuda") -> None:
        if backend_id not in {"cuda", "rocm"}:
            raise ValueError(f"CudaBackendProfiler backend_id must be 'cuda' or 'rocm', got {backend_id!r}")
        self.device_index = int(device_index)
        self.backend_id = backend_id

    def _ensure_available(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(f"{self.backend_id.upper()} profiler requires torch.cuda.is_available()")
        if self.backend_id == "rocm" and not getattr(torch.version, "hip", None):
            raise RuntimeError("ROCm profiler requires a HIP-enabled PyTorch build")
        if self.backend_id == "cuda" and (
            not getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None)
        ):
            raise RuntimeError("CUDA profiler requires a CUDA-enabled non-ROCm PyTorch build")

    def _is_device_resource(self, value: str) -> bool:
        lowered = value.lower()
        return self.backend_id in lowered or "cuda" in lowered

    def _device(self) -> torch.device:
        return torch.device(f"cuda:{self.device_index}")

    def _sync(self) -> None:
        torch.cuda.synchronize(self._device())

    def _to_device(self, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        device = self._device()
        placed: list[Any] = []
        for value in inputs:
            if isinstance(value, torch.Tensor):
                placed.append(value.to(device, non_blocking=False))
            else:
                placed.append(value)
        return tuple(placed)

    def profile_region(
        self,
        fn: Callable[..., Any],
        inputs: tuple[Any, ...],
        *,
        device_fingerprint: str,
        region_graph_hash: str,
        warm_up: int = 2,
        samples: int = 5,
    ) -> ProfileRecord:
        self._ensure_available()
        device = self._device()
        # Prefer an explicit backend_gpu_N fingerprint when the planner supplies one.
        for token in str(device_fingerprint).split():
            if token.startswith(f"{self.backend_id}_gpu_"):
                try:
                    self.device_index = int(token.rsplit("_", 1)[-1])
                    device = self._device()
                except ValueError:
                    pass
                break
        module = fn
        moved_module = False
        if isinstance(module, torch.nn.Module):
            module = module.to(device)
            moved_module = True
            call: Callable[..., Any] = module
        else:
            call = fn
        placed = self._to_device(inputs)
        shapes, dtypes = _shapes_dtypes(placed)
        try:
            for _ in range(max(0, warm_up)):
                call(*placed)
            self._sync()
            timings: list[float] = []
            for _ in range(max(1, samples)):
                self._sync()
                t0 = time.perf_counter()
                call(*placed)
                self._sync()
                timings.append(time.perf_counter() - t0)
        finally:
            if moved_module and isinstance(module, torch.nn.Module):
                module.to("cpu")
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash=region_graph_hash,
            shape=shapes,
            dtype=dtypes,
            backend_implementation=self.backend_id,
            kind="region",
            thread_configuration=f"{self.backend_id}:{self.device_index}",
            notes=(f"{self.backend_id}:{self.device_index}",),
        )

    def profile_transfer(
        self,
        nbytes: int,
        *,
        source: str,
        destination: str,
        device_fingerprint: str,
        warm_up: int = 1,
        samples: int = 5,
        transfer_fn: Callable[[], None] | None = None,
    ) -> ProfileRecord:
        self._ensure_available()
        device = self._device()
        measured_nbytes, scale = _bounded_transfer_size(nbytes)
        host = torch.empty(measured_nbytes, dtype=torch.uint8, pin_memory=True)
        device_buf = torch.empty_like(host, device=device)

        def _h2d() -> None:
            device_buf.copy_(host, non_blocking=False)
            self._sync()

        def _d2h() -> None:
            host.copy_(device_buf, non_blocking=False)
            self._sync()

        peer = torch.empty_like(device_buf)

        def _d2d() -> None:
            peer.copy_(device_buf, non_blocking=False)
            self._sync()

        if transfer_fn is not None:
            fn = transfer_fn
        elif self._is_device_resource(destination) and not self._is_device_resource(source):
            fn = _h2d
        elif self._is_device_resource(source) and not self._is_device_resource(destination):
            fn = _d2h
        elif self._is_device_resource(source) and self._is_device_resource(destination):
            fn = _d2d
        else:
            fn = _h2d
        for _ in range(max(0, warm_up)):
            fn()
        timings: list[float] = []
        for _ in range(max(1, samples)):
            t0 = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - t0) * scale)
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash=f"transfer:{source}->{destination}:{nbytes}",
            shape=((nbytes,),),
            dtype=("uint8",),
            backend_implementation=f"{self.backend_id}_memcpy",
            kind="transfer",
            notes=(
                f"source={source}",
                f"destination={destination}",
                f"measured_bytes={measured_nbytes}",
                f"requested_bytes={max(0, int(nbytes))}",
            ),
        )

    def profile_overlap(
        self,
        compute_fn: Callable[[], None],
        transfer_fn: Callable[[], None],
        *,
        device_fingerprint: str,
        warm_up: int = 1,
        samples: int = 3,
    ) -> ProfileRecord:
        self._ensure_available()
        for _ in range(max(0, warm_up)):
            compute_fn()
            transfer_fn()
            self._sync()
        timings: list[float] = []
        for _ in range(max(1, samples)):
            self._sync()
            t0 = time.perf_counter()
            compute_fn()
            transfer_fn()
            self._sync()
            timings.append(time.perf_counter() - t0)
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash=f"overlap:{self.backend_id}",
            backend_implementation=self.backend_id,
            kind="overlap",
        )

    def profile_memory_behavior(
        self,
        alloc_fn: Callable[[], Any],
        free_fn: Callable[[Any], None],
        *,
        device_fingerprint: str,
        nbytes: int,
        samples: int = 3,
    ) -> ProfileRecord:
        self._ensure_available()
        timings: list[float] = []
        for _ in range(max(1, samples)):
            self._sync()
            t0 = time.perf_counter()
            handle = alloc_fn()
            self._sync()
            free_fn(handle)
            self._sync()
            timings.append(time.perf_counter() - t0)
        return _summarize(
            timings,
            warm_up=0,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash=f"memory:{nbytes}",
            backend_implementation=self.backend_id,
            kind="memory",
            workspace_memory_bytes=nbytes,
        )


class XpuBackendProfiler(BackendProfiler):
    """Measured Intel XPU profiler through PyTorch's ``torch.xpu`` API."""

    backend_id = "xpu"

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = int(device_index)

    def _module(self) -> Any:
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not callable(getattr(xpu, "is_available", None)) or not xpu.is_available():
            raise RuntimeError("XPU profiler requires torch.xpu.is_available()")
        return xpu

    def _device(self) -> torch.device:
        return torch.device(f"xpu:{self.device_index}")

    def _sync(self) -> None:
        xpu = self._module()
        synchronize = getattr(xpu, "synchronize", None)
        if callable(synchronize):
            try:
                synchronize(self.device_index)
            except TypeError:
                synchronize()

    def _to_device(self, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        device = self._device()
        return tuple(
            value.to(device, non_blocking=False) if isinstance(value, torch.Tensor) else value for value in inputs
        )

    def profile_region(
        self,
        fn: Callable[..., Any],
        inputs: tuple[Any, ...],
        *,
        device_fingerprint: str,
        region_graph_hash: str,
        warm_up: int = 2,
        samples: int = 5,
    ) -> ProfileRecord:
        self._module()
        device = self._device()
        for token in str(device_fingerprint).split():
            if token.startswith("xpu_gpu_"):
                tail = token.rsplit("_", 1)[-1]
                if tail.isdigit():
                    self.device_index = int(tail)
                    device = self._device()
                break
        module = fn
        moved_module = False
        if isinstance(module, torch.nn.Module):
            module = module.to(device)
            moved_module = True
            call: Callable[..., Any] = module
        else:
            call = fn
        placed = self._to_device(inputs)
        shapes, dtypes = _shapes_dtypes(placed)
        try:
            for _ in range(max(0, warm_up)):
                call(*placed)
            self._sync()
            timings: list[float] = []
            for _ in range(max(1, samples)):
                self._sync()
                started = time.perf_counter()
                call(*placed)
                self._sync()
                timings.append(time.perf_counter() - started)
        finally:
            if moved_module and isinstance(module, torch.nn.Module):
                module.to("cpu")
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash=region_graph_hash,
            shape=shapes,
            dtype=dtypes,
            backend_implementation="xpu",
            kind="region",
            thread_configuration=f"xpu:{self.device_index}",
            notes=(f"xpu:{self.device_index}",),
        )

    def profile_transfer(
        self,
        nbytes: int,
        *,
        source: str,
        destination: str,
        device_fingerprint: str,
        warm_up: int = 1,
        samples: int = 5,
        transfer_fn: Callable[[], None] | None = None,
    ) -> ProfileRecord:
        self._module()
        device = self._device()
        measured_nbytes, scale = _bounded_transfer_size(nbytes)
        host = torch.empty(measured_nbytes, dtype=torch.uint8)
        device_buf = torch.empty_like(host, device=device)

        def h2d() -> None:
            device_buf.copy_(host, non_blocking=False)
            self._sync()

        def d2h() -> None:
            host.copy_(device_buf, non_blocking=False)
            self._sync()

        peer = torch.empty_like(device_buf)

        def d2d() -> None:
            peer.copy_(device_buf, non_blocking=False)
            self._sync()

        if transfer_fn is not None:
            callback = transfer_fn
        elif "xpu" in destination.lower() and "xpu" not in source.lower():
            callback = h2d
        elif "xpu" in source.lower() and "xpu" not in destination.lower():
            callback = d2h
        elif "xpu" in source.lower() and "xpu" in destination.lower():
            callback = d2d
        else:
            callback = h2d
        for _ in range(max(0, warm_up)):
            callback()
        timings: list[float] = []
        for _ in range(max(1, samples)):
            started = time.perf_counter()
            callback()
            timings.append((time.perf_counter() - started) * scale)
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash=f"transfer:{source}->{destination}:{nbytes}",
            shape=((nbytes,),),
            dtype=("uint8",),
            backend_implementation="xpu_memcpy",
            kind="transfer",
            notes=(
                f"source={source}",
                f"destination={destination}",
                f"measured_bytes={measured_nbytes}",
                f"requested_bytes={max(0, int(nbytes))}",
            ),
        )

    def profile_overlap(
        self,
        compute_fn: Callable[[], None],
        transfer_fn: Callable[[], None],
        *,
        device_fingerprint: str,
        warm_up: int = 1,
        samples: int = 3,
    ) -> ProfileRecord:
        self._module()
        for _ in range(max(0, warm_up)):
            compute_fn()
            transfer_fn()
            self._sync()
        timings: list[float] = []
        for _ in range(max(1, samples)):
            self._sync()
            started = time.perf_counter()
            compute_fn()
            transfer_fn()
            self._sync()
            timings.append(time.perf_counter() - started)
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash="overlap:xpu",
            backend_implementation="xpu",
            kind="overlap",
        )

    def profile_memory_behavior(
        self,
        alloc_fn: Callable[[], Any],
        free_fn: Callable[[Any], None],
        *,
        device_fingerprint: str,
        nbytes: int,
        samples: int = 3,
    ) -> ProfileRecord:
        self._module()
        timings: list[float] = []
        for _ in range(max(1, samples)):
            self._sync()
            started = time.perf_counter()
            handle = alloc_fn()
            self._sync()
            free_fn(handle)
            self._sync()
            timings.append(time.perf_counter() - started)
        return _summarize(
            timings,
            warm_up=0,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash=f"memory:{nbytes}",
            backend_implementation="xpu",
            kind="memory",
            workspace_memory_bytes=nbytes,
        )


def profiler_for_backend(backend_id: str, **kwargs: Any) -> BackendProfiler:
    if backend_id in {"cpu", "cpu_numa"}:
        return CpuBackendProfiler()
    if backend_id == "cuda":
        return CudaBackendProfiler(**kwargs, backend_id="cuda")
    if backend_id == "rocm":
        return CudaBackendProfiler(**kwargs, backend_id="rocm")
    if backend_id == "xpu":
        return XpuBackendProfiler(**kwargs)
    if backend_id in {"mock_accel", "simulated_device"}:
        return VirtualAccelBackendProfiler(**kwargs)
    raise NotImplementedError(
        f"BackendProfiler for {backend_id!r} is not implemented; "
        f"real accelerator profiling requires a validated backend profiler"
    )
