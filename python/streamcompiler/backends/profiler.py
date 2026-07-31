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
        payload = torch.empty(max(1, nbytes // 4), dtype=torch.float32)

        def _default() -> None:
            _ = payload.clone()

        fn = transfer_fn or _default
        for _ in range(max(0, warm_up)):
            fn()
        timings = []
        for _ in range(max(1, samples)):
            t0 = time.perf_counter()
            fn()
            timings.append(time.perf_counter() - t0)
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
            notes=(f"source={source}", f"destination={destination}"),
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
    """Measured CUDA region/transfer profiler (requires torch.cuda)."""

    backend_id = "cuda"

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = int(device_index)

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
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profiler requires torch.cuda.is_available()")
        device = self._device()
        # Prefer an explicit cuda_gpu_N fingerprint when the planner supplies one.
        for token in str(device_fingerprint).split():
            if token.startswith("cuda_gpu_"):
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
            backend_implementation="cuda",
            kind="region",
            thread_configuration=f"cuda:{self.device_index}",
            notes=(f"cuda:{self.device_index}",),
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
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profiler requires torch.cuda.is_available()")
        device = self._device()
        host = torch.empty(max(1, nbytes // 4), dtype=torch.float32, pin_memory=True)
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
        elif "cuda" in destination.lower() and "cuda" not in source.lower():
            fn = _h2d
        elif "cuda" in source.lower() and "cuda" not in destination.lower():
            fn = _d2h
        elif "cuda" in source.lower() and "cuda" in destination.lower():
            fn = _d2d
        else:
            fn = _h2d
        for _ in range(max(0, warm_up)):
            fn()
        timings: list[float] = []
        for _ in range(max(1, samples)):
            t0 = time.perf_counter()
            fn()
            timings.append(time.perf_counter() - t0)
        return _summarize(
            timings,
            warm_up=warm_up,
            measured=True,
            simulated=False,
            device_fingerprint=device_fingerprint,
            region_graph_hash=f"transfer:{source}->{destination}:{nbytes}",
            shape=((nbytes,),),
            dtype=("uint8",),
            backend_implementation="cuda_memcpy",
            kind="transfer",
            notes=(f"source={source}", f"destination={destination}"),
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
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profiler requires torch.cuda.is_available()")
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
            region_graph_hash="overlap:cuda",
            backend_implementation="cuda",
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
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profiler requires torch.cuda.is_available()")
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
            backend_implementation="cuda",
            kind="memory",
            workspace_memory_bytes=nbytes,
        )


def profiler_for_backend(backend_id: str, **kwargs: Any) -> BackendProfiler:
    if backend_id in {"cpu", "cpu_numa"}:
        return CpuBackendProfiler()
    if backend_id == "cuda":
        return CudaBackendProfiler(**kwargs)
    if backend_id in {"mock_accel", "simulated_device"}:
        return VirtualAccelBackendProfiler(**kwargs)
    raise NotImplementedError(
        f"BackendProfiler for {backend_id!r} is not implemented; "
        f"real accelerator profiling requires a validated backend profiler"
    )
