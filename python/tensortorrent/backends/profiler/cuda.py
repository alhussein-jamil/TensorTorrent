"""CUDA/ROCm backend profiler."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from tensortorrent.backends.profiler.base import (
    BackendProfiler,
    ProfileRecord,
    _bounded_transfer_size,
    _shapes_dtypes,
    _summarize,
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
