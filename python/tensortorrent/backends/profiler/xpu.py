"""Intel XPU backend profiler."""

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
