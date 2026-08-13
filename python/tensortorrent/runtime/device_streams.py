"""CUDA copy vs compute streams so H2D can overlap GEMM.

Schedule IR already stamps ``{resource}::copy0`` vs ``{resource}::compute`` and
lets Transfer(i) race Compute(i-1) when ``prefetch_distance > 0``. Those ids were
labels: ``copy_sync`` and region bodies ran on the default stream, so the GPU
serialized the work the DAG permitted.

This module binds one copy stream and one compute stream per CUDA (or HIP)
device. Transfers record a CUDA event; Compute waits on that event *on the
compute stream* (no CPU sync). Release/collect still CPU-synchronize the event
before dropping storage.

Training (autograd) and mock resources stay on the blocking path — stream
capture is unsafe across saved tensors and virtual backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from tensortorrent.runtime.resource_names import is_device_resource


class CudaEventHandle:
    """``ResidentCopy.ready_event`` wrapper around ``torch.cuda.Event``."""

    __slots__ = ("event", "device")

    def __init__(self, event: Any, device: torch.device) -> None:
        self.event = event
        self.device = device

    def wait(self, *, timeout: float | None = None) -> None:
        del timeout
        self.event.synchronize()

    def is_complete(self) -> bool:
        return bool(self.event.query())

    def wait_stream(self, stream: Any) -> None:
        stream.wait_event(self.event)


@dataclass
class _DevicePair:
    index: int
    device: torch.device
    copy_stream: Any
    compute_stream: Any


@dataclass
class DeviceStreamRuntime:
    """Executor-owned CUDA stream pair, reused across forwards.

    ``None`` from :meth:`maybe_create` means the plan has no CUDA/ROCm device
    or CUDA is unavailable — callers keep the existing blocking ``.to``.
    """

    pairs: dict[int, _DevicePair] = field(default_factory=dict)
    _pinned_cache: dict[int, torch.Tensor] = field(default_factory=dict)
    _buffer_pool: dict[tuple[int, tuple[int, ...], torch.dtype], list[torch.Tensor]] = field(default_factory=dict)
    _issued: set[int] = field(default_factory=set)
    _pool_keep: int = 8

    @classmethod
    def maybe_create(cls, bindings: dict[str, Any]) -> DeviceStreamRuntime | None:
        """Build streams for CUDA/ROCm bindings; ``None`` when nothing to overlap."""
        if not torch.cuda.is_available():
            return None
        indices: set[int] = set()
        for binding in bindings.values():
            backend = str(getattr(binding, "backend_id", "") or "")
            resource = str(getattr(binding, "device", "") or "")
            if backend not in {"cuda", "rocm"} and not _resource_is_cudaish(resource):
                continue
            index = _device_index(binding)
            if index is not None:
                indices.add(index)
        if not indices:
            return None
        pairs: dict[int, _DevicePair] = {}
        for index in sorted(indices):
            device = torch.device("cuda", index)
            pairs[index] = _DevicePair(
                index=index,
                device=device,
                copy_stream=torch.cuda.Stream(device=device),  # type: ignore[no-untyped-call]
                compute_stream=torch.cuda.Stream(device=device),  # type: ignore[no-untyped-call]
            )
        return cls(pairs=pairs)

    def pair_for_device(self, device: torch.device) -> _DevicePair | None:
        if device.type != "cuda":
            return None
        index = int(device.index) if device.index is not None else 0
        return self.pairs.get(index)

    def transfer(
        self,
        value: torch.Tensor,
        target: torch.device,
    ) -> tuple[torch.Tensor, CudaEventHandle | None]:
        """Async H2D/D2H/D2D on the copy stream. Returns ``(dest, event)``.

        ``event`` is ``None`` when the tensor is already on ``target`` (no-op).
        Pageable host sources still copy; PyTorch may internally sync. Pinned
        host sources overlap with compute on the other stream.
        """
        if value.device == target:
            return value, None
        pair = self.pair_for_device(target if target.type == "cuda" else value.device)
        if pair is None:
            moved = value.to(target)
            return moved, None
        if target.type == "cuda" and value.device.type == "cpu":
            value = self._pinned_host(value)
        dest_buffer = None
        if target.type == "cuda":
            dest_buffer = self.acquire_buffer(tuple(int(d) for d in value.shape), value.dtype, target)
        non_blocking = bool(value.is_pinned()) or value.device.type == "cuda"
        with torch.cuda.stream(pair.copy_stream):
            if dest_buffer is not None and dest_buffer.shape == value.shape and dest_buffer.dtype == value.dtype:
                dest_buffer.copy_(value, non_blocking=non_blocking)
                dest = dest_buffer
            else:
                dest = value.to(target, non_blocking=non_blocking)
            event = torch.cuda.Event()  # type: ignore[no-untyped-call]
            event.record(pair.copy_stream)
        return dest, CudaEventHandle(event, pair.device)

    def _pinned_host(self, value: torch.Tensor) -> torch.Tensor:
        """Cache page-locked views of overflow host weights for async H2D."""
        key = id(value)
        cached = self._pinned_cache.get(key)
        if cached is not None:
            return cached
        from tensortorrent.runtime.pinning import pin_for_dma

        pinned = pin_for_dma(value)
        self._pinned_cache[key] = pinned
        return pinned

    def acquire_buffer(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Reusable device storage for one in-flight H2D. Returned via :meth:`release_buffer`."""
        index = int(device.index) if device.index is not None else 0
        key = (index, shape, dtype)
        pool = self._buffer_pool.setdefault(key, [])
        buf = pool.pop() if pool else torch.empty(shape, dtype=dtype, device=device)
        self._issued.add(id(buf))
        return buf

    def release_buffer(self, tensor: torch.Tensor) -> None:
        """Return an acquired H2D dest to the pool after its CUDA event completes."""
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            return
        ident = id(tensor)
        if ident not in self._issued:
            return
        self._issued.discard(ident)
        index = int(tensor.device.index) if tensor.device.index is not None else 0
        key = (index, tuple(int(d) for d in tensor.shape), tensor.dtype)
        pool = self._buffer_pool.setdefault(key, [])
        if len(pool) < self._pool_keep:
            pool.append(tensor)

    def wait_on_compute(self, device: torch.device, handle: CudaEventHandle | None) -> None:
        if handle is None:
            return
        pair = self.pair_for_device(device)
        if pair is None:
            handle.wait()
            return
        handle.wait_stream(pair.compute_stream)

    def run_compute(self, device: torch.device, fn: Any, *args: Any) -> tuple[Any, CudaEventHandle | None]:
        """Launch ``fn`` on the compute stream; record completion event."""
        pair = self.pair_for_device(device)
        if pair is None:
            return fn(*args), None
        with torch.cuda.stream(pair.compute_stream):
            result = fn(*args)
            event = torch.cuda.Event()  # type: ignore[no-untyped-call]
            event.record(pair.compute_stream)
        return result, CudaEventHandle(event, pair.device)

    def synchronize_all(self) -> None:
        for pair in self.pairs.values():
            pair.compute_stream.synchronize()
            pair.copy_stream.synchronize()

    def close(self) -> None:
        self.synchronize_all()
        self.pairs.clear()
        self._pinned_cache.clear()
        self._buffer_pool.clear()
        self._issued.clear()


def streams_enabled_for_context(ctx: Any) -> bool:
    """False under autograd — saved tensors must not race async copies."""
    if ctx is None or bool(getattr(ctx, "enable_grad", False)):
        return False
    runtime = getattr(ctx, "device_streams", None)
    return isinstance(runtime, DeviceStreamRuntime)


def runtime_for_context(ctx: Any) -> DeviceStreamRuntime | None:
    """Typed stream runtime when overlap is enabled for ``ctx``."""
    if not streams_enabled_for_context(ctx):
        return None
    runtime = getattr(ctx, "device_streams", None)
    return runtime if isinstance(runtime, DeviceStreamRuntime) else None


def _resource_is_cudaish(resource: str) -> bool:
    name = str(resource or "").lower()
    if "mock" in name:
        return False
    return "cuda" in name or "rocm" in name or (is_device_resource(name) and "xpu" not in name)


def _device_index(binding: Any) -> int | None:
    compiled = getattr(binding, "compiled", None)
    torch_device = getattr(compiled, "torch_device", None)
    if torch_device is not None:
        try:
            dev = torch_device if isinstance(torch_device, torch.device) else torch.device(torch_device)
        except (RuntimeError, TypeError, ValueError):
            dev = None
        if dev is not None and dev.type == "cuda":
            return int(dev.index) if dev.index is not None else 0
    resource = str(getattr(binding, "device", "") or "")
    mapped = _torch_device(resource)
    if mapped is not None and mapped.type == "cuda":
        return int(mapped.index) if mapped.index is not None else 0
    return None


def _torch_device(resource: str) -> torch.device | None:
    from tensortorrent.runtime.native_bridge.residency import _torch_device_for_resource

    mapped = _torch_device_for_resource(resource)
    if isinstance(mapped, torch.device) and mapped.type == "cuda":
        return mapped
    return None
