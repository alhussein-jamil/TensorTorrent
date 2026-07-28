"""Async execution and transfer streams (mock + CUDA-shaped API).

On CPU-only hosts, mock streams use background workers with deterministic delays
so RecordEvent stays incomplete until work finishes. CUDA will later map the same
API onto dedicated compute/copy streams and ``stream.wait_event``.

Development VMs without GPUs validate asynchronous semantics via
:class:`MockStream` / :class:`StreamEvent` only — never claim real CUDA execution.

CUDA helpers (:func:`make_event`, :func:`make_stream`, :func:`synchronize_device`)
live here so there is one event/stream module — not a parallel ``async_events`` path.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from streamcompiler.errors import RuntimePlanError


@runtime_checkable
class BackendEvent(Protocol):
    """Backend-owned completion handle (CUDA event / host future)."""

    def wait(self, *, timeout: float | None = None) -> None: ...

    def is_complete(self) -> bool: ...


@runtime_checkable
class ExecutionStream(Protocol):
    """Backend-owned ordered work queue (CUDA stream / host executor)."""

    def submit_compute(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]: ...

    def submit_transfer(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]: ...

    def record_event(self, name: str) -> BackendEvent: ...

    def wait_event(self, event: BackendEvent) -> None: ...


@dataclass
class StreamEvent:
    """Completion handle. Incomplete until the owning future finishes."""

    name: str
    device: str
    _future: Future[Any] | None = None
    _flag: threading.Event = field(default_factory=threading.Event)
    completed: bool = False
    enqueue_start_s: float = 0.0
    enqueue_end_s: float = 0.0
    complete_s: float = 0.0

    def bind_future(self, future: Future[Any], *, enqueue_start_s: float, enqueue_end_s: float) -> None:
        self._future = future
        self.enqueue_start_s = enqueue_start_s
        self.enqueue_end_s = enqueue_end_s

        def _mark(fut: Future[Any]) -> None:
            self.complete_s = time.perf_counter()
            self.completed = True
            self._flag.set()

        future.add_done_callback(_mark)
        if future.done():
            _mark(future)

    def record(self) -> None:
        """Mark host-side bookkeeping complete (CPU sync path)."""
        self.completed = True
        self.complete_s = time.perf_counter()
        self._flag.set()

    def is_complete(self) -> bool:
        return bool(self.completed)

    def wait(self, *, timeout: float | None = None) -> None:
        if self.completed:
            return
        if self._future is not None:
            self._future.result(timeout=timeout)
            return
        if not self._flag.wait(timeout=timeout):
            raise RuntimePlanError(f"WaitEvent {self.name!r} timed out on {self.device!r}")
        if not self.completed:
            raise RuntimePlanError(f"WaitEvent {self.name!r} has no recorded completion on device {self.device!r}")


@dataclass
class EventRegistry:
    _events: dict[str, StreamEvent] = field(default_factory=dict)

    def store(self, name: str, event: StreamEvent) -> None:
        self._events[name] = event

    def get(self, name: str) -> StreamEvent:
        event = self._events.get(name)
        if event is None:
            raise RuntimePlanError(f"WaitEvent references unknown RecordEvent {name!r}")
        return event

    def clear(self) -> None:
        self._events.clear()


class MockStream:
    """Independent background queue with a fixed per-op delay (seconds)."""

    def __init__(self, name: str, *, delay_s: float = 0.0, workers: int = 1) -> None:
        self.name = name
        self.delay_s = float(delay_s)
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix=f"mock-{name}")

    def submit(self, fn: Callable[..., Any], *args: Any, delay_s: float | None = None, **kwargs: Any) -> Future[Any]:
        delay = self.delay_s if delay_s is None else float(delay_s)

        def _run() -> Any:
            if delay > 0:
                time.sleep(delay)
            return fn(*args, **kwargs)

        return self._pool.submit(_run)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=not wait)


class DeviceStreams:
    """Per-resource compute and copy streams."""

    def __init__(self) -> None:
        self._compute: dict[str, MockStream] = {}
        self._copy: dict[str, MockStream] = {}
        self._lock = threading.Lock()

    def compute_stream(self, resource_id: str, *, delay_s: float = 0.0, workers: int = 4) -> MockStream:
        with self._lock:
            stream = self._compute.get(resource_id)
            want = max(1, int(workers))
            if stream is None or getattr(stream, "_wanted_workers", 1) < want:
                if stream is not None:
                    stream.shutdown()
                stream = MockStream(f"compute:{resource_id}", delay_s=delay_s, workers=want)
                stream._wanted_workers = want  # type: ignore[attr-defined]
                self._compute[resource_id] = stream
            return stream

    def copy_stream(self, resource_id: str, *, delay_s: float = 0.0, workers: int = 2) -> MockStream:
        with self._lock:
            stream = self._copy.get(resource_id)
            want = max(1, int(workers))
            if stream is None or getattr(stream, "_wanted_workers", 1) < want:
                if stream is not None:
                    stream.shutdown()
                stream = MockStream(f"copy:{resource_id}", delay_s=delay_s, workers=want)
                stream._wanted_workers = want  # type: ignore[attr-defined]
                self._copy[resource_id] = stream
            return stream

    def configure_mock(self, resource_id: str, *, compute_delay_s: float, transfer_delay_s: float) -> None:
        with self._lock:
            self._compute[resource_id] = MockStream(f"compute:{resource_id}", delay_s=compute_delay_s)
            self._copy[resource_id] = MockStream(f"copy:{resource_id}", delay_s=transfer_delay_s)

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            for stream in list(self._compute.values()) + list(self._copy.values()):
                stream.shutdown(wait=wait)
            self._compute.clear()
            self._copy.clear()


class HostExecutionStream:
    """CPU :class:`ExecutionStream` backed by a persistent thread pool."""

    def __init__(self, name: str = "host", *, workers: int = 2) -> None:
        self.name = name
        self._compute = MockStream(f"{name}:compute", delay_s=0.0, workers=workers)
        self._copy = MockStream(f"{name}:copy", delay_s=0.0, workers=max(1, workers // 2))
        self._last_compute: Future[Any] | None = None
        self._last_transfer: Future[Any] | None = None

    def submit_compute(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        fut = self._compute.submit(fn, *args, **kwargs)
        self._last_compute = fut
        return fut

    def submit_transfer(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        fut = self._copy.submit(fn, *args, **kwargs)
        self._last_transfer = fut
        return fut

    def record_event(self, name: str) -> StreamEvent:
        event = StreamEvent(name=name, device=self.name)
        last = self._last_transfer or self._last_compute
        if last is not None:
            now = time.perf_counter()
            event.bind_future(last, enqueue_start_s=now, enqueue_end_s=now)
        else:
            event.record()
        return event

    def wait_event(self, event: BackendEvent) -> None:
        event.wait()

    def shutdown(self, wait: bool = True) -> None:
        self._compute.shutdown(wait=wait)
        self._copy.shutdown(wait=wait)


class MockExecutionStream:
    """Deterministic virtual-accelerator stream (simulated DMA / compute delays)."""

    def __init__(
        self,
        name: str,
        *,
        compute_delay_s: float = 0.05,
        transfer_delay_s: float = 0.08,
    ) -> None:
        self.name = name
        self._compute = MockStream(f"{name}:compute", delay_s=compute_delay_s, workers=1)
        self._copy = MockStream(f"{name}:copy", delay_s=transfer_delay_s, workers=1)
        self._last_compute: Future[Any] | None = None
        self._last_transfer: Future[Any] | None = None

    def submit_compute(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        fut = self._compute.submit(fn, *args, **kwargs)
        self._last_compute = fut
        return fut

    def submit_transfer(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        fut = self._copy.submit(fn, *args, **kwargs)
        self._last_transfer = fut
        return fut

    def record_event(self, name: str) -> StreamEvent:
        event = StreamEvent(name=name, device=self.name)
        last = self._last_transfer or self._last_compute
        if last is not None:
            now = time.perf_counter()
            event.bind_future(last, enqueue_start_s=now, enqueue_end_s=now)
        else:
            event.record()
        return event

    def wait_event(self, event: BackendEvent) -> None:
        event.wait()

    def shutdown(self, wait: bool = True) -> None:
        self._compute.shutdown(wait=wait)
        self._copy.shutdown(wait=wait)


@dataclass
class CudaEvent:
    """Host-visible CUDA completion handle when a CUDA device is available."""

    name: str
    device: str
    cuda_event: Any | None = None
    completed: bool = False

    def record(self, stream: Any | None = None) -> None:
        if self.cuda_event is not None:
            self.cuda_event.record(stream)
            return
        self.completed = True

    def is_complete(self) -> bool:
        if self.cuda_event is not None:
            return bool(self.cuda_event.query())
        return bool(self.completed)

    def wait(self, stream: Any | None = None, *, timeout: float | None = None) -> None:
        del timeout  # CUDA events have no host timeout API here.
        if self.cuda_event is not None:
            if stream is not None:
                stream.wait_event(self.cuda_event)
            else:
                self.cuda_event.synchronize()
            self.completed = True
            return
        if not self.completed:
            raise RuntimePlanError(f"WaitEvent {self.name!r} has no recorded completion on device {self.device!r}")
        self.completed = True


def make_event(name: str, device: str) -> StreamEvent | CudaEvent:
    """Build a completion handle for ``device`` (CUDA event when available)."""
    if "cuda" in device.lower() and torch.cuda.is_available():
        return CudaEvent(name=name, device=device, cuda_event=torch.cuda.Event(enable_timing=True))  # type: ignore[no-untyped-call]
    event = StreamEvent(name=name, device=device)
    return event


def make_stream(device: str) -> Any | None:
    """Return a ``torch.cuda.Stream`` when ``device`` names an available CUDA index."""
    if "cuda" in device.lower() and torch.cuda.is_available():
        digits = "".join(ch for ch in device if ch.isdigit())
        index = int(digits) if digits else 0
        with torch.cuda.device(index):
            return torch.cuda.Stream()  # type: ignore[no-untyped-call]
    return None


def synchronize_device(device: str) -> None:
    """Host-sync a CUDA device; no-op for CPU / mock resources."""
    if "cuda" in device.lower() and torch.cuda.is_available():
        torch.cuda.synchronize()
        return
    if "cuda" in device.lower():
        raise RuntimePlanError(f"Cannot synchronize unavailable device {device!r}")
