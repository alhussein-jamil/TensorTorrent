"""Test-only mock/CUDA-shaped stream and event helpers.

Production schedule execution uses the native runtime. These types exist so
CPU-only CI can still exercise RecordEvent / WaitEvent ordering semantics.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from tensortorrent.errors import RuntimePlanError


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
    """Background work queue with optional ordered submission.

    Ordered by default regardless of worker count. Pass ``ordered=False`` only for
    host CPU pools where independent region overlap is intentional. Prior future
    failures never poison later work.
    """

    def __init__(
        self,
        name: str,
        *,
        delay_s: float = 0.0,
        workers: int = 1,
        ordered: bool = True,
    ) -> None:
        self.name = name
        self.delay_s = float(delay_s)
        self.workers = max(1, int(workers))
        self._ordered = bool(ordered)
        self._pool = ThreadPoolExecutor(
            max_workers=1 if self._ordered else self.workers,
            thread_name_prefix=f"mock-{name}",
        )
        self._chain: Future[Any] | None = None
        self._lock = threading.Lock()

    def submit(self, fn: Callable[..., Any], *args: Any, delay_s: float | None = None, **kwargs: Any) -> Future[Any]:
        delay = self.delay_s if delay_s is None else float(delay_s)

        def _body() -> Any:
            if delay > 0:
                time.sleep(delay)
            return fn(*args, **kwargs)

        if not self._ordered:
            return self._pool.submit(_body)

        with self._lock:
            prev = self._chain

            def _run() -> Any:
                if prev is not None:
                    # Preserve submission order only — prior errors stay on their future.
                    with contextlib.suppress(Exception):
                        prev.result()
                return _body()

            fut = self._pool.submit(_run)
            self._chain = fut
            return fut

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=not wait)


class HostExecutionStream:
    """CPU :class:`ExecutionStream` with ordered compute and copy queues."""

    def __init__(self, name: str = "host", *, workers: int = 1) -> None:
        del workers
        self.name = name
        self._compute = MockStream(f"{name}:compute", delay_s=0.0, workers=1)
        self._copy = MockStream(f"{name}:copy", delay_s=0.0, workers=1)
        self._last_compute: Future[Any] | None = None
        self._last_transfer: Future[Any] | None = None
        self._compute_chain: Future[Any] | None = None
        self._transfer_chain: Future[Any] | None = None

    def submit_compute(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        # HostExecutionStream's MockStream already orders when workers=1.
        fut = self._compute.submit(fn, *args, **kwargs)
        self._compute_chain = fut
        self._last_compute = fut
        return fut

    def submit_transfer(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        fut = self._copy.submit(fn, *args, **kwargs)
        self._transfer_chain = fut
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
    """Deterministic virtual-accelerator stream (simulated DMA / compute delays).

    One stream preserves submission order via an explicit future chain.
    """

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
        self._compute_chain: Future[Any] | None = None
        self._transfer_chain: Future[Any] | None = None

    def submit_compute(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        fut = self._compute.submit(fn, *args, **kwargs)
        self._compute_chain = fut
        self._last_compute = fut
        return fut

    def submit_transfer(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        fut = self._copy.submit(fn, *args, **kwargs)
        self._transfer_chain = fut
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
