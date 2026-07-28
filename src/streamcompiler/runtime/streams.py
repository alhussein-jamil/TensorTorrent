"""Async execution and transfer streams (mock + CUDA-shaped API).

On CPU-only hosts, mock streams use background workers with deterministic delays
so RecordEvent stays incomplete until work finishes. CUDA will later map the same
API onto dedicated compute/copy streams and ``stream.wait_event``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from streamcompiler.errors import RuntimePlanError


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
