"""Native async completion helpers (CUDA events / streams when present)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from streamcompiler.errors import RuntimePlanError


@dataclass
class AsyncEvent:
    """Host-visible completion handle. Wraps ``torch.cuda.Event`` when on CUDA."""

    name: str
    device: str
    cuda_event: Any | None = None
    completed: bool = False

    def record(self, stream: Any | None = None) -> None:
        if self.cuda_event is not None:
            self.cuda_event.record(stream)
            return
        self.completed = True

    def wait(self, stream: Any | None = None) -> None:
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


@dataclass
class EventRegistry:
    """Named event store: RecordEvent inserts; WaitEvent waits the same handle."""

    _events: dict[str, AsyncEvent] = field(default_factory=dict)

    def store(self, name: str, event: AsyncEvent) -> None:
        self._events[name] = event

    def get(self, name: str) -> AsyncEvent:
        event = self._events.get(name)
        if event is None:
            raise RuntimePlanError(f"WaitEvent references unknown RecordEvent {name!r}")
        return event

    def clear(self) -> None:
        self._events.clear()


def make_event(name: str, device: str) -> AsyncEvent:
    if "cuda" in device.lower() and torch.cuda.is_available():
        return AsyncEvent(name=name, device=device, cuda_event=torch.cuda.Event(enable_timing=True))  # type: ignore[no-untyped-call]
    return AsyncEvent(name=name, device=device)


def make_stream(device: str) -> Any | None:
    if "cuda" in device.lower() and torch.cuda.is_available():
        digits = "".join(ch for ch in device if ch.isdigit())
        index = int(digits) if digits else 0
        with torch.cuda.device(index):
            return torch.cuda.Stream()  # type: ignore[no-untyped-call]
    return None


def synchronize_device(device: str) -> None:
    if "cuda" in device.lower() and torch.cuda.is_available():
        torch.cuda.synchronize()
        return
    if "cuda" in device.lower():
        raise RuntimePlanError(f"Cannot synchronize unavailable device {device!r}")
