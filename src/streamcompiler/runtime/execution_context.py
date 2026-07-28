"""Per-call mutable execution state — never stored on ExecutableSchedule.

The immutable schedule is the program. This context holds futures, events,
residency, leases, telemetry, and cancellation for a single ``run()`` call.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from streamcompiler.runtime.copies import CopyStore
from streamcompiler.runtime.streams import EventRegistry


@dataclass
class InstructionState:
    """Mutable per-instruction runtime state for one execution."""

    submitted_s: float | None = None
    start_s: float | None = None
    completion_s: float | None = None
    future: Future[Any] | None = None
    completion_event: Any | None = None
    result: Any = None
    exception: BaseException | None = None
    wait_duration_s: float = 0.0
    resource_lease: Any | None = None
    async_future: Future[Any] | None = None
    enqueue_start_s: float = 0.0
    enqueue_end_s: float = 0.0


@dataclass
class CancellationState:
    cancelled: bool = False

    def request(self) -> None:
        self.cancelled = True

    def clear(self) -> None:
        self.cancelled = False

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            from streamcompiler.errors import ExecutionCancelled

            self.cancelled = False
            raise ExecutionCancelled("Schedule execution cancelled")


@dataclass
class TelemetryRecorder:
    """Collects instruction-level telemetry for one execution."""

    events: list[Any] = field(default_factory=list)
    spill_events: list[dict[str, Any]] = field(default_factory=list)
    multi_copy_peaks: list[dict[str, Any]] = field(default_factory=list)
    activation_bytes_written: int = 0
    activation_bytes_read: int = 0
    spill_latency_s: float = 0.0
    reload_latency_s: float = 0.0

    def record_spill(self, *, name: str, nbytes: int, latency_s: float, **extra: Any) -> None:
        self.activation_bytes_written += max(0, nbytes)
        self.spill_latency_s += max(0.0, latency_s)
        self.spill_events.append({"event": "spill", "name": name, "nbytes": nbytes, **extra})

    def record_reload(self, *, name: str, nbytes: int, latency_s: float, **extra: Any) -> None:
        self.activation_bytes_read += max(0, nbytes)
        self.reload_latency_s += max(0.0, latency_s)
        self.spill_events.append({"event": "reload", "name": name, "nbytes": nbytes, **extra})


@dataclass
class AllocationTable:
    """Physical allocation accounting keyed by allocation id."""

    _allocs: dict[str, dict[str, Any]] = field(default_factory=dict)
    _live_bytes: int = 0
    _peak_bytes: int = 0

    def register(self, allocation_id: str, *, resource_id: str, capacity_bytes: int, handle: Any = None) -> None:
        if allocation_id in self._allocs:
            self._allocs[allocation_id]["reference_count"] += 1
            return
        self._allocs[allocation_id] = {
            "allocation_id": allocation_id,
            "resource_id": resource_id,
            "capacity_bytes": capacity_bytes,
            "physical_handle": handle,
            "reference_count": 1,
        }
        self._live_bytes += capacity_bytes
        self._peak_bytes = max(self._peak_bytes, self._live_bytes)

    def release(self, allocation_id: str) -> int:
        rec = self._allocs.get(allocation_id)
        if rec is None:
            return 0
        rec["reference_count"] -= 1
        if rec["reference_count"] > 0:
            return 0
        freed = int(rec["capacity_bytes"])
        del self._allocs[allocation_id]
        self._live_bytes = max(0, self._live_bytes - freed)
        return freed

    def live_bytes(self) -> int:
        return self._live_bytes

    def peak_bytes(self) -> int:
        return self._peak_bytes

    def snapshot(self) -> dict[str, Any]:
        return {
            "live_bytes": self._live_bytes,
            "peak_bytes": self._peak_bytes,
            "allocations": {k: dict(v) for k, v in self._allocs.items()},
        }


@dataclass
class ExecutionContext:
    """Mutable state for one schedule execution. Schedule itself stays frozen."""

    instruction_states: dict[str, InstructionState] = field(default_factory=dict)
    events: EventRegistry = field(default_factory=EventRegistry)
    copies: CopyStore = field(default_factory=CopyStore)
    allocations: AllocationTable = field(default_factory=AllocationTable)
    telemetry: TelemetryRecorder = field(default_factory=TelemetryRecorder)
    cancellation: CancellationState = field(default_factory=CancellationState)
    pending_transfers: dict[tuple[str, str], Future[Any]] = field(default_factory=dict)
    host_resource: str = "cpu"

    def state_for(self, instruction_name: str) -> InstructionState:
        st = self.instruction_states.get(instruction_name)
        if st is None:
            st = InstructionState()
            self.instruction_states[instruction_name] = st
        return st
