"""Per-call mutable execution state — never stored on ExecutableSchedule.

The immutable schedule is the program. This context holds residency, telemetry,
and cancellation for a single ``run()`` call. Physical allocation accounting
lives in Rust (``tt_memory::AllocationTable``); Python only mirrors copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tensortorrent.runtime.copies import CopyStore


@dataclass
class InstructionRuntimeState:
    """Mutable per-instruction timing/result for one execution."""

    submitted_s: float | None = None
    start_s: float | None = None
    completion_s: float | None = None
    result: Any = None


@dataclass
class CancellationState:
    """Python-side cancel flag (native path also checks NativeCancelToken)."""

    cancelled: bool = False


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
class ExecutionContext:
    """Mutable state for one schedule execution. Schedule itself stays frozen."""

    instruction_states: dict[str, InstructionRuntimeState] = field(default_factory=dict)
    copies: CopyStore = field(default_factory=CopyStore)
    telemetry: TelemetryRecorder = field(default_factory=TelemetryRecorder)
    cancellation: CancellationState = field(default_factory=CancellationState)
    host_resource: str = "cpu"
    activation_peak_bytes: int = 0
    # Per-run train flag: must live on the context so Rust worker-thread
    # region callbacks see the same value as the submitting thread.
    enable_grad: bool = False
    # When set, Rust NativeResidencySession is authoritative for residency metadata.
    native_residency: Any | None = None
    # Shared NativeExecutionContext (virtual buffers, cancel, streaming).
    native_execution_context: Any | None = None

    def note_activation_live(self, live_bytes: int) -> None:
        self.activation_peak_bytes = max(self.activation_peak_bytes, max(0, int(live_bytes)))

    def state_for(self, instruction_name: str) -> InstructionRuntimeState:
        st = self.instruction_states.get(instruction_name)
        if st is None:
            st = InstructionRuntimeState()
            self.instruction_states[instruction_name] = st
        return st

    def mirror_native_put(
        self,
        tensor_id: str,
        resource_id: str,
        value: Any,
        *,
        nbytes: int | None = None,
        view_meta: dict[str, Any] | None = None,
    ) -> None:
        bridge = self.native_residency
        if bridge is None:
            return
        if nbytes is None:
            import torch

            nbytes = int(value.nbytes) if isinstance(value, torch.Tensor) else 0
        bridge.mirror_put(tensor_id, resource_id, value, nbytes=int(nbytes), view_meta=view_meta)

    def publish(
        self,
        tensor_id: str,
        resource_id: str,
        value: Any,
        *,
        tier: str = "system_ram",
        ownership: str = "activation",
        nbytes: int | None = None,
        precomputed: Any = None,
        view_meta: dict[str, Any] | None = None,
        authoritative: bool = True,
    ) -> None:
        """Register a tensor in the Python handle bag and mirror into Rust.

        Rust owns residency metadata; CopyStore holds the local value handle.
        Call sites should prefer this over separate ``copies.put`` +
        ``mirror_native_put`` so both stay in sync.
        """
        from tensortorrent.runtime.copies import describe_tensor
        from tensortorrent.runtime.handles import _tensor_view_meta

        meta = precomputed if precomputed is not None else describe_tensor(value, tensor_id, resource_id)
        self.copies.put(
            tensor_id,
            resource_id,
            value,
            tier=tier,
            ownership=ownership,
            precomputed=meta,
        )
        if nbytes is None:
            nbytes = int(getattr(meta, "nbytes", 0) or getattr(value, "nbytes", 0) or 0)
        if view_meta is None:
            view_meta = _tensor_view_meta(value)
        bridge = self.native_residency
        if bridge is not None:
            bridge.mirror_put(
                tensor_id,
                resource_id,
                value,
                nbytes=int(nbytes),
                view_meta=view_meta,
                authoritative=authoritative,
            )

    def native_require(self, tensor_id: str, resource_id: str) -> None:
        bridge = self.native_residency
        if bridge is None:
            return
        bridge.require_handle(tensor_id, resource_id)
