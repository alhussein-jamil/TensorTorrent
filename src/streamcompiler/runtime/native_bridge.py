"""Bridge: Rust schedules instructions; Python executes tensor-bearing ops."""

from __future__ import annotations

import time
from typing import Any

from streamcompiler.errors import ExecutionCancelled, RuntimePlanError
from streamcompiler.ir.graph import OpCode
from streamcompiler.native import allow_python_runtime, native_available, require_native
from streamcompiler.runtime.execution_context import ExecutionContext
from streamcompiler.runtime.schedule import PlanInstruction
from streamcompiler.runtime.schedule_executor import InstructionEvent, ScheduleReport


def _exec_inline(executor: Any, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> InstructionEvent:
    """Run one instruction body without an extra Python thread-pool hop."""
    opcode = inst.opcode
    if opcode == OpCode.PREFETCH:
        event = executor._exec_prefetch(inst, ctx, submitted)
    elif opcode == OpCode.LOAD:
        event = executor._exec_load(inst, ctx, submitted)
    elif opcode == OpCode.TRANSFER:
        event = executor._submit_transfer(inst, ctx, submitted).result()
    elif opcode == OpCode.RECORD_EVENT:
        event = executor._exec_record(inst, ctx, submitted)
    elif opcode == OpCode.WAIT_EVENT:
        event = executor._exec_wait(inst, ctx, submitted)
    elif opcode == OpCode.COMPUTE:
        event = executor._submit_compute(inst, ctx, submitted).result()
    elif opcode == OpCode.RELEASE:
        event = executor._exec_release(inst, ctx, submitted)
    elif opcode == OpCode.EVICT:
        event = executor._exec_evict(inst, ctx, submitted)
    else:
        raise RuntimePlanError(f"Unsupported schedule opcode {opcode}")
    assert isinstance(event, InstructionEvent)
    return event


def run_schedule_native(executor: Any, flat_inputs: list[Any]) -> tuple[list[Any], ScheduleReport]:
    """Run ``executor.schedule`` under the Rust dispatcher.

    Rust owns dependency counters, ready queues, worker pools, and waits with
    the GIL released. Each ready instruction re-enters Python only for the
    existing ``_dispatch`` body (tensor I/O / region compute).
    """
    native = require_native()
    if executor._closed:
        raise RuntimePlanError("ScheduleExecutor is closed")
    if executor._cancel:
        executor._cancel = False
        raise ExecutionCancelled("Schedule execution cancelled")
    executor._cancel = False

    ctx = ExecutionContext(host_resource=executor._default_host_resource())
    report = ScheduleReport(wall_time_s=0.0)
    host = ctx.host_resource
    if len(flat_inputs) != len(executor.program.user_inputs):
        raise RuntimePlanError(f"Expected {len(executor.program.user_inputs)} inputs, got {len(flat_inputs)}")
    for name, value in zip(executor.program.user_inputs, flat_inputs, strict=True):
        ctx.copies.put(name, host, value, tier="system_ram", authoritative=True, ownership="input")
        if host != "cpu":
            ctx.copies.alias(name, host, "cpu")
        if host != "host":
            ctx.copies.alias(name, host, "host")

    executor.parameter_store.begin_execution()
    wall0 = time.perf_counter()
    completed: set[str] = set()
    events_by_name: dict[str, InstructionEvent] = {}

    def handler(name: str) -> dict[str, Any]:
        if executor._cancel or ctx.cancellation.cancelled:
            raise ExecutionCancelled("Schedule execution cancelled")
        inst = executor._by_name[name]
        submitted = time.perf_counter()
        ctx.state_for(name).submitted_s = submitted
        # Execute instruction body inline: Rust already owns concurrency.
        event = _exec_inline(executor, inst, ctx, submitted)
        events_by_name[name] = event
        report.events.append(event)
        st = ctx.state_for(name)
        st.start_s = event.start_s
        st.completion_s = event.end_s
        st.result = event
        completed.add(name)
        executor._assert_activation_budget(ctx, completed)
        return {
            "nbytes": int(event.nbytes),
            "simulated": bool(event.simulated),
            "notes": str(event.notes or ""),
        }

    try:
        native_report = native.execute_schedule(
            executor.schedule,
            instruction_handler=handler,
            dry_run=False,
            cpu_workers=max(4, int(executor.max_inflight)),
        )
    except Exception as exc:
        if isinstance(exc, (ExecutionCancelled, RuntimePlanError)):
            raise
        raise RuntimePlanError(f"native schedule execution failed: {exc}") from exc

    missing = [i.name for i in executor.schedule.instructions if i.name not in completed]
    if missing:
        raise RuntimePlanError(f"Schedule left unfinished instructions: {missing}")

    executor._assert_activation_budget(ctx, completed)
    report.wall_time_s = float(native_report.get("wall_time_s") or (time.perf_counter() - wall0))
    report.parallel_overlaps = len(report.overlapping_pairs())
    report.copy_snapshot = ctx.copies.snapshot()
    report.multi_copy_peaks = list(ctx.telemetry.multi_copy_peaks)
    report.peak_activation_bytes = max(ctx.activation_peak_bytes, ctx.copies.activation_live_bytes())
    report.activation_bytes_written = ctx.telemetry.activation_bytes_written
    report.activation_bytes_read = ctx.telemetry.activation_bytes_read
    report.spill_latency_s = ctx.telemetry.spill_latency_s
    report.reload_latency_s = ctx.telemetry.reload_latency_s
    report.allocation_peak_bytes = ctx.allocations.peak_bytes()
    if report.allocation_peak_bytes == 0 and report.peak_activation_bytes > 0:
        report.allocation_peak_bytes = report.peak_activation_bytes
    executor.copies = ctx.copies
    executor._spill_events.extend(ctx.telemetry.spill_events)
    compute_intervals = [(e.start_s, e.end_s) for e in report.events if e.opcode == "Compute"]
    if hasattr(executor.parameter_store, "record_compute_intervals"):
        executor.parameter_store.record_compute_intervals(compute_intervals)
    stats = executor.parameter_store.stats()
    if isinstance(stats, dict):
        stats = dict(stats)
        stats["schedule_instruction_events"] = len(report.events)
        stats["schedule_driven"] = True
        stats["native_runtime"] = True
        stats["peak_activation_bytes"] = report.peak_activation_bytes
        stats["activation_bytes_written"] = report.activation_bytes_written
        stats["activation_bytes_read"] = report.activation_bytes_read
    report.parameter_store = stats if isinstance(stats, dict) else {}
    report.max_concurrent = max(1, len({e.name for e in report.events}))
    return executor._collect_outputs(ctx), report


def should_use_native_runtime() -> bool:
    if native_available():
        return True
    if allow_python_runtime():
        return False
    require_native()  # raises
    return False
