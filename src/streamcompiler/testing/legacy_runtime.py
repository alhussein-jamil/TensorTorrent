"""Developer-only legacy Python DAG runner for benchmarks. Never auto-activates."""

from __future__ import annotations

import time
from typing import Any

from streamcompiler.errors import RuntimePlanError
from streamcompiler.runtime.execution_context import ExecutionContext
from streamcompiler.runtime.schedule_executor import InstructionEvent, ScheduleReport, max_concurrency_from_intervals


def run_schedule_legacy_python(executor: Any, flat_inputs: list[Any]) -> tuple[list[Any], ScheduleReport]:
    """Execute via Python ``_dispatch`` DAG (oracle / bench only).

    Production forwards use :func:`streamcompiler.runtime.native_bridge.run_schedule_native`.
    """
    if executor._closed:
        raise RuntimePlanError("ScheduleExecutor is closed")
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

    # Match native resident seeding — no fake Load for already-mapped packs.
    from streamcompiler.runtime.native_bridge import _register_persistent_residency

    _register_persistent_residency(executor, ctx)

    executor.parameter_store.begin_execution()
    wall0 = time.perf_counter()
    completed: set[str] = set()
    remaining = {i.name: len(i.depends_on) for i in executor.schedule.instructions}
    dependents: dict[str, list[str]] = {i.name: [] for i in executor.schedule.instructions}
    for inst in executor.schedule.instructions:
        for dep in inst.depends_on:
            dependents.setdefault(dep, []).append(inst.name)
    ready = [n for n, d in remaining.items() if d == 0]
    events: list[InstructionEvent] = []

    while ready:
        name = ready.pop(0)
        inst = executor._by_name[name]
        submitted = time.perf_counter()
        from streamcompiler.runtime._legacy_dispatch import dispatch

        fut = dispatch(executor, inst, ctx, submitted)
        event = fut.result()
        assert isinstance(event, InstructionEvent)
        events.append(event)
        completed.add(name)
        for nxt in dependents.get(name, []):
            remaining[nxt] -= 1
            if remaining[nxt] == 0:
                ready.append(nxt)

    missing = [i.name for i in executor.schedule.instructions if i.name not in completed]
    if missing:
        raise RuntimePlanError(f"Legacy schedule left unfinished: {missing}")

    report.events = events
    report.wall_time_s = time.perf_counter() - wall0
    report.parallel_overlaps = len(report.overlapping_pairs())
    report.copy_snapshot = ctx.copies.snapshot()
    report.max_concurrent = max(
        1,
        max_concurrency_from_intervals([(e.start_s, e.end_s) for e in events]),
    )
    report.parameter_store = {
        "legacy_python_dag": True,
        "native_data_plane": False,
        "native_runtime": False,
    }
    return executor._collect_outputs(ctx), report
