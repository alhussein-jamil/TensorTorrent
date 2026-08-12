"""Discrete-event simulator for :class:`ExecutableSchedule` instruction DAGs.

Analytic only: kernels are not executed. Makespan, transfer exposure, peak
memory, and contention come from schedule instruction costs, explicit
dependencies, and the machine's transfer links. ``simulate_plan`` is a thin
wrapper that lowers an ``ExecutionPlan`` to an executable schedule first —
the simulator never invents transfers absent from that schedule.

Implementation is the Rust discrete-event walk (native extension required).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tensortorrent.closed import SimulationStatus, closed_str
from tensortorrent.ir.resource_graph import ResourceGraph
from tensortorrent.planner.maximal import ExecutionPlan


@dataclass
class SimulationResult:
    makespan_s: float
    peak_bytes: dict[str, int]
    timeline: list[dict[str, Any]]
    exposed_transfer_latency_s: float
    resource_busy_s: dict[str, float]
    transfer_events: list[dict[str, Any]] = field(default_factory=list)
    release_events: list[dict[str, Any]] = field(default_factory=list)
    simulated: bool = True
    """Always True: this path never executes kernels."""
    critical_path: list[str] = field(default_factory=list)
    bytes_read: int = 0
    bytes_transferred: int = 0
    instruction_count: int = 0
    resource_utilization: dict[str, float] = field(default_factory=dict)
    """Busy time / makespan per compute resource (0..1+ under oversub)."""
    activation_peak_bytes: int = 0
    """Peak bytes of distinct live physical activation allocations."""
    initiation_interval_s: float = 0.0
    """Steady-state bottleneck (max resource busy time) from DES."""


def simulate_plan(
    plan: ExecutionPlan,
    machine: ResourceGraph,
    *,
    residency: Any | None = None,
    streaming: bool = False,
    prefetch_distance: int = 1,
    program: Any | None = None,
) -> SimulationResult:
    """Lower ``plan`` to an :class:`ExecutableSchedule`, then simulate that DAG.

    Does not invent transfers: cross-device movement must appear as Transfer
    instructions from residency / schedule building.
    """
    from tensortorrent.runtime.residency import build_residency_schedule
    from tensortorrent.runtime.schedule import build_executable_schedule

    if residency is None:
        residency = build_residency_schedule(plan, program)
    if streaming is False:
        for note in plan.notes:
            if note.startswith("prefetch_distance="):
                try:
                    prefetch_distance = max(0, int(note.split("=", 1)[1]))
                    streaming = prefetch_distance > 0
                except ValueError:
                    pass
                break
    schedule = build_executable_schedule(
        plan,
        residency,
        streaming=streaming,
        prefetch_distance=prefetch_distance,
        program=program,
    )
    return simulate_schedule(schedule, machine)


def _result_from_raw(raw: dict[str, Any], *, instruction_count: int = 0) -> SimulationResult:
    timeline = list(raw.get("timeline") or [])
    busy = {str(k): float(v) for k, v in dict(raw.get("resource_busy_s") or {}).items()}
    makespan = float(raw.get("makespan_s") or 0.0)
    ii = float(raw.get("initiation_interval_s") or 0.0)
    if ii <= 0 and busy:
        ii = max(busy.values())
    denom = makespan if makespan > 0 else 0.0
    utilization = {k: (v / denom if denom > 0 else 0.0) for k, v in busy.items()}
    return SimulationResult(
        makespan_s=makespan,
        peak_bytes={str(k): int(v) for k, v in dict(raw.get("peak_bytes") or {}).items()},
        timeline=timeline,
        transfer_events=list(raw.get("transfer_events") or []),
        release_events=list(raw.get("release_events") or []),
        exposed_transfer_latency_s=float(raw.get("exposed_transfer_latency_s") or 0.0),
        resource_busy_s=busy,
        simulated=True,
        critical_path=[str(x) for x in list(raw.get("critical_path") or [])],
        bytes_read=int(raw.get("bytes_read") or 0),
        bytes_transferred=int(raw.get("bytes_transferred") or 0),
        instruction_count=int(raw.get("instruction_count") or instruction_count),
        resource_utilization=utilization,
        activation_peak_bytes=int(raw.get("activation_peak_bytes") or 0),
        initiation_interval_s=max(ii, 1e-12) if ii > 0 else 0.0,
    )


def simulate_schedule(schedule: Any, machine: ResourceGraph) -> SimulationResult:
    """Simulate an :class:`ExecutableSchedule` via the native Rust discrete-event walk."""
    from tensortorrent.errors import MemoryCapacityError
    from tensortorrent.native import require_native
    from tensortorrent.runtime.schedule import ExecutableSchedule

    if not isinstance(schedule, ExecutableSchedule):
        raise TypeError(f"simulate_schedule expects ExecutableSchedule, got {type(schedule).__name__}")

    native = require_native()
    try:
        raw = native.simulate_schedule(schedule, machine)
    except ValueError as exc:
        # Native DES fails closed on peak-memory oversubscription; surface as the
        # public capacity error so compile/force-GPU paths stay typed.
        msg = str(exc)
        if "infeasible" in msg.lower() or "memory" in msg.lower():
            raise MemoryCapacityError(msg) from exc
        raise
    return _result_from_raw(raw, instruction_count=len(schedule.instructions))


def simulate_schedules(
    schedules: list[Any],
    machine: ResourceGraph,
    *,
    workers: int = 0,
) -> list[SimulationResult | dict[str, Any]]:
    """Batch-simulate schedules (native Rayon). Preserves input order.

    Each entry is a :class:`SimulationResult` or a status dict for infeasible /
    invalid siblings. One failure does not discard the batch.
    """
    outcomes, _stats = simulate_schedules_with_stats(schedules, machine, workers=workers)
    return outcomes


def simulate_schedules_with_stats(
    schedules: list[Any],
    machine: ResourceGraph,
    *,
    workers: int = 0,
) -> tuple[list[SimulationResult | dict[str, Any]], dict[str, Any]]:
    """Batch DES with authoritative Rust parallelism statistics.

    Statistics keys:
      schedule_count, simulator_workers_requested, simulator_workers_available,
      simulator_workers_used, parallel_simulation_used
    """
    from tensortorrent.native import require_native
    from tensortorrent.runtime.schedule import ExecutableSchedule

    for schedule in schedules:
        if not isinstance(schedule, ExecutableSchedule):
            raise TypeError(f"simulate_schedules expects ExecutableSchedule, got {type(schedule).__name__}")
    if not schedules:
        return [], {
            "schedule_count": 0,
            "simulator_workers_requested": int(workers),
            "simulator_workers_available": 1,
            "simulator_workers_used": 1,
            "parallel_simulation_used": False,
        }
    native = require_native()
    payload = native.simulate_schedules(schedules, machine, workers)
    if not isinstance(payload, dict) or "outcomes" not in payload:
        raise TypeError(
            "native.simulate_schedules must return "
            "{'outcomes': [...], 'statistics': {...}}; "
            f"got {type(payload).__name__}"
        )
    raw_list = list(payload.get("outcomes") or [])
    stats = dict(payload.get("statistics") or {})
    out: list[SimulationResult | dict[str, Any]] = []
    for i, raw in enumerate(raw_list):
        status = closed_str(raw.get("status") or SimulationStatus.VALID.value)
        if status == SimulationStatus.VALID and "makespan_s" in raw:
            out.append(_result_from_raw(raw, instruction_count=len(schedules[i].instructions)))
        else:
            out.append(dict(raw))
    return out, stats
