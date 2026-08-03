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

    Does **not** infer transfers inside the simulator. Cross-device movement must
    appear as Transfer instructions (via residency / schedule builder). Kept as a
    thin compatibility entry for tests that still hand-build ``ExecutionPlan``s.
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


def simulate_schedule(schedule: Any, machine: ResourceGraph) -> SimulationResult:
    """Simulate an :class:`ExecutableSchedule` via the native Rust discrete-event walk."""
    from tensortorrent.native import require_native
    from tensortorrent.runtime.schedule import ExecutableSchedule

    if not isinstance(schedule, ExecutableSchedule):
        raise TypeError(f"simulate_schedule expects ExecutableSchedule, got {type(schedule).__name__}")

    native = require_native()
    raw = native.simulate_schedule(schedule, machine)
    timeline = list(raw.get("timeline") or [])
    return SimulationResult(
        makespan_s=float(raw.get("makespan_s") or 0.0),
        peak_bytes={str(k): int(v) for k, v in dict(raw.get("peak_bytes") or {}).items()},
        timeline=timeline,
        transfer_events=list(raw.get("transfer_events") or []),
        release_events=list(raw.get("release_events") or []),
        exposed_transfer_latency_s=float(raw.get("exposed_transfer_latency_s") or 0.0),
        resource_busy_s={str(k): float(v) for k, v in dict(raw.get("resource_busy_s") or {}).items()},
        simulated=True,
        critical_path=[str(x) for x in list(raw.get("critical_path") or [])],
        bytes_read=int(raw.get("bytes_read") or 0),
        bytes_transferred=int(raw.get("bytes_transferred") or 0),
        instruction_count=int(raw.get("instruction_count") or len(schedule.instructions)),
        activation_peak_bytes=int(raw.get("activation_peak_bytes") or 0),
    )
