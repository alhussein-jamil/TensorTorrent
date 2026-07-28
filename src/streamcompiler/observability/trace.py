"""Chrome-tracing / HTML plan visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from streamcompiler.planner.maximal import ExecutionPlan
from streamcompiler.simulator.discrete_event import SimulationResult


def plan_to_chrome_trace(plan: ExecutionPlan, sim: SimulationResult) -> dict[str, Any]:
    """Build a Chrome trace from an analytic simulation (never claims measured)."""
    events: list[dict[str, Any]] = []
    for item in sim.timeline:
        kind = item.get("event", "compute")
        if kind == "compute" and "start_s" in item and "end_s" in item:
            events.append(
                {
                    "name": item["region"],
                    "cat": "compute",
                    "ph": "X",
                    "ts": item["start_s"] * 1e6,
                    "dur": max(1.0, (item["end_s"] - item["start_s"]) * 1e6),
                    "pid": item["device"],
                    "tid": item.get("backend", "compute"),
                    "args": {
                        "dtype": item.get("dtype"),
                        "simulated": True,
                        "working_set_bytes": item.get("working_set_bytes"),
                    },
                }
            )
        elif kind == "release":
            events.append(
                {
                    "name": f"release:{item.get('region')}",
                    "cat": "memory",
                    "ph": "i",
                    "ts": float(item.get("at_s", 0.0)) * 1e6,
                    "pid": item.get("memory", "memory"),
                    "tid": "release",
                    "s": "t",
                    "args": {
                        "nbytes": item.get("nbytes"),
                        "kind": item.get("kind"),
                        "simulated": True,
                    },
                }
            )
        elif kind == "prefetch_hint":
            events.append(
                {
                    "name": f"prefetch_hint:{item.get('region')}",
                    "cat": "storage",
                    "ph": "i",
                    "ts": 0.0,
                    "pid": item.get("device", "storage"),
                    "tid": "prefetch",
                    "s": "t",
                    "args": {"after": item.get("after"), "simulated": True},
                }
            )
        elif kind == "eviction_pressure":
            events.append(
                {
                    "name": f"eviction_pressure:{item.get('memory')}",
                    "cat": "memory",
                    "ph": "i",
                    "ts": float(item.get("at_s", 0.0)) * 1e6,
                    "pid": item.get("memory", "memory"),
                    "tid": "pressure",
                    "s": "t",
                    "args": {
                        "resident_bytes": item.get("resident_bytes"),
                        "allocatable_bytes": item.get("allocatable_bytes"),
                        "simulated": True,
                        "validated": False,
                    },
                }
            )
        elif kind == "transfer_landed":
            events.append(
                {
                    "name": f"landed:{item.get('source_region')}->{item.get('destination_region')}",
                    "cat": "memory",
                    "ph": "i",
                    "ts": float(item.get("at_s", 0.0)) * 1e6,
                    "pid": item.get("memory", "memory"),
                    "tid": "transfer_copy",
                    "s": "t",
                    "args": {"nbytes": item.get("nbytes"), "simulated": True},
                }
            )
    for transfer in sim.transfer_events:
        events.append(
            {
                "name": f"transfer:{transfer['source_region']}->{transfer['destination_region']}",
                "cat": "transfer",
                "ph": "X",
                "ts": transfer["start_s"] * 1e6,
                "dur": max(1.0, (transfer["end_s"] - transfer["start_s"]) * 1e6),
                "pid": transfer.get("link", "transfer"),
                "tid": "dma",
                "args": {
                    "nbytes": transfer.get("nbytes"),
                    "simulated": True,
                    "contention_factor": transfer.get("contention_factor"),
                    "source_device": transfer.get("source_device"),
                    "destination_device": transfer.get("destination_device"),
                },
            }
        )
    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "plan_objective": plan.objective,
            "strategy": plan.strategy,
            "devices_used": list(plan.devices_used),
            "predicted_latency_s": plan.predicted_latency_s,
            "simulated_makespan_s": sim.makespan_s,
            "exposed_transfer_latency_s": sim.exposed_transfer_latency_s,
            "simulated": sim.simulated,
        },
    }


def write_chrome_trace(plan: ExecutionPlan, sim: SimulationResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan_to_chrome_trace(plan, sim), indent=2), encoding="utf-8")
    return path


def report_to_chrome_trace(
    report: Any,
    *,
    plan: ExecutionPlan | None = None,
    residency_events: list[dict[str, Any]] | None = None,
    transfer_events: list[dict[str, Any]] | None = None,
    io_intervals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Chrome trace from a **measured** execution report (not simulated)."""
    events: list[dict[str, Any]] = []
    report_events = getattr(report, "events", ())
    t0 = min((e.start_s for e in report_events), default=0.0)
    wall_time_s = float(getattr(report, "wall_time_s", 0.0))
    for event in report_events:
        predicted = None
        if plan is not None:
            for placement in plan.placements:
                if placement.region_id == event.region_id:
                    predicted = placement.estimated_latency_s
                    break
        events.append(
            {
                "name": event.region_id,
                "cat": "compute",
                "ph": "X",
                "ts": (event.start_s - t0) * 1e6,
                "dur": max(1.0, event.duration_s * 1e6),
                "pid": event.device,
                "tid": event.worker,
                "args": {
                    "backend_id": event.backend_id,
                    "simulated": False,
                    "measured": True,
                    "predicted_duration_s": predicted,
                    "error_s": None if predicted is None else event.duration_s - predicted,
                },
            }
        )
    for item in transfer_events or ():
        start = float(item.get("start_s", 0.0))
        end = float(item.get("end_s", start))
        events.append(
            {
                "name": item.get("name", "transfer"),
                "cat": "transfer",
                "ph": "X",
                "ts": (start - t0) * 1e6,
                "dur": max(1.0, (end - start) * 1e6),
                "pid": item.get("resource", "transfer"),
                "tid": item.get("backend", "dma"),
                "args": {
                    "nbytes": item.get("nbytes"),
                    "simulated": bool(item.get("simulated", False)),
                    "measured": not bool(item.get("simulated", False)),
                    "cache_hit": item.get("cache_hit"),
                    "elided": item.get("elided"),
                },
            }
        )
    for item in io_intervals or ():
        start = float(item.get("start_s", 0.0))
        end = float(item.get("end_s", start))
        events.append(
            {
                "name": item.get("name", "read"),
                "cat": "storage",
                "ph": "X",
                "ts": (start - t0) * 1e6,
                "dur": max(1.0, (end - start) * 1e6),
                "pid": "storage",
                "tid": "pread",
                "args": {
                    "nbytes": item.get("nbytes"),
                    "simulated": False,
                    "measured": True,
                    "prefetch_hit": item.get("prefetch_hit"),
                    "cache_hit": item.get("cache_hit"),
                },
            }
        )
    for item in residency_events or ():
        events.append(
            {
                "name": f"{item.get('event')}:{item.get('tensor_id')}",
                "cat": "memory",
                "ph": "i",
                "ts": float(item.get("ts_s", 0.0)) * 1e6,
                "pid": "residency",
                "tid": str(item.get("event", "memory")),
                "s": "t",
                "args": {k: v for k, v in item.items() if k not in {"event", "ts_s"}},
            }
        )
    predicted = plan.predicted_latency_s if plan is not None else None
    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "simulated": False,
            "measured": True,
            "wall_time_s": wall_time_s,
            "peak_activation_bytes": getattr(report, "peak_activation_bytes", 0),
            "released_values": getattr(report, "released_values", 0),
            "predicted_latency_s": predicted,
            "prediction_error_s": None if predicted is None else wall_time_s - predicted,
            "devices_used": list(plan.devices_used) if plan is not None else [],
        },
    }


def write_execution_trace(
    report: Any,
    path: Path,
    *,
    plan: ExecutionPlan | None = None,
    residency_events: list[dict[str, Any]] | None = None,
    transfer_events: list[dict[str, Any]] | None = None,
    io_intervals: list[dict[str, Any]] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report_to_chrome_trace(
        report,
        plan=plan,
        residency_events=residency_events,
        transfer_events=transfer_events,
        io_intervals=io_intervals,
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_execution_timeline_html(
    report: Any,
    path: Path,
    *,
    plan: ExecutionPlan | None = None,
) -> Path:
    """Self-contained HTML timeline of measured region events."""
    rows = []
    report_events = getattr(report, "events", ())
    wall_time_s = float(getattr(report, "wall_time_s", 0.0))
    for event in sorted(report_events, key=lambda e: e.start_s):
        predicted = ""
        if plan is not None:
            for placement in plan.placements:
                if placement.region_id == event.region_id:
                    predicted = f"{placement.estimated_latency_s * 1e3:.3f}"
                    break
        rows.append(
            "<tr>"
            f"<td>{event.region_id}</td><td>{event.device}</td><td>{event.backend_id}</td>"
            f"<td>{event.worker}</td><td>{event.duration_s * 1e3:.3f}</td>"
            f"<td>{predicted}</td></tr>"
        )
    predicted_wall = f"{plan.predicted_latency_s * 1e3:.3f}" if plan is not None else "n/a"
    html = (
        "<html><body><h1>StreamCompiler measured execution</h1>"
        "<p><b>Timeline is measured runtime telemetry</b> "
        f"(simulated=False; wall={wall_time_s * 1e3:.3f} ms; "
        f"predicted={predicted_wall} ms).</p>"
        "<table border=1><tr><th>region</th><th>device</th><th>backend</th>"
        "<th>worker</th><th>measured_ms</th><th>predicted_ms</th></tr>" + "".join(rows) + "</table></body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
