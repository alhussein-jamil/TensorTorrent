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
