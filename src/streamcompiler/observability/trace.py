"""Chrome-tracing / HTML plan visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from streamcompiler.planner.maximal import ExecutionPlan
from streamcompiler.simulator.discrete_event import SimulationResult


def plan_to_chrome_trace(plan: ExecutionPlan, sim: SimulationResult) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for item in sim.timeline:
        events.append(
            {
                "name": item["region"],
                "cat": "compute",
                "ph": "X",
                "ts": item["start_s"] * 1e6,
                "dur": max(1.0, (item["end_s"] - item["start_s"]) * 1e6),
                "pid": item["device"],
                "tid": item["backend"],
                "args": {"dtype": item["dtype"]},
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
        },
    }


def write_chrome_trace(plan: ExecutionPlan, sim: SimulationResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan_to_chrome_trace(plan, sim), indent=2), encoding="utf-8")
    return path
