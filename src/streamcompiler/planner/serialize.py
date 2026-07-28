"""Serialize and inspect execution plans."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from streamcompiler.planner.maximal import ExecutionPlan, Placement, ResourceDecision


def plan_to_dict(plan: ExecutionPlan) -> dict:
    return {
        "graph_name": plan.graph_name,
        "fingerprint": plan.fingerprint,
        "objective": plan.objective,
        "placements": [asdict(p) for p in plan.placements],
        "decisions": [asdict(d) for d in plan.decisions],
        "devices_used": list(plan.devices_used),
        "communication_backend": plan.communication_backend,
        "predicted_latency_s": plan.predicted_latency_s,
        "predicted_peak_bytes": plan.predicted_peak_bytes,
        "strategy": plan.strategy,
        "notes": plan.notes,
    }


def write_plan(plan: ExecutionPlan, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan_to_dict(plan), indent=2), encoding="utf-8")
    return path


def load_plan(path: Path) -> ExecutionPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    placements = [Placement(**p) for p in payload["placements"]]
    decisions = [ResourceDecision(**d) for d in payload["decisions"]]
    return ExecutionPlan(
        graph_name=payload["graph_name"],
        fingerprint=payload["fingerprint"],
        objective=payload["objective"],
        placements=placements,
        decisions=decisions,
        devices_used=tuple(payload["devices_used"]),
        communication_backend=payload["communication_backend"],
        predicted_latency_s=payload["predicted_latency_s"],
        predicted_peak_bytes=payload.get("predicted_peak_bytes", {}),
        strategy=payload.get("strategy", ""),
        notes=list(payload.get("notes", [])),
    )
