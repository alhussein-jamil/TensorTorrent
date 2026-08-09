"""Extract GPU/CPU compute and transfer observability from CompiledModule runs."""

from __future__ import annotations

from typing import Any


def _backend_kind(token: str) -> str:
    t = (token or "").lower()
    if "cuda" in t or "rocm" in t or "xpu" in t or t.startswith("gpu"):
        return "gpu"
    if "cpu" in t or "host" in t:
        return "cpu"
    return "other"


def summarize_execution(compiled: Any) -> dict[str, Any]:
    """Build a machine-readable observability summary after ≥1 forward."""
    report = compiled.last_execution_report() if hasattr(compiled, "last_execution_report") else {}
    if not isinstance(report, dict):
        report = {}
    plan = compiled.specialized.plan
    profile = compiled.profile() if hasattr(compiled, "profile") else {}
    srep = getattr(getattr(compiled, "executor", None), "_last_schedule_report", None)

    region_rows = list(report.get("regions") or [])
    compute_by_kind: dict[str, float] = {"gpu": 0.0, "cpu": 0.0, "other": 0.0}
    regions_by_kind: dict[str, int] = {"gpu": 0, "cpu": 0, "other": 0}
    for row in region_rows:
        kind = _backend_kind(str(row.get("backend_id") or row.get("device") or ""))
        dur = float(row.get("duration_s") or 0.0)
        compute_by_kind[kind] = compute_by_kind.get(kind, 0.0) + dur
        regions_by_kind[kind] = regions_by_kind.get(kind, 0) + 1
    compute_total = sum(compute_by_kind.values()) or 1e-12

    transfer_bytes_h2d = 0
    transfer_bytes_d2h = 0
    transfer_bytes_other = 0
    transfer_count = 0
    transfer_time_s = 0.0
    compute_time_sched_s = 0.0
    compute_by_resource: dict[str, float] = {}
    spill_events = 0
    activation_bytes_written = 0
    activation_bytes_read = 0
    spill_latency_s = 0.0
    reload_latency_s = 0.0

    if srep is not None:
        activation_bytes_written = int(getattr(srep, "activation_bytes_written", 0) or 0)
        activation_bytes_read = int(getattr(srep, "activation_bytes_read", 0) or 0)
        spill_latency_s = float(getattr(srep, "spill_latency_s", 0.0) or 0.0)
        reload_latency_s = float(getattr(srep, "reload_latency_s", 0.0) or 0.0)
        spill_events = len(getattr(srep, "spill_events", None) or [])
        for ev in getattr(srep, "events", None) or []:
            opcode = str(getattr(ev, "opcode", "") or "")
            nbytes = int(getattr(ev, "nbytes", 0) or 0)
            dur = float(getattr(ev, "duration_s", 0.0) or 0.0)
            resource = str(getattr(ev, "resource", "") or "")
            notes = str(getattr(ev, "notes", "") or "").lower()
            if opcode == "Compute":
                compute_time_sched_s += dur
                compute_by_resource[resource] = compute_by_resource.get(resource, 0.0) + dur
            elif opcode in {"Transfer", "Prefetch", "Load", "Evict"}:
                transfer_count += 1
                transfer_time_s += dur
                direction = "other"
                if "h2d" in notes or "host" in notes and "device" in notes:
                    direction = "h2d"
                if resource.startswith("cuda") or resource.startswith("gpu"):
                    direction = "d2h" if "evict" in opcode.lower() or "d2h" in notes or "host" in resource else "h2d"
                if opcode == "Evict":
                    direction = "d2h"
                    transfer_bytes_d2h += nbytes
                elif opcode in {"Transfer", "Prefetch", "Load"}:
                    if _backend_kind(resource) == "gpu":
                        transfer_bytes_h2d += nbytes
                        direction = "h2d"
                    elif _backend_kind(resource) == "cpu":
                        transfer_bytes_d2h += nbytes
                        direction = "d2h"
                    else:
                        transfer_bytes_other += nbytes
                else:
                    transfer_bytes_other += nbytes
                _ = direction

    profile_transfers = {}
    if isinstance(profile, dict):
        profile_transfers = dict(profile.get("transfers") or {})
        residency = dict(profile.get("residency") or {})
    else:
        residency = {}

    sim = dict(profile.get("simulator") or {}) if isinstance(profile, dict) else {}
    param_store = dict(report.get("parameter_store") or {})
    if not param_store and hasattr(compiled, "executor"):
        store = getattr(compiled.executor, "parameter_store", None)
        if store is not None and hasattr(store, "stats"):
            param_store = dict(store.stats())

    wall = float(report.get("wall_time_s") or 0.0)
    transfer_frac = (transfer_time_s / wall) if wall > 0 else None

    predicted = {
        "latency_s": float(getattr(plan, "predicted_latency_s", 0.0) or 0.0),
        "peak_bytes": dict(getattr(plan, "predicted_peak_bytes", None) or {}),
        "transfer_bytes": int(getattr(plan, "predicted_transfer_bytes", 0) or 0),
        "transfer_latency_s": float(getattr(plan, "predicted_transfer_latency_s", 0.0) or 0.0),
    }

    return {
        "devices_used": list(plan.devices_used),
        "n_regions": len(region_rows) or len(getattr(plan, "placements", []) or []),
        "region_compute_s": compute_by_kind,
        "region_compute_fraction": {k: v / compute_total for k, v in compute_by_kind.items()},
        "regions_by_kind": regions_by_kind,
        "schedule_compute_s": compute_time_sched_s,
        "schedule_compute_by_resource": compute_by_resource,
        "transfer_count": transfer_count,
        "transfer_time_s": transfer_time_s,
        "transfer_bytes_h2d": transfer_bytes_h2d,
        "transfer_bytes_d2h": transfer_bytes_d2h,
        "transfer_bytes_other": transfer_bytes_other,
        "transfer_wall_fraction": transfer_frac,
        "wall_time_s": wall,
        "peak_activation_bytes": int(report.get("peak_activation_bytes") or 0),
        "allocation_peak_bytes": int(
            report.get("allocation_peak_bytes") or getattr(srep, "allocation_peak_bytes", 0) or 0
        ),
        "activation_bytes_written": activation_bytes_written,
        "activation_bytes_read": activation_bytes_read,
        "spill_events": spill_events,
        "spill_latency_s": spill_latency_s,
        "reload_latency_s": reload_latency_s,
        "parameter_store": param_store,
        "residency_profile": residency,
        "profile_transfers": profile_transfers,
        "simulator": {
            "peak_bytes": sim.get("peak_bytes"),
            "bytes_transferred": sim.get("bytes_transferred"),
            "exposed_transfer_latency_s": sim.get("exposed_transfer_latency_s"),
        },
        "predicted": predicted,
        "measured_vs_predicted": {
            "latency_ratio": (wall / predicted["latency_s"]) if predicted["latency_s"] > 0 and wall > 0 else None,
            "transfer_bytes_ratio": (
                (transfer_bytes_h2d + transfer_bytes_d2h) / predicted["transfer_bytes"]
                if predicted["transfer_bytes"] > 0
                else None
            ),
        },
        "plan_notes": list(plan.notes)[:24],
        "validation": dict(getattr(compiled.specialized, "validation", None) or {}),
    }
