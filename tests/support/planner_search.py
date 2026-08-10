"""Single-subset native placement search helper for tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tensortorrent.backends.base import KernelCandidate
from tensortorrent.config import CompileConfig
from tensortorrent.ir.graph import HeterogeneousGraph
from tensortorrent.ir.resource_graph import ResourceGraph
from tensortorrent.planner.native import build_planning_problem, placements_from_native, run_native_planner

if TYPE_CHECKING:
    from tensortorrent.planner.maximal import Placement

__all__ = ["SearchResult", "search_placements"]


@dataclass(frozen=True)
class SearchResult:
    placements: tuple[Placement, ...]
    latency_s: float
    throughput_per_s: float
    peak_bytes: dict[str, int]
    transfer_bytes: int
    transfer_latency_s: float
    unmeasured_transfer_count: int
    host_staged_transfer_count: int
    states_expanded: int
    states_pruned: int
    beam_width: int
    local_improvements: int


def search_placements(
    graph: HeterogeneousGraph,
    machine: ResourceGraph,
    region_candidates: dict[str, list[KernelCandidate]],
    allowed_devices: set[str],
    byte_counts: dict[str, tuple[int, int]],
    config: CompileConfig,
) -> SearchResult | None:
    """Run the native planner on one device subset."""
    devices = [machine.compute[name] for name in sorted(allowed_devices) if name in machine.compute]
    if not devices:
        return None
    problem = build_planning_problem(
        graph,
        machine,
        region_candidates,
        [tuple(devices)],
        byte_counts,
        config,
    )
    if problem is None:
        return None
    problem["config"]["allow_parallel_subsets"] = False
    problem["config"]["planner_workers"] = 1
    problem["config"]["finalist_count"] = 1
    out = run_native_planner(problem)
    finalists = list(out.get("finalists") or [])
    if not finalists:
        return None
    f = finalists[0]
    placements = tuple(placements_from_native(f))
    stats = out.get("statistics") or {}
    return SearchResult(
        placements=placements,
        latency_s=float(f.get("latency_s") or 0.0),
        throughput_per_s=float(f.get("throughput_per_s") or 0.0),
        peak_bytes={str(k): int(v) for k, v in dict(f.get("peak_bytes") or {}).items()},
        transfer_bytes=int(f.get("transfer_bytes") or 0),
        transfer_latency_s=float(f.get("transfer_latency_s") or 0.0),
        unmeasured_transfer_count=int(f.get("unmeasured_transfer_count") or 0),
        host_staged_transfer_count=int(f.get("host_staged_transfer_count") or 0),
        states_expanded=int(stats.get("states_expanded") or f.get("states_expanded") or 0),
        states_pruned=int(stats.get("states_pruned") or f.get("states_pruned") or 0),
        beam_width=int(stats.get("beam_width") or config.planner_beam_width),
        local_improvements=int(stats.get("local_improvements") or 0),
    )
