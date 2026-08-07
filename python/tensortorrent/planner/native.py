"""Convert planning inputs and invoke the native Rust planner."""

from __future__ import annotations

from typing import Any

from tensortorrent.backends.base import KernelCandidate
from tensortorrent.config import CompileConfig
from tensortorrent.ir.graph import HeterogeneousGraph
from tensortorrent.ir.resource_graph import ComputeClass, ComputeResource, ResourceGraph
from tensortorrent.native import require_native


def device_capacity_bytes(
    machine: ResourceGraph,
    device_name: str,
    *,
    vram_budget_bytes: int | None,
) -> int:
    device = machine.compute.get(device_name)
    if device is None:
        return 0
    total = sum(
        max(0, int(machine.memory[name].allocatable_bytes)) for name in device.memory_affinity if name in machine.memory
    )
    if (
        device.compute_class
        in {
            ComputeClass.DISCRETE_GPU,
            ComputeClass.INTEGRATED_GPU,
            ComputeClass.ACCELERATOR,
        }
        and vram_budget_bytes is not None
    ):
        return min(total, vram_budget_bytes) if total > 0 else vram_budget_bytes
    return total


def device_memory_name(machine: ResourceGraph, device_name: str) -> str:
    device = machine.compute.get(device_name)
    if device is None:
        return device_name
    for name in device.memory_affinity:
        if name in machine.memory:
            return name
    return device_name


def dependency_edge_bytes(graph: HeterogeneousGraph) -> dict[tuple[str, str], int]:
    producer: dict[str, str] = {}
    for region in graph.compute_regions():
        for output in region.outputs:
            producer[output] = region.name

    edges: dict[tuple[str, str], int] = {}
    for region in graph.compute_regions():
        for input_name in region.inputs:
            source = producer.get(input_name)
            if source is None or source == region.name:
                continue
            meta = graph.tensors.get(input_name)
            edges[(source, region.name)] = edges.get((source, region.name), 0) + max(
                0, int(getattr(meta, "size_bytes", 0) or 0)
            )
    return edges


def consumer_counts(graph: HeterogeneousGraph) -> dict[str, int]:
    counts = {region.name: 0 for region in graph.compute_regions()}
    for region in graph.compute_regions():
        for dep in tuple(str(x) for x in region.attributes.get("depends_on", ())):
            if dep in counts:
                counts[dep] += 1
    producer: dict[str, str] = {}
    for region in graph.compute_regions():
        for output in region.outputs:
            producer[output] = region.name
    for output in graph.outputs:
        source = producer.get(output)
        if source is not None:
            counts[source] = counts.get(source, 0) + 1
    return counts


def topological_order(graph: HeterogeneousGraph) -> list[str] | None:
    regions = graph.compute_regions()
    declared_order = [region.name for region in regions]
    known = set(declared_order)
    dependencies = {
        region.name: tuple(str(dep) for dep in region.attributes.get("depends_on", ()) if str(dep) in known)
        for region in regions
    }
    remaining = set(declared_order)
    order: list[str] = []
    while remaining:
        ready = [
            region_id
            for region_id in declared_order
            if region_id in remaining and all(dep not in remaining for dep in dependencies[region_id])
        ]
        if not ready:
            return None
        for region_id in ready:
            order.append(region_id)
            remaining.remove(region_id)
    return order


def placements_from_native(finalist: dict[str, Any]) -> list[Any]:
    """Decode a native finalist dict into ``Placement`` rows."""
    from tensortorrent.planner.maximal import Placement

    placements: list[Any] = []
    for raw in finalist.get("placements") or []:
        placements.append(
            Placement(
                region_id=str(raw["region_id"]),
                device=str(raw["device"]),
                backend_id=str(raw["backend_id"]),
                dtype=str(raw["dtype"]),
                kernel_id=str(raw["kernel_id"]),
                estimated_latency_s=float(raw.get("estimated_latency_s") or 0.0),
                depends_on=tuple(str(d) for d in (raw.get("depends_on") or ())),
                measured=bool(raw.get("measured", False)),
                output_bytes=int(raw.get("output_bytes") or 0),
                state_bytes=int(raw.get("state_bytes") or 0),
                workspace_bytes=int(raw.get("workspace_bytes") or 0),
            )
        )
    return placements


def build_planning_problem(
    graph_ir: HeterogeneousGraph,
    machine: ResourceGraph,
    region_candidates: dict[str, list[KernelCandidate]],
    subsets: list[tuple[ComputeResource, ...]],
    byte_counts: dict[str, tuple[int, int]],
    config: CompileConfig,
) -> dict[str, Any] | None:
    """Build the compact dict consumed by ``native.plan_placements``."""
    order_names = topological_order(graph_ir)
    if order_names is None:
        return None
    name_to_idx = {name: i for i, name in enumerate(order_names)}
    # Regions in order-index space; candidates aligned by region index.
    regions_by_name = {r.name: r for r in graph_ir.compute_regions()}
    consumers = consumer_counts(graph_ir)
    edge_named = dependency_edge_bytes(graph_ir)

    # Stable device index space: union of all subset devices, sorted.
    device_names_set: set[str] = set()
    for subset in subsets:
        for device in subset:
            device_names_set.add(device.id.name)
    device_names = sorted(device_names_set)
    device_to_idx = {n: i for i, n in enumerate(device_names)}

    regions = []
    candidates: list[list[dict[str, Any]]] = []
    for name in order_names:
        region = regions_by_name[name]
        deps = [
            name_to_idx[dep] for dep in (str(d) for d in region.attributes.get("depends_on", ())) if dep in name_to_idx
        ]
        out_b, state_b = byte_counts.get(name, (0, 0))
        regions.append(
            {
                "name": name,
                "depends_on": deps,
                "output_bytes": int(out_b),
                "state_bytes": int(state_b),
                "consumer_count": int(consumers.get(name, 0)),
            }
        )
        pool = []
        for cand in region_candidates.get(name, []):
            if cand.device not in device_to_idx:
                continue
            pool.append(
                {
                    "device": device_to_idx[cand.device],
                    "backend_id": cand.backend_id,
                    "kernel_id": cand.kernel_id,
                    "dtype": cand.dtype,
                    "estimated_latency_s": float(cand.estimated_latency_s or 0.0),
                    "workspace_bytes": int(cand.workspace_bytes or 0),
                    "measured": bool(cand.attributes.get("measured", False)),
                }
            )
        candidates.append(pool)

    edge_bytes = [
        (name_to_idx[src], name_to_idx[dst], nbytes)
        for (src, dst), nbytes in edge_named.items()
        if src in name_to_idx and dst in name_to_idx
    ]

    capacities = [
        device_capacity_bytes(machine, name, vram_budget_bytes=config.vram_budget_bytes) for name in device_names
    ]
    device_memory = [device_memory_name(machine, name) for name in device_names]

    subset_specs = []
    for subset in subsets:
        indices = sorted(device_to_idx[d.id.name] for d in subset if d.id.name in device_to_idx)
        if indices:
            subset_specs.append(indices)

    weights = config.objective_weights
    machine.allow_host_staged_transfers = bool(config.allow_host_staged_transfers)

    return {
        "config": {
            "objective": config.objective.value,
            "weight_latency": float(weights.get("latency", 0.0)),
            "weight_throughput": float(weights.get("throughput", 0.0)),
            "weight_memory": float(weights.get("memory", 0.0)),
            "beam_width": int(config.planner_beam_width),
            "candidates_per_device": int(config.planner_candidates_per_device),
            "local_search_iters": int(config.planner_local_search_iters),
            "target_inflight_requests": int(config.target_inflight_requests),
            "allow_host_staged_transfers": bool(config.allow_host_staged_transfers),
            "vram_budget_bytes": config.vram_budget_bytes,
            "planner_workers": int(config.planner_workers),
            "allow_parallel_subsets": bool(config.planner_parallel_subsets),
            "finalist_count": int(config.planner_des_candidates),
        },
        "machine": machine,
        "device_names": device_names,
        "capacities": capacities,
        "device_memory": device_memory,
        "regions": regions,
        "order": list(range(len(order_names))),
        "candidates": candidates,
        "edge_bytes": edge_bytes,
        "subsets": subset_specs,
    }


def run_native_planner(problem: dict[str, Any]) -> dict[str, Any]:
    native = require_native()
    out = native.plan_placements(problem)
    if not isinstance(out, dict):
        raise TypeError(f"plan_placements returned {type(out).__name__}, expected dict")
    return out
