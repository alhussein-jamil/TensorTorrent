"""Maximal heterogeneous planner.

Searches subsets and combinations of the machine. A device participates only
when it improves the selected objective. Vendor-specific logic stays behind
backend capability queries.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations

from tensortorrent.backends import backend_by_id
from tensortorrent.backends.base import KernelCandidate
from tensortorrent.backends.communication import select_communication_backend
from tensortorrent.compile.measure import MeasurementSet
from tensortorrent.config import CompileConfig, Objective
from tensortorrent.errors import PlanningError
from tensortorrent.ir.graph import HeterogeneousGraph, Instruction
from tensortorrent.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    ResourceDecision,
    ResourceGraph,
)

logger = logging.getLogger(__name__)


@dataclass
class Placement:
    region_id: str
    device: str
    backend_id: str
    dtype: str
    kernel_id: str
    estimated_latency_s: float
    depends_on: tuple[str, ...] = ()
    measured: bool = False
    """True when ``estimated_latency_s`` came from running the region, not a prior."""
    output_bytes: int = 0
    """Bytes this region writes, summed from the lowered tensor metadata."""
    state_bytes: int = 0
    """Parameter and buffer bytes the region reads."""
    workspace_bytes: int = 0
    """Backend-declared temporary workspace charged during region execution."""

    @property
    def working_set_bytes(self) -> int:
        return self.output_bytes + self.state_bytes + self.workspace_bytes


@dataclass
class ExecutionPlan:
    graph_name: str
    fingerprint: str
    objective: str
    placements: list[Placement]
    decisions: list[ResourceDecision]
    devices_used: tuple[str, ...]
    communication_backend: str
    predicted_latency_s: float
    predicted_peak_bytes: dict[str, int] = field(default_factory=dict)
    predicted_throughput_per_s: float = 0.0
    predicted_transfer_bytes: int = 0
    predicted_transfer_latency_s: float = 0.0
    prefetch_distance: int = 0
    search_statistics: dict[str, object] = field(default_factory=dict)
    strategy: str = ""
    notes: list[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = [
            f"plan for {self.graph_name}",
            f"objective: {self.objective}",
            f"strategy: {self.strategy}",
            f"predicted_latency_s: {self.predicted_latency_s:.6f}",
            f"predicted_throughput_per_s: {self.predicted_throughput_per_s:.3f}",
            f"predicted_transfer_bytes: {self.predicted_transfer_bytes}",
            f"prefetch_distance: {self.prefetch_distance}",
            f"devices_used: {', '.join(self.devices_used) or '(none)'}",
            f"communication: {self.communication_backend}",
            "resource decisions:",
        ]
        for d in self.decisions:
            flag = "SELECTED" if d.selected else "EXCLUDED"
            lines.append(f"  {flag} {d.resource}: {d.reason}")
        lines.append("placements:")
        for p in self.placements:
            tag = "measured" if p.measured else "prior"
            lines.append(
                f"  {p.region_id} -> {p.device} [{p.backend_id}/{p.dtype}] ~{p.estimated_latency_s:.6f}s ({tag})"
            )
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)


def _eligible_compute(graph: ResourceGraph, config: CompileConfig) -> list[ComputeResource]:
    out: list[ComputeResource] = []
    for device in graph.compute.values():
        if device.compute_class == ComputeClass.COPY_ENGINE:
            continue
        if device.compute_class == ComputeClass.CPU_SOCKET:
            # Prefer NUMA pools for placement; keep sockets for reporting.
            continue
        if device.compute_class in (ComputeClass.CPU_NUMA_POOL,) and not config.allow_cpu:
            continue
        # Real vendor GPUs (CUDA/ROCm/XPU) use DISCRETE_GPU / INTEGRATED_GPU.
        # ACCELERATOR is reserved for capability-gated mocks/plugins that must
        # remain placeable on CPU-only configs when present in the resource graph.
        if device.compute_class in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU) and not config.allow_gpu:
            continue
        if device.compute_class == ComputeClass.INTEGRATED_GPU and not config.allow_integrated_gpu:
            continue
        out.append(device)
    if not config.allow_mixed_vendor:
        gpu_vendors = {
            (d.vendor or d.backend_id)
            for d in out
            if d.compute_class
            in (
                ComputeClass.DISCRETE_GPU,
                ComputeClass.INTEGRATED_GPU,
                ComputeClass.ACCELERATOR,
            )
        }
        if len(gpu_vendors) > 1:
            # Keep first vendor only when mixed vendors disabled.
            keep_vendor = sorted(gpu_vendors)[0]
            out = [
                d
                for d in out
                if d.compute_class
                not in (
                    ComputeClass.DISCRETE_GPU,
                    ComputeClass.INTEGRATED_GPU,
                    ComputeClass.ACCELERATOR,
                )
                or (d.vendor or d.backend_id) == keep_vendor
            ]
    return out


def _region_candidates(
    graph_ir: HeterogeneousGraph,
    devices: list[ComputeResource],
    measurements: MeasurementSet | None = None,
) -> dict[str, list[KernelCandidate]]:
    """Enumerate per-device kernels for every real compute region.

    Latencies come from ``measurements`` whenever the region was actually run on
    that device. Regions without a measurement fall back to a clearly labelled
    prior and set ``attributes["measured"] = False``.
    """
    regions = graph_ir.compute_regions()
    if not regions:
        raise PlanningError(
            "IR contains no compute regions; nothing can be planned. "
            "This indicates a lowering failure rather than a hardware limitation."
        )

    by_region: dict[str, list[KernelCandidate]] = {}
    for region in regions:
        cands: list[KernelCandidate] = []
        for device in devices:
            backend = backend_by_id(device.backend_id)
            if backend is None:
                continue
            for cand in backend.enumerate_kernels(region, device):
                measurement = measurements.get(region.name, device.id.name) if measurements else None
                if (
                    measurement is not None
                    and measurement.latency_s < float("inf")
                    and (measurement.measured or measurement.simulated)
                ):
                    latency = measurement.latency_s
                    measured = bool(measurement.measured)
                    attributes = dict(cand.attributes)
                    attributes["measured"] = measured
                    attributes["simulated"] = bool(measurement.simulated)
                else:
                    latency = _scaled_prior(region, device, cand.dtype, measurements)
                    measured = False
                    attributes = dict(cand.attributes)
                    attributes["measured"] = False
                    attributes["simulated"] = False
                cands.append(
                    KernelCandidate(
                        region_id=cand.region_id,
                        device=cand.device,
                        backend_id=cand.backend_id,
                        kernel_id=cand.kernel_id,
                        dtype=cand.dtype,
                        estimated_latency_s=latency,
                        workspace_bytes=cand.workspace_bytes,
                        attributes=attributes,
                    )
                )
        by_region[region.name] = cands
    return by_region


def _scaled_prior(
    region: Instruction,
    device: ComputeResource,
    dtype: str,
    measurements: MeasurementSet | None,
) -> float:
    """Prior for a device that was never measured for this region.

    When the region *was* measured somewhere, the prior scales that real number by
    the declared relative device speed instead of using an absolute constant. This
    keeps unmeasured estimates anchored to observed work.

    With no measurement, relative device ratios scale a measured host CPU region
    prior (GEMM sample) — never treat the ratio table as absolute seconds.

    The host GEMM sample is ~one Linear worth of work. Unmeasured regions scale
    that sample by ``node_count`` (FX ops in the region) so a giant fused model
    is not priced like a 64×64 Linear (which previously made CPU-only look like
    tens of microseconds and excluded every GPU).
    """
    reference = measurements.best_usable(region.name) if measurements else None
    ratio = _relative_device_cost(device, dtype) / max(1e-12, _CPU_REFERENCE_COST)
    work = _region_prior_work_units(region)
    if reference is None:
        from tensortorrent.planner.cost.calibration import host_cpu_region_prior_s

        return max(1e-7, host_cpu_region_prior_s() * ratio * work)
    return reference.latency_s * ratio


def _region_prior_work_units(region: Instruction) -> float:
    """Relative work vs the calibrated host Linear prior (≥1)."""
    nodes = region.attributes.get("node_count")
    try:
        count = int(nodes) if nodes is not None else 0
    except (TypeError, ValueError):
        count = 0
    if count < 1:
        # Fall back to input arity when lowering omitted node_count.
        count = max(1, len(region.inputs))
    return float(max(1, count))


#: Declared relative device costs used only when a region was never measured on a
#: device. They are ratios against ``_CPU_REFERENCE_COST``, not latency claims.
_RELATIVE_DEVICE_COST: dict[ComputeClass, float] = {
    ComputeClass.DISCRETE_GPU: 0.002,
    ComputeClass.INTEGRATED_GPU: 0.004,
    ComputeClass.CPU_NUMA_POOL: 0.02,
    ComputeClass.CPU_SOCKET: 0.02,
    ComputeClass.ACCELERATOR: 0.003,
    ComputeClass.COPY_ENGINE: 0.001,
}
_CPU_REFERENCE_COST = _RELATIVE_DEVICE_COST[ComputeClass.CPU_NUMA_POOL]


def _relative_device_cost(device: ComputeResource, dtype: str) -> float:
    """Declared, unmeasured relative cost. Never reported as a benchmark."""
    base = _RELATIVE_DEVICE_COST.get(device.compute_class, 0.05)
    # Prefer BF16/FP16 on accelerators when listed as supported.
    if device.compute_class in (
        ComputeClass.DISCRETE_GPU,
        ComputeClass.INTEGRATED_GPU,
        ComputeClass.ACCELERATOR,
    ):
        if dtype == "bfloat16" and device.supports_dtype("bfloat16"):
            base *= 0.7
        elif dtype == "float16" and device.supports_dtype("float16"):
            base *= 0.75
        elif dtype == "float32":
            base *= 1.0
    else:
        if dtype == "float32":
            base *= 0.9
        elif dtype in ("float16", "bfloat16"):
            base *= 1.1  # CPU soft-float paths often slower
    # Scale mildly by inverse core count when known.
    if device.core_count > 0:
        base *= 32.0 / max(8.0, float(device.core_count))
    return base


def _device_subsets(devices: list[ComputeResource], limit: int = 32) -> list[tuple[ComputeResource, ...]]:
    """Enumerate useful resource subsets, ranked rather than truncated by discovery order.

    Small machines are searched exhaustively. Larger machines retain every singleton,
    CPU pool, all-accelerator set, accelerator+local-CPU set, and the highest-capacity
    combinations up to the configured limit.
    """
    if not devices:
        return []

    def rank(device: ComputeResource) -> tuple[float, int, str]:
        speed = 1.0 / max(1e-12, _relative_device_cost(device, next(iter(device.supported_dtypes), "float32")))
        return (-speed, -max(0, device.core_count), device.id.name)

    ordered = sorted(devices, key=rank)
    cpus = [d for d in ordered if d.compute_class == ComputeClass.CPU_NUMA_POOL]
    accelerators = [
        d
        for d in ordered
        if d.compute_class in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU, ComputeClass.ACCELERATOR)
    ]
    candidates: list[tuple[ComputeResource, ...]] = []

    # Exhaustive search is practical on ordinary workstations.
    if len(ordered) <= 6:
        for size in range(1, len(ordered) + 1):
            candidates.extend(combinations(ordered, size))
    else:
        candidates.extend((device,) for device in ordered)
        if cpus:
            candidates.append(tuple(cpus))
        if accelerators:
            candidates.append(tuple(accelerators))
        if accelerators and cpus:
            candidates.append(tuple(accelerators) + (cpus[0],))
            candidates.append(tuple(accelerators) + tuple(cpus))
        for size in (2, 3, 4):
            if len(accelerators) >= size:
                candidates.extend(combinations(accelerators, size))
        # Include mixed accelerator/CPU subsets so CPU-efficient tails are not lost.
        for accelerator in accelerators[: min(8, len(accelerators))]:
            for cpu in cpus[: min(2, len(cpus))]:
                candidates.append((accelerator, cpu))

    def subset_rank(subset: tuple[ComputeResource, ...]) -> tuple[int, float, int, tuple[str, ...]]:
        names = tuple(sorted(device.id.name for device in subset))
        has_accel = any(device in accelerators for device in subset)
        aggregate_speed = sum(
            1.0 / max(1e-12, _relative_device_cost(device, next(iter(device.supported_dtypes), "float32")))
            for device in subset
        )
        # Prefer diverse multi-device candidates, then stronger aggregate compute.
        return (0 if has_accel else 1, -aggregate_speed, -len(subset), names)

    seen: set[tuple[str, ...]] = set()
    unique: list[tuple[ComputeResource, ...]] = []
    for subset in sorted(candidates, key=subset_rank):
        key = tuple(sorted(device.id.name for device in subset))
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(tuple(sorted(subset, key=lambda device: device.id.name)))
        if len(unique) >= max(1, limit):
            break
    return unique


def _score_plan(latency_s: float, config: CompileConfig, *, peak_working_set_bytes: int = 0) -> float:
    """Plan score where **lower is always better**.

    Throughput for this single-batch planner is the reciprocal of makespan, so
    maximizing requests/s is exactly minimizing predicted latency. Memory
    objective minimizes a per-device peak working-set estimate (max region on
    each device, summed across devices), with latency as a tiny tie-break.
    """
    if config.objective == Objective.MEMORY:
        return float(max(0, peak_working_set_bytes)) + 1e-9 * latency_s
    if config.objective in (Objective.LATENCY, Objective.THROUGHPUT, Objective.BALANCED):
        return latency_s
    weights = config.objective_weights
    return weights.get("latency", 1.0) * latency_s + weights.get("memory", 0.0) * float(max(0, peak_working_set_bytes))


def _peak_working_set_bytes(placements: list[Placement]) -> int:
    """Crude peak: largest region per device, summed across devices."""
    per_device: dict[str, int] = {}
    for placement in placements:
        per_device[placement.device] = max(
            per_device.get(placement.device, 0),
            placement.working_set_bytes,
        )
    return sum(per_device.values())


def _device_memory_bytes(
    device: ComputeResource,
    machine: ResourceGraph | None = None,
    *,
    vram_budget_bytes: int | None = None,
) -> int:
    if machine is None:
        return 0
    total = 0
    for name in device.memory_affinity:
        mem = machine.memory.get(name)
        if mem is not None:
            total += mem.allocatable_bytes
    if vram_budget_bytes is not None and device.compute_class in (
        ComputeClass.DISCRETE_GPU,
        ComputeClass.INTEGRATED_GPU,
        ComputeClass.ACCELERATOR,
    ):
        total = min(total, vram_budget_bytes) if total > 0 else vram_budget_bytes
    return total


def region_byte_counts(graph_ir: HeterogeneousGraph) -> dict[str, tuple[int, int]]:
    """Output and state bytes per compute region, taken from the lowered IR.

    Used for memory pressure and peak estimates, which previously assumed a flat
    1 MiB per region regardless of the model. Shared weights (same alias/storage)
    count once per region.
    """
    counts: dict[str, tuple[int, int]] = {}
    for region in graph_ir.compute_regions():
        outputs = 0
        for name in region.outputs:
            meta = graph_ir.tensors.get(name)
            if meta is not None:
                outputs += int(meta.size_bytes)
        state = 0
        seen_storage: set[str] = set()
        for name in region.inputs:
            meta = graph_ir.tensors.get(name)
            if meta is None or meta.kind not in ("parameter", "buffer", "constant"):
                continue
            key = meta.alias_group or meta.storage_id or name
            if key in seen_storage:
                continue
            seen_storage.add(key)
            state += int(meta.size_bytes)
        counts[region.name] = (outputs, state)
    return counts


def _decide_resources(
    graph: ResourceGraph,
    eligible: list[ComputeResource],
    used: set[str],
    best_latency: float,
    solo_latencies: dict[str, float],
) -> list[ResourceDecision]:
    decisions: list[ResourceDecision] = []
    for device in eligible:
        name = device.id.name
        solo = solo_latencies.get(name)
        if name in used:
            benefit = None if solo is None else max(0.0, solo - best_latency)
            if benefit is not None and benefit > 0 and solo is not None:
                solo_s = solo
                reason = (
                    f"{name} selected because it reduced predicted critical-path latency "
                    f"by {benefit * 1e3:.3f} ms versus running alone on this device "
                    f"(solo={solo_s * 1e3:.3f} ms, plan={best_latency * 1e3:.3f} ms)"
                )
            elif device.compute_class in (
                ComputeClass.DISCRETE_GPU,
                ComputeClass.INTEGRATED_GPU,
                ComputeClass.ACCELERATOR,
            ):
                reason = (
                    f"{name} selected as the fastest measured/prior backend for critical regions "
                    f"(plan={best_latency * 1e3:.3f} ms)"
                )
            elif device.compute_class == ComputeClass.CPU_NUMA_POOL:
                reason = (
                    f"{name} selected; CPU-efficient work stays on the host critical path "
                    f"(plan={best_latency * 1e3:.3f} ms)"
                )
            else:
                reason = f"{name} selected by objective improvement (plan={best_latency * 1e3:.3f} ms)"
            decisions.append(
                ResourceDecision(
                    resource=name,
                    selected=True,
                    reason=reason,
                    estimated_benefit_s=benefit,
                    estimated_cost_s=None,
                )
            )
        else:
            cost = None if solo is None else max(0.0, (solo or 0) - best_latency)
            if (
                device.compute_class in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU)
                and solo is not None
                and cost is not None
            ):
                # Solo on this GPU vs best multi-device plan: positive cost means
                # adding it alone would be slower than the chosen plan.
                delta_ms = (solo - best_latency) * 1e3
                if delta_ms > 0:
                    reason = (
                        f"{name} excluded because using it alone predicts "
                        f"{solo * 1e3:.3f} ms versus the chosen plan at "
                        f"{best_latency * 1e3:.3f} ms "
                        f"(+{delta_ms:.3f} ms on the critical path)"
                    )
                else:
                    reason = (
                        f"{name} excluded; combining it did not beat the chosen plan at {best_latency * 1e3:.3f} ms"
                    )
            elif device.compute_class == ComputeClass.CPU_NUMA_POOL and solo is not None:
                reason = (
                    f"{name} excluded because NUMA-remote / alternate-pool placement "
                    f"predicts {solo * 1e3:.3f} ms versus chosen plan "
                    f"{best_latency * 1e3:.3f} ms"
                )
            else:
                reason = (
                    f"{name} excluded; participation did not improve the selected objective "
                    f"(plan={best_latency * 1e3:.3f} ms)"
                )
            decisions.append(
                ResourceDecision(
                    resource=name,
                    selected=False,
                    reason=reason,
                    estimated_benefit_s=None,
                    estimated_cost_s=cost,
                )
            )
    # Also report copy engines / storage when present.
    for device in graph.compute.values():
        if device.compute_class == ComputeClass.COPY_ENGINE:
            decisions.append(
                ResourceDecision(
                    resource=device.id.name,
                    selected=True,
                    reason="copy engine retained to overlap weight/activation movement with compute",
                )
            )
    return decisions


def _subset_passes_vendor_filter(subset: tuple[ComputeResource, ...], config: CompileConfig) -> bool:
    vendors = {
        (device.vendor or device.backend_id)
        for device in subset
        if device.compute_class in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU, ComputeClass.ACCELERATOR)
    }
    return len(vendors) <= 1 or config.allow_mixed_vendor


def _comparable_search_score(
    result: object,
    config: CompileConfig,
    machine: ResourceGraph,
) -> float:
    from tensortorrent.planner.search import SearchResult

    if not isinstance(result, SearchResult):
        raise TypeError(f"expected SearchResult, got {type(result).__name__}")
    peak_total = sum(result.peak_bytes.values())
    if config.objective == Objective.THROUGHPUT:
        return float(1.0 / max(result.throughput_per_s, 1e-12))
    if config.objective == Objective.MEMORY:
        return float(peak_total) + 1e-9 * float(result.latency_s)
    if config.objective == Objective.BALANCED:
        comparable = float(result.latency_s) + 1.0 / max(result.throughput_per_s, 1e-12)
        comparable += 0.05 * sum(
            result.peak_bytes.get(name, 0)
            / max(
                1,
                _device_memory_bytes(
                    machine.compute[name],
                    machine,
                    vram_budget_bytes=config.vram_budget_bytes,
                ),
            )
            for name in result.peak_bytes
            if name in machine.compute
        )
        return float(comparable)
    if config.objective == Objective.WEIGHTED:
        pressure = sum(
            result.peak_bytes.get(name, 0)
            / max(
                1,
                _device_memory_bytes(
                    machine.compute[name],
                    machine,
                    vram_budget_bytes=config.vram_budget_bytes,
                ),
            )
            for name in result.peak_bytes
            if name in machine.compute
        )
        return float(
            config.objective_weights.get("latency", 0.0) * result.latency_s
            + config.objective_weights.get("throughput", 0.0) * (1.0 / max(result.throughput_per_s, 1e-12))
            + config.objective_weights.get("memory", 0.0) * pressure
        )
    return float(result.latency_s)


def plan_execution(
    graph_ir: HeterogeneousGraph,
    machine: ResourceGraph,
    config: CompileConfig | None = None,
    measurements: MeasurementSet | None = None,
) -> ExecutionPlan:
    """Jointly select resources, kernels, transfers, and memory-feasible placement.

    Every candidate subset is solved with a bounded beam search that carries the
    dependency critical path, serialized copy paths, streamed state working sets,
    and activation lifetimes.
    """
    from tensortorrent.planner.search import search_placements

    config = config or CompileConfig()
    eligible = _eligible_compute(machine, config)
    if not eligible:
        raise PlanningError("No eligible compute resources discovered")

    region_candidates = _region_candidates(graph_ir, eligible, measurements)
    byte_counts = region_byte_counts(graph_ir)
    subsets = _device_subsets(eligible, limit=config.max_plan_candidates)
    empty_cand_regions = sorted(region for region, candidates in region_candidates.items() if not candidates)
    if empty_cand_regions:
        raise PlanningError(
            "No backend kernel candidates exist for one or more regions: "
            f"{empty_cand_regions}. Check backend supported_ops/dtypes and lowering."
        )

    # Singleton subset searches below already cover per-device solos; do not
    # run a separate solo pass (that duplicated beam search for every device).
    solo_latencies: dict[str, float] = {}
    solo_results: dict[str, object] = {}

    storage = [memory for memory in machine.memory.values() if memory.memory_class.value in {"nvme", "disk_cache"}]
    subset_diagnostics: list[str] = []
    best: ExecutionPlan | None = None
    best_score = float("inf")

    subset_work: list[tuple[tuple[ComputeResource, ...], set[str]]] = []
    for subset in subsets:
        names = {device.id.name for device in subset}
        if not _subset_passes_vendor_filter(subset, config):
            subset_diagnostics.append(f"subset=[{','.join(sorted(names))}] mixed_vendor_disallowed")
            continue
        subset_work.append((subset, names))

    from tensortorrent.planner.search import SearchResult

    parallel_subsets = config.planner_parallel_subsets and len(subset_work) >= 3 and len(region_candidates) >= 2
    subset_results: list[tuple[tuple[ComputeResource, ...], set[str], SearchResult | None]] = []

    if parallel_subsets:
        try:
            max_workers = min(len(subset_work), os.cpu_count() or 1)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_map = {
                    pool.submit(
                        search_placements,
                        graph_ir,
                        machine,
                        region_candidates,
                        names,
                        byte_counts,
                        config,
                    ): (subset, names)
                    for subset, names in subset_work
                }
                results_by_key: dict[tuple[str, ...], SearchResult | None] = {}
                for future in as_completed(future_map):
                    subset, names = future_map[future]
                    key = tuple(sorted(names))
                    results_by_key[key] = future.result()
                for subset, names in subset_work:
                    subset_results.append((subset, names, results_by_key.get(tuple(sorted(names)))))
        except Exception as exc:  # noqa: BLE001 - fall back to serial search
            logger.warning("parallel subset search failed (%s); falling back to serial", exc)
            parallel_subsets = False
            subset_results = []

    if not parallel_subsets:
        for subset, names in subset_work:
            result = search_placements(
                graph_ir,
                machine,
                region_candidates,
                names,
                byte_counts,
                config,
            )
            subset_results.append((subset, names, result))

    for subset, names, result in subset_results:
        if result is None:
            subset_diagnostics.append(f"subset=[{','.join(sorted(names))}] infeasible")
            continue
        if len(names) == 1:
            only = next(iter(names))
            solo_latencies[only] = result.latency_s
            solo_results[only] = result

        comparable = _comparable_search_score(result, config, machine)

        used = {placement.device for placement in result.placements}
        communication = select_communication_backend(tuple(sorted(used)))
        notes = [
            f"subset={','.join(sorted(names))}",
            _measurement_note(list(result.placements)),
            "planner=joint_beam_search",
            (
                f"planner_search expanded={result.states_expanded} pruned={result.states_pruned} "
                f"beam_width={result.beam_width} local_improvements={result.local_improvements}"
            ),
        ]
        if result.unmeasured_transfer_count:
            notes.append(
                f"unmeasured_transfer_priors={result.unmeasured_transfer_count}; validate links on target hardware"
            )
        if result.host_staged_transfer_count:
            notes.append(f"host_staged_transfers={result.host_staged_transfer_count}")
        if storage:
            notes.append(
                f"storage_resources_detected={len(storage)}; adaptive prefetch is bounded by the host RAM budget"
            )

        plan = ExecutionPlan(
            graph_name=graph_ir.name,
            fingerprint=machine.fingerprint,
            objective=config.objective.value,
            placements=list(result.placements),
            decisions=[],
            devices_used=tuple(sorted(used)),
            communication_backend=communication.backend_id,
            predicted_latency_s=result.latency_s,
            predicted_peak_bytes=dict(result.peak_bytes),
            predicted_throughput_per_s=result.throughput_per_s,
            predicted_transfer_bytes=result.transfer_bytes,
            predicted_transfer_latency_s=result.transfer_latency_s,
            strategy=_strategy_name(tuple(device for device in subset if device.id.name in used)),
            search_statistics={
                "states_expanded": result.states_expanded,
                "states_pruned": result.states_pruned,
                "beam_width": result.beam_width,
                "candidate_subsets": len(subsets),
                "target_inflight_requests": config.target_inflight_requests,
                "local_improvements": result.local_improvements,
                "parallel_subsets": parallel_subsets,
            },
            notes=notes,
        )
        if comparable < best_score:
            best_score = comparable
            best = plan

    if best is None:
        eligible_names = [device.id.name for device in eligible]
        capacities = ",".join(
            f"{device.id.name}={_device_memory_bytes(device, machine, vram_budget_bytes=config.vram_budget_bytes)}"
            for device in eligible
        )
        counts = ",".join(f"{region}={len(candidates)}" for region, candidates in region_candidates.items())
        details = "; ".join(subset_diagnostics) if subset_diagnostics else "no subsets tried"
        largest_region = max(
            ((region, sum(byte_counts.get(region, (0, 0)))) for region in region_candidates),
            key=lambda item: item[1],
            default=("", 0),
        )
        raise PlanningError(
            "Joint planner failed to produce a memory- and transfer-feasible plan. "
            f"eligible={eligible_names} vram_budget_bytes={config.vram_budget_bytes} "
            f"largest_region_working_set={largest_region} candidate_counts={{{counts}}} "
            f"device_capacity_bytes={{{capacities}}} subset_failures=[{details}]. "
            "If a single indivisible operator exceeds every device, lower it into smaller regions or enable an "
            "operation-specific sharded backend; region placement cannot split arbitrary operator semantics."
        )

    used_set = set(best.devices_used)
    best.decisions = _decide_resources(machine, eligible, used_set, best.predicted_latency_s, solo_latencies)
    for decision in best.decisions:
        if decision.selected and decision.resource in solo_results:
            solo = solo_results[decision.resource]
            throughput = getattr(solo, "throughput_per_s", 0.0)
            if throughput:
                decision.reason += f"; solo_throughput={throughput:.3f}/s"
    return best


def _measurement_note(placements: list[Placement]) -> str:
    measured = sum(1 for p in placements if p.measured)
    total = len(placements)
    if measured == total:
        return f"region_costs=measured ({measured}/{total} placements benchmarked on their device)"
    if measured == 0:
        return f"region_costs=priors_only (0/{total} placements benchmarked; run on target hardware)"
    return (
        f"region_costs=mixed ({measured}/{total} measured; the rest scaled from measured "
        "work by declared relative device speed)"
    )


def _strategy_name(subset: tuple[ComputeResource, ...]) -> str:
    cpus = [d for d in subset if d.compute_class == ComputeClass.CPU_NUMA_POOL]
    gpus = [
        d
        for d in subset
        if d.compute_class in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU, ComputeClass.ACCELERATOR)
    ]
    if gpus and cpus:
        return "pipeline_gpu_cpu"
    if len(gpus) > 1:
        return "multi_gpu"
    if len(gpus) == 1:
        return "single_gpu"
    if len(cpus) > 1:
        return "multi_numa_cpu"
    return "cpu_only"


def enumerate_plan_strategies() -> tuple[str, ...]:
    """Labels emitted by ``_strategy_name`` for published plans."""
    return (
        "cpu_only",
        "single_gpu",
        "multi_gpu",
        "multi_numa_cpu",
        "pipeline_gpu_cpu",
    )
