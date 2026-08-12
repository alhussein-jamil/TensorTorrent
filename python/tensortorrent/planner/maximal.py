"""Maximal heterogeneous planner.

Searches subsets and combinations of the machine. A device participates only
when it improves the selected objective. Vendor-specific logic stays behind
backend capability queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from tensortorrent.backends import backend_by_id
from tensortorrent.backends.base import KernelCandidate
from tensortorrent.backends.communication import select_communication_backend
from tensortorrent.closed import TensorKind, closed_str
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
    objective: Objective
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
    finalist_plans: list[ExecutionPlan] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.objective, Objective):
            self.objective = Objective(str(self.objective))

    def explain(self) -> str:
        lines = [
            f"plan for {self.graph_name}",
            f"objective: {closed_str(self.objective)}",
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
    that sample by ``node_count`` (FX ops in the region) so a large fused model
    is not priced like a single 64×64 Linear.
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
    """Simple plan score helper (tests / diagnostics). Lower is better."""
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


def region_byte_counts(graph_ir: HeterogeneousGraph) -> dict[str, tuple[int, int]]:
    """Output and state bytes per compute region from lowered IR.

    Shared weights (same alias/storage) count once per region.
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
            if meta is None or meta.kind != TensorKind.PARAMETER:
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


def plan_execution(
    graph_ir: HeterogeneousGraph,
    machine: ResourceGraph,
    config: CompileConfig | None = None,
    measurements: MeasurementSet | None = None,
) -> ExecutionPlan:
    """Jointly select resources, kernels, transfers, and memory-feasible placement.

    Uses the native Rust planner: parallel device-subset beam search shortlists
    distinct finalists. Discrete-event simulation (in specialize) picks the winner.
    """
    from tensortorrent.planner.native import (
        build_planning_problem,
        device_capacity_bytes,
        placements_from_native,
        run_native_planner,
    )

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

    subset_work: list[tuple[ComputeResource, ...]] = []
    subset_diagnostics: list[str] = []
    for subset in subsets:
        names = {device.id.name for device in subset}
        if not _subset_passes_vendor_filter(subset, config):
            subset_diagnostics.append(f"subset=[{','.join(sorted(names))}] mixed_vendor_disallowed")
            continue
        subset_work.append(subset)

    if not subset_work:
        raise PlanningError(
            "No eligible device subsets after vendor filters. "
            f"subset_failures=[{'; '.join(subset_diagnostics) or 'none'}]"
        )

    problem = build_planning_problem(
        graph_ir,
        machine,
        region_candidates,
        subset_work,
        byte_counts,
        config,
    )
    if problem is None:
        raise PlanningError("Graph dependency cycle prevents topological planning")

    try:
        native_out = run_native_planner(problem)
    except Exception as exc:
        raise PlanningError(f"Native planner failed: {exc}") from exc

    finalists_raw = list(native_out.get("finalists") or [])
    stats = dict(native_out.get("statistics") or {})
    if not finalists_raw:
        eligible_names = [device.id.name for device in eligible]
        capacities = ",".join(
            f"{device.id.name}={device_capacity_bytes(machine, device.id.name, vram_budget_bytes=config.vram_budget_bytes)}"
            for device in eligible
        )
        counts = ",".join(f"{region}={len(candidates)}" for region, candidates in region_candidates.items())
        details = "; ".join(subset_diagnostics) if subset_diagnostics else "all subsets infeasible"
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

    storage = [memory for memory in machine.memory.values() if memory.memory_class.value in {"nvme", "disk_cache"}]
    solo_latencies: dict[str, float] = {}
    plans: list[ExecutionPlan] = []

    for finalist in finalists_raw:
        placements = placements_from_native(finalist)
        used = {p.device for p in placements}
        subset_names = tuple(str(x) for x in (finalist.get("subset_devices") or ()))
        if len(subset_names) == 1:
            solo_latencies[subset_names[0]] = float(finalist.get("latency_s") or 0.0)
        if len(used) == 1:
            only = next(iter(used))
            solo_latencies.setdefault(only, float(finalist.get("latency_s") or 0.0))

        communication = select_communication_backend(tuple(sorted(used)))
        device_objs = tuple(device for device in eligible if device.id.name in used)
        notes = [
            f"subset={','.join(subset_names)}",
            _measurement_note(placements),
            "planner=native_rust_beam_search",
            (
                f"planner_search expanded={finalist.get('states_expanded', 0)} "
                f"pruned={finalist.get('states_pruned', 0)} "
                f"analytic_rank={finalist.get('analytic_rank', finalist.get('search_rank', 0))} "
                f"finalist_rank={finalist.get('finalist_rank', 0)}"
            ),
        ]
        if int(finalist.get("unmeasured_transfer_count") or 0):
            notes.append(
                f"unmeasured_transfer_priors={finalist['unmeasured_transfer_count']}; validate links on target hardware"
            )
        if int(finalist.get("host_staged_transfer_count") or 0):
            notes.append(f"host_staged_transfers={finalist['host_staged_transfer_count']}")
        if storage:
            notes.append(
                f"storage_resources_detected={len(storage)}; adaptive prefetch is bounded by the host RAM budget"
            )

        analytic_rank = int(finalist.get("analytic_rank", finalist.get("search_rank") or 0) or 0)
        finalist_rank = int(finalist.get("finalist_rank") or 0)
        plans.append(
            ExecutionPlan(
                graph_name=graph_ir.name,
                fingerprint=machine.fingerprint,
                objective=config.objective,
                placements=placements,
                decisions=[],
                devices_used=tuple(sorted(used)),
                communication_backend=communication.backend_id,
                predicted_latency_s=float(finalist.get("latency_s") or 0.0),
                predicted_peak_bytes={str(k): int(v) for k, v in dict(finalist.get("peak_bytes") or {}).items()},
                predicted_throughput_per_s=float(finalist.get("throughput_per_s") or 0.0),
                predicted_transfer_bytes=int(finalist.get("transfer_bytes") or 0),
                predicted_transfer_latency_s=float(finalist.get("transfer_latency_s") or 0.0),
                strategy=_strategy_name(device_objs),
                search_statistics={
                    **stats,
                    "analytic_score": float(finalist.get("analytic_score") or 0.0),
                    "analytic_rank": analytic_rank,
                    "search_rank": analytic_rank,  # alias: real analytical rank
                    "finalist_rank": finalist_rank,
                    "placement_signature": str(finalist.get("placement_signature") or ""),
                    "host_staged_transfer_count": int(finalist.get("host_staged_transfer_count") or 0),
                    "target_inflight_requests": config.target_inflight_requests,
                },
                notes=notes,
            )
        )

    best = plans[0]
    best.finalist_plans = plans
    used_set = set(best.devices_used)
    best.decisions = _decide_resources(machine, eligible, used_set, best.predicted_latency_s, solo_latencies)
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
