"""Maximal heterogeneous planner.

Searches subsets and combinations of the machine. A device participates only
when it improves the selected objective. Vendor-specific logic stays behind
backend capability queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from streamcompiler.backends import backend_by_id
from streamcompiler.backends.base import KernelCandidate
from streamcompiler.communication import select_communication_backend
from streamcompiler.compile.measure import MeasurementSet
from streamcompiler.config import CompileConfig, Objective
from streamcompiler.errors import PlanningError
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction
from streamcompiler.ir.resource_graph import (
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

    @property
    def working_set_bytes(self) -> int:
        return self.output_bytes + self.state_bytes


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
    strategy: str = ""
    notes: list[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = [
            f"plan for {self.graph_name}",
            f"objective: {self.objective}",
            f"strategy: {self.strategy}",
            f"predicted_latency_s: {self.predicted_latency_s:.6f}",
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
        if device.compute_class == ComputeClass.DISCRETE_GPU and not config.allow_gpu:
            continue
        if device.compute_class == ComputeClass.INTEGRATED_GPU and not config.allow_integrated_gpu:
            continue
        if not config.allow_mixed_vendor:
            # Will filter after seeing first GPU vendor.
            pass
        out.append(device)
    if not config.allow_mixed_vendor:
        gpu_vendors = {
            d.vendor for d in out if d.compute_class in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU)
        }
        if len(gpu_vendors) > 1:
            # Keep first vendor only when mixed vendors disabled.
            keep_vendor = sorted(gpu_vendors)[0]
            out = [
                d
                for d in out
                if d.compute_class not in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU)
                or d.vendor == keep_vendor
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
                if measurement is not None and measurement.measured:
                    latency = measurement.latency_s
                    measured = True
                else:
                    latency = _scaled_prior(region, device, cand.dtype, measurements)
                    measured = False
                attributes = dict(cand.attributes)
                attributes["measured"] = measured
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
    """
    reference = measurements.best_measured(region.name) if measurements else None
    if reference is None:
        return _relative_device_cost(device, dtype)
    ratio = _relative_device_cost(device, dtype) / max(1e-12, _CPU_REFERENCE_COST)
    return reference.latency_s * ratio


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
    if device.compute_class in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU):
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


def _device_subsets(devices: list[ComputeResource], limit: int = 24) -> list[tuple[ComputeResource, ...]]:
    """Generate meaningful machine subsets without assuming all-devices is best."""
    cpus = [d for d in devices if d.compute_class == ComputeClass.CPU_NUMA_POOL]
    gpus = [
        d
        for d in devices
        if d.compute_class in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU, ComputeClass.ACCELERATOR)
    ]
    subsets: list[tuple[ComputeResource, ...]] = []

    # CPU only
    if cpus:
        subsets.append((cpus[0],))
        if len(cpus) > 1:
            subsets.append(tuple(cpus))

    # Each GPU independently
    for g in gpus:
        subsets.append((g,))

    # All GPUs
    if len(gpus) >= 2:
        subsets.append(tuple(gpus))

    # All GPUs + selected CPU cores
    if gpus and cpus:
        subsets.append(tuple(gpus) + (cpus[0],))
        if len(cpus) > 1:
            subsets.append(tuple(gpus) + tuple(cpus))

    # Unequal GPU pairs / small combinations
    for r in range(2, min(3, len(gpus)) + 1):
        for combo in combinations(gpus, r):
            subsets.append(combo)

    # Deduplicate while preserving order
    seen: set[tuple[str, ...]] = set()
    unique: list[tuple[ComputeResource, ...]] = []
    for subset in subsets:
        key = tuple(sorted(d.id.name for d in subset))
        if key in seen:
            continue
        seen.add(key)
        unique.append(subset)
        if len(unique) >= limit:
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


def _assign_regions(
    region_candidates: dict[str, list[KernelCandidate]],
    subset: tuple[ComputeResource, ...],
    machine: ResourceGraph | None = None,
    dependencies: dict[str, tuple[str, ...]] | None = None,
    byte_counts: dict[str, tuple[int, int]] | None = None,
    *,
    vram_budget_bytes: int | None = None,
) -> list[Placement] | None:
    allowed = {d.id.name for d in subset}
    # Larger VRAM / more cores attract heavier shards; faster priors attract compute.
    capacity = {d.id.name: _device_memory_bytes(d, machine, vram_budget_bytes=vram_budget_bytes) for d in subset}
    speed = {
        d.id.name: 1.0 / max(1e-9, _relative_device_cost(d, next(iter(d.supported_dtypes), "float32"))) for d in subset
    }
    placements: list[Placement] = []
    device_load: dict[str, float] = {d.id.name: 0.0 for d in subset}
    device_bytes: dict[str, int] = {d.id.name: 0 for d in subset}

    region_ids = list(region_candidates.keys())
    for idx, region_id in enumerate(region_ids):
        cands = region_candidates[region_id]
        usable = [c for c in cands if c.device in allowed]
        if not usable:
            return None
        # Alternate preference: early regions prefer speed, later prefer remaining capacity.
        prefer_capacity = idx >= max(1, len(region_ids) // 2)

        def key(
            c: KernelCandidate,
            *,
            _prefer_capacity: bool = prefer_capacity,
        ) -> tuple[float, float, float]:
            lat = c.estimated_latency_s or 1.0
            load = device_load[c.device]
            # Soft capacity pressure: penalize devices already near allocatable bytes.
            cap = capacity.get(c.device, 0)
            used = device_bytes.get(c.device, 0)
            pressure = (used / cap) if cap > 0 else 0.0
            speed_score = -speed.get(c.device, 0.0) if _prefer_capacity else 0.0
            cap_score = -float(cap) if _prefer_capacity else 0.0
            return (lat + load + 0.25 * pressure, speed_score, cap_score)

        best = min(usable, key=key)
        lat = float(best.estimated_latency_s or 1.0)
        output_bytes, state_bytes = (byte_counts or {}).get(region_id, (0, 0))
        device_load[best.device] += lat
        device_bytes[best.device] += output_bytes + state_bytes
        placements.append(
            Placement(
                region_id=region_id,
                device=best.device,
                backend_id=best.backend_id,
                dtype=best.dtype,
                kernel_id=best.kernel_id,
                estimated_latency_s=lat,
                depends_on=(dependencies or {}).get(region_id, ()),
                measured=bool(best.attributes.get("measured", False)),
                output_bytes=output_bytes,
                state_bytes=state_bytes,
            )
        )
    return placements


def _pipeline_latency(placements: list[Placement]) -> float:
    """Critical-path latency honouring both device serialization and dependencies.

    Each device executes its own placements sequentially; a region additionally
    cannot start before the regions it depends on have finished. The result is the
    longest completion time across all regions, i.e. a genuine critical path
    rather than a per-device load estimate.
    """
    if not placements:
        return float("inf")
    finish: dict[str, float] = {}
    device_free: dict[str, float] = {}
    for placement in placements:
        dep_ready = max((finish.get(d, 0.0) for d in placement.depends_on), default=0.0)
        start = max(dep_ready, device_free.get(placement.device, 0.0))
        end = start + placement.estimated_latency_s
        finish[placement.region_id] = end
        device_free[placement.device] = end
    sync_tax = 0.0 if len(device_free) <= 1 else 0.0005 * (len(device_free) - 1)
    # sync_tax is an unmeasured multi-device coordination prior, not a benchmark.
    return max(finish.values()) + sync_tax


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
            elif device.compute_class in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU):
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


def plan_execution(
    graph_ir: HeterogeneousGraph,
    machine: ResourceGraph,
    config: CompileConfig | None = None,
    measurements: MeasurementSet | None = None,
) -> ExecutionPlan:
    config = config or CompileConfig()
    eligible = _eligible_compute(machine, config)
    if not eligible:
        raise PlanningError("No eligible compute resources discovered")

    dependencies = {
        inst.name: tuple(str(d) for d in inst.attributes.get("depends_on", ())) for inst in graph_ir.compute_regions()
    }
    region_candidates = _region_candidates(graph_ir, eligible, measurements)
    byte_counts = region_byte_counts(graph_ir)
    subsets = _device_subsets(eligible, limit=config.max_plan_candidates)

    # Solo latencies for exclusion explanations.
    solo_latencies: dict[str, float] = {}
    for device in eligible:
        placed = _assign_regions(
            region_candidates,
            (device,),
            machine,
            dependencies,
            byte_counts,
            vram_budget_bytes=config.vram_budget_bytes,
        )
        if placed:
            solo_latencies[device.id.name] = _pipeline_latency(placed)

    best: ExecutionPlan | None = None
    best_score = float("inf")

    # Storage pipeline note: presence of NVMe does not force participation.
    storage = [m for m in machine.memory.values() if m.memory_class.value in {"nvme", "disk_cache"}]

    for subset in subsets:
        placements = _assign_regions(
            region_candidates,
            subset,
            machine,
            dependencies,
            byte_counts,
            vram_budget_bytes=config.vram_budget_bytes,
        )
        if not placements:
            continue
        latency = _pipeline_latency(placements)  # Penalize mixed-vendor plans that require host staging when disabled.
        vendors = {d.vendor for d in subset if d.vendor}
        if len(vendors) > 1:
            if not config.allow_mixed_vendor:
                continue
            if not config.allow_host_staged_transfers:
                # Require that every GPU pair has P2P; otherwise skip.
                gpu_mems: list[str] = []
                for d in subset:
                    gpu_mems.extend(d.memory_affinity)
                ok = True
                for a in gpu_mems:
                    for b in gpu_mems:
                        if a == b:
                            continue
                        link = machine.link_between(a, b)
                        if link is None or not link.peer_to_peer:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
            latency *= 1.15  # host-staged tax prior
            host_staged_tax = True
        else:
            host_staged_tax = False

        score = _score_plan(
            latency,
            config,
            peak_working_set_bytes=_peak_working_set_bytes(placements),
        )
        comparable = score

        used = {p.device for p in placements}
        comm = select_communication_backend(tuple(sorted(used)))
        strategy = _strategy_name(subset)
        plan = ExecutionPlan(
            graph_name=graph_ir.name,
            fingerprint=machine.fingerprint,
            objective=config.objective.value,
            placements=placements,
            decisions=[],  # filled after best known
            devices_used=tuple(sorted(used)),
            communication_backend=comm.backend_id,
            predicted_latency_s=latency,
            strategy=strategy,
            notes=[
                f"subset={','.join(d.id.name for d in subset)}",
                _measurement_note(placements),
            ],
        )
        if host_staged_tax:
            plan.notes.append("host_staged_tax_prior=1.15x on mixed-vendor latency (unmeasured; not a benchmark)")
        if storage:
            plan.notes.append(
                f"storage_resources_detected={len(storage)}; included only when streaming reduces critical-path stalls"
            )
        if comparable < best_score:
            best_score = comparable
            best = plan

    if best is None:
        raise PlanningError("Planner failed to produce any feasible plan")

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
        return "tensor_or_pipeline_multi_gpu"
    if len(gpus) == 1:
        return "single_gpu"
    if len(cpus) > 1:
        return "multi_numa_cpu"
    return "cpu_only"


def enumerate_plan_strategies() -> tuple[str, ...]:
    return (
        "cpu_only",
        "each_gpu_independently",
        "all_gpus",
        "all_gpus_plus_selected_cpu",
        "pipeline_across_gpus_and_cpus",
        "tensor_partition_unequal_gpus",
        "tensor_partition_gpus_and_cpus",
        "independent_branches",
        "multi_request_separate_resources",
        "shared_weight_streaming",
        "separate_storage_pipelines",
    )
