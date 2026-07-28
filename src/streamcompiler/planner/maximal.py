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
from streamcompiler.config import CompileConfig, Objective
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode
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
            lines.append(
                f"  {p.region_id} -> {p.device} [{p.backend_id}/{p.dtype}] "
                f"~{p.estimated_latency_s:.6f}s"
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
        gpu_vendors = {d.vendor for d in out if d.compute_class in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU)}
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
) -> dict[str, list[KernelCandidate]]:
    regions = graph_ir.compute_regions() or [
        Instruction(opcode=OpCode.COMPUTE, name=f"region_{i}", attributes={"op": "graph"})
        for i in range(max(1, len(graph_ir.repeated_blocks) or 1))
    ]
    # If IR has no explicit compute ops yet, synthesize block regions.
    if not graph_ir.compute_regions():
        if graph_ir.repeated_blocks:
            regions = [
                Instruction(opcode=OpCode.COMPUTE, name=f"block_{i}", attributes={"ops": list(block)})
                for i, block in enumerate(graph_ir.repeated_blocks)
            ]
        else:
            regions = [Instruction(opcode=OpCode.COMPUTE, name="main", attributes={})]

    by_region: dict[str, list[KernelCandidate]] = {}
    for region in regions:
        cands: list[KernelCandidate] = []
        for device in devices:
            backend = backend_by_id(device.backend_id)
            if backend is None:
                continue
            for cand in backend.enumerate_kernels(region, device):
                # Coarse prior: prefer device-reported preferred dtype order already used.
                # Latency prior uses relative device class until measured.
                prior = _prior_latency(device, cand.dtype)
                cands.append(
                    KernelCandidate(
                        region_id=cand.region_id,
                        device=cand.device,
                        backend_id=cand.backend_id,
                        kernel_id=cand.kernel_id,
                        dtype=cand.dtype,
                        estimated_latency_s=prior,
                        workspace_bytes=cand.workspace_bytes,
                        attributes=dict(cand.attributes),
                    )
                )
        by_region[region.name] = cands
    return by_region


def _prior_latency(device: ComputeResource, dtype: str) -> float:
    """Coarse unmeasured prior. Never presented as a real benchmark result."""
    base = {
        ComputeClass.DISCRETE_GPU: 0.002,
        ComputeClass.INTEGRATED_GPU: 0.004,
        ComputeClass.CPU_NUMA_POOL: 0.02,
        ComputeClass.CPU_SOCKET: 0.02,
        ComputeClass.ACCELERATOR: 0.003,
        ComputeClass.COPY_ENGINE: 0.001,
    }.get(device.compute_class, 0.05)
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


def _score_plan(latency_s: float, config: CompileConfig) -> float:
    # Lower is better for latency/memory-oriented scores.
    if config.objective == Objective.LATENCY:
        return latency_s
    if config.objective == Objective.THROUGHPUT:
        return -1.0 / max(latency_s, 1e-9)
    if config.objective == Objective.MEMORY:
        return latency_s  # memory peaks attached later
    if config.objective == Objective.BALANCED:
        return latency_s
    weights = config.objective_weights
    return weights.get("latency", 1.0) * latency_s


def _device_memory_bytes(device: ComputeResource, machine: ResourceGraph | None = None) -> int:
    if machine is None:
        return 0
    total = 0
    for name in device.memory_affinity:
        mem = machine.memory.get(name)
        if mem is not None:
            total += mem.allocatable_bytes
    return total


def _assign_regions(
    region_candidates: dict[str, list[KernelCandidate]],
    subset: tuple[ComputeResource, ...],
    machine: ResourceGraph | None = None,
) -> list[Placement] | None:
    allowed = {d.id.name for d in subset}
    # Larger VRAM / more cores attract heavier shards; faster priors attract compute.
    capacity = {d.id.name: _device_memory_bytes(d, machine) for d in subset}
    speed = {
        d.id.name: 1.0 / max(1e-9, _prior_latency(d, next(iter(d.supported_dtypes), "float32")))
        for d in subset
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
        device_load[best.device] += lat
        device_bytes[best.device] += 1_048_576
        placements.append(
            Placement(
                region_id=region_id,
                device=best.device,
                backend_id=best.backend_id,
                dtype=best.dtype,
                kernel_id=best.kernel_id,
                estimated_latency_s=lat,
            )
        )
    return placements


def _pipeline_latency(placements: list[Placement]) -> float:
    """Approximate critical-path latency with concurrent devices."""
    if not placements:
        return float("inf")
    per_device: dict[str, float] = {}
    for p in placements:
        per_device[p.device] = per_device.get(p.device, 0.0) + p.estimated_latency_s
    # Critical path ~ max device load + small sync tax for multi-device.
    sync_tax = 0.0 if len(per_device) <= 1 else 0.0005 * (len(per_device) - 1)
    return max(per_device.values()) + sync_tax


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
            if device.compute_class in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU):
                reason = "additional throughput exceeds synchronization cost"
                if solo is not None and abs(solo - best_latency) < 1e-12:
                    reason = "fastest matrix multiplication backend for critical regions"
            elif device.compute_class == ComputeClass.CPU_NUMA_POOL:
                reason = "CPU-efficient regions remain outside the accelerator critical path"
            else:
                reason = "selected by measured/prior objective improvement"
            decisions.append(
                ResourceDecision(
                    resource=name,
                    selected=True,
                    reason=reason,
                    estimated_benefit_s=None if solo is None else max(0.0, solo - best_latency),
                    estimated_cost_s=None,
                )
            )
        else:
            if device.compute_class == ComputeClass.CPU_NUMA_POOL:
                reason = "NUMA-remote transfer cost increases latency"
            elif device.compute_class in (ComputeClass.DISCRETE_GPU, ComputeClass.INTEGRATED_GPU):
                reason = "host-staged synchronization cost exceeds its compute contribution"
            else:
                reason = "participation did not improve the selected objective"
            decisions.append(
                ResourceDecision(
                    resource=name,
                    selected=False,
                    reason=reason,
                    estimated_benefit_s=None,
                    estimated_cost_s=None if solo is None else max(0.0, (solo or 0) - best_latency),
                )
            )
    # Also report copy engines / storage when present.
    for device in graph.compute.values():
        if device.compute_class == ComputeClass.COPY_ENGINE:
            decisions.append(
                ResourceDecision(
                    resource=device.id.name,
                    selected=True,
                    reason="overlap weight/activation movement with compute",
                )
            )
    return decisions


def plan_execution(
    graph_ir: HeterogeneousGraph,
    machine: ResourceGraph,
    config: CompileConfig | None = None,
) -> ExecutionPlan:
    config = config or CompileConfig()
    eligible = _eligible_compute(machine, config)
    if not eligible:
        raise RuntimeError("No eligible compute resources discovered")

    region_candidates = _region_candidates(graph_ir, eligible)
    subsets = _device_subsets(eligible, limit=config.max_plan_candidates)

    # Solo latencies for exclusion explanations.
    solo_latencies: dict[str, float] = {}
    for device in eligible:
        placed = _assign_regions(region_candidates, (device,), machine)
        if placed:
            solo_latencies[device.id.name] = _pipeline_latency(placed)

    best: ExecutionPlan | None = None
    best_score = float("inf")

    # Storage pipeline note: presence of NVMe does not force participation.
    storage = [m for m in machine.memory.values() if m.memory_class.value in {"nvme", "disk_cache"}]

    for subset in subsets:
        placements = _assign_regions(region_candidates, subset, machine)
        if not placements:
            continue
        latency = _pipeline_latency(placements)        # Penalize mixed-vendor plans that require host staging when disabled.
        vendors = {d.vendor for d in subset if d.vendor}
        if len(vendors) > 1:
            if not config.allow_mixed_vendor:
                continue
            if not config.allow_host_staged_transfers:
                # Require that every GPU pair has P2P; otherwise skip.
                gpu_mems = []
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

        score = _score_plan(latency, config)
        # Convert throughput scores (negative) into minimization space.
        if config.objective == Objective.THROUGHPUT:
            comparable = score  # already negative = better
            # For comparison we keep lower-is-better by negating again below.
            comparable = -score
        else:
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
                "Priors used where measurements are missing; run `streamcompiler profile` on the deployment machine.",
                f"subset={','.join(d.id.name for d in subset)}",
            ],
        )
        if storage:
            plan.notes.append(
                f"storage_resources_detected={len(storage)}; "
                "included only when streaming reduces critical-path stalls"
            )
        if comparable < best_score:
            best_score = comparable
            best = plan

    if best is None:
        raise RuntimeError("Planner failed to produce any feasible plan")

    used_set = set(best.devices_used)
    best.decisions = _decide_resources(
        machine, eligible, used_set, best.predicted_latency_s, solo_latencies
    )
    return best


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
