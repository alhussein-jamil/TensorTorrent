"""Static GPU-prefix + CPU-overflow placement (Accelerate ``device_map=auto`` analog).

Sequential beyond-VRAM graphs can keep a leading region prefix resident on the
accelerator and run the remainder on host. Overflow weights never H2D; only
activations cross the cut. Bakeoff measures this against streamed GPU and fused CPU.
"""

from __future__ import annotations

from dataclasses import replace

from tensortorrent.compile.regions import RegionProgram
from tensortorrent.ir.resource_graph import ComputeClass, ResourceGraph
from tensortorrent.planner.maximal import ExecutionPlan


def host_cpu_placement_target(machine: ResourceGraph) -> tuple[str, str] | None:
    """First NUMA CPU pool ``(device_id, backend_id)``, if the graph has one."""
    for device in machine.compute.values():
        if device.compute_class == ComputeClass.CPU_NUMA_POOL:
            return device.id.name, str(device.backend_id)
    return None


def gpu_prefix_count(program: RegionProgram, budget_bytes: int) -> int:
    """How many leading regions' unique state fits ``budget_bytes``.

    ``program.regions`` order is lowering order (a chain on the sequential
    beyond-VRAM path). Returns 0 when the first region does not fit, and
    ``len(regions)`` when the whole model fits.
    """
    budget = int(budget_bytes)
    if budget < 1 or not program.regions:
        return 0
    seen: list[str] = []
    n = 0
    for region in program.regions:
        candidate = [*seen, *region.state_inputs]
        total = program.unique_state_bytes(candidate)
        if total > budget:
            break
        seen = candidate
        n += 1
    return n


def gpu_prefix_overflow_plan(
    source_plan: ExecutionPlan,
    program: RegionProgram,
    *,
    n_gpu: int,
    cpu_device: str,
    cpu_backend: str,
) -> ExecutionPlan:
    """Copy ``source_plan`` with suffix regions rewritten onto the host CPU.

    Prefix region ids come from ``program.regions[:n_gpu]``. GPU placements for
    those ids are kept as-is (device, dtype, kernel).
    """
    regions = program.regions
    if n_gpu < 1 or n_gpu >= len(regions):
        raise ValueError(f"gpu prefix count must be interior, got {n_gpu}/{len(regions)}")
    prefix_ids = {region.region_id for region in regions[:n_gpu]}
    placements = []
    for placement in source_plan.placements:
        if placement.region_id in prefix_ids:
            placements.append(placement)
            continue
        dtype = str(placement.dtype or "float32")
        placements.append(
            replace(
                placement,
                device=cpu_device,
                backend_id=cpu_backend,
                kernel_id=f"{cpu_backend}_fx_{dtype}",
            )
        )
    devices = tuple(sorted({p.device for p in placements}))
    notes = [n for n in source_plan.notes if not str(n).startswith("baseline_compare")]
    notes.append(f"gpu_prefix_cpu_overflow: prefix={n_gpu}/{len(regions)}")
    return replace(
        source_plan,
        placements=placements,
        devices_used=devices,
        strategy="gpu_prefix_cpu_overflow",
        prefetch_distance=0,
        finalist_plans=[],
        notes=notes,
        fingerprint=f"{source_plan.fingerprint}:gpu_prefix_overflow",
    )
