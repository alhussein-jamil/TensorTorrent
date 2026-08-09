"""Machine-specific specialization of portable artifacts."""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

from tensortorrent.backends import backend_by_id
from tensortorrent.compile.artifacts import PortableArtifact, SpecializedArtifact
from tensortorrent.compile.concurrency import ConcurrencyDecision, dependency_levels
from tensortorrent.compile.measure import (
    MeasurementSet,
    RegionMeasurement,
    measure_regions_on_devices,
    region_source,
)
from tensortorrent.compile.regions import RegionBinding, RegionProgram
from tensortorrent.config import CompileConfig
from tensortorrent.errors import SpecializationError
from tensortorrent.hardware.discovery import discover_resource_graph
from tensortorrent.hardware.fingerprint import machine_fingerprint
from tensortorrent.ir.resource_graph import ResourceGraph
from tensortorrent.planner.maximal import ExecutionPlan, plan_execution

logger = logging.getLogger(__name__)


def specialize_for_machine(
    portable: PortableArtifact,
    *,
    config: CompileConfig | None = None,
    output_dir: Path | None = None,
    example_inputs: list[Any] | None = None,
    machine: Any | None = None,
    profile_feedback: Any | None = None,
    measurements: MeasurementSet | None = None,
    compile_regions: bool = True,
) -> SpecializedArtifact:
    """Deployment-time specialization against the actual machine resource graph.

    When ``compile_regions`` is False, measure/plan/schedule/concurrency still run
    but region kernel compile is skipped. Used as a cheap probe before the fusion
    path decides whether a second full specialize is worth paying for.
    """
    config = config or CompileConfig()
    machine = machine if machine is not None else discover_resource_graph()
    current_fp = machine.fingerprint or machine_fingerprint()
    program = portable.program
    timing: dict[str, float] = {}
    t_specialize0 = perf_counter()

    region_inputs: dict[str, tuple[Any, ...]] = {}
    # Region-input capture materializes a full sequential forward (and can retain a
    # second weight footprint). Skip it unless region measurement needs those
    # tensors — planning works from IR priors alone when measure_regions=False.
    if measurements is None:
        measurements = MeasurementSet()
        if program is not None and example_inputs is not None and config.measure_regions:
            # Call through pipeline so tests can monkeypatch pipeline.capture_region_inputs.
            from tensortorrent.compile import pipeline as _pipeline

            t0 = perf_counter()
            # Capture timings fill gaps when a later CPU probe is missing/failed.
            try:
                capture_out = _pipeline.capture_region_inputs(program, example_inputs, time_regions=True)
            except TypeError:
                # Monkeypatched capture helpers may not accept time_regions.
                capture_out = _pipeline.capture_region_inputs(program, example_inputs)
            if isinstance(capture_out, tuple):
                region_inputs, capture_times = capture_out
            else:
                region_inputs, capture_times = capture_out, {}
            timing["capture_s"] = perf_counter() - t0
            profile_devices = []
            for device in machine.compute.values():
                backend_id = str(device.backend_id)
                is_cpu = backend_id in {"cpu", "cpu_numa"}
                if is_cpu and not config.allow_cpu:
                    continue
                if not is_cpu and not config.allow_gpu:
                    continue
                profile_devices.append(device)
            t0 = perf_counter()
            measurements = measure_regions_on_devices(
                program,
                region_inputs,
                profile_devices,
                iters=config.region_measure_iters,
                workers=config.measure_workers,
            )
            _seed_cpu_measurements_from_capture(measurements, capture_times, profile_devices)
            timing["measure_s"] = perf_counter() - t0
    elif program is not None and example_inputs is not None and config.measure_regions:
        from tensortorrent.compile import pipeline as _pipeline

        t0 = perf_counter()
        captured = _pipeline.capture_region_inputs(program, example_inputs)
        region_inputs = captured[0] if isinstance(captured, tuple) else captured
        timing["capture_s"] = perf_counter() - t0

    if profile_feedback is not None and hasattr(profile_feedback, "merge_into_measurements"):
        measurements = profile_feedback.merge_into_measurements(measurements)

    if program is not None and not program.regions:
        return _passthrough_specialization(program, current_fp, output_dir)

    t0 = perf_counter()
    plan = plan_execution(portable.ir, machine, config, measurements)
    from tensortorrent.compile.fit import needs_parameter_streaming
    from tensortorrent.planner.collectives import plan_collectives
    from tensortorrent.planner.local_search import refine_prefetch_distance
    from tensortorrent.runtime.residency import attach_residency_to_plan

    finalists = list(plan.finalist_plans) or [plan]
    storage_bps = _planning_storage_bandwidth(machine)
    for cand in finalists:
        refined = refine_prefetch_distance(
            cand,
            distance=config.prefetch_distance,
            adaptive=config.adaptive_prefetch,
            ram_budget_bytes=config.ram_budget_bytes,
            storage_bytes_per_s=storage_bps,
        )
        cand.prefetch_distance = refined.prefetch_distance
        cand.notes = list(refined.notes)
    timing["plan_s"] = perf_counter() - t0

    compiled: list[dict[str, Any]] = []
    bindings: dict[str, RegionBinding] = {}
    profile: dict[str, Any] = {
        "devices": {},
        "transfers": {},
        "missing_measurements": [],
        "region_measurements": measurements.as_dict(),
    }
    if program is not None:
        planned = {p.region_id for p in plan.placements}
        expected = {r.region_id for r in program.regions}
        if planned != expected:
            raise SpecializationError(
                "Plan does not cover the compiled regions exactly once: "
                f"missing={sorted(expected - planned)} unexpected={sorted(planned - expected)}"
            )

    total_state = int(program.total_state_bytes()) if program is not None else 0
    streaming = needs_parameter_streaming(config, state_bytes=total_state)
    profile["tensor_size_metadata"] = "exact" if program is not None else "estimated_from_portable_ir"

    # Rank finalists with DES *before* expensive region compilation.
    t0 = perf_counter()
    plan, executable_schedule, sim, prefetch, select_stats = _select_finalist_by_simulation(
        finalists,
        program=program,
        streaming=streaming,
        activation_budget_bytes=config.activation_budget_bytes,
        machine=machine,
        config=config,
    )
    timing["simulate_s"] = select_stats.get("batch_simulation_s", 0.0)
    timing["schedule_build_s"] = select_stats.get("schedule_build_s", 0.0)
    timing["finalist_selection_s"] = select_stats.get("finalist_selection_s", 0.0)
    plan.prefetch_distance = prefetch
    plan.search_statistics = {**dict(plan.search_statistics), **select_stats}
    residency = attach_residency_to_plan(plan, program)
    profile["residency"] = residency.as_dict()

    # Compile only the DES-selected winner.
    t0 = perf_counter()
    compiled, bindings = _compile_plan_placements(
        plan,
        program=program,
        machine=machine,
        config=config,
        current_fp=current_fp,
        region_inputs=region_inputs,
        compile_regions=compile_regions,
        profile=profile,
    )
    timing["region_compile_s"] = perf_counter() - t0

    for placement in plan.placements:
        binding = bindings.get(placement.region_id)
        if binding is not None and (binding.device != placement.device or binding.backend_id != placement.backend_id):
            raise SpecializationError(
                f"Binding for {placement.region_id} is {binding.backend_id}/{binding.device} "
                f"but plan says {placement.backend_id}/{placement.device}"
            )

    collectives = plan_collectives(portable.ir, machine, plan.devices_used)
    if collectives:
        plan.notes.append("collectives=" + ",".join(f"{c.op}:{c.backend_id}" for c in collectives))

    from tensortorrent.ir.graph import OpCode
    from tensortorrent.planner.cost.calibration import runtime_predicted_makespan_s

    n_compute = sum(1 for i in executable_schedule.instructions if i.opcode == OpCode.COMPUTE)
    plan.predicted_latency_s = runtime_predicted_makespan_s(sim.makespan_s, n_compute=n_compute)
    plan.predicted_peak_bytes = sim.peak_bytes
    plan.predicted_transfer_bytes = sim.bytes_transferred
    plan.predicted_transfer_latency_s = sim.exposed_transfer_latency_s
    ii = float(getattr(sim, "initiation_interval_s", 0.0) or 0.0)
    if ii > 0:
        plan.predicted_throughput_per_s = min(
            1.0 / ii,
            config.target_inflight_requests / max(plan.predicted_latency_s, 1e-12),
        )
    elif plan.predicted_latency_s > 0:
        finite_concurrency_bound = config.target_inflight_requests / plan.predicted_latency_s
        if plan.predicted_throughput_per_s > 0:
            plan.predicted_throughput_per_s = min(plan.predicted_throughput_per_s, finite_concurrency_bound)
        else:
            plan.predicted_throughput_per_s = finite_concurrency_bound
    for decision in plan.decisions:
        if "simulated_makespan=" not in decision.reason:
            decision.reason = (
                f"{decision.reason}; simulated_makespan={plan.predicted_latency_s * 1e3:.3f} ms "
                f"(analytic={sim.makespan_s * 1e3:.3f} ms + host bridge)"
            )
    plan.notes.append(
        f"simulator makespan={plan.predicted_latency_s:.6f}s analytic={sim.makespan_s:.6f}s "
        f"exposed_transfer={sim.exposed_transfer_latency_s:.6f}s "
        f"(simulated={sim.simulated}; schedule_instructions={len(executable_schedule.instructions)})"
    )
    profile["executable_schedule"] = executable_schedule.as_dict()
    profile["planner_search"] = dict(plan.search_statistics)
    timing["total_s"] = perf_counter() - t_specialize0
    profile["specialize_timing"] = timing
    if portable.metadata.get("buffer_reuse"):
        profile["buffer_reuse"] = portable.metadata["buffer_reuse"]
    eviction_events = sum(1 for e in sim.timeline if e.get("event") == "eviction_pressure")
    transfer_landed = sum(1 for e in sim.timeline if e.get("event") in ("Transfer", "transfer_landed"))
    profile["simulator"] = {
        "simulated": sim.simulated,
        "makespan_s": sim.makespan_s,
        "initiation_interval_s": ii,
        "exposed_transfer_latency_s": sim.exposed_transfer_latency_s,
        "transfer_events": len(sim.transfer_events),
        "transfer_landed_events": transfer_landed,
        "release_events": len(sim.release_events),
        "eviction_pressure_events": eviction_events,
        "peak_bytes": dict(sim.peak_bytes),
        "critical_path": list(sim.critical_path),
        "bytes_read": sim.bytes_read,
        "bytes_transferred": sim.bytes_transferred,
        "schedule_instructions": len(executable_schedule.instructions),
        "schedule_transfers": len(executable_schedule.transfer_ops()),
        "source": "executable_schedule",
    }
    if eviction_events:
        plan.notes.append(
            f"simulator eviction_pressure_events={eviction_events} (analytic; schedule spill when budgeted)"
        )

    profile["transfers"] = {
        f"{link.source}->{link.destination}": {
            "link_class": link.link_class.value,
            "bytes_per_s": link.bytes_per_s,
            "latency_s": link.latency_s,
            "measured": link.measured,
            "peer_to_peer": link.peer_to_peer,
        }
        for link in machine.links.values()
    }

    # Memory feasibility: ensure each device's peak estimate fits allocatable memory.
    for mem_name, used in sim.peak_bytes.items():
        mem = machine.memory.get(mem_name)
        if mem is None:
            continue
        limit = mem.allocatable_bytes
        if config.vram_budget_bytes is not None and mem.memory_class.value == "device_vram" and limit > 0:
            limit = min(limit, config.vram_budget_bytes)
        elif config.vram_budget_bytes is not None and mem.memory_class.value == "device_vram":
            limit = config.vram_budget_bytes
        if used > limit > 0:
            raise SpecializationError(f"Plan exceeds allocatable memory on {mem_name}: {used} > {limit}")

    if program is not None and config.activation_budget_bytes is not None:
        peak_act = program.estimate_peak_activation_bytes()
        plan.predicted_peak_bytes["activations"] = peak_act
        spill_ops = sum(
            1
            for i in executable_schedule.instructions
            if i.opcode.value == "Evict" and i.attributes.get("kind") == "activation_spill"
        )
        if peak_act > config.activation_budget_bytes or spill_ops:
            plan.notes.append(
                f"activation_peak={peak_act} budget={config.activation_budget_bytes}; "
                f"schedule activation spill ops={spill_ops}"
            )

    concurrency = _decide_concurrency(program, region_inputs, plan, machine, config)
    plan.notes.append(f"concurrency={'enabled' if concurrency.enabled else 'disabled'}: {concurrency.reason}")

    inductor_regions = sum(1 for c in compiled if str(c.get("impl", "")).startswith("torch_compile_"))
    fallback_regions = sum(1 for c in compiled if c.get("fallback"))
    if inductor_regions:
        plan.notes.append(f"torch_compile_regions={inductor_regions}")
    if fallback_regions:
        plan.notes.append(f"eager_fallback_regions={fallback_regions}")
    devices_used = set(plan.devices_used)
    has_cuda = any(d.startswith("cuda_gpu_") for d in devices_used)
    has_cpu = any(d.startswith("cpu_") or "numa" in d for d in devices_used)
    multi_accel = sum(1 for d in devices_used if d.startswith(("cuda_gpu_", "rocm_gpu_"))) >= 2
    if multi_accel:
        cross_device = "multi_gpu"
    elif has_cuda and has_cpu:
        cross_device = "cpu_gpu"
    elif has_cuda:
        cross_device = "single_gpu"
    else:
        cross_device = "cpu_only"
    validation = {
        "fingerprint_matched": True,
        "memory_feasible": True,
        "concurrency": concurrency.as_dict(),
        "backends_used": sorted({p.backend_id for p in plan.placements}),
        "simulated_makespan_s": sim.makespan_s,
        "exposed_transfer_latency_s": sim.exposed_transfer_latency_s,
        "timeline_events": len(sim.timeline),
        "regions_compiled": len(compiled),
        "regions_measured": sum(1 for p in plan.placements if p.measured),
        "regions_total": len(plan.placements),
        "scheduled_transfers": len(residency.transfers),
        "schedule_instructions": len(executable_schedule.instructions),
        "torch_compile_regions": inductor_regions,
        "eager_fallback_regions": fallback_regions,
        "cross_device_execution": cross_device,
    }
    artifact = SpecializedArtifact(
        fingerprint=current_fp,
        plan=plan,
        compiled_regions=compiled,
        profile=profile,
        validation=validation,
        bindings=bindings,
        schedule=executable_schedule,
    )
    if output_dir is not None:
        artifact.save(output_dir)
        # Invalidate notice for future fingerprint mismatch.
        (output_dir / "fingerprint").write_text(current_fp + "\n", encoding="utf-8")
    return artifact


def _seed_cpu_measurements_from_capture(
    measurements: MeasurementSet,
    capture_times: dict[str, float],
    profile_devices: list[Any],
) -> None:
    """Fill missing/failed CPU probes from the capture forward (never overwrite)."""
    if not capture_times:
        return
    cpu_devices = [
        device for device in profile_devices if str(getattr(device, "backend_id", "")) in {"cpu", "cpu_numa"}
    ]
    for device in cpu_devices:
        name = device.id.name
        backend_id = str(device.backend_id)
        for region_id, latency_s in capture_times.items():
            if latency_s <= 0 or latency_s == float("inf"):
                continue
            existing = measurements.get(region_id, name)
            if existing is not None and existing.measured and existing.latency_s < float("inf"):
                continue
            measurements.add(
                RegionMeasurement(
                    region_id=region_id,
                    device=name,
                    backend_id=backend_id,
                    latency_s=float(latency_s),
                    measured=True,
                    simulated=False,
                    notes="capture_forward_sample",
                )
            )


def _compile_one_placement(
    placement: Any,
    *,
    program: RegionProgram,
    machine: Any,
    config: CompileConfig,
    current_fp: str,
    region_inputs: dict[str, tuple[Any, ...]],
    compile_regions: bool,
) -> tuple[dict[str, Any], RegionBinding | None, dict[str, Any] | None, str | None]:
    """Compile one placement. Returns (compiled_row, binding, device_profile, missing_region)."""
    from tensortorrent.backends.base import KernelCandidate

    backend = backend_by_id(placement.backend_id)
    device = machine.compute.get(placement.device)
    if backend is None or device is None:
        raise SpecializationError(
            f"Placement for {placement.region_id} targets unknown "
            f"backend={placement.backend_id} device={placement.device}"
        )
    cand = KernelCandidate(
        region_id=placement.region_id,
        device=placement.device,
        backend_id=placement.backend_id,
        kernel_id=placement.kernel_id,
        dtype=placement.dtype,
        attributes={
            "use_torch_compile": config.use_torch_compile,
            "profile_level": config.profile_level,
            "torch_compile_backend": config.torch_compile_backend,
            "machine_fingerprint": current_fp,
        },
    )
    missing = None if placement.measured else placement.region_id
    device_profile = None
    if config.profile_level in ("competitive", "full"):
        bench = backend.benchmark(cand)
        device_profile = {
            "latency_s": bench.latency_s,
            "measured": bench.measured,
            "notes": bench.notes,
        }
    if not compile_regions:
        return (
            {
                "region_id": placement.region_id,
                "device": placement.device,
                "backend_id": placement.backend_id,
                "dtype": placement.dtype,
                "compile_skipped": True,
            },
            None,
            device_profile,
            missing,
        )
    region = program.region_by_id(placement.region_id)
    source = region_source(program, region, region_inputs.get(placement.region_id))
    compiled_region = backend.compile(source, cand)
    binding = RegionBinding(
        region=region,
        compiled=compiled_region,
        backend_id=placement.backend_id,
        device=placement.device,
    )
    row = {
        "region_id": compiled_region.region_id,
        "device": compiled_region.device,
        "backend_id": compiled_region.backend_id,
        "dtype": compiled_region.dtype,
        "torch_device": compiled_region.torch_device,
        "aten_ops": list(region.aten_ops),
        "node_count": region.node_count,
        "executable": type(compiled_region.executable).__name__,
        "impl": compiled_region.attributes.get("impl"),
        "compile_time_s": compiled_region.attributes.get("compile_time_s"),
        "fallback": compiled_region.attributes.get("fallback", False),
        "fallback_reason": compiled_region.attributes.get("fallback_reason"),
        "cache_key": compiled_region.attributes.get("cache_key"),
    }
    return row, binding, device_profile, missing


def _compile_plan_placements(
    plan: ExecutionPlan,
    *,
    program: RegionProgram | None,
    machine: Any,
    config: CompileConfig,
    current_fp: str,
    region_inputs: dict[str, tuple[Any, ...]],
    compile_regions: bool,
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, RegionBinding]]:
    """Compile plan placements; optional parallel CPU compiles when configured."""
    compiled: list[dict[str, Any]] = []
    bindings: dict[str, RegionBinding] = {}
    if program is None:
        return compiled, bindings

    def _apply(
        placement: Any,
        row: dict[str, Any],
        binding: RegionBinding | None,
        device_profile: dict[str, Any] | None,
        missing: str | None,
    ) -> None:
        if missing is not None:
            profile["missing_measurements"].append(missing)
        if device_profile is not None:
            profile["devices"][placement.device] = device_profile
        compiled.append(row)
        if binding is not None:
            bindings[placement.region_id] = binding

    def _safe_compile(placement: Any) -> tuple[dict[str, Any], RegionBinding | None, dict[str, Any] | None, str | None]:
        try:
            return _compile_one_placement(
                placement,
                program=program,
                machine=machine,
                config=config,
                current_fp=current_fp,
                region_inputs=region_inputs,
                compile_regions=compile_regions,
            )
        except SpecializationError:
            raise
        except Exception as exc:
            raise SpecializationError(
                f"Failed to specialize region {placement.region_id} on {placement.device}: {exc}"
            ) from exc

    cpu_placements = []
    serial_placements = []
    for placement in plan.placements:
        if str(placement.backend_id) in {"cpu", "cpu_numa"} and compile_regions:
            cpu_placements.append(placement)
        else:
            serial_placements.append(placement)

    for placement in serial_placements:
        _apply(placement, *_safe_compile(placement))

    workers = int(config.region_compile_workers)
    use_parallel = compile_regions and workers != 1 and len(cpu_placements) > 1
    if not use_parallel:
        for placement in cpu_placements:
            _apply(placement, *_safe_compile(placement))
        return compiled, bindings

    max_workers = workers if workers > 0 else min(len(cpu_placements), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_safe_compile, placement): placement for placement in cpu_placements}
        by_region = {}
        for future in as_completed(futures):
            placement = futures[future]
            by_region[placement.region_id] = future.result()
        for placement in cpu_placements:
            _apply(placement, *by_region[placement.region_id])
    return compiled, bindings


def _prefetch_variants(estimated: int, *, streaming: bool, max_distance: int = 8) -> list[int]:
    """Bounded prefetch distances around the analytic estimate.

    Streaming always includes ``0``. Under a bounded breadth-first DES budget,
    the analytically preferred distance is first so every finalist gets its
    primary schedule before exploratory alternatives consume slots.

    Priority (clamp + dedup): estimated, 0, 1, estimated-1, estimated+1.
    When the estimate is 0 (hard disable / no overlap), do not explore positives.
    """
    if not streaming:
        return [0]
    est = max(0, min(int(estimated), max_distance))
    if est == 0:
        return [0]
    ordered = [est, 0, 1, max(0, est - 1), min(max_distance, est + 1)]
    out: list[int] = []
    seen: set[int] = set()
    for v in ordered:
        if 0 <= v <= max_distance and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _simulated_objective_score(
    *,
    config: CompileConfig,
    makespan_s: float,
    initiation_interval_s: float,
    peak_bytes: dict[str, int],
    machine: ResourceGraph,
) -> float:
    """Authoritative finalist score from DES metrics (lower better)."""
    from tensortorrent.config import Objective

    latency = float(makespan_s)
    cycle = max(float(initiation_interval_s), 1e-12)
    peak_total = float(sum(peak_bytes.values()))
    pressure = 0.0
    for name, used in peak_bytes.items():
        mem = machine.memory.get(name)
        if mem is None:
            # Device-keyed peaks from analytic search.
            device = machine.compute.get(name)
            if device is None:
                if used > 0:
                    pressure += 1.0
                continue
            cap = sum(int(machine.memory[m].allocatable_bytes) for m in device.memory_affinity if m in machine.memory)
        else:
            cap = int(mem.allocatable_bytes)
        if config.vram_budget_bytes is not None and (
            (mem is not None and mem.memory_class.value == "device_vram") or (machine.compute.get(name) is not None)
        ):
            cap = min(cap, config.vram_budget_bytes) if cap > 0 else config.vram_budget_bytes
        if cap > 0:
            pressure += used / cap
        elif used > 0:
            pressure += 1.0

    if config.objective == Objective.LATENCY:
        return latency + 1e-6 * cycle + 1e-9 * pressure
    if config.objective == Objective.THROUGHPUT:
        return cycle + 1e-3 * latency + 1e-9 * pressure
    if config.objective == Objective.MEMORY:
        return peak_total + 1e-3 * pressure + 1e-9 * latency
    if config.objective == Objective.BALANCED:
        return latency + cycle + 0.05 * pressure
    weights = config.objective_weights
    return (
        weights.get("latency", 0.0) * latency
        + weights.get("throughput", 0.0) * cycle
        + weights.get("memory", 0.0) * pressure
    )


def _host_or_pinned_pressure(outcome: Any) -> bool:
    """True when a DES/status rejection looks like host/pinned capacity pressure.

    Intentionally narrow: device-VRAM / generic ``infeasible`` must not trigger
    pageable recovery (that burns recovery slots and can starve real pinned fails).

    Batch DES reports ``status=infeasible_memory`` with the resource in ``memory``.
    Single-path raises include the resource name in the exception message.
    """
    if isinstance(outcome, dict):
        status = str(outcome.get("status") or "").lower()
        mem = str(outcome.get("memory") or "").lower()
        if status == "infeasible_memory":
            hostish = (
                "pinned",
                "host",
                "numa",
                "pageable",
                "system_ram",
                "host_ram",
                "host_memory",
            )
            return any(k in mem for k in hostish)
        blob = " ".join(str(outcome.get(k) or "") for k in ("status", "error", "message", "memory", "detail")).lower()
    else:
        blob = str(outcome).lower()
    keys = (
        "pinned",
        "host_ram",
        "host memory",
        "host_memory",
        "pageable",
        "numa",
        "host staging",
        "host_staging",
    )
    return any(k in blob for k in keys)


def _plan_analytic_rank(plan: ExecutionPlan, default: int = 0) -> int:
    """Real analytical rank (not finalist shortlist index)."""
    raw = plan.search_statistics.get("analytic_rank")
    if raw is None:
        raw = plan.search_statistics.get("search_rank", default)
    return int(raw) if isinstance(raw, (int, float)) else default


def _plan_finalist_rank(plan: ExecutionPlan, selected: list[ExecutionPlan], default: int = 0) -> int:
    """Shortlist position among DES placement finalists."""
    raw = plan.search_statistics.get("finalist_rank")
    if raw is None:
        return next((i for i, p in enumerate(selected) if p is plan), default)
    return int(raw) if isinstance(raw, (int, float)) else default


def _scores_near_equal(
    a: float,
    b: float,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> bool:
    """True when ``a`` is within float-noise of fixed anchor ``b``.

    Only for Stage-2 membership against a single best raw DES score.
    Never used pairwise inside a sort comparator (non-transitive).
    """
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def _des_candidate_sort_key(c: dict[str, Any]) -> tuple[float, int, int, int, int, int]:
    """Strict diagnostic order: raw DES score, then stable secondary keys."""
    return (
        float(c["raw_score"]),
        int(c["analytic_rank"]),
        int(c["finalist_rank"]),
        int(c["pref_distance"]),
        int(c["pref_tie"]),
        int(c["variant_idx"]),
    )


def _select_des_winner(
    candidates: list[dict[str, Any]],
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> tuple[dict[str, Any], dict[int, int]]:
    """Two-stage DES winner: min raw score, then tie-break near that anchor only.

    Stage 1 — ``best_raw_score = min(raw_scores)`` (no secondary keys).
    Stage 2 — among candidates ``isclose`` to that fixed best, pick by
    analytic_rank → finalist_rank → prefetch keys → variant index.

    Returns ``(winner, simulated_rank_by_variant_idx)`` where ranks come from
    a strict raw-score sort (never fuzzy pairwise compare).
    """
    if not candidates:
        raise ValueError("DES winner selection requires at least one feasible candidate")
    best_raw = min(float(c["raw_score"]) for c in candidates)
    tie_candidates = [
        c for c in candidates if _scores_near_equal(float(c["raw_score"]), best_raw, rel_tol=rel_tol, abs_tol=abs_tol)
    ]
    winner = min(
        tie_candidates,
        key=lambda c: (
            int(c["analytic_rank"]),
            int(c["finalist_rank"]),
            int(c["pref_distance"]),
            int(c["pref_tie"]),
            int(c["variant_idx"]),
        ),
    )
    ordered = sorted(candidates, key=_des_candidate_sort_key)
    simulated_rank_by_variant = {int(c["variant_idx"]): rank for rank, c in enumerate(ordered)}
    return winner, simulated_rank_by_variant


def _merge_batch_sim_stats(acc: dict[str, Any], batch: dict[str, Any]) -> None:
    """Accumulate authoritative Rust batch-DES statistics across DES calls."""
    acc["simulator_workers_requested"] = int(
        batch.get("simulator_workers_requested", acc.get("simulator_workers_requested", 0)) or 0
    )
    acc["simulator_workers_available"] = max(
        int(acc.get("simulator_workers_available", 1) or 1),
        int(batch.get("simulator_workers_available", 1) or 1),
    )
    if batch.get("parallel_simulation_used"):
        acc["parallel_simulation_used"] = True
        acc["simulator_workers_used"] = max(
            int(acc.get("simulator_workers_used", 1) or 1),
            int(batch.get("simulator_workers_used", 1) or 1),
        )
    elif not acc.get("parallel_simulation_used"):
        acc["simulator_workers_used"] = 1


def _recompute_winner_metadata(
    win_plan: ExecutionPlan,
    finalists: list[ExecutionPlan],
    *,
    machine: ResourceGraph,
    config: CompileConfig,
    prefetch: int,
    sim: Any,
) -> None:
    """Derive public plan fields from the DES-selected winner (never analytic #1)."""
    from tensortorrent.backends.communication import select_communication_backend
    from tensortorrent.planner.maximal import (
        _decide_resources,
        _eligible_compute,
        _strategy_name,
    )

    win_plan.prefetch_distance = int(prefetch)
    win_plan.devices_used = tuple(sorted({p.device for p in win_plan.placements}))
    win_plan.communication_backend = select_communication_backend(win_plan.devices_used).backend_id
    used = set(win_plan.devices_used)
    solo_latencies: dict[str, float] = {}
    for cand in finalists:
        if len(cand.devices_used) == 1:
            solo_latencies[cand.devices_used[0]] = float(cand.predicted_latency_s)
    eligible = _eligible_compute(machine, config)
    device_objs = tuple(device for device in eligible if device.id.name in used)
    # DES may select devices filtered from eligibility; still classify from the graph.
    if len(device_objs) < len(used):
        have = {d.id.name for d in device_objs}
        extras = []
        for name in sorted(used - have):
            dev = machine.compute.get(name) if hasattr(machine, "compute") else None
            if dev is not None:
                extras.append(dev)
        device_objs = device_objs + tuple(extras)
    win_plan.strategy = _strategy_name(device_objs)
    latency = float(getattr(sim, "makespan_s", None) or win_plan.predicted_latency_s)
    win_plan.decisions = _decide_resources(machine, eligible, used, latency, solo_latencies)
    # Drop stale notes that belong to other finalists' staging/prefetch paths.
    win_plan.notes = [n for n in win_plan.notes if not n.startswith("host_staging=")]


def _select_finalist_by_simulation(
    finalists: list[ExecutionPlan],
    *,
    program: RegionProgram | None,
    streaming: bool,
    activation_budget_bytes: int | None,
    machine: ResourceGraph,
    config: CompileConfig,
) -> tuple[ExecutionPlan, Any, Any, int, dict[str, Any]]:
    """Build schedule variants for finalists, batch-DES, pick objective winner.

    Variant slots are allocated breadth-first across finalists so every selected
    placement gets a primary DES evaluation before optional prefetch/staging extras.
    """
    from tensortorrent.config import Objective
    from tensortorrent.runtime.residency import attach_residency_to_plan
    from tensortorrent.runtime.schedule import (
        build_executable_schedule,
        hoist_resident_parameter_transfers,
        schedule_matches_plan,
        validate_schedule,
        validate_schedule_resources,
        validate_schedule_tensor_sizes,
    )
    from tensortorrent.runtime.simulator.discrete_event import (
        SimulationResult,
        simulate_schedules_with_stats,
    )

    t_select0 = perf_counter()
    t_build = 0.0
    # Cap total DES schedule variants (not just placement finalists).
    cap = max(1, int(config.planner_des_candidates) * 3)
    # Never spend multiple slots on one placement while another gets zero.
    placement_budget = min(len(finalists), max(1, int(config.planner_des_candidates)))
    selected_finalists = finalists[:placement_budget]

    def _try_build(
        plan: ExecutionPlan,
        residency: Any,
        prefetch: int,
        *,
        force_pageable: bool,
    ) -> Any | None:
        nonlocal t_build
        t0 = perf_counter()
        try:
            sched = build_executable_schedule(
                plan,
                residency,
                streaming=streaming,
                prefetch_distance=prefetch,
                program=program,
                activation_budget_bytes=activation_budget_bytes,
                machine=machine,
                force_pageable_host_staging=force_pageable,
            )
            errors = (
                schedule_matches_plan(sched, plan)
                or validate_schedule(sched)
                or validate_schedule_resources(sched, machine)
                or (validate_schedule_tensor_sizes(sched) if program is not None else [])
            )
            if errors:
                return None
            return sched
        except Exception as exc:  # noqa: BLE001 - skip bad variants
            logger.debug("finalist schedule build skipped: %s", exc)
            return None
        finally:
            t_build += perf_counter() - t0

    residencies = [attach_residency_to_plan(plan, program) for plan in selected_finalists]
    pref_lists = [_prefetch_variants(plan.prefetch_distance, streaming=streaming) for plan in selected_finalists]
    max_rounds = max((len(p) for p in pref_lists), default=1)

    # (finalist_index, plan, prefetch, pageable, schedule)
    variants: list[tuple[int, ExecutionPlan, int, bool, Any]] = []
    seen_keys: set[tuple[int, int, bool]] = set()

    # Breadth-first: round r = r-th prefetch for every finalist before r+1.
    for round_i in range(max_rounds):
        for fi, plan in enumerate(selected_finalists):
            if len(variants) >= cap:
                break
            prefs = pref_lists[fi]
            if round_i >= len(prefs):
                continue
            prefetch = prefs[round_i]
            key = (fi, prefetch, False)
            if key in seen_keys:
                continue
            sched = _try_build(plan, residencies[fi], prefetch, force_pageable=False)
            if sched is not None:
                seen_keys.add(key)
                variants.append((fi, plan, prefetch, False, sched))
        if len(variants) >= cap:
            break

    if not variants:
        raise SpecializationError("No feasible executable schedule among planner finalists")

    machine.allow_host_staged_transfers = bool(config.allow_host_staged_transfers)

    def _des_view(sched: Any) -> Any:
        # Non-streaming: score the same residency policy the runtime will use.
        if streaming:
            return sched
        from tensortorrent.compile.fit import should_hoist_resident_parameters

        state_bytes = int(program.total_state_bytes()) if program is not None else 0
        if program is not None and not should_hoist_resident_parameters(
            config, state_bytes=state_bytes, machine=machine
        ):
            return sched
        return hoist_resident_parameter_transfers(sched, drop_parameter_evicts=True)

    def _simulate(batch: list[Any]) -> tuple[list[Any], float, dict[str, Any]]:
        t0 = perf_counter()
        workers = int(config.planner_workers)
        outs, sim_stats = simulate_schedules_with_stats(batch, machine, workers=workers)
        return outs, perf_counter() - t0, sim_stats

    batch_sim_stats: dict[str, Any] = {
        "parallel_simulation_used": False,
        "simulator_workers_requested": int(config.planner_workers),
        "simulator_workers_available": 1,
        "simulator_workers_used": 1,
    }
    outcomes, batch_s, first_stats = _simulate([_des_view(v[4]) for v in variants])
    _merge_batch_sim_stats(batch_sim_stats, first_stats)

    # DES-driven pageable fallback: only for pinned variants rejected on host pressure.
    # Replace failed pinned slots — do not require spare cap headroom (pinned fill must
    # not starve pageable recovery). At most one pageable recovery per finalist, and at
    # most ``placement_budget`` recoveries in a batch.
    if config.allow_host_staged_transfers and streaming:
        pageable_batch: list[tuple[int, ExecutionPlan, int, bool, Any]] = []
        recovery_cap = max(1, placement_budget)
        recovered_finalists: set[int] = set()
        for i, outcome in enumerate(outcomes):
            if len(pageable_batch) >= recovery_cap:
                break
            if isinstance(outcome, SimulationResult):
                continue
            if not _host_or_pinned_pressure(outcome):
                continue
            fi, plan_i, pref_i, pageable_i, _ = variants[i]
            if pageable_i or fi in recovered_finalists:
                continue
            key = (fi, pref_i, True)
            if key in seen_keys:
                continue
            sched = _try_build(plan_i, residencies[fi], pref_i, force_pageable=True)
            if sched is not None:
                seen_keys.add(key)
                recovered_finalists.add(fi)
                pageable_batch.append((fi, plan_i, pref_i, True, sched))
        if pageable_batch:
            page_out, page_s, page_stats = _simulate([_des_view(v[4]) for v in pageable_batch])
            batch_s += page_s
            _merge_batch_sim_stats(batch_sim_stats, page_stats)
            variants.extend(pageable_batch)
            outcomes.extend(page_out)

    scored: list[dict[str, Any]] = []
    for i, outcome in enumerate(outcomes):
        if not isinstance(outcome, SimulationResult):
            continue
        score = _simulated_objective_score(
            config=config,
            makespan_s=outcome.makespan_s,
            initiation_interval_s=float(outcome.initiation_interval_s or 0.0),
            peak_bytes=outcome.peak_bytes,
            machine=machine,
        )
        _, plan_i, pref_i, _, _ = variants[i]
        analytic_rank_i = _plan_analytic_rank(plan_i, default=i)
        finalist_rank_i = _plan_finalist_rank(plan_i, selected_finalists, default=analytic_rank_i)
        analytic_pref = int(plan_i.prefetch_distance)
        # Raw DES score is authoritative. Secondary keys apply only in Stage 2
        # against the fixed best raw score (never pairwise fuzzy sort).
        pref_distance = abs(pref_i - analytic_pref)
        pref_tie = pref_i if config.objective == Objective.MEMORY else -pref_i
        scored.append(
            {
                "raw_score": score,
                "analytic_rank": analytic_rank_i,
                "finalist_rank": finalist_rank_i,
                "pref_distance": pref_distance,
                "pref_tie": pref_tie,
                "variant_idx": i,
                "outcome": outcome,
            }
        )

    if not scored:
        reject_reasons = [
            str(o.get("status") or o.get("error") or "infeasible") for o in outcomes if isinstance(o, dict)
        ]
        detail = "; ".join(reject_reasons[:5]) if reject_reasons else "no feasible SimulationResult"
        raise SpecializationError(
            f"All {len(variants)} DES schedule variants infeasible after pageable recovery: {detail}"
        )

    winner, simulated_rank_by_variant = _select_des_winner(scored)
    best_idx = int(winner["variant_idx"])
    best_sim = winner["outcome"]
    best_score = float(winner["raw_score"])

    _, win_plan, win_prefetch, win_pageable, win_sched = variants[best_idx]
    analytic_rank = _plan_analytic_rank(win_plan, default=0)
    finalist_rank = _plan_finalist_rank(win_plan, selected_finalists, default=analytic_rank)
    simulated_rank = int(simulated_rank_by_variant.get(best_idx, 0))
    _recompute_winner_metadata(
        win_plan,
        finalists,
        machine=machine,
        config=config,
        prefetch=win_prefetch,
        sim=best_sim,
    )
    if win_pageable:
        win_plan.notes.append("host_staging=pageable_after_des_reject")
    win_plan.search_statistics["analytic_rank"] = analytic_rank
    win_plan.search_statistics["finalist_rank"] = finalist_rank
    win_plan.search_statistics["simulated_rank"] = simulated_rank
    ok_plans = {id(variants[i][1]) for i, o in enumerate(outcomes) if isinstance(o, SimulationResult)}
    # Carry planner worker stats from the native shortlist when present.
    planner_stats: dict[str, Any] = {
        k: win_plan.search_statistics[k]
        for k in (
            "planner_workers_requested",
            "planner_workers_available",
            "planner_workers_used",
            "planner_pool_threads",
            "parallel_search_used",
            "parallel_beam_used",
        )
        if k in win_plan.search_statistics
    }
    stats = {
        "finalists_simulated": len(ok_plans),
        "placement_finalists_considered": len(selected_finalists),
        "schedule_variants_simulated": len(variants),
        "parallel_simulation_used": bool(batch_sim_stats.get("parallel_simulation_used")),
        "simulator_workers_requested": batch_sim_stats.get("simulator_workers_requested"),
        "simulator_workers_available": batch_sim_stats.get("simulator_workers_available"),
        "simulator_workers_used": batch_sim_stats.get("simulator_workers_used"),
        "batch_simulation_s": batch_s,
        "schedule_build_s": t_build,
        "finalist_selection_s": perf_counter() - t_select0,
        "winning_analytic_rank": analytic_rank,
        "winning_finalist_rank": finalist_rank,
        "winning_simulated_rank": simulated_rank,
        "simulated_objective_score": best_score,
        **planner_stats,
    }
    if analytic_rank != 0:
        stats["simulator_changed_winner"] = f"analytic_rank={analytic_rank}"
        logger.debug("DES changed planner winner: analytic_rank=%s", analytic_rank)
    win_plan.finalist_plans = finalists
    return win_plan, win_sched, best_sim, win_prefetch, stats


def _schedule_and_simulate(
    plan: ExecutionPlan,
    residency: Any,
    *,
    streaming: bool,
    program: RegionProgram | None,
    activation_budget_bytes: int | None,
    machine: ResourceGraph,
) -> tuple[Any, Any, int]:
    """Build + validate + DES-simulate one schedule (ratchet prefetch on pressure)."""
    from tensortorrent.errors import MemoryCapacityError
    from tensortorrent.runtime.schedule import (
        build_executable_schedule,
        schedule_matches_plan,
        validate_schedule,
        validate_schedule_resources,
        validate_schedule_tensor_sizes,
    )
    from tensortorrent.runtime.simulator.discrete_event import simulate_schedule

    prefetch = int(plan.prefetch_distance)
    force_pageable = False
    while True:
        executable_schedule = build_executable_schedule(
            plan,
            residency,
            streaming=streaming,
            prefetch_distance=prefetch,
            program=program,
            activation_budget_bytes=activation_budget_bytes,
            machine=machine,
            force_pageable_host_staging=force_pageable,
        )
        schedule_errors = schedule_matches_plan(executable_schedule, plan)
        if schedule_errors:
            raise SpecializationError(f"Executable schedule inconsistent with plan: {schedule_errors}")
        structural_errors = validate_schedule(executable_schedule)
        if structural_errors:
            raise SpecializationError(f"Executable schedule failed validation: {structural_errors}")
        resource_errors = validate_schedule_resources(executable_schedule, machine)
        if resource_errors:
            raise SpecializationError(f"Executable schedule references unknown resources: {resource_errors}")
        size_errors = validate_schedule_tensor_sizes(executable_schedule) if program is not None else []
        if size_errors:
            raise SpecializationError(f"Executable schedule lacks exact tensor sizes: {size_errors}")
        try:
            sim = simulate_schedule(executable_schedule, machine)
        except MemoryCapacityError as exc:
            msg = str(exc)
            if prefetch > 0:
                logger.warning(
                    "schedule simulate infeasible at prefetch_distance=%s (%s); retrying with %s",
                    prefetch,
                    exc,
                    prefetch - 1,
                )
                prefetch -= 1
                plan.notes.append(f"prefetch_distance_reduced_to={prefetch} after pinned/host pressure")
                continue
            if streaming and not force_pageable and "pinned" in msg.lower():
                logger.warning(
                    "schedule simulate infeasible after prefetch ratchet (%s); rebuilding with pageable host staging",
                    exc,
                )
                force_pageable = True
                plan.notes.append("host_staging=pageable_after_pinned_pressure")
                continue
            detail = "prefetch_distance=0"
            if force_pageable:
                detail += " and pageable host staging"
            raise MemoryCapacityError(
                f"{msg}. Schedule still exceeds allocatable memory after {detail}. "
                "Raise ram_budget_bytes / vram_budget_bytes, lower "
                "CompileConfig.max_region_nodes so each region fits device+host staging, "
                "or free host RAM / pinned headroom."
            ) from exc
        return executable_schedule, sim, prefetch


def _passthrough_specialization(
    program: RegionProgram,
    fingerprint: str,
    output_dir: Path | None,
) -> SpecializedArtifact:
    """Specialize a graph that returns its inputs or state without computing.

    There is nothing to place, measure or overlap, so the plan is empty by
    construction and the runtime resolves outputs straight from its environment.
    """
    from tensortorrent.runtime.schedule import ExecutableSchedule

    plan = ExecutionPlan(
        graph_name=program.graph_name,
        fingerprint=fingerprint,
        objective="latency",
        placements=[],
        decisions=[],
        devices_used=(),
        communication_backend="none",
        predicted_latency_s=0.0,
        strategy="pass_through",
        notes=["graph returns inputs or state directly; no compute regions to place"],
    )
    concurrency = ConcurrencyDecision(enabled=False, workers=1, reason="graph has no compute regions")
    empty_schedule = ExecutableSchedule(
        graph_name=program.graph_name,
        fingerprint=fingerprint,
        instructions=(),
        notes=("pass_through: empty instruction DAG",),
    )
    artifact = SpecializedArtifact(
        fingerprint=fingerprint,
        plan=plan,
        compiled_regions=[],
        profile={"devices": {}, "transfers": {}, "missing_measurements": [], "region_measurements": {}},
        validation={
            "fingerprint_matched": True,
            "memory_feasible": True,
            "concurrency": concurrency.as_dict(),
            "backends_used": [],
            "regions_compiled": 0,
            "regions_measured": 0,
            "regions_total": 0,
            "pass_through": True,
        },
        bindings={},
        schedule=empty_schedule,
    )
    if output_dir is not None:
        artifact.save(output_dir)
        (output_dir / "fingerprint").write_text(fingerprint + "\n", encoding="utf-8")
    return artifact


def concurrency_budget(plan: ExecutionPlan, machine: ResourceGraph) -> int:
    """Upper bound on simultaneous regions the selected devices can absorb.

    Distinct accelerator devices contribute one worker each. A CPU pool can host
    as many regions as it has cores (measurement may still pick fewer). For a
    mixed CPU+accelerator plan the useful overlap is typically one wave across
    device classes, so the budget is capped at the class count once bindings
    exist — still an upper bound for the measurement / fusion bake-off.
    """
    total = 0
    has_cpu = False
    accel = 0
    for name in plan.devices_used:
        device = machine.compute.get(name)
        if device is None or device.backend_id != "cpu":
            total += 1
            accel += 1
            continue
        has_cpu = True
        total += max(2, device.concurrency_limit)
    if has_cpu and accel:
        # Cross-device overlap: one worker per accelerator + one CPU sibling.
        return max(2, accel + 1)
    return max(1, total)


def _plan_is_cpu_accelerator(plan: ExecutionPlan) -> bool:
    devices = set(plan.devices_used)
    has_cpu = any(d == "cpu" or d.startswith("cpu_") or "numa" in d for d in devices)
    has_accel = any(d.startswith(("cuda_gpu_", "rocm_gpu_")) for d in devices)
    return has_cpu and has_accel


def _decide_concurrency(
    program: RegionProgram | None,
    region_inputs: dict[str, tuple[Any, ...]],
    plan: ExecutionPlan,
    machine: ResourceGraph,
    config: CompileConfig,
) -> ConcurrencyDecision:
    """Decide whether independent regions should overlap, by measurement.

    CPU-submodule timing is not representative for CPU+accelerator placements:
    both branches would be timed on host FX modules and often "lose" to
    sequential CPU contention. Those plans keep overlap for the full executor
    bake-off instead.
    """
    if not config.allow_concurrent_regions:
        return ConcurrencyDecision(
            enabled=False, workers=1, reason="disabled by CompileConfig.allow_concurrent_regions"
        )
    if config.max_concurrent_regions > 0:
        return ConcurrencyDecision(
            enabled=config.max_concurrent_regions > 1,
            workers=config.max_concurrent_regions,
            reason=(f"forced to {config.max_concurrent_regions} workers by CompileConfig.max_concurrent_regions"),
        )
    if program is None or not region_inputs:
        return ConcurrencyDecision(enabled=False, workers=1, reason="no example inputs available to measure with")
    budget = concurrency_budget(plan, machine)
    levels = dependency_levels(program) if program is not None else None
    widest = levels.widest() if levels is not None else ()
    if _plan_is_cpu_accelerator(plan) and len(widest) >= 2 and budget > 1:
        workers = min(budget, max(2, len(widest)))
        return ConcurrencyDecision(
            enabled=True,
            workers=workers,
            group=widest,
            reason=(
                "heterogeneous CPU+accelerator plan retained for full executor "
                "benchmark; CPU-only region microbenchmark is not representative"
            ),
            measured=False,
            intraop_threads=0,
        )
    from tensortorrent.compile import pipeline as _pipeline

    return _pipeline.measure_concurrency_benefit(
        program, region_inputs, max_workers=budget, iters=max(1, config.region_measure_iters)
    )


def _planning_storage_bandwidth(machine: ResourceGraph) -> float | None:
    """Best discovered storage bandwidth for adaptive prefetch planning."""
    measured = [
        float(link.bytes_per_s)
        for link in machine.links.values()
        if link.link_class.value == "storage" and link.measured and link.bytes_per_s and link.bytes_per_s > 0
    ]
    if measured:
        return max(measured)
    declared = [
        float(memory.bandwidth_bytes_per_s)
        for memory in machine.memory.values()
        if memory.memory_class.value in {"nvme", "disk_cache"}
        and memory.bandwidth_bytes_per_s
        and memory.bandwidth_bytes_per_s > 0
    ]
    return max(declared) if declared else None
