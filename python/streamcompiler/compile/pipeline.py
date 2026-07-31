"""Two-stage compilation: portable artifact + machine specialization."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from streamcompiler.backends import backend_by_id
from streamcompiler.compile.concurrency import ConcurrencyDecision, measure_concurrency_benefit
from streamcompiler.compile.measure import (
    MeasurementSet,
    capture_region_inputs,
    measure_regions_on_devices,
    region_source,
)
from streamcompiler.compile.regions import RegionBinding, RegionProgram
from streamcompiler.config import CompileConfig
from streamcompiler.errors import SpecializationError
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.hardware.fingerprint import machine_fingerprint
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode, TensorMeta
from streamcompiler.ir.resource_graph import ResourceGraph
from streamcompiler.planner.maximal import ExecutionPlan, plan_execution

if TYPE_CHECKING:
    from streamcompiler.runtime.module import CompiledModule
    from streamcompiler.runtime.schedule import ExecutableSchedule
from streamcompiler.storage.pack import pack_state_dict


@dataclass
class PortableArtifact:
    """Hardware-independent compilation product."""

    name: str
    ir: HeterogeneousGraph
    alias_groups: dict[str, str] = field(default_factory=dict)
    liveness: dict[str, tuple[int | None, int | None]] = field(default_factory=dict)
    candidate_partitions: list[list[str]] = field(default_factory=list)
    packed_model_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    program: RegionProgram | None = None
    """Executable regions. Present in-process; reconstructed from ``exported.pt2`` on load."""
    exported: Any = None
    """The captured ``ExportedProgram``, saved separately because it is not JSON."""

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "ir": {
                "name": self.ir.name,
                "tensors": {k: asdict(v) for k, v in self.ir.tensors.items()},
                "instructions": [asdict(i) for i in self.ir.instructions],
                "parameters": list(self.ir.parameters),
                "outputs": list(self.ir.outputs),
                "repeated_blocks": [list(b) for b in self.ir.repeated_blocks],
                "metadata": self.ir.metadata,
            },
            "alias_groups": self.alias_groups,
            "liveness": {k: list(v) for k, v in self.liveness.items()},
            "candidate_partitions": self.candidate_partitions,
            "packed_model_path": self.packed_model_path,
            "metadata": self.metadata,
        }
        path = directory / "portable.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        (directory / "MANIFEST").write_text(
            "streamcompiler-portable-artifact-v1\n"
            f"name={self.name}\n"
            "stages=exported_graph,heterogeneous_ir,"
            "alias_liveness,packed_model,candidate_partitions,hw_independent_metadata\n",
            encoding="utf-8",
        )
        return path

    @staticmethod
    def load(directory: Path) -> PortableArtifact:
        payload = json.loads((directory / "portable.json").read_text(encoding="utf-8"))
        ir_data = payload["ir"]
        ir = HeterogeneousGraph(
            name=ir_data["name"],
            parameters=tuple(ir_data.get("parameters", [])),
            outputs=tuple(ir_data.get("outputs", [])),
            repeated_blocks=tuple(tuple(b) for b in ir_data.get("repeated_blocks", [])),
            metadata=ir_data.get("metadata", {}),
        )
        for tdata in ir_data.get("tensors", {}).values():
            ir.add_tensor(TensorMeta(**tdata))
        for idata in ir_data.get("instructions", []):
            opcode = idata["opcode"]
            if not isinstance(opcode, OpCode):
                idata = dict(idata)
                idata["opcode"] = OpCode(opcode)
            ir.add_instruction(Instruction(**idata))
        return PortableArtifact(
            name=payload["name"],
            ir=ir,
            alias_groups=payload.get("alias_groups", {}),
            liveness={k: (v[0], v[1]) for k, v in payload.get("liveness", {}).items()},
            candidate_partitions=payload.get("candidate_partitions", []),
            packed_model_path=payload.get("packed_model_path"),
            metadata=payload.get("metadata", {}),
        )


@dataclass
class SpecializedArtifact:
    """Machine-specific execution plan and compiled region stubs."""

    fingerprint: str
    plan: ExecutionPlan
    compiled_regions: list[dict[str, Any]] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    bindings: dict[str, RegionBinding] = field(default_factory=dict)
    """Live executables per region. Not serialized; rebuilt by re-specializing."""
    schedule: ExecutableSchedule | None = None
    """Shared executable schedule when built."""

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": self.fingerprint,
            "plan": {
                "graph_name": self.plan.graph_name,
                "fingerprint": self.plan.fingerprint,
                "objective": self.plan.objective,
                "placements": [asdict(p) for p in self.plan.placements],
                "decisions": [asdict(d) for d in self.plan.decisions],
                "devices_used": list(self.plan.devices_used),
                "communication_backend": self.plan.communication_backend,
                "predicted_latency_s": self.plan.predicted_latency_s,
                "predicted_peak_bytes": self.plan.predicted_peak_bytes,
                "strategy": self.plan.strategy,
                "notes": self.plan.notes,
            },
            "compiled_regions": self.compiled_regions,
            "profile": self.profile,
            "validation": self.validation,
            "executable_schedule": None if self.schedule is None else self.schedule.as_dict(),
        }
        path = directory / "specialized.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path


def portable_compile_from_ir(
    ir: HeterogeneousGraph,
    *,
    state_dict: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    program: RegionProgram | None = None,
    exported: Any = None,
) -> PortableArtifact:
    """Produce a portable artifact from an already-lowered heterogeneous IR."""
    from streamcompiler.ir.alias import run_alias_analysis
    from streamcompiler.ir.liveness import run_liveness_analysis
    from streamcompiler.runtime.buffer_reuse import plan_buffer_reuse

    alias_result = run_alias_analysis(ir)
    alias = alias_result.groups
    liveness_result = run_liveness_analysis(ir)
    liveness = liveness_result.intervals
    reuse = plan_buffer_reuse(ir, liveness_result, alias_result)
    partitions: list[list[str]] = []
    if ir.repeated_blocks:
        partitions = [list(b) for b in ir.repeated_blocks]
    else:
        compute = [i.name for i in ir.compute_regions()]
        if compute:
            partitions = [compute]

    packed_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if state_dict:
            pack_state_dict(state_dict, output_dir / "model.pack")
            # Relative to the artifact directory so reload cannot escape via absolute paths.
            packed_path = "model.pack"

    artifact = PortableArtifact(
        name=ir.name,
        ir=ir,
        alias_groups=alias,
        liveness=liveness,
        candidate_partitions=partitions,
        packed_model_path=packed_path,
        metadata={
            "stage": "portable",
            "created_unix": time.time(),
            "hardware_independent": True,
            "region_count": len(ir.compute_regions()),
            "buffer_reuse": reuse.as_dict(),
            "liveness_mismatches": list(liveness_result.mismatches),
            "alias_view_of": dict(alias_result.view_of),
        },
        program=program,
        exported=exported,
    )
    if output_dir is not None:
        artifact.save(output_dir)
    return artifact


def specialize_for_machine(
    portable: PortableArtifact,
    *,
    config: CompileConfig | None = None,
    output_dir: Path | None = None,
    example_inputs: list[Any] | None = None,
    machine: Any | None = None,
    profile_feedback: Any | None = None,
    measurements: MeasurementSet | None = None,
) -> SpecializedArtifact:
    """Deployment-time specialization against the actual machine resource graph."""
    config = config or CompileConfig()
    machine = machine if machine is not None else discover_resource_graph()
    current_fp = machine.fingerprint or machine_fingerprint()
    program = portable.program

    region_inputs: dict[str, tuple[Any, ...]] = {}
    if measurements is None:
        measurements = MeasurementSet()
        if program is not None and example_inputs is not None:
            region_inputs = capture_region_inputs(program, example_inputs)
            if config.measure_regions:
                profile_devices = []
                for device in machine.compute.values():
                    backend_id = str(device.backend_id)
                    is_cpu = backend_id in {"cpu", "cpu_numa"}
                    if is_cpu and not config.allow_cpu:
                        continue
                    if not is_cpu and not config.allow_gpu:
                        continue
                    profile_devices.append(device)
                measurements = measure_regions_on_devices(
                    program,
                    region_inputs,
                    profile_devices,
                    iters=config.region_measure_iters,
                )
    elif program is not None and example_inputs is not None:
        region_inputs = capture_region_inputs(program, example_inputs)

    if profile_feedback is not None and hasattr(profile_feedback, "merge_into_measurements"):
        measurements = profile_feedback.merge_into_measurements(measurements)

    if program is not None and not program.regions:
        return _passthrough_specialization(program, current_fp, output_dir)

    plan = plan_execution(portable.ir, machine, config, measurements)
    from streamcompiler.planner.collectives import plan_collectives
    from streamcompiler.planner.local_search import rebalance_partitions, refine_prefetch_distance

    # Refine placements before compiling so bindings match the final plan devices.
    plan = rebalance_partitions(plan)
    plan = refine_prefetch_distance(plan, distance=config.prefetch_distance)

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

    for placement in plan.placements:
        backend = backend_by_id(placement.backend_id)
        device = machine.compute.get(placement.device)
        if backend is None or device is None:
            raise SpecializationError(
                f"Placement for {placement.region_id} targets unknown "
                f"backend={placement.backend_id} device={placement.device}"
            )
        from streamcompiler.backends.base import KernelCandidate

        cand = KernelCandidate(
            region_id=placement.region_id,
            device=placement.device,
            backend_id=placement.backend_id,
            kernel_id=placement.kernel_id,
            dtype=placement.dtype,
            attributes={
                "use_torch_compile": config.use_torch_compile,
                "torch_compile_backend": config.torch_compile_backend,
                "machine_fingerprint": current_fp,
            },
        )
        if not placement.measured:
            profile["missing_measurements"].append(placement.region_id)
        try:
            if config.profile_level in ("competitive", "full"):
                bench = backend.benchmark(cand)
                profile["devices"][placement.device] = {
                    "latency_s": bench.latency_s,
                    "measured": bench.measured,
                    "notes": bench.notes,
                }
            if program is None:
                continue
            region = program.region_by_id(placement.region_id)
            source = region_source(program, region, region_inputs.get(placement.region_id))
            compiled_region = backend.compile(source, cand)
            bindings[placement.region_id] = RegionBinding(
                region=region,
                compiled=compiled_region,
                backend_id=placement.backend_id,
                device=placement.device,
            )
            compiled.append(
                {
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
            )
        except Exception as exc:
            raise SpecializationError(
                f"Failed to specialize region {placement.region_id} on {placement.device}: {exc}"
            ) from exc

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
    from streamcompiler.runtime.residency import attach_residency_to_plan
    from streamcompiler.runtime.schedule import (
        build_executable_schedule,
        schedule_matches_plan,
        validate_schedule,
        validate_schedule_resources,
        validate_schedule_tensor_sizes,
    )
    from streamcompiler.runtime.simulator.discrete_event import simulate_schedule

    residency = attach_residency_to_plan(plan, program)
    profile["residency"] = residency.as_dict()
    streaming = bool(config.ram_budget_bytes is not None and config.allow_nvme_streaming)
    executable_schedule = build_executable_schedule(
        plan,
        residency,
        streaming=streaming,
        prefetch_distance=config.prefetch_distance,
        program=program,
        activation_budget_bytes=config.activation_budget_bytes,
        machine=machine,
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
    profile["tensor_size_metadata"] = "exact" if program is not None else "estimated_from_portable_ir"
    # Simulate the exact instruction DAG the runtime will execute.
    sim = simulate_schedule(executable_schedule, machine)
    from streamcompiler.ir.graph import OpCode
    from streamcompiler.planner.cost.calibration import runtime_predicted_makespan_s

    n_compute = sum(1 for i in executable_schedule.instructions if i.opcode == OpCode.COMPUTE)
    # Runtime prediction = analytic DES + measured host-bridge tax.
    plan.predicted_latency_s = runtime_predicted_makespan_s(sim.makespan_s, n_compute=n_compute)
    plan.predicted_peak_bytes = sim.peak_bytes
    # Refresh decision text with post-simulation makespan so explanations cite
    # the same critical-path number the runtime/simulator share.
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
    if portable.metadata.get("buffer_reuse"):
        profile["buffer_reuse"] = portable.metadata["buffer_reuse"]
    eviction_events = sum(1 for e in sim.timeline if e.get("event") == "eviction_pressure")
    transfer_landed = sum(1 for e in sim.timeline if e.get("event") in ("Transfer", "transfer_landed"))
    profile["simulator"] = {
        "simulated": sim.simulated,
        "makespan_s": sim.makespan_s,
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


def _passthrough_specialization(
    program: RegionProgram,
    fingerprint: str,
    output_dir: Path | None,
) -> SpecializedArtifact:
    """Specialize a graph that returns its inputs or state without computing.

    There is nothing to place, measure or overlap, so the plan is empty by
    construction and the runtime resolves outputs straight from its environment.
    """
    from streamcompiler.runtime.schedule import ExecutableSchedule

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


def _decide_concurrency(
    program: RegionProgram | None,
    region_inputs: dict[str, tuple[Any, ...]],
    plan: ExecutionPlan,
    machine: ResourceGraph,
    config: CompileConfig,
) -> ConcurrencyDecision:
    """Decide whether independent regions should overlap, by measurement."""
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
    return measure_concurrency_benefit(
        program, region_inputs, max_workers=budget, iters=max(1, config.region_measure_iters)
    )


def concurrency_budget(plan: ExecutionPlan, machine: ResourceGraph) -> int:
    """Upper bound on simultaneous regions the selected devices can absorb.

    Distinct devices always contribute one worker each. A CPU pool can host as many
    regions as it has cores, because the concurrency measurement divides the intra-op
    threads between the workers rather than letting each one claim every core. This is
    only an upper bound: the measurement picks the configuration that is actually
    fastest, and often that is one worker.
    """
    total = 0
    for name in plan.devices_used:
        device = machine.compute.get(name)
        if device is None or device.backend_id != "cpu":
            total += 1
            continue
        total += max(2, device.concurrency_limit)
    return max(1, total)


def compile_exported_program(
    exported: Any,
    *,
    config: CompileConfig | None = None,
    name: str = "model",
    artifact_dir: Path | None = None,
    pack_lookup_dirs: tuple[Path, ...] = (),
    machine: Any | None = None,
    measurements: Any | None = None,
) -> CompiledModule:
    """Compile an already-captured ``ExportedProgram`` into a runnable module.

    This is the single implementation behind both :func:`streamcompiler.compile`
    and artifact reloading, so both paths exercise identical code.
    """
    from dataclasses import replace

    from streamcompiler.runtime.graph_executor import GraphExecutor
    from streamcompiler.runtime.module import CompiledModule
    from streamcompiler.runtime.provisioning import (
        build_parameter_store,
        intraop_threads,
        worker_count,
    )

    config = config or CompileConfig()
    force_single = (not config.allow_concurrent_regions) or config.max_concurrent_regions == 1
    program, portable = _lower_to_portable(
        exported,
        name=name,
        config=config,
        artifact_dir=artifact_dir,
        force_single_region=force_single,
    )
    example_flat = _example_flat_inputs(exported, program)
    specialized = specialize_for_machine(
        portable,
        config=config,
        output_dir=(artifact_dir / "specialized") if artifact_dir else None,
        example_inputs=example_flat,
        machine=machine,
        measurements=measurements,
    )
    workers = worker_count(specialized, config)
    # Prefer a fused single-region fast path when it beats multi-region execution.
    # Auto-concurrency may win against a multi-region sequential schedule yet still
    # lose to one fused region that avoids per-region dispatch.
    if (
        len(program.regions) > 1
        and config.ram_budget_bytes is None
        and config.allow_concurrent_regions
        and config.max_concurrent_regions == 0
        and not force_single
        and example_flat is not None
    ):
        decision = specialized.validation.get("concurrency", {})
        fused_config = replace(config, allow_concurrent_regions=False, max_concurrent_regions=1)
        fused_program, fused_portable = _lower_to_portable(
            exported,
            name=name,
            config=fused_config,
            artifact_dir=artifact_dir,
            force_single_region=True,
        )
        fused_specialized = specialize_for_machine(
            fused_portable,
            config=fused_config,
            output_dir=(artifact_dir / "specialized") if artifact_dir else None,
            example_inputs=example_flat,
            machine=machine,
            measurements=measurements,
        )
        prefer_fused = workers == 1
        if not prefer_fused:
            concurrent_s = _time_executor(
                program,
                specialized.bindings,
                example_flat,
                workers=workers,
                intraop_threads=intraop_threads(specialized, config),
            )
            fused_s = _time_executor(
                fused_program,
                fused_specialized.bindings,
                example_flat,
                workers=1,
                intraop_threads=0,
            )
            prefer_fused = fused_s * 1.02 <= concurrent_s
            specialized.plan.notes.append(
                f"fusion_compare: concurrent={concurrent_s * 1e3:.3f}ms "
                f"fused={fused_s * 1e3:.3f}ms prefer_fused={prefer_fused}"
            )
        if prefer_fused:
            program, portable, specialized = fused_program, fused_portable, fused_specialized
            specialized.validation["concurrency"] = decision
            specialized.validation["fused_after_sequential_decision"] = True
            specialized.plan.notes.append("fused_to_single_region: single region is faster than multi-region execution")
            workers = 1

    store = build_parameter_store(
        program,
        portable,
        config,
        artifact_dir=artifact_dir,
        pack_lookup_dirs=pack_lookup_dirs,
    )
    _attach_storage_measurement(store, specialized)
    reuse_meta = portable.metadata.get("buffer_reuse") or specialized.profile.get("buffer_reuse") or {}
    reuse_assignment = dict(reuse_meta.get("assignment") or {})
    executor = GraphExecutor(
        program,
        specialized.bindings,
        parameter_store=store,
        max_workers=workers,
        prefetch_distance=config.prefetch_distance,
        intraop_threads=intraop_threads(specialized, config),
        activation_budget_bytes=config.activation_budget_bytes,
        schedule=getattr(specialized, "schedule", None),
        buffer_reuse_assignment=reuse_assignment or None,
        process_workers=int(config.process_workers),
        machine=machine,
    )
    return CompiledModule(
        portable=portable,
        specialized=specialized,
        config=config,
        program=program,
        executor=executor,
        machine=machine,
        example_flat=example_flat,
    )


def _attach_storage_measurement(store: Any, specialized: SpecializedArtifact) -> None:
    """Record measured pack pread bandwidth when the runtime streams from disk."""
    if getattr(store, "kind", None) != "streaming":
        return
    from streamcompiler.hardware.storage_bench import benchmark_pack_payload
    from streamcompiler.storage.pack import load_pack_manifest

    stats = store.stats()
    pack_path = Path(stats["pack_path"])
    manifest = load_pack_manifest(pack_path)
    tensors = manifest.get("tensors") or []
    if not tensors:
        return
    largest = max(tensors, key=lambda entry: int(entry.get("nbytes", 0)))
    result = benchmark_pack_payload(
        pack_path,
        offset=int(largest["offset"]),
        nbytes=int(largest["nbytes"]),
    )
    specialized.profile["storage"] = result.as_dict()
    specialized.validation["storage"] = result.as_dict()
    if result.measured:
        mbps = result.bytes_per_s / (1 << 20)
        specialized.plan.notes.append(
            f"storage_pread_measured={mbps:.1f} MiB/s "
            f"({result.nbytes} bytes in {result.latency_s * 1e3:.3f} ms; {result.notes})"
        )
    else:
        specialized.plan.notes.append(f"storage_pread_unmeasured: {result.notes}")


def _time_executor(
    program: RegionProgram,
    bindings: dict[str, RegionBinding],
    flat_inputs: list[Any],
    *,
    workers: int,
    intraop_threads: int,
    iters: int = 3,
) -> float:
    """Best-of-N wall time for one GraphExecutor configuration."""
    from streamcompiler.runtime.graph_executor import GraphExecutor
    from streamcompiler.runtime.tensor_store import ResidentParameterStore

    store = ResidentParameterStore(program.state_tensors())
    executor = GraphExecutor(
        program,
        bindings,
        parameter_store=store,
        max_workers=workers,
        prefetch_distance=0,
        intraop_threads=intraop_threads,
    )
    try:
        for _ in range(2):
            executor.run(list(flat_inputs))
        best = float("inf")
        for _ in range(max(1, iters)):
            start = time.perf_counter()
            executor.run(list(flat_inputs))
            best = min(best, time.perf_counter() - start)
        return best
    finally:
        store.close()


def _lower_to_portable(
    exported: Any,
    *,
    name: str,
    config: CompileConfig,
    artifact_dir: Path | None,
    force_single_region: bool,
) -> tuple[RegionProgram, PortableArtifact]:
    """Lower, analyze, and pack one portable artifact from an exported program."""
    from streamcompiler.frontend.lower import lower_exported_program
    from streamcompiler.ir import (
        detect_repeated_blocks,
        run_alias_analysis,
        run_liveness_analysis,
    )

    lowered = lower_exported_program(
        exported,
        name=name,
        max_region_nodes=config.max_region_nodes,
        max_region_state_bytes=_streaming_region_budget(config),
        force_single_region=force_single_region,
    )
    ir = lowered.ir
    program = lowered.program
    alias = run_alias_analysis(ir)
    live = run_liveness_analysis(ir)
    ir.repeated_blocks = detect_repeated_blocks(ir)
    ir.metadata["alias_groups"] = alias.groups
    ir.metadata["liveness"] = {k: list(v) for k, v in live.intervals.items()}
    portable = portable_compile_from_ir(
        ir,
        state_dict=program.state_tensors(),
        output_dir=artifact_dir,
        program=program,
        exported=exported,
    )
    return program, portable


def _streaming_region_budget(config: CompileConfig) -> int | None:
    """Per-region parameter budget implied by the host RAM budget.

    With prefetching enabled the runtime may hold the current region's pins plus
    up to ``prefetch_distance`` successor regions, so each region is capped to
    ``budget / (1 + prefetch_distance)``.
    """
    if config.ram_budget_bytes is None:
        return None
    divisor = max(1, 1 + max(0, int(config.prefetch_distance)))
    return max(1, config.ram_budget_bytes // divisor)


def _example_flat_inputs(exported: Any, program: RegionProgram) -> list[Any] | None:
    """Recover the flat example inputs recorded by ``torch.export``."""
    args = getattr(exported, "example_inputs", None)
    if args is None:
        return None
    try:
        if isinstance(args, tuple) and len(args) == 2 and isinstance(args[1], dict):
            return program.flatten_inputs(args[0], args[1])
        return program.flatten_inputs(tuple(args), {})
    except Exception:  # noqa: BLE001 - measurement is optional, correctness is not
        return None


def needs_respecialization(artifact_dir: Path, current_fingerprint: str | None = None) -> bool:
    """True when no matching fingerprint exists for this machine.

    Looks at both the artifact root and ``specialized/fingerprint`` because
    ``SpecializedArtifact.save`` writes under ``specialized/`` while
    ``CompiledModule.save`` also mirrors the fingerprint at the root.
    """
    current = current_fingerprint or machine_fingerprint()
    for relative in ("fingerprint", "specialized/fingerprint"):
        fp_path = artifact_dir / relative
        if not fp_path.exists():
            continue
        stored = fp_path.read_text(encoding="utf-8").strip()
        return stored != current
    return True
