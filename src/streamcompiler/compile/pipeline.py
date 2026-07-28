"""Two-stage compilation: portable artifact + machine specialization."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from streamcompiler.backends import backend_by_id
from streamcompiler.codegen.regions import RegionBinding, RegionProgram
from streamcompiler.compile.concurrency import ConcurrencyDecision, measure_concurrency_benefit
from streamcompiler.compile.measure import (
    MeasurementSet,
    capture_region_inputs,
    measure_regions_on_devices,
    region_source,
)
from streamcompiler.config import CompileConfig
from streamcompiler.errors import SpecializationError
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.hardware.fingerprint import machine_fingerprint
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode, TensorMeta
from streamcompiler.ir.resource_graph import ResourceGraph
from streamcompiler.planner.maximal import ExecutionPlan, plan_execution

if TYPE_CHECKING:
    from streamcompiler.runtime.module import CompiledModule
from streamcompiler.simulator.discrete_event import simulate_plan
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
    alias = {tid: (t.alias_group or tid) for tid, t in ir.tensors.items()}
    liveness = {tid: (t.produced_at, t.last_use_at) for tid, t in ir.tensors.items()}
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
) -> SpecializedArtifact:
    """Deployment-time specialization against the actual machine resource graph."""
    config = config or CompileConfig()
    machine = discover_resource_graph()
    current_fp = machine.fingerprint or machine_fingerprint()
    program = portable.program

    region_inputs: dict[str, tuple[Any, ...]] = {}
    measurements = MeasurementSet()
    if program is not None and example_inputs is not None:
        region_inputs = capture_region_inputs(program, example_inputs)
        if config.measure_regions:
            measurements = measure_regions_on_devices(
                program,
                region_inputs,
                [d for d in machine.compute.values() if d.backend_id == "cpu"],
                iters=config.region_measure_iters,
            )

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
                }
            )
        except Exception as exc:
            raise SpecializationError(
                f"Failed to specialize region {placement.region_id} on {placement.device}: {exc}"
            ) from exc

    for placement in plan.placements:
        binding = bindings.get(placement.region_id)
        if binding is not None and (
            binding.device != placement.device or binding.backend_id != placement.backend_id
        ):
            raise SpecializationError(
                f"Binding for {placement.region_id} is {binding.backend_id}/{binding.device} "
                f"but plan says {placement.backend_id}/{placement.device}"
            )

    collectives = plan_collectives(portable.ir, machine, plan.devices_used)
    if collectives:
        plan.notes.append("collectives=" + ",".join(f"{c.op}:{c.backend_id}" for c in collectives))
    sim = simulate_plan(plan, machine)
    plan.predicted_latency_s = sim.makespan_s
    plan.predicted_peak_bytes = sim.peak_bytes
    plan.notes.append(
        f"simulator makespan={sim.makespan_s:.6f}s exposed_transfer={sim.exposed_transfer_latency_s:.6f}s"
    )

    # Memory feasibility: ensure each device's peak estimate fits allocatable memory.
    for mem_name, used in sim.peak_bytes.items():
        mem = machine.memory.get(mem_name)
        if mem is None:
            continue
        if used > mem.allocatable_bytes > 0:
            raise SpecializationError(
                f"Plan exceeds allocatable memory on {mem_name}: {used} > {mem.allocatable_bytes}"
            )

    concurrency = _decide_concurrency(program, region_inputs, plan, machine, config)
    plan.notes.append(f"concurrency={'enabled' if concurrency.enabled else 'disabled'}: {concurrency.reason}")

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
    }
    artifact = SpecializedArtifact(
        fingerprint=current_fp,
        plan=plan,
        compiled_regions=compiled,
        profile=profile,
        validation=validation,
        bindings=bindings,
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
    )
    workers = worker_count(specialized, config)
    # When auto concurrency measured no benefit, fuse into one region so the call
    # hits the single-region fast path instead of paying per-branch dispatch.
    if (
        workers == 1
        and len(program.regions) > 1
        and config.ram_budget_bytes is None
        and config.allow_concurrent_regions
        and config.max_concurrent_regions == 0
        and not force_single
    ):
        fused_config = replace(config, allow_concurrent_regions=False, max_concurrent_regions=1)
        program, portable = _lower_to_portable(
            exported,
            name=name,
            config=fused_config,
            artifact_dir=artifact_dir,
            force_single_region=True,
        )
        decision = specialized.validation.get("concurrency", {})
        specialized = specialize_for_machine(
            portable,
            config=fused_config,
            output_dir=(artifact_dir / "specialized") if artifact_dir else None,
            example_inputs=example_flat,
        )
        # Keep the original measured concurrency verdict; fusion is a latency win
        # after that verdict, not a claim that concurrency was never considered.
        specialized.validation["concurrency"] = decision
        specialized.validation["fused_after_sequential_decision"] = True
        specialized.plan.notes.append("fused_to_single_region: concurrency measured no benefit; one region for latency")
        workers = 1

    store = build_parameter_store(program, portable, config, artifact_dir=artifact_dir)
    _attach_storage_measurement(store, specialized)
    executor = GraphExecutor(
        program,
        specialized.bindings,
        parameter_store=store,
        max_workers=workers,
        prefetch_distance=config.prefetch_distance,
        intraop_threads=intraop_threads(specialized, config),
    )
    return CompiledModule(
        portable=portable,
        specialized=specialized,
        config=config,
        program=program,
        executor=executor,
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


def _lower_to_portable(
    exported: Any,
    *,
    name: str,
    config: CompileConfig,
    artifact_dir: Path | None,
    force_single_region: bool,
) -> tuple[RegionProgram, PortableArtifact]:
    """Lower, analyze, and pack one portable artifact from an exported program."""
    from streamcompiler.analysis import (
        detect_repeated_blocks,
        run_alias_analysis,
        run_liveness_analysis,
    )
    from streamcompiler.frontend.lower import lower_exported_program

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

    With prefetching enabled the runtime may hold the current and the next
    region's weights at once, so each region gets at most half the budget.
    """
    if config.ram_budget_bytes is None:
        return None
    divisor = 2 if config.prefetch_distance >= 1 else 1
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
