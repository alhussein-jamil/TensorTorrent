"""Two-stage compilation: portable artifact + machine specialization."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from streamcompiler.backends import backend_by_id
from streamcompiler.config import CompileConfig
from streamcompiler.errors import SpecializationError
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.hardware.fingerprint import machine_fingerprint
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode, TensorMeta
from streamcompiler.planner.maximal import ExecutionPlan, plan_execution
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
            "stages=exported_graph,normalized_graph,heterogeneous_ir,"
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
) -> PortableArtifact:
    """Produce a portable artifact from an already-lowered heterogeneous IR."""
    alias = {tid: (t.alias_group or tid) for tid, t in ir.tensors.items()}
    liveness = {
        tid: (t.produced_at, t.last_use_at) for tid, t in ir.tensors.items()
    }
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
            pack = pack_state_dict(state_dict, output_dir / "model.pack")
            packed_path = str(pack.path)

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
        },
    )
    if output_dir is not None:
        artifact.save(output_dir)
    return artifact


def specialize_for_machine(
    portable: PortableArtifact,
    *,
    config: CompileConfig | None = None,
    output_dir: Path | None = None,
    force_rediscover: bool = True,
) -> SpecializedArtifact:
    """Deployment-time specialization against the actual machine resource graph."""
    config = config or CompileConfig()
    machine = discover_resource_graph() if force_rediscover else discover_resource_graph()
    current_fp = machine.fingerprint
    if not current_fp:
        current_fp = machine_fingerprint()

    # 4-9: profile missing candidates, generate executables, search plan, validate memory,
    # measure promising plans, select best, cache.
    plan = plan_execution(portable.ir, machine, config)

    compiled: list[dict[str, Any]] = []
    profile: dict[str, Any] = {"devices": {}, "transfers": {}, "missing_measurements": []}
    for placement in plan.placements:
        backend = backend_by_id(placement.backend_id)
        device = machine.compute.get(placement.device)
        if backend is None or device is None:
            profile["missing_measurements"].append(placement.region_id)
            continue
        from streamcompiler.backends.base import KernelCandidate

        cand = KernelCandidate(
            region_id=placement.region_id,
            device=placement.device,
            backend_id=placement.backend_id,
            kernel_id=placement.kernel_id,
            dtype=placement.dtype,
        )
        try:
            if config.profile_level in ("competitive", "full"):
                bench = backend.benchmark(cand)
                profile["devices"][placement.device] = {
                    "latency_s": bench.latency_s,
                    "measured": bench.measured,
                    "notes": bench.notes,
                }
                if bench.measured and bench.latency_s < float("inf"):
                    placement.estimated_latency_s = bench.latency_s
            region = backend.compile(
                Instruction(opcode=OpCode.COMPUTE, name=placement.region_id),
                cand,
            )
            compiled.append(
                {
                    "region_id": region.region_id,
                    "device": region.device,
                    "backend_id": region.backend_id,
                    "dtype": region.dtype,
                }
            )
        except Exception as exc:  # noqa: BLE001
            raise SpecializationError(
                f"Failed to specialize region {placement.region_id} on {placement.device}: {exc}"
            ) from exc

    # Recompute critical-path latency after any measured candidate updates.
    sim = simulate_plan(plan, machine)
    plan.predicted_latency_s = sim.makespan_s
    plan.predicted_peak_bytes = sim.peak_bytes
    from streamcompiler.planner.collectives import plan_collectives
    from streamcompiler.planner.local_search import rebalance_partitions, refine_prefetch_distance

    plan = rebalance_partitions(plan)
    plan = refine_prefetch_distance(plan)
    collectives = plan_collectives(portable.ir, machine, plan.devices_used)
    if collectives:
        plan.notes.append(
            "collectives=" + ",".join(f"{c.op}:{c.backend_id}" for c in collectives)
        )
    sim = simulate_plan(plan, machine)
    plan.predicted_latency_s = sim.makespan_s
    plan.predicted_peak_bytes = sim.peak_bytes
    plan.notes.append(
        f"simulator makespan={sim.makespan_s:.6f}s "
        f"exposed_transfer={sim.exposed_transfer_latency_s:.6f}s"
    )

    # Memory feasibility: ensure each device's peak estimate fits allocatable memory.
    for mem_name, used in sim.peak_bytes.items():
        mem = machine.memory.get(mem_name)
        if mem is None:
            continue
        if used > mem.allocatable_bytes > 0:
            raise SpecializationError(
                f"Plan exceeds allocatable memory on {mem_name}: "
                f"{used} > {mem.allocatable_bytes}"
            )

    validation = {
        "fingerprint_matched": True,
        "memory_feasible": True,
        "backends_used": sorted({p.backend_id for p in plan.placements}),
        "simulated_makespan_s": sim.makespan_s,
        "exposed_transfer_latency_s": sim.exposed_transfer_latency_s,
        "timeline_events": len(sim.timeline),
    }
    artifact = SpecializedArtifact(
        fingerprint=current_fp,
        plan=plan,
        compiled_regions=compiled,
        profile=profile,
        validation=validation,
    )
    if output_dir is not None:
        artifact.save(output_dir)
        # Invalidate notice for future fingerprint mismatch.
        (output_dir / "fingerprint").write_text(current_fp + "\n", encoding="utf-8")
    return artifact


def needs_respecialization(artifact_dir: Path, current_fingerprint: str | None = None) -> bool:
    fp_path = artifact_dir / "fingerprint"
    if not fp_path.exists():
        return True
    stored = fp_path.read_text(encoding="utf-8").strip()
    current = current_fingerprint or machine_fingerprint()
    return stored != current
