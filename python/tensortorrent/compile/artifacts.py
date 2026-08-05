"""Portable and specialized compilation artifacts."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tensortorrent.artifact_io import atomic_write_json, atomic_write_text
from tensortorrent.compile.regions import RegionBinding, RegionProgram
from tensortorrent.ir.graph import HeterogeneousGraph, Instruction, OpCode, TensorMeta
from tensortorrent.planner.maximal import ExecutionPlan
from tensortorrent.storage.pack import pack_state_dict

if TYPE_CHECKING:
    from tensortorrent.runtime.schedule import ExecutableSchedule


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
        atomic_write_json(path, payload)
        atomic_write_text(
            directory / "MANIFEST",
            "tensortorrent-portable-artifact-v1\n"
            f"name={self.name}\n"
            "stages=exported_graph,heterogeneous_ir,"
            "alias_liveness,packed_model,candidate_partitions,hw_independent_metadata\n",
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
                "predicted_throughput_per_s": self.plan.predicted_throughput_per_s,
                "predicted_transfer_bytes": self.plan.predicted_transfer_bytes,
                "predicted_transfer_latency_s": self.plan.predicted_transfer_latency_s,
                "prefetch_distance": self.plan.prefetch_distance,
                "search_statistics": self.plan.search_statistics,
                "strategy": self.plan.strategy,
                "notes": self.plan.notes,
            },
            "compiled_regions": self.compiled_regions,
            "profile": self.profile,
            "validation": self.validation,
            "executable_schedule": None if self.schedule is None else self.schedule.as_dict(),
        }
        path = directory / "specialized.json"
        atomic_write_json(path, payload)
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
    from tensortorrent.ir.alias import run_alias_analysis
    from tensortorrent.ir.liveness import run_liveness_analysis
    from tensortorrent.runtime.buffer_reuse import plan_buffer_reuse

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
