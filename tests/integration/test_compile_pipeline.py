"""Two-stage compilation tests: portable artifact then machine specialization."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from streamcompiler.compile.measure import capture_region_inputs
from streamcompiler.compile.pipeline import (
    needs_respecialization,
    portable_compile_from_ir,
    specialize_for_machine,
)
from streamcompiler.config import CompileConfig
from streamcompiler.frontend.lower import lower_exported_program
from streamcompiler.storage.pack import load_pack_manifest, pack_state_dict


def _lower(model: nn.Module, example: torch.Tensor):
    exported = torch.export.export(model.eval(), (example,))
    return lower_exported_program(exported, name="Tiny"), exported


def test_portable_then_specialize(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    example = torch.randn(2, 8)
    lowered, exported = _lower(model, example)
    portable = portable_compile_from_ir(
        lowered.ir,
        state_dict=lowered.program.state_tensors(),
        output_dir=tmp_path / "artifact",
        program=lowered.program,
        exported=exported,
    )
    assert (tmp_path / "artifact" / "portable.json").exists()
    assert portable.packed_model_path
    manifest = load_pack_manifest(Path(portable.packed_model_path))
    assert manifest["tensor_count"] == len(lowered.program.state_bindings)

    flat = lowered.program.flatten_inputs((example,), {})
    specialized = specialize_for_machine(
        portable,
        config=CompileConfig(profile_level="coarse"),
        output_dir=tmp_path / "artifact" / "specialized",
        example_inputs=flat,
    )
    assert specialized.plan.devices_used
    assert (tmp_path / "artifact" / "specialized" / "fingerprint").exists()
    assert needs_respecialization(tmp_path / "artifact" / "specialized", "different-fp")

    # Every planned region must have a real callable executable.
    assert set(specialized.bindings) == {r.region_id for r in lowered.program.regions}
    for binding in specialized.bindings.values():
        assert callable(binding.compiled.executable)


def test_specialization_uses_measured_region_costs(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))
    example = torch.randn(4, 64)
    lowered, exported = _lower(model, example)
    portable = portable_compile_from_ir(lowered.ir, program=lowered.program, exported=exported)
    flat = lowered.program.flatten_inputs((example,), {})
    specialized = specialize_for_machine(portable, config=CompileConfig(), example_inputs=flat)
    assert specialized.validation["regions_measured"] == specialized.validation["regions_total"]
    assert all(p.measured for p in specialized.plan.placements)
    assert all(p.estimated_latency_s > 0.0 for p in specialized.plan.placements)
    measurements = specialized.profile["region_measurements"]
    assert measurements
    for devices in measurements.values():
        for entry in devices.values():
            assert entry["measured"] is True
            assert entry["latency_s"] > 0.0
    assert any("region_costs=measured" in note for note in specialized.plan.notes)


def test_regions_without_measurements_are_reported_as_priors() -> None:
    model = nn.Sequential(nn.Linear(8, 8))
    example = torch.randn(2, 8)
    lowered, exported = _lower(model, example)
    portable = portable_compile_from_ir(lowered.ir, program=lowered.program, exported=exported)
    specialized = specialize_for_machine(portable, config=CompileConfig(measure_regions=False), example_inputs=None)
    assert not any(p.measured for p in specialized.plan.placements)
    assert any("priors_only" in note for note in specialized.plan.notes)
    assert specialized.profile["missing_measurements"]


def test_capture_region_inputs_matches_region_arity() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    example = torch.randn(2, 8)
    lowered, _ = _lower(model, example)
    flat = lowered.program.flatten_inputs((example,), {})
    captured = capture_region_inputs(lowered.program, flat)
    for region in lowered.program.regions:
        assert len(captured[region.region_id]) == len(region.inputs)


def test_pack_roundtrip(tmp_path: Path) -> None:
    t = torch.randn(4, 4)
    pack = pack_state_dict({"w": t}, tmp_path / "m.pack")
    manifest = load_pack_manifest(pack.path)
    assert manifest["tensors"][0]["logical_id"] == "w"


def test_ir_matches_the_executable_region_program() -> None:
    """The planner IR must describe exactly the regions the runtime will execute."""
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    lowered, _ = _lower(model, torch.randn(2, 8))
    ir_regions = [i.name for i in lowered.ir.compute_regions()]
    assert ir_regions == [r.region_id for r in lowered.program.regions]
    for inst, region in zip(lowered.ir.compute_regions(), lowered.program.regions, strict=True):
        assert inst.inputs == region.inputs
        assert inst.outputs == region.outputs
        assert tuple(inst.attributes["depends_on"]) == region.depends_on
    for name, spec in lowered.program.values.items():
        tensor = lowered.ir.tensors[name]
        assert tuple(tensor.shape) == spec.shape
        assert tensor.dtype == spec.dtype
        assert tensor.size_bytes == spec.nbytes
    assert lowered.ir.parameters == tuple(lowered.program.state_bindings)


def test_lowering_preserves_real_shapes_and_dtypes() -> None:
    model = nn.Sequential(nn.Linear(8, 5))
    lowered, _ = _lower(model, torch.randn(3, 8))
    weight = next(n for n, t in lowered.program.state_bindings.items() if t.endswith("0.weight"))
    spec = lowered.program.values[weight]
    assert spec.shape == (5, 8)
    assert spec.dtype == "float32"
    assert spec.nbytes == 5 * 8 * 4
    assert spec.kind == "parameter"
    inputs = [lowered.program.values[n] for n in lowered.program.user_inputs]
    assert [i.shape for i in inputs] == [(3, 8)]


def test_planner_rejects_ir_without_compute_regions() -> None:
    from streamcompiler.errors import PlanningError
    from streamcompiler.hardware.discovery import discover_resource_graph
    from streamcompiler.ir.graph import HeterogeneousGraph
    from streamcompiler.planner.maximal import plan_execution

    empty = HeterogeneousGraph(name="empty")
    with pytest.raises(PlanningError, match="no compute regions"):
        plan_execution(empty, discover_resource_graph(), CompileConfig())
