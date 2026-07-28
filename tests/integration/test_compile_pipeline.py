"""Two-stage compilation smoke tests."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from streamcompiler.compile.pipeline import (
    needs_respecialization,
    portable_compile_from_ir,
    specialize_for_machine,
)
from streamcompiler.config import CompileConfig
from streamcompiler.frontend.lower import lower_exported_program
from streamcompiler.storage.pack import load_pack_manifest, pack_state_dict


def test_portable_then_specialize(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    model.eval()
    exported = torch.export.export(model, (torch.randn(2, 8),))
    ir = lower_exported_program(exported, name="Tiny")
    portable = portable_compile_from_ir(
        ir,
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        output_dir=tmp_path / "artifact",
    )
    assert (tmp_path / "artifact" / "portable.json").exists()
    assert portable.packed_model_path
    manifest = load_pack_manifest(Path(portable.packed_model_path))
    assert manifest["tensor_count"] >= 1

    specialized = specialize_for_machine(
        portable,
        config=CompileConfig(profile_level="coarse"),
        output_dir=tmp_path / "artifact" / "specialized",
    )
    assert specialized.plan.devices_used
    assert (tmp_path / "artifact" / "specialized" / "fingerprint").exists()
    assert needs_respecialization(tmp_path / "artifact" / "specialized", "different-fp")


def test_pack_roundtrip(tmp_path: Path) -> None:
    t = torch.randn(4, 4)
    pack = pack_state_dict({"w": t}, tmp_path / "m.pack")
    manifest = load_pack_manifest(pack.path)
    assert manifest["tensors"][0]["logical_id"] == "w"
