"""Production hardening regressions for packs, fingerprints, and reentrancy."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.compile.pipeline import needs_respecialization
from streamcompiler.config import CompileConfig, Objective
from streamcompiler.errors import RuntimePlanError, StorageError
from streamcompiler.storage.pack import (
    MAGIC,
    VERSION,
    load_pack_manifest,
    pack_state_dict,
    resolve_pack_path,
    verify_block_checksum,
)


def test_pack_manifest_rejects_out_of_bounds_blocks(tmp_path: Path) -> None:
    pack = pack_state_dict({"w": torch.randn(4)}, tmp_path / "ok.pack")
    data = bytearray(pack.path.read_bytes())
    version, manifest_len = struct.unpack_from("<II", data, 8)
    assert version == VERSION
    manifest = json.loads(data[16 : 16 + manifest_len].decode("utf-8"))
    manifest["tensors"][0]["offset"] = len(data) + 100
    corrupt = MAGIC + struct.pack("<II", VERSION, len(json.dumps(manifest))) + json.dumps(manifest).encode()
    bad = tmp_path / "bad.pack"
    bad.write_bytes(corrupt + b"\0" * 64)
    with pytest.raises(StorageError, match="outside file size"):
        load_pack_manifest(bad)


def test_pack_checksum_mismatch_is_rejected() -> None:
    with pytest.raises(StorageError, match="Checksum mismatch"):
        verify_block_checksum(b"abc", "deadbeef", logical_id="w", path="x.pack")


def test_resolve_pack_path_rejects_escape(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    artifact.mkdir()
    outside = tmp_path / "secret.pack"
    pack_state_dict({"w": torch.ones(2)}, outside)
    with pytest.raises(StorageError, match="outside"):
        resolve_pack_path(outside, artifact_dir=artifact, cache_dir=tmp_path / "cache")


def test_resolve_pack_path_accepts_relative_under_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "art"
    artifact.mkdir()
    pack_state_dict({"w": torch.ones(2)}, artifact / "model.pack")
    resolved = resolve_pack_path("model.pack", artifact_dir=artifact, cache_dir=tmp_path / "cache")
    assert resolved == (artifact / "model.pack").resolve()


def test_needs_respecialization_reads_specialized_fingerprint(tmp_path: Path) -> None:
    specialized = tmp_path / "specialized"
    specialized.mkdir()
    (specialized / "fingerprint").write_text("machine-a\n", encoding="utf-8")
    assert needs_respecialization(tmp_path, current_fingerprint="machine-a") is False
    assert needs_respecialization(tmp_path, current_fingerprint="machine-b") is True


def test_compile_config_round_trips_through_json() -> None:
    original = CompileConfig(
        objective=Objective.THROUGHPUT,
        max_region_nodes=7,
        max_concurrent_regions=2,
        ram_budget_bytes=1_000_000,
        prefetch_distance=3,
        allow_concurrent_regions=False,
        profile_level="full",
    )
    restored = CompileConfig.from_json_dict(original.to_json_dict())
    assert restored.objective is Objective.THROUGHPUT
    assert restored.max_region_nodes == 7
    assert restored.max_concurrent_regions == 2
    assert restored.ram_budget_bytes == 1_000_000
    assert restored.prefetch_distance == 3
    assert restored.allow_concurrent_regions is False
    assert restored.profile_level == "full"


def test_bindings_match_plan_devices_after_specialize() -> None:
    compiled = sc.compile(nn.Linear(4, 4).eval(), (torch.randn(2, 4),))
    by_id = {p.region_id: p for p in compiled.specialized.plan.placements}
    for region_id, binding in compiled.specialized.bindings.items():
        assert binding.device == by_id[region_id].device
        assert binding.backend_id == by_id[region_id].backend_id


def test_concurrent_forward_is_rejected() -> None:
    compiled = sc.compile(nn.Linear(8, 4).eval(), (torch.randn(2, 8),))
    x = torch.randn(2, 8)
    assert compiled.executor._run_lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimePlanError, match="not reentrant"):
            compiled(x)
    finally:
        compiled.executor._run_lock.release()


def test_save_writes_root_fingerprint_and_full_config(tmp_path: Path) -> None:
    compiled = sc.compile(
        nn.Linear(4, 2).eval(),
        (torch.randn(1, 4),),
        config=CompileConfig(max_region_nodes=8, prefetch_distance=2),
    )
    out = compiled.save(tmp_path / "artifact")
    assert (out / "fingerprint").read_text(encoding="utf-8").strip() == compiled.specialized.fingerprint
    data = json.loads((out / "compile_config.json").read_text(encoding="utf-8"))
    assert data["max_region_nodes"] == 8
    assert data["prefetch_distance"] == 2
    assert "allow_mixed_vendor" in data
