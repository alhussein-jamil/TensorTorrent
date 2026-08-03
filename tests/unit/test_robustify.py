"""Pack, fingerprint, configuration, and concurrency regressions."""

from __future__ import annotations

import json
import struct
import threading
from pathlib import Path
from typing import Any

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
    manifest["tensors"][0]["offset"] = ((len(data) + 100 + 63) // 64) * 64
    corrupt = MAGIC + struct.pack("<II", VERSION, len(json.dumps(manifest))) + json.dumps(manifest).encode()
    bad = tmp_path / "bad.pack"
    bad.write_bytes(corrupt + b"\0" * 64)
    with pytest.raises(StorageError, match="outside file size"):
        load_pack_manifest(bad)


def test_truncated_pack_file_is_rejected(tmp_path: Path) -> None:
    pack = pack_state_dict({"w": torch.randn(64)}, tmp_path / "ok.pack")
    full = pack.path.read_bytes()
    truncated = tmp_path / "truncated.pack"
    truncated.write_bytes(full[: len(full) // 2])
    with pytest.raises(StorageError, match="outside file size"):
        load_pack_manifest(truncated)


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


def test_concurrent_forwards_on_same_module() -> None:
    compiled = sc.compile(nn.Linear(8, 4).eval(), (torch.randn(2, 8),))
    x = torch.randn(2, 8)
    expected = compiled(x)
    errors: list[BaseException] = []
    outputs: list[torch.Tensor] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            out = compiled(x)
            with lock:
                outputs.append(out)
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(outputs) == 8
    for out in outputs:
        torch.testing.assert_close(out, expected)
    compiled.close()


def test_bindings_match_plan_devices_after_specialize() -> None:
    compiled = sc.compile(nn.Linear(4, 4).eval(), (torch.randn(2, 4),))
    by_id = {p.region_id: p for p in compiled.specialized.plan.placements}
    for region_id, binding in compiled.specialized.bindings.items():
        assert binding.device == by_id[region_id].device
        assert binding.backend_id == by_id[region_id].backend_id


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


def test_compile_config_rejects_recompute_overflow_policy() -> None:
    with pytest.raises(ValueError, match="recompute is not implemented"):
        CompileConfig(activation_overflow_policy="recompute")


def test_pack_write_is_atomic_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import streamcompiler.storage.pack as pack_mod

    destination = tmp_path / "model.pack"
    pack_state_dict({"w": torch.ones(4)}, destination)
    original = destination.read_bytes()

    def fail_chunked(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("inject write failure")

    monkeypatch.setattr(pack_mod, "_write_payload_chunked", fail_chunked)
    with pytest.raises(RuntimeError, match="inject write failure"):
        pack_state_dict({"w": torch.ones(8)}, destination)
    assert destination.read_bytes() == original
    assert not destination.with_name(destination.name + ".tmp").exists()


def test_run_after_close_is_rejected() -> None:
    compiled = sc.compile(nn.Linear(4, 2).eval(), (torch.randn(2, 4),))
    x = torch.randn(2, 4)
    compiled.close()
    with pytest.raises(RuntimePlanError, match="closed"):
        compiled(x)


def test_schedule_report_tracks_peak_activation_bytes() -> None:
    compiled = sc.compile(nn.Linear(8, 4).eval(), (torch.randn(2, 8),))
    try:
        compiled(torch.randn(2, 8))
        report = compiled.last_report
        assert report is not None
        assert report.peak_activation_bytes > 0
        sreport = compiled.executor._last_schedule_report
        assert sreport is not None
        assert sreport.peak_activation_bytes == report.peak_activation_bytes
    finally:
        compiled.close()


def test_compile_config_rejects_ambiguous_json_types() -> None:
    with pytest.raises(TypeError, match="max_region_nodes"):
        CompileConfig.from_json_dict({"max_region_nodes": "9"})
    with pytest.raises(TypeError, match="allow_training"):
        CompileConfig.from_json_dict({"allow_training": 0})
    with pytest.raises(TypeError, match="ram_budget_bytes"):
        CompileConfig.from_json_dict({"ram_budget_bytes": "1024"})
    with pytest.raises(TypeError, match="atol"):
        CompileConfig.from_json_dict({"atol": "1e-5"})
    restored = CompileConfig.from_json_dict(
        {
            "objective": "latency",
            "max_region_nodes": 9,
            "prefetch_distance": 2,
            "allow_training": False,
            "process_workers": 0,
            "ram_budget_bytes": 1024,
            "atol": 1e-5,
        }
    )
    assert restored.max_region_nodes == 9
    assert restored.prefetch_distance == 2
    assert restored.allow_training is False
    assert restored.process_workers == 0
    assert restored.ram_budget_bytes == 1024
    assert restored.atol == pytest.approx(1e-5)


def test_schedule_executor_close_is_idempotent_and_rejects_run() -> None:
    compiled = sc.compile(
        nn.Linear(4, 4).eval(),
        (torch.randn(2, 4),),
        config=CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        sched = compiled.executor._schedule_executor
        assert sched is not None
        sched.close()
        sched.close()  # second close must not raise
        with pytest.raises(RuntimePlanError, match="closed"):
            sched.run([torch.randn(2, 4)])
    finally:
        compiled.close()


def test_streams_make_event_cpu_and_registry() -> None:
    from streamcompiler.runtime.streams import EventRegistry, make_event, make_stream, synchronize_device

    event = make_event("e", "cpu")
    assert make_stream("cpu") is None
    synchronize_device("cpu")  # no-op
    event.record()
    registry = EventRegistry()
    registry.store("e", event)
    assert registry.get("e") is event
    event.wait()
    assert event.is_complete()


def test_compiled_module_close_is_idempotent() -> None:
    compiled = sc.compile(
        nn.Linear(4, 4).eval(),
        (torch.randn(2, 4),),
        config=CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    compiled.close()
    compiled.close()
    with pytest.raises(RuntimePlanError, match="closed"):
        compiled(torch.randn(2, 4))
