"""Native storage / profiler / virtual-backend / stream-id smoke tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from tensortorrent.errors import StorageError
from tensortorrent.ir.graph import OpCode
from tensortorrent.native import native_available, require_native
from tensortorrent.runtime.profile_feedback import ProfileFeedback
from tensortorrent.runtime.schedule import (
    ExecutableSchedule,
    MemoryTier,
    PlanInstruction,
    ensure_explicit_streams,
    validate_schedule,
)
from tensortorrent.storage.native_pack import open_native_pack_reader, scpack_to_native_manifest_json
from tensortorrent.storage.pack import pack_tensors

pytestmark = pytest.mark.skipif(not native_available(), reason="native extension required")


def test_native_pack_reader_reads_scpack():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "m.pack"
        w = torch.randn(4, 4)
        pack_tensors([("w", lambda: w.clone())], path)
        reader = open_native_pack_reader(path)
        assert reader is not None
        raw = bytes(reader.pread("w"))
        assert len(raw) == w.nbytes
        restored = torch.frombuffer(bytearray(raw), dtype=w.dtype).reshape(w.shape)
        assert torch.allclose(restored, w)


def test_native_profile_database_and_feedback():
    fb = ProfileFeedback(cache_key="t")
    assert fb._native_db is not None

    class _Ev:
        region_id = "r0"
        start_s = 0.0
        end_s = 0.02
        device = "cpu"

    class _Rep:
        events = [_Ev()]
        parameter_store = {}

    fb.observe_report(_Rep())
    assert fb.prior_for("r0", 1.0) == pytest.approx(0.02)
    assert fb.as_dict()["native_profiler"] is True


def test_native_virtual_backend_pending_async():
    native = require_native()
    assert native.virtual_backend_pending_is_async() is True
    be = native.NativeVirtualBackend(name="mock_accel0", compute_delay_s=0.02)
    caps = dict(be.capabilities())
    assert caps["simulated"] is True
    a = be.allocate("mock_accel0", 1024)
    b = be.allocate("mock_accel0", 1024)
    ev = be.transfer(a, b, 1024, "copy0")
    assert be.query_event(ev) == "pending"
    be.wait_event(ev)
    assert be.query_event(ev) == "complete"


def test_public_mock_compute_events_are_labelled_simulated():
    """Mock-resource Compute on public path sets simulated via Rust VirtualBackend."""
    import torch.nn as nn

    import tensortorrent as tt
    from tensortorrent.config import CompileConfig

    model = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    compiled = tt.compile(model, (x,), config=CompileConfig(use_torch_compile=False, measure_regions=False))
    try:
        from dataclasses import replace

        sched = compiled.specialized.schedule
        assert sched is not None
        new_inst = []
        for inst in sched.instructions:
            if inst.opcode == OpCode.COMPUTE:
                attrs = dict(inst.attributes)
                attrs["mock_compute_delay_s"] = 0.01
                new_inst.append(replace(inst, resource="mock_accel0", attributes=attrs))
            else:
                new_inst.append(inst)
        new_sched = ensure_explicit_streams(replace(sched, instructions=tuple(new_inst)))
        compiled.executor._schedule_executor.replace_schedule(new_sched)
        out = compiled(x)
        torch.testing.assert_close(out, model(x), check_device=False)
        sreport = compiled.executor._last_schedule_report
        assert sreport is not None
        assert sreport.parameter_store.get("native_data_plane") is True
        assert any(e.simulated for e in sreport.events if e.opcode == "Compute")
    finally:
        compiled.close()


def test_explicit_stream_ids_on_schedule():
    schedule = ExecutableSchedule(
        graph_name="g",
        fingerprint="fp",
        instructions=(
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="c0",
                resource="cpu",
                executable_ref="r0",
                inputs=("x",),
                outputs=("y",),
                nbytes=8,
                memory_tier=MemoryTier.SYSTEM_RAM,
            ),
            PlanInstruction(
                opcode=OpCode.LOAD,
                name="l0",
                resource="cpu",
                inputs=("w",),
                outputs=("w",),
                nbytes=8,
                memory_tier=MemoryTier.SYSTEM_RAM,
            ),
        ),
    )
    filled = ensure_explicit_streams(schedule)
    assert filled.instructions[0].stream_id == "cpu::compute"
    assert filled.instructions[1].stream_id == "cpu::copy0"
    assert filled.instructions[1].copy_engine_id == "cpu::copy0"
    assert filled.instructions[1].io_queue_id == "cpu::io0"
    assert validate_schedule(filled) == []


def test_scpack_manifest_adapter_keys():
    manifest = {
        "version": 1,
        "tensors": [
            {
                "logical_id": "w",
                "offset": 100,
                "nbytes": 64,
                "stored_dtype": "float32",
                "stored_shape": [4, 4],
            }
        ],
    }
    import json

    adapted = json.loads(scpack_to_native_manifest_json(manifest))
    assert adapted["tensors"][0]["name"] == "w"
    assert adapted["tensors"][0]["length"] == 64


@pytest.mark.parametrize("field,value", (("offset", 1.5), ("nbytes", True), ("stored_shape", [False])))
def test_scpack_manifest_adapter_rejects_ambiguous_metadata(field: str, value: object) -> None:
    entry = {
        "logical_id": "w",
        "offset": 0,
        "nbytes": 4,
        "stored_dtype": "float32",
        "stored_shape": [1],
    }
    entry[field] = value
    with pytest.raises(StorageError):
        scpack_to_native_manifest_json({"version": 1, "tensors": [entry]})
