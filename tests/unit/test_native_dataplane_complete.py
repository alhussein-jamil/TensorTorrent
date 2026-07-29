"""Native storage / profiler / virtual-backend / stream-id smoke tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from streamcompiler.native import native_available, require_native
from streamcompiler.runtime.profile_feedback import ProfileFeedback
from streamcompiler.runtime.schedule import (
    ExecutableSchedule,
    MemoryTier,
    PlanInstruction,
    ensure_explicit_streams,
    validate_schedule,
)
from streamcompiler.ir.graph import OpCode
from streamcompiler.storage.native_pack import open_native_pack_reader, scpack_to_native_manifest_json
from streamcompiler.storage.pack import pack_tensors


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
        ),
    )
    filled = ensure_explicit_streams(schedule)
    assert filled.instructions[0].stream_id == "cpu::compute"
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
