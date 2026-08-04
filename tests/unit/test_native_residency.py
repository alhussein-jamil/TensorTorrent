"""Opaque handle + native residency authority tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from tests.support.native import assert_native_runtime_used

import tensortorrent as tt
from tensortorrent.native import require_native
from tensortorrent.runtime.handles import NativeResidencyBridge, TensorHandleTable


def test_tensor_handle_table_roundtrip() -> None:
    table = TensorHandleTable()
    t = torch.ones(2, 2)
    hid = table.insert(t)
    assert table.get(hid) is t
    assert table.drop(hid) is t
    with pytest.raises(Exception, match="unknown tensor handle"):
        table.get(hid)


def test_native_residency_session_strict() -> None:
    native = require_native()
    session = native.NativeResidencySession()
    session.put("x", "cpu", 1, 16)
    assert session.require("x", "cpu") == 1
    assert session.has("x", "cpu")
    with pytest.raises(RuntimeError, match="not resident"):
        session.require("x", "cuda:0")
    session.release("x", "cpu")
    with pytest.raises(RuntimeError, match="not resident"):
        session.release("x", "cpu")


def test_native_residency_lease_blocks_release() -> None:
    bridge = NativeResidencyBridge.create()
    t = torch.randn(4)
    bridge.mirror_put("w", "cpu", t, nbytes=int(t.nbytes))
    bridge.session.acquire_lease("w", "cpu")
    with pytest.raises(RuntimeError, match="lease|alias"):
        bridge.release("w", "cpu")
    bridge.session.release_lease("w", "cpu")
    bridge.release("w", "cpu")
    assert not bridge.session.has("w", "cpu")
    assert ("w", "cpu") not in bridge._index
    # Final release drops the opaque Python handle immediately.
    assert len(bridge.handles) == 0


def test_public_compile_uses_native_residency_on_region_path() -> None:
    model = nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        devices="cpu",
        config=tt.CompileConfig(use_torch_compile=False, prefer_direct_path=False),
    )
    try:
        out = compiled(x)
        torch.testing.assert_close(out, model(x))
        report = compiled.executor._last_schedule_report
        assert report is not None
        stats = report.parameter_store
        assert_native_runtime_used(stats)
        assert stats.get("native_data_plane") is True
        assert stats.get("native_residency") is True
        rs = stats.get("native_residency_stats") or {}
        assert int(rs.get("put_count") or 0) >= 1
        # Host-path Compute verifies residency via session.has; require may be 0.
        assert int(rs.get("put_count") or 0) + int(rs.get("require_count") or 0) >= 1
        assert rs.get("native_residency") is True
    finally:
        compiled.close()
