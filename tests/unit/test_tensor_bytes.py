"""Host byte conversion for dtypes NumPy cannot represent."""

from __future__ import annotations

import torch

from tensortorrent.runtime.tensor_store import StreamingParameterStore
from tensortorrent.storage.pack import load_pack_manifest, pack_state_dict
from tensortorrent.tensor_bytes import tensor_as_bytes, tensor_as_memoryview


def test_bfloat16_tensor_as_bytes_roundtrip_size() -> None:
    t = torch.randn(8, 8, dtype=torch.bfloat16)
    raw = tensor_as_bytes(t)
    assert len(raw) == t.numel() * t.element_size()
    mv = tensor_as_memoryview(t)
    assert len(mv) == len(raw)
    assert bytes(mv) == raw


def test_float32_tensor_as_bytes_matches_numpy() -> None:
    t = torch.randn(4, 4, dtype=torch.float32)
    assert tensor_as_bytes(t) == bytes(t.numpy().tobytes())


def test_pack_and_stream_bfloat16(tmp_path) -> None:
    weight = torch.randn(32, 16, dtype=torch.bfloat16)
    path = tmp_path / "bf16.pack"
    pack_state_dict({"linear.weight": weight}, path)
    manifest = load_pack_manifest(path)
    entry = manifest["tensors"][0]
    assert entry["logical_id"] == "linear.weight"
    assert entry["stored_dtype"] == "bfloat16"
    assert entry["nbytes"] == weight.numel() * weight.element_size()

    store = StreamingParameterStore(path, {"w": "linear.weight"}, budget_bytes=1 << 20)
    try:
        got = store.acquire("w")
        assert got.dtype == torch.bfloat16
        assert got.shape == weight.shape
        assert torch.equal(got, weight)
        store.release(("w",))
    finally:
        store.close()
