"""NativeStreamingStore owns pack bytes + prefetch; Python tensorizes."""

from __future__ import annotations

from pathlib import Path

import torch

from streamcompiler.runtime.tensor_store import StreamingParameterStore
from streamcompiler.storage.pack import pack_state_dict


def test_native_streaming_store_used_for_pack_io(tmp_path: Path) -> None:
    tensors = {"w": torch.randn(64, 64)}
    pack = pack_state_dict(tensors, tmp_path / "m.pack")
    store = StreamingParameterStore(pack.path, {"env_w": "w"}, budget_bytes=1 << 20)
    try:
        stats = store.stats()
        assert stats["native_streaming"] is True
        assert store._native_store is not None
        store.prefetch(("env_w",))
        got = store.acquire("env_w")
        torch.testing.assert_close(got, tensors["w"])
        store.release(("env_w",))
        native = store.stats()["native_store"]
        assert int(native["bytes_read"]) >= tensors["w"].nbytes
        assert native["native_streaming"] is True
    finally:
        store.close()
