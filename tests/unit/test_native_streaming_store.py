"""NativeStreamingStore owns pack bytes + prefetch; Python tensorizes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from streamcompiler.errors import StorageError
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
        assert native["prefetch_dropped"] == 0
        assert native["native_streaming"] is True
    finally:
        store.close()


def test_failed_decode_releases_native_lease(tmp_path: Path) -> None:
    pack = pack_state_dict({"w": torch.ones(1)}, tmp_path / "m.pack")
    store = StreamingParameterStore(pack.path, {"env_w": "w"}, budget_bytes=1 << 20)

    class BrokenDtypeStore:
        releases = 0

        def stats(self) -> dict[str, int]:
            return {}

        def acquire_bytes(self, name: str) -> bytearray:
            return bytearray(4)

        def release(self, name: str) -> None:
            self.releases += 1

        def close(self) -> None:
            return None

    original = store._native_store
    assert original is not None
    original.close()
    broken = BrokenDtypeStore()
    store._native_store = broken
    store._blocks["w"].dtype = "not_a_torch_dtype"
    try:
        with pytest.raises(StorageError, match="Unsupported stored dtype"):
            store.acquire("env_w")
        assert broken.releases == 1
        assert store._staging == {}
    finally:
        store.close()
