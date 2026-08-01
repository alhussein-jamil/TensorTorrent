"""Model pack format tests: the streaming runtime depends on these offsets."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from streamcompiler.errors import StorageError
from streamcompiler.storage.pack import load_pack_manifest, pack_state_dict


def test_pack_header_grows_for_many_tensors(tmp_path: Path) -> None:
    """A fixed header size used to break packs with more than a few dozen tensors."""
    tensors = {f"layer{i}.weight": torch.randn(4, 4) for i in range(300)}
    pack = pack_state_dict(tensors, tmp_path / "many.pack")
    manifest = load_pack_manifest(pack.path)
    assert manifest["tensor_count"] == 300
    assert pack.metadata["header_bytes"] > 4096

    fd = os.open(pack.path, os.O_RDONLY)
    try:
        for name, entry in zip(tensors, manifest["tensors"], strict=True):
            raw = os.pread(fd, entry["nbytes"], entry["offset"])
            restored = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(4, 4)
            torch.testing.assert_close(restored, tensors[name])
    finally:
        os.close(fd)


def test_load_pack_manifest_does_not_read_payload_bytes(tmp_path: Path) -> None:
    """Manifest load must not pull the whole pack into RAM."""
    payload = torch.randn(256, 256)
    pack = pack_state_dict({"w": payload}, tmp_path / "big.pack")
    file_size = pack.path.stat().st_size
    assert file_size > 200_000

    reads: list[int] = []
    real_pread = os.pread

    def counting_pread(fd: int, nbytes: int, offset: int) -> bytes:
        reads.append(nbytes)
        return real_pread(fd, nbytes, offset)

    with (
        patch("streamcompiler.storage.pack.os.pread", counting_pread),
        patch.object(Path, "read_bytes", side_effect=AssertionError("read_bytes must not be used")),
    ):
        manifest = load_pack_manifest(pack.path)

    assert manifest["tensor_count"] == 1
    assert sum(reads) < 64_000, f"manifest load read {sum(reads)} bytes of a {file_size}-byte pack"
    assert all(n < 64_000 for n in reads)


def test_pack_state_dict_writes_without_one_giant_bytearray(tmp_path: Path, monkeypatch) -> None:
    """Writer streams payloads to disk instead of assembling the full pack in RAM."""
    calls: list[int] = []
    real_bytearray = bytearray

    def tracking_bytearray(source=0):  # type: ignore[no-untyped-def]
        if isinstance(source, int) and source > 4096:
            calls.append(source)
        return real_bytearray(source)

    monkeypatch.setattr("builtins.bytearray", tracking_bytearray)
    tensors = {f"w{i}": torch.randn(64, 64) for i in range(8)}
    pack = pack_state_dict(tensors, tmp_path / "streamed.pack")
    assert pack.path.exists()
    assert not calls, f"unexpected large bytearray allocations: {calls}"
    manifest = load_pack_manifest(pack.path)
    assert manifest["tensor_count"] == 8


def test_pack_roundtrip_one_tensor_at_a_time(tmp_path: Path) -> None:
    """Two-pass packing still restores every block via pread."""
    tensors = {f"w{i}": torch.randn(96, 96) for i in range(5)}
    pack = pack_state_dict(tensors, tmp_path / "two_pass.pack")
    manifest = load_pack_manifest(pack.path)
    fd = os.open(pack.path, os.O_RDONLY)
    try:
        for name, entry in zip(tensors, manifest["tensors"], strict=True):
            raw = os.pread(fd, entry["nbytes"], entry["offset"])
            restored = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(96, 96)
            torch.testing.assert_close(restored, tensors[name])
    finally:
        os.close(fd)


def test_empty_state_dict_packs_cleanly(tmp_path: Path) -> None:
    pack = pack_state_dict({}, tmp_path / "empty.pack")
    manifest = load_pack_manifest(pack.path)
    assert manifest["tensor_count"] == 0
    assert pack.tensors == []


def test_chunked_tensor_source_writes_without_materializing_full_tensor(tmp_path: Path) -> None:
    from streamcompiler.storage.pack import ChunkedTensorSource, pack_tensors

    chunk = bytes(range(256)) * 1024
    chunk_calls: list[int] = []
    loader_calls = 0

    def loader() -> ChunkedTensorSource:
        nonlocal loader_calls
        loader_calls += 1

        def chunks():  # type: ignore[no-untyped-def]
            for index in range(4):
                chunk_calls.append(index)
                yield chunk

        return ChunkedTensorSource(
            nbytes=len(chunk) * 4,
            stored_shape=(len(chunk) * 4,),
            logical_shape=(len(chunk) * 4,),
            stored_dtype="uint8",
            logical_dtype="uint8",
            chunks=chunks,
        )

    pack = pack_tensors([("huge", loader)], tmp_path / "chunked.pack")
    assert loader_calls == 2  # metadata pass + streaming write pass
    assert chunk_calls == [0, 1, 2, 3]
    manifest = load_pack_manifest(pack.path)
    entry = manifest["tensors"][0]
    assert entry["nbytes"] == len(chunk) * 4
    fd = os.open(pack.path, os.O_RDONLY)
    try:
        raw = os.pread(fd, entry["nbytes"], entry["offset"])
    finally:
        os.close(fd)
    assert raw == chunk * 4


def test_pack_rejects_duplicate_tensor_names(tmp_path: Path) -> None:
    import pytest

    from streamcompiler.errors import StorageError
    from streamcompiler.storage.pack import pack_tensors

    with pytest.raises(StorageError, match="Duplicate tensor name"):
        pack_tensors(
            [("w", lambda: torch.ones(1)), ("w", lambda: torch.zeros(1))],
            tmp_path / "duplicate.pack",
        )


def test_pack_rejects_loader_layout_change_between_passes(tmp_path: Path) -> None:
    import pytest

    from streamcompiler.errors import StorageError
    from streamcompiler.storage.pack import pack_tensors

    calls = 0

    def loader() -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.ones(2 if calls == 1 else 3)

    with pytest.raises(StorageError, match="changed metadata between pack passes"):
        pack_tensors([("w", loader)], tmp_path / "changing.pack")
    assert not (tmp_path / "changing.pack").exists()


def test_manifest_rejects_duplicate_and_overlapping_blocks(tmp_path: Path) -> None:
    import json
    import struct

    import pytest

    from streamcompiler.errors import StorageError
    from streamcompiler.storage.pack import MAGIC, VERSION

    path = tmp_path / "bad.pack"
    entries = [
        {
            "logical_id": "w",
            "offset": 4096,
            "nbytes": 16,
            "stored_shape": [4],
            "logical_shape": [4],
            "stored_dtype": "float32",
            "logical_dtype": "float32",
            "alignment": 64,
            "checksum": "",
        },
        {
            "logical_id": "w",
            "offset": 4096,
            "nbytes": 16,
            "stored_shape": [4],
            "logical_shape": [4],
            "stored_dtype": "float32",
            "logical_dtype": "float32",
            "alignment": 64,
            "checksum": "",
        },
    ]
    manifest = json.dumps({"version": VERSION, "tensor_count": 2, "tensors": entries}).encode()
    header = MAGIC + struct.pack("<II", VERSION, len(manifest)) + manifest
    path.write_bytes(header + b"\0" * (8192 - len(header)))
    with pytest.raises(StorageError, match="Duplicate pack tensor"):
        load_pack_manifest(path)


def test_pack_rejects_symlink_paths(tmp_path: Path) -> None:
    real = tmp_path / "real.pack"
    pack_state_dict({"w": torch.ones(4)}, real)
    link = tmp_path / "link.pack"
    link.symlink_to(real)
    with pytest.raises(StorageError, match="symlink"):
        load_pack_manifest(link)
    with pytest.raises(StorageError, match="symlink"):
        pack_state_dict({"w": torch.ones(4)}, link)


def test_quantized_state_rejects_symlink_paths(tmp_path: Path) -> None:
    from streamcompiler.storage.quantized import load_quantized_state_dict, pack_quantized_state_dict

    real = tmp_path / "q.pt"
    pack_quantized_state_dict({"w": torch.ones(4)}, real)
    link = tmp_path / "q.link"
    link.symlink_to(real)
    with pytest.raises(StorageError, match="symlink"):
        load_quantized_state_dict(link)


def test_pack_concurrent_writers_use_isolated_temps(tmp_path: Path) -> None:
    import threading

    errors: list[BaseException] = []

    def worker(worker_id: int) -> None:
        try:
            for step in range(15):
                path = tmp_path / f"w{worker_id}_{step}.pack"
                pack_state_dict({"w": torch.ones(8)}, path)
                assert load_pack_manifest(path)["tensor_count"] == 1
        except BaseException as exc:  # noqa: BLE001 - collect for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert not errors
