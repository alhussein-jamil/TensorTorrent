"""Model pack format tests: the streaming runtime depends on these offsets."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import torch

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
