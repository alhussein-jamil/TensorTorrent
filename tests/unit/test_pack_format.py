"""Model pack format tests: the streaming runtime depends on these offsets."""

from __future__ import annotations

import os
from pathlib import Path

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
