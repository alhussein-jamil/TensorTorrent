"""Storage bandwidth probes used by streaming specialization."""

from __future__ import annotations

from pathlib import Path

import torch

from tensortorrent.hardware.storage_bench import (
    benchmark_pack_payload,
    benchmark_pread,
    benchmark_sequential_read,
)
from tensortorrent.storage.pack import pack_state_dict


def test_benchmark_pread_measures_real_bytes(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    payload = b"\xab" * (1 << 20)
    target.write_bytes(payload)
    result = benchmark_pread(target, offset=0, nbytes=len(payload), iters=2)
    assert result.measured
    assert result.nbytes == len(payload)
    assert result.bytes_per_s > 0
    assert result.latency_s > 0


def test_benchmark_pack_payload_uses_manifest_offsets(tmp_path: Path) -> None:
    tensor = torch.randn(128, 128)
    pack = pack_state_dict({"w": tensor}, tmp_path / "m.pack")
    block = pack.tensors[0]
    result = benchmark_pack_payload(pack.path, offset=block.offset, nbytes=block.nbytes)
    assert result.measured
    assert result.nbytes == block.nbytes
    assert "model pack" in result.notes


def test_benchmark_sequential_read_on_directory(tmp_path: Path) -> None:
    result = benchmark_sequential_read(tmp_path, nbytes=1 << 16)
    assert result.measured
    assert result.nbytes == 1 << 16
