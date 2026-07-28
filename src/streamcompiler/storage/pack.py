"""Packed model storage format (aligned single-file packs)."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from streamcompiler.errors import StorageError

MAGIC = b"SCPACK1\0"
VERSION = 1


@dataclass
class TensorBlock:
    logical_id: str
    offset: int
    nbytes: int
    stored_shape: tuple[int, ...]
    logical_shape: tuple[int, ...]
    stored_dtype: str
    logical_dtype: str
    layout: str = "contiguous"
    compression: str = "none"
    alignment: int = 64
    shard: str | None = None
    checksum: str = ""


@dataclass
class ModelPack:
    path: Path
    tensors: list[TensorBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _align(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _block_entry(block: TensorBlock) -> dict[str, Any]:
    return {
        "logical_id": block.logical_id,
        "offset": block.offset,
        "nbytes": block.nbytes,
        "stored_shape": list(block.stored_shape),
        "logical_shape": list(block.logical_shape),
        "stored_dtype": block.stored_dtype,
        "logical_dtype": block.logical_dtype,
        "layout": block.layout,
        "compression": block.compression,
        "alignment": block.alignment,
        "shard": block.shard,
        "checksum": block.checksum,
    }


def _manifest_bytes(blocks: list[TensorBlock]) -> bytes:
    manifest = {
        "version": VERSION,
        "tensor_count": len(blocks),
        "tensors": [_block_entry(b) for b in blocks],
    }
    return json.dumps(manifest, sort_keys=True).encode("utf-8")


def pack_state_dict(state_dict: dict[str, Any], path: Path, alignment: int = 64) -> ModelPack:
    """Pack tensors into one aligned file with a manifest and offset table.

    The header is sized from the manifest it must hold, so packs with thousands of
    tensors work. Because offsets appear inside the manifest, the header size and
    the manifest length are solved by iterating until the layout is stable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payloads: list[tuple[str, bytes, tuple[int, ...], str]] = []
    for name, value in state_dict.items():
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            data = bytes(tensor.numpy().tobytes())
            shape = tuple(int(x) for x in tensor.shape)
            dtype = str(tensor.dtype).replace("torch.", "")
        else:
            # Accept raw bytes with metadata dict.
            data = bytes(value.get("data", b""))
            shape = tuple(value.get("shape", ()))
            dtype = str(value.get("dtype", "uint8"))
        payloads.append((name, data, shape, dtype))

    checksums = [hashlib.sha256(data).hexdigest()[:16] for _, data, _, _ in payloads]
    header_reserve = max(4096, 512 * (len(payloads) + 1))
    for _ in range(8):
        blocks: list[TensorBlock] = []
        cursor = header_reserve
        for (name, data, shape, dtype), checksum in zip(payloads, checksums, strict=True):
            cursor = _align(cursor, alignment)
            blocks.append(
                TensorBlock(
                    logical_id=name,
                    offset=cursor,
                    nbytes=len(data),
                    stored_shape=shape,
                    logical_shape=shape,
                    stored_dtype=dtype,
                    logical_dtype=dtype,
                    alignment=alignment,
                    checksum=checksum,
                )
            )
            cursor += len(data)
        manifest_bytes = _manifest_bytes(blocks)
        needed = len(MAGIC) + 8 + len(manifest_bytes)
        if needed <= header_reserve:
            break
        header_reserve = _align(needed + needed // 8 + 1024, alignment)
    else:  # pragma: no cover - the loop converges after one growth in practice
        raise StorageError(f"Could not lay out a pack header for {len(payloads)} tensors")

    blob = bytearray(b"\0" * header_reserve)
    for block, (_, data, _, _) in zip(blocks, payloads, strict=True):
        if block.offset > len(blob):
            blob.extend(b"\0" * (block.offset - len(blob)))
        blob.extend(data)
    header = MAGIC + struct.pack("<II", VERSION, len(manifest_bytes)) + manifest_bytes
    blob[: len(header)] = header
    path.write_bytes(blob)
    return ModelPack(
        path=path,
        tensors=blocks,
        metadata={"version": VERSION, "header_bytes": header_reserve},
    )


def load_pack_manifest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"Not a StreamCompiler pack: {path}")
    version, manifest_len = struct.unpack_from("<II", data, 8)
    if version != VERSION:
        raise ValueError(f"Unsupported pack version {version}")
    manifest = json.loads(data[16 : 16 + manifest_len].decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Pack manifest is not an object: {path}")
    return manifest
