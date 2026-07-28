"""Packed model storage format (aligned single-file packs)."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def pack_state_dict(state_dict: dict[str, Any], path: Path, alignment: int = 64) -> ModelPack:
    """Pack tensors into one aligned file with manifest + offset table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[TensorBlock] = []
    blob = bytearray()
    # Reserve header space rewritten at the end.
    header_placeholder = 4096
    blob.extend(b"\0" * header_placeholder)

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
        offset = _align(len(blob), alignment)
        if offset > len(blob):
            blob.extend(b"\0" * (offset - len(blob)))
        checksum = hashlib.sha256(data).hexdigest()[:16]
        blob.extend(data)
        blocks.append(
            TensorBlock(
                logical_id=name,
                offset=offset,
                nbytes=len(data),
                stored_shape=shape,
                logical_shape=shape,
                stored_dtype=dtype,
                logical_dtype=dtype,
                alignment=alignment,
                checksum=checksum,
            )
        )

    manifest = {
        "version": VERSION,
        "tensor_count": len(blocks),
        "tensors": [
            {
                "logical_id": b.logical_id,
                "offset": b.offset,
                "nbytes": b.nbytes,
                "stored_shape": list(b.stored_shape),
                "logical_shape": list(b.logical_shape),
                "stored_dtype": b.stored_dtype,
                "logical_dtype": b.logical_dtype,
                "layout": b.layout,
                "compression": b.compression,
                "alignment": b.alignment,
                "shard": b.shard,
                "checksum": b.checksum,
            }
            for b in blocks
        ],
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    header = MAGIC + struct.pack("<II", VERSION, len(manifest_bytes)) + manifest_bytes
    if len(header) > header_placeholder:
        raise ValueError("manifest exceeds reserved header; increase header_placeholder")
    blob[: len(header)] = header
    path.write_bytes(blob)
    return ModelPack(path=path, tensors=blocks, metadata={"version": VERSION})


def load_pack_manifest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"Not a StreamCompiler pack: {path}")
    version, manifest_len = struct.unpack_from("<II", data, 8)
    if version != VERSION:
        raise ValueError(f"Unsupported pack version {version}")
    return json.loads(data[16 : 16 + manifest_len].decode("utf-8"))
