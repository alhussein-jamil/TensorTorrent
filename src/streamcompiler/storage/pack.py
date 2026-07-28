"""Packed model storage format (aligned single-file packs)."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

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
    scale: float | None = None
    zero_point: int | None = None


@dataclass
class ModelPack:
    path: Path
    tensors: list[TensorBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _align(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _block_entry(block: TensorBlock) -> dict[str, Any]:
    entry: dict[str, Any] = {
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
    if block.scale is not None:
        entry["scale"] = float(block.scale)
    if block.zero_point is not None:
        entry["zero_point"] = int(block.zero_point)
    return entry


def _manifest_bytes(blocks: list[TensorBlock]) -> bytes:
    manifest = {
        "version": VERSION,
        "tensor_count": len(blocks),
        "tensors": [_block_entry(b) for b in blocks],
    }
    return json.dumps(manifest, sort_keys=True).encode("utf-8")


def pack_state_dict(
    state_dict: dict[str, Any],
    path: Path,
    alignment: int = 64,
    *,
    quantize: bool = False,
) -> ModelPack:
    """Pack tensors into one aligned file with a manifest and offset table.

    The header is sized from the manifest it must hold, so packs with thousands of
    tensors work. Because offsets appear inside the manifest, the header size and
    the manifest length are solved by iterating until the layout is stable.

    Each tensor is serialized once, at write time. Layout uses ``numel`` /
    ``element_size`` (or declared raw sizes) so payloads are not retained as a
    second full-model byte copy.

    When ``quantize`` is true, floating tensors are stored as int8 with an affine
    scale (``compression=int8_affine``); the streaming loader dequantizes on read.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    metas: list[tuple[str, int, tuple[int, ...], tuple[int, ...], str, str, str, float | None, int | None, Any]] = []
    for name, value in state_dict.items():
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            logical_shape = tuple(int(x) for x in tensor.shape)
            logical_dtype = str(tensor.dtype).replace("torch.", "")
            scale: float | None = None
            zero_point: int | None = None
            compression = "none"
            if quantize and tensor.dtype in (torch.float32, torch.float16, torch.bfloat16):
                from streamcompiler.storage.quantized import quantize_per_tensor

                q = quantize_per_tensor(tensor)
                payload = q.qdata.contiguous()
                stored_shape = tuple(int(x) for x in payload.shape)
                stored_dtype = "int8"
                scale = float(q.scale)
                zero_point = int(q.zero_point)
                compression = "int8_affine"
                nbytes = int(payload.numel() * payload.element_size())
            else:
                payload = tensor
                stored_shape = logical_shape
                stored_dtype = logical_dtype
                nbytes = int(tensor.numel() * tensor.element_size())
            metas.append(
                (
                    name,
                    nbytes,
                    stored_shape,
                    logical_shape,
                    stored_dtype,
                    logical_dtype,
                    compression,
                    scale,
                    zero_point,
                    payload,
                )
            )
        else:
            data = bytes(value.get("data", b""))
            shape = tuple(value.get("shape", ()))
            dtype = str(value.get("dtype", "uint8"))
            metas.append((name, len(data), shape, shape, dtype, dtype, "none", None, None, data))

    header_reserve = max(4096, 512 * (len(metas) + 1))
    for _ in range(8):
        blocks: list[TensorBlock] = []
        cursor = header_reserve
        for (
            name,
            nbytes,
            stored_shape,
            logical_shape,
            stored_dtype,
            logical_dtype,
            compression,
            scale,
            zero_point,
            _payload,
        ) in metas:
            cursor = _align(cursor, alignment)
            blocks.append(
                TensorBlock(
                    logical_id=name,
                    offset=cursor,
                    nbytes=nbytes,
                    stored_shape=stored_shape,
                    logical_shape=logical_shape,
                    stored_dtype=stored_dtype,
                    logical_dtype=logical_dtype,
                    compression=compression,
                    alignment=alignment,
                    checksum="",  # filled while writing payloads
                    scale=scale,
                    zero_point=zero_point,
                )
            )
            cursor += nbytes
        # Checksums are 16 hex chars; reserve that length so the header size is stable.
        for block in blocks:
            block.checksum = "0" * 16
        manifest_bytes = _manifest_bytes(blocks)
        needed = len(MAGIC) + 8 + len(manifest_bytes)
        if needed <= header_reserve:
            break
        header_reserve = _align(needed + needed // 8 + 1024, alignment)
    else:  # pragma: no cover - the loop converges after one growth in practice
        raise StorageError(f"Could not lay out a pack header for {len(metas)} tensors")

    with path.open("wb") as handle:
        handle.write(b"\0" * header_reserve)
        for block, (_name, _nbytes, _s, _l, _sd, _ld, _c, _sc, _zp, payload) in zip(blocks, metas, strict=True):
            data = bytes(payload.numpy().tobytes()) if hasattr(payload, "numpy") else bytes(payload)
            if len(data) != block.nbytes:
                raise StorageError(
                    f"Pack payload size mismatch for {block.logical_id}: "
                    f"layout {block.nbytes} vs serialized {len(data)}"
                )
            block.checksum = hashlib.sha256(data).hexdigest()[:16]
            handle.seek(block.offset)
            handle.write(data)
            del data
        manifest_bytes = _manifest_bytes(blocks)
        header = MAGIC + struct.pack("<II", VERSION, len(manifest_bytes)) + manifest_bytes
        if len(header) > header_reserve:
            raise StorageError(f"Pack header ({len(header)} bytes) exceeds reserve {header_reserve}")
        handle.seek(0)
        handle.write(header)

    return ModelPack(
        path=path,
        tensors=blocks,
        metadata={
            "version": VERSION,
            "header_bytes": header_reserve,
            "quantize": bool(quantize),
        },
    )


def load_pack_manifest(path: Path) -> dict[str, Any]:
    """Read only the pack header and JSON manifest.

    Tensor payloads stay on disk. Callers that need a block use ``os.pread`` at the
    offsets recorded in the manifest.
    """
    path = Path(path)
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise StorageError(f"Cannot stat pack {path}: {exc}") from exc
    fd = os.open(path, os.O_RDONLY)
    try:
        head = os.pread(fd, 16, 0)
        if len(head) < 16:
            raise StorageError(f"Pack file too small for header: {path}")
        if head[:8] != MAGIC:
            raise StorageError(f"Not a StreamCompiler pack: {path}")
        version, manifest_len = struct.unpack_from("<II", head, 8)
        if version != VERSION:
            raise StorageError(f"Unsupported pack version {version}")
        if manifest_len < 0 or 16 + manifest_len > file_size:
            raise StorageError(f"Pack manifest length {manifest_len} exceeds file size for {path}")
        raw_manifest = os.pread(fd, manifest_len, 16)
        if len(raw_manifest) != manifest_len:
            raise StorageError(f"Short read of pack manifest in {path}")
    finally:
        os.close(fd)
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StorageError(f"Corrupt pack manifest in {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise StorageError(f"Pack manifest is not an object: {path}")
    validate_pack_manifest(manifest, file_size=file_size, path=path)
    return manifest


def validate_pack_manifest(manifest: dict[str, Any], *, file_size: int, path: Path | str) -> None:
    """Reject manifests whose tensor blocks fall outside the pack file."""
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise StorageError(f"Pack manifest tensors list missing: {path}")
    for entry in tensors:
        if not isinstance(entry, dict):
            raise StorageError(f"Pack manifest entry is not an object: {path}")
        try:
            offset = int(entry["offset"])
            nbytes = int(entry["nbytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StorageError(f"Pack manifest entry missing offset/nbytes: {path}") from exc
        if offset < 0 or nbytes < 0 or offset + nbytes > file_size:
            raise StorageError(
                f"Pack block {entry.get('logical_id', '?')} spans [{offset}, {offset + nbytes}) "
                f"outside file size {file_size} for {path}"
            )


def resolve_pack_path(
    packed_model_path: str | Path,
    *,
    artifact_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Resolve a pack path and refuse escapes outside the artifact or cache roots."""
    candidate = Path(packed_model_path)
    if not candidate.is_absolute():
        roots = [p for p in (artifact_dir, cache_dir) if p is not None]
        if not roots:
            raise StorageError(f"Relative packed_model_path {packed_model_path!r} needs an artifact or cache directory")
        for root in roots:
            resolved = (root / candidate).resolve()
            if resolved.exists() and _is_within(resolved, root.resolve()):
                return resolved
        raise StorageError(f"Packed model {packed_model_path!r} not found under {roots}")
    resolved = candidate.resolve()
    allowed = [p.resolve() for p in (artifact_dir, cache_dir) if p is not None]
    if allowed and not any(_is_within(resolved, root) for root in allowed):
        raise StorageError(f"Packed model path {resolved} is outside the allowed artifact/cache directories {allowed}")
    if not resolved.exists():
        raise StorageError(f"Packed model path does not exist: {resolved}")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def verify_block_checksum(raw: bytes, checksum: str, *, logical_id: str, path: Path | str) -> None:
    """Verify a truncated SHA-256 checksum recorded in the pack manifest."""
    if not checksum:
        return
    digest = hashlib.sha256(raw).hexdigest()[: len(checksum)]
    if digest != checksum:
        raise StorageError(f"Checksum mismatch for {logical_id} in {path}")
