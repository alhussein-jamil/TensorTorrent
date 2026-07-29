"""Packed model storage format (aligned single-file packs)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import struct
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from streamcompiler.errors import StorageError

MAGIC = b"SCPACK1\0"
VERSION = 1

TensorLoader = Callable[[], Any]


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


@dataclass
class _TensorMeta:
    name: str
    nbytes: int
    stored_shape: tuple[int, ...]
    logical_shape: tuple[int, ...]
    stored_dtype: str
    logical_dtype: str
    compression: str
    scale: float | None
    zero_point: int | None
    loader: TensorLoader


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


def _describe_value(name: str, value: Any, *, quantize: bool) -> tuple[_TensorMeta, Any]:
    """Return layout metadata and the materialised payload for one tensor."""
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
            payload: Any = q.qdata.contiguous()
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
        meta = _TensorMeta(
            name=name,
            nbytes=nbytes,
            stored_shape=stored_shape,
            logical_shape=logical_shape,
            stored_dtype=stored_dtype,
            logical_dtype=logical_dtype,
            compression=compression,
            scale=scale,
            zero_point=zero_point,
            loader=lambda: None,
        )
        return meta, payload
    data = bytes(value.get("data", b""))
    shape = tuple(value.get("shape", ()))
    dtype = str(value.get("dtype", "uint8"))
    meta = _TensorMeta(
        name=name,
        nbytes=len(data),
        stored_shape=shape,
        logical_shape=shape,
        stored_dtype=dtype,
        logical_dtype=dtype,
        compression="none",
        scale=None,
        zero_point=None,
        loader=lambda: None,
    )
    return meta, data


def _write_payload_chunked(handle: Any, payload: Any, *, offset: int, expected_nbytes: int) -> str:
    """Write payload in chunks with an incremental checksum; avoid one giant bytearray."""
    chunk = 1 << 20
    hasher = hashlib.sha256()
    handle.seek(offset)
    written = 0
    if hasattr(payload, "detach"):
        tensor = payload.detach().cpu().contiguous()
        # memoryview over numpy shares storage when possible (incl. int8 quantized)
        mv = memoryview(tensor.numpy()).cast("B")
        pos = 0
        while pos < len(mv):
            piece = mv[pos : pos + chunk]
            handle.write(piece)
            hasher.update(piece)
            written += len(piece)
            pos += len(piece)
    elif hasattr(payload, "numpy"):
        mv = memoryview(payload.numpy()).cast("B")
        pos = 0
        while pos < len(mv):
            piece = mv[pos : pos + chunk]
            handle.write(piece)
            hasher.update(piece)
            written += len(piece)
            pos += len(piece)
    else:
        raw = payload if isinstance(payload, (bytes, bytearray, memoryview)) else bytes(payload)
        data = memoryview(raw)
        pos = 0
        while pos < len(data):
            piece = data[pos : pos + chunk]
            handle.write(piece)
            hasher.update(piece)
            written += len(piece)
            pos += len(piece)
    if written != expected_nbytes:
        raise StorageError(f"Pack payload size mismatch: layout {expected_nbytes} vs written {written}")
    return hasher.hexdigest()[:16]


def pack_tensors(
    named_loaders: Iterable[tuple[str, TensorLoader]],
    destination: Path,
    alignment: int = 64,
    *,
    quantize: bool = False,
) -> ModelPack:
    """Two-pass pack writer that never retains a full second model copy.

    Pass one materialises each loader only long enough to record shape/dtype/
    nbytes, then drops the payload. Pass two re-invokes loaders one at a time,
    writes each payload to its reserved offset, and releases it before the next.
    """
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    metas: list[_TensorMeta] = []
    for name, loader in named_loaders:
        value = loader()
        meta, payload = _describe_value(name, value, quantize=quantize)
        del value, payload
        metas.append(
            _TensorMeta(
                name=meta.name,
                nbytes=meta.nbytes,
                stored_shape=meta.stored_shape,
                logical_shape=meta.logical_shape,
                stored_dtype=meta.stored_dtype,
                logical_dtype=meta.logical_dtype,
                compression=meta.compression,
                scale=meta.scale,
                zero_point=meta.zero_point,
                loader=loader,
            )
        )

    header_reserve = max(4096, 512 * (len(metas) + 1))
    for _ in range(8):
        blocks: list[TensorBlock] = []
        offset = header_reserve
        for meta in metas:
            offset = _align(offset, alignment)
            blocks.append(
                TensorBlock(
                    logical_id=meta.name,
                    offset=offset,
                    nbytes=meta.nbytes,
                    stored_shape=meta.stored_shape,
                    logical_shape=meta.logical_shape,
                    stored_dtype=meta.stored_dtype,
                    logical_dtype=meta.logical_dtype,
                    compression=meta.compression,
                    alignment=alignment,
                    checksum="",
                    scale=meta.scale,
                    zero_point=meta.zero_point,
                )
            )
            offset += meta.nbytes
        for block in blocks:
            block.checksum = "0" * 16
        manifest_bytes = _manifest_bytes(blocks)
        needed = len(MAGIC) + 8 + len(manifest_bytes)
        if needed <= header_reserve:
            break
        header_reserve = _align(needed + needed // 8 + 1024, alignment)
    else:  # pragma: no cover
        raise StorageError(f"Could not lay out a pack header for {len(metas)} tensors")

    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(b"\0" * header_reserve)
            for block, meta in zip(blocks, metas, strict=True):
                value = meta.loader()
                _meta2, payload = _describe_value(meta.name, value, quantize=quantize)
                del value
                # Always stream via chunked writer (incl. int8 quantized tensors).
                block.checksum = _write_payload_chunked(
                    handle, payload, offset=block.offset, expected_nbytes=block.nbytes
                )
                del payload
            manifest_bytes = _manifest_bytes(blocks)
            header = MAGIC + struct.pack("<II", VERSION, len(manifest_bytes)) + manifest_bytes
            if len(header) > header_reserve:
                raise StorageError(f"Pack header ({len(header)} bytes) exceeds reserve {header_reserve}")
            handle.seek(0)
            handle.write(header)
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise

    return ModelPack(
        path=path,
        tensors=blocks,
        metadata={
            "version": VERSION,
            "header_bytes": header_reserve,
            "quantize": bool(quantize),
        },
    )


def pack_state_dict(
    state_dict: dict[str, Any],
    path: Path,
    alignment: int = 64,
    *,
    quantize: bool = False,
) -> ModelPack:
    """Pack tensors into one aligned file with a manifest and offset table.

    Delegates to :func:`pack_tensors` so peak memory does not scale as an extra
    full-model payload copy.
    """

    def _loaders() -> Iterator[tuple[str, TensorLoader]]:
        for name in state_dict:

            def _load(n: str = name) -> Any:
                return state_dict[n]

            yield name, _load

    return pack_tensors(_loaders(), path, alignment=alignment, quantize=quantize)


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
