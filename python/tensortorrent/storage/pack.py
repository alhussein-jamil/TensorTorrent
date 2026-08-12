"""Packed model storage format (aligned single-file packs)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import struct
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from tensortorrent.closed import CompressionKind, TensorLayout
from tensortorrent.errors import StorageError
from tensortorrent.tensor_bytes import tensor_as_memoryview

MAGIC = b"SCPACK1\0"
VERSION = 1
MAX_MANIFEST_BYTES = 64 << 20
MAX_TENSOR_COUNT = 100_000
MAX_TENSOR_BYTES = 64 << 30
MAX_TENSOR_NAME_BYTES = 4096
MAX_DTYPE_BYTES = 128
MAX_TENSOR_RANK = 64

TensorLoader = Callable[[], Any]


def _coerce_compression(value: CompressionKind | str) -> CompressionKind:
    if isinstance(value, CompressionKind):
        return value
    raw = value.value if isinstance(value, Enum) else value
    return CompressionKind(str(raw))


def _coerce_layout(value: TensorLayout | str) -> TensorLayout:
    if isinstance(value, TensorLayout):
        return value
    raw = value.value if isinstance(value, Enum) else value
    return TensorLayout(str(raw))


def _is_bounded_utf8(value: Any, maximum: int) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        return len(value.encode("utf-8")) <= maximum
    except UnicodeEncodeError:
        return False


@dataclass(frozen=True)
class ChunkedTensorSource:
    """Lazy tensor payload for a single extremely large logical tensor.

    ``chunks`` must return a fresh iterable for each pack pass. Chunks are written
    incrementally and never concatenated in memory. Quantization is intentionally
    unsupported here until a streaming quantizer is implemented.
    """

    nbytes: int
    stored_shape: tuple[int, ...]
    logical_shape: tuple[int, ...]
    stored_dtype: str
    logical_dtype: str
    chunks: Callable[[], Iterable[bytes | bytearray | memoryview]]
    compression: CompressionKind = CompressionKind.NONE
    scale: float | None = None
    zero_point: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "compression", _coerce_compression(self.compression))


@dataclass
class TensorBlock:
    logical_id: str
    offset: int
    nbytes: int
    stored_shape: tuple[int, ...]
    logical_shape: tuple[int, ...]
    stored_dtype: str
    logical_dtype: str
    layout: TensorLayout = TensorLayout.CONTIGUOUS
    compression: CompressionKind = CompressionKind.NONE
    alignment: int = 64
    shard: str | None = None
    checksum: str = ""
    checksum_crc32: int | None = None
    scale: float | None = None
    zero_point: int | None = None

    def __post_init__(self) -> None:
        self.layout = _coerce_layout(self.layout)
        self.compression = _coerce_compression(self.compression)


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
    compression: CompressionKind
    scale: float | None
    zero_point: int | None
    loader: TensorLoader

    def __post_init__(self) -> None:
        self.compression = _coerce_compression(self.compression)


def _align(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _validate_alignment(alignment: int) -> None:
    if isinstance(alignment, bool) or not isinstance(alignment, int):
        raise StorageError(f"alignment must be an integer, got {type(alignment).__name__}")
    if alignment < 1 or alignment & (alignment - 1):
        raise StorageError(f"alignment must be a positive power of two, got {alignment}")


def _require_manifest_int(value: Any, *, field_name: str, path: Path | str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StorageError(f"Pack manifest {field_name} must be an integer: {path}")
    return value


def _validate_tensor_meta(meta: _TensorMeta) -> None:
    if not _is_bounded_utf8(meta.name, MAX_TENSOR_NAME_BYTES):
        raise StorageError(f"Pack tensor name exceeds {MAX_TENSOR_NAME_BYTES} bytes: {meta.name!r}")
    if isinstance(meta.nbytes, bool) or not isinstance(meta.nbytes, int) or not 0 <= meta.nbytes <= MAX_TENSOR_BYTES:
        raise StorageError(f"Pack tensor {meta.name!r} has invalid nbytes {meta.nbytes!r}")
    for label, shape in (("stored", meta.stored_shape), ("logical", meta.logical_shape)):
        if len(shape) > MAX_TENSOR_RANK or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
        ):
            raise StorageError(f"Pack tensor {meta.name!r} has invalid {label} shape")
    if not _is_bounded_utf8(meta.stored_dtype, MAX_DTYPE_BYTES):
        raise StorageError(f"Pack tensor {meta.name!r} has invalid stored dtype")
    if not _is_bounded_utf8(meta.logical_dtype, MAX_DTYPE_BYTES):
        raise StorageError(f"Pack tensor {meta.name!r} has invalid logical dtype")


def _assert_same_layout(expected: _TensorMeta, actual: _TensorMeta) -> None:
    fields = (
        "nbytes",
        "stored_shape",
        "logical_shape",
        "stored_dtype",
        "logical_dtype",
        "compression",
        "scale",
        "zero_point",
    )
    changed = [name for name in fields if getattr(expected, name) != getattr(actual, name)]
    if changed:
        detail = ", ".join(f"{name}: {getattr(expected, name)!r} -> {getattr(actual, name)!r}" for name in changed)
        raise StorageError(f"Tensor loader {expected.name!r} changed metadata between pack passes ({detail})")


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path.parent, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _block_entry(block: TensorBlock) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "logical_id": block.logical_id,
        "offset": block.offset,
        "nbytes": block.nbytes,
        "stored_shape": list(block.stored_shape),
        "logical_shape": list(block.logical_shape),
        "stored_dtype": block.stored_dtype,
        "logical_dtype": block.logical_dtype,
        "layout": block.layout.value,
        "compression": block.compression.value,
        "alignment": block.alignment,
        "shard": block.shard,
        "checksum": block.checksum,
    }
    if block.checksum_crc32 is not None:
        entry["checksum_crc32"] = int(block.checksum_crc32) & 0xFFFFFFFF
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
    """Return layout metadata and the materialised or lazy payload for one tensor."""
    if isinstance(value, ChunkedTensorSource):
        if quantize:
            raise StorageError("ChunkedTensorSource does not support quantize=True without a streaming quantizer")
        if isinstance(value.nbytes, bool) or not isinstance(value.nbytes, int):
            raise StorageError("ChunkedTensorSource.nbytes must be an integer")
        meta = _TensorMeta(
            name=name,
            nbytes=value.nbytes,
            stored_shape=tuple(value.stored_shape),
            logical_shape=tuple(value.logical_shape),
            stored_dtype=str(value.stored_dtype),
            logical_dtype=str(value.logical_dtype),
            compression=_coerce_compression(value.compression),
            scale=value.scale,
            zero_point=value.zero_point,
            loader=lambda: None,
        )
        return meta, value
    if hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        logical_shape = tuple(int(x) for x in tensor.shape)
        logical_dtype = str(tensor.dtype).replace("torch.", "")
        scale: float | None = None
        zero_point: int | None = None
        compression = CompressionKind.NONE
        if quantize and tensor.dtype in (torch.float32, torch.float16, torch.bfloat16):
            from tensortorrent.storage.quantized import quantize_per_tensor

            q = quantize_per_tensor(tensor)
            payload: Any = q.qdata.contiguous()
            stored_shape = tuple(int(x) for x in payload.shape)
            stored_dtype = "int8"
            scale = float(q.scale)
            zero_point = int(q.zero_point)
            compression = CompressionKind.INT8_AFFINE
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
        compression=CompressionKind.NONE,
        scale=None,
        zero_point=None,
        loader=lambda: None,
    )
    return meta, data


def _write_memoryview_chunks(
    handle: Any,
    data: memoryview,
    *,
    hasher: Any,
    crc: int,
    chunk: int,
) -> tuple[int, int]:
    """Write ``data`` in ``chunk``-sized pieces; return (bytes_written, crc32)."""
    import zlib

    written = 0
    pos = 0
    while pos < len(data):
        piece = data[pos : pos + chunk]
        handle.write(piece)
        hasher.update(piece)
        crc = zlib.crc32(piece, crc) & 0xFFFFFFFF
        written += len(piece)
        pos += len(piece)
    return written, crc


def _payload_as_memoryview(payload: Any) -> memoryview:
    """Coerce a tensor-like or buffer payload to a ``uint8`` memoryview."""
    if isinstance(payload, torch.Tensor):
        return tensor_as_memoryview(payload)
    if hasattr(payload, "detach"):
        return tensor_as_memoryview(torch.as_tensor(payload))
    if hasattr(payload, "numpy"):
        try:
            return memoryview(payload.numpy()).cast("B")
        except TypeError:
            return tensor_as_memoryview(torch.as_tensor(payload))
    raw = payload if isinstance(payload, (bytes, bytearray, memoryview)) else bytes(payload)
    return memoryview(raw).cast("B") if not isinstance(raw, memoryview) else raw.cast("B")


def _write_payload_chunked(handle: Any, payload: Any, *, offset: int, expected_nbytes: int) -> tuple[str, int]:
    """Write payload in chunks; return (truncated SHA-256, IEEE CRC32)."""
    import zlib

    chunk = 1 << 20
    hasher = hashlib.sha256()
    crc = 0
    handle.seek(offset)
    written = 0
    if isinstance(payload, ChunkedTensorSource):
        for raw_chunk in payload.chunks():
            piece = memoryview(raw_chunk).cast("B")
            if not piece:
                continue
            handle.write(piece)
            hasher.update(piece)
            crc = zlib.crc32(piece, crc) & 0xFFFFFFFF
            written += len(piece)
    else:
        n, crc = _write_memoryview_chunks(
            handle,
            _payload_as_memoryview(payload),
            hasher=hasher,
            crc=crc,
            chunk=chunk,
        )
        written += n
    if written != expected_nbytes:
        raise StorageError(f"Pack payload size mismatch: layout {expected_nbytes} vs written {written}")
    return hasher.hexdigest()[:16], crc


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
    if path.is_symlink():
        raise StorageError(f"Refusing to write pack via symlink: {path}")
    _validate_alignment(alignment)
    path.parent.mkdir(parents=True, exist_ok=True)
    metas: list[_TensorMeta] = []
    seen_names: set[str] = set()
    for name, loader in named_loaders:
        if not isinstance(name, str) or not name:
            raise StorageError("Pack tensor names must be non-empty strings")
        if name in seen_names:
            raise StorageError(f"Duplicate tensor name in pack input: {name!r}")
        if not callable(loader):
            raise StorageError(f"Tensor loader for {name!r} is not callable")
        seen_names.add(name)
        value = loader()
        meta, payload = _describe_value(name, value, quantize=quantize)
        _validate_tensor_meta(meta)
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
        if len(metas) > MAX_TENSOR_COUNT:
            raise StorageError(f"Pack contains too many tensors: > {MAX_TENSOR_COUNT}")

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
        if len(manifest_bytes) > MAX_MANIFEST_BYTES:
            raise StorageError(f"Pack manifest is too large: {len(manifest_bytes)} bytes > {MAX_MANIFEST_BYTES}")
        needed = len(MAGIC) + 8 + len(manifest_bytes)
        if needed <= header_reserve:
            break
        header_reserve = _align(needed + needed // 8 + 1024, alignment)
    else:  # pragma: no cover
        raise StorageError(f"Could not lay out a pack header for {len(metas)} tensors")

    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(b"\0" * header_reserve)
            for block, meta in zip(blocks, metas, strict=True):
                value = meta.loader()
                meta2, payload = _describe_value(meta.name, value, quantize=quantize)
                del value
                _assert_same_layout(meta, meta2)
                # Always stream via chunked writer (incl. int8 quantized tensors).
                block.checksum, block.checksum_crc32 = _write_payload_chunked(
                    handle, payload, offset=block.offset, expected_nbytes=block.nbytes
                )
                del payload
            manifest_bytes = _manifest_bytes(blocks)
            header = MAGIC + struct.pack("<II", VERSION, len(manifest_bytes)) + manifest_bytes
            if len(header) > header_reserve:
                raise StorageError(f"Pack header ({len(header)} bytes) exceeds reserve {header_reserve}")
            handle.seek(0)
            handle.write(header)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_parent(path)
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
    if path.is_symlink():
        raise StorageError(f"Refusing to load pack via symlink: {path}")
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
            raise StorageError(f"Not a TensorTorrent pack: {path}")
        version, manifest_len = struct.unpack_from("<II", head, 8)
        if version != VERSION:
            raise StorageError(f"Unsupported pack version {version}")
        if manifest_len > MAX_MANIFEST_BYTES:
            raise StorageError(
                f"Pack manifest length {manifest_len} exceeds safety limit {MAX_MANIFEST_BYTES} for {path}"
            )
        if 16 + manifest_len > file_size:
            raise StorageError(f"Pack manifest length {manifest_len} exceeds file size for {path}")
        raw_manifest = os.pread(fd, manifest_len, 16)
        if len(raw_manifest) != manifest_len:
            raise StorageError(f"Short read of pack manifest in {path}")
    finally:
        os.close(fd)
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise StorageError(f"Corrupt pack manifest in {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise StorageError(f"Pack manifest is not an object: {path}")
    validate_pack_manifest(manifest, file_size=file_size, path=path, header_bytes=16 + manifest_len)
    return manifest


def validate_pack_manifest(
    manifest: dict[str, Any],
    *,
    file_size: int,
    path: Path | str,
    header_bytes: int = 0,
) -> None:
    """Reject malformed, duplicate, overlapping, or out-of-range tensor blocks."""
    if _require_manifest_int(manifest.get("version"), field_name="version", path=path) != VERSION:
        raise StorageError(f"Pack manifest version mismatch: {path}")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise StorageError(f"Pack manifest tensors list missing: {path}")
    if len(tensors) > MAX_TENSOR_COUNT:
        raise StorageError(f"Pack manifest contains too many tensors: {len(tensors)}")
    declared_count = manifest.get("tensor_count")
    if declared_count is not None and _require_manifest_int(
        declared_count, field_name="tensor_count", path=path
    ) != len(tensors):
        raise StorageError(f"Pack tensor_count mismatch: declared {declared_count}, found {len(tensors)} in {path}")

    dtype_sizes = {
        "bool": 1,
        "uint8": 1,
        "int8": 1,
        "int16": 2,
        "float16": 2,
        "bfloat16": 2,
        "int32": 4,
        "float32": 4,
        "int64": 8,
        "float64": 8,
    }
    seen: set[str] = set()
    spans: list[tuple[int, int, str]] = []
    for entry in tensors:
        if not isinstance(entry, dict):
            raise StorageError(f"Pack manifest entry is not an object: {path}")
        logical_id = entry.get("logical_id")
        if not isinstance(logical_id, str) or not logical_id:
            raise StorageError(f"Pack manifest entry has invalid logical_id: {path}")
        if not _is_bounded_utf8(logical_id, MAX_TENSOR_NAME_BYTES):
            raise StorageError(f"Pack manifest logical_id exceeds {MAX_TENSOR_NAME_BYTES} bytes: {path}")
        if logical_id in seen:
            raise StorageError(f"Duplicate pack tensor logical_id {logical_id!r}: {path}")
        seen.add(logical_id)
        try:
            offset = _require_manifest_int(entry["offset"], field_name="offset", path=path)
            nbytes = _require_manifest_int(entry["nbytes"], field_name="nbytes", path=path)
            alignment = _require_manifest_int(entry.get("alignment", 1), field_name="alignment", path=path)
        except KeyError as exc:
            raise StorageError(f"Pack manifest entry missing offset/nbytes: {path}") from exc
        _validate_alignment(alignment)
        if offset < header_bytes:
            raise StorageError(
                f"Pack block {logical_id!r} starts inside the manifest/header: {offset} < {header_bytes}"
            )
        if offset % alignment:
            raise StorageError(f"Pack block {logical_id!r} offset {offset} violates alignment {alignment}")
        if nbytes < 0 or nbytes > MAX_TENSOR_BYTES or offset > file_size or nbytes > file_size - offset:
            raise StorageError(
                f"Pack block {logical_id} spans [{offset}, {offset + nbytes}) outside file size {file_size} for {path}"
            )
        stored_shape = entry.get("stored_shape")
        if not isinstance(stored_shape, list) or len(stored_shape) > MAX_TENSOR_RANK:
            raise StorageError(f"Pack block {logical_id!r} has invalid stored_shape")
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in stored_shape):
            raise StorageError(f"Pack block {logical_id!r} has negative shape dimensions")
        shape = tuple(stored_shape)
        stored_dtype = entry.get("stored_dtype")
        if not isinstance(stored_dtype, str) or not _is_bounded_utf8(stored_dtype, MAX_DTYPE_BYTES):
            raise StorageError(f"Pack block {logical_id!r} has invalid stored_dtype")
        itemsize = dtype_sizes.get(stored_dtype)
        if itemsize is not None:
            expected_nbytes = 1
            for dim in shape:
                expected_nbytes *= dim
            expected_nbytes *= itemsize
            if expected_nbytes != nbytes:
                raise StorageError(f"Pack block {logical_id!r} metadata expects {expected_nbytes} bytes, got {nbytes}")
        logical_shape = entry.get("logical_shape", stored_shape)
        if not isinstance(logical_shape, list) or len(logical_shape) > MAX_TENSOR_RANK:
            raise StorageError(f"Pack block {logical_id!r} has invalid logical_shape")
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in logical_shape):
            raise StorageError(f"Pack block {logical_id!r} has invalid logical shape dimensions")
        logical_dtype = entry.get("logical_dtype", stored_dtype)
        if not _is_bounded_utf8(logical_dtype, MAX_DTYPE_BYTES):
            raise StorageError(f"Pack block {logical_id!r} has invalid logical_dtype")
        compression = entry.get("compression", CompressionKind.NONE)
        try:
            compression_kind = _coerce_compression(compression)
        except ValueError as exc:
            raise StorageError(f"Pack block {logical_id!r} has unsupported compression {compression!r}") from exc
        scale = entry.get("scale")
        zero_point = entry.get("zero_point")
        if scale is not None and (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or scale <= 0
        ):
            raise StorageError(f"Pack block {logical_id!r} has an invalid quantization scale")
        if zero_point is not None and (
            isinstance(zero_point, bool) or not isinstance(zero_point, int) or not -128 <= zero_point <= 127
        ):
            raise StorageError(f"Pack block {logical_id!r} has an invalid quantization zero_point")
        if compression_kind == CompressionKind.NONE:
            if logical_shape != stored_shape or logical_dtype != stored_dtype:
                raise StorageError(f"Uncompressed pack block {logical_id!r} has mismatched logical metadata")
        elif stored_dtype != "int8" or scale is None or logical_dtype not in dtype_sizes:
            raise StorageError(
                f"int8_affine pack block {logical_id!r} requires int8 storage, a scale, and a supported logical dtype"
            )
        layout = entry.get("layout", TensorLayout.CONTIGUOUS)
        try:
            _coerce_layout(layout)
        except ValueError as exc:
            raise StorageError(f"Pack block {logical_id!r} has unsupported layout {layout!r}") from exc
        checksum = entry.get("checksum", "")
        if not isinstance(checksum, str) or (
            checksum and (len(checksum) > 64 or any(ch not in "0123456789abcdef" for ch in checksum))
        ):
            raise StorageError(f"Pack block {logical_id!r} has an invalid checksum")
        checksum_crc32 = entry.get("checksum_crc32")
        if checksum_crc32 is not None and (
            isinstance(checksum_crc32, bool)
            or not isinstance(checksum_crc32, int)
            or not 0 <= checksum_crc32 <= 0xFFFFFFFF
        ):
            raise StorageError(f"Pack block {logical_id!r} has an invalid CRC32 checksum")
        spans.append((offset, offset + nbytes, logical_id))

    spans.sort()
    active: tuple[int, str] | None = None
    for offset, end, logical_id in spans:
        if active is not None and offset < active[0] and offset < end:
            raise StorageError(f"Pack blocks {active[1]!r} and {logical_id!r} overlap in {path}")
        if offset < end and (active is None or end > active[0]):
            active = (end, logical_id)


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
