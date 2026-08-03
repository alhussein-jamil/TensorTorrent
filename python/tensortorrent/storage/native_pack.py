"""Native pack reader bridge for SCPACK1 production loads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tensortorrent.errors import StorageError
from tensortorrent.native import native_available, require_native
from tensortorrent.storage.pack import load_pack_manifest


def scpack_to_native_manifest_json(manifest: dict[str, Any]) -> str:
    """Adapt SCPACK1 manifest fields to the Rust PackManifest schema."""
    if not isinstance(manifest, dict):
        raise StorageError("Pack manifest must be a dictionary")
    version = manifest.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise StorageError("Pack manifest version must be an integer")
    entries = manifest.get("tensors")
    if not isinstance(entries, list):
        raise StorageError("Pack manifest tensors must be a list")
    tensors = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise StorageError("Pack manifest tensor entry must be a dictionary")
        name = entry.get("logical_id", entry.get("name"))
        if not isinstance(name, str) or not name:
            raise StorageError("Pack manifest tensor name must be a non-empty string")
        offset = entry.get("offset")
        length = entry["nbytes"] if "nbytes" in entry else entry.get("length")
        dtype = entry.get("stored_dtype", entry.get("dtype"))
        shape = entry.get("stored_shape", entry.get("shape"))
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise StorageError(f"Pack tensor {name!r} offset must be an integer")
        if isinstance(length, bool) or not isinstance(length, int):
            raise StorageError(f"Pack tensor {name!r} length must be an integer")
        if not isinstance(dtype, str) or not dtype:
            raise StorageError(f"Pack tensor {name!r} dtype must be a non-empty string")
        if not isinstance(shape, list) or any(isinstance(dim, bool) or not isinstance(dim, int) for dim in shape):
            raise StorageError(f"Pack tensor {name!r} shape must be a list of integers")
        checksum = entry.get("checksum_crc32")
        if checksum is not None and (
            isinstance(checksum, bool) or not isinstance(checksum, int) or not 0 <= checksum <= 0xFFFFFFFF
        ):
            raise StorageError(f"Pack tensor {name!r} CRC32 checksum is invalid")
        tensors.append(
            {
                "name": name,
                "offset": offset,
                "length": length,
                "dtype": dtype,
                "shape": shape,
                "checksum_crc32": checksum,
            }
        )
    return json.dumps({"version": version, "tensors": tensors})


def open_native_pack_reader(pack_path: Path | str, manifest: dict[str, Any] | None = None) -> Any | None:
    """Open NativePackReader when the extension is loaded; else None."""
    if not native_available():
        return None
    path = Path(pack_path)
    if manifest is None:
        manifest = load_pack_manifest(path)
    try:
        native = require_native()
        return native.NativePackReader.open(str(path), scpack_to_native_manifest_json(manifest))
    except Exception as exc:  # pragma: no cover - surface as StorageError
        raise StorageError(f"Failed to open native pack reader for {path}: {exc}") from exc


def open_native_streaming_store(
    pack_path: Path | str,
    manifest: dict[str, Any] | None = None,
    *,
    capacity_bytes: int,
) -> Any | None:
    """Open NativeStreamingStore when the extension is loaded; else None."""
    if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int) or capacity_bytes < 1:
        raise StorageError("Native streaming capacity_bytes must be an integer >= 1")
    if not native_available():
        return None
    path = Path(pack_path)
    if manifest is None:
        manifest = load_pack_manifest(path)
    try:
        native = require_native()
        return native.NativeStreamingStore.open(
            str(path),
            scpack_to_native_manifest_json(manifest),
            capacity_bytes,
        )
    except Exception as exc:  # pragma: no cover
        raise StorageError(f"Failed to open native streaming store for {path}: {exc}") from exc
