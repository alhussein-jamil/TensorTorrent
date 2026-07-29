"""Native pack reader bridge for SCPACK1 production loads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from streamcompiler.errors import StorageError
from streamcompiler.native import native_available, require_native
from streamcompiler.storage.pack import load_pack_manifest


def scpack_to_native_manifest_json(manifest: dict[str, Any]) -> str:
    """Adapt SCPACK1 manifest fields to the Rust PackManifest schema."""
    tensors = []
    for entry in manifest.get("tensors") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("logical_id") or entry.get("name") or "")
        if not name:
            continue
        tensors.append(
            {
                "name": name,
                "offset": int(entry["offset"]),
                "length": int(entry.get("nbytes") or entry.get("length") or 0),
                "dtype": str(entry.get("stored_dtype") or entry.get("dtype") or "u8"),
                "shape": [int(x) for x in (entry.get("stored_shape") or entry.get("shape") or [])],
                "checksum_crc32": None,
            }
        )
    return json.dumps({"version": int(manifest.get("version") or 1), "tensors": tensors})


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
            int(capacity_bytes),
        )
    except Exception as exc:  # pragma: no cover
        raise StorageError(f"Failed to open native streaming store for {path}: {exc}") from exc
