"""Disk spill / reload helpers for schedule-driven activation Evict/Load ops.

Rust owns spill file paths, writes, reads, and cleanup via the native spill
format. Python only converts ``torch.Tensor`` ↔ contiguous host bytes.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from streamcompiler.errors import RuntimePlanError
from streamcompiler.native import native_available, require_native


@dataclass
class SpilledTensor:
    """Host tensor relocated to a native spill file until a consumer needs it."""

    path: Path
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "shape": list(self.shape),
            "dtype": str(self.dtype).replace("torch.", ""),
            "nbytes": self.nbytes,
        }


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _parse_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimePlanError(f"Unsupported spill dtype {name!r}")
    return dtype


def tensor_to_spill_bytes(tensor: torch.Tensor) -> tuple[str, list[int], bytes]:
    """Contiguous host bytes + dtype/shape for native spill write."""
    if not isinstance(tensor, torch.Tensor):
        raise RuntimePlanError(f"Cannot spill non-tensor value: {type(tensor)!r}")
    host = tensor.detach().contiguous().cpu()
    dtype = _dtype_name(host.dtype)
    shape = [int(x) for x in host.shape]
    raw = bytes(host.numpy().tobytes())
    return dtype, shape, raw


def spill_bytes_to_tensor(dtype_name: str, shape: list[int], raw: bytes) -> torch.Tensor:
    dtype = _parse_dtype(dtype_name)
    expected = 1
    for d in shape:
        expected *= int(d)
    expected *= int(torch.empty((), dtype=dtype).element_size())
    if len(raw) != expected:
        raise RuntimePlanError(
            f"Spill payload size mismatch: got {len(raw)} expected {expected} for {shape} {dtype_name}"
        )
    # Keep bytearray alive for frombuffer storage (no clone).
    buf = bytearray(raw) if not isinstance(raw, bytearray) else raw
    tensor = torch.frombuffer(buf, dtype=dtype).reshape(tuple(int(x) for x in shape))
    tensor._sc_spill_buf = buf  # type: ignore[attr-defined]
    return tensor


def spill_tensor(tensor: torch.Tensor, *, directory: Path | None = None) -> SpilledTensor:
    """Write ``tensor`` via native spill format and return a spill handle."""
    dtype, shape, raw = tensor_to_spill_bytes(tensor)
    nbytes = len(raw)
    dir_path = Path(directory) if directory is not None else Path(tempfile.gettempdir())
    dir_path.mkdir(parents=True, exist_ok=True)
    if not native_available():
        raise RuntimePlanError("native extension required for activation spill")
    native = require_native()
    path_str = native.write_activation_spill(str(dir_path), dtype, shape, raw)
    return SpilledTensor(
        path=Path(path_str),
        shape=tuple(shape),
        dtype=_parse_dtype(dtype),
        nbytes=nbytes,
    )


def reload_spilled(spilled: SpilledTensor, *, delete: bool = True) -> torch.Tensor:
    """Read a spilled tensor back into a new host tensor."""
    if not native_available():
        raise RuntimePlanError("native extension required for activation reload")
    native = require_native()
    meta = native.read_activation_spill(str(spilled.path))
    dtype_name = str(meta["dtype"])
    shape = [int(x) for x in meta["shape"]]
    raw = bytes(meta["bytes"])
    tensor = spill_bytes_to_tensor(dtype_name, shape, raw)
    if tuple(tensor.shape) != spilled.shape or tensor.dtype != spilled.dtype:
        raise RuntimePlanError(
            f"Spilled activation {spilled.path} meta mismatch: "
            f"got shape={tuple(tensor.shape)} dtype={tensor.dtype}, "
            f"expected shape={spilled.shape} dtype={spilled.dtype}"
        )
    if delete:
        native.remove_activation_spill(str(spilled.path))
    return tensor


def is_spilled(value: Any) -> bool:
    return isinstance(value, SpilledTensor)
