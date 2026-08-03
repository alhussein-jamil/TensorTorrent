"""Optional quantized weight packing.

When ``CompileConfig.allow_quantized_storage`` is true and ``numerical_mode`` is
``quantized``, :func:`tensortorrent.storage.pack.pack_state_dict` stores float
weights as int8 with an affine scale (``compression=int8_affine``). The streaming
parameter store dequantizes on ``pread``. Standalone helpers below remain for
safe direct ``torch.save`` round-trips.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tensortorrent.errors import StorageError, UnsupportedFeatureError

_ALLOWED_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class QuantizedTensor:
    qdata: torch.Tensor  # int8
    scale: float
    zero_point: int
    shape: tuple[int, ...]
    dtype: str = "float32"

    def dequantize(self) -> torch.Tensor:
        target_dtype = _ALLOWED_DTYPES.get(self.dtype)
        if target_dtype is None:
            raise StorageError(f"Unsupported quantized logical dtype {self.dtype!r}")
        value = (self.qdata.float() - float(self.zero_point)) * float(self.scale)
        return value.reshape(self.shape).to(target_dtype)


def quantize_per_tensor(tensor: torch.Tensor) -> QuantizedTensor:
    if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise UnsupportedFeatureError(f"Cannot quantize dtype {tensor.dtype}")
    host = tensor.detach().float().contiguous().cpu()
    if not bool(torch.isfinite(host).all()):
        raise StorageError("Quantized storage rejects NaN or infinity values")
    max_abs = float(host.abs().max().item()) if host.numel() else 0.0
    scale = max(max_abs / 127.0, 1e-8)
    q = torch.clamp(torch.round(host / scale), -127, 127).to(torch.int8)
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    return QuantizedTensor(
        qdata=q,
        scale=scale,
        zero_point=0,
        shape=tuple(host.shape),
        dtype=dtype_name,
    )


def pack_quantized_state_dict(state: dict[str, torch.Tensor], path: Path) -> dict[str, Any]:
    """Atomically write a weights-only-compatible quantized state dictionary."""
    payload: dict[str, Any] = {"format": "tensortorrent_q8_v1", "tensors": {}}
    for name, tensor in state.items():
        if not isinstance(name, str) or not name:
            raise StorageError("Quantized tensor names must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise StorageError(f"Quantized state entry {name!r} is not a tensor")
        q = quantize_per_tensor(tensor)
        payload["tensors"][name] = {
            "qdata": q.qdata,
            "scale": q.scale,
            "zero_point": q.zero_point,
            "shape": list(q.shape),
            "dtype": q.dtype,
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        torch.save(payload, tmp)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            parent_fd = os.open(path.parent, flags)
        except OSError:
            parent_fd = -1
        if parent_fd >= 0:
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    finally:
        tmp.unlink(missing_ok=True)
    return {"path": str(path), "count": len(state), "format": "tensortorrent_q8_v1"}


def load_quantized_state_dict(path: Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.is_symlink():
        raise StorageError(f"Refusing to load quantized state via symlink: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - convert serialization boundary to StorageError
        raise StorageError(f"Unable to load quantized state {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != "tensortorrent_q8_v1":
        raise StorageError(f"{path} is not a tensortorrent_q8_v1 pack")
    tensors = payload.get("tensors")
    if not isinstance(tensors, dict):
        raise StorageError(f"{path} has an invalid tensor table")
    out: dict[str, torch.Tensor] = {}
    for name, entry in tensors.items():
        if not isinstance(name, str) or not name or not isinstance(entry, dict):
            raise StorageError(f"{path} contains an invalid quantized tensor entry")
        qdata = entry.get("qdata")
        if not isinstance(qdata, torch.Tensor) or qdata.dtype != torch.int8:
            raise StorageError(f"Quantized tensor {name!r} must contain int8 qdata")
        if qdata.device.type != "cpu":
            qdata = qdata.cpu()
        try:
            scale = float(entry["scale"])
            zero_point = int(entry["zero_point"])
            shape = tuple(int(dim) for dim in entry["shape"])
            dtype = str(entry.get("dtype", "float32"))
        except (KeyError, TypeError, ValueError) as exc:
            raise StorageError(f"Invalid metadata for quantized tensor {name!r}: {exc}") from exc
        if not math.isfinite(scale) or scale <= 0:
            raise StorageError(f"Invalid scale for quantized tensor {name!r}: {scale}")
        if any(dim < 0 for dim in shape):
            raise StorageError(f"Invalid shape for quantized tensor {name!r}: {shape}")
        expected_numel = math.prod(shape)
        if int(qdata.numel()) != expected_numel:
            raise StorageError(
                f"Quantized tensor {name!r} shape {shape} expects {expected_numel} values, got {int(qdata.numel())}"
            )
        q = QuantizedTensor(
            qdata=qdata,
            scale=scale,
            zero_point=zero_point,
            shape=shape,
            dtype=dtype,
        )
        out[name] = q.dequantize()
    return out
