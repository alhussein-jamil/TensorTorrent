"""Optional quantized weight packing (explicit user mode).

Enabled only when ``CompileConfig.allow_quantized_storage`` is true and
``numerical_mode`` is ``quantized``. Packs float32 weights as int8 with a
per-tensor scale; dequantizes on load. Not a drop-in for exact numerics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from streamcompiler.errors import StorageError, UnsupportedFeatureError


@dataclass
class QuantizedTensor:
    qdata: torch.Tensor  # int8
    scale: float
    zero_point: int
    shape: tuple[int, ...]
    dtype: str = "float32"

    def dequantize(self) -> torch.Tensor:
        return (self.qdata.float() - float(self.zero_point)) * float(self.scale)


def quantize_per_tensor(tensor: torch.Tensor) -> QuantizedTensor:
    if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise UnsupportedFeatureError(f"Cannot quantize dtype {tensor.dtype}")
    host = tensor.detach().float().contiguous().cpu()
    max_abs = float(host.abs().max().item()) if host.numel() else 1.0
    scale = max(max_abs / 127.0, 1e-8)
    q = torch.clamp(torch.round(host / scale), -128, 127).to(torch.int8)
    return QuantizedTensor(qdata=q, scale=scale, zero_point=0, shape=tuple(host.shape), dtype="float32")


def pack_quantized_state_dict(state: dict[str, torch.Tensor], path: Path) -> dict[str, Any]:
    """Write a torch-saved dict of QuantizedTensor payloads."""
    payload: dict[str, Any] = {"format": "streamcompiler_q8_v1", "tensors": {}}
    for name, tensor in state.items():
        q = quantize_per_tensor(tensor)
        payload["tensors"][name] = {
            "qdata": q.qdata,
            "scale": q.scale,
            "zero_point": q.zero_point,
            "shape": list(q.shape),
            "dtype": q.dtype,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"path": str(path), "count": len(state), "format": "streamcompiler_q8_v1"}


def load_quantized_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "streamcompiler_q8_v1":
        raise StorageError(f"{path} is not a streamcompiler_q8_v1 pack")
    out: dict[str, torch.Tensor] = {}
    for name, entry in payload["tensors"].items():
        q = QuantizedTensor(
            qdata=entry["qdata"],
            scale=float(entry["scale"]),
            zero_point=int(entry["zero_point"]),
            shape=tuple(entry["shape"]),
            dtype=str(entry.get("dtype", "float32")),
        )
        out[name] = q.dequantize()
    return out
