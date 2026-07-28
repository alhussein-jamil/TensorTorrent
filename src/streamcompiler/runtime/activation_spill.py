"""Disk spill / reload helpers for schedule-driven activation Evict/Load ops.

The planner emits explicit ``Evict`` (kind=activation_spill) and ``Load``
(kind=activation_reload) instructions. The runtime executes those ops via
:func:`spill_tensor` / :func:`reload_spilled`. No transparent spill outside
the schedule. This is real I/O, not a stub.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from streamcompiler.errors import RuntimePlanError


@dataclass
class SpilledTensor:
    """Host tensor relocated to a private temp file until a consumer needs it."""

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


def spill_tensor(tensor: torch.Tensor, *, directory: Path | None = None) -> SpilledTensor:
    """Write ``tensor`` to a new temp file and return a spill handle."""
    if not isinstance(tensor, torch.Tensor):
        raise RuntimePlanError(f"Cannot spill non-tensor value: {type(tensor)!r}")
    host = tensor.detach().contiguous().cpu()
    nbytes = int(host.numel() * host.element_size())
    dir_path = Path(directory) if directory is not None else Path(tempfile.gettempdir())
    dir_path.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="sc_act_", suffix=".pt", dir=str(dir_path))
    path = Path(name)
    try:
        # Close the fd first so torch.save can open the path on all platforms.
        import os

        os.close(fd)
        torch.save(host, path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return SpilledTensor(path=path, shape=tuple(host.shape), dtype=host.dtype, nbytes=nbytes)


def reload_spilled(spilled: SpilledTensor, *, delete: bool = True) -> torch.Tensor:
    """Read a spilled tensor back into a new host tensor."""
    tensor = torch.load(spilled.path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise RuntimePlanError(f"Spilled file {spilled.path} did not contain a tensor")
    if tuple(tensor.shape) != spilled.shape or tensor.dtype != spilled.dtype:
        raise RuntimePlanError(
            f"Spilled activation {spilled.path} meta mismatch: "
            f"got shape={tuple(tensor.shape)} dtype={tensor.dtype}, "
            f"expected shape={spilled.shape} dtype={spilled.dtype}"
        )
    if delete:
        spilled.path.unlink(missing_ok=True)
    return tensor


def is_spilled(value: Any) -> bool:
    return isinstance(value, SpilledTensor)
