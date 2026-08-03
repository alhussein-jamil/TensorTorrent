"""Host tensor ↔ raw byte views for packs, spill, and virtual buffers.

NumPy cannot represent every ``torch.dtype`` (notably ``bfloat16``). All
storage paths that need a contiguous byte payload go through this module so
pack / spill / virtual-device code share one conversion contract.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from tensortorrent.errors import RuntimePlanError


def tensor_as_memoryview(tensor: torch.Tensor) -> memoryview:
    """Return a ``uint8`` memoryview over the contiguous CPU storage of ``tensor``."""
    if not isinstance(tensor, torch.Tensor):
        raise RuntimePlanError(f"Expected torch.Tensor, got {type(tensor)!r}")
    host = tensor.detach().cpu().contiguous()
    # Prefer a same-width integer view for dtypes NumPy cannot host.
    if host.dtype == torch.bfloat16:
        arr: Any = host.view(torch.uint16).numpy()
        return memoryview(cast(Any, arr)).cast("B")
    try:
        arr = host.numpy()
        return memoryview(cast(Any, arr)).cast("B")
    except TypeError:
        arr = host.view(torch.uint8).numpy()
        return memoryview(cast(Any, arr)).cast("B")


def tensor_as_bytes(tensor: torch.Tensor) -> bytes:
    """Copy contiguous CPU storage of ``tensor`` into a ``bytes`` object."""
    return bytes(tensor_as_memoryview(tensor))
