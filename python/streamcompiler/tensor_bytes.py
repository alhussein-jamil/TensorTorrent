"""Host tensor ↔ raw byte views for packs, spill, and virtual buffers.

NumPy cannot represent every ``torch.dtype`` (notably ``bfloat16``). All
storage paths that need a contiguous byte payload go through this module so
pack / spill / virtual-device code share one conversion contract.
"""

from __future__ import annotations

import torch

from streamcompiler.errors import RuntimePlanError


def tensor_as_memoryview(tensor: torch.Tensor) -> memoryview:
    """Return a ``uint8`` memoryview over the contiguous CPU storage of ``tensor``."""
    if not isinstance(tensor, torch.Tensor):
        raise RuntimePlanError(f"Expected torch.Tensor, got {type(tensor)!r}")
    host = tensor.detach().cpu().contiguous()
    # Prefer a same-width integer view for dtypes NumPy cannot host.
    if host.dtype == torch.bfloat16:
        return memoryview(host.view(torch.uint16).numpy()).cast("B")
    try:
        return memoryview(host.numpy()).cast("B")
    except TypeError:
        return memoryview(host.view(torch.uint8).numpy()).cast("B")


def tensor_as_bytes(tensor: torch.Tensor) -> bytes:
    """Copy contiguous CPU storage of ``tensor`` into a ``bytes`` object."""
    return bytes(tensor_as_memoryview(tensor))
