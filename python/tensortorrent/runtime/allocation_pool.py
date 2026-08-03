"""Physical allocation pool backing buffer-reuse decisions.

``runtime.buffer_reuse.plan_buffer_reuse`` decides, from liveness, which
non-overlapping logical activations may share one slot. That is a planning
decision over tensor ids; it says nothing about physical memory. This module
gives each slot one real byte buffer on the same device as the producer
tensor, so two non-overlapping tensors placed in the same slot provably share
one allocation (equal ``data_ptr()``), and overlapping tensors are refused a
shared slot.

``GraphExecutor`` acquires into these slots when a ``buffer_reuse`` assignment
is supplied at construction, and releases the slot when the schedule's
corresponding ``Release`` fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from tensortorrent.errors import RuntimePlanError


@dataclass
class AllocationRecord:
    """Bookkeeping for one physical slot, per the residency-tracking spec."""

    allocation_id: int
    capacity_bytes: int = 0
    current_tensor_id: str | None = None
    reuse_history: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    device: str | None = None


class ActivationAllocator:
    """One growable physical buffer per buffer-reuse slot.

    Reuse only happens when the caller explicitly acquires the same
    ``slot_id`` again; it is the caller's responsibility (normally
    ``BufferReusePlan``, which is liveness-safe) to never assign two
    simultaneously-live tensors to the same slot.

    Buffers follow ``like.device`` so CUDA (or other accelerator) activations
    are not silently staged onto host RAM.
    """

    def __init__(self) -> None:
        self._buffers: dict[int, torch.Tensor] = {}
        self._records: dict[int, AllocationRecord] = {}

    def acquire(self, slot_id: int, tensor_id: str, like: torch.Tensor) -> torch.Tensor:
        """Return a tensor view over the slot's physical storage holding ``like``'s data."""
        nbytes = like.numel() * like.element_size()
        device = like.device
        device_key = str(device)
        record = self._records.get(slot_id)
        if record is None:
            record = AllocationRecord(allocation_id=slot_id, device=device_key)
            self._records[slot_id] = record

        buf = self._buffers.get(slot_id)
        needs_alloc = buf is None or buf.numel() < nbytes or buf.device != device or record.device != device_key
        if needs_alloc:
            buf = torch.empty(max(nbytes, 1), dtype=torch.uint8, device=device)
            self._buffers[slot_id] = buf
            record.capacity_bytes = buf.numel()
            record.device = device_key
            record.events.append(
                {
                    "event": "allocate",
                    "slot_id": slot_id,
                    "tensor_id": tensor_id,
                    "capacity_bytes": buf.numel(),
                    "device": device_key,
                }
            )
        else:
            record.events.append({"event": "reuse", "slot_id": slot_id, "tensor_id": tensor_id})

        assert buf is not None
        record.reuse_history.append(tensor_id)
        record.current_tensor_id = tensor_id
        view = buf[:nbytes].view(like.dtype).view(like.shape)
        view.copy_(like)
        return view

    def release(self, slot_id: int) -> None:
        record = self._records.get(slot_id)
        if record is None:
            return
        record.events.append({"event": "release", "slot_id": slot_id, "tensor_id": record.current_tensor_id})
        record.current_tensor_id = None

    def storage_ptr(self, slot_id: int) -> int:
        buf = self._buffers.get(slot_id)
        if buf is None:
            raise RuntimePlanError(f"No physical allocation exists yet for slot {slot_id}")
        return buf.data_ptr()

    def snapshot(self) -> dict[int, AllocationRecord]:
        return dict(self._records)
