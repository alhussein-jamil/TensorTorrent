"""Physical buffers for buffer-reuse slots.

``plan_buffer_reuse`` picks which non-overlapping activations share a slot
(logical ids only). This module owns the real byte buffer per slot so two
tensors in the same slot share one ``data_ptr()``, and overlapping live
tensors never get the same slot.

``GraphExecutor`` acquires on compute and releases when the schedule's
``Release`` fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from tensortorrent.errors import RuntimePlanError


@dataclass
class AllocationRecord:
    """One physical slot's bookkeeping."""

    allocation_id: int
    capacity_bytes: int = 0
    current_tensor_id: str | None = None
    reuse_history: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    device: str | None = None


class ActivationAllocator:
    """Growable physical buffer per reuse slot.

    Reuse only when the same ``slot_id`` is acquired again — caller
    (normally ``BufferReusePlan``) must not put two live tensors in one slot.

    Buffers follow ``like.device`` so accelerator activations don't silently
    land on host RAM.
    """

    def __init__(self) -> None:
        self._buffers: dict[int, torch.Tensor] = {}
        self._records: dict[int, AllocationRecord] = {}

    def acquire(self, slot_id: int, tensor_id: str, like: torch.Tensor) -> torch.Tensor:
        """View over the slot buffer holding ``like``'s data."""
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
