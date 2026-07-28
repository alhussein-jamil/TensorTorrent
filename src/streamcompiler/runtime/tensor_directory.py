"""Central tensor residency directory.

Tracks every logical tensor the runtime knows about: valid copies, memory
location, version, size, dtype, layout, last use, and active consumers.
Mutations bump the version and invalidate stale copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import Event, RLock
from typing import Any

import torch

from streamcompiler.errors import RuntimePlanError
from streamcompiler.runtime.schedule import MemoryTier


class TensorState(str, Enum):
    ON_DISK = "on_disk"
    IN_RAM = "in_ram"
    IN_PINNED_RAM = "in_pinned_ram"
    ON_DEVICE = "on_device"
    TRANSFERRING = "transferring"
    COMPUTING = "computing"
    RELEASED = "released"


@dataclass
class PendingTransfer:
    """Tracks one in-flight transfer so concurrent consumers join instead of duplicating it."""

    event: Event = field(default_factory=Event)
    result_value: Any = None


@dataclass
class TensorCopy:
    location: str
    tier: MemoryTier
    nbytes: int
    device: str | None = None


@dataclass
class TensorRecord:
    tensor_id: str
    size_bytes: int
    dtype: str
    layout: str = "contiguous"
    shape: tuple[int, ...] = ()
    version: int = 0
    state: TensorState = TensorState.RELEASED
    valid_copies: list[TensorCopy] = field(default_factory=list)
    last_use_region: str | None = None
    active_consumers: int = 0
    alias_group: str | None = None
    storage_id: str | None = None
    mutable: bool = False
    pending_transfers: dict[str, PendingTransfer] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def has_valid_copy(self) -> bool:
        return bool(self.valid_copies) and self.state != TensorState.RELEASED


class TensorDirectory:
    """Authoritative residency map for logical tensors."""

    def __init__(self) -> None:
        self._records: dict[str, TensorRecord] = {}
        self._lock = RLock()
        self._events: list[dict[str, Any]] = []

    def get(self, tensor_id: str) -> TensorRecord | None:
        with self._lock:
            return self._records.get(tensor_id)

    def ensure(
        self,
        tensor_id: str,
        *,
        size_bytes: int = 0,
        dtype: str = "float32",
        layout: str = "contiguous",
        shape: tuple[int, ...] = (),
        alias_group: str | None = None,
        storage_id: str | None = None,
        mutable: bool = False,
    ) -> TensorRecord:
        with self._lock:
            record = self._records.get(tensor_id)
            if record is None:
                record = TensorRecord(
                    tensor_id=tensor_id,
                    size_bytes=size_bytes,
                    dtype=dtype,
                    layout=layout,
                    shape=shape,
                    alias_group=alias_group,
                    storage_id=storage_id,
                    mutable=mutable,
                )
                self._records[tensor_id] = record
            else:
                if size_bytes:
                    record.size_bytes = size_bytes
                if dtype:
                    record.dtype = dtype
                if shape:
                    record.shape = shape
                if alias_group is not None:
                    record.alias_group = alias_group
                if storage_id is not None:
                    record.storage_id = storage_id
                record.mutable = mutable or record.mutable
            return record

    def materialize(
        self,
        tensor_id: str,
        *,
        location: str,
        tier: MemoryTier,
        nbytes: int,
        device: str | None = None,
        value: Any = None,
    ) -> TensorRecord:
        with self._lock:
            record = self.ensure(tensor_id, size_bytes=nbytes)
            if value is not None and isinstance(value, torch.Tensor):
                record.size_bytes = int(value.numel() * value.element_size())
                record.dtype = str(value.dtype).removeprefix("torch.")
                record.shape = tuple(int(d) for d in value.shape)
            copy = TensorCopy(location=location, tier=tier, nbytes=nbytes or record.size_bytes, device=device)
            # Replace any copy at the same location; keep others until invalidated.
            record.valid_copies = [c for c in record.valid_copies if c.location != location]
            record.valid_copies.append(copy)
            record.state = _state_for_tier(tier)
            self._events.append(
                {
                    "event": "materialize",
                    "tensor_id": tensor_id,
                    "location": location,
                    "tier": tier.value,
                    "nbytes": copy.nbytes,
                    "version": record.version,
                }
            )
            return record

    def begin_transfer(self, tensor_id: str, destination: str | None = None) -> PendingTransfer | None:
        """Register the start of a transfer.

        When ``destination`` names a concrete resident copy and a transfer to that
        same destination is already in flight, returns the existing
        :class:`PendingTransfer` so the caller can join it (wait, then reuse
        ``result_value``) instead of performing a duplicate transfer. Returns
        ``None`` when the caller is the one that must actually do the work.
        """
        with self._lock:
            record = self.ensure(tensor_id)
            if record.state == TensorState.RELEASED and not record.valid_copies:
                raise RuntimePlanError(f"Cannot transfer released tensor {tensor_id}")
            if destination is not None:
                existing = record.pending_transfers.get(destination)
                if existing is not None:
                    return existing
                record.pending_transfers[destination] = PendingTransfer()
            record.state = TensorState.TRANSFERRING
            self._events.append({"event": "transfer_start", "tensor_id": tensor_id, "version": record.version})
            return None

    def complete_transfer(
        self,
        tensor_id: str,
        *,
        location: str,
        tier: MemoryTier,
        nbytes: int,
        device: str | None = None,
        invalidate_source: bool = False,
        source_location: str | None = None,
        value: Any = None,
    ) -> TensorRecord:
        with self._lock:
            record = self.ensure(tensor_id, size_bytes=nbytes)
            if invalidate_source and source_location is not None:
                record.valid_copies = [c for c in record.valid_copies if c.location != source_location]
            record.valid_copies = [c for c in record.valid_copies if c.location != location]
            record.valid_copies.append(
                TensorCopy(location=location, tier=tier, nbytes=nbytes or record.size_bytes, device=device)
            )
            record.state = _state_for_tier(tier)
            pending = record.pending_transfers.pop(location, None)
            if pending is not None:
                pending.result_value = value
                pending.event.set()
            self._events.append(
                {
                    "event": "transfer_end",
                    "tensor_id": tensor_id,
                    "location": location,
                    "tier": tier.value,
                    "nbytes": nbytes,
                    "version": record.version,
                }
            )
            return record

    def begin_compute(self, tensor_id: str) -> None:
        with self._lock:
            record = self.ensure(tensor_id)
            record.state = TensorState.COMPUTING

    def mark_produced(
        self,
        tensor_id: str,
        *,
        location: str,
        tier: MemoryTier,
        value: Any = None,
        device: str | None = None,
    ) -> TensorRecord:
        with self._lock:
            nbytes = 0
            if isinstance(value, torch.Tensor):
                nbytes = int(value.numel() * value.element_size())
            record = self.materialize(
                tensor_id, location=location, tier=tier, nbytes=nbytes, device=device, value=value
            )
            self._events.append({"event": "allocate", "tensor_id": tensor_id, "nbytes": record.size_bytes})
            return record

    def add_consumer(self, tensor_id: str) -> None:
        with self._lock:
            record = self.ensure(tensor_id)
            record.active_consumers += 1

    def finish_consumer(self, tensor_id: str, *, region_id: str | None = None) -> int:
        """Decrement consumer count. Returns remaining consumers."""
        with self._lock:
            record = self.ensure(tensor_id)
            record.active_consumers = max(0, record.active_consumers - 1)
            if region_id is not None:
                record.last_use_region = region_id
            return record.active_consumers

    def release(self, tensor_id: str, *, force: bool = False) -> bool:
        """Release tensor storage when no consumers remain (or ``force``)."""
        with self._lock:
            record = self._records.get(tensor_id)
            if record is None:
                return False
            if not force and record.active_consumers > 0:
                return False
            record.valid_copies.clear()
            record.state = TensorState.RELEASED
            record.active_consumers = 0
            self._events.append({"event": "release", "tensor_id": tensor_id, "version": record.version})
            return True

    def mutate(self, tensor_id: str) -> None:
        """Invalidate all copies after an in-place mutation; bump version."""
        with self._lock:
            record = self.ensure(tensor_id)
            if not record.mutable:
                raise RuntimePlanError(
                    f"Mutation of immutable tensor {tensor_id} rejected; "
                    "mark mutable=True or avoid in-place ops on this value"
                )
            record.version += 1
            # Keep one canonical location if present; mark others invalid.
            if record.valid_copies:
                keep = record.valid_copies[0]
                record.valid_copies = [keep]
            self._events.append({"event": "invalidate", "tensor_id": tensor_id, "version": record.version})

    def has_copy_at(self, tensor_id: str, location: str) -> bool:
        with self._lock:
            record = self._records.get(tensor_id)
            if record is None or record.state == TensorState.RELEASED:
                return False
            return any(c.location == location for c in record.valid_copies)

    def locations(self, tensor_id: str) -> tuple[str, ...]:
        with self._lock:
            record = self._records.get(tensor_id)
            if record is None:
                return ()
            return tuple(c.location for c in record.valid_copies)

    def drain_events(self) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                tid: {
                    "size_bytes": r.size_bytes,
                    "dtype": r.dtype,
                    "layout": r.layout,
                    "shape": list(r.shape),
                    "version": r.version,
                    "state": r.state.value,
                    "locations": [c.location for c in r.valid_copies],
                    "tiers": [c.tier.value for c in r.valid_copies],
                    "active_consumers": r.active_consumers,
                    "last_use_region": r.last_use_region,
                    "alias_group": r.alias_group,
                    "storage_id": r.storage_id,
                    "mutable": r.mutable,
                }
                for tid, r in self._records.items()
            }


def _state_for_tier(tier: MemoryTier) -> TensorState:
    if tier == MemoryTier.DISK:
        return TensorState.ON_DISK
    if tier == MemoryTier.PINNED_RAM:
        return TensorState.IN_PINNED_RAM
    if tier == MemoryTier.DEVICE:
        return TensorState.ON_DEVICE
    if tier == MemoryTier.SYSTEM_RAM:
        return TensorState.IN_RAM
    return TensorState.IN_RAM
