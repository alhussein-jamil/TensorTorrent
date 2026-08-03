"""Python-side passive tensor value bag.

Maps ``(logical_tensor_id, resource_id) → torch.Tensor`` (or virtual handle).
Rust ``ResidencyStore`` is the sole authority for residency, versions, leases,
aliases, allocations, transfers, and lifetime. This module never invents or
repairs residency state.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import torch

from tensortorrent.errors import RuntimePlanError


@dataclass
class ResidentCopy:
    """One Python-side value for a logical tensor on one resource label."""

    tensor_id: str
    resource_id: str
    value: Any
    nbytes: int
    tier: str = "system_ram"
    ready_event: Any | None = None
    # Stored label only — never drives invalidation (Rust owns that).
    authoritative: bool = False
    ownership: str = "runtime"
    allocation_id: str | None = None
    storage_offset: int = 0
    shape: tuple[int, ...] = ()
    stride: tuple[int, ...] = ()

    @property
    def valid(self) -> bool:
        """Presence implies usable; Rust owns stale/version authority."""
        return True

    def wait_ready(self, *, timeout: float | None = None) -> None:
        event = self.ready_event
        if event is None:
            return
        if hasattr(event, "wait"):
            event.wait(timeout=timeout)
            return
        if hasattr(event, "result"):
            event.result(timeout=timeout)


@dataclass
class CopyStore:
    """Passive ``(tensor_id, resource_id) → ResidentCopy`` value bag.

    No version bump, sibling-stale, consumer leases, or AllocationTable authority.
    """

    _copies: dict[tuple[str, str], ResidentCopy] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def put(
        self,
        tensor_id: str,
        resource_id: str,
        value: Any,
        *,
        tier: str = "system_ram",
        authoritative: bool = True,
        ownership: str = "runtime",
        ready_event: Any | None = None,
    ) -> ResidentCopy:
        """Store or replace the Python value for ``(tensor_id, resource_id)``."""
        nbytes = _nbytes(value)
        with self._lock:
            return self._install(
                tensor_id,
                resource_id,
                value,
                nbytes=nbytes,
                tier=tier,
                authoritative=authoritative,
                ownership=ownership,
                ready_event=ready_event,
            )

    def replicate(
        self,
        tensor_id: str,
        resource_id: str,
        value: Any,
        *,
        tier: str = "system_ram",
        ownership: str = "runtime",
        ready_event: Any | None = None,
        source_resource: str | None = None,
    ) -> ResidentCopy:
        """Store an additional resource label for the same logical tensor."""
        del source_resource
        nbytes = _nbytes(value)
        with self._lock:
            return self._install(
                tensor_id,
                resource_id,
                value,
                nbytes=nbytes,
                tier=tier,
                authoritative=False,
                ownership=ownership,
                ready_event=ready_event,
            )

    def alias(self, tensor_id: str, source_resource: str, alias_resource: str) -> ResidentCopy:
        """Register the same Python value under another resource label."""
        with self._lock:
            src = self._copies.get((tensor_id, source_resource))
            if src is None:
                raise RuntimePlanError(f"Cannot alias {tensor_id!r}: no copy on {source_resource!r}")
            return self._install(
                tensor_id,
                alias_resource,
                src.value,
                nbytes=src.nbytes,
                tier=src.tier,
                authoritative=False,
                ownership=src.ownership,
                ready_event=src.ready_event,
            )

    def replace_handle(
        self,
        tensor_id: str,
        resource_id: str,
        value: Any,
        *,
        tier: str | None = None,
        ready_event: Any | None = None,
    ) -> ResidentCopy:
        """Replace the Python value in place (spill/reload)."""
        nbytes = _nbytes(value)
        with self._lock:
            prev = self._copies.get((tensor_id, resource_id))
            if prev is None:
                raise RuntimePlanError(f"Cannot replace missing copy of {tensor_id!r} on {resource_id!r}")
            return self._install(
                tensor_id,
                resource_id,
                value,
                nbytes=nbytes,
                tier=tier if tier is not None else prev.tier,
                authoritative=prev.authoritative,
                ownership=prev.ownership,
                ready_event=ready_event if ready_event is not None else prev.ready_event,
            )

    def _install(
        self,
        tensor_id: str,
        resource_id: str,
        value: Any,
        *,
        nbytes: int,
        tier: str,
        authoritative: bool,
        ownership: str,
        ready_event: Any | None,
    ) -> ResidentCopy:
        key = (tensor_id, resource_id)
        alloc_id = _allocation_id(value, tensor_id, resource_id)
        copy = ResidentCopy(
            tensor_id=tensor_id,
            resource_id=resource_id,
            value=value,
            nbytes=nbytes,
            tier=tier,
            ready_event=ready_event,
            authoritative=authoritative,
            ownership=ownership,
            allocation_id=alloc_id,
            storage_offset=_storage_offset(value),
            shape=_shape(value),
            stride=_stride(value),
        )
        self._copies[key] = copy
        return copy

    def live_bytes(self) -> int:
        with self._lock:
            seen: set[str] = set()
            total = 0
            for (tid, rid), copy in self._copies.items():
                alloc = copy.allocation_id or _allocation_id(copy.value, tid, rid)
                if alloc in seen:
                    continue
                seen.add(alloc)
                total += _physical_capacity_bytes(copy.value, copy.nbytes)
            return total

    def peak_bytes(self) -> int:
        return self.live_bytes()

    def activation_live_bytes(
        self,
        *,
        exclude_tensors: set[str] | None = None,
        exclude_resources: set[str] | None = None,
        ownerships: set[str] | None = None,
    ) -> int:
        """Distinct physical bytes of activation values (aliases counted once)."""
        skip_t = exclude_tensors or set()
        skip_r = exclude_resources or {"disk"}
        own = ownerships if ownerships is not None else {"activation"}
        with self._lock:
            allocations: dict[str, int] = {}
            for (tid, rid), copy in self._copies.items():
                if tid in skip_t or rid in skip_r:
                    continue
                if copy.ownership not in own:
                    continue
                alloc_id = copy.allocation_id or _allocation_id(copy.value, tid, rid)
                capacity = _physical_capacity_bytes(copy.value, copy.nbytes)
                allocations[alloc_id] = max(allocations.get(alloc_id, 0), capacity)
            return int(sum(allocations.values()))

    def activation_tensor_ids(self, *, exclude_resources: set[str] | None = None) -> set[str]:
        skip_r = exclude_resources or {"disk"}
        with self._lock:
            out: set[str] = set()
            for (tid, rid), copy in self._copies.items():
                if rid in skip_r or copy.ownership != "activation":
                    continue
                out.add(tid)
            return out

    def get(self, tensor_id: str, resource_id: str) -> ResidentCopy:
        with self._lock:
            copy = self._copies.get((tensor_id, resource_id))
            if copy is None:
                raise RuntimePlanError(f"No resident copy of {tensor_id!r} on resource {resource_id!r}")
            return copy

    def require(self, tensor_id: str, resource_id: str) -> ResidentCopy:
        """Exact planned value: must exist (strict — no silent invent)."""
        with self._lock:
            copy = self._copies.get((tensor_id, resource_id))
            if copy is None:
                raise RuntimePlanError(
                    f"Required copy missing: tensor={tensor_id!r} resource={resource_id!r} "
                    f"(schedule error — no silent fallback)"
                )
        copy.wait_ready()
        with self._lock:
            again = self._copies.get((tensor_id, resource_id))
            if again is None:
                raise RuntimePlanError(f"Required copy vanished during wait: {tensor_id!r}@{resource_id!r}")
            return again

    def try_get(self, tensor_id: str, resource_id: str) -> ResidentCopy | None:
        with self._lock:
            return self._copies.get((tensor_id, resource_id))

    def has(self, tensor_id: str, resource_id: str, *, valid_only: bool = False) -> bool:
        del valid_only  # presence implies usable; Rust owns validity
        with self._lock:
            return (tensor_id, resource_id) in self._copies

    def resources_for(self, tensor_id: str, *, valid_only: bool = False) -> tuple[str, ...]:
        del valid_only
        with self._lock:
            return tuple(rid for (tid, rid) in self._copies if tid == tensor_id)

    def mark_ready(self, tensor_id: str, resource_id: str, event: Any | None = None) -> None:
        with self._lock:
            copy = self._copies.get((tensor_id, resource_id))
            if copy is None:
                raise RuntimePlanError(f"mark_ready: no copy of {tensor_id!r} on {resource_id!r}")
            copy.ready_event = event

    def require_unlocked(self, tensor_id: str, resource_id: str) -> ResidentCopy:
        copy = self._copies.get((tensor_id, resource_id))
        if copy is None:
            raise RuntimePlanError(f"Required copy missing: tensor={tensor_id!r} resource={resource_id!r}")
        return copy

    def drop(self, tensor_id: str, resource_id: str) -> int:
        """Drop the exact ``(tensor_id, resource_id)`` value. Never drops siblings."""
        with self._lock:
            key = (tensor_id, resource_id)
            if key not in self._copies:
                return 0
            copy = self._copies.pop(key)
            return copy.nbytes

    def move(
        self,
        tensor_id: str,
        source_resource: str,
        dest_resource: str,
        value: Any,
        *,
        tier: str,
    ) -> ResidentCopy:
        """Relocate a value label without inventing residency."""
        with self._lock:
            src = self._copies.get((tensor_id, source_resource))
            if src is None:
                raise RuntimePlanError(f"Cannot move {tensor_id!r}: source {source_resource!r} missing")
            authoritative = src.authoritative
            ownership = src.ownership
        self.drop(tensor_id, source_resource)
        with self._lock:
            return self._install(
                tensor_id,
                dest_resource,
                value,
                nbytes=_nbytes(value),
                tier=tier,
                authoritative=authoritative,
                ownership=ownership,
                ready_event=None,
            )

    def snapshot(self) -> dict[str, Any]:
        """Handle/value inventory only — not residency authority."""
        with self._lock:
            return {
                f"{tid}@{rid}": {
                    "nbytes": c.nbytes,
                    "tier": c.tier,
                    "authoritative": c.authoritative,
                    "ownership": c.ownership,
                    "allocation_id": c.allocation_id,
                    "storage_offset": c.storage_offset,
                    "shape": list(c.shape),
                    "stride": list(c.stride),
                }
                for (tid, rid), c in self._copies.items()
            }


def _nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    return 0


def _storage_offset(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        try:
            return int(value.storage_offset())
        except Exception:  # noqa: BLE001
            return 0
    return 0


def _shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return tuple(int(x) for x in value.shape)
    return ()


def _stride(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return tuple(int(x) for x in value.stride())
    return ()


def _physical_capacity_bytes(value: Any, logical_nbytes: int) -> int:
    if isinstance(value, torch.Tensor):
        try:
            return int(value.untyped_storage().nbytes())
        except Exception:  # noqa: BLE001
            pass
    capacity = getattr(value, "allocation_nbytes", None)
    if isinstance(capacity, int) and capacity >= 0:
        return capacity
    return max(0, int(logical_nbytes))


def _allocation_id(value: Any, tensor_id: str, resource_id: str) -> str:
    """Identity of the backing physical allocation for alias byte counting."""
    del resource_id
    if value is None:
        return f"null::{tensor_id}"
    if isinstance(value, torch.Tensor):
        try:
            storage = value.untyped_storage()
            ptr = int(storage.data_ptr())
            nbytes = int(storage.nbytes())
            device = str(value.device)
            return f"torch::{device}::{ptr}::{nbytes}"
        except Exception:  # noqa: BLE001
            pass
    explicit = getattr(value, "allocation_key", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    return f"phys::{type(value).__name__}::{id(value)}"
