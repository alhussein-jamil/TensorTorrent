"""Python-side multi-copy tensor value bag.

Keyed by ``(logical_tensor_id, resource_id)``. One logical tensor may hold
simultaneous valid copies on several resources. Replication (transfer) does not
change the logical version; mutation does, and marks sibling copies stale.

On the native execution path, Rust ``ResidencyStore`` owns valid/lease/release
authority; this store holds the Python tensor objects those opaque handles map to.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import torch

from streamcompiler.errors import RuntimePlanError


@dataclass
class ResidentCopy:
    """One physical residency of a logical tensor on one resource."""

    tensor_id: str
    resource_id: str
    value: Any
    nbytes: int
    tier: str = "system_ram"
    version: int = 0
    ready_event: Any | None = None
    active_consumers: int = 0
    authoritative: bool = False
    stale: bool = False
    ownership: str = "runtime"
    allocation_id: str | None = None
    storage_offset: int = 0
    shape: tuple[int, ...] = ()
    stride: tuple[int, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.stale

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
    """Map ``(tensor_id, resource_id) -> ResidentCopy`` — Python tensor value bag.

    On the native path, Rust residency metadata is authoritative; this bag stores
    values only (``value_bag_only=True`` skips version bump / sibling stale marks /
    Python AllocationTable). Physical memory accounting otherwise delegates to
    :class:`AllocationTable` so aliases that share one handle are counted once.
    """

    _copies: dict[tuple[str, str], ResidentCopy] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _versions: dict[str, int] = field(default_factory=dict)
    _alloc_by_copy: dict[tuple[str, str], str] = field(default_factory=dict)
    _allocations: Any | None = None
    value_bag_only: bool = False

    def bind_allocations(self, allocations: Any) -> None:
        """Attach the call's AllocationTable (must be called before put/replicate)."""
        self._allocations = allocations

    def logical_version(self, tensor_id: str) -> int:
        with self._lock:
            return int(self._versions.get(tensor_id, 0))

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
        """Materialize or mutate the logical tensor on ``resource_id``.

        Increments the logical version and marks every other resource copy stale.
        Use :meth:`replicate` for immutable cross-resource copies.
        """
        nbytes = _nbytes(value)
        with self._lock:
            if self.value_bag_only:
                version = self._versions.get(tensor_id, 0) or 1
                self._versions[tensor_id] = version
            else:
                version = self._versions.get(tensor_id, 0) + 1
                self._versions[tensor_id] = version
                for (tid, rid), existing in list(self._copies.items()):
                    if tid == tensor_id and rid != resource_id:
                        existing.stale = True
                        existing.authoritative = False
            return self._install(
                tensor_id,
                resource_id,
                value,
                nbytes=nbytes,
                tier=tier,
                version=version,
                authoritative=authoritative,
                ownership=ownership,
                ready_event=ready_event,
                stale=False,
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
        """Create an immutable copy on another resource without bumping version."""
        nbytes = _nbytes(value)
        with self._lock:
            if self.value_bag_only:
                if tensor_id not in self._versions:
                    self._versions[tensor_id] = 1
                version = self._versions[tensor_id]
                return self._install(
                    tensor_id,
                    resource_id,
                    value,
                    nbytes=nbytes,
                    tier=tier,
                    version=version,
                    authoritative=False,
                    ownership=ownership,
                    ready_event=ready_event,
                    stale=False,
                )
            if tensor_id not in self._versions:
                if source_resource is not None:
                    src = self._copies.get((tensor_id, source_resource))
                    if src is None or src.stale:
                        raise RuntimePlanError(
                            f"Cannot replicate {tensor_id!r}: source {source_resource!r} missing or stale"
                        )
                    self._versions[tensor_id] = src.version
                else:
                    self._versions[tensor_id] = 1
            version = self._versions[tensor_id]
            return self._install(
                tensor_id,
                resource_id,
                value,
                nbytes=nbytes,
                tier=tier,
                version=version,
                authoritative=False,
                ownership=ownership,
                ready_event=ready_event,
                stale=False,
            )

    def alias(self, tensor_id: str, source_resource: str, alias_resource: str) -> ResidentCopy:
        """Register the same physical handle under another resource label (no version bump)."""
        with self._lock:
            src = self._copies.get((tensor_id, source_resource))
            if src is None:
                raise RuntimePlanError(f"Cannot alias {tensor_id!r}: no copy on {source_resource!r}")
            if src.stale:
                raise RuntimePlanError(f"Cannot alias stale copy of {tensor_id!r} on {source_resource!r}")
            return self._install(
                tensor_id,
                alias_resource,
                src.value,
                nbytes=src.nbytes,
                tier=src.tier,
                version=src.version,
                authoritative=False,
                ownership=src.ownership,
                ready_event=src.ready_event,
                stale=False,
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
        """Replace the physical handle in place (spill/reload) without version bump."""
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
                version=prev.version,
                authoritative=prev.authoritative,
                ownership=prev.ownership,
                ready_event=ready_event if ready_event is not None else prev.ready_event,
                stale=prev.stale,
            )

    def _install(
        self,
        tensor_id: str,
        resource_id: str,
        value: Any,
        *,
        nbytes: int,
        tier: str,
        version: int,
        authoritative: bool,
        ownership: str,
        ready_event: Any | None,
        stale: bool,
    ) -> ResidentCopy:
        key = (tensor_id, resource_id)
        prev = self._copies.get(key)
        prev_alloc = self._alloc_by_copy.get(key)
        alloc_id = _allocation_id(value, tensor_id, resource_id)
        physical_capacity = _physical_capacity_bytes(value, nbytes)
        if self._allocations is not None and not self.value_bag_only:
            if prev_alloc is not None and prev_alloc != alloc_id:
                self._allocations.release(prev_alloc)
            if prev_alloc != alloc_id:
                self._allocations.register(
                    alloc_id,
                    resource_id=resource_id,
                    capacity_bytes=max(0, physical_capacity),
                    handle=value,
                )
            self._alloc_by_copy[key] = alloc_id
        copy = ResidentCopy(
            tensor_id=tensor_id,
            resource_id=resource_id,
            value=value,
            nbytes=nbytes,
            tier=tier,
            version=version,
            ready_event=ready_event,
            active_consumers=prev.active_consumers if prev is not None else 0,
            authoritative=authoritative,
            stale=stale,
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
            if self._allocations is not None:
                return int(self._allocations.live_bytes())
            # Fallback when unbound (unit tests constructing CopyStore alone).
            seen: set[str] = set()
            total = 0
            for (tid, rid), copy in self._copies.items():
                alloc = copy.allocation_id or _allocation_id(copy.value, tid, rid)
                if alloc in seen:
                    continue
                seen.add(alloc)
                total += copy.nbytes
            return total

    def peak_bytes(self) -> int:
        with self._lock:
            if self._allocations is not None:
                return int(self._allocations.peak_bytes())
            return self.live_bytes()

    def activation_live_bytes(
        self,
        *,
        exclude_tensors: set[str] | None = None,
        exclude_resources: set[str] | None = None,
        ownerships: set[str] | None = None,
    ) -> int:
        """Physical bytes of resident activation copies (aliases counted once).

        Defaults to ``ownerships={"activation"}`` and excludes disk spill handles.
        """
        skip_t = exclude_tensors or set()
        skip_r = exclude_resources or {"disk"}
        own = ownerships if ownerships is not None else {"activation"}
        with self._lock:
            # Count distinct physical allocations, not logical tensors. A CPU and
            # device copy consume memory independently; aliases/views sharing one
            # backing storage count once.
            allocations: dict[str, int] = {}
            for (tid, rid), copy in self._copies.items():
                if tid in skip_t or rid in skip_r or copy.stale:
                    continue
                if copy.ownership not in own:
                    continue
                alloc_id = copy.allocation_id or _allocation_id(copy.value, tid, rid)
                capacity = _physical_capacity_bytes(copy.value, copy.nbytes)
                if self._allocations is not None:
                    capacity = max(capacity, int(self._allocations.capacity_bytes(alloc_id)))
                allocations[alloc_id] = max(allocations.get(alloc_id, 0), capacity)
            return int(sum(allocations.values()))

    def activation_tensor_ids(self, *, exclude_resources: set[str] | None = None) -> set[str]:
        """Logical tensor ids with a valid non-disk activation residency."""
        skip_r = exclude_resources or {"disk"}
        with self._lock:
            out: set[str] = set()
            for (tid, rid), copy in self._copies.items():
                if rid in skip_r or copy.stale or copy.ownership != "activation":
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
        """Exact planned copy: must exist, be valid, and be ready (or wait)."""
        with self._lock:
            copy = self._copies.get((tensor_id, resource_id))
            if copy is None:
                raise RuntimePlanError(
                    f"Required copy missing: tensor={tensor_id!r} resource={resource_id!r} "
                    f"(schedule error — no silent fallback)"
                )
            if copy.stale:
                raise RuntimePlanError(
                    f"Required copy stale: tensor={tensor_id!r} resource={resource_id!r} "
                    f"version={copy.version} logical={self._versions.get(tensor_id, 0)}"
                )
        copy.wait_ready()
        with self._lock:
            again = self._copies.get((tensor_id, resource_id))
            if again is None:
                raise RuntimePlanError(f"Required copy vanished during wait: {tensor_id!r}@{resource_id!r}")
            if again.stale:
                raise RuntimePlanError(f"Required copy became stale during wait: {tensor_id!r}@{resource_id!r}")
            return again

    def try_get(self, tensor_id: str, resource_id: str) -> ResidentCopy | None:
        with self._lock:
            return self._copies.get((tensor_id, resource_id))

    def has(self, tensor_id: str, resource_id: str, *, valid_only: bool = False) -> bool:
        with self._lock:
            copy = self._copies.get((tensor_id, resource_id))
            if copy is None:
                return False
            return (not copy.stale) if valid_only else True

    def resources_for(self, tensor_id: str, *, valid_only: bool = False) -> tuple[str, ...]:
        with self._lock:
            out = []
            for (tid, rid), copy in self._copies.items():
                if tid != tensor_id:
                    continue
                if valid_only and copy.stale:
                    continue
                out.append(rid)
            return tuple(out)

    def mark_ready(self, tensor_id: str, resource_id: str, event: Any | None = None) -> None:
        with self._lock:
            copy = self._copies.get((tensor_id, resource_id))
            if copy is None:
                raise RuntimePlanError(f"mark_ready: no copy of {tensor_id!r} on {resource_id!r}")
            copy.ready_event = event

    def add_consumer(self, tensor_id: str, resource_id: str) -> None:
        with self._lock:
            copy = self.require_unlocked(tensor_id, resource_id)
            copy.active_consumers += 1

    def release_consumer(self, tensor_id: str, resource_id: str) -> None:
        with self._lock:
            copy = self._copies.get((tensor_id, resource_id))
            if copy is None:
                return
            copy.active_consumers = max(0, copy.active_consumers - 1)

    def require_unlocked(self, tensor_id: str, resource_id: str) -> ResidentCopy:
        copy = self._copies.get((tensor_id, resource_id))
        if copy is None:
            raise RuntimePlanError(f"Required copy missing: tensor={tensor_id!r} resource={resource_id!r}")
        if copy.stale:
            raise RuntimePlanError(f"Required copy stale: tensor={tensor_id!r} resource={resource_id!r}")
        return copy

    def drop(self, tensor_id: str, resource_id: str) -> int:
        """Drop the exact ``(tensor_id, resource_id)`` copy. Never drops siblings."""
        freed = 0
        with self._lock:
            key = (tensor_id, resource_id)
            if key not in self._copies:
                return 0
            copy = self._copies.pop(key)
            alloc_id = self._alloc_by_copy.pop(key, copy.allocation_id)
            if self._allocations is not None and alloc_id is not None and not self.value_bag_only:
                freed = int(self._allocations.release(alloc_id))
            elif self.value_bag_only:
                freed = copy.nbytes
            else:
                # Unbound: free nbytes only when no sibling still holds the same allocation.
                alloc = copy.allocation_id or _allocation_id(copy.value, tensor_id, resource_id)
                still = any(
                    (c.allocation_id or _allocation_id(c.value, tid, rid)) == alloc
                    for (tid, rid), c in self._copies.items()
                )
                freed = 0 if still else copy.nbytes
        return freed

    def move(
        self,
        tensor_id: str,
        source_resource: str,
        dest_resource: str,
        value: Any,
        *,
        tier: str,
    ) -> ResidentCopy:
        """Relocate a copy to another resource without bumping the logical version."""
        with self._lock:
            src = self._copies.get((tensor_id, source_resource))
            if src is None or src.stale:
                raise RuntimePlanError(f"Cannot move {tensor_id!r}: source {source_resource!r} missing or stale")
            version = src.version
            authoritative = src.authoritative
            ownership = src.ownership
            self.drop(tensor_id, source_resource)
            return self._install(
                tensor_id,
                dest_resource,
                value,
                nbytes=_nbytes(value),
                tier=tier,
                version=version,
                authoritative=authoritative,
                ownership=ownership,
                ready_event=None,
                stale=False,
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                f"{tid}@{rid}": {
                    "nbytes": c.nbytes,
                    "tier": c.tier,
                    "version": c.version,
                    "stale": c.stale,
                    "authoritative": c.authoritative,
                    "active_consumers": c.active_consumers,
                    "valid": c.valid,
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
    """Identity of the backing physical allocation, independent of tensor views.

    Shape, stride and storage offset describe a view and deliberately do not enter
    the allocation key. Separate host/device copies have separate storage pointers
    or explicit virtual allocation keys.
    """
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
