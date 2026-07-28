"""Authoritative multi-copy tensor residency.

Keyed by ``(logical_tensor_id, resource_id)``. One logical tensor may hold
simultaneous valid copies on several resources. Replication (transfer) does not
change the logical version; mutation does, and marks sibling copies stale.
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
    """Map ``(tensor_id, resource_id) -> ResidentCopy`` — sole residency authority."""

    _copies: dict[tuple[str, str], ResidentCopy] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _versions: dict[str, int] = field(default_factory=dict)
    _live_bytes: int = 0
    _peak_bytes: int = 0

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
        prev = self._copies.get((tensor_id, resource_id))
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
        )
        self._copies[(tensor_id, resource_id)] = copy
        if prev is not None:
            self._live_bytes -= prev.nbytes
        self._live_bytes += nbytes
        self._peak_bytes = max(self._peak_bytes, self._live_bytes)
        return copy

    def live_bytes(self) -> int:
        with self._lock:
            return self._live_bytes

    def peak_bytes(self) -> int:
        with self._lock:
            return self._peak_bytes

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

    def drop(self, tensor_id: str, resource_id: str | None = None) -> int:
        """Drop one resource copy, or all copies of ``tensor_id`` when resource is None."""
        freed = 0
        with self._lock:
            if resource_id is None:
                keys = [k for k in self._copies if k[0] == tensor_id]
            else:
                keys = [(tensor_id, resource_id)] if (tensor_id, resource_id) in self._copies else []
            for key in keys:
                freed += self._copies.pop(key).nbytes
            self._live_bytes = max(0, self._live_bytes - freed)
        return freed

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
