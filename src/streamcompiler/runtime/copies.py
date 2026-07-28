"""Multi-copy tensor residency keyed by (logical tensor id, resource id).

One logical tensor may hold valid copies on several resources at once. Creating a
GPU copy must not invalidate or overwrite the CPU copy.
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


@dataclass
class CopyStore:
    """Map ``(tensor_id, resource_id) -> ResidentCopy``."""

    _copies: dict[tuple[str, str], ResidentCopy] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _versions: dict[str, int] = field(default_factory=dict)

    def put(self, tensor_id: str, resource_id: str, value: Any, *, tier: str = "system_ram") -> ResidentCopy:
        nbytes = 0
        if isinstance(value, torch.Tensor):
            nbytes = int(value.numel() * value.element_size())
        with self._lock:
            version = self._versions.get(tensor_id, 0) + 1
            self._versions[tensor_id] = version
            copy = ResidentCopy(
                tensor_id=tensor_id,
                resource_id=resource_id,
                value=value,
                nbytes=nbytes,
                tier=tier,
                version=version,
            )
            self._copies[(tensor_id, resource_id)] = copy
            return copy

    def get(self, tensor_id: str, resource_id: str) -> ResidentCopy:
        with self._lock:
            copy = self._copies.get((tensor_id, resource_id))
            if copy is None:
                raise RuntimePlanError(
                    f"No resident copy of {tensor_id!r} on resource {resource_id!r}"
                )
            return copy

    def try_get(self, tensor_id: str, resource_id: str) -> ResidentCopy | None:
        with self._lock:
            return self._copies.get((tensor_id, resource_id))

    def has(self, tensor_id: str, resource_id: str) -> bool:
        with self._lock:
            return (tensor_id, resource_id) in self._copies

    def resources_for(self, tensor_id: str) -> tuple[str, ...]:
        with self._lock:
            return tuple(rid for (tid, rid) in self._copies if tid == tensor_id)

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
        return freed

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                f"{tid}@{rid}": {"nbytes": c.nbytes, "tier": c.tier, "version": c.version}
                for (tid, rid), c in self._copies.items()
            }
