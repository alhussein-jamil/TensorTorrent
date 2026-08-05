"""Opaque tensor handles: Python owns values; Rust owns residency metadata."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import torch

from tensortorrent.errors import RuntimePlanError
from tensortorrent.native import require_native


def _tensor_view_meta(value: Any) -> dict[str, Any]:
    """Extract backing-storage identity and view layout for native residency."""
    if not isinstance(value, torch.Tensor):
        return {
            "shape": None,
            "strides": None,
            "storage_offset": 0,
            "dtype": "",
            "storage_nbytes": 0,
            "storage_id": None,
        }
    t = value.detach()
    storage = t.untyped_storage()
    return {
        "shape": list(t.shape),
        "strides": list(t.stride()),
        "storage_offset": int(t.storage_offset()),
        "dtype": str(t.dtype).removeprefix("torch."),
        "storage_nbytes": int(storage.nbytes()),
        "storage_id": f"{t.device}:{storage.data_ptr()}",
    }


@dataclass
class TensorHandleTable:
    """Maps opaque u64 handle ids to owned Python tensor objects."""

    _values: dict[int, Any] = field(default_factory=dict)
    _next: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def insert(self, value: Any) -> int:
        with self._lock:
            handle = self._next
            self._next += 1
            self._values[handle] = value
            return handle

    def get(self, handle_id: int) -> Any:
        with self._lock:
            try:
                return self._values[handle_id]
            except KeyError as exc:
                raise RuntimePlanError(f"unknown tensor handle {handle_id}") from exc

    def drop(self, handle_id: int) -> Any | None:
        with self._lock:
            return self._values.pop(int(handle_id), None)

    def set(self, handle_id: int, value: Any) -> None:
        """Bind or replace the Python value for an existing opaque handle id."""
        with self._lock:
            self._values[int(handle_id)] = value

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)


@dataclass
class NativeResidencyBridge:
    """Mirror CopyStore puts into Rust ``NativeResidencySession``.

    Rust is authoritative for valid/missing/stale/lease. Python table holds values.
    """

    session: Any
    handles: TensorHandleTable = field(default_factory=TensorHandleTable)
    # (tensor_id, resource_id) -> handle_id for fast CopyStore sync
    _index: dict[tuple[str, str], int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def create(cls) -> NativeResidencyBridge:
        """Orphan session — tests only. Production must use create_from_context."""
        native = require_native()
        return cls(session=native.NativeResidencySession())

    @classmethod
    def create_from_context(cls, execution_context: Any) -> NativeResidencyBridge:
        """Bind bridge to a shared ``NativeExecutionContext`` residency store."""
        native = require_native()
        session = native.NativeResidencySession.from_execution_context(execution_context)
        return cls(session=session)

    def mirror_put(
        self,
        tensor_id: str,
        resource_id: str,
        value: Any,
        *,
        nbytes: int,
        authoritative: bool = True,
        view_meta: dict[str, Any] | None = None,
    ) -> int:
        """Register ``value`` in the Rust residency session.

        ``view_meta`` lets callers whose tensor identity never changes across
        forwards (e.g. resident parameters) skip re-deriving storage/view
        metadata every time; see :func:`_tensor_view_meta`.

        If Rust already owns ``(tensor_id, resource_id)`` (e.g. Transfer completed
        before CopyStore catch-up), refresh the Python handle table only — do not
        ``session.put`` again.
        """
        tid, rid = str(tensor_id), str(resource_id)
        with self._lock:
            if self.session.has(tid, rid):
                handle = int(self.session.require(tid, rid))
                self.handles.set(handle, value)
                self._index[(tid, rid)] = handle
                return handle
            handle = self.handles.insert(value)
            meta = view_meta if view_meta is not None else _tensor_view_meta(value)
            self.session.put(
                tid,
                rid,
                int(handle),
                int(max(0, nbytes)),
                authoritative,
                shape=meta["shape"],
                strides=meta["strides"],
                storage_offset=meta["storage_offset"],
                dtype=meta["dtype"],
                storage_nbytes=int(meta["storage_nbytes"] or max(0, nbytes)),
                storage_id=meta["storage_id"],
            )
            self._index[(tid, rid)] = handle
            return handle

    def mirror_alias(self, tensor_id: str, src_resource: str, dst_resource: str) -> None:
        if src_resource == dst_resource:
            return
        if self.session.has(str(tensor_id), str(dst_resource)):
            return
        self.session.alias(str(tensor_id), str(src_resource), str(dst_resource))
        with self._lock:
            handle = self._index.get((str(tensor_id), str(src_resource)))
            if handle is not None:
                self._index[(str(tensor_id), str(dst_resource))] = handle

    def require_handle(self, tensor_id: str, resource_id: str) -> int:
        return int(self.session.require(str(tensor_id), str(resource_id)))

    def require_value(self, tensor_id: str, resource_id: str) -> Any:
        handle = self.require_handle(tensor_id, resource_id)
        return self.handles.get(handle)

    def release(self, tensor_id: str, resource_id: str) -> int:
        """Release Rust residency then drop Python index + opaque value if unused.

        Lease / alias-unsafe errors propagate. Missing copies are tolerated when
        Rust already freed the copy (native mid-schedule path).
        """
        freed = 0
        if self.session.has(str(tensor_id), str(resource_id)):
            freed = int(self.session.release(str(tensor_id), str(resource_id)))
        with self._lock:
            handle = self._index.pop((str(tensor_id), str(resource_id)), None)
            if handle is not None and handle not in self._index.values():
                self.handles.drop(handle)
        return freed

    def drop_python_only(self, tensor_id: str, resource_id: str) -> None:
        """Drop opaque handle after Rust already final-released the allocation."""
        with self._lock:
            handle = self._index.pop((str(tensor_id), str(resource_id)), None)
            if handle is not None and handle not in self._index.values():
                self.handles.drop(handle)

    def live_handle_bytes(self) -> int:
        with self._lock:
            total = 0
            for value in self.handles._values.values():
                total += int(getattr(value, "nbytes", 0) or 0)
            return total

    def stats(self) -> dict[str, Any]:
        raw = dict(self.session.stats())
        raw["handle_live"] = len(self.handles)
        raw["handle_live_bytes"] = self.live_handle_bytes()
        return raw
