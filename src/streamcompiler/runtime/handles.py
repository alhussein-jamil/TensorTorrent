"""Opaque tensor handles: Python owns values; Rust owns residency metadata."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from streamcompiler.errors import RuntimePlanError
from streamcompiler.native import require_native


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
    ) -> int:
        with self._lock:
            existing = self._index.get((str(tensor_id), str(resource_id)))
            if existing is not None and self.session.has(str(tensor_id), str(resource_id)):
                return existing
            handle = self.handles.insert(value)
            self.session.put(
                str(tensor_id),
                str(resource_id),
                int(handle),
                int(max(0, nbytes)),
                authoritative,
            )
            self._index[(str(tensor_id), str(resource_id))] = handle
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
        # Rust first — missing/leased fails closed.
        freed = int(self.session.release(str(tensor_id), str(resource_id)))
        with self._lock:
            handle = self._index.pop((str(tensor_id), str(resource_id)), None)
        if handle is not None:
            self.handles.drop(handle)
        return freed

    def stats(self) -> dict[str, Any]:
        raw = dict(self.session.stats())
        raw["handle_live"] = len(self.handles)
        return raw
