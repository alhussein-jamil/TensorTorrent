"""Model load / unload / warm / atomic replace."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from streamcompiler.errors import StreamCompilerError
from streamcompiler.runtime.module import CompiledModule

logger = logging.getLogger("streamcompiler.server.models")


@dataclass
class ModelSlot:
    model_id: str
    version: str
    module: CompiledModule
    loaded_at: float
    warm: bool = False
    concurrency_limit: int = 8
    in_flight: int = 0
    retired: bool = False
    closed: bool = False


@dataclass
class ModelManager:
    """Owns loaded CompiledModule instances. Thread-safe."""

    _lock: threading.RLock = field(default_factory=threading.RLock)
    _models: dict[str, ModelSlot] = field(default_factory=dict)
    _retired: dict[str, ModelSlot] = field(default_factory=dict)
    _condition: threading.Condition = field(init=False)

    def __post_init__(self) -> None:
        self._condition = threading.Condition(self._lock)

    def _drain_in_flight(self, slot: ModelSlot, *, timeout_s: float = 5.0) -> None:
        deadline = time.time() + timeout_s
        with self._condition:
            while slot.in_flight > 0:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
        if slot.in_flight > 0:
            logger.warning(
                "model %s still has %s in-flight requests after %.1fs drain",
                slot.model_id,
                slot.in_flight,
                timeout_s,
            )

    @staticmethod
    def _close_slot(slot: ModelSlot) -> None:
        try:
            slot.module.close()
        except Exception:
            logger.exception(
                "failed closing model %s generation %s",
                slot.model_id,
                slot.version,
            )

    def load(self, model_id: str, module: CompiledModule, *, concurrency_limit: int = 8) -> str:
        """Publish a new generation without blocking on prior in-flight work.

        Idle prior generations close immediately. Busy ones retire and close
        when their final :meth:`release_slot` runs.
        """
        version = uuid.uuid4().hex[:12]
        close_now: ModelSlot | None = None
        with self._lock:
            old = self._models.get(model_id)
            self._models[model_id] = ModelSlot(
                model_id=model_id,
                version=version,
                module=module,
                loaded_at=time.time(),
                concurrency_limit=max(1, concurrency_limit),
            )
            if old is not None:
                old.retired = True
                if old.in_flight == 0:
                    old.closed = True
                    close_now = old
                else:
                    self._retired[old.version] = old
        if close_now is not None:
            self._close_slot(close_now)
        return version

    def unload(self, model_id: str) -> None:
        with self._lock:
            slot = self._models.pop(model_id, None)
            if slot is None:
                raise StreamCompilerError(f"model not loaded: {model_id}")
            slot.retired = True
            if slot.in_flight > 0:
                self._retired[slot.version] = slot

        self._drain_in_flight(slot)
        should_close = False
        with self._lock:
            self._retired.pop(slot.version, None)
            if not slot.closed:
                slot.closed = True
                should_close = True
        if should_close:
            self._close_slot(slot)

    def warm(self, model_id: str, example_inputs: Any) -> None:
        with self._lock:
            slot = self._models.get(model_id)
            if slot is None:
                raise StreamCompilerError(f"model not loaded: {model_id}")
            module = slot.module
        _ = module(*example_inputs) if isinstance(example_inputs, tuple) else module(example_inputs)
        with self._lock:
            current = self._models.get(model_id)
            if current is slot:
                current.warm = True

    def get(self, model_id: str) -> ModelSlot:
        with self._lock:
            slot = self._models.get(model_id)
            if slot is None:
                raise StreamCompilerError(f"model not loaded: {model_id}")
            return slot

    def acquire(self, model_id: str) -> ModelSlot:
        with self._lock:
            slot = self._models.get(model_id)
            if slot is None:
                raise StreamCompilerError(f"model not loaded: {model_id}")
            if slot.in_flight >= slot.concurrency_limit:
                raise StreamCompilerError(
                    f"backpressure: model {model_id} at concurrency limit {slot.concurrency_limit}"
                )
            slot.in_flight += 1
            return slot

    def release_slot(self, slot: ModelSlot) -> None:
        """Release the exact model generation acquired by a request.

        Releasing by model id is unsafe during atomic model replacement because
        the id may already point to a newer generation. Closing a retired
        generation is deferred until its final in-flight request releases.
        """
        close_now = False
        with self._condition:
            if slot.in_flight > 0:
                slot.in_flight -= 1
            if slot.retired and slot.in_flight == 0 and not slot.closed:
                slot.closed = True
                self._retired.pop(slot.version, None)
                close_now = True
            self._condition.notify_all()
        if close_now:
            self._close_slot(slot)

    def release(self, model_id: str) -> None:
        """Release the *current* generation for ``model_id`` only.

        Prefer :meth:`release_slot` for request paths. This helper must not
        touch retired generations — doing so would under-count replacements.
        """
        with self._lock:
            slot = self._models.get(model_id)
            if slot is None or slot.retired:
                return
        self.release_slot(slot)

    def list_models(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "model_id": s.model_id,
                    "version": s.version,
                    "warm": s.warm,
                    "in_flight": s.in_flight,
                    "concurrency_limit": s.concurrency_limit,
                    "loaded_at": s.loaded_at,
                }
                for s in self._models.values()
            ]

    def shutdown(self) -> None:
        with self._lock:
            ids = list(self._models.keys())
        for mid in ids:
            with contextlib.suppress(StreamCompilerError):
                self.unload(mid)

        # Retired generations can remain only when their requests outlive the
        # normal drain timeout. Close any that have since become idle.
        close_now: list[ModelSlot] = []
        with self._lock:
            for version, slot in list(self._retired.items()):
                if slot.in_flight == 0 and not slot.closed:
                    slot.closed = True
                    close_now.append(slot)
                    self._retired.pop(version, None)
        for slot in close_now:
            self._close_slot(slot)
