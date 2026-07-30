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


def _drain_in_flight(slot: ModelSlot, *, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while slot.in_flight > 0 and time.time() < deadline:
        time.sleep(0.01)
    if slot.in_flight > 0:
        logger.warning(
            "model %s still has %s in-flight requests after %.1fs drain",
            slot.model_id,
            slot.in_flight,
            timeout_s,
        )


@dataclass
class ModelManager:
    """Owns loaded CompiledModule instances. Thread-safe."""

    _lock: threading.RLock = field(default_factory=threading.RLock)
    _models: dict[str, ModelSlot] = field(default_factory=dict)

    def load(self, model_id: str, module: CompiledModule, *, concurrency_limit: int = 8) -> str:
        version = uuid.uuid4().hex[:12]
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
            # New slot is published; drain previous generation before close.
            _drain_in_flight(old)
            try:
                old.module.close()
            except Exception:
                logger.exception("failed closing replaced model %s", model_id)
        return version

    def unload(self, model_id: str) -> None:
        with self._lock:
            slot = self._models.pop(model_id, None)
        if slot is None:
            raise StreamCompilerError(f"model not loaded: {model_id}")
        _drain_in_flight(slot)
        slot.module.close()

    def warm(self, model_id: str, example_inputs: Any) -> None:
        with self._lock:
            slot = self._models.get(model_id)
            if slot is None:
                raise StreamCompilerError(f"model not loaded: {model_id}")
            module = slot.module
        _ = module(*example_inputs) if isinstance(example_inputs, tuple) else module(example_inputs)
        with self._lock:
            if model_id in self._models:
                self._models[model_id].warm = True

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

    def release(self, model_id: str) -> None:
        with self._lock:
            slot = self._models.get(model_id)
            if slot is not None and slot.in_flight > 0:
                slot.in_flight -= 1

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
