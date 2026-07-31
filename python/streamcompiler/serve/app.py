"""Inference service: request lifecycle, health, readiness, metrics."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from streamcompiler.errors import ExecutionCancelled, StreamCompilerError
from streamcompiler.runtime.device_workers import DeviceWorkerSupervisor
from streamcompiler.serve.model_manager import ModelManager

logger = logging.getLogger("streamcompiler.server")


@dataclass
class ServiceConfig:
    max_queue_depth: int = 64
    default_timeout_s: float = 30.0
    default_concurrency: int = 8


@dataclass
class RequestRecord:
    request_id: str
    model_id: str
    status: str
    error: str | None = None
    started_at: float = 0.0
    finished_at: float | None = None


@dataclass
class InferenceService:
    """Minimal production service surface around CompiledModule.

    Not an HTTP server by itself — bind via ASGI/WSGI or gRPC separately.
    Survives failed requests without corrupting future ones.
    """

    config: ServiceConfig = field(default_factory=ServiceConfig)
    models: ModelManager = field(default_factory=ModelManager)
    device_workers: DeviceWorkerSupervisor | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _queue_depth: int = 0
    _shutting_down: bool = False
    _ready: bool = False
    _requests: deque[RequestRecord] = field(default_factory=lambda: deque(maxlen=1024))
    _metrics: dict[str, float] = field(
        default_factory=lambda: {
            "requests_total": 0.0,
            "requests_success": 0.0,
            "requests_failed": 0.0,
            "requests_cancelled": 0.0,
            "queue_reject_total": 0.0,
            "inference_latency_sum_s": 0.0,
        }
    )

    def start(self) -> None:
        with self._lock:
            self._shutting_down = False
            self._ready = True
        if self.device_workers is not None:
            self.device_workers.ensure_healthy()
        logger.info("inference service ready")

    def stop(self) -> None:
        with self._lock:
            self._ready = False
            self._shutting_down = True
        self.models.shutdown()
        if self.device_workers is not None:
            self.device_workers.shutdown()
        logger.info("inference service stopped")

    def health(self) -> dict[str, Any]:
        workers = None
        if self.device_workers is not None:
            workers = [
                {
                    "device_id": s.device_id,
                    "alive": s.alive,
                    "pid": s.pid,
                    "restarts": s.restarts,
                    "pending": s.pending,
                }
                for s in self.device_workers.health()
            ]
        with self._lock:
            return {
                "status": "ok" if not self._shutting_down else "stopping",
                "ready": self._ready,
                "queue_depth": self._queue_depth,
                "models": len(self.models.list_models()),
                "device_workers": workers,
            }

    def readiness(self) -> dict[str, Any]:
        workers_ok = True
        if self.device_workers is not None:
            # Restart dead workers once, then require all alive.
            self.device_workers.ensure_healthy()
            workers_ok = all(s.alive for s in self.device_workers.health())
        with self._lock:
            ready = self._ready and not self._shutting_down and workers_ok
        return {"ready": ready, "device_workers_ok": workers_ok}

    def metrics_prometheus(self) -> str:
        with self._lock:
            lines = [
                "# HELP streamcompiler_requests_total Total inference requests",
                "# TYPE streamcompiler_requests_total counter",
                f"streamcompiler_requests_total {int(self._metrics['requests_total'])}",
                "# HELP streamcompiler_requests_success_total Successful inferences",
                "# TYPE streamcompiler_requests_success_total counter",
                f"streamcompiler_requests_success_total {int(self._metrics['requests_success'])}",
                "# HELP streamcompiler_requests_failed_total Failed inferences",
                "# TYPE streamcompiler_requests_failed_total counter",
                f"streamcompiler_requests_failed_total {int(self._metrics['requests_failed'])}",
                "# HELP streamcompiler_queue_depth Current request queue depth",
                "# TYPE streamcompiler_queue_depth gauge",
                f"streamcompiler_queue_depth {self._queue_depth}",
                "# HELP streamcompiler_inference_latency_seconds_sum Cumulative inference latency",
                "# TYPE streamcompiler_inference_latency_seconds_sum counter",
                f"streamcompiler_inference_latency_seconds_sum {self._metrics['inference_latency_sum_s']}",
            ]
        return "\n".join(lines) + "\n"

    def infer(
        self,
        model_id: str,
        inputs: Any,
        *,
        request_id: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        rid = request_id or uuid.uuid4().hex
        if timeout_s is None:
            timeout = self.config.default_timeout_s
        else:
            try:
                timeout = float(timeout_s)
            except (TypeError, ValueError) as exc:
                raise StreamCompilerError("timeout_s must be a number") from exc
            if timeout <= 0:
                raise StreamCompilerError("timeout_s must be > 0")
            timeout = min(timeout, 3600.0)
        with self._lock:
            if self._shutting_down or not self._ready:
                raise StreamCompilerError("service not ready")
            if self._queue_depth >= self.config.max_queue_depth:
                self._metrics["queue_reject_total"] += 1.0
                raise StreamCompilerError("backpressure: request queue full")
            self._queue_depth += 1
            self._metrics["requests_total"] += 1.0
            rec = RequestRecord(request_id=rid, model_id=model_id, status="running", started_at=time.time())
            self._requests.append(rec)

        slot = None
        try:
            slot = self.models.acquire(model_id)
            t0 = time.perf_counter()
            # Timeout is cooperative via cancel token when available.
            cancel = getattr(slot.module, "cancel_token", None)
            if cancel is not None and hasattr(cancel, "reset"):
                cancel.reset()
            deadline = time.time() + timeout

            def _run() -> Any:
                if isinstance(inputs, tuple):
                    return slot.module(*inputs)
                return slot.module(inputs)

            # Run inline; production HTTP layer should offload to a worker pool.
            # Enforce timeout by polling cancel if the module supports it.
            result_box: dict[str, Any] = {}
            error_box: dict[str, BaseException] = {}

            def target() -> None:
                try:
                    result_box["out"] = _run()
                except BaseException as exc:  # noqa: BLE001 — record then re-raise path
                    error_box["err"] = exc

            th = threading.Thread(target=target, name=f"sc-infer-{rid[:8]}", daemon=True)
            th.start()
            while th.is_alive():
                if time.time() > deadline:
                    if cancel is not None and hasattr(cancel, "cancel"):
                        cancel.cancel()
                    th.join(timeout=1.0)
                    raise ExecutionCancelled(f"request {rid} timed out after {timeout}s")
                th.join(timeout=0.01)
            if "err" in error_box:
                raise error_box["err"]
            out = result_box["out"]
            elapsed = time.perf_counter() - t0
            with self._lock:
                self._metrics["requests_success"] += 1.0
                self._metrics["inference_latency_sum_s"] += elapsed
                rec.status = "ok"
                rec.finished_at = time.time()
            return {
                "request_id": rid,
                "model_id": model_id,
                "version": slot.version,
                "output": out,
                "latency_s": elapsed,
            }
        except ExecutionCancelled as exc:
            with self._lock:
                self._metrics["requests_cancelled"] += 1.0
                rec.status = "cancelled"
                rec.error = str(exc)
                rec.finished_at = time.time()
            raise
        except Exception as exc:
            with self._lock:
                self._metrics["requests_failed"] += 1.0
                rec.status = "failed"
                rec.error = str(exc)
                rec.finished_at = time.time()
            logger.exception("request %s failed", rid)
            raise
        finally:
            if slot is not None:
                self.models.release(model_id)
            with self._lock:
                self._queue_depth = max(0, self._queue_depth - 1)
