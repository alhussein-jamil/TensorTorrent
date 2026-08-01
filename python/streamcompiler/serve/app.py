"""Inference service: request lifecycle, health, readiness, metrics."""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
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
    worker_threads: int = 0
    cancellation_grace_s: float = 1.0

    def __post_init__(self) -> None:
        for name in ("max_queue_depth", "default_concurrency", "worker_threads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        for name in ("default_timeout_s", "cancellation_grace_s"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.max_queue_depth < 1:
            raise ValueError("max_queue_depth must be >= 1")
        if self.default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be > 0")
        if self.default_concurrency < 1:
            raise ValueError("default_concurrency must be >= 1")
        if self.worker_threads < 0:
            raise ValueError("worker_threads must be >= 0")
        if self.cancellation_grace_s < 0:
            raise ValueError("cancellation_grace_s must be >= 0")


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
            "timeout_total": 0.0,
            "inference_latency_sum_s": 0.0,
        }
    )
    _pool: ThreadPoolExecutor | None = None
    _active: dict[str, tuple[Future[Any], Any | None]] = field(default_factory=dict)
    _reserved_request_ids: set[str] = field(default_factory=set)

    def _ensure_pool(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._pool is None:
                workers = self.config.worker_threads or self.config.default_concurrency
                self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="sc-infer")
            return self._pool

    def start(self) -> None:
        with self._lock:
            self._shutting_down = False
            self._ready = True
        self._ensure_pool()
        if self.device_workers is not None:
            self.device_workers.ensure_healthy()
        logger.info("inference service ready")

    def stop(self) -> None:
        with self._lock:
            self._ready = False
            self._shutting_down = True
            active = list(self._active.values())
            self._reserved_request_ids.clear()
            pool = self._pool
            self._pool = None
        for future, token in active:
            if token is not None and hasattr(token, "cancel"):
                token.cancel()
            future.cancel()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
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
                "active_requests": len(self._active),
                "requests_total": int(self._metrics["requests_total"]),
                "requests_success": int(self._metrics["requests_success"]),
                "requests_failed": int(self._metrics["requests_failed"]),
                "requests_cancelled": int(self._metrics["requests_cancelled"]),
                "queue_rejects": int(self._metrics["queue_reject_total"]),
                "timeouts": int(self._metrics["timeout_total"]),
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
                "# HELP streamcompiler_requests_cancelled_total Cancelled or timed-out inferences",
                "# TYPE streamcompiler_requests_cancelled_total counter",
                f"streamcompiler_requests_cancelled_total {int(self._metrics['requests_cancelled'])}",
                "# HELP streamcompiler_queue_depth Current request queue depth",
                "# TYPE streamcompiler_queue_depth gauge",
                f"streamcompiler_queue_depth {self._queue_depth}",
                "# HELP streamcompiler_queue_rejects_total Requests rejected by queue backpressure",
                "# TYPE streamcompiler_queue_rejects_total counter",
                f"streamcompiler_queue_rejects_total {int(self._metrics['queue_reject_total'])}",
                "# HELP streamcompiler_timeouts_total Timed out inference requests",
                "# TYPE streamcompiler_timeouts_total counter",
                f"streamcompiler_timeouts_total {int(self._metrics['timeout_total'])}",
                "# HELP streamcompiler_active_requests Current executing or draining requests",
                "# TYPE streamcompiler_active_requests gauge",
                f"streamcompiler_active_requests {len(self._active)}",
                "# HELP streamcompiler_inference_latency_seconds_sum Cumulative inference latency",
                "# TYPE streamcompiler_inference_latency_seconds_sum counter",
                f"streamcompiler_inference_latency_seconds_sum {self._metrics['inference_latency_sum_s']}",
            ]
        return "\n".join(lines) + "\n"

    def cancel(self, request_id: str) -> bool:
        """Request cooperative cancellation of one active inference."""
        with self._lock:
            active = self._active.get(request_id)
        if active is None:
            return False
        future, token = active
        if token is not None and hasattr(token, "cancel"):
            token.cancel()
        future.cancel()
        return True

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
            if not math.isfinite(timeout) or timeout <= 0:
                raise StreamCompilerError("timeout_s must be > 0 and finite")
            timeout = min(timeout, 3600.0)

        with self._lock:
            if self._shutting_down or not self._ready:
                raise StreamCompilerError("service not ready")
            if rid in self._active or rid in self._reserved_request_ids:
                raise StreamCompilerError(f"duplicate active request_id: {rid}")
            if self._queue_depth >= self.config.max_queue_depth:
                self._metrics["queue_reject_total"] += 1.0
                raise StreamCompilerError("backpressure: request queue full")
            self._reserved_request_ids.add(rid)
            self._queue_depth += 1
            self._metrics["requests_total"] += 1.0
            rec = RequestRecord(request_id=rid, model_id=model_id, status="running", started_at=time.time())
            self._requests.append(rec)

        try:
            slot = self.models.acquire(model_id)
        except BaseException as exc:
            with self._lock:
                self._reserved_request_ids.discard(rid)
                self._queue_depth = max(0, self._queue_depth - 1)
                self._metrics["requests_failed"] += 1.0
                rec.status = "failed"
                rec.error = str(exc)
                rec.finished_at = time.time()
            raise

        cancel_token: Any | None = None
        try:
            from streamcompiler.native import require_native

            cancel_token = require_native().NativeCancelToken()
        except Exception:  # noqa: BLE001 - service still supports test doubles
            cancel_token = None

        task_registered = threading.Event()
        task_started = threading.Event()

        def _run() -> Any:
            # Prevent an ultra-fast task from completing before the request is
            # visible in _active, which would leave stale request reservations.
            task_registered.wait()
            task_started.set()
            try:
                module = slot.module
                if isinstance(inputs, tuple):
                    if cancel_token is not None and hasattr(module, "_forward_with_cancel_token"):
                        return module._forward_with_cancel_token(cancel_token, *inputs)
                    return module(*inputs)
                if cancel_token is not None and hasattr(module, "_forward_with_cancel_token"):
                    return module._forward_with_cancel_token(cancel_token, inputs)
                return module(inputs)
            finally:
                self.models.release_slot(slot)

        pool = self._ensure_pool()
        try:
            future = pool.submit(_run)
        except BaseException as exc:
            self.models.release_slot(slot)
            with self._lock:
                self._reserved_request_ids.discard(rid)
                self._queue_depth = max(0, self._queue_depth - 1)
                self._metrics["requests_failed"] += 1.0
                rec.status = "failed"
                rec.error = str(exc)
                rec.finished_at = time.time()
            raise

        def _finished(_future: Future[Any]) -> None:
            if _future.cancelled() and not task_started.is_set():
                self.models.release_slot(slot)
            with self._lock:
                self._active.pop(rid, None)
                self._reserved_request_ids.discard(rid)
                self._queue_depth = max(0, self._queue_depth - 1)

        with self._lock:
            self._active[rid] = (future, cancel_token)
            self._reserved_request_ids.discard(rid)
        future.add_done_callback(_finished)
        task_registered.set()
        t0 = time.perf_counter()
        try:
            out = future.result(timeout=timeout)
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
        except FutureTimeoutError as exc:
            if cancel_token is not None and hasattr(cancel_token, "cancel"):
                cancel_token.cancel()
            elif hasattr(slot.module, "request_cancel"):
                slot.module.request_cancel()
            grace = float(self.config.cancellation_grace_s)
            if grace > 0:
                with contextlib.suppress(BaseException):
                    future.result(timeout=grace)
            with self._lock:
                self._metrics["requests_cancelled"] += 1.0
                self._metrics["timeout_total"] += 1.0
                rec.status = "cancelled"
                rec.error = f"timed out after {timeout}s"
                rec.finished_at = time.time()
            raise ExecutionCancelled(f"request {rid} timed out after {timeout}s") from exc
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
