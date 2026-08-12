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

from tensortorrent.errors import ExecutionCancelled, TensorTorrentError
from tensortorrent.runtime.device_workers import DeviceWorkerSupervisor
from tensortorrent.serve.model_manager import ModelManager
from tensortorrent.serve.service_config import ServiceConfig

logger = logging.getLogger("tensortorrent.server")

# Histogram buckets for tensortorrent_inference_latency_seconds
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


@dataclass
class RequestRecord:
    request_id: str
    model_id: str
    status: str
    error: str | None = None
    started_at: float = 0.0
    finished_at: float | None = None


def _make_model_latency() -> dict[str, Any]:
    return {
        "buckets": [0] * len(_LATENCY_BUCKETS),  # counts per le bucket
        "count": 0,
        "sum": 0.0,
    }


def _make_model_requests() -> dict[str, int]:
    return {"success": 0, "failed": 0, "cancelled": 0, "timeout": 0}


@dataclass
class InferenceService:
    """Service surface around CompiledModule (not the HTTP server itself).

    Survives failed requests without poisoning later ones.
    """

    config: ServiceConfig = field(default_factory=ServiceConfig)
    models: ModelManager = field(default_factory=ModelManager)
    device_workers: DeviceWorkerSupervisor | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _queue_depth: int = 0
    _shutting_down: bool = False
    _ready: bool = False
    _requests: deque[RequestRecord] = field(init=False)
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

    # Per-model histogram: model_id -> {"buckets": [...], "count": int, "sum": float}
    _model_latency: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Per-model outcome counter: model_id -> {"success": int, "failed": int, "cancelled": int, "timeout": int}
    _model_requests: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._requests = deque(maxlen=self.config.request_history_size)

    def _ensure_pool(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._pool is None:
                workers = self.config.worker_threads or self.config.default_concurrency
                self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="tt-infer")
            return self._pool

    def _ensure_model_metrics(self, model_id: str) -> None:
        """Ensure per-model metric dicts exist. Must be called under _lock."""
        if model_id not in self._model_latency:
            self._model_latency[model_id] = _make_model_latency()
        if model_id not in self._model_requests:
            self._model_requests[model_id] = _make_model_requests()

    def _record_latency(self, model_id: str, elapsed: float) -> None:
        """Update the per-model histogram. Must be called under _lock."""
        self._ensure_model_metrics(model_id)
        hist = self._model_latency[model_id]
        hist["count"] += 1
        hist["sum"] += elapsed
        for i, le in enumerate(_LATENCY_BUCKETS):
            if elapsed <= le:
                hist["buckets"][i] += 1

    def _record_outcome(self, model_id: str, outcome: str) -> None:
        """Increment per-model outcome counter. Must be called under _lock."""
        self._ensure_model_metrics(model_id)
        counters = self._model_requests[model_id]
        if outcome in counters:
            counters[outcome] += 1

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
            # Readiness is side-effect-free. The supervisor collector owns
            # bounded restart attempts; probes only report current state.
            workers_ok = all(s.alive for s in self.device_workers.health())
        models_loaded = len(self.models.list_models())
        with self._lock:
            ready = self._ready and not self._shutting_down and workers_ok and models_loaded > 0
        return {"ready": ready, "device_workers_ok": workers_ok, "models_loaded": models_loaded}

    def metrics_prometheus(self) -> str:
        with self._lock:
            # Snapshot per-model data while holding the lock.
            model_latency_snap = {
                mid: {
                    "buckets": list(hist["buckets"]),
                    "count": hist["count"],
                    "sum": hist["sum"],
                }
                for mid, hist in self._model_latency.items()
            }
            model_requests_snap = {mid: dict(counters) for mid, counters in self._model_requests.items()}
            lines = [
                "# HELP tensortorrent_requests_total Total inference requests",
                "# TYPE tensortorrent_requests_total counter",
                f"tensortorrent_requests_total {int(self._metrics['requests_total'])}",
                "# HELP tensortorrent_requests_success_total Successful inferences",
                "# TYPE tensortorrent_requests_success_total counter",
                f"tensortorrent_requests_success_total {int(self._metrics['requests_success'])}",
                "# HELP tensortorrent_requests_failed_total Failed inferences",
                "# TYPE tensortorrent_requests_failed_total counter",
                f"tensortorrent_requests_failed_total {int(self._metrics['requests_failed'])}",
                "# HELP tensortorrent_requests_cancelled_total Cancelled or timed-out inferences",
                "# TYPE tensortorrent_requests_cancelled_total counter",
                f"tensortorrent_requests_cancelled_total {int(self._metrics['requests_cancelled'])}",
                "# HELP tensortorrent_queue_depth Current request queue depth",
                "# TYPE tensortorrent_queue_depth gauge",
                f"tensortorrent_queue_depth {self._queue_depth}",
                "# HELP tensortorrent_queue_rejects_total Requests rejected by queue backpressure",
                "# TYPE tensortorrent_queue_rejects_total counter",
                f"tensortorrent_queue_rejects_total {int(self._metrics['queue_reject_total'])}",
                "# HELP tensortorrent_timeouts_total Timed out inference requests",
                "# TYPE tensortorrent_timeouts_total counter",
                f"tensortorrent_timeouts_total {int(self._metrics['timeout_total'])}",
                "# HELP tensortorrent_active_requests Current executing or draining requests",
                "# TYPE tensortorrent_active_requests gauge",
                f"tensortorrent_active_requests {len(self._active)}",
                "# HELP tensortorrent_inference_latency_seconds_sum Cumulative inference latency",
                "# TYPE tensortorrent_inference_latency_seconds_sum counter",
                f"tensortorrent_inference_latency_seconds_sum {self._metrics['inference_latency_sum_s']}",
            ]

        # Per-model latency histogram (tensortorrent_inference_latency_seconds).
        if model_latency_snap:
            lines.append("# HELP tensortorrent_inference_latency_seconds Per-model inference latency histogram")
            lines.append("# TYPE tensortorrent_inference_latency_seconds histogram")
            for mid, hist in sorted(model_latency_snap.items()):
                label = f'model="{mid}"'
                cumulative = 0
                for i, le in enumerate(_LATENCY_BUCKETS):
                    cumulative += hist["buckets"][i]
                    lines.append(f'tensortorrent_inference_latency_seconds_bucket{{{label},le="{le}"}} {cumulative}')
                # +Inf bucket equals total count.
                lines.append(f'tensortorrent_inference_latency_seconds_bucket{{{label},le="+Inf"}} {hist["count"]}')
                lines.append(f"tensortorrent_inference_latency_seconds_sum{{{label}}} {hist['sum']}")
                lines.append(f"tensortorrent_inference_latency_seconds_count{{{label}}} {hist['count']}")

        # Per-model outcome counter (tensortorrent_model_requests_total).
        if model_requests_snap:
            lines.append("# HELP tensortorrent_model_requests_total Per-model request outcome counter")
            lines.append("# TYPE tensortorrent_model_requests_total counter")
            _outcome_map = {
                "success": "success",
                "failed": "failed",
                "cancelled": "cancelled",
                "timeout": "timeout",
            }
            for mid, counters in sorted(model_requests_snap.items()):
                for key, outcome in _outcome_map.items():
                    lines.append(
                        f'tensortorrent_model_requests_total{{model="{mid}",outcome="{outcome}"}} {counters[key]}'
                    )

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
        if not isinstance(model_id, str) or not model_id:
            raise TensorTorrentError("model_id must be a non-empty string")
        if request_id is not None and (not isinstance(request_id, str) or not request_id):
            raise TensorTorrentError("request_id must be a non-empty string when provided")
        rid = request_id if request_id is not None else uuid.uuid4().hex
        if timeout_s is None:
            timeout = self.config.default_timeout_s
        else:
            try:
                timeout = float(timeout_s)
            except (TypeError, ValueError) as exc:
                raise TensorTorrentError("timeout_s must be a number") from exc
            if not math.isfinite(timeout) or timeout <= 0:
                raise TensorTorrentError("timeout_s must be > 0 and finite")
            timeout = min(timeout, self.config.max_request_timeout_s)

        with self._lock:
            if self._shutting_down or not self._ready:
                raise TensorTorrentError("service not ready")
            if rid in self._active or rid in self._reserved_request_ids:
                raise TensorTorrentError(f"duplicate active request_id: {rid}")
            if self._queue_depth >= self.config.max_queue_depth:
                self._metrics["queue_reject_total"] += 1.0
                raise TensorTorrentError("backpressure: request queue full")
            self._reserved_request_ids.add(rid)
            self._queue_depth += 1
            self._metrics["requests_total"] += 1.0
            self._ensure_model_metrics(model_id)
            rec = RequestRecord(request_id=rid, model_id=model_id, status="running", started_at=time.time())
            self._requests.append(rec)

        try:
            slot = self.models.acquire(model_id)
        except BaseException as exc:
            with self._lock:
                self._reserved_request_ids.discard(rid)
                self._queue_depth = max(0, self._queue_depth - 1)
                self._metrics["requests_failed"] += 1.0
                self._record_outcome(model_id, "failed")
                rec.status = "failed"
                rec.error = str(exc)
                rec.finished_at = time.time()
            raise

        cancel_token: Any | None = None
        try:
            from tensortorrent.native import require_native

            cancel_token = require_native().NativeCancelToken()
        except Exception:  # noqa: BLE001 - native optional for test doubles / CPU CI
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
                if cancel_token is not None:
                    if isinstance(inputs, tuple):
                        return module.forward_with_cancel_token(cancel_token, *inputs)
                    return module.forward_with_cancel_token(cancel_token, inputs)
                if isinstance(inputs, tuple):
                    return module(*inputs)
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
                self._record_outcome(model_id, "failed")
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
                self._record_latency(model_id, elapsed)
                self._record_outcome(model_id, "success")
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
            # Per-request cancel only — never module.request_cancel on timeout.
            if cancel_token is not None:
                cancel_token.cancel()
            grace = float(self.config.cancellation_grace_s)
            if grace > 0:
                with contextlib.suppress(BaseException):
                    future.result(timeout=grace)
            with self._lock:
                self._metrics["requests_cancelled"] += 1.0
                self._metrics["timeout_total"] += 1.0
                self._record_outcome(model_id, "timeout")
                rec.status = "cancelled"
                rec.error = f"timed out after {timeout}s"
                rec.finished_at = time.time()
            raise ExecutionCancelled(f"request {rid} timed out after {timeout}s") from exc
        except ExecutionCancelled as exc:
            with self._lock:
                self._metrics["requests_cancelled"] += 1.0
                self._record_outcome(model_id, "cancelled")
                rec.status = "cancelled"
                rec.error = str(exc)
                rec.finished_at = time.time()
            raise
        except Exception as exc:
            with self._lock:
                self._metrics["requests_failed"] += 1.0
                self._record_outcome(model_id, "failed")
                rec.status = "failed"
                rec.error = str(exc)
                rec.finished_at = time.time()
            logger.exception("request %s failed", rid)
            raise
