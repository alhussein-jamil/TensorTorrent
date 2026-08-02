"""Validated operational limits for the inference service."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

DEFAULT_MAX_QUEUE_DEPTH = 64
DEFAULT_REQUEST_TIMEOUT_S = 30.0
# Cooperative cancellation cannot safely leave request threads waiting forever.
DEFAULT_MAX_REQUEST_TIMEOUT_S = 60.0 * 60.0
DEFAULT_MODEL_CONCURRENCY = 8
DEFAULT_WORKER_THREADS = 0
DEFAULT_CANCELLATION_GRACE_S = 1.0
DEFAULT_REQUEST_HISTORY_SIZE = 1024
DEFAULT_MODEL_DRAIN_TIMEOUT_S = 5.0

DEFAULT_HTTP_MAX_BODY_BYTES = 32 * 1024 * 1024
DEFAULT_HTTP_SOCKET_TIMEOUT_S = 30.0
MIN_HTTP_SOCKET_TIMEOUT_S = 0.1
DEFAULT_HTTP_SHUTDOWN_TIMEOUT_S = 5.0
MAX_HTTP_TENSOR_RANK = 64


def env_int(name: str, default: int) -> int:
    """Read one strict integer environment setting."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw, 10)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def env_float(name: str, default: float) -> float:
    """Read one finite floating-point environment setting."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be finite, got {value}")
    return value


@dataclass(frozen=True)
class ServiceConfig:
    """Service limits. Defaults are conservative and every field is tunable."""

    max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH
    default_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    max_request_timeout_s: float = DEFAULT_MAX_REQUEST_TIMEOUT_S
    default_concurrency: int = DEFAULT_MODEL_CONCURRENCY
    worker_threads: int = DEFAULT_WORKER_THREADS
    cancellation_grace_s: float = DEFAULT_CANCELLATION_GRACE_S
    request_history_size: int = DEFAULT_REQUEST_HISTORY_SIZE

    def __post_init__(self) -> None:
        for name in (
            "max_queue_depth",
            "default_concurrency",
            "worker_threads",
            "request_history_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        for name in ("default_timeout_s", "max_request_timeout_s", "cancellation_grace_s"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.max_queue_depth < 1:
            raise ValueError("max_queue_depth must be >= 1")
        if self.default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be > 0")
        if self.max_request_timeout_s <= 0:
            raise ValueError("max_request_timeout_s must be > 0")
        if self.default_timeout_s > self.max_request_timeout_s:
            raise ValueError("default_timeout_s must be <= max_request_timeout_s")
        if self.default_concurrency < 1:
            raise ValueError("default_concurrency must be >= 1")
        if self.worker_threads < 0:
            raise ValueError("worker_threads must be >= 0")
        if self.cancellation_grace_s < 0:
            raise ValueError("cancellation_grace_s must be >= 0")
        if self.request_history_size < 1:
            raise ValueError("request_history_size must be >= 1")

    @classmethod
    def from_env(cls) -> ServiceConfig:
        """Load strict production overrides from ``SC_SERVE_*`` variables."""
        return cls(
            max_queue_depth=env_int("SC_SERVE_MAX_QUEUE_DEPTH", DEFAULT_MAX_QUEUE_DEPTH),
            default_timeout_s=env_float("SC_SERVE_DEFAULT_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S),
            max_request_timeout_s=env_float("SC_SERVE_MAX_REQUEST_TIMEOUT_S", DEFAULT_MAX_REQUEST_TIMEOUT_S),
            default_concurrency=env_int("SC_SERVE_DEFAULT_CONCURRENCY", DEFAULT_MODEL_CONCURRENCY),
            worker_threads=env_int("SC_SERVE_WORKER_THREADS", DEFAULT_WORKER_THREADS),
            cancellation_grace_s=env_float("SC_SERVE_CANCELLATION_GRACE_S", DEFAULT_CANCELLATION_GRACE_S),
            request_history_size=env_int("SC_SERVE_REQUEST_HISTORY_SIZE", DEFAULT_REQUEST_HISTORY_SIZE),
        )
