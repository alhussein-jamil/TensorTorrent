"""Structured logging setup for the TensorTorrent serving layer.

Environment variables
---------------------
TT_LOG_LEVEL   : Logging level name (default ``INFO``). Valid: DEBUG, INFO, WARNING, ERROR, CRITICAL.
TT_LOG_FORMAT  : ``text`` (default) or ``json``.
"""

from __future__ import annotations

import datetime
import json
import logging
import logging.handlers
import os
from contextvars import ContextVar
from typing import Any

# Context variable that carries the request-id for the current thread/task.
request_id_var: ContextVar[str | None] = ContextVar("request_id_var", default=None)

_VALID_LEVELS = frozenset(("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
_VALID_FORMATS = frozenset(("text", "json"))

_DEFAULT_LEVEL = "INFO"
_DEFAULT_FORMAT = "text"


def _validate_level(raw: str) -> int:
    name = raw.strip().upper()
    if name not in _VALID_LEVELS:
        raise RuntimeError(f"TT_LOG_LEVEL must be one of {sorted(_VALID_LEVELS)}, got {raw!r}")
    return int(getattr(logging, name))


def _validate_format(raw: str) -> str:
    name = raw.strip().lower()
    if name not in _VALID_FORMATS:
        raise RuntimeError(f"TT_LOG_FORMAT must be 'text' or 'json', got {raw!r}")
    return name


class _RequestIdFilter(logging.Filter):
    """Inject the current request_id context variable into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        rid = request_id_var.get(None)
        record.__dict__["request_id"] = rid
        return True


class _JsonFormatter(logging.Formatter):
    """Newline-delimited JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        ts = (
            datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid: str | None = getattr(record, "request_id", None)
        if rid is not None:
            payload["request_id"] = rid
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Forward any extra fields set by callers via `extra=`
        _template = logging.LogRecord("", 0, "", 0, "", (), None)
        skip = frozenset(_template.__dict__.keys()) | frozenset(("msg", "request_id", "message", "asctime"))
        for key, value in record.__dict__.items():
            if key not in skip and not key.startswith("_"):
                try:
                    json.dumps(value)  # only include JSON-serialisable extras
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = repr(value)
        return json.dumps(payload)


_TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
_TEXT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(
    level: str | None = None,
    fmt: str | None = None,
) -> None:
    """Configure root logger for the TensorTorrent serving layer.

    Reads from environment if parameters are not supplied:
    - ``TT_LOG_LEVEL``  — validated against {DEBUG,INFO,WARNING,ERROR,CRITICAL}
    - ``TT_LOG_FORMAT`` — ``text`` (default) or ``json``

    The function is idempotent and safe to call multiple times; subsequent
    calls reconfigure the root logger in-place.
    """
    raw_level = level if level is not None else os.environ.get("TT_LOG_LEVEL", _DEFAULT_LEVEL)
    raw_fmt = fmt if fmt is not None else os.environ.get("TT_LOG_FORMAT", _DEFAULT_FORMAT)

    numeric_level = _validate_level(raw_level)
    fmt_name = _validate_format(raw_fmt)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Ensure exactly one StreamHandler (stdout-like) is attached.
    # Remove stale handlers of the same type to avoid duplicates on re-call.
    for handler in list(root.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler()
    handler.setLevel(numeric_level)
    handler.addFilter(_RequestIdFilter())

    if fmt_name == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(fmt=_TEXT_FORMAT, datefmt=_TEXT_DATE_FORMAT))

    root.addHandler(handler)
