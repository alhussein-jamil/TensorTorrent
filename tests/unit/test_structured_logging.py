"""Tests for observability/logging.py structured logging setup."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from tensortorrent.observability.logging import (
    _JsonFormatter,
    _RequestIdFilter,
    request_id_var,
    setup_logging,
)

# ---------------------------------------------------------------------------
# JSON format produces parseable JSON lines with ts/level/logger/msg
# ---------------------------------------------------------------------------


def _capture_json_record(msg: str = "hello", level: int = logging.INFO) -> dict:
    """Emit one log record through the JSON formatter and parse the result."""
    formatter = _JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=level,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
    # Inject the request_id filter
    filt = _RequestIdFilter()
    filt.filter(record)

    line = formatter.format(record)
    return json.loads(line)


def test_json_format_produces_required_fields() -> None:
    """JSON output must contain ts, level, logger, msg."""
    data = _capture_json_record("test message", level=logging.WARNING)
    assert "ts" in data, f"missing ts: {data}"
    assert "level" in data, f"missing level: {data}"
    assert "logger" in data, f"missing logger: {data}"
    assert "msg" in data, f"missing msg: {data}"
    assert data["msg"] == "test message"
    assert data["level"] == "WARNING"
    assert data["logger"] == "test.logger"


def test_json_format_ts_is_iso_utc() -> None:
    """ts field must end with 'Z' (UTC ISO-8601 with milliseconds)."""
    data = _capture_json_record()
    ts = data["ts"]
    assert ts.endswith("Z"), f"expected UTC Z suffix: {ts!r}"
    # Must be parseable
    import datetime

    datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def test_json_format_is_parseable_json() -> None:
    """Output must be valid JSON (single line)."""
    formatter = _JsonFormatter()
    record = logging.LogRecord("t", logging.INFO, "", 0, "hi", (), None)
    _RequestIdFilter().filter(record)
    line = formatter.format(record)
    parsed = json.loads(line)
    assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# TT_LOG_LEVEL invalid → RuntimeError
# ---------------------------------------------------------------------------


def test_invalid_log_level_raises_runtime_error(monkeypatch: Any) -> None:
    """setup_logging with an invalid level must raise RuntimeError."""
    from tensortorrent.observability.logging import _validate_level

    with pytest.raises(RuntimeError, match="TT_LOG_LEVEL"):
        _validate_level("NONSENSE")


def test_invalid_log_format_raises_runtime_error() -> None:
    """setup_logging with an invalid format must raise RuntimeError."""
    from tensortorrent.observability.logging import _validate_format

    with pytest.raises(RuntimeError, match="TT_LOG_FORMAT"):
        _validate_format("yaml")


def test_setup_logging_env_invalid_level(monkeypatch: Any) -> None:
    """setup_logging reads TT_LOG_LEVEL and raises on invalid value."""
    monkeypatch.setenv("TT_LOG_LEVEL", "GARBAGE")
    with pytest.raises(RuntimeError, match="TT_LOG_LEVEL"):
        setup_logging()


# ---------------------------------------------------------------------------
# request_id_var injection appears in records
# ---------------------------------------------------------------------------


def test_request_id_injected_when_set() -> None:
    """When request_id_var is set, JSON output contains request_id field."""
    token = request_id_var.set("req-abc-123")
    try:
        data = _capture_json_record("with request id")
        assert data.get("request_id") == "req-abc-123", f"expected request_id in record: {data}"
    finally:
        request_id_var.reset(token)


def test_request_id_absent_when_not_set() -> None:
    """When request_id_var is not set (default=None), request_id absent from JSON."""
    # Ensure no token is set
    token = request_id_var.set(None)
    try:
        data = _capture_json_record("no request id")
        # request_id key should not appear when None
        assert "request_id" not in data, f"unexpected request_id in record: {data}"
    finally:
        request_id_var.reset(token)


# ---------------------------------------------------------------------------
# setup_logging is idempotent (no duplicate handlers)
# ---------------------------------------------------------------------------


def test_setup_logging_idempotent() -> None:
    """Calling setup_logging twice must not add duplicate handlers."""
    root = logging.getLogger()
    len(root.handlers)
    setup_logging(level="WARNING", fmt="text")
    setup_logging(level="WARNING", fmt="text")
    len(root.handlers)
    # At most one StreamHandler added
    stream_handlers = [
        h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert len(stream_handlers) == 1, f"duplicate StreamHandlers: {stream_handlers}"


def test_setup_logging_json_format() -> None:
    """setup_logging with json format must attach a JSON formatter."""
    setup_logging(level="INFO", fmt="json")
    root = logging.getLogger()
    stream_handlers = [
        h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert stream_handlers
    formatter = stream_handlers[0].formatter
    assert isinstance(formatter, _JsonFormatter), f"expected JSON formatter, got {formatter}"
