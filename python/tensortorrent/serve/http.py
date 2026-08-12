"""Stdlib HTTP front for InferenceService.

JSON bodies only (nested float/int lists → tensors). No extra deps.
Binary transports can wait for a later gRPC/Arrow layer.

Env knobs:
- TT_HTTP_MAX_CONNECTIONS (default 128) — saturated → 503
- TT_HTTP_BACKLOG (default 64)
- TT_HTTP_MAX_BODY_BYTES (default 32 MiB)
- TT_HTTP_MAX_RESPONSE_BYTES (default 128 MiB)
- TT_HTTP_SOCKET_TIMEOUT_S (default 30 s)
- TT_SERVE_AUTH_TOKEN — Bearer required on non-health endpoints when set
"""

from __future__ import annotations

import contextlib
import hmac
import json
import logging
import math
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import torch

from tensortorrent.errors import ExecutionCancelled, TensorTorrentError
from tensortorrent.serve.app import InferenceService
from tensortorrent.serve.logging_setup import request_id_var
from tensortorrent.serve.service_config import (
    DEFAULT_HTTP_BACKLOG,
    DEFAULT_HTTP_MAX_BODY_BYTES,
    DEFAULT_HTTP_MAX_CONNECTIONS,
    DEFAULT_HTTP_MAX_RESPONSE_BYTES,
    DEFAULT_HTTP_SHUTDOWN_TIMEOUT_S,
    DEFAULT_HTTP_SOCKET_TIMEOUT_S,
    MAX_HTTP_TENSOR_RANK,
    MIN_HTTP_SOCKET_TIMEOUT_S,
    _validate_http_backlog,
    _validate_http_max_connections,
    _validate_http_max_response_bytes,
    env_float,
    env_int,
)

logger = logging.getLogger("tensortorrent.server.http")

# Fail fast at import / server startup.
_MAX_BODY_BYTES = env_int("TT_HTTP_MAX_BODY_BYTES", DEFAULT_HTTP_MAX_BODY_BYTES)
if _MAX_BODY_BYTES < 1:
    raise RuntimeError(f"TT_HTTP_MAX_BODY_BYTES must be >= 1, got {_MAX_BODY_BYTES}")

_SOCKET_TIMEOUT_S = env_float("TT_HTTP_SOCKET_TIMEOUT_S", DEFAULT_HTTP_SOCKET_TIMEOUT_S)

_MAX_CONNECTIONS: int = _validate_http_max_connections(env_int("TT_HTTP_MAX_CONNECTIONS", DEFAULT_HTTP_MAX_CONNECTIONS))
_BACKLOG: int = _validate_http_backlog(env_int("TT_HTTP_BACKLOG", DEFAULT_HTTP_BACKLOG))
_MAX_RESPONSE_BYTES: int = _validate_http_max_response_bytes(
    env_int("TT_HTTP_MAX_RESPONSE_BYTES", DEFAULT_HTTP_MAX_RESPONSE_BYTES)
)

# Auth token — never log the value.
_AUTH_TOKEN: str | None = os.environ.get("TT_SERVE_AUTH_TOKEN") or None

_ALLOWED_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
    "uint8": torch.uint8,
}

# Endpoints that skip Bearer auth (health probes must always be reachable).
_OPEN_PATHS = frozenset({"/health", "/v1/health", "/ready", "/readiness", "/v1/ready"})


def _json_to_tensor(value: Any) -> Any:
    if isinstance(value, dict) and "data" in value:
        data = value["data"]
        dtype_name = str(value.get("dtype", "float32"))
        if dtype_name.startswith("torch."):
            dtype_name = dtype_name.removeprefix("torch.")
        dtype = _ALLOWED_DTYPES.get(dtype_name)
        if dtype is None:
            raise TensorTorrentError(f"unsupported dtype: {dtype_name}")
        try:
            t = torch.tensor(data, dtype=dtype)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TensorTorrentError(f"invalid tensor data: {exc}") from exc
        shape = value.get("shape")
        if shape is not None:
            if not isinstance(shape, list):
                raise TensorTorrentError("shape must be a JSON array of integers")
            if len(shape) > MAX_HTTP_TENSOR_RANK:
                raise TensorTorrentError(f"tensor rank exceeds maximum {MAX_HTTP_TENSOR_RANK}")
            if any(isinstance(dim, bool) or not isinstance(dim, int) for dim in shape):
                raise TensorTorrentError("shape dims must be integers")
            dims = list(shape)
            if any(d < 0 for d in dims):
                raise TensorTorrentError("shape dims must be non-negative")
            numel = 1
            for d in dims:
                numel *= d
            if int(t.numel()) != numel:
                raise TensorTorrentError(f"shape {dims} expects {numel} elements, got {int(t.numel())}")
            t = t.reshape(dims)
        return t
    if isinstance(value, list):
        # A rectangular numeric tree is one tensor, including matrices and
        # higher-rank arrays. Multiple model arguments must use explicit tensor
        # descriptors (a list containing ``{"data": ...}`` objects), avoiding
        # the old ambiguity where a 2-D matrix became several positional args.
        if not any(isinstance(item, dict) for item in value):
            try:
                return torch.tensor(value, dtype=torch.float32)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TensorTorrentError(f"invalid numeric tensor input: {exc}") from exc
        return [_json_to_tensor(item) for item in value]
    return value


def _tensor_to_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "data": value.detach().cpu().tolist(),
        }
    if isinstance(value, (list, tuple)):
        return [_tensor_to_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _tensor_to_json(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _require_json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TensorTorrentError("request body must be a JSON object")
    return value


def _decode_json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TensorTorrentError(f"invalid JSON: {exc}") from exc


def _walk_output_estimate(value: Any, total_ref: list[int]) -> None:
    if isinstance(value, torch.Tensor):
        total_ref[0] += int(value.nbytes) * 3
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk_output_estimate(item, total_ref)
    elif isinstance(value, dict):
        for v in value.values():
            _walk_output_estimate(v, total_ref)


def _estimate_response_bytes(output: Any) -> int:
    """Estimate serialised response size without actually serialising.

    Walks tensors and sums nbytes * 3 (JSON ASCII overhead per element) plus a
    fixed overhead for JSON framing and metadata.
    """
    total_ref: list[int] = [1024]  # fixed JSON framing overhead
    _walk_output_estimate(output, total_ref)
    return total_ref[0]


class _HttpRequestError(Exception):
    """Raised by _read_json to signal a non-TensorTorrentError HTTP error."""

    def __init__(self, body: bytes, *, code: int = 400) -> None:
        self.body = body
        self.code = code
        super().__init__()


class _Handler(BaseHTTPRequestHandler):
    service: InferenceService
    max_body_bytes: int = _MAX_BODY_BYTES
    socket_timeout_s: float = _SOCKET_TIMEOUT_S
    max_response_bytes: int = _MAX_RESPONSE_BYTES
    auth_token: str | None = _AUTH_TOKEN

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(max(MIN_HTTP_SOCKET_TIMEOUT_S, float(self.socket_timeout_s)))

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 — stdlib API
        logger.info("%s - %s", self.address_string(), fmt % args)

    # Force Connection: close — no keep-alive.
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _send_json(self, code: int, payload: Any, *, request_id: str | None = None) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        if request_id is not None:
            self.send_header("X-Request-ID", request_id)
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def _send_401(self) -> None:
        body = json.dumps({"error": "missing or invalid Authorization header"}).encode("utf-8")
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.send_header("WWW-Authenticate", "Bearer")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _check_auth_full(self, path: str) -> bool:
        """True if ok; sends 401 + WWW-Authenticate and returns False otherwise."""
        token = self.auth_token
        if token is None:
            return True
        if path in _OPEN_PATHS:
            return True
        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            self._send_401()
            return False
        provided = auth_header[len("Bearer ") :]
        if not hmac.compare_digest(provided.encode(), token.encode()):
            self._send_401()
            return False
        return True

    def _read_json(self) -> Any:
        # Transfer-Encoding is not supported (only Content-Length).
        if self.headers.get("Transfer-Encoding"):
            raise _HttpRequestError(
                json.dumps({"error": "Transfer-Encoding not supported; use Content-Length"}).encode("utf-8"),
                code=400,
            )
        length_hdr = self.headers.get("Content-Length")
        if length_hdr is None or length_hdr.strip() == "" or length_hdr.strip() == "0":
            # POST with missing or zero Content-Length → 411 Length Required.
            raise _HttpRequestError(
                json.dumps({"error": "Content-Length required and must be > 0"}).encode("utf-8"),
                code=411,
            )
        try:
            length = int(length_hdr)
        except ValueError as exc:
            raise TensorTorrentError("invalid Content-Length") from exc
        if length < 0:
            raise TensorTorrentError("invalid Content-Length")
        if length > self.max_body_bytes:
            raise _HttpRequestError(
                json.dumps({"error": f"request body too large: {length} bytes (max {self.max_body_bytes})"}).encode(
                    "utf-8"
                ),
                code=413,
            )
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise TensorTorrentError(f"incomplete request body: expected {length} bytes, got {len(raw)}")
        if not raw:
            return {}
        return _decode_json(raw)

    def _check_response_size(self, output: Any) -> None:
        estimate = _estimate_response_bytes(output)
        cap = self.max_response_bytes
        if estimate > cap:
            raise TensorTorrentError(
                f"estimated response size {estimate} bytes exceeds TT_HTTP_MAX_RESPONSE_BYTES={cap}"
            )

    def _send_raw_error(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler
        path = urlparse(self.path).path.rstrip("/") or "/"
        if not self._check_auth_full(path):
            return
        if path in ("/health", "/v1/health"):
            self._send_json(200, self.service.health())
            return
        if path in ("/ready", "/readiness", "/v1/ready"):
            ready = self.service.readiness()
            self._send_json(200 if ready.get("ready") else 503, ready)
            return
        if path in ("/metrics", "/v1/metrics"):
            text = self.service.metrics_prometheus().encode("utf-8")
            self._send(200, text, "text/plain; version=0.0.4; charset=utf-8")
            return
        self._send_json(404, {"error": "not found", "path": path})

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler
        path = urlparse(self.path).path.rstrip("/") or "/"
        if not self._check_auth_full(path):
            return
        if path in ("/cancel", "/v1/cancel"):
            self._handle_cancel()
            return
        if path not in ("/infer", "/v1/infer"):
            self._send_json(404, {"error": "not found", "path": path})
            return
        self._handle_infer()

    def _handle_cancel(self) -> None:
        rid: str | None = None
        try:
            body = _require_json_object(self._read_json())
            request_id = body.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise TensorTorrentError("request_id must be a non-empty string")
            rid = request_id
            cancelled = self.service.cancel(request_id)
            self._send_json(
                200 if cancelled else 404,
                {"request_id": request_id, "cancelled": cancelled},
                request_id=request_id,
            )
        except _HttpRequestError as exc:
            self._send_raw_error(exc.code, exc.body)
        except TensorTorrentError as exc:
            self._send_json(400, {"error": str(exc)}, request_id=rid)
        except Exception:  # noqa: BLE001 — HTTP boundary
            logger.exception("cancel failed")
            self._send_json(500, {"error": "internal error"}, request_id=rid)

    def _handle_infer(self) -> None:
        rid: str | None = None
        try:
            body = _require_json_object(self._read_json())
            model_id = body.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                raise TensorTorrentError("model_id must be a non-empty string")
            inputs = body.get("inputs")
            if inputs is None:
                raise TensorTorrentError("missing inputs")
            raw_request_id = body.get("request_id")
            if raw_request_id is not None and (not isinstance(raw_request_id, str) or not raw_request_id):
                raise TensorTorrentError("request_id must be a non-empty string when provided")
            rid = raw_request_id

            # Wire the request_id context variable for structured logging.
            token = request_id_var.set(rid or "")
            try:
                converted = _json_to_tensor(inputs)
                if isinstance(converted, list):
                    args: Any = tuple(converted)
                else:
                    args = converted
                out = self.service.infer(
                    model_id,
                    args,
                    request_id=rid,
                    timeout_s=body.get("timeout_s"),
                )
            finally:
                request_id_var.reset(token)

            rid = out["request_id"]

            # Response-size guard before serialisation.
            self._check_response_size(out["output"])

            payload = {
                "request_id": out["request_id"],
                "model_id": out["model_id"],
                "version": out["version"],
                "latency_s": out["latency_s"],
                "output": _tensor_to_json(out["output"]),
            }
            self._send_json(200, payload, request_id=rid)
        except _HttpRequestError as exc:
            self._send_raw_error(exc.code, exc.body)
        except KeyError as exc:
            self._send_json(400, {"error": f"missing field: {exc}"}, request_id=rid)
        except ExecutionCancelled as exc:
            self._send_json(408, {"error": str(exc)}, request_id=rid)
        except TensorTorrentError as exc:
            # Distinguish payload-too-large for load balancers.
            msg = str(exc)
            if "estimated response size" in msg:
                self._send_json(500, {"error": msg}, request_id=rid)
            else:
                code = 413 if "too large" in msg else 400
                self._send_json(code, {"error": msg}, request_id=rid)
        except Exception:  # noqa: BLE001 — HTTP boundary
            logger.exception("infer failed")
            self._send_json(500, {"error": "internal error"}, request_id=rid)


class _ProductionThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a connection cap and configurable listen backlog.

    Connection cap implementation
    ------------------------------
    ``process_request`` is the server-level method called in the main accept
    loop **before** a worker thread is spawned.  We do a non-blocking acquire
    of ``_conn_sem`` here: if the cap is saturated we write a raw HTTP 503 with
    ``Retry-After: 1`` and close the socket WITHOUT handing it to the thread
    pool.  ``shutdown_request`` (called in every finish path, including errors)
    releases the semaphore exactly once.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type,
        *,
        max_connections: int = _MAX_CONNECTIONS,
        backlog: int = _BACKLOG,
    ) -> None:
        self._conn_sem: threading.Semaphore = threading.Semaphore(max_connections)
        self.request_queue_size = backlog
        super().__init__(server_address, RequestHandlerClass)

    def process_request(  # type: ignore[override]
        self, request: socket.socket, client_address: Any
    ) -> None:
        """Non-blocking connection-cap acquire then thread dispatch."""
        if not self._conn_sem.acquire(blocking=False):
            # Connection cap saturated — write minimal HTTP 503 and close.
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Content-Length: 0\r\n"
                    b"Connection: close\r\n"
                    b"Retry-After: 1\r\n"
                    b"\r\n"
                )
            except OSError:
                pass
            finally:
                with contextlib.suppress(OSError):
                    request.close()
            return
        # Semaphore acquired — proceed to spawn a worker thread.
        super().process_request(request, client_address)

    def shutdown_request(self, request: Any) -> None:
        """Release the connection-cap semaphore (exactly once) after the request."""
        try:
            super().shutdown_request(request)
        finally:
            self._conn_sem.release()


class HttpServer:
    """Threading HTTP server bound to an InferenceService."""

    def __init__(
        self,
        service: InferenceService,
        host: str = "127.0.0.1",
        port: int = 8080,
        *,
        max_body_bytes: int | None = None,
        socket_timeout_s: float | None = None,
        max_connections: int | None = None,
        backlog: int | None = None,
        max_response_bytes: int | None = None,
        auth_token: str | None = None,
    ) -> None:
        if not isinstance(host, str) or not host:
            raise TypeError("host must be a non-empty string")
        if isinstance(port, bool) or not isinstance(port, int):
            raise TypeError("port must be an integer")
        if not 0 <= port <= 65_535:
            raise ValueError("port must be between 0 and 65535")
        if max_body_bytes is not None and (isinstance(max_body_bytes, bool) or not isinstance(max_body_bytes, int)):
            raise TypeError("max_body_bytes must be an integer or None")
        if socket_timeout_s is not None and (
            isinstance(socket_timeout_s, bool) or not isinstance(socket_timeout_s, (int, float))
        ):
            raise TypeError("socket_timeout_s must be numeric or None")
        self.service = service
        self.host = host
        self.port = port
        self.max_body_bytes = _MAX_BODY_BYTES if max_body_bytes is None else max_body_bytes
        self.socket_timeout_s = _SOCKET_TIMEOUT_S if socket_timeout_s is None else float(socket_timeout_s)
        self.max_connections = _MAX_CONNECTIONS if max_connections is None else max_connections
        self.backlog = _BACKLOG if backlog is None else backlog
        self.max_response_bytes = _MAX_RESPONSE_BYTES if max_response_bytes is None else max_response_bytes
        self.auth_token = _AUTH_TOKEN if auth_token is None else auth_token
        if self.max_body_bytes < 1:
            raise ValueError("max_body_bytes must be >= 1")
        if not math.isfinite(self.socket_timeout_s) or self.socket_timeout_s <= 0:
            raise ValueError("socket_timeout_s must be > 0 and finite")
        if self.max_connections < 1:
            raise ValueError("max_connections must be >= 1")
        if self.backlog < 1:
            raise ValueError("backlog must be >= 1")
        if self.max_response_bytes < 1:
            raise ValueError("max_response_bytes must be >= 1")
        self._httpd: _ProductionThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, *, background: bool = True) -> None:
        if self._httpd is not None:
            raise RuntimeError("HttpServer is already running; call stop() before start()")
        # Build a per-server semaphore via the bound handler class.
        conn_sem = threading.Semaphore(self.max_connections)
        max_body = self.max_body_bytes
        max_resp = self.max_response_bytes
        tok = self.auth_token
        sock_timeout = self.socket_timeout_s
        handler = type(
            "BoundHandler",
            (_Handler,),
            {
                "service": self.service,
                "max_body_bytes": max_body,
                "socket_timeout_s": sock_timeout,
                "max_response_bytes": max_resp,
                "auth_token": tok,
            },
        )
        self._httpd = _ProductionThreadingHTTPServer(
            (self.host, self.port),
            handler,
            max_connections=self.max_connections,
            backlog=self.backlog,
        )
        # Override the server's semaphore with the per-instance one so that
        # different HttpServer instances don't share a semaphore.
        self._httpd._conn_sem = conn_sem
        # Ephemeral port support: port 0 → real bound port.
        # server_address may be (host, port) or an IPv6 4-tuple; index, don't unpack.
        addr = self._httpd.server_address
        bound_host = addr[0]
        if isinstance(bound_host, (bytes, bytearray)):
            bound_host = bound_host.decode()
        self.host = str(bound_host)
        self.port = int(addr[1])
        if background:
            self._thread = threading.Thread(target=self._httpd.serve_forever, name="tt-http", daemon=True)
            self._thread.start()
        else:
            self._httpd.serve_forever()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=DEFAULT_HTTP_SHUTDOWN_TIMEOUT_S)
            self._thread = None
