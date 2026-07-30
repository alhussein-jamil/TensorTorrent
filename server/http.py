"""Minimal stdlib HTTP front for InferenceService.

No extra dependency. JSON bodies only — float/int nested lists become tensors.
Binary tensor transports belong in a later gRPC/Arrow layer.

Auth is out of scope: bind to loopback or put behind a trusted reverse proxy.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import torch

from server.app import InferenceService
from streamcompiler.errors import ExecutionCancelled, StreamCompilerError

logger = logging.getLogger("streamcompiler.server.http")

# Default 32 MiB — enough for small JSON tensors; raise via env for larger demos.
_MAX_BODY_BYTES = int(os.environ.get("SC_HTTP_MAX_BODY_BYTES", str(32 * 1024 * 1024)))
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


def _json_to_tensor(value: Any) -> Any:
    if isinstance(value, dict) and "data" in value:
        data = value["data"]
        dtype_name = str(value.get("dtype", "float32"))
        if dtype_name.startswith("torch."):
            dtype_name = dtype_name.removeprefix("torch.")
        dtype = _ALLOWED_DTYPES.get(dtype_name)
        if dtype is None:
            raise StreamCompilerError(f"unsupported dtype: {dtype_name}")
        t = torch.tensor(data, dtype=dtype)
        shape = value.get("shape")
        if shape is not None:
            dims = [int(x) for x in shape]
            if any(d < 0 for d in dims):
                raise StreamCompilerError("shape dims must be non-negative")
            numel = 1
            for d in dims:
                numel *= d
            if int(t.numel()) != numel:
                raise StreamCompilerError(f"shape {dims} expects {numel} elements, got {int(t.numel())}")
            t = t.reshape(dims)
        return t
    if isinstance(value, list):
        if value and isinstance(value[0], (list, dict)):
            return [_json_to_tensor(v) for v in value]
        return torch.tensor(value, dtype=torch.float32)
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


def _parse_timeout_s(raw: Any, *, default: float) -> float:
    if raw is None:
        return default
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise StreamCompilerError("timeout_s must be a number") from exc
    if timeout <= 0:
        raise StreamCompilerError("timeout_s must be > 0")
    # Hard ceiling avoids runaway cooperative-cancel threads.
    return min(timeout, 3600.0)


class _Handler(BaseHTTPRequestHandler):
    service: InferenceService
    max_body_bytes: int = _MAX_BODY_BYTES

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 — stdlib API
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: Any) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self._send(code, raw, "application/json")

    def _read_json(self) -> Any:
        length_hdr = self.headers.get("Content-Length", "0")
        try:
            length = int(length_hdr)
        except ValueError as exc:
            raise StreamCompilerError("invalid Content-Length") from exc
        if length < 0:
            raise StreamCompilerError("invalid Content-Length")
        if length > self.max_body_bytes:
            raise StreamCompilerError(f"request body too large: {length} bytes (max {self.max_body_bytes})")
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise StreamCompilerError(f"invalid JSON: {exc}") from exc

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler
        path = urlparse(self.path).path.rstrip("/") or "/"
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
        if path not in ("/infer", "/v1/infer"):
            self._send_json(404, {"error": "not found", "path": path})
            return
        try:
            body = self._read_json()
            model_id = str(body["model_id"])
            if not model_id:
                raise StreamCompilerError("model_id must be non-empty")
            inputs = body.get("inputs")
            if inputs is None:
                raise StreamCompilerError("missing inputs")
            converted = _json_to_tensor(inputs)
            if isinstance(converted, list):
                args: Any = tuple(converted)
            else:
                args = converted
            timeout_s = _parse_timeout_s(body.get("timeout_s"), default=self.service.config.default_timeout_s)
            out = self.service.infer(
                model_id,
                args,
                request_id=body.get("request_id"),
                timeout_s=timeout_s,
            )
            payload = {
                "request_id": out["request_id"],
                "model_id": out["model_id"],
                "version": out["version"],
                "latency_s": out["latency_s"],
                "output": _tensor_to_json(out["output"]),
            }
            self._send_json(200, payload)
        except KeyError as exc:
            self._send_json(400, {"error": f"missing field: {exc}"})
        except ExecutionCancelled as exc:
            self._send_json(408, {"error": str(exc)})
        except StreamCompilerError as exc:
            # Distinguish payload-too-large for load balancers.
            msg = str(exc)
            code = 413 if "too large" in msg else 400
            self._send_json(code, {"error": msg})
        except Exception:  # noqa: BLE001 — HTTP boundary
            logger.exception("infer failed")
            self._send_json(500, {"error": "internal error"})


class HttpServer:
    """Threading HTTP server bound to an InferenceService."""

    def __init__(
        self,
        service: InferenceService,
        host: str = "127.0.0.1",
        port: int = 8080,
        *,
        max_body_bytes: int | None = None,
    ) -> None:
        self.service = service
        self.host = host
        self.port = int(port)
        self.max_body_bytes = _MAX_BODY_BYTES if max_body_bytes is None else int(max_body_bytes)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, *, background: bool = True) -> None:
        max_body = self.max_body_bytes
        handler = type(
            "BoundHandler",
            (_Handler,),
            {"service": self.service, "max_body_bytes": max_body},
        )
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        # Ephemeral port support: port 0 → real bound port.
        # server_address may be (host, port) or an IPv6 4-tuple; index, don't unpack.
        addr = self._httpd.server_address
        bound_host = addr[0]
        if isinstance(bound_host, (bytes, bytearray)):
            bound_host = bound_host.decode()
        self.host = str(bound_host)
        self.port = int(addr[1])
        if background:
            self._thread = threading.Thread(target=self._httpd.serve_forever, name="sc-http", daemon=True)
            self._thread.start()
        else:
            self._httpd.serve_forever()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
