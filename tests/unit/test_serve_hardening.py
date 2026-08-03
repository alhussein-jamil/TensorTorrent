"""HTTP server hardening tests: 411, 400 chunked, auth, metrics, connection cap."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any

import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.serve import InferenceService, ServiceConfig
from tensortorrent.serve.http import HttpServer

# ---------------------------------------------------------------------------
# Helper fixture: running HttpServer on 127.0.0.1:0
# ---------------------------------------------------------------------------


class _FakeModule:
    """Minimal module that echoes inputs."""

    def __call__(self, x: Any) -> Any:
        return x

    def close(self) -> None:
        pass


def _make_server(svc: InferenceService, **kwargs: Any) -> HttpServer:
    http = HttpServer(svc, host="127.0.0.1", port=0, **kwargs)
    http.start(background=True)
    return http


# ---------------------------------------------------------------------------
# 411 Length Required for missing Content-Length
# ---------------------------------------------------------------------------


def test_post_without_content_length_returns_411() -> None:
    svc = InferenceService()
    svc.start()
    http = _make_server(svc)
    try:
        # Send a raw HTTP request without Content-Length header
        raw = (
            f"POST /v1/infer HTTP/1.1\r\n"
            f"Host: {http.host}:{http.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Connection: close\r\n\r\n"
        ).encode()
        with socket.create_connection((http.host, http.port), timeout=5) as sock:
            sock.sendall(raw)
            sock.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        first_line = data.split(b"\r\n", 1)[0]
        assert b"411" in first_line, f"Expected 411, got: {first_line}"
    finally:
        http.stop()
        svc.stop()


# ---------------------------------------------------------------------------
# 400 for Transfer-Encoding: chunked
# ---------------------------------------------------------------------------


def test_transfer_encoding_chunked_returns_400() -> None:
    svc = InferenceService()
    svc.start()
    http = _make_server(svc)
    try:
        raw = (
            f"POST /v1/infer HTTP/1.1\r\n"
            f"Host: {http.host}:{http.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Transfer-Encoding: chunked\r\n"
            f"Connection: close\r\n\r\n"
        ).encode()
        with socket.create_connection((http.host, http.port), timeout=5) as sock:
            sock.sendall(raw)
            sock.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        first_line = data.split(b"\r\n", 1)[0]
        assert b"400" in first_line, f"Expected 400, got: {first_line}"
    finally:
        http.stop()
        svc.stop()


# ---------------------------------------------------------------------------
# Bearer auth: 401 without token, pass with correct token, /health exempt
# ---------------------------------------------------------------------------


def test_auth_token_required(monkeypatch: Any) -> None:
    """With auth_token set, /infer without token returns 401 + WWW-Authenticate."""
    svc = InferenceService()
    svc.start()

    # Pass auth_token directly to HttpServer (per-instance, not module-level)
    http = HttpServer(svc, host="127.0.0.1", port=0, auth_token="secret-token")
    http.start(background=True)
    try:
        # /health is exempt — must be 200
        with urllib.request.urlopen(f"{http.url}/health", timeout=5) as resp:
            assert resp.status == 200

        # /infer without token → 401
        body = json.dumps({"model_id": "any", "inputs": [1.0]}).encode()
        req = urllib.request.Request(
            f"{http.url}/v1/infer",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected 401")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401, f"expected 401, got {exc.code}"
            assert exc.headers.get("WWW-Authenticate") is not None

        # /infer with correct token → not 401 (may be 4xx for other reasons)
        req_auth = urllib.request.Request(
            f"{http.url}/v1/infer",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer secret-token",
            },
            method="POST",
        )
        try:
            urllib.request.urlopen(req_auth, timeout=5)
        except urllib.error.HTTPError as exc:
            # We expect some 4xx for missing model, but NOT 401
            assert exc.code != 401, f"got 401 even with correct token: {exc.code}"
    finally:
        http.stop()
        svc.stop()


# ---------------------------------------------------------------------------
# /metrics contains histogram labels after one inference
# ---------------------------------------------------------------------------


def test_metrics_contains_histogram_after_inference() -> None:
    """After one successful inference, /metrics must include histogram bucket labels."""
    model = nn.Linear(4, 2).eval()
    x = torch.randn(2, 4)

    svc = InferenceService(config=ServiceConfig(max_queue_depth=4, default_timeout_s=30.0))
    svc.start()

    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    svc.models.load("metrics_model", compiled, concurrency_limit=1)
    svc.models.warm("metrics_model", (x,))
    svc.infer("metrics_model", (x,))

    http = _make_server(svc)
    try:
        with urllib.request.urlopen(f"{http.url}/metrics", timeout=5) as resp:
            metrics = resp.read().decode()

        assert "tensortorrent_inference_latency_seconds_bucket" in metrics, (
            "histogram metric missing from /metrics output"
        )
        assert 'le="' in metrics, "le= labels missing from histogram"
        assert 'model="metrics_model"' in metrics, "model= label missing from histogram"
    finally:
        http.stop()
        svc.stop()


# ---------------------------------------------------------------------------
# X-Request-ID header on infer response
# ---------------------------------------------------------------------------


def test_x_request_id_header_present() -> None:
    """POST /infer response must include X-Request-ID header."""
    model = nn.Linear(4, 2).eval()
    x = torch.randn(2, 4)
    svc = InferenceService(config=ServiceConfig(max_queue_depth=4, default_timeout_s=30.0))
    svc.start()
    compiled = tt.compile(model, (x,), config=tt.CompileConfig(use_torch_compile=False, measure_regions=False))
    svc.models.load("rid_model", compiled, concurrency_limit=1)
    svc.models.warm("rid_model", (x,))

    http = _make_server(svc)
    try:
        body = json.dumps(
            {
                "model_id": "rid_model",
                "inputs": [{"dtype": "float32", "shape": list(x.shape), "data": x.tolist()}],
            }
        ).encode()
        req = urllib.request.Request(
            f"{http.url}/v1/infer",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            rid = resp.headers.get("X-Request-ID")
        assert rid is not None, "X-Request-ID header missing from infer response"
    finally:
        http.stop()
        svc.stop()


# ---------------------------------------------------------------------------
# Connection cap: semaphore mechanics tested directly (deterministic)
# ---------------------------------------------------------------------------


def test_connection_cap_semaphore_saturated_sends_503() -> None:
    """Test _ProductionThreadingHTTPServer connection cap via semaphore mechanics directly.

    Testing via a live connection is racy on a 2-core box because the second
    connection may complete before the first holds the semaphore. Instead we test
    the semaphore acquisition logic directly: drain the semaphore, then call
    process_request with a mock socket and verify the 503 raw response path.
    This is deterministic and exercises the exact same code branch.
    """
    from unittest.mock import MagicMock

    # Create a server with max_connections=1
    svc = InferenceService()
    svc.start()
    http = _make_server(svc, max_connections=1)
    try:
        httpd = http._httpd
        assert httpd is not None

        # Drain the semaphore (simulate one active connection)
        acquired = httpd._conn_sem.acquire(blocking=False)
        assert acquired, "should acquire from fresh semaphore"

        # Now the semaphore is at 0; next process_request should 503
        mock_sock = MagicMock()
        mock_sock.sendall = MagicMock()
        mock_sock.close = MagicMock()

        httpd.process_request(mock_sock, ("127.0.0.1", 9999))

        # The 503 raw bytes must have been sent
        calls = mock_sock.sendall.call_args_list
        assert calls, "sendall should have been called for 503 response"
        sent_data = b"".join(call[0][0] for call in calls)
        assert b"503" in sent_data, f"expected 503 in raw response, got: {sent_data[:200]}"

        # Release so the server can clean up
        httpd._conn_sem.release()
    finally:
        http.stop()
        svc.stop()
