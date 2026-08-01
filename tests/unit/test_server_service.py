"""Unit tests for the in-process inference service."""

from __future__ import annotations

import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.serve import InferenceService, ServiceConfig


def test_service_health_and_infer_roundtrip() -> None:
    svc = InferenceService(config=ServiceConfig(max_queue_depth=4, default_timeout_s=30.0))
    svc.start()
    try:
        assert svc.health()["ready"] is True
        assert svc.readiness()["ready"] is True
        model = nn.Linear(4, 2).eval()
        x = torch.randn(2, 4)
        compiled = sc.compile(
            model,
            (x,),
            config=sc.CompileConfig(use_torch_compile=False, measure_regions=False),
        )
        version = svc.models.load("m0", compiled, concurrency_limit=2)
        assert version
        svc.models.warm("m0", (x,))
        out = svc.infer("m0", (x,))
        assert out["request_id"]
        assert out["output"].shape == (2, 2)
        text = svc.metrics_prometheus()
        assert "streamcompiler_requests_total" in text
    finally:
        svc.stop()


def test_service_config_rejects_zero_queue_capacity() -> None:
    import pytest

    with pytest.raises(ValueError, match="max_queue_depth"):
        ServiceConfig(max_queue_depth=0)


def test_http_health_ready_metrics_and_infer() -> None:
    import json
    import urllib.error
    import urllib.request

    svc = InferenceService(config=ServiceConfig(max_queue_depth=4, default_timeout_s=30.0))
    svc.start()
    from streamcompiler.serve.http import HttpServer

    http = HttpServer(svc, host="127.0.0.1", port=0)
    http.start(background=True)
    try:
        base = http.url
        with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
            health = json.loads(resp.read().decode())
        assert health["ready"] is True
        with urllib.request.urlopen(f"{base}/ready", timeout=5) as resp:
            assert json.loads(resp.read().decode())["ready"] is True
        with urllib.request.urlopen(f"{base}/metrics", timeout=5) as resp:
            metrics = resp.read()
        assert b"streamcompiler_requests_total" in metrics
        assert b"streamcompiler_requests_cancelled_total" in metrics
        assert b"streamcompiler_queue_rejects_total" in metrics

        model = nn.Linear(4, 2).eval()
        x = torch.randn(2, 4)
        compiled = sc.compile(
            model,
            (x,),
            config=sc.CompileConfig(use_torch_compile=False, measure_regions=False),
        )
        svc.models.load("m0", compiled, concurrency_limit=2)
        svc.models.warm("m0", (x,))
        body = json.dumps(
            {
                "model_id": "m0",
                "inputs": [{"dtype": "float32", "shape": list(x.shape), "data": x.tolist()}],
            }
        ).encode()
        req = urllib.request.Request(
            f"{base}/v1/infer",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
        assert payload["model_id"] == "m0"
        assert payload["output"]["shape"] == [2, 2]

        # Body over max → 413
        tiny = HttpServer(svc, host="127.0.0.1", port=0, max_body_bytes=32)
        tiny.start(background=True)
        try:
            big = json.dumps({"model_id": "m0", "inputs": [{"data": list(range(100))}]}).encode()
            assert len(big) > 32
            bad = urllib.request.Request(
                f"{tiny.url}/v1/infer",
                data=big,
                headers={"Content-Type": "application/json", "Content-Length": str(len(big))},
                method="POST",
            )
            try:
                urllib.request.urlopen(bad, timeout=5)
                raise AssertionError("expected HTTPError")
            except urllib.error.HTTPError as exc:
                assert exc.code == 413
                err = json.loads(exc.read().decode())
                assert "too large" in err["error"]
        finally:
            tiny.stop()

        # Bad dtype rejected
        bad_dtype = json.dumps(
            {"model_id": "m0", "inputs": [{"dtype": "float32", "shape": [1, 1], "data": [[1.0]]}]}
        ).replace("float32", "not_a_dtype")
        bad_req = urllib.request.Request(
            f"{base}/v1/infer",
            data=bad_dtype.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(bad_req, timeout=5)
            raise AssertionError("expected HTTPError")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            assert "unsupported dtype" in json.loads(exc.read().decode())["error"]
    finally:
        http.stop()
        svc.stop()


def test_infer_rejects_non_positive_timeout() -> None:
    import pytest

    from streamcompiler.errors import StreamCompilerError

    svc = InferenceService()
    svc.start()
    try:
        with pytest.raises(StreamCompilerError, match="timeout_s must be > 0"):
            svc.infer("missing", (torch.randn(1, 1),), timeout_s=0)
    finally:
        svc.stop()


def test_http_json_nested_numeric_lists_form_one_tensor() -> None:
    from streamcompiler.serve.http import _json_to_tensor

    value = _json_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(value, torch.Tensor)
    assert tuple(value.shape) == (2, 2)


def test_http_json_explicit_descriptors_form_multiple_inputs() -> None:
    from streamcompiler.serve.http import _json_to_tensor

    value = _json_to_tensor(
        [
            {"dtype": "float32", "shape": [2], "data": [1.0, 2.0]},
            {"dtype": "int64", "shape": [1], "data": [3]},
        ]
    )
    assert isinstance(value, list)
    assert len(value) == 2
    assert value[0].dtype == torch.float32
    assert value[1].dtype == torch.int64


def test_http_server_rejects_invalid_limits() -> None:
    import pytest

    from streamcompiler.serve.http import HttpServer

    svc = InferenceService()
    with pytest.raises(ValueError, match="max_body_bytes"):
        HttpServer(svc, max_body_bytes=0)
    with pytest.raises(ValueError, match="socket_timeout_s"):
        HttpServer(svc, socket_timeout_s=0)


def test_http_server_rejects_double_start() -> None:
    import pytest

    from streamcompiler.serve.http import HttpServer

    svc = InferenceService()
    svc.start()
    http = HttpServer(svc, host="127.0.0.1", port=0)
    http.start(background=True)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            http.start(background=True)
    finally:
        http.stop()
        svc.stop()


def test_http_cancel_unknown_request_returns_404() -> None:
    import json
    import urllib.error
    import urllib.request

    svc = InferenceService()
    svc.start()
    from streamcompiler.serve.http import HttpServer

    http = HttpServer(svc, host="127.0.0.1", port=0)
    http.start(background=True)
    try:
        body = json.dumps({"request_id": "missing-id"}).encode()
        req = urllib.request.Request(
            f"{http.url}/v1/cancel",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected HTTPError")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
            payload = json.loads(exc.read().decode())
            assert payload["cancelled"] is False
    finally:
        http.stop()
        svc.stop()


def test_http_incomplete_body_is_rejected() -> None:
    import socket

    svc = InferenceService()
    svc.start()
    from streamcompiler.serve.http import HttpServer

    http = HttpServer(svc, host="127.0.0.1", port=0)
    http.start(background=True)
    try:
        payload = b'{"model_id":"x","inputs":[]}'
        request = (
            f"POST /v1/infer HTTP/1.1\r\nHost: {http.host}:{http.port}\r\n"
            f"Content-Type: application/json\r\nContent-Length: 1000\r\n"
            f"Connection: close\r\n\r\n"
        ).encode() + payload
        with socket.create_connection((http.host, http.port), timeout=5) as sock:
            sock.sendall(request)
            sock.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        status = data.split(b"\r\n", 1)[0]
        assert b"400" in status
        assert b"incomplete" in data.lower()
    finally:
        http.stop()
        svc.stop()
