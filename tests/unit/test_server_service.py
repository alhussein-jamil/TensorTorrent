"""Unit tests for the in-process inference service."""

from __future__ import annotations

import torch
import torch.nn as nn
from server import InferenceService, ServiceConfig

import streamcompiler as sc


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


def test_backpressure_rejects_when_queue_full() -> None:
    svc = InferenceService(config=ServiceConfig(max_queue_depth=0))
    svc.start()
    try:
        import pytest

        from streamcompiler.errors import StreamCompilerError

        with pytest.raises(StreamCompilerError, match="queue full"):
            svc.infer("missing", (torch.randn(1, 1),))
    finally:
        svc.stop()


def test_http_health_ready_metrics_and_infer() -> None:
    import json
    import urllib.error
    import urllib.request

    svc = InferenceService(config=ServiceConfig(max_queue_depth=4, default_timeout_s=30.0))
    svc.start()
    from server.http import HttpServer

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
            assert b"streamcompiler_requests_total" in resp.read()

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
