"""Backend contract and discovery tests."""

from __future__ import annotations

from tensortorrent.backends import all_backends, available_backends
from tensortorrent.backends.base import ExecutionBackend
from tensortorrent.hardware.discovery import discover_resource_graph


def test_all_backends_expose_contract() -> None:
    required = [
        "available",
        "discover_devices",
        "supported_ops",
        "supported_dtypes",
        "enumerate_kernels",
        "benchmark",
        "compile",
        "execute",
        "transfer_capabilities",
    ]
    for backend in all_backends():
        assert isinstance(backend, ExecutionBackend)
        for name in required:
            assert callable(getattr(backend, name))


def test_cpu_backend_always_available() -> None:
    ids = {b.backend_id for b in available_backends()}
    assert "cpu" in ids


def test_discovery_does_not_claim_missing_gpus() -> None:
    graph = discover_resource_graph()
    assert graph.fingerprint
    # On this development host there may be no NVIDIA devices; that must not be
    # reported as successful CUDA validation elsewhere.
    if "cuda" not in graph.backends_present:
        assert (
            graph.attributes.get("cuda_available") in (False, None) or graph.attributes.get("cuda_available") is False
        )
    assert "dev_machine_note" in graph.attributes
