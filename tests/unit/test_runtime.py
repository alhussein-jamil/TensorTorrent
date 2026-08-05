"""Runtime helper tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from tensortorrent.parallel import inference_thread_pool
from tensortorrent.runtime.artifact_fingerprint import specialized_fingerprint_mismatch


def test_region_worker_threads_run_in_inference_mode() -> None:
    """torch.inference_mode is thread-local, so pool workers must opt in themselves."""
    with inference_thread_pool(max_workers=2, thread_name_prefix="test-region") as pool:
        modes = list(pool.map(lambda _: torch.is_inference_mode_enabled(), range(4)))
    assert modes == [True] * 4


def test_default_thread_pool_would_not_inherit_inference_mode() -> None:
    """Guards the reason the initializer exists: plain pools drop the mode."""
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=1) as pool:
        assert torch.is_inference_mode_enabled() is True
        assert pool.submit(torch.is_inference_mode_enabled).result() is False


def test_specialized_fingerprint_mismatch_detects_drift() -> None:
    @dataclass
    class _Artifact:
        fingerprint: str

    @dataclass
    class _Machine:
        fingerprint: str

    assert specialized_fingerprint_mismatch(_Artifact("a"), _Machine("b")) is True  # type: ignore[arg-type]
    assert specialized_fingerprint_mismatch(_Artifact("a"), _Machine("a")) is False  # type: ignore[arg-type]
    assert specialized_fingerprint_mismatch(_Artifact(""), _Machine("a")) is False  # type: ignore[arg-type]
