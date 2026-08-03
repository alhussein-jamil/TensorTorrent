"""Runtime, storage, communication, and hardware-skip coverage."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.backends.communication import GlooComm, HostStagedComm
from tensortorrent.errors import RuntimePlanError
from tensortorrent.planner.cost.contention import concurrent_slowdown, set_measured_compute_contention
from tensortorrent.runtime.process_workers import ProcessWorkerPool
from tensortorrent.runtime.profile_feedback import refine_contention_from_overlaps
from tensortorrent.runtime.streams import make_event, make_stream
from tensortorrent.storage.quantized import load_quantized_state_dict, pack_quantized_state_dict


def test_quantized_storage_roundtrip(tmp_path: Path) -> None:
    state = {"w": torch.randn(16, 8)}
    pack_quantized_state_dict(state, tmp_path / "q.pt")
    loaded = load_quantized_state_dict(tmp_path / "q.pt")
    assert loaded["w"].shape == state["w"].shape
    # int8 quant is lossy; allow coarse tolerance
    err = (loaded["w"] - state["w"]).abs().max().item()
    assert err < 0.5


def test_measured_contention_factor_applies() -> None:
    set_measured_compute_contention(None)
    base = concurrent_slowdown(active_compute=2, active_transfers=0, active_storage=0)
    set_measured_compute_contention(1.5)
    measured = concurrent_slowdown(active_compute=2, active_transfers=0, active_storage=0)
    assert measured.compute >= 1.5
    assert measured.compute >= base.compute
    set_measured_compute_contention(None)
    factor = refine_contention_from_overlaps(sequential_s=10.0, concurrent_s=6.0, workers=2)
    assert factor == pytest.approx(1.2)


def test_profile_feedback_observes_reports() -> None:
    model = nn.Linear(8, 4).eval()
    x = torch.randn(2, 8)
    compiled = tt.compile(model, (x,), config=tt.CompileConfig(use_torch_compile=False))
    try:
        with torch.no_grad():
            compiled(x)
            compiled(x)
        fb = compiled._profile_feedback
        assert fb.updates >= 2
        assert fb.as_dict()["updates"] >= 2
    finally:
        compiled.close()


def _add_for_process_worker(a: int, b: int) -> int:
    return a + b


def _sleep_for_process_worker(seconds: float) -> None:
    time.sleep(seconds)


def test_process_worker_pool_runs_callable() -> None:
    pool = ProcessWorkerPool(max_workers=1)
    try:
        fut = pool.submit(_add_for_process_worker, 2, 3)
        assert fut.result() == 5
    finally:
        pool.shutdown()


@pytest.mark.parametrize("max_workers", (True, 1.5, 0))
def test_process_worker_pool_rejects_invalid_worker_count(max_workers: object) -> None:
    with pytest.raises(RuntimePlanError, match="max_workers"):
        ProcessWorkerPool(max_workers=max_workers)


def test_process_worker_pool_enforces_backpressure() -> None:
    pool = ProcessWorkerPool(max_workers=1, max_pending=1)
    try:
        first = pool.submit(_sleep_for_process_worker, 0.2)
        with pytest.raises(RuntimePlanError, match="backpressure"):
            pool.submit(_add_for_process_worker, 2, 3)
        first.result(timeout=5)
    finally:
        pool.shutdown()


def test_process_worker_crash_fails_pending_future() -> None:
    if sys.platform != "linux":
        pytest.skip("signal-based process test requires Linux")
    pool = ProcessWorkerPool(max_workers=1, start_method="fork")
    try:
        future = pool.submit(_sleep_for_process_worker, 30.0)
        proc = pool._workers[0]
        os.kill(proc.pid, signal.SIGKILL)
        with pytest.raises(RuntimePlanError, match="exited unexpectedly"):
            future.result(timeout=5)
        with pytest.raises(RuntimePlanError, match="broken"):
            pool.submit(_add_for_process_worker, 1, 2)
    finally:
        pool.shutdown()


def test_gloo_falls_back_to_host_sum_without_process_group() -> None:
    backend = GlooComm()
    if not backend.available():
        pytest.skip("Gloo not available in this torch build")
    a = torch.ones(4)
    b = torch.ones(4) * 2
    out = backend.allreduce([a, b], devices=("cpu_0", "cpu_1"))
    torch.testing.assert_close(out, torch.ones(4) * 3)


def test_async_event_cpu_completes() -> None:
    event = make_event("e0", "cpu_numa_0")
    event.record()
    event.wait()
    assert event.is_complete()
    assert make_stream("cpu_numa_0") is None


def test_host_staged_comm_still_sums() -> None:
    out = HostStagedComm().allreduce([torch.ones(3), torch.ones(3)], devices=("a", "b"))
    torch.testing.assert_close(out, torch.ones(3) * 2)


def test_quantized_storage_preserves_supported_logical_dtype(tmp_path: Path) -> None:
    state = {"w": torch.randn(8, 4, dtype=torch.float16)}
    pack_quantized_state_dict(state, tmp_path / "q16.pt")
    loaded = load_quantized_state_dict(tmp_path / "q16.pt")
    assert loaded["w"].dtype == torch.float16
    assert loaded["w"].shape == state["w"].shape


def test_quantized_storage_rejects_non_finite_values(tmp_path: Path) -> None:
    from tensortorrent.errors import StorageError

    with pytest.raises(StorageError, match="NaN or infinity"):
        pack_quantized_state_dict({"w": torch.tensor([float("nan")])}, tmp_path / "bad.pt")


def test_quantized_storage_rejects_malformed_shape(tmp_path: Path) -> None:
    from tensortorrent.errors import StorageError

    path = tmp_path / "bad-shape.pt"
    torch.save(
        {
            "format": "tensortorrent_q8_v1",
            "tensors": {
                "w": {
                    "qdata": torch.ones(3, dtype=torch.int8),
                    "scale": 0.1,
                    "zero_point": 0,
                    "shape": [2, 2],
                    "dtype": "float32",
                }
            },
        },
        path,
    )
    with pytest.raises(StorageError, match="expects 4 values"):
        load_quantized_state_dict(path)
