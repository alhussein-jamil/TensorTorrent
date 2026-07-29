"""Milestone 2/3 feature coverage: real CPU paths + honest hardware skips."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.communication import GlooComm, HostStagedComm
from streamcompiler.cost_model.contention import concurrent_slowdown, set_measured_compute_contention
from streamcompiler.runtime.process_workers import ProcessWorkerPool
from streamcompiler.runtime.profile_feedback import refine_contention_from_overlaps
from streamcompiler.runtime.streams import make_event, make_stream
from streamcompiler.storage.fastpath import read_storage_bytes, storage_fastpath_status
from streamcompiler.storage.quantized import load_quantized_state_dict, pack_quantized_state_dict


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
    compiled = sc.compile(model, (x,), config=sc.CompileConfig(use_torch_compile=False))
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


def test_process_worker_pool_runs_callable() -> None:
    pool = ProcessWorkerPool(max_workers=1)
    try:
        fut = pool.submit(_add_for_process_worker, 2, 3)
        assert fut.result() == 5
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


def test_storage_fastpath_pread(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    payload = b"abcdefghijklmnop"
    path.write_bytes(payload)
    result = read_storage_bytes(path, offset=4, nbytes=4)
    assert result.data == b"efgh"
    assert result.backend == "os_pread"
    status = storage_fastpath_status()
    assert status["os_pread"] is True


def test_async_event_cpu_completes() -> None:
    event = make_event("e0", "cpu_numa_0")
    event.record()
    event.wait()
    assert event.is_complete()
    assert make_stream("cpu_numa_0") is None


def test_host_staged_comm_still_sums() -> None:
    out = HostStagedComm().allreduce([torch.ones(3), torch.ones(3)], devices=("a", "b"))
    torch.testing.assert_close(out, torch.ones(3) * 2)


def test_training_mode_skips_inference_guard() -> None:
    model = nn.Linear(4, 2)
    x = torch.randn(2, 4, requires_grad=True)
    compiled = sc.compile(
        model,
        (torch.randn(2, 4),),
        config=sc.CompileConfig(allow_training=True, use_torch_compile=False, measure_regions=False),
    )
    try:
        assert compiled.config.allow_training is True
        out = compiled(x)
        assert out.requires_grad
        out.sum().backward()
        assert x.grad is not None
        assert any(p.grad is not None for p in compiled.parameters())
    finally:
        compiled.close()
