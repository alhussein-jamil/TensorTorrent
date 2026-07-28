"""Milestone 2/3 feature coverage: real CPU paths + honest hardware skips."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.communication import GlooComm, HostStagedComm
from streamcompiler.cost_model.contention import concurrent_slowdown, set_measured_compute_contention
from streamcompiler.runtime.async_events import make_event, make_stream
from streamcompiler.runtime.intraop_split import IntraOpSplit, run_intraop_split
from streamcompiler.runtime.pipeline import MicrobatchPlan, run_pipeline_microbatched
from streamcompiler.runtime.process_workers import ProcessWorkerPool
from streamcompiler.runtime.profile_feedback import refine_contention_from_overlaps
from streamcompiler.runtime.shape_buckets import BucketedModule, ShapeBucket
from streamcompiler.runtime.tensor_parallel import allreduce_sum_host, tensor_parallel_linear_host_staged
from streamcompiler.runtime.transfers import TorchDeviceTransfer, device_transfer_available, select_transfer_backend
from streamcompiler.storage.fastpath import read_storage_bytes, storage_fastpath_status
from streamcompiler.storage.quantized import load_quantized_state_dict, pack_quantized_state_dict


def test_shape_buckets_dispatch_by_batch() -> None:
    small = nn.Linear(8, 4).eval()
    large = nn.Linear(8, 4).eval()
    large.load_state_dict(small.state_dict())
    buckets = BucketedModule(
        [
            ShapeBucket("s", 1, 4, small),
            ShapeBucket("l", 5, 16, large),
        ]
    )
    y = buckets(torch.randn(2, 8))
    assert y.shape == (2, 4)
    y2 = buckets(torch.randn(8, 8))
    assert y2.shape == (8, 4)
    with pytest.raises(Exception, match="outside every specialized bucket"):
        buckets(torch.randn(32, 8))


def test_tensor_parallel_host_staged_matches_dense() -> None:
    x = torch.randn(4, 16)
    w = torch.randn(32, 16)
    b = torch.randn(32)
    dense = x.matmul(w.t()) + b
    sharded = tensor_parallel_linear_host_staged(x, w, b, world_size=4)
    torch.testing.assert_close(sharded, dense, atol=1e-5, rtol=1e-5)
    reduced = allreduce_sum_host([dense * 0.5, dense * 0.5])
    torch.testing.assert_close(reduced, dense, atol=1e-5, rtol=1e-5)


def test_pipeline_microbatch_matches_full_batch() -> None:
    stage1 = nn.Linear(8, 8).eval()
    stage2 = nn.Linear(8, 4).eval()

    def s1(t: torch.Tensor) -> torch.Tensor:
        return torch.relu(stage1(t))

    def s2(t: torch.Tensor) -> torch.Tensor:
        return stage2(t)

    x = torch.randn(10, 8)
    plan = MicrobatchPlan(microbatch_size=3, stages=(s1, s2))
    out = run_pipeline_microbatched(plan, x)
    torch.testing.assert_close(out, s2(s1(x)), atol=1e-5, rtol=1e-5)


def test_intraop_split_cat_matches_full() -> None:
    x = torch.randn(8, 16)

    def op(t: torch.Tensor) -> torch.Tensor:
        return t * 2

    out = run_intraop_split(x, op, IntraOpSplit(dim=0, workers=4, reduce="cat"))
    torch.testing.assert_close(out, op(x))


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
    assert event.completed
    assert make_stream("cpu_numa_0") is None


def test_device_transfer_backend_selection_without_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    assert device_transfer_available("cuda_gpu_0") is False
    backend = select_transfer_backend("host_device_copy", destination="cuda_gpu_0")
    assert backend.backend_id == "simulated_device"
    host = select_transfer_backend("host_device_copy", destination="cpu_numa_0")
    assert isinstance(host, TorchDeviceTransfer) or host.backend_id in {"torch_device_copy", "host_memcpy"}


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
