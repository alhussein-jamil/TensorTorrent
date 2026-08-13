"""CUDA copy/compute stream overlap (CPU-safe; GPU tests skip without CUDA)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from tensortorrent.runtime.device_streams import CudaEventHandle, DeviceStreamRuntime, streams_enabled_for_context


def test_maybe_create_skips_cpu_bindings() -> None:
    bindings = {
        "r0": SimpleNamespace(backend_id="cpu", device="cpu_numa_0", compiled=SimpleNamespace(torch_device="cpu"))
    }
    assert DeviceStreamRuntime.maybe_create(bindings) is None


def test_streams_disabled_under_autograd() -> None:
    ctx = SimpleNamespace(enable_grad=True, device_streams=object())
    assert streams_enabled_for_context(ctx) is False
    ctx.enable_grad = False
    ctx.device_streams = None
    assert streams_enabled_for_context(ctx) is False


def test_cuda_event_handle_protocol() -> None:
    if not torch.cuda.is_available():
        return
    event = torch.cuda.Event()
    torch.cuda.synchronize()
    event.record()
    handle = CudaEventHandle(event, torch.device("cuda", 0))
    handle.wait()
    assert handle.is_complete() is True


def test_overflow_h2d_pins_and_caches_host_source() -> None:
    if not torch.cuda.is_available():
        return
    from tensortorrent.runtime.pinning import pin_for_dma

    pageable = torch.randn(64, 64)
    assert not pageable.is_pinned()
    pinned = pin_for_dma(pageable)
    assert pinned.is_pinned()
    assert pin_for_dma(pinned) is pinned

    bindings = {
        "r0": SimpleNamespace(
            backend_id="cuda",
            device="cuda_gpu_0",
            compiled=SimpleNamespace(torch_device="cuda:0"),
        )
    }
    runtime = DeviceStreamRuntime.maybe_create(bindings)
    assert runtime is not None
    try:
        host = torch.randn(128, 128)
        dest, event = runtime.transfer(host, torch.device("cuda", 0))
        assert event is not None
        assert dest.device.type == "cuda"
        cached = runtime._pinned_cache[id(host)]
        assert cached.is_pinned()
        dest2, _event2 = runtime.transfer(host, torch.device("cuda", 0))
        assert runtime._pinned_cache[id(host)] is cached
        event.wait()
        torch.testing.assert_close(dest2.cpu(), host)
    finally:
        runtime.close()
        assert runtime._pinned_cache == {}


def test_async_h2d_then_compute_matches_blocking() -> None:
    if not torch.cuda.is_available():
        return
    bindings = {
        "r0": SimpleNamespace(
            backend_id="cuda",
            device="cuda_gpu_0",
            compiled=SimpleNamespace(torch_device="cuda:0"),
        )
    }
    runtime = DeviceStreamRuntime.maybe_create(bindings)
    assert runtime is not None
    try:
        host = torch.randn(256, 256, pin_memory=True)
        target = torch.device("cuda", 0)
        dest, copy_event = runtime.transfer(host, target)
        assert copy_event is not None
        runtime.wait_on_compute(target, copy_event)

        def _mm() -> torch.Tensor:
            return dest @ dest

        result, compute_event = runtime.run_compute(target, _mm)
        assert compute_event is not None
        compute_event.wait()
        expected = host.to(target) @ host.to(target)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
    finally:
        runtime.close()


def test_schedule_path_numeric_with_streams() -> None:
    """Compile a tiny GPU model and check DirectPlan/schedule still matches eager."""
    if not torch.cuda.is_available():
        return
    import tensortorrent as tt

    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    x = torch.randn(4, 32)
    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=True,
            allow_cpu=False,
            validate_numerics=False,
            prefer_direct_path=False,
        ),
    )
    try:
        streams = getattr(compiled._executor._schedule_executor, "_device_streams", None)
        assert streams is not None
        y = compiled(x)
        torch.testing.assert_close(y.cpu(), model(x), atol=1e-4, rtol=1e-4)
    finally:
        compiled.close()
