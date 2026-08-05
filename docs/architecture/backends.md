# Backends

Planner queries capabilities. It does not branch on vendor names.

**Production support** for a backend means measured execution and numerical
correctness on the **target** host (`tensortorrent validate-hardware`,
`tests/hardware/`), not merely that the library imports or appears in discovery.
CI runs CPU/virtual paths; CUDA/ROCm/XPU evidence is target-local (see
`make hardware-test`).

## Execution

| Backend | ID | Notes |
| --- | --- | --- |
| CPU | `cpu` | NUMA domains, affinity, host buffers |
| CUDA | `cuda` | NVIDIA GPUs via PyTorch; placement, measure, execute |
| ROCm | `rocm` | AMD GPUs via HIP-enabled PyTorch; measured region and transfer profiling |
| Intel XPU | `xpu` | Intel GPU/XPU through capability-gated `torch.xpu`; measured profiling |
| Virtual | `mock_accel` / Rust virtual | Deterministic simulated accelerator for CI |

PyTorch-backed devices share `backends/torch_device.py`. Absent devices raise
`BackendError`.

With `use_torch_compile=True`, Inductor is kept only when it is not slower than
eager FX on the specialization examples.

## Communication

| Backend | Notes |
| --- | --- |
| NCCL | Selected for CUDA device sets when available |
| RCCL | Selected for ROCm device sets when available |
| oneCCL | Selected when the Intel oneCCL binding is present |
| Gloo | CPU / host collectives |
| host-staged | Portable fallback via host memory |

`select_communication_backend(devices)` picks the first capable backend for the
device set, otherwise host-staged.

## Third-party backends

External packages can register a backend without modifying planner code by exposing
an entry point in the `tensortorrent.backends` group. The entry point may return
an `ExecutionBackend` instance, subclass, or zero-argument factory.

```toml
[project.entry-points."tensortorrent.backends"]
my_accelerator = "my_package.backend:create_backend"
```

Plugin discovery is isolated: a broken optional backend is reported by
`tensortorrent doctor` and in the resource graph, but it does not prevent CPU or
other backends from starting. Set `TENSORTORRENT_DISABLE_BACKEND_PLUGINS=1` for
hermetic deployments. Built-in backend IDs take precedence over duplicates.

## Hardware truthfulness

Discovery is not validation. Every backend is capability-gated and target-machine
validation must run before production use. Unmeasured host-device and storage links
are represented explicitly as conservative fallbacks so the runtime never hides
data movement in a backend call.

## Profiling safety

Built-in CPU, CUDA, ROCm, Intel XPU, and virtual backends use a common profiler
record format. Large transfer probes are bounded to avoid allocating model-sized
temporary buffers; measured bandwidth is extrapolated to the requested byte count
and the sampled byte count is retained in profile notes. A failed optional probe is
recorded as unavailable rather than aborting specialization of other resources.
