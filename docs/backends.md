# Backends

Planner queries capabilities. It does not branch on vendor names.

## Execution

| Backend | ID | Notes |
| --- | --- | --- |
| CPU | `cpu` | Always available with PyTorch |
| Mock accelerator | `mock_accel` | Virtual; inject via `make_mock_accel_graph` — never auto-discovered |
| CUDA | `cuda` | When `torch.cuda` is usable |
| ROCm | `rocm` | When HIP runtime is present |
| MPS | `mps` | Apple Metal |
| SYCL | `sycl` | Intel XPU / `dpctl` |
| OpenCL / Vulkan | `opencl`, `vulkan` | Raise `UnsupportedFeatureError` |

PyTorch-backed devices share `backends/torch_device.py`. Absent devices raise
`BackendError` — no silent host fallback.

With `use_torch_compile=True`, Inductor is kept only when it is not slower than
eager FX on the specialization examples.

## Communication

| Backend | Notes |
| --- | --- |
| Gloo | Host / CPU collectives |
| NCCL / RCCL / oneCCL | Selected by capability when present |
| host-staged | Fallback when direct interconnect is missing |

`select_communication_backend(devices)` picks the first capable backend, else
host-staged.
