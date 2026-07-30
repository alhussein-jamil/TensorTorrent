# Backends

Planner queries capabilities. It does not branch on vendor names.

## Execution

| Backend | ID | Notes | Readiness |
| --- | --- | --- | --- |
| CPU | `cpu` | NUMA domains, affinity, host buffers | measured on host |
| Virtual | `mock_accel` / Rust virtual | Deterministic simulated accelerator — never auto-discovered | simulated |
| CUDA | `cuda` | When `torch.cuda` is usable; multi-process workers planned | untested / blocked without hardware |
| ROCm | `rocm` | When HIP runtime is present | untested / blocked without hardware |

Unsupported accelerator stubs (MPS, SYCL, OpenCL, Vulkan) were removed. Do not claim them.

PyTorch-backed devices share `backends/torch_device.py`. Absent devices raise
`BackendError` — no silent host fallback.

With `use_torch_compile=True`, Inductor is kept only when it is not slower than
eager FX on the specialization examples.

## Communication

| Backend | Notes |
| --- | --- |
| Gloo | Host / CPU collectives |
| NCCL / RCCL | Selected by capability when present (multi-node later) |
| host-staged | Fallback when direct interconnect is missing |

`select_communication_backend(devices)` picks the first capable backend, else
host-staged. Multi-node collectives are out of product scope until single-node
is reliable.
