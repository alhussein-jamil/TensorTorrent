# Backends

Planner queries capabilities. It does not branch on vendor names.

## Execution

| Backend | ID | Notes |
| --- | --- | --- |
| CPU | `cpu` | NUMA domains, affinity, host buffers |
| CUDA | `cuda` | NVIDIA GPUs via PyTorch; placement, measure, execute |
| ROCm | `rocm` | AMD GPUs when HIP runtime is present |
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
