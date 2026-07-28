# Architecture

StreamCompiler is organized as a real compiler + runtime:

```
PyTorch nn.Module
  → torch.export
  → normalization / lowering
  → alias, liveness, repeated-block analysis
  → portable heterogeneous IR + packed weights
  → deployment-time hardware discovery & profiling
  → maximal heterogeneous planner
  → discrete-event simulation (milestone)
  → specialized async execution plan
  → PyTorch-compatible CompiledModule
```

## Module map

| Package | Responsibility |
|---------|----------------|
| `frontend` | `torch.export` capture and IR lowering |
| `ir` | Heterogeneous graph IR + resource graph |
| `analysis` | Alias, liveness, redundancy, repeated blocks |
| `hardware` | Fingerprint, discovery, caches |
| `backends` | CPU / CUDA / ROCm / MPS / SYCL contracts |
| `communication` | NCCL / RCCL / oneCCL / Gloo / host-staged |
| `planner` | Maximal subset search + resource decisions |
| `compile` | Portable + specialization pipeline |
| `storage` | Aligned model packs |
| `runtime` | Compiled module wrapper / executors |
| `validation` | Production hardware validation suite |
| `cli` | `doctor`, `profile`, `validate-hardware`, … |

Vendor-specific code stays inside backends. The planner consumes capability queries and measured costs only.

## Native runtime

CMake-based native components (streams, events, IO queues) live under `native/` and are optional for the pure-Python milestone. They must not encode single-vendor assumptions.
