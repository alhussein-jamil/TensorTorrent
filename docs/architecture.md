# Architecture

StreamCompiler is organized as a real compiler + runtime:

```
PyTorch nn.Module
  → torch.export
  → region partition / IR lowering
  → alias, liveness, repeated-block analysis
  → portable heterogeneous IR + packed weights
  → deployment-time hardware discovery & profiling
  → maximal heterogeneous planner
  → discrete-event simulation (analytic, used for planning)
  → specialized async execution plan
  → PyTorch-compatible CompiledModule
```

## Module map

| Package | Responsibility |
|---------|----------------|
| `frontend` | `torch.export` capture and IR lowering |
| `ir` | Heterogeneous graph IR + resource graph |
| `analysis` | Alias, liveness, repeated blocks |
| `hardware` | Fingerprint, discovery, storage microbench |
| `backends` | CPU / CUDA / ROCm / MPS / SYCL contracts |
| `communication` | NCCL / RCCL / oneCCL / Gloo / host-staged |
| `planner` | Maximal subset search + local refinement |
| `compile` | Portable + specialization pipeline |
| `storage` | Aligned model packs |
| `runtime` | Compiled module, graph executor, weight store |
| `validation` | Production hardware validation suite |
| `cli` | `doctor`, `profile`, `validate-hardware`, … |

Vendor-specific code stays inside backends. The planner consumes capability queries and measured costs only.

## Implementation status

Everything is Python on top of PyTorch today. There is no native extension: an
earlier `native/` C++ stub was removed because nothing built or called it, and a
header-only placeholder is indistinguishable from a missing feature.

The stages above are implemented. Discrete-event simulation is an analytic model
used for planning and is reported as simulated. Cross-device concurrent execution
is planned.
