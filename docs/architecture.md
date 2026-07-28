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
| `runtime` | Compiled module, graph executor, weight store (resident or streaming with timed I/O∩compute) |
| `validation` | Production hardware validation suite |
| `cli` | `doctor`, `profile`, `validate-hardware`, … |
| `cost_model` | Transfer / contention models fed by measurements |

Vendor-specific code stays inside backends. The planner consumes capability queries and measured costs only. Region concurrency is enabled only after measured wins on the widest level, the full DAG, and versus a fused single-region candidate.

## Implementation status

Everything is Python on top of PyTorch today. There is no native extension: an
earlier `native/` C++ stub was removed because nothing built or called it, and a
header-only placeholder is indistinguishable from a missing feature.

The stages above are implemented. Discrete-event simulation is an analytic model
used for planning and is always labelled `simulated=True`; it models tensor
lifetimes, transfers, destination residency, release, eviction pressure, and
contention but never executes kernels. Parameter/state bytes remain resident for
each region's `[start, end]` interval so overlapping peers that share a memory
pool contribute to peak together. Eviction pressure marks over-capacity
residency without claiming a validated spill/recompute path.
Cross-device concurrent execution has an explicit residency/transfer schedule
(`runtime/residency.py`) and a shared `ExecutableSchedule`
(`runtime/schedule.py`) consumed by the simulator; simultaneous CPU–GPU
execution remains unvalidated until run on real accelerators.
Enumerating GPUs is hardware detection, not concurrent-execution validation.

Region realization uses FX subgraphs by default. With
`CompileConfig.use_torch_compile=True`, regions wrap `torch.compile` (Inductor)
and keep an explicit eager FX fallback that still executes the real graph.
Measured runtime telemetry exports via `CompiledModule.visualize(..., measured=True)`
and `observability.report_to_chrome_trace` (distinct from simulated plan traces).

`GraphExecutor` remains the single production runtime. It walks the region DAG
for Compute (fast path / static resident / concurrent workers) while consulting
the shared `ExecutableSchedule` for Transfer ops and rejecting schedule/program
order mismatches. It does not interpret every IR opcode as a separate interpreter
loop; Prefetch/Load for weights still go through `ParameterStore`, and Release is
driven by consumer counts aligned with schedule Release markers.
