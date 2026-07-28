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
| `runtime` | Compiled module, `GraphExecutor`→`ScheduleExecutor`, weight store, `ExecutableSchedule`, authoritative `CopyStore` residency, buffer reuse |
| `observability` | Simulated Chrome traces + measured execution Chrome/HTML timelines |
| `validation` | Production hardware validation suite |
| `cli` | `doctor`, `profile`, `validate-hardware`, … |
| `cost_model` | Transfer / contention models fed by measurements |

Vendor-specific code stays inside backends. The planner consumes capability queries and measured costs only. Region concurrency is enabled only after measured wins on the widest level, the full DAG, and versus a fused single-region candidate.

## Implementation status

Everything is Python on top of PyTorch today. There is no native extension: an
earlier `native/` C++ stub was removed because nothing built or called it, and a
header-only placeholder is indistinguishable from a missing feature.

The stages above are implemented. Discrete-event simulation walks the same
`ExecutableSchedule` instruction DAG the runtime executes (Prefetch/Load/Transfer/
RecordEvent/WaitEvent/Compute/Evict/Release) and is always labelled
`simulated=True`; it never executes kernels. `simulate_plan` is only a thin
wrapper that lowers an `ExecutionPlan` through `build_residency_schedule` +
`build_executable_schedule` before calling `simulate_schedule` — it does not
infer transfers inside the simulator. Development hosts without GPUs use
deterministic virtual accelerators for heterogeneous scheduling semantics —
CUDA/ROCm/multi-GPU concurrent execution is **not** validated on GPU-less VMs.
Parameter/state bytes remain resident for each region's lifetime so overlapping
peers that share a memory pool contribute to peak together. Eviction pressure
marks over-capacity residency without claiming a validated spill/recompute path.
Cross-device concurrent execution builds an explicit residency/transfer plan
(`runtime/residency.py` → planner) and a shared `ExecutableSchedule`
(`runtime/schedule.py`) consumed by both simulator and `ScheduleExecutor`.
Runtime physical copies live only in `CopyStore` keyed by
`(logical_tensor_id, resource_id)`; replication does not bump logical versions.
`TensorDirectory` remains for legacy telemetry hooks on the GraphExecutor facade
and is not the schedule-path residency authority.
Enumerating GPUs is hardware detection, not concurrent-execution validation.

Region realization uses FX subgraphs by default. With
`CompileConfig.use_torch_compile=True`, regions wrap `torch.compile` (Inductor)
and keep an explicit eager FX fallback that still executes the real graph.
Measured runtime telemetry exports via `CompiledModule.visualize(..., measured=True)`
and `observability.report_to_chrome_trace` (distinct from simulated plan traces).

`GraphExecutor` dispatches exclusively through `ScheduleExecutor` on the
`ExecutableSchedule` DAG. Prefetch/Load materialize parameters into RAM;
Transfer creates a separate destination copy (never a silent `.to(device)` inside
Compute). Async streams/events (`runtime/streams.py`) own CPU, mock-accel, and
CUDA-shaped completion handles; real CUDA streams/events remain future work on
GPU hosts.
