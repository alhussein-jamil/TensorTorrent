# Architecture

StreamCompiler is a two-stage compiler plus schedule-driven runtime for PyTorch
inference.

```
nn.Module
  → torch.export
  → region partition / IR lowering
  → alias + liveness analysis
  → portable IR + packed weights
  → machine discovery + region measurement
  → planner (placement + residency + ExecutableSchedule)
  → discrete-event simulation of that schedule (analytic)
  → ScheduleExecutor on the same instruction DAG
  → CompiledModule (nn.Module)
```

## Packages

| Package | Responsibility |
|---------|----------------|
| `frontend` | `torch.export` capture and IR lowering |
| `ir` | Heterogeneous graph IR + resource graph |
| `analysis` | Alias, liveness, repeated blocks |
| `hardware` | Fingerprint, discovery, storage microbench |
| `backends` | CPU / CUDA / ROCm / MPS / SYCL / mock_accel contracts |
| `communication` | NCCL / RCCL / oneCCL / Gloo / host-staged |
| `planner` | Maximal subset search; capacity hard-filters |
| `compile` | Portable compile + specialization pipeline |
| `storage` | Aligned model packs (chunked write, atomic replace) |
| `runtime` | `CompiledModule`, `GraphExecutor` → `ScheduleExecutor`, `CopyStore`, streams, parameter stores |
| `simulator` | Analytic walk of `ExecutableSchedule` (`simulated=True`) |
| `observability` | Simulated plan traces + measured Chrome/HTML timelines |
| `validation` | Hardware validation suite |
| `cli` | `doctor`, `profile`, `validate-hardware`, … |
| `cost_model` | Transfer / contention models |

Vendor-specific code stays in `backends/` and `communication/`. The planner
queries capabilities and measured (or honestly labelled simulated) costs only.

## Schedule is the program

Specialization builds one immutable `ExecutableSchedule`: Prefetch, Load,
Transfer, RecordEvent, WaitEvent, Compute, Evict, Release.

- **Runtime** (`ScheduleExecutor`) and **simulator** (`simulate_schedule`) consume
  the same instruction IDs and dependencies.
- Load is disk→host only. Device residency requires Transfer.
- Activation spill/reload are explicit Evict/Load ops under
  `activation_budget_bytes` — never transparent spill inside Compute.
- Physical copies live in `CopyStore` keyed by `(logical_tensor_id, resource_id)`.
  Replication does not bump logical versions; aliases share one allocation id.
- Mock accelerators use `VirtualDeviceTensor` handles so host tensors are not
  silently treated as device-resident.
- Release waits for consumer Computes and for Record/Wait edges that completed
  Transfers of that tensor.

`GraphExecutor` dispatches exclusively through `ScheduleExecutor`. There is no
second production executor.

## Measurement honesty

`BackendProfiler` prefers CPU (measured) and mock_accel (simulated). Cache hits
preserve `measured` / `simulated` flags. Planner may use simulated latencies for
placement without claiming them as measured hardware.

## Concurrency

Region concurrency is enabled only after measured wins on the widest independent
level, the full DAG, and versus a fused single-region candidate. GPU presence is
discovery only; concurrent CPU+GPU execution is unvalidated until a real
overlapping run exists on accelerator hardware. Heterogeneous schedule semantics
on GPU-less hosts use `mock_accel` (including multi-device host-staged topologies).

## Telemetry

- Plan simulation: `compiled.visualize(path)` — labelled `simulated=True`.
- Measured run: after `forward`, `compiled.visualize(path, measured=True)`.
