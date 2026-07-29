# Architecture

StreamCompiler is a hybrid compiler/runtime: Python control plane + Rust data plane.

```
nn.Module
  → torch.export (Python)
  → region partition / IR lowering (Python)
  → alias + liveness analysis (Python)
  → portable IR + packed weights (Python)
  → machine discovery + region measurement (Python)
  → planner (Python) → immutable ExecutableSchedule
  → discrete-event simulation (Rust default; Python oracle via STREAMCOMPILER_PYTHON_SIM=1)
  → Rust schedule dispatcher (dependency DAG, workers, GIL released)
      ↳ Python region callbacks for Compute only (resident / Transfer-metadata path)
  → CompiledModule (nn.Module)
```

## Rust crates (`crates/`)

| Crate | Responsibility |
|---------|----------------|
| `streamcompiler-core` | IDs, opcodes, immutable schedule, validation, serde |
| `streamcompiler-memory` | Logical residency + physical allocation accounting |
| `streamcompiler-simulator` | Deterministic DES; `SimulationOutcome` (Valid / InfeasibleMemory / …) |
| `streamcompiler-runtime` | Event-driven dispatcher, `NativeExecutionContext`, `ResourceState` |
| `streamcompiler-backend-api` | Backend trait + C ABI stubs |
| `streamcompiler-virtual-backend` | Deterministic simulated accelerator |
| `streamcompiler-storage` | Pack manifest, positional pread, chunk cache, `StreamingStore` |
| `streamcompiler-profiler` | Cost records with measured/simulated/estimated/unknown |
| `streamcompiler-python` | PyO3 module `streamcompiler._native` |

Build: `maturin develop` or `pip install .`. Missing native extension fails closed
unless `STREAMCOMPILER_DEV_PYTHON_RUNTIME=1` (benchmark oracle only;
deprecated alias `STREAMCOMPILER_ALLOW_PYTHON_RUNTIME` still accepted).

`cargo test --workspace` and `cargo clippy --workspace --all-targets --all-features`
include `streamcompiler-python` (rlib + optional `extension-module` feature).

`make native-gate` proves the public `compile()` path uses the native artifact
with zero hot-path schedule conversion.

Streaming parameters: `NativeStreamingStore` owns pack pread, byte LRU, shared
in-flight reads, and prefetch workers. Python tensorizes at Load and enforces the
decoded-tensor RAM budget.
## Python packages

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
| `runtime` | `CompiledModule`, `GraphExecutor`, instruction handlers, `CopyStore` |
| `simulator` | Analytic walk of `ExecutableSchedule` (`simulated=True`) |
| `observability` | Simulated plan traces + measured Chrome/HTML timelines |
| `validation` | Hardware validation suite |
| `cli` | `doctor`, `profile`, `validate-hardware`, … |
| `cost_model` | Transfer / contention models |
| `native` | Extension loader |

Vendor-specific code stays behind backend traits. Core scheduling never imports
CUDA/ROCm. Virtual accelerators are explicitly labelled simulated.

## Schedule is the program

Specialization builds one immutable `ExecutableSchedule`: Prefetch, Load,
Transfer, RecordEvent, WaitEvent, Compute, Evict, Release.

- **Rust runtime** owns dependency counters, ready queues, ordered streams, and
  waits with the GIL released between instruction callbacks.
- **Simulator** and **runtime** share the same schedule model (Rust types via
  JSON/bindings; Python planner still emits the schedule).
- Load is disk→host only. Device residency requires Transfer.
- Activation spill/reload are explicit Evict/Load ops under
  `activation_budget_bytes`.
- Physical copies: views share one allocation; distinct resources count separately.
- Release waits for consumer Computes and Record/Wait edges.

`GraphExecutor` dispatches through `ScheduleExecutor.run`, which prefers the
native Rust dispatcher.
