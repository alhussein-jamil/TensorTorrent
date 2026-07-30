# Architecture

Python control plane. Rust data plane. One immutable **ExecutableArtifact** is the program.

```mermaid
flowchart LR
  M[nn.Module] --> E[torch.export / FX]
  E --> N[normalize + partition]
  N --> A[AOT region compile]
  A --> ART[ExecutableArtifact]
  ART --> RT[Rust request context]
  RT --> CPU[CPU backend]
  RT --> VIRT[virtual backend]
  RT --> GPU[GPU worker processes]
```

## Control plane (`python/streamcompiler`)

Real packages (no empty facade layers):

| Package | Role |
| --- | --- |
| root / `config` | `compile`, `load`, `CompileConfig`, `CompiledModule` |
| `frontend/` | export capture, IR lowering |
| `ir/` · `analysis/` · `planner/` | graph IR, alias/liveness, planner |
| `compile/` · `codegen/` | measure, plan, specialize, pack, regions |
| `validation/` · `observability/` · `cli/` | validation, traces, doctor |
| `runtime/` | migration bridge (callbacks, CompiledModule, device workers) |
| `_legacy/` | bench/oracle Python DAG only |

Python does **not** own residency, events, stream ordering, or transfer bookkeeping at runtime.

## Data plane (`rust/`)

| Crate | Role |
| --- | --- |
| `sc-ir` | IDs, opcodes, schedule, validation, artifact schema |
| `sc-runtime` | dispatcher, execution context, scheduler, simulator, profiler |
| `sc-memory` | logical tensors, views, copies, allocations, leases |
| `sc-storage` | packs, prefetch, cache, spill, checksums |
| `sc-backend-api` | device-agnostic backend trait |
| `sc-backend-cpu` | NUMA domains, affinity hooks, host buffers, copy bandwidth |
| `sc-backend-virtual` | deterministic simulated accelerators |
| `sc-python` | PyO3 `streamcompiler._native` |

`sc-backend-cuda` is **not** in-tree until hardware-validated.

## ExecutableArtifact

Versioned, immutable, non-pickle. Contains:

- normalized graph identity + compatibility version
- compiled compute regions + tensor metadata
- resource requirements + execution schedule
- streams / events / storage manifest / memory plan
- initial persistent residency (loaded parameters — not fake `Load` ops)
- profile keys

Compilation produces the artifact once. Forward does not rebuild or reinterpret the graph.

Compute with `attributes.native_launch=true` runs via the virtual/CPU backend launch path **without** a Python region callback (GIL-free). Torch regions still use the Python callback until AOT artifacts land.

## ExecutionContext (per request)

```text
ExecutionContext {
    artifact, request_id, cancellation,
    instruction_state, ready_queues, events,
    tensors, views, copies, allocations, transfers,
    storage_state, telemetry
}
```

Rust is sole authority for versions, residency, views/aliases, physical allocations, leases, transfers, release, eviction, budgets, events, stream ordering, storage lifetime.

## Backends

`Backend` discovers devices, queries memory/capabilities, allocates/frees, copies async, launches compiled regions, records/waits events, synchronizes, reports health/memory, cancels queued work.

Resources expose: compute/copy streams, copy engines, links, memory domains, peer-access matrix, NUMA affinity, dtypes, supported artifact formats.

Core never embeds CUDA/ROCm assumptions.

## Serving (`server/`)

Load / unload / warm / infer / cancel / health / readiness / metrics / graceful shutdown.

HTTP (stdlib, no extra deps): `GET /health`, `GET /ready`, `GET /metrics`, `POST /v1/infer`.

```bash
PYTHONPATH=python:. python -m server.cli --listen 127.0.0.1:8080
PYTHONPATH=python:. python -m server.cli --devices virtual_0,virtual_1 --health
```

Bounded queues, backpressure, per-model concurrency, timeouts, request IDs, structured errors/logs, Prometheus metrics, tracing.

## GPU deployment model

```text
Rust coordinator ── owns schedule, topology, request lifecycle, global memory plan
        │
        ├── DeviceWorkerSupervisor (python/runtime) — health / restart / Compute submit
        ├── GPU worker process 0  (one physical GPU)
        ├── GPU worker process 1
        └── ...
```

`ScheduleExecutor` routes Compute to a device worker when `resource` matches a supervised `device_id`. Isolation + restart exercised on virtual labels / CPU. Not production-ready until real multi-GPU tests pass and workers own CUDA contexts.

## NUMA / affinity

`sc-backend-cpu` discovers NUMA nodes and reports topology. Multi-socket `numactl`/cgroup binding is a follow-up for multi-node hosts (this class of host often has 1 socket).

## Honesty labels

| Claim | Status |
| --- | --- |
| CPU + virtual path | measured on CI/dev hosts |
| Multi-GPU CUDA | **blocked** without hardware |
| Simulator numbers | analytic / simulated |
| Torch region AOT | partial (`native_launch`); full Inductor AOT not done |
