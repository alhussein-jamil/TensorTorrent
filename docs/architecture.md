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

| Package | Role |
| --- | --- |
| `api/` | `compile`, `load`, `CompileConfig`, `CompiledModule` |
| `frontend/` | export capture, IR lowering |
| `partitioning/` | regions, alias/liveness, graph IR |
| `compilation/` | measure, plan, specialize, pack, backend discovery |
| `diagnostics/` | validation, traces, CLI doctor |

Python does **not** own residency, events, stream ordering, or transfer bookkeeping at runtime.

## Data plane (`rust/`)

| Crate | Role |
| --- | --- |
| `sc-ir` | IDs, opcodes, schedule, validation, artifact schema |
| `sc-runtime` | dispatcher, execution context, scheduler, simulator, profiler |
| `sc-memory` | logical tensors, views, copies, allocations, leases |
| `sc-storage` | packs, prefetch, cache, spill, checksums |
| `sc-backend-api` | device-agnostic backend trait |
| `sc-backend-cpu` | NUMA domains, affinity, host buffers, copy bandwidth |
| `sc-backend-virtual` | deterministic simulated accelerators |
| `sc-backend-cuda` | real CUDA (only when hardware-validated) |
| `sc-python` | PyO3 `streamcompiler._native` |

## ExecutableArtifact

Versioned, immutable, non-pickle. Contains:

- normalized graph identity + compatibility version
- compiled compute regions + tensor metadata
- resource requirements + execution schedule
- streams / events / storage manifest / memory plan
- initial persistent residency (loaded parameters — not fake `Load` ops)
- profile keys

Compilation produces the artifact once. Forward does not rebuild or reinterpret the graph.

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
```

Bounded queues, backpressure, per-model concurrency, timeouts, request IDs, structured errors/logs, Prometheus metrics, tracing.

## GPU deployment model

```text
Rust coordinator ── owns schedule, topology, request lifecycle, global memory plan
        │
        ├── DeviceWorkerSupervisor (python/runtime) — health / restart
        ├── GPU worker process 0  (one physical GPU)
        ├── GPU worker process 1
        └── ...
```

Isolation + restart exercised on virtual device labels. Not production-ready until real multi-GPU tests pass and workers own CUDA contexts on the schedule path.

## Simulator

One Rust DES using the same artifact and topology types as the runtime.

Outcomes: `Valid` | `InfeasibleMemory` | `InvalidResidency` | `InvalidEvent` | `Unsupported`.

Never invents missing copies or completed events. Runtime-vs-simulator error is reported explicitly.
