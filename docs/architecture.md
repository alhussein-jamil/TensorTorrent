# Architecture

Python control plane. Rust data plane. One immutable schedule is the program.

<p align="center">
  <img src="figures/pipeline.svg" alt="Compile pipeline" width="720" />
</p>

```mermaid
flowchart LR
  M[nn.Module] --> E[torch.export]
  E --> R[regions + IR]
  R --> P[packs]
  P --> S[specialize]
  S --> Q[ExecutableSchedule]
  Q --> X[Rust dispatcher]
  X --> C[Python Compute]
  X --> N[Rust residency / I/O]
```

## Pipeline

1. **Portable** — export, partition, lower IR, pack weights (hardware-independent)
2. **Specialize** — discover resources, measure regions, plan placement, build
   `ExecutableSchedule`, fingerprint-gated cache
3. **Run** — `NativeCompiledArtifact` + per-forward `NativeExecutionContext`

<p align="center">
  <img src="figures/runtime.svg" alt="Runtime data plane" width="720" />
</p>

```mermaid
flowchart TB
  subgraph rust [Rust]
    D[dispatcher]
    RS[ResidencyStore]
    ST[StreamingStore]
    VB[VirtualBackend]
    D --> RS
    D --> ST
    D --> VB
  end
  subgraph py [Python]
    CB[region callback]
    PL[parameter materialize]
    SP[spill tensorize]
    BAG[CopyStore value bag]
  end
  D -->|Compute wave| CB
  D -->|Load wave| PL
  D -->|spill| SP
  CB --> BAG
  PL --> BAG
  RS -.->|authority| BAG
```

## Schedule opcodes

`Prefetch` · `Load` · `Transfer` · `RecordEvent` · `WaitEvent` · `Compute` ·
`Evict` · `Release`

- Runtime and simulator share the same schedule model and resource IDs
  (`stream_id`, `copy_engine_id`, `link_id`, `io_queue_id`).
- Load is disk→host. Device residency requires Transfer.
- Missing / stale copies, unknown events, invalid releases fail closed.
- Resident parameters register once; forward does not invent cosmetic Loads.

## Crates

| Crate | Role |
| --- | --- |
| `streamcompiler-core` | Schedule IR, validation, serde |
| `streamcompiler-memory` | Residency, leases, allocations |
| `streamcompiler-runtime` | Dispatcher, context, workers |
| `streamcompiler-simulator` | Deterministic DES |
| `streamcompiler-storage` | Packs, pread, spill, streaming |
| `streamcompiler-virtual-backend` | Simulated accelerators |
| `streamcompiler-profiler` | Cost records |
| `streamcompiler-python` | PyO3 `streamcompiler._native` |

## Python packages

| Package | Role |
| --- | --- |
| `frontend` / `ir` / `analysis` | Export, IR, liveness |
| `hardware` / `planner` / `compile` | Discovery, plan, specialize |
| `backends` / `communication` | Device and collective contracts |
| `runtime` | `CompiledModule`, native bridge, value bag |
| `storage` / `simulator` / `cli` | Packs, DES wrapper, doctor |

Build: `maturin develop --release`. Missing native extension fails closed on
`compile()` / forward.
