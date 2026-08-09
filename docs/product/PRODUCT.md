# Product scope

Single-host heterogeneous compiler/runtime for PyTorch. Inference is primary; training is an opt-in resident-parameter path.

Alpha: production readiness is per target host, not inferred from discovery.

## In scope

- Capture and region partition of PyTorch graphs
- One host: CPU/NUMA plus zero or more supported accelerators
- Unequal devices and asymmetric links
- Placement search, memory-aware schedules, streaming, activation spill
- Native Rust planner + DES
- Concurrent inference with shared capacity accounting
- Versioned save/load artifacts
- HTTP serving (queue, concurrency, cancel, health, Prometheus metrics)
- Opt-in training (resident params + autograd)
- Backend plugins via Python entry points

## Non-goals

- Multi-node cluster scheduling
- Exhaustive placement enumeration
- Guaranteeing every detected GPU is used
- Out-of-core NVMe training
- Hiding unsupported hardware behind optimistic discovery
- Replacing PyTorch as a tensor/kernel framework

## Support levels

1. **Discovered** — resource/backend is visible
2. **Capability-eligible** — backend reports the ops needed for the path
3. **Validated on this host** — target validation passed numerical/execution checks

Only (3) is a production claim for a specific accelerator host.

## Ownership

| Python | Rust |
| --- | --- |
| Capture, partition, public API, serving | Planner, DES, schedule execution |
| Discovery, region compile, diagnostics | Residency, storage, cancellation, telemetry |

Torch region bodies may run via Python callback; scheduling and residency stay runtime-owned. Detail: [Architecture](../architecture/architecture.md).
