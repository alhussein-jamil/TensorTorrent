# Product scope

TensorTorrent is a **single-host heterogeneous execution compiler/runtime for PyTorch**, with inference as the primary workload and an opt-in resident-parameter training path.

The project is alpha. Production readiness is evaluated per target host, not inferred from backend discovery.

## In scope

- PyTorch graph capture and region partitioning.
- One host with CPU/NUMA resources and zero or more supported accelerators.
- Unequal devices and asymmetric transfer paths.
- Placement search across device subsets.
- Memory-aware schedules under explicit or resolved RAM/VRAM limits.
- Parameter streaming from slower tiers.
- Activation spill where enabled and compatible.
- Native Rust planner and discrete-event schedule simulation.
- Concurrent inference with shared capacity accounting.
- Save/load of versioned compiled artifacts.
- HTTP serving with queue/concurrency limits, cancellation, health, readiness, and Prometheus-format metrics.
- Opt-in training with resident parameters and autograd.
- Backend plugins through Python entry points.

## Explicit non-goals

- Multi-node cluster scheduling.
- Exhaustive enumeration of every legal placement.
- Guaranteeing that every detected GPU should be used.
- Out-of-core NVMe training.
- Hiding unsupported hardware behind optimistic discovery results.
- Replacing PyTorch kernels with a completely independent tensor framework.

## Support model

There are three distinct levels:

1. **Discovered** — TensorTorrent can see a resource/backend.
2. **Capability-eligible** — the backend reports the operations required for the attempted path.
3. **Validated on this host** — target-machine validation has exercised execution and numerical checks.

Only the third is a meaningful production statement for a specific accelerator host.

## Ownership boundary

| Python | Rust |
| --- | --- |
| PyTorch capture and integration | planner search |
| graph normalization/partitioning | executable schedule and validation |
| backend discovery/orchestration | DES |
| region implementation compilation | residency/accounting |
| public API and serving interface | data movement/storage coordination |
| diagnostics and artifact orchestration | request execution/cancellation/telemetry |

Torch region bodies may execute through a Python callback. Scheduling and residency remain runtime-owned.
