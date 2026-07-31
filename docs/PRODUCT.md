# Product scope

StreamCompiler is a **single-machine multi-CPU / multi-GPU inference runtime** for PyTorch.

## In scope

- PyTorch inference (`torch.export` / FX control plane)
- One host: many CPU cores, NUMA domains, one or many GPUs
- Models larger than device or host RAM (parameter streaming, activation spill)
- Concurrent inference requests with shared capacity accounting
- Ahead-of-time compiled regions + immutable `ExecutableArtifact`
- Rust data plane owns scheduling, residency, transfers, storage, telemetry

## Out of scope

- Training / autograd through the schedule
- Multi-node distributed training clusters
- Arbitrary dynamic Python in the serving hot path

## Ownership

| Plane | Owns |
| --- | --- |
| Python | export, normalize, partition, AOT region compile, PyTrees, public API, diagnostics |
| Rust | artifact, topology, schedule, workers, memory, transfers, storage, streams/events, cancel, telemetry, request lifecycle |

After `load` / `warm`, the Rust dispatcher runs the schedule. Torch compute regions may still invoke a Python callback to execute the region body; scheduling, residency, and transfers stay in Rust.
