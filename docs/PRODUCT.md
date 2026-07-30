# Product scope

StreamCompiler is a **single-machine heterogeneous inference runtime** for PyTorch.

## In scope

- PyTorch inference (`torch.export` / FX control plane)
- One host: many CPU cores, NUMA domains, one or many GPUs
- Models larger than device or host RAM (parameter streaming, activation spill)
- Concurrent inference requests with shared capacity accounting
- Ahead-of-time compiled regions + immutable `ExecutableArtifact`
- Rust data plane owns scheduling, residency, transfers, storage, telemetry

## Out of scope (now)

- Training / autograd through the schedule
- Multi-node distributed execution
- Tensor-parallel collectives that need custom all-reduce fabrics
- Arbitrary dynamic Python in the serving hot path
- Claiming production readiness for untested accelerators (CUDA/ROCm until hardware-validated)

Multi-node interfaces may exist as stubs; they must not run on the production path.

## Ownership

| Plane | Owns |
| --- | --- |
| Python | export, normalize, partition, AOT region compile, PyTrees, public API, diagnostics |
| Rust | artifact, topology, schedule, workers, memory, transfers, storage, streams/events, cancel, telemetry, request lifecycle |

After `load` / `warm`, the serving hot path must not call Python once per schedule instruction for scheduling. Temporary Python compute callbacks may remain during migration; the target is native AOT region launch.

## One runtime

There is one production runtime: the Rust dispatcher.

A Python DAG executor may exist only under a testing-only namespace for differential / oracle benchmarks. It must never activate from `CompiledModule.forward` or the serving layer.

## Readiness labels

| Label | Meaning |
| --- | --- |
| **measured** | Exercised on this commit / machine with numbers |
| **simulated** | Analytic DES / virtual backend only |
| **experimental** | Wired but incomplete |
| **untested** | Code present; no real-hardware validation |
| **blocked** | Architecture complete; waiting on hardware or dependency |

Real multi-GPU production readiness is **blocked** until GPU worker isolation tests pass on real hardware.
