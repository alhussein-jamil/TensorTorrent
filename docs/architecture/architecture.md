# Architecture

TensorTorrent separates **portable compilation** from **machine specialization**. The portable stage reasons about the PyTorch program. The specialization stage reasons about the machine that will execute it.

<p align="center">
  <img src="../figures/pipeline.svg" alt="TensorTorrent pipeline" width="100%">
</p>

## Design goals

The architecture is built around four constraints:

1. Placement must account for compute, memory capacity, and transfer cost together.
2. The planner must be fast enough to search alternatives, but detailed schedule validation must still model overlap and contention.
3. Runtime residency and data movement must be explicit; backend calls must not hide unscheduled transfers.
4. Hardware support is capability-driven and validated on the target host.

## Phase 1: portable compilation

Portable compilation does not commit to a concrete host topology.

The Python frontend:

- captures the module through the PyTorch export/FX path,
- normalizes the graph,
- partitions it into regions,
- records tensor metadata and liveness,
- prepares parameter packs and portable metadata,
- produces a `PortableArtifact`.

This stage lives primarily under `python/tensortorrent/frontend`, `python/tensortorrent/ir`, and `python/tensortorrent/compile`.

## Phase 2: specialization

Specialization binds a portable program to one machine.

The sequence is:

1. Discover compute, memory, storage, and transfer resources.
2. Resolve effective CPU/RAM/VRAM/disk budgets.
3. Measure region and transfer performance when configured.
4. Build a compact planning problem.
5. Search placements in `tt-planner`.
6. Keep a diverse set of strong finalists.
7. Build executable schedule variants for those finalists.
8. Evaluate them with the Rust discrete-event simulator.
9. Reject infeasible variants and select the winner according to the requested objective.
10. Compile region implementations for the winner only.
11. Build the specialized immutable artifact and runtime executor.

The planner deliberately does **not** call the full DES for every partial beam state. The fast native search is a shortlist mechanism; detailed simulation is the final selector. See [Planner](planner.md).

## Control plane and data plane

### Python control plane

Python owns the parts that need close PyTorch integration:

- public API (`tt.compile`, `tt.load_compiled`),
- export and graph normalization,
- region construction,
- hardware/backend discovery,
- profile orchestration,
- backend-specific region compilation,
- PyTree input/output handling,
- diagnostics and serving integration.

### Rust data plane

Rust owns the high-frequency or stateful systems machinery:

| Crate | Responsibility |
| --- | --- |
| `tt-ir` | schedule and artifact types, IDs, validation |
| `tt-planner` | placement search, pruning, finalist generation |
| `tt-runtime` | execution context, workers, simulator, telemetry |
| `tt-memory` | allocations, residency, copies, leases |
| `tt-storage` | packs, streaming, cache, spill |
| `tt-backend-api` | backend trait surface |
| `tt-backend-cpu` | CPU/NUMA support and host budget enforcement |
| `tt-backend-virtual` | deterministic virtual accelerator used in tests |
| `tt-python` | PyO3 boundary exposed as `tensortorrent._native` |

Torch-backed compute regions may call back into Python for the region body. That does not transfer scheduling authority back to Python: schedule order, residency metadata, events, and storage lifetime remain runtime concerns.

## Artifacts

TensorTorrent uses two artifact levels.

### `PortableArtifact`

Contains graph/region information that can be specialized for another compatible host.

### `SpecializedArtifact`

Contains the machine-specific plan, executable schedule, compiled-region metadata, profile information, and hardware fingerprint.

`CompiledModule.save()` persists the artifact directory. `tt.load_compiled()` verifies the bundle, reloads the exported program, and specializes it for the current machine. `refresh_artifacts=True` writes that fresh specialization back to the directory.

## Execution plan versus executable schedule

These are different concepts.

An **execution plan** answers questions such as:

- which device executes each region,
- which kernel/backend candidate is selected,
- what the estimated latency/throughput/memory cost is,
- which resources were included or rejected.

An **executable schedule** is the concrete ordered/dependency-constrained instruction program consumed by the runtime. It includes compute, transfers, loads, releases, events, and spill-related operations.

DES evaluates the executable schedule, not an abstract device list.

## Direct path

Some resident static plans do not benefit from schedule dispatch. When `prefer_direct_path=True`, TensorTorrent can select a lower-overhead direct path for eligible cases after specialization-time checks.

The schedule path remains mandatory when its semantics are required, including streaming, spill, and training-capable execution.

## Request isolation

Runtime state is request-scoped through `ExecutionContext`. The immutable artifact is shared; mutable instruction state, cancellation, residency/accounting state, and telemetry are associated with the active request.

See [Runtime](runtime.md) for the execution model.

## Hardware assumptions

The planner does not assume:

- identical GPUs,
- symmetric links,
- a single CPU socket,
- CUDA-only execution,
- that every discovered accelerator should be used.

A device is useful only if a feasible plan containing it improves the chosen objective after transfer and memory costs are considered.

See [Heterogeneous hardware](heterogeneous_hardware.md) and [Backends](backends.md).
