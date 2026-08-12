# Architecture

Portable compilation (the PyTorch program) stays separate from machine specialization (this host).

<p align="center">
  <img src="../figures/pipeline.svg" alt="TensorTorrent pipeline" width="100%">
</p>

## Goals

1. Placement weighs compute, memory capacity, and transfer cost together.
2. Search stays cheap; DES only ranks a bounded finalist set.
3. Residency and data movement are explicit — backends must not hide unscheduled transfers.
4. Hardware support is capability-driven and validated on the target host.

## Portable compilation

No host topology commitment. Frontend captures (`torch.export` / FX), partitions into regions, records tensor metadata and parameter packs, emits a `PortableArtifact`.

Code: `python/tensortorrent/frontend`, `ir`, `compile`.

## Specialization

Bind a portable program to one machine:

1. Discover resources; resolve CPU/RAM/VRAM/disk budgets.
2. Measure region/transfer perf when configured.
3. Search placements in `tt-planner`; keep a diverse top-K.
4. Build schedule variants; rank with Rust DES.
5. Compile region impls for the **winner only**.
6. Emit specialized artifact + runtime executor.

Native search shortlists; DES picks. See [Planner](planner.md).

## Python vs Rust

| Python | Rust |
| --- | --- |
| Public API, export, partitioning | `tt-planner` search and finalists |
| Discovery, profiling, region compile | `tt-runtime` schedule, residency, DES |
| Serving, diagnostics, artifacts | `tt-storage` packs, prefetch, spill |
| | `tt-ir`, `tt-memory`, backend crates |

Torch region bodies may call back into Python. Schedule order, residency, and storage lifetime stay runtime-owned. Crate map: `crates/`.

## Artifacts

- **`PortableArtifact`** — graph/regions that can be specialized on another compatible host.
- **`SpecializedArtifact`** — plan, executable schedule, compiled-region metadata, profile, hardware fingerprint.

`CompiledModule.save()` / `tt.load_compiled()` persist and reload. `refresh_artifacts=True` rewrites specialization for the current machine.

## Plan vs schedule

An **execution plan** chooses devices, kernels, and estimated cost. An **executable schedule** is the ordered instruction program the runtime runs (compute, transfer, load, release, events, spill). DES ranks schedules, not abstract device lists.

## Direct path

When `prefer_direct_path=True`, eligible resident static plans can skip schedule dispatch. Streaming, spill, training, and mid-forward cancel tokens still require the schedule path. Details: [Runtime](runtime.md).

## Hardware

No assumption of identical GPUs, symmetric links, or “use every accelerator.” A device is kept only if a feasible plan containing it improves the objective after transfer and memory cost. See [Heterogeneous hardware](heterogeneous_hardware.md) and [Backends](backends.md).
