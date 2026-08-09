# Runtime

Runs one specialized `ExecutableSchedule`: order work, keep residency, move data, enforce limits, report progress.

<p align="center">
  <img src="../figures/runtime.svg" alt="TensorTorrent runtime execution model" width="92%">
</p>

## ExecutionContext

Per-request state: ID, cancellation, instruction progress, events, tensor/copy handles, allocations, transfers, storage, telemetry. The artifact is immutable and shareable; mutable state is not stored in it.

## Instructions

Vocabulary lives in `tt-ir` (compute, transfer, load, release, events, storage/residency). Work is explicit — backends must not sneak a model-wide `.to(device)` past the schedule.

## Residency

Rust owns which logical tensor version lives where, which copies are live, and which resources are leased. Python may hold torch tensors for a region callback; that is not placement policy.

## Compute

CPU/native/virtual backends, or torch region callbacks (CUDA/ROCm/XPU, etc.). Concurrent regions when `allow_concurrent_regions=True` and the plan benefits. `process_workers` is opt-in CPU-only — leave it off for accelerator plans (unsafe after accelerator init).

## Storage and spill

`tt-storage` handles packs, prefetch, cache, and spill. Activation spill uses per-forward session dirs cleaned on completion/cancel/failure. See [Large models](../guides/large-models.md) and [Resource budgets](../product/resource_budgets.md).

## Stalls and cancellation

No progress for `stall_timeout_s` → diagnosable stall (not the same as a request timeout). Cancellation is cooperative: stop launching at safe boundaries, raise `ExecutionCancelled`; in-flight work may finish.

## Direct path

`prefer_direct_path=True` can skip the dispatcher for eligible resident static plans. Use the schedule path for streaming, spill, training, or schedule telemetry.

## Observability

`explain()`, visualization, serving metrics, and validation share the same schedule representation the simulator ranked.
