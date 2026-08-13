# Runtime

Runs one specialized `ExecutableSchedule`: order work, keep residency, move data, enforce limits, report progress.

<p align="center">
  <img src="../figures/runtime.svg" alt="TensorTorrent runtime execution model" width="92%">
</p>

## ExecutionContext

Per-request state: ID, cancellation, instruction progress, events, tensor/copy handles, allocations, transfers, storage, telemetry. The artifact is immutable and shareable; mutable state is not stored in it. Each forward builds a fresh context so cancel tokens and residency lifetime stay request-scoped.

## Instructions

Vocabulary lives in `tt-ir` (compute, transfer, load, release, events, storage/residency). Work is explicit — backends must not sneak a model-wide `.to(device)` past the schedule.

## Residency

Rust owns which logical tensor version lives where, which copies are live, and which resources are leased. Python may hold torch tensors for a region callback; that is not placement policy.

For inference with a resident parameter store, the runtime may **hoist** scheduled device copies into an executor-owned cache and drop the matching host→device Transfers from the steady-state DAG. Hoist budget authority is shared (`accelerator_hoist_budget_bytes`, live VRAM clamp). If hoist OOMs, recovery rebuilds transfer/evict for that schedule generation only — hoist intent is not permanently disabled. Training never consumes the device cache (optimizers mutate host parameters).

Already-on-device user inputs are seeded on the Transfer destination so native Transfer becomes a residency no-op (no redundant H2D).

## CUDA copy / compute overlap

Schedule IR already names `{resource}::copy0` vs `{resource}::compute` and, with `prefetch_distance > 0`, lets Transfer(i) race Compute(i-1). The runtime binds those ids to real `torch.cuda.Stream` objects: H2D records a CUDA event on the copy stream; Compute waits on that event *on the compute stream* (no CPU barrier). Release and output collection still synchronize the event before dropping storage. Training and CPU-only plans stay on the blocking path.

## Capacity leases

`CompiledModule` owns a `CapacityLedger`. Each forward leases incremental host/device/disk bytes under a module lock; serve (`ModelManager`) tracks request slots only and requires a real ledger. Zero device or disk budgets fail closed. Empty leases still reserve a 1-byte floor so concurrency math stays honest.

## Compute

CPU/native/virtual backends, or torch region callbacks (CUDA/ROCm/XPU, etc.). Concurrent regions when `allow_concurrent_regions=True` and the plan benefits. `process_workers` is opt-in CPU-only — leave it off for accelerator plans (unsafe after accelerator init).

## Storage and spill

`tt-storage` handles packs, prefetch, cache, and spill. Activation spill uses per-forward session dirs cleaned on completion/cancel/failure. See [Large models](../guides/large-models.md) and [Resource budgets](../product/resource_budgets.md).

## Stalls and cancellation

No progress for `stall_timeout_s` → diagnosable stall (not the same as a request timeout). Cancellation is cooperative: stop launching at safe boundaries, raise `ExecutionCancelled`; in-flight work may finish.

Public API: `CompiledModule.forward_with_cancel_token(...)`. Cancel uses a **generation** so a completing sibling cannot clear a sticky cancel meant for another in-flight forward. Serve request timeouts cancel the **per-request** token only — never `module.request_cancel()`, which would abort every concurrent forward.

## Direct path

`prefer_direct_path=True` can skip the dispatcher for eligible resident static plans (single-region `DirectPlan`, or multi-region `DataflowDirectPlan` when enabled). Streaming, spill, training, mid-forward cancel tokens, and schedule telemetry force the schedule path.

Direct plans keep ambient PyTorch intra-op thread budgets (intra-op pinch applies only to the schedule microbench path). Accelerator bindings must resolve to a concrete torch device or DirectPlan is refused.

## Observability

`explain()`, visualization, serving metrics, and validation share the same schedule representation the simulator ranked.
