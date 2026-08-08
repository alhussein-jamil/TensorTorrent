# Runtime

The runtime executes one specialized `ExecutableSchedule`. Its main responsibilities are ordering work, maintaining residency state, coordinating data movement, enforcing resource limits, and reporting progress.

<p align="center">
  <img src="../figures/runtime.svg" alt="TensorTorrent runtime execution model" width="92%">
</p>

## `ExecutionContext`

Each request receives an execution context containing request-local state such as:

- request ID and cancellation state,
- instruction progress,
- event state,
- tensor/copy handles,
- allocations and residency metadata,
- transfers and storage state,
- telemetry.

The compiled artifact is immutable and can be shared across requests. Mutable execution state is not stored in the artifact.

## Schedule instructions

The exact instruction vocabulary is defined by `tt-ir`. At runtime, schedules can express operations such as compute, transfer, load, release, synchronization/events, and storage/residency operations.

The runtime treats those operations as explicit work. A backend should not silently perform a hidden model-wide `.to(device)` that bypasses the schedule.

## Residency authority

Rust owns the authoritative residency model: which logical tensor version exists in which physical allocation, which copies are live, and which resources are leased.

Python may hold local tensor objects required to execute a torch region, but that object store is not the source of placement policy.

This boundary prevents the planner, simulator, and runtime from disagreeing about where data is expected to live.

## Compute execution

Compute can be executed through:

- CPU/native or virtual backend paths where supported,
- torch-backed region callbacks for CUDA/ROCm/XPU and other PyTorch execution paths.

Independent regions can execute concurrently when `allow_concurrent_regions=True` and the selected plan contains useful concurrency.

`process_workers` is a separate, opt-in CPU-oriented mechanism. It is disabled by default and should remain disabled for accelerator plans because forking after accelerator runtime initialization is unsafe.

## Storage and spill

`tt-storage` provides parameter-pack access, prefetch, caching, and spill primitives.

When activation spill is enabled, runtime sessions use per-forward spill directories and clean them after completion, cancellation, or failure. Startup cleanup can remove orphaned session directories left by dead processes.

See [Large models](../guides/large-models.md) and [Resource budgets](../product/resource_budgets.md).

## Progress and stalls

Resource waits are progress-aware. The executor tracks a progress generation and raises a diagnosable stalled error when no work completes for `stall_timeout_s`.

This is different from a request timeout: it is intended to detect a runtime that has stopped making progress while waiting for a resource or completion.

## Cancellation

Cancellation is cooperative. The dispatcher stops launching new work at safe schedule boundaries and raises `ExecutionCancelled`; work already in flight may complete before the request exits.

## Direct path versus schedule path

The schedule executor is not forced on simple resident graphs. `prefer_direct_path=True` lets eligible static plans use a direct lower-overhead path when specialization-time timing shows that the schedule adds no value.

Use the schedule path when you need streaming, activation spill, training-capable execution, schedule telemetry, or other semantics that require the full dispatcher.

## Observability

Runtime and specialization expose structured information used by `compiled.explain()`, visualization, serving metrics, and validation tools. The simulator and runtime share the same schedule representation so the simulated object is the object eventually executed.
