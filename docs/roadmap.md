# Milestone roadmap

Status words follow the README: **implemented** means a test in this repository runs
it, **untested** means the code path exists but no machine here could execute it,
**simulated** means an analytic model stands in for hardware, and **planned** means it
is not built.

This document has **no planned items left**. Everything formerly listed under
Milestones 1–3 is either implemented with tests, implemented but untested on
absent hardware, or explicitly simulated and labelled as such.

## Milestone 1 — truthful CPU vertical path

Implemented (tests in-repo):

- `torch.export` capture, region partitioning, heterogeneous IR lowering
- CPU backend compiling and executing every region
- `CompiledModule` as a real `nn.Module` with eager-matching outputs
- Dependency-aware region scheduling; concurrency only when measured faster
- Measured region latencies; packed weight streaming under RAM budget
- Alias analysis, activation budget with disk spill / recompute policies
- Shared `ExecutableSchedule`; GraphExecutor schedule-driven Compute/Release
- TorchInductor optional regions (default on; keep when measured ≥ eager)
- `TensorDirectory`, host memcpy / disk-pread transfers, transfer join
- `ActivationAllocator` on single-worker runs; cancel API; dispatch ceiling test
- Hardware discovery + validation CLI

Untested here: CUDA / ROCm / MPS / SYCL region execute; NCCL / RCCL / oneCCL.

Simulated: analytic transfer/makespan models (always labelled).

## Milestone 2 — heterogeneous execution

Implemented (tests in-repo):

- `TorchDeviceTransfer` real `Tensor.to` path selected when the destination device
  is available; otherwise `SimulatedDeviceTransfer` (labelled simulated)
- Dynamic-shape `ShapeBucket` / `BucketedModule` dispatch by batch-size range
- Activation overflow policies: `spill` and `recompute` (`NeedsRecompute`)
- Host-staged tensor-parallel shard / gather / allreduce-sum
  (`runtime/tensor_parallel.py`) for unequal-device fallback
- Pipeline microbatching (`MicrobatchPlan` / `run_pipeline_microbatched`)
- CPU intra-op chunk split (`IntraOpSplit` / `run_intraop_split`)
- Quantized storage pack/dequant when `allow_quantized_storage` /
  `numerical_mode=quantized` (`storage/quantized.py`)
- Measured contention injection (`set_measured_compute_contention`) plus
  `refine_contention_from_overlaps`
- Online profile feedback (`ProfileFeedback` on each `forward`)

Untested here: measured overlapping CPU+GPU run with real device DMA; vendor
device collectives on accelerators.

Simulated: device transfers when no accelerator is present.

## Milestone 3 — beyond one process

Implemented (tests in-repo):

- `ProcessWorkerPool` (spawn + tensor-safe IPC) for mixed-vendor isolation
- `CompileConfig.allow_training` opts out of the inference-mode guard
- Gloo allreduce: uses `torch.distributed` when a process group is initialized;
  otherwise host-staged sum (multi-node bring-up path)
- Storage fast-path selector: validated `os.pread`, optional io_uring/GDS hooks
  that engage only when bindings exist (`storage/fastpath.py`)
- Native async helpers: `AsyncEvent` / streams wrap CUDA events when CUDA is
  present; CPU WaitEvent bookkeeping otherwise (`runtime/async_events.py`)

Untested here: multi-node process groups on a real cluster; cuFile/GDS; io_uring
with a production binding; mixed CUDA+ROCm in one plan on hardware.

## Remaining planned work

None.
