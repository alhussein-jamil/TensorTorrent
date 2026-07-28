# Milestone roadmap

Status words follow the README: **implemented** means a test in this repository runs
it end-to-end through the compiler/runtime path that matters, **experimental
scaffolding** means a helper API exists but is not connected to `compile()` as the
sole schedule-driven path, **untested** means the code path exists but no machine
here could execute it, **simulated** means an analytic model stands in for
hardware, and **planned** means it is not built.

## Implemented (end-to-end / schedule-driven)

- `torch.export` capture, region partitioning, heterogeneous IR lowering
- CPU backend compiling and executing regions; `CompiledModule` as a real `nn.Module`
- Dependency-aware region scheduling; concurrency only when measured faster
- Shared `ExecutableSchedule` executed as an instruction dependency DAG
  (`ScheduleExecutor`): Prefetch, Load, Transfer, RecordEvent, WaitEvent,
  Compute, Evict, Release — not converted back into a region-prelude scheduler
- Discrete-event simulator walks that same `ExecutableSchedule` (same instruction
  IDs/deps); planning profile source=`executable_schedule`
- Multi-copy residency: `CopyStore` keyed by `(tensor_id, resource_id)`; CPU and
  accelerator copies coexist; replication does not bump logical versions
- Real event registry / mock async streams: RecordEvent incomplete until transfer
  future completes; WaitEvent waits that handle; no host sync at Transfer enqueue
- Backend-owned resource→`torch.device` mapping (CUDA / ROCm / XPU / MPS / CPU /
  mock accel); ROCm and SYCL are not mis-routed to CPU
- Schedule-managed placement: compute regions do not hide `.to(device)` moves
- Async `Tensor.to(..., non_blocking=True)` on real device transfers when available
- Host memcpy / disk-pread transfers; simulated device DMA when no accelerator
- Mock accelerator backend; `compile(..., machine=, measurements=)` drives CPU+accel
  partition without a GPU (stream delays, not caller-thread `sleep()`)
- `allow_training=True` is an autograd-compatible **graph-module fallback**
  (grads work); it is **not** heterogeneous compiled training through the schedule
- Online `ProfileFeedback` → `apply_profile_feedback()` / `replan_with_profile_feedback()`
  re-specializes and swaps the live executor
- Persistent nonblocking `ProcessWorkerPool`; `CompileConfig.process_workers>0` attaches
  a Linux-fork pool (not mixed-vendor process isolation; fork CoW / CUDA caveats apply)
- Quantized storage on the pack path is **experimental / opt-in**
  (`allow_quantized_storage` + `numerical_mode=quantized`); streaming dequantizes —
  not a quantized compute-kernel path
- Alias analysis, activation budget with disk spill / recompute policies
- TorchInductor optional regions (default on; keep when measured ≥ eager); compile/
  warm-up may fall back to eager FX, but accepted Inductor regions propagate
  runtime errors instead of silently switching
- Hardware discovery + validation CLI

## Experimental scaffolding (not compile()-integrated)

Helpers with unit tests that are **not** yet first-class planner/schedule strategies:

- Shape buckets / `BucketedModule`
- Host-staged tensor-parallel shard / gather (`runtime/tensor_parallel.py`)
- Pipeline microbatching (`MicrobatchPlan`)
- CPU intra-op chunk split
- Storage fast-path selector hooks beyond validated `os.pread` (`storage/fastpath.py`)
- Gloo allreduce helper (uses `torch.distributed` when a process group exists)

## Untested here (need real hardware / cluster)

- CUDA / ROCm / MPS / SYCL region execute on production accelerators
- Measured overlapping CPU+GPU run with real device DMA
- NCCL / RCCL / oneCCL collectives; multi-node process groups
- cuFile/GDS; io_uring with a production binding
- Mixed CUDA+ROCm in one plan on hardware

## Simulated

- Analytic transfer/makespan models (always labelled)
- `SimulatedDeviceTransfer` when destination hardware is absent
- Mock accelerator host sleeps modeling device work

## Planned

- Promote tensor / pipeline parallelism to schedule-driven planner strategies
- Wire shape buckets into `compile()` when dynamic shapes are supported
