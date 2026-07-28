# Milestone roadmap

Status words follow the README: **implemented** means a test in this repository runs
it, **untested** means the code path exists but no machine here could execute it,
**simulated** means an analytic model stands in for hardware, and **planned** means it
is not built.

## Milestone 1 — truthful CPU vertical path (complete on CPU hosts)

Implemented:

- `torch.export` capture, region partitioning, heterogeneous IR lowering
- CPU backend compiling and executing every region
- `CompiledModule` as a real `nn.Module` with eager-matching outputs
- Dependency-aware region scheduling, with concurrency enabled only when measured
  faster
- Measured region latencies feeding the planner; unmeasured candidates labelled
- Packed weight storage with disk streaming under a RAM budget, prefetch and double
  buffering
- Alias analysis groups shared parameter/buffer storage; streaming caches by storage id
- Host activation peak estimate / `activation_budget_bytes`; runtime disk spill + reload
  when the live peak would exceed the budget
- Artifact save and reload through `torch.export.save`
- Hardware discovery for CPUs, NUMA pools, memory tiers and links
- Hardware validation CLI that runs the compiled path and skips absent accelerators
- Throughput objective minimizes makespan (no inverted score)
- Device-specific profiling cache keys (device, shapes, dtype, kernel, threads)
- Explicit residency/transfer schedule for future CPU–GPU plans (unvalidated)
- Shared `ExecutableSchedule` (Compute/Transfer/Prefetch/Load/Release) for
  planner, simulator, and runtime; GraphExecutor walks schedule Compute order and
  fires Release ops when all consumer Computes complete (concurrency-safe)
- Optional `torch.compile` / TorchInductor region compilation with eager FX
  fallback; **on by default**, kept only when measured ≥ eager (within 5%)
- Central `TensorDirectory` residency state machine; explicit host memcpy and
  disk-pread transfer backends (device transfers simulated until validated)
- Liveness derived from producer–consumer edges; alias groups cover views and
  reject mutable shared weights
- Measured Chrome/HTML execution telemetry (`visualize(measured=True)`)
- `ExecutableSchedule` structural validation (`validate_schedule`) and
  `validate_schedule_resources`
- `TensorDirectory` joins concurrent same-destination transfers
- `ActivationAllocator` wired into live dispatch for single-worker runs:
  buffer-reuse slots share one physical buffer (`data_ptr()` equality). Disabled
  when `max_workers > 1` because sequential liveness is unsafe under concurrent
  region overlap
- `GraphExecutor.request_cancel` / `CompiledModule.request_cancel` /
  `ExecutionCancelled`
- Micro-dispatch overhead bounded by regression test (tiny `Linear` stays under
  a 40 µs delta ceiling; larger models approach eager parity)

Untested here (no accelerator available): CUDA / ROCm / MPS / SYCL backends, NCCL /
RCCL / oneCCL collectives. GPU region execution code exists and refuses to
fabricate results when the device is absent; measured CPU–GPU overlap is a
Milestone 2 validation item.

Simulated: transfer costs and plan makespan with tensor lifetime accounting
(including overlapping shared-memory state); always labelled simulated, never
claimed validated.

Remaining in this milestone: none.

## Milestone 2 — heterogeneous execution

Next concrete step toward real CPU–GPU simultaneous execution: on a machine with
at least one GPU, specialize a two-region plan with one CPU placement and one GPU
placement, execute the `ExecutableSchedule` Transfer/Wait path against real
device copies (replace `SimulatedDeviceTransfer`), and assert measured overlap
plus numerical equivalence. Do not mark concurrent execution validated until that
run exists.


- CPU and GPU regions executing concurrently in one plan (residency/transfer
  schedule exists; measured overlapping CPU+GPU run still required)
- Dynamic-shape bucket specialization with measured plans per bucket
- Stronger activation offload policies (recompute as an alternative to disk spill)
- Tensor parallelism across unequal GPUs
- Pipeline microbatching
- CPU/GPU intra-op splitting with measured schedules
- Quantized storage representations (explicit user mode)
- Stronger contention modeling from profiles
- Online profile refinement feedback loop

## Milestone 3 — beyond one process

- Separate worker processes for mixed-vendor stacks
- Training support
- Multi-node collectives beyond single-machine host staging
- GPUDirect Storage / io_uring fast paths when beneficial
- Native async runtime completion (streams, events, IO queues)
