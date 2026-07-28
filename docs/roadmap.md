# Milestone roadmap

Status words follow the README: **implemented** means a test in this repository runs
it, **untested** means the code path exists but no machine here could execute it,
**simulated** means an analytic model stands in for hardware, and **planned** means it
is not built.

## Milestone 1 — truthful CPU vertical path (current)

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
- Host activation peak estimate / `activation_budget_bytes`; optional `vram_budget_bytes`
- Artifact save and reload through `torch.export.save`
- Hardware discovery for CPUs, NUMA pools, memory tiers and links
- Hardware validation CLI that runs the compiled path and skips absent accelerators
- Throughput objective minimizes makespan (no inverted score)
- Device-specific profiling cache keys (device, shapes, dtype, kernel, threads)
- Explicit residency/transfer schedule for future CPU–GPU plans (unvalidated)
- Shared `ExecutableSchedule` (Compute/Transfer/Prefetch/Load/Release) for
  planner, simulator, and runtime introspection
- Optional `torch.compile` / TorchInductor region compilation with eager FX
  fallback; fingerprint-keyed process cache
- Central `TensorDirectory` residency state machine; explicit host memcpy and
  disk-pread transfer backends (device transfers simulated until validated)
- Liveness derived from producer–consumer edges; alias groups cover views and
  reject mutable shared weights
- Measured Chrome/HTML execution telemetry (`visualize(measured=True)`)
- `ExecutableSchedule` structural validation (`validate_schedule`): duplicate
  instruction ids, dangling dependencies, dependency cycles,
  release-before-use, and compute-before-transfer-completion are rejected
  before the planner's output reaches the simulator or `GraphExecutor`
- `TensorDirectory` joins two concurrent requests for the same
  tensor+destination transfer into one real copy instead of duplicating I/O
- `ActivationAllocator`: liveness-derived buffer-reuse slots are backed by one
  real physical byte buffer (`data_ptr()` equality proves reuse is not just a
  compile-time count); not yet wired into `GraphExecutor`'s live dispatch

Untested here (no accelerator available): CUDA / ROCm / MPS / SYCL backends, NCCL /
RCCL / oneCCL collectives.

Simulated: transfer costs and plan makespan with tensor lifetime accounting
(including overlapping shared-memory state); always labelled simulated, never
claimed validated.

Remaining in this milestone:

- Reduce the fixed per-call dispatch overhead (~6–15 microseconds today on tiny
  linears; larger models already sit at eager parity) so micro-models are not
  slower than eager
- Execute a region on a GPU on a machine that has one, and record the measurement
- Drive GraphExecutor Compute purely from `ExecutableSchedule` opcodes (today
  Compute still walks `RegionProgram`; Transfers already come from the schedule)
- Route `GraphExecutor` activation releases through the schedule's `Release`
  instructions instead of the executor's own consumer-count bookkeeping (both
  are liveness-correct today but are two independent implementations of the
  same decision)
- Wire `ActivationAllocator` into live region dispatch so two non-overlapping
  activations physically share memory during a real compiled run, not only in
  the standalone allocator test
- No cancellation API exists (`GraphExecutor` has no way to abort an in-flight
  `run()` and release partial resources); a caller can only wait for
  completion or let an exception propagate
- Activation disk offload / recompute (peak `activation_budget_bytes` is enforced;
  spilling activations is still planned)
- Enable `use_torch_compile` by default after broader CPU wins are confirmed
  (optional today; Inductor is kept only when measured faster than eager FX)

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
- Activation offloading with residency tracking
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
