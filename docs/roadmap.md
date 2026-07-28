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
- Artifact save and reload through `torch.export.save`
- Hardware discovery for CPUs, NUMA pools, memory tiers and links
- Hardware validation CLI that runs the compiled path and skips absent accelerators

Untested here (no accelerator available): CUDA / ROCm / MPS / SYCL backends, NCCL /
RCCL / oneCCL collectives.

Simulated: transfer costs and plan makespan, reported as `simulated_makespan_s`.

Remaining in this milestone:

- Reduce the fixed per-call dispatch overhead (~6–15 microseconds today on tiny
  linears; larger models already sit at eager parity) so micro-models are not
  slower than eager
- Execute a region on a GPU on a machine that has one, and record the measurement
- Activation RAM budgets / offload (parameter streaming is implemented; activation
  residency tracking beyond peak-byte reporting is still thin)

## Milestone 2 — heterogeneous execution

- CPU and GPU regions executing concurrently in one plan
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
