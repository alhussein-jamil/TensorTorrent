# Milestone roadmap

## Milestone 1 (current)

- Linux host support
- CPU backend always available
- CUDA / ROCm / MPS / SYCL backends behind capability contracts
- Heterogeneous resource graph (compute, memory, links)
- Portable compile + machine specialization
- Maximal planner with inclusion/exclusion reasons
- Host-staged mixed-vendor fallback
- Discrete-event simulator + chrome tracing
- Packed model storage
- Hardware validation CLI
- Correctness tests on CPU; accelerator checks honest when absent

## Milestone 2

- Dynamic-shape bucket specialization with measured plans per bucket
- Activation offloading with residency tracking
- Tensor parallelism across unequal GPUs
- Pipeline microbatching
- CPU/GPU intra-op splitting with measured schedules
- Quantized storage representations (explicit user mode)
- Stronger contention modeling from profiles
- Online profile refinement feedback loop

## Milestone 3

- Training support
- Multi-node collectives beyond single-machine host staging
- GPUDirect Storage / io_uring fast paths when beneficial
- Native async runtime completion (streams, events, IO queues)
