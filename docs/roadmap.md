# Milestone roadmap

Status words match the README capability table. This file lists only what is
**not** yet first-class through `compile()` / production validation — so it does
not repeat the implemented matrix.

## Experimental scaffolding (helpers exist; not schedule-driven via `compile()`)

- Shape buckets / `BucketedModule`
- Host-staged tensor-parallel shard / gather (`runtime/tensor_parallel.py`)
- Pipeline microbatching (`MicrobatchPlan`)
- CPU intra-op chunk split
- Storage fast-path selector hooks beyond validated `os.pread` (`storage/fastpath.py`)
- Gloo allreduce helper when a process group exists (not emitted by the planner)

## Untested here (need real hardware / cluster)

- CUDA / ROCm / MPS / SYCL region execute on production accelerators
- Measured overlapping CPU+GPU with real device DMA / CUDA events
- NCCL / RCCL / oneCCL collectives; multi-node process groups
- cuFile/GDS; production io_uring bindings
- Mixed CUDA+ROCm in one plan on hardware

## Simulated (analytic / virtual)

- Discrete-event makespan and transfer exposure on `ExecutableSchedule`
- `SimulatedDeviceTransfer` when destination hardware is absent
- `mock_accel` stream delays modelling device work (schedule semantics only)

## Planned

- Promote tensor / pipeline parallelism to schedule-driven planner strategies
- Wire shape buckets into `compile()` when dynamic shapes are supported
- Activation **recompute** overflow policy (today only schedule spill is supported;
  `activation_overflow_policy="recompute"` is rejected)
- Real CUDA streams/events behind existing `ExecutionStream` / `BackendEvent` APIs
