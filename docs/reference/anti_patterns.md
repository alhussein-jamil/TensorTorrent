# Architectural anti-patterns

This document records invariants that should not be weakened during feature work or optimization.

## Planning and hardware

### Hard-coding a CUDA/identical-GPU worldview

Do not bake assumptions such as identical accelerator memory, symmetric bandwidth, one CPU socket, or mandatory CUDA into planner code. Express hardware through backend capabilities and the resource graph.

### Forcing every discovered device into the plan

Discovery is an option set. The planner must be free to exclude resources whose transfer/memory cost makes the objective worse.

### Treating discovery as validation

A visible device is not proof that execution works correctly on the target driver/runtime combination. Keep target-host validation explicit.

### Maintaining two production planners

The native Rust planner is authoritative for hot placement search. Do not reintroduce a second full Python beam-search implementation that can drift semantically.

### Divergent transfer models

Planner and DES may differ in fidelity, but they must share the same underlying link identity and coefficients. Do not add an unrelated transfer-cost formula to one side.

## Scheduling and runtime

### Hidden data movement

Do not insert backend-side model/tensor copies that are invisible to the executable schedule. Residency, memory feasibility, and simulation become meaningless when transfers occur outside the schedule model.

### Two executable schedule representations

Keep one authoritative schedule IR. Simulation and execution should consume the same representation rather than translating into two independently evolving executors.

### Unbounded resource waits

A wait loop without progress detection can turn a lost completion into an infinite hang. Runtime waits must remain bounded by progress-aware stall detection unless the user explicitly disables it.

### Unbounded service threads or queues

Do not accept unlimited HTTP connections or inference queue growth. Backpressure must happen before process resources are exhausted.

## Memory and storage

### Planning from machine totals

Raw host RAM or GPU capacity is not the usable budget. Respect cgroups, affinity, current availability, explicit limits, reserves, and headroom.

### Spilling to RAM-backed filesystems

`tmpfs`/`ramfs` does not relieve RAM pressure. Keep the default refusal for activation spill.

### Ignoring workspace memory

A kernel whose parameters fit can still be infeasible because workspace/activation requirements exceed the memory resource. Capacity checks must include the actual working set represented by the planner/runtime.

## Performance

### Compiling every finalist

Detailed schedule selection happens before expensive region compilation. Keep it that way; compile only the winner.

### Parallelism by default at any cost

“Parallel enabled” means parallelism is permitted and bounded, not that every tiny search should create worker overhead. Preserve automatic serial fallback for small planner/DES workloads.

### Keeping a slower compiled kernel because compilation succeeded

Under competitive profiling, compiled implementations should be retained because measurements justify them, not because `torch.compile` returned successfully.

### Presenting simulation as measurement

Label measured and modeled values accurately. Do not present cache hits, priors, or DES predictions as hardware measurements.
