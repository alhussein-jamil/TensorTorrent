# Anti-patterns (rejected)

The following shortcuts are explicitly rejected by this architecture:

1. Assuming CUDA is the only accelerator backend
2. Assuming all GPUs are identical
3. Assuming one CPU socket
4. Assuming unified CPU memory latency
5. Assuming direct GPU peer-to-peer transfers
6. Assuming one disk
7. Assuming fixed bandwidth values
8. Assuming theoretical peak throughput predicts real performance
9. Compiling one global kernel format for every backend
10. Duplicating the compiler for each hardware vendor
11. Putting vendor-specific logic throughout the planner
12. Silently ignoring detected resources
13. Using every resource without measuring whether it helps
14. Claiming compatibility without real backend validation
15. Treating a CPU-only development host as proof that GPU execution works
16. Maintaining two production executors or two plan IRs that disagree
17. Hiding device transfers inside `tensor.to(device)` without plan instructions
18. Keeping `torch.compile` when it is slower than eager FX on the measured examples

Concrete enforcement mechanisms:

- `ExecutionBackend` capability queries
- independent `ResourceGraph` nodes/edges
- host-staged communication fallback
- validation statuses distinguishing unsupported vs validated
- planner decision explanations for inclusion/exclusion
- fingerprint-gated specialization cache invalidation
- single `GraphExecutor` + shared `ExecutableSchedule`
- measured Inductor keep-or-fallback in `torch_device.compile_region_for_torch_device`
