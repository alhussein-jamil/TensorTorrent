# Anti-patterns

Rejected by design:

1. CUDA-only or identical-GPU assumptions
2. One-socket / unified-memory / fixed-bandwidth assumptions
3. Vendor conditionals scattered through the planner
4. Silent ignore of detected resources, or forcing every device busy
5. Claiming compatibility without backend validation
6. Treating a CPU-only host as proof of GPU execution
7. Two executors or two plan IRs that disagree
8. Hidden `tensor.to(device)` transfers outside the schedule
9. Keeping slower `torch.compile` over eager FX
10. Labelling simulated / cache-hit latencies as measured
11. Advertising unimplemented overflow policies (`recompute`)

Enforced by: `ExecutionBackend` queries, `ResourceGraph`, validation statuses,
fingerprint-gated cache, single `ExecutableSchedule`, and fail-closed residency.
