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
11. Advertising overflow policies other than activation spill

Enforced by: `ExecutionBackend` queries, `ResourceGraph`, validation statuses,
fingerprint-gated cache, single `ExecutableSchedule`, and fail-closed residency.

## Resource-budget anti-patterns

The following patterns are rejected by design or produce incorrect behaviour.

12. **Sizing from machine totals instead of resolved budgets.**
    Reading `/proc/meminfo MemTotal` or `nvidia-smi` VRAM to decide how much
    memory a workload may use ignores cgroup limits, current allocations, OS
    overhead, and headroom. Always use the resolved budget from
    `python/tensortorrent/hardware/budget.py` or `tt-backend-cpu::host_budget`.
    Containers sized from host totals are OOMKilled; GPUs sized from total VRAM
    produce OOM errors mid-inference. Enforced by the budget resolver and the
    early fit gate (`MemoryCapacityError`).

13. **Spilling to RAM-backed filesystems (`tmpfs` / `ramfs`).**
    Writing activation spill to `/tmp` (or any tmpfs/ramfs mount) does not free
    device memory — it consumes RAM instead. It also silently exhausts the
    system page cache under sustained load. TensorTorrent refuses this
    configuration with `ConfigurationError` at startup. Set `TT_SPILL_DIR` or
    `CompileConfig.spill_dir` to a persistent path. Use `TT_ALLOW_TMPFS_SPILL=1`
    only for isolated testing.

14. **Unbounded waits without a progress watchdog.**
    Spinning on `yield_now()` or sleeping without a deadline turns a lost
    completion into a silent hang that consumes a CPU core indefinitely. All
    resource-acquisition loops must track a progress generation counter and
    raise a diagnosable error after `stall_timeout_s` seconds of no progress.
    The Rust data plane enforces this via `wait_for_resource` and the completion
    watchdog in `tt-runtime::executor`. Enforced by: `RuntimeError::Stalled`.

15. **Per-connection unbounded threads.**
    Spawning one OS thread per HTTP connection with no cap leads to resource
    exhaustion under load. The HTTP server uses `TT_HTTP_MAX_CONNECTIONS` (default 128)
    as a hard connection cap enforced at `accept` time. Excess connections
    receive an immediate HTTP 503 with `Retry-After: 1` and are never handed to
    the thread pool. Thread-spawn failures in worker pools are propagated as
    Python exceptions rather than panicking the interpreter.
