# Resource budgets

TensorTorrent plans against **effective capacity**, not raw machine totals. The budget resolver is the common source of truth for host memory, CPU count, accelerator memory, and spill disk.

## Host memory

Resolution order:

1. explicit `CompileConfig.ram_budget_bytes`,
2. the most restrictive usable cgroup v2/v1 memory limit (Linux) and live OS availability (Linux `MemAvailable`, macOS sysctl/vm pages, elsewhere psutil).

The automatic host reserve is 5% of the resolved raw capacity, clamped between 256 MiB and 2 GiB. Override it with `host_memory_reserve_bytes` or `TT_HOST_MEMORY_RESERVE_BYTES`.

TensorTorrent reports the provenance of the resulting budget through `tensortorrent doctor`.

## CPU count

Automatic worker capacity is the minimum of the constraints the process can actually use, including:

- `sched_getaffinity`,
- cgroup v2 CPU quota,
- cgroup v1 CFS quota,
- `os.cpu_count()`.

This prevents a container limited to two CPUs from sizing runtime pools for a 96-core host.

## Device memory

When no explicit `vram_budget_bytes` is supplied, TensorTorrent prefers live free accelerator memory minus a safety headroom.

Default headroom:

- 768 MiB when a display is considered active,
- 256 MiB for headless use.

Override with `vram_headroom_bytes` or `TT_VRAM_HEADROOM_BYTES`.

The planner can also use a physical-capacity floor (`total - headroom`) to avoid overreacting to transient free-memory readings held by a caching allocator. Set `TT_DISABLE_VRAM_CAPACITY_FLOOR=1` only when you explicitly want to disable that behavior.

## Spill disk

For a spill path, automatic disk allowance is 80% of currently free space. An explicit `max_total_spill_bytes` can impose a smaller limit.

TensorTorrent checks free space before writes and uses session-scoped spill directories.

## Spill safety

A RAM-backed spill directory defeats the purpose of moving pressure out of memory. TensorTorrent refuses `tmpfs`/`ramfs` spill roots by default.

Use a persistent local path, for example:

```bash
export TT_SPILL_DIR=/mnt/nvme/tensortorrent-spill
```

or:

```python
config = tt.CompileConfig(spill_dir="/mnt/nvme/tensortorrent-spill")
```

`TT_ALLOW_TMPFS_SPILL=1` exists for controlled testing, not normal deployment.

## Containers

The resolver is cgroup-aware. A container with a 4 GiB memory limit on a 512 GiB host plans against the container's available budget, not 512 GiB.

Run diagnostics inside the same container/cgroup that will serve the model:

```bash
tensortorrent doctor
```

## Early fit gate

Specialization fails closed when the model cannot fit the resolved combination of host/device/storage budgets under the enabled policies. Do not disable capacity checks to force a plan through; change the budget, storage policy, or model.

## Shared capacity

Concurrent serving uses a module-owned `CapacityLedger` so simultaneous requests cannot each assume the entire host/device/disk budget is available independently.

- `CompiledModule` creates and owns the ledger; each forward acquires/releases a byte lease under a module lock.
- Serve (`ModelManager`) tracks request counts only and requires `module.capacity_ledger` — there is no parallel capacity ContextVar or serve-side lease ownership.
- Zero resolved device or disk budgets fail closed (no silent admit).
- Base parameter reservation and per-request incremental leases are distinct; empty incremental leases still take a 1-byte floor.
- When the host budget comes from live remaining memory (`os_available` / cgroup), resident model bytes are already reflected in that ceiling and are not deducted again. Explicit `ram_budget_bytes` still reserves resident state as a base allocation.

See [Runtime](../architecture/runtime.md).

## Practical presets

For a shared workstation:

```python
config = tt.CompileConfig.polite()
```

For a dedicated service, explicit budgets are preferable because they make capacity behavior reproducible across restarts.
