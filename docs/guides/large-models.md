# Running models under memory pressure

TensorTorrent can trade memory residency for data movement when a model does not fit the preferred device tier. The compiler does this through explicit budgets and scheduled movement; it does not rely on an implicit backend `.to(device)` policy.

## Start with explicit budgets

For a reproducible deployment, set the limits that matter to the workload:

```python
import tensortorrent as tt

config = tt.CompileConfig(
    vram_budget_bytes=6 * (1 << 30),
    ram_budget_bytes=24 * (1 << 30),
    allow_nvme_streaming=True,
)
```

When a field is left unset, TensorTorrent resolves an effective limit from the host/cgroup state. See [Resource budgets](../product/resource_budgets.md).

## Parameter streaming

If parameters cannot remain resident in device memory but fit a slower tier, specialization can emit loads/transfers so regions receive parameters when needed.

The relevant controls are:

- `allow_nvme_streaming`
- `ram_budget_bytes`
- `vram_budget_bytes`
- `prefetch_distance`
- `adaptive_prefetch`
- `storage_io_workers`
- `storage_queue_depth`

Prefetch is part of schedule selection. The planner provides an analytical preference, then DES evaluates a bounded set of variants, including `prefetch=0` for streaming schedules. A deeper prefetch can hide I/O/transfer latency, but it consumes more resident staging memory.

## Activation spill

`activation_budget_bytes` limits the host-side live activation budget used by the planning/runtime path. When spill is required and supported by the plan, TensorTorrent emits explicit spill/reload operations.

```python
config = tt.CompileConfig(
    activation_budget_bytes=8 * (1 << 30),
    spill_dir="/mnt/nvme/tensortorrent-spill",
)
```

Do not use RAM-backed filesystems such as `tmpfs` for real spill. TensorTorrent rejects them by default because they do not move pressure out of RAM.

## Spill directory

Resolution order is:

1. `CompileConfig.spill_dir`
2. `TT_SPILL_DIR`
3. `<cache_dir>/spill`
4. system temporary directory

For production, use an explicitly provisioned persistent local path with adequate free space.

TensorTorrent creates request/session spill directories and cleans them on normal completion, cancellation, and exceptions. Startup cleanup handles orphaned sessions left by dead processes.

## Pinned versus pageable staging

Pinned host staging is useful for accelerator transfers but consumes a constrained host resource. The DES finalist stage prefers the intended pinned path when feasible. If detailed simulation rejects that path for relevant memory pressure and host-staged fallback is allowed, TensorTorrent can evaluate a pageable recovery schedule.

## Linear sharding

`enable_linear_sharding=True` allows oversized `aten.linear` operations to be rewritten into exact output-feature shards when required by the planning path. `max_linear_shards` bounds the rewrite.

This is a capacity mechanism; it is not automatically a performance win.

## Fit failures are deliberate

TensorTorrent is designed to fail before execution when the selected model cannot fit the resolved host/device/disk capacity even with enabled streaming/spill policies.

Treat `MemoryCapacityError` as a configuration/capacity result, not as a cue to disable checks.

## Practical workflow

1. Run `tensortorrent doctor` and inspect effective budgets.
2. Set explicit deployment budgets if reproducibility matters.
3. Put spill on real disk/NVMe, not tmpfs.
4. Compile with profiling on the deployment class of machine.
5. Inspect `compiled.explain()` for placement and streaming decisions.
6. Run target hardware validation.
7. Benchmark steady-state behavior with the same shape/concurrency expected in service.

## Performance expectation

Streaming solves a capacity problem first. If every forward must move many gigabytes across PCIe, a CPU-resident execution can be faster when the model fits RAM. The published oversized-model benchmark demonstrates exactly this trade-off. See [Benchmarks](../product/benchmarks.md).
