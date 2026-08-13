# Running models under memory pressure

Trade residency for data movement when the model does not fit the preferred device tier. Budgets and scheduled movement are explicit — no implicit backend `.to(device)` policy.

## Start with explicit budgets

```python
import tensortorrent as tt

config = tt.CompileConfig(
    vram_budget_bytes=6 * (1 << 30),
    ram_budget_bytes=24 * (1 << 30),
    allow_nvme_streaming=True,
)
```

Unset fields resolve from host/cgroup state. See [Resource budgets](../product/resource_budgets.md).

## Parameter streaming

If params cannot stay resident on device but fit a slower tier, specialization emits loads/transfers so regions get them when needed.

When they **do** fit (or partially fit) the hoist budget, inference may keep device-resident copies across forwards and drop matching H2D Transfers from the steady-state schedule. Hoist sizing shares authority with fit policy (`accelerator_hoist_budget_bytes`, clamped to live free VRAM). A residency OOM demotes hoist for that schedule generation and rebuilds transfer/evict — it does not permanently flip hoist off.

Controls that matter:

- `allow_nvme_streaming`
- `ram_budget_bytes`
- `vram_budget_bytes`
- `prefetch_distance`
- `adaptive_prefetch`
- `storage_io_workers`
- `storage_queue_depth`

Prefetch is part of schedule selection. Planner gives an analytical preference; DES evaluates a bounded set of variants (including `prefetch=0` for streaming). Deeper prefetch can hide I/O, but burns more staging memory.

## Activation spill

`activation_budget_bytes` caps host-side live activations. When spill is required and the plan supports it, explicit spill/reload ops get emitted.

```python
config = tt.CompileConfig(
    activation_budget_bytes=8 * (1 << 30),
    spill_dir="/mnt/nvme/tensortorrent-spill",
)
```

Don't put real spill on `tmpfs` — rejected by default because it does not move pressure out of RAM.

## Spill directory

Resolution order is:

1. `CompileConfig.spill_dir`
2. `TT_SPILL_DIR`
3. `<cache_dir>/spill`
4. system temporary directory

For production, use an explicitly provisioned persistent local path with adequate free space.

TensorTorrent creates request/session spill directories and cleans them on normal completion, cancellation, and exceptions. Startup cleanup handles orphaned sessions left by dead processes.

## Pinned versus pageable staging

Pinned host staging is useful for accelerator transfers but consumes a constrained host resource. The DES finalist stage prefers the intended pinned path when feasible. If simulation rejects that path for host/pinned pressure and host-staged fallback is allowed, TensorTorrent evaluates a pageable recovery schedule.

Resident beyond-VRAM plans that Transfer/Evict per region pin the full parameter set only when it fits the discovered `pinned_host` allocatable pool; otherwise H2D stays pageable. CUDA copy streams still overlap when the source *is* pinned.

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
