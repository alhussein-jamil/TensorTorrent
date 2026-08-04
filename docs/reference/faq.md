# FAQ

**Why does `doctor` say CUDA is unsupported?**
No usable CUDA runtime. Status is `unsupported_capability`, not a pass. Validate
on the target machine with `tensortorrent validate-hardware`.

**Does the planner use every GPU?**
No. A device is included only when it improves the objective after transfer cost.
See `compiled.explain()`.

**Can I mix NVIDIA and AMD in one process?**
Not today. Mixed-vendor links may be host-staged; real execution needs separate
workers per backend.

**Do I need a GPU to compile?**
No. Portable artifacts are hardware-independent. Specialize per host.

**Tiny-model latency?**
At scale (tens to hundreds of ms of work) TensorTorrent matches eager and can
lead. Sub-millisecond forwards pay schedule dispatch on the default path; set
`TT_DIRECT_PATH=1` for resident single-region cases to skip that path. The main
product win is capacity under RAM / VRAM budgets and multi-device schedules.
See [Benchmarks](../product/benchmarks.md).

**Different batch size?**
No. Example shapes/dtypes are fixed. Mismatch raises `UnsupportedFeatureError`.

**Where do output tensors live?**
On the device holding the final scheduled copy. TensorTorrent does not add an
unscheduled copy back to CPU. Use `output.cpu()` when the caller requires host
residency; explicit output-placement policy is not yet a compile option.

**Can I compile several modules together?**
Yes. `compile_modules([...], example_inputs=...)` composes a series into one
exported graph and schedule. Use `ModuleGraph`, `ModuleNode`, `GraphInput`, and
`NodeOutput` for branches, joins, multiple inputs, structured arguments, and
tuple/list/dict output pytrees.

**Training?**
Default compile is inference-only (`.train()` raises). Pass
`CompileConfig(allow_training=True)` for a normal loop: `.train()` runs the
ExecutableSchedule with autograd (`backward` / `optimizer.step()`); `.eval()`
returns to the inference schedule with the updated weights. Parameters must stay
resident (no NVMe streaming or activation spill yet). Incompatible with
`process_workers`. Works with multi-region schedules and `use_torch_compile`.

**Execution timeline?**

```python
compiled(x)
compiled.visualize("run.html", measured=True)
```

Default `visualize` is analytic simulation of the same schedule (`simulated=True`).

**Cancel?**
`request_cancel()` flips per-forward tokens. The dispatcher stops launching new
work at wave boundaries, then raises `ExecutionCancelled`. In-flight Compute in
the current wave still finishes.

---

**Why does TensorTorrent refuse to spill to `/tmp`?**

On desktop Linux and most container runtimes, `/tmp` is a `tmpfs` mount — data
lives in RAM, not on disk. Spilling activations to tmpfs defeats the purpose of
spilling (freeing device memory) and can exhaust RAM instead. TensorTorrent
reads `/proc/mounts` and raises `ConfigurationError` if the chosen spill path
is on `tmpfs` or `ramfs`.

Set `TT_SPILL_DIR` or `CompileConfig.spill_dir` to a persistent path (e.g. an
NVMe-backed volume). To override the check (not recommended):

```bash
export TT_ALLOW_TMPFS_SPILL=1
```

See [Resource budgets — spill safety](../product/resource_budgets.md#spill--lifecycle-and-safety).

**Why is my container seeing less memory than the host has?**

This is intentional, not a bug. When a container cgroup limit is set
(`--memory 4g`), TensorTorrent reads that limit from cgroup v2 or v1 and uses
it as the raw budget ceiling instead of the host total. A 4 GiB container on a
512 GiB host correctly receives a ~3.75 GiB allowed budget.

This prevents OOMKilled containers caused by sizing from machine totals. Run
`tensortorrent doctor` inside the container to see the resolved budget and its
provenance (`source=cgroup_v2`).

**What does the "Stalled" error mean?**

```
RuntimeError: stalled: no progress for N.Ns while waiting for WHAT.
This usually means a lost completion or a deadlocked resource; if this host
legitimately has I/O this slow, raise CompileConfig.stall_timeout_s
```

The stall watchdog fires when **nothing** in the execution has made progress
for `stall_timeout_s` seconds (default 300 s). Likely causes:

- A Rust worker panic or a closed completion channel.
- A GPU kernel hang — check `nvidia-smi` and `dmesg`.
- Pathologically slow I/O (network-backed storage, thermal throttling).

If the host legitimately has slow I/O, raise `stall_timeout_s` in
`CompileConfig`. Setting it to `0` disables the watchdog entirely (not
recommended for production). The `CompileConfig.polite()` preset uses 120 s.


**Does it run on Windows or WSL2?**

The supported target is **Linux**. Windows is unsupported.

WSL2 is detected (`/proc/version` containing `microsoft`). When
`process_workers > 0` under WSL2, TensorTorrent emits a warning: `fork()` after
CUDA initialization is unstable and may cause hangs or corruption. Set
`process_workers=0` if you encounter issues. WSL2 without `process_workers` may
work for CPU-only development but is not a supported production target.

Setting `process_workers > 0` on any non-Linux platform raises
`ConfigurationError` immediately at `CompileConfig` construction.

**Why is my GPU budget smaller than its VRAM capacity?**

TensorTorrent reserves a **headroom** from the GPU budget for the display
compositor and driver resident state:

- **Display attached** (desktop): 768 MiB reserved
- **Headless** (server, container): 256 MiB reserved

Additionally, the budget is computed from **live free VRAM**
(`torch.cuda.mem_get_info`), not the total. If other processes are using the
GPU, their allocations reduce the available budget further.

To see the resolved VRAM budget and its source, run `tensortorrent doctor`.
Override the headroom with `CompileConfig.vram_headroom_bytes` or
`TT_VRAM_HEADROOM_BYTES`. The `CompileConfig.polite()` preset uses 1.5 GiB
headroom for shared desktops.
