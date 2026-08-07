# Resource Budgets and Guardrails

Every byte limit, CPU count, and disk quota TensorTorrent uses flows through a
single resolver before any allocation happens. This document describes how
budgets are computed, where they come from, and how to tune them.

---

## Budget resolver — precedence chain

Two resolvers work in tandem: a Python resolver (`python/tensortorrent/hardware/budget.py`)
used during compilation and planning, and a Rust resolver
(`crates/tt-backend-cpu/src/host_budget.rs`) used by the CPU backend at
runtime. Both implement the same precedence chain.

### Host memory

| Priority | Source | When used |
| -------: | ------ | --------- |
| 1 | Explicit caller value (`CompileConfig.host_memory_reserve_bytes`, or `ram_budget_bytes`) | Always wins when set |
| 2 | cgroup v2 (`memory.max` or `memory.high` minus `memory.current`) | Container or cgroup-limited host |
| 3 | cgroup v1 (`memory.limit_in_bytes` minus `memory.usage_in_bytes`) | Legacy container runtimes |
| 4 | OS available (`MemAvailable` from `/proc/meminfo` / `psutil.virtual_memory().available`) | Bare-metal or unlimited cgroup |
| 5 | OS total (`MemTotal` / `psutil.virtual_memory().total`) | Last resort; always noted in provenance |

The minimum of all applicable sources is used, so a container with a 4 GiB
cgroup limit on a 512 GiB host receives a 4 GiB ceiling, not 512 GiB.

After the raw figure is resolved, a **reserve floor** is withheld:

> **Reserve = 5 % of raw, clamped to [256 MiB, 2 GiB]**

The remaining figure is the **allowed budget**. It is never less than 128 MiB.
Override the reserve with `CompileConfig.host_memory_reserve_bytes` or
`TT_HOST_MEMORY_RESERVE_BYTES`.

### CPU worker count

| Priority | Source |
| -------: | ------ |
| 1 | Explicit caller value |
| 2 | `sched_getaffinity(0)` (respects `taskset` / cpuset masks) |
| 3 | cgroup v2 `cpu.max` quota (rounded up to whole CPUs) |
| 4 | cgroup v1 `cpu.cfs_quota_us / cpu.cfs_period_us` (rounded up) |
| 5 | `os.cpu_count()` / `available_parallelism()` |

The minimum of all applicable sources is used. A container with `--cpus 2` on
a 96-core host receives two logical CPUs, not 96.

### Device VRAM

| Priority | Source |
| -------: | ------ |
| 1 | `CompileConfig.vram_budget_bytes` |
| 2 | Live free VRAM (`torch.cuda.mem_get_info`) minus headroom |
| 3 | 90 % of total VRAM minus headroom (noted as `total_fallback`) |

**Headroom** is reserved for the display compositor and driver resident state:

| Display attached | Default headroom |
| :--- | ---: |
| Yes (desktop) | 768 MiB |
| No (headless server) | 256 MiB |

Override with `CompileConfig.vram_headroom_bytes` or
`TT_VRAM_HEADROOM_BYTES`.

### Disk (spill path)

| Priority | Source |
| -------: | ------ |
| 1 | `CompileConfig.max_total_spill_bytes` |
| 2 | 80 % of `shutil.disk_usage(spill_path).free` |

---

## Provenance

Every resolved budget carries a **provenance tag** (`BudgetSource`) with a kind
and a human-readable detail string:

| Kind | Meaning |
| ---- | ------- |
| `explicit` | Caller-supplied value; no probing done |
| `cgroup_v2` | cgroup v2 limit file |
| `cgroup_v1` | cgroup v1 limit file |
| `os_available` | Live OS available figure |
| `total_fallback` | OS total — live availability was unavailable |

Run `tensortorrent doctor` to see every resolved budget and its provenance:

```bash
tensortorrent doctor
tensortorrent doctor --full --json artifacts/doctor.json
```

---

## Environment variables and config fields

All environment variables are read at import / startup. Invalid values raise
`RuntimeError` immediately rather than silently using the default.

### Budget env vars

| Environment variable | Config field | Default | Purpose |
| -------------------- | ------------ | ------- | ------- |
| `TT_HOST_MEMORY_RESERVE_BYTES` | `host_memory_reserve_bytes` | 5 % of raw, [256 MiB, 2 GiB] | Bytes withheld from host budget for OS and other processes |
| `TT_VRAM_HEADROOM_BYTES` | `vram_headroom_bytes` | 768 MiB (display) / 256 MiB (headless) | Bytes withheld per GPU for compositor and driver |
| `TT_SPILL_DIR` | `spill_dir` | `<cache_dir>/spill` | Root directory for activation spill files |
| `TT_ALLOW_TMPFS_SPILL` | — | `0` | Set to `1` to allow spill on tmpfs/ramfs (not recommended) |

### CompileConfig fields (budget-related)

| Field | Default | Purpose |
| ----- | ------- | ------- |
| `ram_budget_bytes` | `None` (auto) | Host RAM cap; exceeding it triggers disk streaming |
| `vram_budget_bytes` | `None` (auto) | Per-device VRAM cap |
| `activation_budget_bytes` | `None` | Host peak for live activations; above this, spill Evict/Load ops are emitted |
| `host_memory_reserve_bytes` | `None` (auto) | Override the reserve floor |
| `vram_headroom_bytes` | `None` (auto) | Override the VRAM headroom |
| `spill_dir` | `None` → `TT_SPILL_DIR` → `<cache_dir>/spill` | Spill root |
| `max_total_spill_bytes` | `None` (auto → 80 % of free disk) | Aggregate spill cap; set via `CompileConfig`, no env var |
| `stall_timeout_s` | `300.0` | Seconds before a stalled execution raises `RuntimeError`; `0` disables |

---

## Polite mode for shared desktops

`CompileConfig.polite()` is a preset designed for machines running a desktop
session or other interactive workloads:

```python
config = tt.CompileConfig.polite()
compiled = tt.compile(model, example_inputs=(x,), config=config)
```

The preset applies:

| Setting | Value | Reason |
| ------- | ----- | ------ |
| `vram_headroom_bytes` | 1.5 GiB | Extra room for display compositor (512 MiB–1 GiB typical) |
| `stall_timeout_s` | 120.0 | Catches hangs quickly; leaves time for moderately slow regions |
| `max_concurrent_regions` | 1 | Avoids CPU/GPU contention that degrades interactive responsiveness |
| `prefetch_distance` | 1 | Minimal prefetch — only double-buffers; reduces pinned-memory footprint |

---

## Container behaviour and cgroup limits

When running inside Docker, Kubernetes, or any cgroup-limited environment,
TensorTorrent automatically reads the container's memory and CPU limits rather
than reporting host totals. No configuration is required.

**Example:** a container started with `--memory 4g` on a 512 GiB host:

- Raw memory budget: 4 GiB (from cgroup v2 `memory.max`)
- Reserve (5 % of 4 GiB = 205 MiB, clamped to 256 MiB): 256 MiB
- Allowed host memory: 3.75 GiB
- `tensortorrent doctor` provenance: `cgroup_v2`

The host's 512 GiB is never visible. This is intentional: sizing from machine
totals instead of resolved budgets leads to OOMKilled containers. See
[Anti-patterns](../reference/anti_patterns.md).

**CPU:** a container with `--cpus 2` on a 96-core host gets a 2-CPU budget,
so TensorTorrent spawns at most 2 concurrent CPU workers.

---

## Spill — lifecycle and safety

Activation spill writes intermediate tensors to disk when live activations
exceed `activation_budget_bytes`. The spill subsystem enforces several safety
properties.

### tmpfs / ramfs refusal

By default, TensorTorrent refuses to spill into directories on tmpfs or ramfs
mounts. On desktop Linux `/tmp` is usually tmpfs; data there occupies RAM and
does not survive reboots. If the chosen spill directory is on tmpfs,
`ConfigurationError` is raised at startup.

To use a persistent path instead:

```bash
# Option 1: environment variable
export TT_SPILL_DIR=/data/tt-spill

# Option 2: config field
config = tt.CompileConfig(spill_dir="/data/tt-spill")
```

To override the refusal (not recommended):

```bash
export TT_ALLOW_TMPFS_SPILL=1
```

**Spill directory precedence:**

1. `CompileConfig.spill_dir` (explicit config value)
2. `TT_SPILL_DIR` environment variable
3. `<cache_dir>/spill` (default cache dir is `~/.cache/tensortorrent/spill`)
4. `tempfile.mkdtemp()` (legacy fallback, used when no cache dir is available)

### Free-space precheck

Before every spill write, TensorTorrent checks that at least 64 MiB of free
space remains on the spill filesystem. When the check fails, `DiskSpaceError`
is raised with the path, bytes needed, and bytes available. This replaces the
previous behavior of receiving an opaque `ENOSPC` mid-inference.

### Aggregate cap

The total bytes written per execution session are bounded by
`CompileConfig.max_total_spill_bytes`. When unset, the default is 80 % of free
disk space at the time the session starts. There is no environment variable for
this setting; configure it via `CompileConfig`.

### Session directories and cleanup

Each execution creates a temporary session directory with the prefix
`tt_native_spill_` inside the spill root. The directory is removed on
completion, cancellation, or error — including when a Python exception is
raised during inference.

On startup, a one-time sweep removes session directories whose owning process
is no longer alive. This handles crash cleanup: if a previous run died without
removing its session directory, the next run cleans it up automatically.

---

## Stall watchdog

The Rust data plane replaces the former infinite busy-wait resource loops with
a progress-aware watchdog. Instead of pinning a CPU at 100 % when a completion
is lost, the scheduler sleeps in 200 µs increments and tracks a
**progress generation counter** that advances whenever any instruction
completes.

When no progress is observed for `stall_timeout_s` seconds, the execution
raises `RuntimeError` with the message:

```
stalled: no progress for N.Ns while waiting for WHAT. This usually means a
lost completion or a deadlocked resource; if this host legitimately has I/O
this slow, raise CompileConfig.stall_timeout_s
```

The watchdog triggers only when **nothing** in the execution makes progress —
legitimately slow disk I/O or large transfers do not trip it as long as
something else completes in the meantime.

Configure via `CompileConfig.stall_timeout_s` (default 300 s; `0` disables).


---

## Shared capacity accounting (concurrent requests)

Concurrent forwards on one `CompiledModule` share the parameter store. Each
in-flight request leases only its **incremental** working set (activations plus
any streaming window) through `CapacityLedger`. Resident parameter bytes are
reserved once as a base.

Admit fails closed when the next lease would exceed the resolved host, device,
or disk budget:

- Serve: `ModelManager.acquire` leases before the request starts and clamps
  `concurrency_limit` to what the budgets allow.
- Direct API: `CompiledModule.forward` leases around each call.

Inspect live usage via `compiled.capacity_ledger` and the `capacity_inflight`
field in `ModelManager.list_models()`.

---

## Early fit gate

Before expensive region capture and benchmark, TensorTorrent checks whether
model parameters can plausibly fit in the available memory. If they cannot,
`MemoryCapacityError` is raised immediately with all numbers and provenance:

```
MemoryCapacityError: Model parameters (12884901888 bytes) definitely cannot
fit in available memory: host_allowed=7516192768 (source=cgroup_v2)
device_allowed_sum=6979321856 total_allowed=14495514624.
Reduce the model size, raise memory budgets, or enable NVMe streaming.
```

This is a conservative early check. A pass does not guarantee the full plan
will succeed, but a failure guarantees it would not.

---

## Worked examples

### 16 GiB gaming PC with an 8 GiB discrete GPU (display attached)

```
Machine:
  RAM:  16 GiB physical, ~12 GiB MemAvailable (OS + browser consuming 4 GiB)
  GPU:  8 GiB GDDR6, display attached
  Disk: 500 GiB NVMe, ~200 GiB free

Budget resolver output:
  Host memory:
    Raw:      12 GiB  (source: os_available)
    Reserve:  614 MiB (5 % × 12 GiB)
    Allowed:  ~11.4 GiB

  VRAM:
    Live free: 6.5 GiB (GPU running desktop compositor)
    Headroom:  768 MiB (display active)
    Allowed:   ~5.7 GiB

  Disk (spill):
    Free:    200 GiB
    Allowed: 160 GiB (80 %)
```

For a model with 10 GiB parameters, NVMe streaming would be required
(`allow_nvme_streaming=True` and `ram_budget_bytes` set). For best
interactive behaviour, use `CompileConfig.polite()`.

### 4 GiB-limit container on a 512 GiB host

```
Container: docker run --memory 4g --cpus 2 ...
Machine:   512 GiB RAM, 96 CPU cores

Budget resolver output:
  Host memory:
    Raw:      4 GiB  (source: cgroup_v2, memory.max=4294967296)
    Reserve:  256 MiB (5 % × 4 GiB = 205 MiB, clamped to 256 MiB)
    Allowed:  3.75 GiB

  CPU workers:
    Count: 2  (source: cgroup_v2 cpu.max quota)

  Disk: depends on volume mount; measured at session start
```

The host's 512 GiB RAM and 96 cores are invisible to the container. Models
larger than 3.75 GiB parameters require NVMe streaming or a larger cgroup
limit.
