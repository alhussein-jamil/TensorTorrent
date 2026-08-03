# Product scope

TensorTorrent is a **single-machine multi-CPU / multi-GPU PyTorch runtime**
(inference-first; opt-in schedule training).

## In scope

- PyTorch inference (`torch.export` / FX control plane)
- One host: many CPU cores, NUMA domains, one or many GPUs
- Models larger than device or host RAM (parameter streaming, activation spill)
- Concurrent inference requests with shared capacity accounting
- Ahead-of-time compiled regions + immutable `ExecutableArtifact`
- Rust data plane owns scheduling, residency, transfers, storage, telemetry
- Resource budget resolver: host memory, VRAM, CPU count, and disk budgets are
  resolved from cgroup v2/v1 limits, live OS availability, or explicit config —
  in that precedence order. Every resolved value carries provenance shown by
  `tensortorrent doctor`. Containers see their cgroup limits, not host totals.
- Spill safety: activation spill refuses RAM-backed tmpfs/ramfs directories;
  free-space precheck before every write; per-session cleanup with orphan sweep.
- Stall watchdog: progress-aware waits replace the former infinite busy-wait
  loops; configurable timeout raises a diagnosable `RuntimeError`.
- Early fit gate: compilation refuses up front with `MemoryCapacityError` when
  parameters cannot fit the resolved host + device + disk budgets.
- Opt-in training (`CompileConfig(allow_training=True)`): `.train()` / `.eval()`
  like a normal module — autograd through the resident ExecutableSchedule, then
  the inference schedule again after `.eval()` (default compile stays
  inference-only). Multi-region partitions are kept for train and eval.
  Optional `tt.fit(...)` wraps a simple optimizer loop on that path.

## Out of scope

- Training under NVMe parameter streaming (needs pack writeback + region-local
  backward/recompute so not all weights stay resident through `backward`)
- Multi-node distributed training clusters
- Arbitrary dynamic Python in the serving hot path

## Ownership

| Plane | Owns |
| --- | --- |
| Python | export, normalize, partition, AOT region compile, PyTrees, public API, diagnostics, resource budget resolution |
| Rust | artifact, topology, schedule, workers, memory, transfers, storage, streams/events, cancel, telemetry, request lifecycle, budget enforcement, stall watchdog |

After `load` / `warm`, the Rust dispatcher runs the schedule. Torch compute regions may still invoke a Python callback to execute the region body; scheduling, residency, and transfers stay in Rust.
