# Changelog

## Unreleased

## 0.3.3

- Load `exported.pt2` onto CPU by default (`load_exported_program` / `load_compiled`) so archive CUDA metadata cannot OOM mid-range cards while the eager module still resides on GPU.
- Public `release_device_residency` on `CompiledModule` / executors: drop hoisted accelerator weights and rebuild Transfer/Evict; residency OOM recovery reuses the same path.

## 0.3.2

- Production hardening: `CompiledModule` owns `CapacityLedger` leases (no serve-side / ContextVar dual ownership); cancel generations; serve timeouts cancel per-request tokens only; hoist OOM demotes for the current schedule generation without permanently disabling hoist; shared live-VRAM hoist clamp; fail-closed zero device/disk budgets.
- Runtime perf: DirectPlan keeps ambient intra-op threads; skip redundant `.to` / on-device input Transfer work; skip host param republish when device-hoisted.
- Serve cancel-token schedule fallback reuses single-region DirectPlan device parameter copies (same cache seed as DataflowDirectPlan).
- Host capacity: live-available RAM budgets no longer double-count resident model bytes; explicit `ram_budget_bytes` still reserves base state.
- Bakeoff: streaming timing that falls back to planner prediction is labeled `measured=False` with explicit provenance in notes/metadata.
- CI: build-only `Dockerfile.cuda` gate on PRs/main (no GPU runtime); longer UV HTTP timeout for large CUDA torch wheels.
- Docs: runtime / budgets / serve / training / FAQ aligned with the above.
- CI: remove temporary helper scripts that broke `ruff format` on `main`.

## 0.3.1

- Planner/runtime: hoist resident parameters only when state fits
  `ACCELERATOR_REGION_STATE_FRACTION` (0.70) of the same effective VRAM as region
  budgets (`min(allocatable, vram_budget_bytes)`), so near-VRAM fits stream via
  Transfer/Evict instead of full residency OOMing on workspace (fixes
  non-monotonic 0.75× crossover).
- Benchmarks: refuse `benchmarks.tooling.freeze` from a dirty worktree unless
  `--allow-dirty`; enrich environment provenance; Qwen CPU eager uses multiple
  timed samples; host abort peak factor 2.5× weights; GPU-eager fit probe labeled
  as feasibility (not timed); crossover records `execution_strategy`.
- Cleanup: public suite is the Makefile/`run_everything` entrypoint; drop alias
  JSON names from public writes; package/docs aligned for clean remasure.
- Public evidence remasured from clean commit `fb503e5` (`git_dirty=false`) under
  `benchmarks/evidence/v0.3.1/` (layout: report → figures → raw).

## 0.3.0

- Public capacity launch suite: `python -m benchmarks.public` (fit / DeepMLP /
  HF transformer / budget / crossover / hetero) with RAM-safe subprocess
  isolation and frozen evidence under `benchmarks/evidence/`.
- Beyond-VRAM correctness: skip fusion when parameters exceed accelerator
  region budget; strip export CPU device asserts; disable buffer reuse when
  the schedule uses Transfer/Evict/Load/Prefetch.
- Benchmark environment records `git_dirty` alongside commit SHA.
- Docs: MEASURED capacity tables for DeepMLP 1.5× VRAM and Qwen3-8B fixed-shape
  logits forward (seq=16); honest Accelerate / CPU / eager baselines.

## 0.2.9

- Docs: brand-matched pipeline/planner/runtime/memory figures; README and
  architecture trimmed so each topic has one home.

## 0.2.8

- Planner top-K finalists: same-subset alternatives reach DES; non-streaming ranks
  steady-state schedules so cold-start H2D cannot overturn GPU.
- Two-stage DES winner selection (min-raw + isclose tie-break); fail closed when
  all variants infeasible; deterministic beam select with subset-pool retry.
- Beam parallelism gated on problem config / workers; finalists expose
  `analytic_rank` / `finalist_rank`.
- Native-extension gating for planner tests; build/dependency script cleanup.

## 0.2.7

- Native placement planner (`tt-planner`): Rayon-parallel subset beam search with
  GIL released; analytic finalists ranked by batch DES before compile.
- Shared transfer cost model between planner and DES (`estimate_transfer`,
  contention / host-staged policy).
- Config defaults: `planner_parallel_subsets=True`, `planner_workers=0` (auto),
  `planner_des_candidates` for DES shortlist size.
- Python planner search reduced to a thin native wrapper; specialize compiles
  only the DES-selected winner.

## 0.2.6

- Shared capacity accounting: `CapacityLedger` leases host/device/disk bytes per
  in-flight forward; serve admit and `CompiledModule.forward` fail closed under
  oversubscription; concurrency clamps to what budgets allow.
- Direct path auto-disabled for streaming, activation spill, and training-capable
  compiles (schedule semantics preserved even if `TT_DIRECT_PATH=1`).
- Docs and validation copy reflect the supported single-host product surface.
- Prune: remove always-error C backend ABI stubs, dead config knobs
  (`activation_overflow_policy`, `reduced_precision`), orphan `tools/dev_check.sh`,
  and unused package re-export facades.

## 0.2.5

- Beyond-VRAM: Prefetch is pack I/O only in DES; NUMA/pageable staging when a
  region exceeds pinned_host; host-resident weights + coalesced Transfer/Evict
  when state exceeds VRAM; pin packs for DMA; omit zero `mock_transfer_delay_s`
  on real CUDA (was capacity-spinning). `benchmarks/micro/oversized_model.py` 1.5× VRAM:
  ~1.1 s/fwd (was ~76 s).

## 0.2.4

- Compile-path timing breakdown (`specialize_timing`) and `make bench-perf`.
- Faster specialize plumbing: CPU-first region measure, optional accelerator
  shards, concurrent-first fusion bake-off, incremental planner local search.
- Dataflow direct path eligibility includes XPU; prune unused Python/Rust APIs.

## 0.2.3

- Disable activation buffer reuse under streaming parameter stores (shared slot
  views could overwrite live activations and produce NaN logits).

## 0.2.2

- `torch>=2.4` floor check; CPython 3.10–3.13 wheels (aarch64 3.12–3.13).

## 0.2.1

- Include `LICENSE` in the sdist for PyPI; tag-triggered release workflow.

## 0.2.0

Renamed to **TensorTorrent** and hardened for production deployment on both
shared desktops and resource-limited servers.

### Rename

- Project, package, and crates renamed: `streamcompiler` → `tensortorrent`,
  `sc-*` → `tt-*`, plugin C ABI `sc_backend_*` → `tt_backend_*`, environment
  prefix `SC_*` → `TT_*`, CLI entry points `tensortorrent` /
  `tensortorrent-serve`, cache dir `~/.cache/tensortorrent`, import alias `tt`.
- MSRV aligned to Rust 1.85 across workspace, CI, and containers.

### Resource budgets (new)

- Single budget resolver (`tensortorrent.hardware.budget` + Rust
  `tt-backend-cpu::host_budget`): explicit config > cgroup v2/v1 limits minus
  current usage > live OS availability (`MemAvailable`,
  `torch.cuda.mem_get_info`, statvfs) > machine totals as last resort, with
  reserve floors always withheld and provenance attached to every number.
- Discovery now reports live, permitted capacity: CUDA/ROCm/XPU allocatable
  memory comes from free VRAM minus a display-aware headroom (768 MiB with a
  display attached, 256 MiB headless) instead of `total * 0.9`; CPU memory and
  worker counts respect cgroup limits and scheduler affinity. Containers see
  their cgroup ceiling, not the host's RAM.
- Early fit gate: compilation refuses impossible models before any expensive
  work, naming every budget and its provenance (`MemoryCapacityError`).
- `tensortorrent doctor` prints the resolved budget table with provenance;
  the hardware validation report gains a `budgets` section.
- `CompileConfig.polite()` preset for shared desktops; new config fields
  `host_memory_reserve_bytes`, `vram_headroom_bytes`, `spill_dir`,
  `max_total_spill_bytes`, `stall_timeout_s`.
- Integrated GPUs (Intel UHD/Iris, NVIDIA Jetson) are now classified as
  `INTEGRATED_GPU` and plannable under `allow_integrated_gpu`.

### Spill lifecycle safety (new)

- Spill refuses RAM-backed filesystems (tmpfs/ramfs) — on desktop Linux `/tmp`
  is usually RAM — with `TT_ALLOW_TMPFS_SPILL=1` as an explicit escape hatch;
  default spill location moved to `<cache_dir>/spill` (or `TT_SPILL_DIR` /
  `CompileConfig.spill_dir`).
- Free-space precheck (64 MiB margin) before every spill write produces an
  actionable `DiskSpace` error instead of mid-inference `ENOSPC`.
- Aggregate spill budget (`max_total_spill_bytes`, default 80% of free disk)
  on top of the per-file cap.
- Per-execution session directories (`tt-spill-<pid>-<exec>`) are removed on
  completion, cancellation, error, and context drop; a startup sweep removes
  sessions whose owning process died, so crashes can no longer leak spill
  files.
- Pack writes precheck disk space (`DiskSpaceError`); invalid stored pack
  paths are logged before repacking instead of silently falling through.

### Runtime robustness

- Thread-spawn failure during CPU backend construction now raises a Python
  exception instead of aborting the interpreter (`BoundedPool::try_new`).
- Worker panics are contained; a completion lost to a panic surfaces through
  the new progress-aware stall watchdog (`RuntimeError::Stalled`,
  `stall_timeout_s`, default 300 s) instead of hanging or pinning a core —
  the former infinite busy-wait acquire loops are gone.
- Hard per-resource capacity ceilings are enforced on the real allocation
  path (`AllocationTable` limits finally wired and tested), and CPU backend
  allocations check the resolved budget with provenance in the error message.

### Serving hardening

- Connection cap with immediate 503 + `Retry-After` on saturation
  (`TT_HTTP_MAX_CONNECTIONS`, default 128) and configurable listen backlog
  (`TT_HTTP_BACKLOG`, default 64) close the unbounded thread-per-connection
  exhaustion path; every response is `Connection: close`.
- Chunked `Transfer-Encoding` rejected (400); missing/zero `Content-Length`
  is 411; response size guard (`TT_HTTP_MAX_RESPONSE_BYTES`, 128 MiB default)
  refuses to serialize oversized outputs.
- Optional bearer auth (`TT_SERVE_AUTH_TOKEN`) on everything except `/health`
  and `/ready`, constant-time compare, never logged.
- Per-model latency histogram `tensortorrent_inference_latency_seconds`
  (14 buckets) and `tensortorrent_model_requests_total{model,outcome}` make
  p95/p99 alerting possible; `X-Request-ID` on responses; structured logging
  (`TT_LOG_LEVEL`, `TT_LOG_FORMAT=text|json`) with request-id correlation.

### Platform guards

- `process_workers > 0` on non-Linux is now a `ConfigurationError` (was a
  silent no-op); WSL2 is detected and warned about (fork + CUDA).
- New error types: `ConfigurationError`, `DiskSpaceError`, `PlatformError`.
- Unknown keys in persisted configs are logged instead of silently dropped;
  example-input flattening failures are logged instead of silently degrading
  region measurement.

### CI, packaging, and deployment

- Every GitHub Action pinned to a verified commit SHA; Rust toolchain pinned;
  Python 3.11 added to the matrix; coverage gate; cargo-audit + pip-audit job;
  wheels uploaded as artifacts; Dependabot for actions/cargo/pip/docker.
- Tag-triggered release workflow: manylinux wheels (cp310–cp312), GitHub
  Release, and PyPI publishing via OIDC trusted publishing (one-time setup
  documented); dormant self-hosted GPU test workflow.
- CPU container healthcheck moved to `/health` so a missing model volume
  degrades to not-ready instead of a restart crash-loop; base images
  digest-pinned; new `Dockerfile.cuda` GPU serving image; `deploy/` gains
  docker-compose and Kubernetes examples with explicit resource limits,
  probes, and security contexts.

### Testing and correctness

- New guardrail suites: budget resolution against faked cgroup trees, spill
  safety (tmpfs refusal, orphan sweep), native FFI limits, the early fit gate,
  platform guards, serving limits, and structured logging.
- Fixed a genuine race in the transfer-execution test: a Transfer appended to
  an already-built schedule was unordered against the generated
  `release::<tensor>`, so the tensor could be freed first. The test now
  expresses the edge in the DAG, removing a ~1-in-3 flake.
- `TT_CACHE_DIR` relocates the artifact/pack cache (needed for read-only
  container roots); the unit suite uses it to stop tests sharing one on-disk
  cache and polluting `~/.cache/tensortorrent`.
- Timing-sensitive assertions widened for low-core hosts.

### Documentation

- New `docs/product/resource_budgets.md` (budget model, spill lifecycle,
  stall watchdog, container behaviour, worked examples); deployment runbook
  and capacity-planning guidance; refreshed FAQ, anti-patterns, architecture,
  and README.

## 0.1.0

Initial public release.

### Compiler and runtime

- PyTorch export, graph partitioning, AOT regions, and immutable
  `ExecutableArtifact` bundles.
- Rust-owned scheduling, residency, transfers, storage, cancellation, and
  telemetry.
- Parameter streaming, activation spill, concurrent execution, and
  profile-guided replanning with safe executor-generation handoff.
- Inference by default, plus opt-in resident-schedule training and `tt.fit`.

### Hardware and planning

- NUMA-aware CPU discovery and placement for NVIDIA CUDA, AMD ROCm, Intel XPU,
  and isolated third-party backend plugins.
- Mixed-vendor resource graphs, conservative topology fallbacks, virtual
  accelerators for deterministic tests, and backend-aware fingerprints.
- Capability-gated profiling, hardware validation, and host-staged collective
  fallbacks.

### Storage and artifacts

- Atomic, checksummed artifact publication with cross-process locking and
  strict integrity verification.
- Hardened model-pack and quantized-state validation, bounded streaming I/O,
  and symlink-safe file handling.

### Serving and operations

- Concurrent in-process serving and a standard-library HTTP service with
  bounded queues, request-scoped cancellation, safe model replacement, and
  health/readiness endpoints.
- Prometheus request, cancellation, rejection, timeout, and runtime telemetry.
- Strict compile and serving configuration validation, target-hardware
  validation commands, and a non-root CPU production container.

### Tooling

- CLI commands for `doctor`, `profile`, `validate-hardware`, `autotune`, and
  `serve`.
- Python 3.10–3.12 packaging, Rust/Python quality gates, pre-commit checks, and
  Linux x86-64/ARM64 CI coverage.
