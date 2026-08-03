# Changelog

## Unreleased

- Repository: add project branding, a task-oriented README, documented SemVer
  releases, and CI/pre-commit checks that reject version or release-tag drift.
- Production hardening: capability-gated Intel XPU support, isolated third-party
  backend entry points, explicit host/device and all-NUMA storage topology fallbacks,
  strict compile and serving configuration validation, backend-aware fingerprints,
  atomic checksummed artifacts with cross-process publication locking, request-scoped
  serving cancellation, safe quantized-state loading, hardened model-pack validation,
  live-executor generation leases during replanning, and bounded CUDA/ROCm/XPU profiling.
- Serving model replacement: non-blocking generation retire/close on final
  `release_slot`, Condition-backed unload drain, warm only marks the current
  generation, and HTTP env knobs reject invalid limits.
- Planner: keep `ACCELERATOR` (mock/plugin) placeable under `allow_gpu=False`;
  `allow_gpu` still gates real DISCRETE/INTEGRATED GPUs only.
- Config: `allow_gpu=False` coerces `allow_integrated_gpu=False` (CPU-only switch).
- Storage: refuse pack/quantized load/write through symlink paths.
- Runtime: `CompiledModule.train()`/`eval()` tolerate torch.export children that
  raise ``NotImplementedError`` instead of aborting construction.
- Serve: expose cancel/queue-reject Prometheus counters; HTTP `/v1/cancel`;
  reject incomplete request bodies; refuse double `HttpServer.start()`;
  include request counters on `/health`.
- Opt-in training UX: `CompileConfig(allow_training=True)` enables normal
  `.train()` / `.eval()` — autograd through the resident ExecutableSchedule
  while training (same framework as inference), inference schedule after
  `.eval()` with updated weights. Multi-region partitions are kept for
  training compiles; mock hetero Transfers keep live tensors under train.
  Default compile stays inference-only (`.train()` raises). Rejects training
  with NVMe parameter streaming, `activation_budget_bytes`, or
  `process_workers>0`.   Optional `sc.fit(...)` wraps a simple train loop on
  that schedule path. Train-mode Transfers use `GradDeviceMove` for real
  torch devices and keep live tensors on mock accelerators.
  `ExecutionContext.enable_grad` carries the train flag into native region
  callbacks; train runs skip online profile feedback; `sc.fit` rejects empty
  batches, non-scalar losses, and closed modules.
- Prune: drop unused schedule helpers (`_copy_tier`, `_ensure_pinned`,
  `_state_env_names`), unused `CompiledModule.matches_current_machine`, and
  redundant train tests; collapse `forward` onto one schedule path.

## 0.1.0

Initial release.

- PyTorch export → partition → AOT regions → immutable `ExecutableArtifact`
- Rust dispatcher owns schedule, residency, transfers, storage, and telemetry
- CPU NUMA discovery, CUDA/ROCm placement, virtual accelerators for CI
- Parameter streaming and activation spill under memory budgets
- In-process serving API and stdlib HTTP (`streamcompiler.serve`)
- CLI: `doctor`, `profile`, `validate-hardware`, `autotune`, `serve`
- Repository layout: `crates/`, `python/streamcompiler/`, `tools/`, `bench/`, `docs/{product,architecture,reference}/`
