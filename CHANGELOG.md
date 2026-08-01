# Changelog

## Unreleased

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
  reject incomplete request bodies; refuse double `HttpServer.start()`.
- Opt-in training UX: `CompileConfig(allow_training=True)` enables normal
  `.train()` / `.eval()` — autograd on the live `graph_module` while training,
  inference schedule after `.eval()` with updated weights. Default compile
  stays inference-only (`.train()` raises). Rejects training with NVMe
  parameter streaming or `process_workers>0`.

## 0.1.0

Initial release.

- PyTorch export → partition → AOT regions → immutable `ExecutableArtifact`
- Rust dispatcher owns schedule, residency, transfers, storage, and telemetry
- CPU NUMA discovery, CUDA/ROCm placement, virtual accelerators for CI
- Parameter streaming and activation spill under memory budgets
- In-process serving API and stdlib HTTP (`streamcompiler.serve`)
- CLI: `doctor`, `profile`, `validate-hardware`, `autotune`, `serve`
- Repository layout: `crates/`, `python/streamcompiler/`, `tools/`, `bench/`, `docs/{product,architecture,reference}/`
