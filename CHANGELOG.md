# Changelog

## Unreleased

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
