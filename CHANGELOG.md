# Changelog

## 0.1.0

Initial release.

- PyTorch export → partition → AOT regions → immutable `ExecutableArtifact`
- Rust dispatcher owns schedule, residency, transfers, storage, and telemetry
- CPU NUMA discovery, CUDA/ROCm placement, virtual accelerators for CI
- Parameter streaming and activation spill under memory budgets
- In-process serving API and stdlib HTTP (`streamcompiler.serve`)
- CLI: `doctor`, `profile`, `validate-hardware`, `autotune`, `serve`
- Repository layout: `crates/`, `python/streamcompiler/`, `tools/`, `bench/`, `docs/{product,architecture,reference}/`
