# Changelog

## 0.1.0

Initial public release.

### Compiler and runtime

- PyTorch export, graph partitioning, AOT regions, and immutable
  `ExecutableArtifact` bundles.
- Rust-owned scheduling, residency, transfers, storage, cancellation, and
  telemetry.
- Parameter streaming, activation spill, concurrent execution, and
  profile-guided replanning with safe executor-generation handoff.
- Inference by default, plus opt-in resident-schedule training and `sc.fit`.

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
