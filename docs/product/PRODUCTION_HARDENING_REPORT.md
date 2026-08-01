# StreamCompiler Production Hardening Report

## Scope

This pass hardened the uploaded repository for heterogeneous, single-machine production deployment while preserving the existing Python/Rust architecture and public API. Changes were concentrated in code paths that can be reasoned about and verified on a CPU-only sandbox. No CUDA, ROCm, Intel XPU, or mixed-vendor result is presented as hardware-validated.

## Implemented changes

### Heterogeneous backend coverage

- Added a capability-gated Intel XPU backend using `torch.xpu`.
- Separated CUDA and ROCm runtime detection so an AMD HIP build cannot be claimed by both backends.
- Upgraded ROCm discovery, dtype probing, peer-link discovery, region execution, and benchmarking.
- Added measured profiling support for CUDA, ROCm, and Intel XPU through the common profiler interface.
- Bounded accelerator and CPU transfer probes to avoid allocating model-sized temporary buffers.
- Added isolated third-party backend discovery through the `streamcompiler.backends` entry-point group.
- Included backend/plugin metadata in hardware fingerprints and surfaced plugin failures in hardware validation.

### Hardware topology and planning

- Added conservative host-to-accelerator links when optional backends do not provide measured links.
- Connected storage to every discovered NUMA RAM node instead of only the first node.
- Generalized planner and validation logic to cover accelerator-class resources and mixed-vendor sets.
- Preserved measured links and marked inferred links as unmeasured conservative fallbacks.

### Configuration and API safety

- Added strict compile-configuration type, range, numerical-mode, profile-level, budget, and weight validation.
- Made JSON configuration loading reject ambiguous strings, booleans-as-integers, and malformed scalar types.
- Prevented device-selection helpers from mutating caller-owned configuration objects.
- Hardened serving configuration and timeout validation against invalid and non-finite values.

### Artifact publication and integrity

- Added SHA-256 integrity manifests for compiled bundles.
- Added staged atomic directory publication with rollback of the prior generation.
- Added cross-process publication locking for concurrent writers.
- Added durable file/directory `fsync` boundaries.
- Rejected checksum mismatches, unexpected files, path escapes, and symlinks during verification.
- Preserved legacy artifact readability while enabling integrity verification for newly saved artifacts.

### Storage hardening

- Hardened model-pack parsing against oversized manifests, duplicate tensor IDs, overlapping blocks, malformed offsets, invalid alignment, truncation, and inconsistent tensor metadata.
- Made lazy two-pass pack loaders prove that their metadata is stable across passes.
- Gave concurrent pack writers unique temporary paths and durable publication.
- Preserved bounded header/manifest reads and incremental payload writes.
- Replaced unrestricted quantized-state loading with `torch.load(..., weights_only=True)`.
- Rejected non-finite quantized inputs and malformed quantized metadata.
- Preserved logical float dtype during dequantization and made quantized writes atomic and durable.

### Serving and lifecycle correctness

- Added request-scoped native cancellation tokens so one timeout does not cancel unrelated forwards.
- Fixed model-generation accounting during concurrent model replacement.
- Added duplicate active-request rejection and explicit cancellation by request ID.
- Hardened HTTP body limits, socket timeouts, server thread shutdown behavior, dtype/shape validation, and error responses.
- Fixed nested numeric JSON matrices so they are treated as one tensor; multiple positional inputs now require explicit tensor descriptors.

### Concurrent replanning

- Added executor-generation leases around live profile-guided replanning.
- New forwards atomically use the replacement executor.
- In-flight forwards finish on their original executor.
- Retired executors and parameter stores close only after their final lease is released.
- Global cancellation reaches both current and draining generations.

## Verification performed

The following checks completed in the sandbox:

```text
python -m compileall -q python tests tools bench examples
changed-module import smoke: 16 modules imported
focused production suite: 57 passed, 4 deselected
production-hardening regression file: 23 passed
maximum Python line length over 160: 0
```

The focused suite covers configuration validation, artifact integrity/publication, backend routing and plugin isolation, resource topology, fingerprints, pack format, quantized storage, serving, storage budgets, and executor-generation leases.

A broader focused run reached 52 passing tests before two tests correctly failed because the required native extension was not built in this sandbox.

## Verification not possible in this sandbox

The environment did not contain Cargo, Rustc, Maturin, Ruff, MyPy, or Hypothesis and had no accelerator hardware. Therefore these claims are intentionally not made here:

- successful Rust workspace compilation after this archive is unpacked;
- native wheel build/install;
- real CUDA, ROCm, Intel XPU, P2P, collective, VRAM-pressure, or CPU/GPU-overlap validation;
- real third-party backend execution;
- full property-test execution.

No Rust source was changed during this pass. The existing native core must still pass its normal CI/toolchain gates.

## Required target-machine release gate

Run on every deployment class:

```bash
uv sync --extra dev
uv run maturin develop --release
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
uv run ruff check python tests
uv run ruff format --check python tests
uv run mypy python
uv run pytest -q
uv run streamcompiler doctor --full --json artifacts/doctor.json
uv run streamcompiler profile --all-resources --output artifacts/profile
uv run streamcompiler validate-hardware --stress --output artifacts/validation.json
uv run maturin build --release -o dist
```

Require measured basic execution, numerical equivalence, transfers, memory-pressure behavior, cleanup, and sustained-concurrency evidence for every enabled backend. Treat discovery-only, inferred links, host-staged paths, plugins, and unsupported features conservatively.

## Remaining hardware-dependent limitations

- Real CUDA, ROCm, and Intel XPU behavior remains dependent on the target PyTorch build, drivers, firmware, topology, and collective libraries.
- The vendor-neutral Rust C ABI remains an experimental extension surface; Python entry-point backends are the supported extensibility mechanism in this revision.
- Cooperative cancellation cannot forcibly terminate an unresponsive vendor kernel or arbitrary Python code; target stability tests remain mandatory.
- Multi-node execution and compiled heterogeneous training are outside the current single-machine inference scope.

## Result

The repository is materially closer to production deployment across CPU NUMA systems and capability-gated NVIDIA, AMD, Intel, and third-party accelerator backends. The remaining uncertainty is hardware validation rather than hidden success claims in the codebase.
