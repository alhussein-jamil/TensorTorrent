# StreamCompiler Production Hardening Report

## Scope

This pass hardened the repository for heterogeneous, single-machine production deployment while preserving the Python/Rust architecture and public API. Validation ran on Linux with one NVIDIA CUDA GPU. CPU and single-GPU smoke results are reported; no ROCm, Intel XPU, multi-GPU, mixed-vendor, or sustained hardware-stress result is claimed.

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

- Centralized validated service limits and strict `SC_SERVE_*` environment overrides.
- Bounded request history and caller-provided timeout ceilings.
- Made network serving require a verified compiled artifact unless `--allow-empty` is explicit.
- Made readiness require a loaded model and healthy workers without mutating worker state.
- Added `SIGINT`/`SIGTERM` handling and bounded HTTP server-thread shutdown.
- Added request-scoped native cancellation tokens so one timeout does not cancel unrelated forwards.
- Fixed model-generation accounting during concurrent model replacement.
- Added duplicate active-request rejection and explicit cancellation by request ID.
- Hardened HTTP body limits, socket timeouts, server thread shutdown behavior, dtype/shape validation, and error responses.
- Fixed nested numeric JSON matrices so they are treated as one tensor; multiple positional inputs now require explicit tensor descriptors.

### Runtime policy and observability

- Named worker queue, restart, warm-up, poll, ping, shutdown, hardware-probe, and planner-contention defaults.
- Kept planner contention coefficients explicitly documented as analytic priors pending target measurements.
- Added native `prefetch_dropped` telemetry when the bounded streaming queue saturates.

### Production container

- Replaced the development image with a multi-stage native-wheel build.
- Removed compilers and development dependencies from the runtime stage.
- Pinned build-tool and CPU PyTorch versions through named build arguments.
- Runs as a dedicated non-root user and supports a read-only root filesystem.
- Requires a read-only model artifact mount by default and exposes a readiness health check.
- Added a CI image build, native import, non-root, and service-health smoke job.

### Concurrent replanning

- Added executor-generation leases around live profile-guided replanning.
- New forwards atomically use the replacement executor.
- In-flight forwards finish on their original executor.
- Retired executors and parameter stores close only after their final lease is released.
- Global cancellation reaches both current and draining generations.

## Verification performed

The complete repository gate passed on 2026-08-03:

```text
Ruff lint: passed
Ruff format: 170 files formatted
MyPy strict: 91 source files, no issues
Cargo fmt: passed
Cargo clippy: workspace/all targets/all features, warnings denied
Cargo tests: all workspace unit, property, and documentation tests passed
Python tests: 523 passed, 3 skipped, 25 hardware deselected
Doctor: CPU and one CUDA GPU detected; basic execution passed on both
Numerical validation: StreamCompiler vs eager max_abs_err=2.384e-07 on CUDA
Native import: passed
Release native rebuild: passed
Native streaming regressions: 23 passed
```

The Docker CLI was present, but no local Docker daemon or daemonless builder was
available, so the production image could not be built in this environment. The CI
workflow now performs the image build plus native-import, non-root, and service-health
smokes; that job must pass before publishing the image.

## Verification still required on deployment targets

- full container CI/build result and registry vulnerability/SBOM policy;
- sustained CPU/NUMA and accelerator load, memory pressure, cleanup, and restart tests;
- multi-GPU overlap, P2P, collectives, and unequal-device partitioning;
- real ROCm, Intel XPU, mixed-vendor, and third-party plugin execution;
- operator-specific reverse proxy, authentication, TLS, rate limiting, secrets, and observability;
- graceful termination under an intentionally stalled vendor kernel.

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
- Multi-node execution remains outside the current single-machine scope.
  Resident schedule training is supported; NVMe streaming training is not.

## Result

The repository passes its local production gate and now has fail-closed serving,
centralized operational policy, observable queue saturation, and a production-shaped
CPU image. Deployment approval remains conditional on the container CI job and the
documented target-hardware/operator release gate; repository checks alone cannot
prove every vendor runtime or production topology.
