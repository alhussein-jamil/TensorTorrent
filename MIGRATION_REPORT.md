# Hybrid Rust migration report

Date: 2026-07-29 (updated)

## Architecture

Python control plane (`torch.export` / FX / Inductor) → persistent
`NativeCompiledArtifact` → Rust data plane → Python only for real PyTorch
region compute.

Public path unchanged: `compile(...)` / `compiled(x)`.

## Stages

| Stage | Status |
|-------|--------|
| Native schedule model | Done |
| Persistent artifact | Done |
| Region-only Python callbacks (resident CPU) | Done |
| Opaque handles + Rust residency metadata | Done (region path) |
| Explicit stream / copy-engine / link ids | Done |
| Rust simulator default | Done (parity with Python oracle; `STREAMCOMPILER_PYTHON_SIM=1` forces oracle) |
| Native pack Load/Prefetch | Done (`NativePackReader` when extension loaded) |
| Native profiler + `apply_profile_feedback` | Done (`NativeProfileDatabase` in `ProfileFeedback`) |
| Async virtual backend | Done (Rust + `NativeVirtualBackend`; **simulated**) |
| Dual-runtime removal | Dev-only: `STREAMCOMPILER_DEV_PYTHON_RUNTIME=1` (deprecated alias kept) |
| CopyStore | Value bag; Rust owns valid/lease/release on native-residency path |

## Measured (CPU-only, Linear 8→4)

| Metric | Value |
|--------|-------|
| Hot-path schedule convert | 0 |
| GIL callbacks / forward | 1 (Compute) |
| pytest | **328 passed** |
| cargo workspace | green |

## Simulated / unvalidated

Virtual accelerators = **simulated**. No CUDA/ROCm/multi-GPU validation on this VM.
