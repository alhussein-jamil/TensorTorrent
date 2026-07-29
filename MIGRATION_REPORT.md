# Hybrid Rust migration report

Date: 2026-07-29 (updated)

## Architecture

Python control plane + persistent `NativeCompiledArtifact` + Rust scheduler.
On resident CPU schedules: region-callback data plane + **native residency
session** (opaque handle ids; Rust metadata authoritative).

Public path unchanged: `compile(...)` / `compiled(x)`.

## Stages

| Stage | Status |
|-------|--------|
| Native schedule model | Done |
| Persistent artifact | Done |
| Region-only Python callbacks (resident CPU) | Done |
| Opaque handles + Rust residency metadata | **Done (region path)** |
| CopyStore still holds torch values | Yes — value bag; Rust owns valid/lease/release metadata on region path |
| Rust simulator default | Blocked — peak-memory parity vs Python oracle; keep `STREAMCOMPILER_NATIVE_SIM=1` |
| Native Transfer / streaming / spill | Open |
| Async virtual backend | Open |
| Dual-runtime removal | Open |

## Measured (CPU-only, Linear 8→4)

| Metric | Value |
|--------|-------|
| Hot-path schedule convert | 0 |
| GIL callbacks / forward | 1 (Compute) |
| `native_residency` | True |
| Dispatch overhead vs eager | ~260–380 µs |
| pytest | **320 passed** |

## Simulated / unvalidated

Virtual accelerators = simulated. No CUDA/ROCm/multi-GPU validation on this VM.
