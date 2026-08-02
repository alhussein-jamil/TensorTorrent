# Deployment

Specialize and validate on the **target** machine — not only the laptop that
built the portable artifact.

```bash
streamcompiler doctor --full --json artifacts/doctor.json
streamcompiler profile --all-resources --output artifacts/profile
streamcompiler benchmark-topology --output artifacts/topology.json
streamcompiler validate-hardware --stress --output artifacts/validation_report.json
streamcompiler autotune model_artifact/ --profile
streamcompiler serve --health
make hardware-test
```

## Status meanings

| Status | Meaning |
| --- | --- |
| `hardware_detected` | Present in the resource graph |
| `backend_available` | Runtime libraries usable |
| `backend_compiled` | Capability / dtype enumerated |
| `basic_execution_validated` | Smoke path succeeded |
| `concurrent_execution_validated` | Measured overlapping execution |
| `numerical_correctness_validated` | Matches eager PyTorch |
| `performance_characterized` | Latency / bandwidth sample stored |
| `unsupported_capability` | Capability absent or unusable |
| `fallback_selected` | e.g. host-staged collectives |
| `failed` / `skipped` | Hard fail / not applicable |

GPU absence on a development host is `unsupported` / `skipped`. Validate GPUs on
the target machine.

Respecialize when hardware, drivers, PyTorch/backends, or resource limits change.

## Artifact integrity and publication

`CompiledModule.save()` writes into a sibling staging directory, generates
`artifact-integrity.json` with SHA-256 digests, and atomically publishes the
completed bundle. A sibling publication lock serializes concurrent writers across
processes. Loading rejects a missing manifest, checksum mismatches, unexpected files,
path escapes, and symlinks by default. Unsigned legacy artifacts require the explicit
`load_compiled(..., verify_integrity=False)` opt-out and must be treated as trusted
code; resave them before production use. Never modify files inside a published artifact.

## Host architecture support

Linux x86-64 with Python 3.10 or 3.12 is the release-tested baseline. The CI matrix
also builds, installs, and tests the native extension and Python package on Linux ARM64
with Python 3.12. ARM64 accelerator and machine-specific NUMA paths still require the
complete gate on the deployment target; CI only covers its generic CPU environment.
Other operating systems are unsupported. Artifact fingerprints include the host
architecture, so specializations are never portable between x86-64 and ARM64.

## Backend plugins

Deployment fingerprints include installed `streamcompiler.backends` entry-point
metadata. Changing a plugin package or version invalidates specialization caches.
Run `streamcompiler doctor --full` after any driver, PyTorch, plugin, firmware, or
hardware change. Plugin errors appear in the validation report rather than being
silently ignored.

## Serving cancellation

Serving requests use independent native cancellation tokens. Timing out one request
does not cancel other in-flight calls on the same compiled model. Model generations
are reference-counted exactly so replacing a model cannot decrement the new
generation when an old request completes. Idle prior generations close immediately;
busy ones close on the final `release_slot`.

HTTP: `POST /v1/cancel` with `{"request_id": "..."}` requests cooperative cancel
(`200` if active, `404` if unknown). Prometheus exposes
`streamcompiler_requests_cancelled_total` and `streamcompiler_queue_rejects_total`.
`GET /health` also returns request success/fail/cancel/reject/timeout counters.

## Target-hardware release gate

Before declaring a machine production-ready, run the validation commands above and
retain their JSON outputs with the deployment. Require at least one measured basic
execution and numerical-correctness result for every enabled backend. CUDA, ROCm,
and Intel XPU are independently capability-gated; a PyTorch ROCm build must never be
classified as CUDA. Treat unmeasured links, host staging, and plugin backends as
conservative fallbacks until profiled on the target.

Recommended matrix:

| Target | Required evidence |
| --- | --- |
| CPU/NUMA | affinity, all NUMA memory paths, region timings, sustained concurrency |
| NVIDIA CUDA | device execution, H2D/D2H, P2P matrix, streams/events, VRAM pressure |
| AMD ROCm | HIP execution, H2D/D2H, peer matrix, RCCL or host-staged fallback |
| Intel XPU | XPU execution, memory-capacity query, transfer path, oneCCL or fallback |
| Third-party plugin | isolated discovery, fingerprinting, execution, transfer and failure tests |

Do not infer support from discovery alone. Preserve `unsupported`, `skipped`, and
`failed` states in deployment reports rather than converting them into success.
