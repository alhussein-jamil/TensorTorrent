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

## Production service contract

Network serving fails closed: `--listen` requires `--artifact` unless the operator
explicitly supplies `--allow-empty` for health diagnostics. Integrity verification
is enabled when the artifact loads. Readiness remains false until at least one model
is loaded and all configured device workers are healthy.

The built-in HTTP server has no authentication, authorization, TLS termination, or
tenant isolation. Bind to a private interface and place it behind an authenticated,
rate-limited reverse proxy. Do not expose it directly to an untrusted network.

The service handles `SIGTERM` and `SIGINT`, stops accepting traffic, cooperatively
cancels active work, drains model generations, and closes workers. Cooperative
cancellation cannot interrupt an unresponsive vendor kernel. Configure the
orchestrator's termination grace period, then rely on its final `SIGKILL` boundary.

### Operational settings

All service limits are validated at startup. Invalid or non-finite values fail the
process rather than silently falling back.

| Environment variable | Default | Purpose |
| --- | ---: | --- |
| `SC_SERVE_MAX_QUEUE_DEPTH` | `64` | Global queued/in-flight request bound |
| `SC_SERVE_DEFAULT_TIMEOUT_S` | `30` | Request timeout when omitted |
| `SC_SERVE_MAX_REQUEST_TIMEOUT_S` | `3600` | Hard ceiling on caller-provided timeouts |
| `SC_SERVE_DEFAULT_CONCURRENCY` | `8` | Default per-model and worker-thread concurrency |
| `SC_SERVE_WORKER_THREADS` | `0` | Inference threads; `0` uses default concurrency |
| `SC_SERVE_CANCELLATION_GRACE_S` | `1` | Cooperative cancellation wait |
| `SC_SERVE_REQUEST_HISTORY_SIZE` | `1024` | Bounded in-memory request history |
| `SC_HTTP_MAX_BODY_BYTES` | `33554432` | Maximum JSON request body |
| `SC_HTTP_SOCKET_TIMEOUT_S` | `30` | Per-connection socket timeout |

Tune queue depth and concurrency from measured target latency, memory pressure, and
backend stability. Larger values are not automatically faster.

## CPU container

The repository `Dockerfile` builds a native wheel in a Rust builder and installs
only runtime dependencies in a non-root Python image. It contains no compiler or
development test dependencies. The default command expects a verified artifact at
`/models/model` and listens on port `8080`.

```bash
docker build --tag streamcompiler:0.1.0 .
docker run --rm --read-only \
	--mount type=bind,src="$PWD/model_artifact",dst=/models/model,readonly \
	--tmpfs /tmp:rw,noexec,nosuid,size=64m \
	--tmpfs /home/streamcompiler:rw,noexec,nosuid,size=64m \
	--publish 127.0.0.1:8080:8080 \
	streamcompiler:0.1.0
```

Build arguments select Rust and Python image versions and pin uv, Maturin, and CPU
PyTorch package versions. Mirror and digest-pin base images and package indexes in
regulated environments. CUDA, ROCm, and Intel XPU deployments require vendor runtime
images and target-specific PyTorch wheels; the supplied image intentionally claims
CPU support only.

Recommended orchestrator policy:

- mount artifacts read-only and never mutate a published bundle;
- run as the image's non-root user with a read-only root filesystem;
- provide writable `tmpfs` mounts only where the selected backend requires them;
- set CPU, memory, process, file-descriptor, and shared-memory limits explicitly;
- use `/health` for liveness and `/ready` for traffic readiness;
- terminate on repeated readiness failures rather than routing around corruption;
- retain target validation JSON, image digest, artifact manifest, and config together.

`/health` reports process lifecycle and is the liveness probe. `/ready` is a
side-effect-free readiness probe that requires a loaded model and healthy device
workers; use it to remove an instance from routing without forcing a restart.

## Artifact integrity and publication

`CompiledModule.save()` writes into a sibling staging directory, generates
`artifact-integrity.json` with SHA-256 digests, and atomically publishes the
completed bundle. A sibling publication lock serializes concurrent writers across
processes. Loading rejects a missing manifest, checksum mismatches, unexpected files,
path escapes, and symlinks by default. Unsigned legacy artifacts require the explicit
`load_compiled(..., verify_integrity=False)` opt-out and must be treated as trusted
code; resave them before production use. Never modify files inside a published artifact.

## Host architecture support

Linux x86-64 with Python 3.10, 3.11, or 3.12 is the release-tested baseline. The CI matrix
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

Planner contention coefficients are conservative analytic priors. Target profiling
currently replaces the compute multiplier; transfer and storage contention remain
priors and must be validated under representative concurrent load.

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
