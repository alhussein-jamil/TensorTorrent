# Deployment

Specialize and validate on the **target** machine — not only the laptop that
built the portable artifact.

## Local release sequence

1. **Deterministic gate (every change):** `make check` / `uv run python tools/check.py`
   (includes `tools/native_gate.py` after native import proof).
2. **On the deployment host:** `make hardware-test` and
   `tensortorrent validate-hardware --stress --output artifacts/validation_report.json`.
3. **Pass criterion:** report `production_ready: true`. Discovery
   (`hardware_detected`) alone is never enough — each enabled accelerator
   backend needs measured `basic_execution_validated` plus
   `numerical_correctness_validated`.
4. **Optional p50 smoke:** `make bench-smoke` (writes `artifacts/bench_smoke.json`).
5. Retain JSON outputs with the deployment. Hosted GPU CI is not required;
   prefer local or self-hosted runners.

```bash
make check
tensortorrent doctor --full --json artifacts/doctor.json
tensortorrent profile --all-resources --output artifacts/profile
tensortorrent benchmark-topology --output artifacts/topology.json
tensortorrent validate-hardware --stress --output artifacts/validation_report.json
# overnight soak on prod machines:
# tensortorrent validate-hardware --overnight --output artifacts/validation_overnight.json
tensortorrent autotune model_artifact/ --profile
make hardware-test
make bench-smoke
```

## Status meanings

| Status | Meaning |
| --- | --- |
| `hardware_detected` | Present in the resource graph (informational — not a production pass) |
| `backend_available` | Runtime libraries usable |
| `backend_compiled` | Capability / dtype enumerated |
| `basic_execution_validated` | Smoke path succeeded |
| `concurrent_execution_validated` | Measured overlapping execution |
| `numerical_correctness_validated` | Matches eager PyTorch |
| `performance_characterized` | Latency / bandwidth sample stored |
| `unsupported_capability` | Capability absent or unusable |
| `fallback_selected` | e.g. host-staged collectives |
| `failed` / `skipped` | Hard fail / not applicable |
| `production_ready` | Summary field: measured execution + numerics for every enabled accelerator |

GPU absence on a development host is `unsupported` / `skipped`. Validate GPUs on
the target machine.

Respecialize when hardware, drivers, PyTorch/backends, or resource limits change.

## Production service contract

Network serving fails closed: `--listen` requires `--artifact` unless the operator
explicitly supplies `--allow-empty` for health diagnostics. Integrity verification
is enabled when the artifact loads. Readiness remains false until at least one model
is loaded and all configured device workers are healthy.

The built-in HTTP server supports bearer-token authentication
(`TT_SERVE_AUTH_TOKEN`). `/health` and `/ready` are exempt from auth and must
always be reachable by liveness/readiness probes. For full TLS termination,
tenant isolation, and rate limiting, place the service behind an authenticated
reverse proxy. Do not expose it directly to an untrusted network.

The service handles `SIGTERM` and `SIGINT`, stops accepting traffic, cooperatively
cancels active work, drains model generations, and closes workers. Cooperative
cancellation cannot interrupt an unresponsive vendor kernel. Configure the
orchestrator's termination grace period, then rely on its final `SIGKILL` boundary.

### HTTP protocol notes

- Chunked transfer-encoding (`Transfer-Encoding: chunked`) is rejected with
  HTTP 400. Callers must send `Content-Length`.
- Every response carries `Connection: close`; keep-alive is not used.
- Successful inference responses carry an `X-Request-ID` header.
- Connection cap saturation returns HTTP 503 with `Retry-After: 1`.

### Operational settings

All service limits are validated at startup. Invalid or non-finite values fail the
process rather than silently falling back.

| Environment variable | Default | Purpose |
| --- | ---: | --- |
| `TT_SERVE_MAX_QUEUE_DEPTH` | `64` | Global queued/in-flight request bound |
| `TT_SERVE_DEFAULT_TIMEOUT_S` | `30` | Request timeout when omitted |
| `TT_SERVE_MAX_REQUEST_TIMEOUT_S` | `3600` | Hard ceiling on caller-provided timeouts |
| `TT_SERVE_DEFAULT_CONCURRENCY` | `8` | Default per-model and worker-thread concurrency |
| `TT_SERVE_WORKER_THREADS` | `0` | Inference threads; `0` uses default concurrency |
| `TT_SERVE_CANCELLATION_GRACE_S` | `1` | Cooperative cancellation wait |
| `TT_SERVE_REQUEST_HISTORY_SIZE` | `1024` | Bounded in-memory request history |
| `TT_HTTP_MAX_BODY_BYTES` | `33554432` (32 MiB) | Maximum JSON request body |
| `TT_HTTP_SOCKET_TIMEOUT_S` | `30` | Per-connection socket timeout |
| `TT_HTTP_MAX_CONNECTIONS` | `128` | Simultaneous connection cap; excess → 503 |
| `TT_HTTP_BACKLOG` | `64` | OS listen backlog (connections waiting to be accepted) |
| `TT_HTTP_MAX_RESPONSE_BYTES` | `134217728` (128 MiB) | Response payload cap; exceeded → 500 |
| `TT_SERVE_AUTH_TOKEN` | _(unset)_ | Bearer token required on all non-health endpoints when set |
| `TT_LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `TT_LOG_FORMAT` | `text` | Log format: `text` or `json` |
| `TT_CACHE_DIR` | `~/.cache/tensortorrent` | Artifact/pack cache root; set when `$HOME` is not writable |

On a read-only container root, point `TT_CACHE_DIR` at a writable volume. Left
unset the cache follows `$HOME`, which cannot be written under
`readOnlyRootFilesystem`.

Budget-related env vars (`TT_HOST_MEMORY_RESERVE_BYTES`, `TT_VRAM_HEADROOM_BYTES`,
`TT_SPILL_DIR`, `TT_ALLOW_TMPFS_SPILL`) and the `max_total_spill_bytes`
`CompileConfig` field are documented in
[Resource budgets and guardrails](resource_budgets.md).

Tune queue depth and concurrency from measured target latency, memory pressure, and
backend stability. Larger values are not automatically faster.

### Metrics

The `/metrics` endpoint exposes Prometheus text format. Key metrics:

| Metric | Type | Labels | Description |
| --- | --- | --- | --- |
| `tensortorrent_inference_latency_seconds` | histogram | `model` | Per-model inference latency; p95/p99 computable from buckets |
| `tensortorrent_model_requests_total` | counter | `model`, `outcome` | Per-model outcome counts (`success`, `failed`, `cancelled`, `timeout`) |
| `tensortorrent_requests_cancelled_total` | counter | — | Cancelled or timed-out inferences |
| `tensortorrent_queue_rejects_total` | counter | — | Requests rejected by queue backpressure |

## CPU container

The repository `Dockerfile` builds a native wheel in a Rust builder and installs
only runtime dependencies in a non-root Python image. It contains no compiler or
development test dependencies. The default command expects a verified artifact at
`/models/model` and listens on port `8080`.

```bash
docker build --tag tensortorrent:0.1.0 .
docker run --rm --read-only \
	--mount type=bind,src="$PWD/model_artifact",dst=/models/model,readonly \
	--tmpfs /tmp:rw,noexec,nosuid,size=64m \
	--tmpfs /home/tensortorrent:rw,noexec,nosuid,size=64m \
	--publish 127.0.0.1:8080:8080 \
	tensortorrent:0.1.0
```

## CUDA GPU container

`Dockerfile.cuda` mirrors the CPU container structure (same non-root UID,
`HEALTHCHECK`, `STOPSIGNAL SIGTERM`, read-only-friendly layout) but installs
CUDA-enabled PyTorch wheels from the `cu124` index. Requires
`nvidia-container-toolkit` on the host.

```bash
docker build -f Dockerfile.cuda -t tensortorrent:cuda .
docker run --gpus all \
  --env TT_SERVE_AUTH_TOKEN="$(cat /run/secrets/tt_token)" \
  --env TT_LOG_FORMAT=json \
  -v /path/to/models:/models:ro \
  -p 127.0.0.1:8080:8080 \
  tensortorrent:cuda
```

> **Important:** `Dockerfile.cuda` has not been validated on a GPU host. Run
> the smoke tests documented inside the file on your GPU host before promoting
> to production. See the `docker run --rm --gpus all` smoke commands at the top
> of `Dockerfile.cuda`.

## Deploy examples

The `deploy/` directory contains ready-to-use examples:

- `deploy/docker-compose.yaml` — single-host CPU serving with an artifact volume mount
- `deploy/k8s/` — Kubernetes manifests (Deployment, Service, ConfigMap)

## Recommended orchestrator policy

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

## Capacity planning

Capacity numbers are machine-dependent. There is no universal formula. Measure
on the target hardware using these commands:

```bash
tensortorrent doctor --full                      # resolved budgets and provenance
tensortorrent validate-hardware --full           # execution and transfer baselines
tensortorrent profile --all-resources            # per-device kernel latencies
tensortorrent benchmark-topology                 # transfer bandwidth matrix
make coverage                                    # Python test suite with coverage
```

Start with the smallest concurrency and queue depth that meets your latency
target under representative load, then increase incrementally while watching
`tensortorrent_queue_rejects_total` and memory pressure from `tensortorrent doctor`.

## Runbook

### /ready returns 503

**Cause:** No model is loaded, or at least one device worker is unhealthy.

**Action:** Check `/health` for worker status. If a model volume is missing or
the artifact failed integrity verification, the service remains unready. Verify
the artifact path, re-run `tensortorrent validate-hardware`, and check
structured logs (`TT_LOG_FORMAT=json`) for the root cause.

### `queue_rejects_total` climbing

**Cause:** Request rate exceeds the combination of concurrency and queue depth
(`TT_SERVE_MAX_QUEUE_DEPTH`).

**Action:** First, check whether latency is increasing — queue saturation often
follows a latency regression. If the machine has headroom, increase
`TT_SERVE_MAX_QUEUE_DEPTH` and `TT_SERVE_DEFAULT_CONCURRENCY`. If the machine
is memory- or compute-saturated, scale out (additional container replicas) or
reduce model size. Never increase queue depth to hide OOM pressure.

### Container OOMKilled

**Cause:** The container's memory limit is too small for the resolved allowed
budget plus OS overhead, or spill is writing to an in-memory tmpfs mount
instead of persistent disk.

**Action:**
1. Run `tensortorrent doctor` inside the container to see the resolved budget
   and its provenance. Check that `source=cgroup_v2` and that `allowed_bytes`
   fits the model.
2. Verify `TT_SPILL_DIR` points to a persistent volume, not `/tmp` (which is
   tmpfs in most container runtimes). Set `TT_SPILL_DIR=/data/spill` and mount
   a real volume there.
3. Increase the container memory limit if the model legitimately requires more.
4. Use `CompileConfig.polite()` on memory-constrained hosts to reduce peak
   pinned-memory footprint.

### Spill `DiskSpaceError`

**Cause:** Less than 64 MiB of free space remains on the spill filesystem, or
the aggregate cap (`CompileConfig.max_total_spill_bytes`) is exhausted.

**Action:** Free disk space or increase the spill volume. Check
`CompileConfig.max_total_spill_bytes` — the default is 80 % of free disk at session start;
if the disk was nearly full when the session started, the cap may be very small.
Session directories under the spill root with prefix `tt_native_spill_` from
dead processes can be removed safely; the startup sweep does this automatically,
but only for processes that have exited.

### Stalled errors

**Cause:** No instruction completed for `stall_timeout_s` seconds (default
300 s). Likely causes: a lost completion channel, a deadlocked resource, or
pathologically slow I/O.

**Action:** Check system load, I/O wait (`iostat`), and GPU health
(`nvidia-smi`). If the host legitimately has very slow I/O (e.g. network-backed
storage), raise `stall_timeout_s` in `CompileConfig`. If GPU workers are
hanging, check `dmesg` for driver errors. If a CUDA kernel is deadlocked,
`SIGKILL` the container and investigate the driver state.

### 401 after enabling `TT_SERVE_AUTH_TOKEN`

**Cause:** The `Authorization: Bearer <token>` header is missing or the token
does not match.

**Action:** `/health` and `/ready` are exempt from auth — use them to confirm
the service is running. All other endpoints require:
```
Authorization: Bearer <your-token>
```
Verify the token value with `echo $TT_SERVE_AUTH_TOKEN`. Note that the token
is compared with `hmac.compare_digest` (constant-time) — whitespace and
newlines at the end of a file-read token are common sources of mismatch.

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

Deployment fingerprints include installed `tensortorrent.backends` entry-point
metadata. Changing a plugin package or version invalidates specialization caches.
Run `tensortorrent doctor --full` after any driver, PyTorch, plugin, firmware, or
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
`tensortorrent_requests_cancelled_total` and `tensortorrent_queue_rejects_total`.
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
`validate-hardware` exits non-zero unless `production_ready` is true (measured
basic execution for every enabled accelerator plus numerical correctness).
Hardware tests live under `tests/hardware/` (CUDA always when present; ROCm/XPU
skip cleanly when silicon is absent).
