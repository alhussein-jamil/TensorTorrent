# Deployment

TensorTorrent deployment has two separate gates: **package/runtime health** and **target-hardware validation**. Do not treat a successful import or device discovery as sufficient evidence for production traffic.

## Release-to-host sequence

A typical deployment flow is:

```text
build/test package
    -> install on target host
    -> tensortorrent doctor
    -> validate-hardware
    -> compile/specialize artifact
    -> benchmark representative traffic
    -> start service
```

For extended host validation:

```bash
tensortorrent validate-hardware --stress --output artifacts/validation_stress.json
tensortorrent validate-hardware --overnight --output artifacts/validation_overnight.json
```

## Serving a compiled artifact

```bash
tensortorrent serve \
  --artifact artifact/ \
  --model-id model \
  --listen 127.0.0.1:8080
```

Equivalent entry point:

```bash
tensortorrent-serve --artifact artifact/ --model-id model --listen 127.0.0.1:8080
```

The stdlib HTTP server is intentionally small. It provides health/readiness, Prometheus-format metrics, inference, cancellation, queue limits, request timeouts, and optional bearer authentication.

## HTTP endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health`, `/v1/health` | process/service health |
| `GET` | `/ready`, `/readiness`, `/v1/ready` | readiness and model/worker state |
| `GET` | `/metrics`, `/v1/metrics` | Prometheus text metrics |
| `POST` | `/infer`, `/v1/infer` | inference |
| `POST` | `/cancel`, `/v1/cancel` | cancel by request ID |

Inference body:

```json
{
  "model_id": "model",
  "inputs": [1.0, 2.0, 3.0],
  "request_id": "optional-client-id",
  "timeout_s": 30.0
}
```

The JSON interface converts supported nested values to tensors and serializes tensor outputs back to JSON-compatible values. It is intended as a practical built-in service boundary, not a high-throughput binary RPC protocol.

## Authentication

Set `TT_SERVE_AUTH_TOKEN` to require a bearer token on non-health endpoints:

```bash
export TT_SERVE_AUTH_TOKEN='replace-me'
```

Clients send:

```text
Authorization: Bearer replace-me
```

Health and readiness endpoints remain unauthenticated so orchestrator probes can function.

## Service limits

| Environment variable | Default | Purpose |
| --- | ---: | --- |
| `TT_SERVE_MAX_QUEUE_DEPTH` | `64` | queued-request cap |
| `TT_SERVE_DEFAULT_TIMEOUT_S` | `30` | default request timeout |
| `TT_SERVE_MAX_REQUEST_TIMEOUT_S` | `3600` | maximum accepted request timeout |
| `TT_SERVE_DEFAULT_CONCURRENCY` | `8` | default per-model concurrency |
| `TT_SERVE_WORKER_THREADS` | `0` | service thread count; `0` derives from concurrency |
| `TT_SERVE_CANCELLATION_GRACE_S` | `1.0` | cancellation grace period |
| `TT_SERVE_REQUEST_HISTORY_SIZE` | `1024` | retained request records |
| `TT_HTTP_MAX_CONNECTIONS` | `128` | accepted connection cap |
| `TT_HTTP_BACKLOG` | `64` | listen backlog |
| `TT_HTTP_MAX_BODY_BYTES` | `33554432` | request body cap |
| `TT_HTTP_MAX_RESPONSE_BYTES` | `134217728` | estimated response cap |
| `TT_HTTP_SOCKET_TIMEOUT_S` | `30` | per-socket idle timeout |
| `TT_LOG_LEVEL` | `INFO` | logging level |
| `TT_LOG_FORMAT` | `text` | `text` or `json` |

Invalid values fail early rather than silently falling back.

## Health versus readiness

`/health` answers whether the service process is functioning and reports current request/worker statistics.

`/ready` is stricter: the service must be started, not shutting down, device workers must be healthy when present, and at least one model must be loaded.

Use readiness for traffic admission.

## Backpressure

The service uses bounded queues and a connection cap. Saturated HTTP connections are rejected instead of spawning an unbounded thread population. Queue saturation is surfaced through metrics.

## Containers

The repository provides:

- `Dockerfile` for CPU-oriented images,
- `Dockerfile.cuda` for CUDA-oriented images,
- `deploy/docker-compose.yaml`,
- Kubernetes examples under `deploy/` when present in the checkout.

Resource planning is cgroup-aware, so container memory/CPU limits affect specialization. Validate inside the final container configuration.

## Operational checks

Before traffic:

```bash
tensortorrent doctor --full
tensortorrent validate-hardware --output artifacts/validation_report.json
```

After loading the service:

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/ready
curl -fsS http://127.0.0.1:8080/metrics
```

## Failure modes

### Readiness is false

Check that an artifact is loaded and any supervised device workers are alive. `/health` contains worker state and request counters.

### Queue rejects increase

Increase capacity only after measuring model concurrency. Raising queue depth without increasing execution capacity increases tail latency rather than throughput.

### Container is OOM-killed

Verify that specialization ran under the same cgroup limits as serving. Inspect `tensortorrent doctor` output and explicit host/VRAM budgets.

### Spill disk errors

Verify the spill root is not tmpfs, has adequate free space, and is writable by the service user.

### Stalled runtime

Inspect device/runtime health and storage latency. Increase `stall_timeout_s` only when the workload genuinely contains long no-progress intervals; do not use a larger timeout to hide a deadlock.

## Deployment principle

A specialized artifact is a machine-level decision. Revalidate or respecialize when relevant hardware, driver/runtime, PyTorch, or resource-limit inputs change.
