# syntax=docker/dockerfile:1.7
# CPU production image. Set TT_SERVE_AUTH_TOKEN for built-in bearer-token auth.
#
# Base images are pinned to immutable digests for supply-chain integrity.
# Dependabot will open PRs to update them weekly.
# Override at build time:
#   docker build --build-arg RUST_IMAGE=rust:1.86-bookworm ...

ARG RUST_VERSION=1.85
ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.11.21
ARG MATURIN_VERSION=1.14.1

# Allow digest overrides at build time while keeping pinned defaults.
ARG RUST_IMAGE=rust:1.85-bookworm@sha256:e51d0265072d2d9d5d320f6a44dde6b9ef13653b035098febd68cce8fa7c0bc4
ARG PYTHON_IMAGE=python:3.11-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

FROM ${RUST_IMAGE} AS wheel-builder
ARG MATURIN_VERSION
COPY --from=uv-bin /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY Cargo.toml Cargo.lock pyproject.toml README.md LICENSE ./
COPY crates ./crates
COPY python ./python
RUN uv venv --python python3 /opt/build-venv \
    && uv pip install --python /opt/build-venv/bin/python "maturin==${MATURIN_VERSION}" \
    && PYO3_PYTHON=/opt/build-venv/bin/python /opt/build-venv/bin/maturin build \
        --release --locked --interpreter /opt/build-venv/bin/python --out /dist

FROM ${PYTHON_IMAGE} AS runtime
ARG TORCH_VERSION=2.13.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG SERVICE_UID=10001
ARG SERVICE_GID=10001
COPY --from=uv-bin /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${SERVICE_GID}" tensortorrent \
    && useradd --uid "${SERVICE_UID}" --gid "${SERVICE_GID}" --create-home tensortorrent
COPY --from=wheel-builder /dist /tmp/wheels
RUN uv pip install --system --no-cache --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" \
    && uv pip install --system --no-cache /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels /root/.cache /usr/local/bin/uv

# ── Environment ──────────────────────────────────────────────────────────────
# TT_SERVE_AUTH_TOKEN: pass a secret token via --env or a secrets manager.
#   docker run --env TT_SERVE_AUTH_TOKEN="$(cat /run/secrets/tt_token)" ...
#   Leave unset to run without auth (not recommended in production).
#
# TT_LOG_FORMAT: set to "json" for structured log ingestion (recommended):
#   docker run --env TT_LOG_FORMAT=json ...
ENV HOME=/home/tensortorrent \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TT_LOG_FORMAT=json

USER tensortorrent:tensortorrent
WORKDIR /home/tensortorrent
VOLUME ["/models"]
EXPOSE 8080
STOPSIGNAL SIGTERM

# HEALTHCHECK uses /health, not /ready.
#
# /ready (readiness) answers "is this instance ready to serve traffic?" and
# belongs to the orchestrator (Kubernetes readinessProbe, compose healthcheck
# used for depends_on condition). A container whose model volume is missing
# will correctly answer /ready=503 — but that does NOT mean the process is
# unhealthy; it just needs its volume mounted.  Wiring /ready into Docker's
# HEALTHCHECK causes the daemon to restart the container in an infinite loop
# whenever the model volume is absent, which is the wrong behaviour.
#
# /health answers "is the process alive and its internal state sane?" It
# returns 200 even before the model is loaded, so the daemon only restarts
# the container when the process itself is genuinely broken.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2).read()"]

CMD ["tensortorrent-serve", "--listen", "0.0.0.0:8080", "--artifact", "/models/model", "--model-id", "default"]
