# syntax=docker/dockerfile:1.7
# CPU production image. HTTP has no built-in auth; deploy behind an authenticated proxy.

ARG RUST_VERSION=1.85
ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.11.21
ARG MATURIN_VERSION=1.14.1

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

FROM rust:${RUST_VERSION}-bookworm AS wheel-builder
ARG MATURIN_VERSION
COPY --from=uv-bin /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY Cargo.toml Cargo.lock pyproject.toml README.md ./
COPY crates ./crates
COPY python ./python
RUN uv venv --python python3 /opt/build-venv \
    && uv pip install --python /opt/build-venv/bin/python "maturin==${MATURIN_VERSION}" \
    && PYO3_PYTHON=/opt/build-venv/bin/python /opt/build-venv/bin/maturin build \
        --release --locked --interpreter /opt/build-venv/bin/python --out /dist

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
ARG TORCH_VERSION=2.13.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG SERVICE_UID=10001
ARG SERVICE_GID=10001
COPY --from=uv-bin /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${SERVICE_GID}" streamcompiler \
    && useradd --uid "${SERVICE_UID}" --gid "${SERVICE_GID}" --create-home streamcompiler
COPY --from=wheel-builder /dist /tmp/wheels
RUN uv pip install --system --no-cache --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" \
    && uv pip install --system --no-cache /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels /root/.cache /usr/local/bin/uv

ENV HOME=/home/streamcompiler \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
USER streamcompiler:streamcompiler
WORKDIR /home/streamcompiler
VOLUME ["/models"]
EXPOSE 8080
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ready', timeout=2).read()"]
CMD ["streamcompiler-serve", "--listen", "0.0.0.0:8080", "--artifact", "/models/model", "--model-id", "default"]
