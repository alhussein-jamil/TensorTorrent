# StreamCompiler container image (CPU / virtual path)
# HTTP has no auth — place behind a trusted network / reverse proxy.

FROM rust:1.85-bookworm AS rust-build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY rust ./rust
RUN cargo build --release -p sc-runtime -p sc-backend-cpu -p sc-ir

FROM python:3.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md Cargo.toml Cargo.lock uv.lock ./
COPY python ./python
COPY rust ./rust
COPY server ./server
COPY docs ./docs
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv sync --extra dev --no-install-package torch \
 && uv pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && uv sync --extra dev --reinstall-package streamcompiler \
 && uv run maturin develop --release
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/python:/app
EXPOSE 8080
CMD ["python", "-m", "server.cli", "--listen", "0.0.0.0:8080"]
