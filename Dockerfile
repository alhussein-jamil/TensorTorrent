# StreamCompiler container image (CPU / virtual path)
# HTTP has no auth — place behind a trusted network / reverse proxy.

FROM rust:1.85-bookworm AS rust-build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY rust ./rust
RUN cargo build --release -p sc-runtime -p sc-backend-cpu -p sc-ir

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md Cargo.toml Cargo.lock ./
COPY python ./python
COPY rust ./rust
COPY server ./server
COPY docs ./docs
RUN pip install --no-cache-dir maturin torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e ".[dev]" \
 && maturin develop --release
ENV PYTHONPATH=/app/python:/app
EXPOSE 8080
CMD ["python", "-m", "server.cli", "--listen", "0.0.0.0:8080"]
