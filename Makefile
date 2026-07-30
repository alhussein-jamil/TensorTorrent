# Local development targets. Prefer `uv run python scripts/check.py` when Make is absent.
UV ?= uv
PYTHON ?= .venv/bin/python
.PHONY: sync check format test doctor build native-gate cargo-test cargo-clippy rust-fmt

sync:
	$(UV) sync --extra dev
	$(UV) run maturin develop --release

check:
	$(UV) run python scripts/check.py

format:
	$(UV) run ruff format python tests server
	$(UV) run ruff check --fix python tests server
	cargo fmt

test:
	$(UV) run pytest -q

doctor:
	$(UV) run streamcompiler doctor

build:
	$(UV) run maturin build --release

native-gate:
	@if [ ! -f rust/sc-python/Cargo.toml ]; then \
		echo "native Rust extension crate missing"; exit 1; \
	fi
	$(UV) run python -c "from streamcompiler.native import require_native; require_native(); print('native import OK')"
	$(UV) run python scripts/native_gate.py

cargo-test:
	cargo test --workspace

cargo-clippy:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

rust-fmt:
	cargo fmt --check

