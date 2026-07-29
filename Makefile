# Local development targets. Prefer `python3 scripts/check.py` when Make is absent.
PYTHON ?= .venv/bin/python
.PHONY: check format test doctor build native-gate cargo-test cargo-clippy rust-fmt

check:
	$(PYTHON) scripts/check.py

format:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests
	cargo fmt

test:
	$(PYTHON) -m pytest -q

doctor:
	$(PYTHON) -m streamcompiler.cli.main doctor

build:
	maturin build --release

native-gate:
	@if [ ! -f crates/streamcompiler-python/Cargo.toml ]; then \
		echo "native Rust extension crate missing"; exit 1; \
	fi
	$(PYTHON) -c "from streamcompiler.native import require_native; require_native(); print('native import OK')"
	$(PYTHON) scripts/native_gate.py

cargo-test:
	cargo test --workspace

cargo-clippy:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

rust-fmt:
	cargo fmt --check
