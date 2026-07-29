# Local development targets. Prefer `python3 scripts/check.py` when Make is absent.
PYTHON ?= python3
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
	@echo "native extension: crates/streamcompiler-python (maturin)"

cargo-test:
	cargo test --workspace --exclude streamcompiler-python

cargo-clippy:
	cargo clippy --workspace --all-targets --exclude streamcompiler-python -- -D warnings

rust-fmt:
	cargo fmt --check
