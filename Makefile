# Local development targets. Prefer `uv run python tools/check.py` when Make is absent.
UV ?= uv
PYTHON ?= .venv/bin/python
.PHONY: sync check format test hardware-test doctor build native-gate cargo-test cargo-clippy rust-fmt pre-commit pre-commit-install

sync:
	$(UV) sync --extra dev
	$(UV) run maturin develop --release

check:
	$(UV) run python tools/check.py

pre-commit-install:
	$(UV) run pre-commit install --hook-type pre-commit --hook-type pre-push

pre-commit:
	$(UV) run pre-commit run --all-files
	$(UV) run pre-commit run --all-files --hook-stage pre-push

format:
	$(UV) run ruff format python tests tools
	$(UV) run ruff check --fix python tests tools
	cargo fmt

test:
	$(UV) run pytest -q -m "not hardware"

hardware-test:
	$(UV) run pytest -q -m hardware

doctor:
	$(UV) run streamcompiler doctor

build:
	$(UV) run maturin build --release

native-gate:
	@if [ ! -f crates/sc-python/Cargo.toml ]; then \
		echo "native Rust extension crate missing"; exit 1; \
	fi
	$(UV) run python -c "from streamcompiler.native import require_native; require_native(); print('native import OK')"
	$(UV) run python tools/native_gate.py

cargo-test:
	PYO3_PYTHON=$(abspath $(PYTHON)) cargo test --workspace

cargo-clippy:
	PYO3_PYTHON=$(abspath $(PYTHON)) cargo clippy --workspace --all-targets --all-features -- -D warnings

rust-fmt:
	cargo fmt --check
