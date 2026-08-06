# Local development targets. Prefer `uv run python tools/check.py` when Make is absent.
UV ?= uv
PYTHON ?= .venv/bin/python
.PHONY: sync check format test hardware-test doctor build native-gate bench-smoke cargo-test cargo-clippy rust-fmt pre-commit pre-commit-install

sync:
	$(UV) sync --extra dev
	# release-quick: no LTO, parallel codegen — faster local rebuilds than
	# profile.release (thin LTO + codegen-units=1). Wheels still use --release.
	$(UV) run maturin develop --profile release-quick

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
	$(UV) run tensortorrent doctor

build:
	$(UV) run maturin build --release

native-gate:
	@if [ ! -f crates/tt-python/Cargo.toml ]; then \
		echo "native Rust extension crate missing"; exit 1; \
	fi
	$(UV) run python -c "from tensortorrent.native import require_native; require_native(); print('native import OK')"
	$(UV) run python tools/native_gate.py

# Optional target-gate smoke (not part of default `check`). Records p50 when possible.
bench-smoke:
	$(UV) run python bench/compare_baselines.py --smoke

cargo-test:
	LIBDIR=$$($(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")'); \
	LD_LIBRARY_PATH="$$LIBDIR$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH}" \
	PYO3_PYTHON=$(abspath $(PYTHON)) cargo test --workspace

cargo-clippy:
	PYO3_PYTHON=$(abspath $(PYTHON)) cargo clippy --workspace --all-targets --all-features -- -D warnings

rust-fmt:
	cargo fmt --check

# ── CI / supply-chain targets (new) ──────────────────────────────────────────
.PHONY: audit coverage

audit:
	# Run both Rust and Python supply-chain audits.
	# To suppress a known RUSTSEC advisory add it to .cargo/audit.toml:
	#   [advisories]
	#   ignore = ["RUSTSEC-XXXX-YYYY"]
	# To suppress a pip-audit finding add --ignore-vuln PYSEC-XXXX-YYYY below.
	cargo audit
	$(UV) run --with pip-audit pip-audit --skip-editable

coverage:
	# Run the test suite with coverage. Build fails if coverage < 70 %.
	# Increase --cov-fail-under as the test suite matures.
	$(UV) run pytest -q -m "not hardware" --cov=tensortorrent --cov-report=term --cov-fail-under=70
