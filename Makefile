# Local development targets. Prefer `python3 scripts/check.py` when Make is absent.
PYTHON ?= python3
.PHONY: check format test doctor build native-gate

check:
	$(PYTHON) scripts/check.py

format:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

test:
	$(PYTHON) -m pytest -q

doctor:
	$(PYTHON) -m streamcompiler.cli.main doctor

build:
	$(PYTHON) -m build

native-gate:
	@if [ -d native ] || [ -f CMakeLists.txt ]; then \
		echo "native sources present without a documented build path"; exit 1; \
	fi
	@echo "no native extension; Python package only"
