# Local development targets. Prefer `python scripts/check.py` when Make is absent.
.PHONY: check format test doctor build native-gate

check:
	python scripts/check.py

format:
	python -m ruff format src tests
	python -m ruff check --fix src tests

test:
	python -m pytest -q

doctor:
	python -m streamcompiler.cli.main doctor

build:
	python -m build

native-gate:
	@if [ -d native ] || [ -f CMakeLists.txt ]; then \
		echo "native sources present without a documented build path"; exit 1; \
	fi
	@echo "no native extension; Python package only"
