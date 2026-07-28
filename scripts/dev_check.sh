#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
python -m ruff check src tests
python -m streamcompiler.cli.main doctor
python -m streamcompiler.cli.main benchmark-topology --output /tmp/streamcompiler-topology.json >/dev/null
echo "dev_check_ok"
