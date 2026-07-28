#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/scripts/check.py"
python -m streamcompiler.cli.main benchmark-topology --output /tmp/streamcompiler-topology.json >/dev/null
if [[ -d "$ROOT/native" || -f "$ROOT/CMakeLists.txt" ]]; then
  echo "native sources present without a documented build; refuse"
  exit 1
fi
echo "dev_check_ok"
