#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run)
else
  RUN=()
fi
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
"${RUN[@]}" python "$ROOT/tools/check.py"
"${RUN[@]}" python -m tensortorrent.cli.main benchmark-topology --output /tmp/tensortorrent-topology.json >/dev/null
if [[ -d "$ROOT/native" || -f "$ROOT/CMakeLists.txt" ]]; then
  echo "native sources present without a documented build; refuse"
  exit 1
fi
echo "dev_check_ok"
