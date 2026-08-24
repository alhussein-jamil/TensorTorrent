#!/usr/bin/env bash
# Run the full local gate plus hardware suite and benches; write logs under OUT.
#
# Use on a machine with the accelerators you care about. Stages that need a GPU
# will fail clearly when hardware is absent; CPU stages still run.
#
#   bash tools/run_everything.sh                  # everything
#   SKIP_BUILD=1 bash tools/run_everything.sh     # reuse an existing .venv
#   OUT=/tmp/tt bash tools/run_everything.sh      # choose the output directory
#
# Hardware tests and the public beyond-VRAM suites are resource-hungry (VRAM
# fill, spill to disk). Avoid running them on a busy shared machine.

set -uo pipefail   # deliberately not -e: a failing stage must not abort the run

OUT="${OUT:-bench-results/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.md"

log()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
stage() {
  local name="$1"; shift
  log "$name"
  local f="$OUT/${name// /_}.log"
  if "$@" >"$f" 2>&1; then
    echo "PASS  $name" | tee -a "$SUMMARY.tmp"
  else
    echo "FAIL  $name  (see ${f##*/})" | tee -a "$SUMMARY.tmp"
  fi
  tail -5 "$f" | sed 's/^/    /'
}

: >"$SUMMARY.tmp"

log "environment"
{
  echo "date:    $(date -Is)"
  echo "host:    $(uname -a)"
  echo "cpu:     $(sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | xargs || echo unknown)"
  echo "cores:   $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo '?')"
  echo "memory:  $(sysctl -n hw.memsize 2>/dev/null || free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo '?')"
  echo "numa:    $(lscpu 2>/dev/null | grep -i 'NUMA node(s)' | xargs || echo 'single domain')"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "nvidia-smi: not present"
  command -v rocm-smi >/dev/null && rocm-smi --showproductname 2>/dev/null | head -5 || true
} | tee "$OUT/environment.txt"

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  log "build (uv sync + maturin)"
  uv sync --extra dev >"$OUT/build.log" 2>&1
  uv run maturin develop --release >>"$OUT/build.log" 2>&1 \
    && echo "PASS  build" | tee -a "$SUMMARY.tmp" \
    || { echo "FAIL  build — stopping" | tee -a "$SUMMARY.tmp"; tail -30 "$OUT/build.log"; exit 1; }
fi

# ---- correctness ---------------------------------------------------------
stage "doctor"            uv run tensortorrent doctor
stage "validate-hardware" uv run tensortorrent validate-hardware --stress
stage "lint-and-types"    uv run python tools/check.py
stage "tests-cpu"         uv run pytest -q -m "not hardware" --timeout 600
stage "tests-hardware"    uv run pytest -q -m hardware --timeout 1800

# ---- the numbers ---------------------------------------------------------
stage "bench-suite"     uv run python -m benchmarks.public --suite all --iters 20 \
                          --out "$OUT/benchmarks"
stage "bench-cpu-peers" uv run python benchmarks/micro/compare_baselines.py --device cpu --iters 30 \
                          --json "$OUT/bench-cpu.json" --markdown "$OUT/bench-cpu.md"
stage "bench-streaming" uv run python benchmarks/micro/run_streaming.py
stage "bench-topology"  uv run tensortorrent benchmark-topology

# ---- report --------------------------------------------------------------
{
  echo "# TensorTorrent hardware run"
  echo
  echo "Generated $(date -Is)"
  echo
  echo '```'
  cat "$OUT/environment.txt"
  echo '```'
  echo
  echo "## Stages"
  echo
  sed 's/^/- /' "$SUMMARY.tmp"
  echo
  for f in bench-gpu.md bench-cpu.md; do
    [ -f "$OUT/$f" ] && { echo "## ${f%.md}"; echo; cat "$OUT/$f"; echo; }
  done
  echo
  echo "## What to do with this"
  echo
  echo "The public beyond-VRAM suites (\`benchmarks.public\`) are the"
  echo "differentiator. If TensorTorrent completes where \`gpu eager\` OOMs,"
  echo "that is the core claim demonstrated. Peer bakeoff numbers from"
  echo "\`compare_baselines.py\` are secondary context."
} >"$SUMMARY"
rm -f "$SUMMARY.tmp"

log "done"
echo "Results: $OUT"
echo "Summary: $SUMMARY"
