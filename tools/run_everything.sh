#!/usr/bin/env bash
# Run every check and benchmark that needs real hardware, and collect the
# output in one place.
#
# Almost everything about TensorTorrent has only ever been validated on a
# CPU-only container: the GPU tests have never executed, and the features that
# justify the project (multi-device placement, parameter streaming, activation
# spill, NUMA) have never been measured. This script runs all of it on a
# machine that actually has a GPU and writes the results to a directory you can
# publish or paste back.
#
#   bash tools/run_everything.sh                  # everything
#   SKIP_BUILD=1 bash tools/run_everything.sh     # reuse an existing .venv
#   OUT=/tmp/tt bash tools/run_everything.sh      # choose the output directory
#
# Nothing here is destructive, but note that the hardware tests and the
# oversized-model benchmark are deliberately resource-hungry: they will try to
# fill VRAM and spill to disk. Don't run them on a machine doing other work.

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
  echo "cpu:     $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | xargs || echo unknown)"
  echo "cores:   $(nproc 2>/dev/null || echo '?')"
  echo "memory:  $(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo '?')"
  echo "numa:    $(lscpu 2>/dev/null | grep -i 'NUMA node(s)' | xargs || echo unknown)"
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
stage "validate-hardware" uv run tensortorrent validate-hardware --full
stage "lint-and-types"    uv run python tools/check.py
stage "tests-cpu"         uv run pytest -q -m "not hardware" --timeout 600
stage "tests-hardware"    uv run pytest -q -m hardware --timeout 1800

# ---- the numbers ---------------------------------------------------------
stage "bench-gpu"       uv run python bench/compare_baselines.py --device cuda --iters 30 \
                          --json "$OUT/bench-gpu.json" --markdown "$OUT/bench-gpu.md"
stage "bench-cpu"       uv run python bench/compare_baselines.py --device cpu --iters 30 \
                          --json "$OUT/bench-cpu.json" --markdown "$OUT/bench-cpu.md"
stage "bench-oversized" uv run python bench/oversized_model.py --vram-multiple 1.5 \
                          --json "$OUT/bench-oversized.json"
stage "bench-streaming" uv run python bench/run_streaming.py
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
  if [ -f "$OUT/bench-oversized.log" ]; then
    echo "## Oversized model (the differentiator)"
    echo
    echo '```'
    tail -25 "$OUT/bench-oversized.log"
    echo '```'
  fi
  echo
  echo "## What to do with this"
  echo
  echo "The oversized-model table is the one that matters. If TensorTorrent"
  echo "completes where \`gpu eager\` OOMs, that is the core claim demonstrated."
  echo "If it is also faster than \`accelerate device_map\`, that is a real"
  echo "result worth publishing. If it is slower, publish it anyway and treat"
  echo "the gap as the roadmap."
} >"$SUMMARY"
rm -f "$SUMMARY.tmp"

log "done"
echo "Results: $OUT"
echo "Summary: $SUMMARY"
