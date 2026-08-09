"""Freeze ephemeral ``benchmarks/results/<run>/`` into a tracked published snapshot.

Refuses dirty worktrees by default so published evidence stays reproducible.
Pass ``--allow-dirty`` only for local debugging (never stderr warning).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from benchmarks.tooling.harness import git_dirty

KEEP_FILES = (
    "environment.json",
    "REPORT.md",
    "fit.json",
    "beyond_vram_deepmlp.json",
    "transformer_beyond_vram.json",
    "memory_budget_curve.json",
    "model_size_crossover.json",
    "heterogeneous.json",
)

_DROP_KEYS = {"samples_ms", "notes", "schedule_notes", "load_meta"}
_INSTR_KEEP = {
    "devices_used",
    "n_regions",
    "region_compute_fraction",
    "regions_by_kind",
    "transfer_count",
    "transfer_bytes_h2d",
    "transfer_bytes_d2h",
    "transfer_wall_fraction",
    "wall_time_s",
    "peak_activation_bytes",
    "allocation_peak_bytes",
}


def slim(obj: Any) -> Any:
    """Drop bulky fields so published JSON stays under pre-commit size limits."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in _DROP_KEYS:
                continue
            if key == "instrumentation" and isinstance(value, dict):
                out[key] = {k: value[k] for k in _INSTR_KEEP if k in value}
                continue
            if key == "parameter_store" and isinstance(value, dict):
                out[key] = {k: value[k] for k in ("kind", "resident_bytes", "tensor_count") if k in value}
                continue
            out[key] = slim(value)
        return out
    if isinstance(obj, list):
        return [slim(v) for v in obj]
    return obj


def freeze(src: Path, dst: Path, *, allow_dirty: bool = False) -> None:
    env_src = src / "environment.json"
    if env_src.exists():
        env_check = json.loads(env_src.read_text(encoding="utf-8"))
        dirty = env_check.get("git_dirty")
        if dirty is True and not allow_dirty:
            raise SystemExit(
                "refusing to publish: environment.json has git_dirty=true. "
                "Commit all changes, remasure from a clean tree, then freeze. "
                "Override with --allow-dirty (not for public evidence)."
            )
        if dirty is True and allow_dirty:
            print(
                "WARNING: publishing with git_dirty=true (--allow-dirty); do not treat this as public evidence.",
                file=sys.stderr,
            )
    live_dirty = git_dirty()
    if live_dirty is True and not allow_dirty:
        raise SystemExit(
            "refusing to publish: worktree is dirty. "
            "Commit all changes, remasure from a clean tree, then freeze. "
            "Override with --allow-dirty (not for public evidence)."
        )

    dst.mkdir(parents=True, exist_ok=True)
    for name in KEEP_FILES:
        path = src / name
        if not path.exists():
            continue
        if path.suffix == ".json":
            payload = slim(json.loads(path.read_text(encoding="utf-8")))
            (dst / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        elif path.name == "REPORT.md":
            # Human report lives in evidence/<ver>/README.md via render_evidence.
            continue
        else:
            shutil.copy2(path, dst / name)
    # Publication figures are regenerated from raw JSON; do not copy ephemeral PNGs.

    # Drop redundant aliases if a previous freeze left them behind.
    for name in ("summary.json", "beyond_vram.json", "memory_pressure.json", "model_size_scaling.json"):
        stale = dst / name
        if stale.exists():
            stale.unlink()

    suites: dict[str, Any] = {}
    mapping = {
        "fit.json": "fit",
        "beyond_vram_deepmlp.json": "beyond_vram_deepmlp",
        "transformer_beyond_vram.json": "transformer_beyond_vram",
        "memory_budget_curve.json": "memory_budget_curve",
        "model_size_crossover.json": "model_size_crossover",
        "heterogeneous.json": "heterogeneous",
    }
    env_path = dst / "environment.json"
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}
    for fname, key in mapping.items():
        path = dst / fname
        if path.exists():
            suites[key] = json.loads(path.read_text(encoding="utf-8"))
    summary = {
        "environment": env,
        "suite": "all",
        "smoke": False,
        "measured_commit": env.get("commit"),
        "git_dirty": env.get("git_dirty"),
        "suites": suites,
    }
    (dst / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    (dst / "README.md").write_text(
        "\n".join(
            [
                "# Raw benchmark evidence",
                "",
                f"- Measured commit: `{env.get('commit', 'unknown')}`",
                f"- git_dirty: `{env.get('git_dirty')}`",
                f"- tensortorrent: `{env.get('tensortorrent', 'unknown')}`",
                f"- timestamp_utc: `{env.get('timestamp_utc', 'unknown')}`",
                "",
                "This directory holds machine-readable JSON only.",
                "Polished report and figures: parent `../README.md` and `../figures/`",
                "(`python -m benchmarks.tooling.render_evidence`).",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow freeze when git_dirty=true (local debug only; not for public evidence).",
    )
    args = ap.parse_args()
    if not args.src.is_dir():
        raise SystemExit(f"missing src dir: {args.src}")
    freeze(args.src.resolve(), args.dst.resolve(), allow_dirty=bool(args.allow_dirty))
    print(f"froze {args.src} → {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
