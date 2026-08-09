"""Freeze ephemeral ``benchmarks/results/<run>/`` into a tracked published snapshot."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KEEP_FILES = (
    "environment.json",
    "summary.json",
    "REPORT.md",
    "fit.json",
    "beyond_vram_deepmlp.json",
    "beyond_vram.json",
    "transformer_beyond_vram.json",
    "memory_budget_curve.json",
    "model_size_crossover.json",
    "heterogeneous.json",
)


def _strip_samples(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "samples_ms":
                continue
            out[k] = _strip_samples(v)
        return out
    if isinstance(obj, list):
        return [_strip_samples(v) for v in obj]
    return obj


def freeze(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in KEEP_FILES:
        path = src / name
        if not path.exists():
            continue
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            (dst / name).write_text(json.dumps(_strip_samples(payload), indent=2) + "\n", encoding="utf-8")
        else:
            shutil.copy2(path, dst / name)
    for png in sorted(src.glob("*.png")):
        shutil.copy2(png, dst / png.name)

    env_path = dst / "environment.json"
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}
    readme = dst / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Published benchmark snapshot",
                "",
                f"- Measured commit: `{env.get('commit', 'unknown')}`",
                f"- git_dirty: `{env.get('git_dirty')}`",
                f"- tensortorrent: `{env.get('tensortorrent', 'unknown')}`",
                f"- timestamp_utc: `{env.get('timestamp_utc', 'unknown')}`",
                "",
                "Ephemeral runs stay under `benchmarks/results/` (gitignored).",
                "This directory is the frozen, reconstructable public evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    if not args.src.is_dir():
        raise SystemExit(f"missing src dir: {args.src}")
    freeze(args.src.resolve(), args.dst.resolve())
    print(f"froze {args.src} → {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
