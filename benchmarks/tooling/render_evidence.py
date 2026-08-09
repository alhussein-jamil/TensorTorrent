"""Render publication figures + polished report from frozen raw JSON.

Does not rerun benchmarks. Reads ``evidence/<version>/raw/`` and writes
``figures/`` plus ``README.md``.

Usage::

    python -m benchmarks.tooling.render_evidence --evidence benchmarks/evidence/v0.3.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Restrained palette (print-friendly).
_TT = "#1F4E79"
_EAGER = "#8B2942"
_ACCENT = "#2F6F4E"
_MUTED = "#5C6670"
_GRID = "#E6E8EB"
_FAIL = "#B85C38"


def _load_summary(raw: Path) -> dict[str, Any]:
    summary_path = raw / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    suites: dict[str, Any] = {}
    mapping = {
        "fit.json": "fit",
        "beyond_vram_deepmlp.json": "beyond_vram_deepmlp",
        "transformer_beyond_vram.json": "transformer_beyond_vram",
        "memory_budget_curve.json": "memory_budget_curve",
        "model_size_crossover.json": "model_size_crossover",
        "heterogeneous.json": "heterogeneous",
    }
    env: dict[str, Any] = {}
    env_path = raw / "environment.json"
    if env_path.exists():
        env = json.loads(env_path.read_text(encoding="utf-8"))
    for fname, key in mapping.items():
        path = raw / fname
        if path.exists():
            suites[key] = json.loads(path.read_text(encoding="utf-8"))
    return {"environment": env, "suites": suites, "smoke": False}


def _run_ok(run: Any) -> bool:
    return isinstance(run, dict) and bool(run.get("ok"))


def _ms(run: Any) -> float | None:
    if not _run_ok(run):
        return None
    med = float(run.get("median_ms") or 0.0)
    return med if med > 0 else None


def _style_axes(ax: Any, *, ylabel: str, xlabel: str, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", color="#1A1D21", pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=_MUTED)
    ax.set_ylabel(ylabel, fontsize=10, color=_MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C5CAD0")
    ax.spines["bottom"].set_color("#C5CAD0")
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _savefig(fig: Any, path: Path) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    png = path.with_suffix(".png")
    svg = path.with_suffix(".svg")
    fig.savefig(png, dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    written.extend([png, svg])
    return written


def _eager_fits(ge: dict[str, Any]) -> bool:
    if not ge.get("ok"):
        return False
    note = str(ge.get("note") or "").lower()
    return "fit" in note or float(ge.get("median_ms") or 0) > 0


def render_crossover(suites: dict[str, Any], out: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    payload = suites.get("model_size_crossover") or {}
    rows = payload.get("results") or []
    if not rows:
        return []

    xs: list[float] = []
    tt_ys: list[float] = []
    eager_ok: list[bool] = []
    for row in rows:
        apps = row.get("approaches") or {}
        tt = _ms(apps.get("tensortorrent"))
        if tt is None:
            continue
        xs.append(float(row.get("vram_multiple") or 0))
        tt_ys.append(tt)
        eager_ok.append(_eager_fits(apps.get("gpu_eager") or {}))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fail_x = [x for x, ok in zip(xs, eager_ok, strict=True) if not ok]
    if fail_x:
        ax.axvspan(min(fail_x) - 0.05, max(xs) + 0.05, color="#F4E8E2", alpha=0.9, zorder=0)
        ax.text(
            (min(fail_x) + max(xs)) / 2,
            max(tt_ys) * 0.92,
            "GPU eager OOM / infeasible",
            ha="center",
            va="top",
            fontsize=9,
            color=_FAIL,
        )

    ax.plot(xs, tt_ys, color=_TT, marker="o", linewidth=2.0, markersize=6, label="TensorTorrent", zorder=3)
    fit_x = [x for x, ok in zip(xs, eager_ok, strict=True) if ok]
    if fit_x:
        y0 = max(tt_ys) * 0.04
        ax.scatter(fit_x, [y0] * len(fit_x), marker="D", s=36, color=_EAGER, label="GPU eager fits", zorder=4)
        for x in fit_x:
            ax.annotate(
                "fits", (x, y0), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=_EAGER
            )

    ax.axvline(1.0, color="#9AA3AB", linestyle="--", linewidth=1.0)
    ax.text(1.02, max(tt_ys) * 0.55, "1× VRAM", rotation=90, va="center", fontsize=8, color=_MUTED)
    _style_axes(
        ax,
        title="Model size vs forward latency (DeepMLP crossover)",
        xlabel="Model parameter footprint (× device VRAM)",
        ylabel="Median latency (ms)",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    written = _savefig(fig, out / "crossover_latency")
    plt.close(fig)
    return written


def render_budget(suites: dict[str, Any], out: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    payload = suites.get("memory_budget_curve") or {}
    points: list[tuple[float, float, float | None]] = []
    for row in payload.get("results") or []:
        tt = row.get("tensortorrent") or {}
        med = _ms(tt)
        if med is None:
            continue
        budget = row.get("vram_budget_gib")
        if budget is None:
            budget = float(row.get("vram_budget_bytes") or 0) / (1024**3)
        extras = tt.get("extras") or {}
        inst = extras.get("instrumentation") if isinstance(extras.get("instrumentation"), dict) else extras
        xfer = (int(inst.get("transfer_bytes_h2d") or 0) + int(inst.get("transfer_bytes_d2h") or 0)) / 1e9
        points.append((float(budget), med, xfer if xfer > 0 else None))
    if not points:
        return []
    points.sort()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xfer = [p[2] for p in points]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(xs, ys, color=_TT, marker="o", linewidth=2.0, markersize=6, label="Latency")
    ax2 = ax.twinx()
    if any(v is not None for v in xfer):
        ax2.plot(
            xs,
            [v if v is not None else float("nan") for v in xfer],
            color=_ACCENT,
            marker="s",
            linewidth=1.6,
            markersize=5,
            linestyle="--",
            label="Transfer volume",
        )
    _style_axes(
        ax,
        title="VRAM budget vs TensorTorrent cost",
        xlabel="Configured VRAM budget (GiB)",
        ylabel="Median latency (ms)",
    )
    ax2.set_ylabel("Transfer volume (GB)", fontsize=10, color=_MUTED)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(colors=_MUTED, labelsize=9)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    written = _savefig(fig, out / "budget_latency_transfer")
    plt.close(fig)
    return written


def render_qwen_memory(suites: dict[str, Any], env: dict[str, Any], out: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    xf = suites.get("transformer_beyond_vram") or {}
    spec = xf.get("transformer_spec") or {}
    params = float(xf.get("params_bytes") or spec.get("param_bytes") or 0) / 1e9
    vram = float((env.get("gpu0") or {}).get("total_memory_bytes") or env.get("gpu_vram_bytes") or 0) / 1e9
    tt = (xf.get("approaches") or {}).get("tensortorrent") or {}
    peak = float(tt.get("peak_device_bytes") or 0) / 1e9
    if params <= 0 or vram <= 0 or peak <= 0:
        return []

    labels = ["Parameters\n(Qwen3-8B BF16)", "Physical\nGPU VRAM", "TensorTorrent\npeak allocated"]
    values = [params, vram, peak]
    colors = [_MUTED, _EAGER, _TT]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    bars = ax.bar(labels, values, color=colors, width=0.62, edgecolor="white")
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(values) * 0.02,
            f"{val:.2f} GB",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1A1D21",
            fontweight="bold",
        )
    _style_axes(
        ax,
        title="Qwen3-8B fixed-shape forward — memory footprint",
        xlabel="",
        ylabel="Gigabytes",
    )
    ax.set_ylim(0, max(values) * 1.18)
    fig.tight_layout()
    written = _savefig(fig, out / "qwen_memory_footprint")
    plt.close(fig)
    return written


def render_fit_overhead(suites: dict[str, Any], out: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = (suites.get("fit") or {}).get("results") or []
    names: list[str] = []
    eager: list[float] = []
    tt: list[float] = []
    for row in rows:
        apps = row.get("approaches") or {}
        em = _ms(apps.get("eager"))
        tm = _ms(apps.get("tensortorrent"))
        if em is None or tm is None:
            continue
        names.append(str(row.get("name") or row.get("workload") or "?").replace("_", " "))
        eager.append(em)
        tt.append(tm)
    if not names:
        return []

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - width / 2, eager, width, color=_EAGER, label="PyTorch eager")
    ax.bar(x + width / 2, tt, width, color=_TT, label="TensorTorrent")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    _style_axes(
        ax,
        title="Fit-in-VRAM overhead — native PyTorch is faster here",
        xlabel="",
        ylabel="Median latency (ms)",
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    written = _savefig(fig, out / "fit_overhead")
    plt.close(fig)
    return written


def write_report(evidence: Path, summary: dict[str, Any], figure_pngs: list[str]) -> None:
    env = summary.get("environment") or {}
    suites = summary.get("suites") or {}
    commit = str(env.get("commit") or "unknown")
    dirty = env.get("git_dirty")
    ver = env.get("tensortorrent") or "?"
    gpu = (env.get("gpu0") or {}).get("name") or env.get("gpu") or "GPU"
    vram = (env.get("gpu0") or {}).get("total_memory_bytes") or env.get("gpu_vram_bytes")
    vram_gib = f"{float(vram) / (1024**3):.2f} GiB" if vram else "?"

    xf = suites.get("transformer_beyond_vram") or {}
    tt = (xf.get("approaches") or {}).get("tensortorrent") or {}
    cpu = (xf.get("approaches") or {}).get("cpu_eager") or {}
    acc = (xf.get("approaches") or {}).get("accelerate") or {}
    extras = tt.get("extras") or {}
    peak_gb = float(tt.get("peak_device_bytes") or 0) / 1e9
    params_gb = float(xf.get("params_bytes") or (xf.get("transformer_spec") or {}).get("param_bytes") or 0) / 1e9

    cross_lines: list[str] = []
    for row in (suites.get("model_size_crossover") or {}).get("results") or []:
        apps = row.get("approaches") or {}
        ttr = apps.get("tensortorrent") or {}
        ge = apps.get("gpu_eager") or {}
        strategy = (ttr.get("extras") or {}).get("execution_strategy") or "?"
        ge_s = "fits" if _eager_fits(ge) else "OOM"
        tt_s = f"{float(ttr.get('median_ms') or 0):.0f} ms" if ttr.get("ok") else "fail"
        cross_lines.append(f"| {float(row.get('vram_multiple') or 0):.2f}× | {ge_s} | {tt_s} | `{strategy}` |")

    fig_block = "\n\n".join(f"![{Path(p).stem}](figures/{p})" for p in figure_pngs)
    ram_gib = float(env.get("host_ram_total_bytes") or 0) / (1024**3)

    body = f"""# TensorTorrent {ver} — capacity benchmarks

Human-readable report for the frozen **v0.3.1** evidence.
Machine-readable JSON: [`raw/`](raw/). Figures regenerated from that JSON
(`python -m benchmarks.tooling.render_evidence`).

## Snapshot

| | |
| --- | --- |
| Package | `{ver}` |
| Measured commit | `{commit}` |
| `git_dirty` | `{dirty}` |
| Host GPU | {gpu} ({vram_gib}) |
| Host RAM | {ram_gib:.0f} GiB |
| PyTorch | {env.get("torch")} |
| CUDA / driver | {env.get("cuda")} / {env.get("nvidia_driver") or env.get("cuda_driver_version")} |

## What this measures

TensorTorrent is a **capacity-oriented** heterogeneous runtime. These benches answer:

1. **Beyond VRAM** — can a fixed-shape forward complete when parameters exceed device memory?
2. **Crossover** — when does residency give way to Transfer/Evict streaming?
3. **Fit-in-VRAM** — what overhead remains when the model already fits?

Autoregressive generation, multi-GPU, and alternate Accelerate configs remain
**SUPPORTED BUT UNMEASURED** on this host.

## Headline — Qwen3-8B fixed-shape logits forward

Not generation. BF16, `seq_len=16`, exportable logits forward only.

| | |
| --- | ---: |
| Parameter footprint | **{params_gb:.2f} GB** |
| Physical VRAM | **{vram_gib}** |
| TensorTorrent median | **{float(tt.get("median_ms") or 0):.0f} ms** |
| TensorTorrent peak allocated VRAM | **{peak_gb:.2f} GB** |
| CPU eager median | {float(cpu.get("median_ms") or 0):.0f} ms |
| Tested Accelerate (`device_map=auto`) | {"OOM" if not acc.get("ok") else "ok"} |
| Correctness | cosine {extras.get("cosine")} · argmax {extras.get("argmax_match")}/{extras.get("argmax_total")} |

TensorTorrent keeps peak VRAM far below the parameter footprint by streaming /
Transfer–Evict through the accelerator.

## Figures

{fig_block}

## Model-size crossover (DeepMLP)

Resident under the safe headroom fraction; Transfer/Evict near and beyond VRAM.

| Size × VRAM | GPU eager | TensorTorrent | Strategy |
| --- | --- | --- | --- |
{chr(10).join(cross_lines)}

## Reproduce

```bash
uv sync --extra dev --extra bench
python -m benchmarks.smoke
python -m benchmarks.public --suite crossover
python -m benchmarks.public --suite transformer --model-id Qwen/Qwen3-8B --seq-len 16

# Freeze JSON into raw/ (clean tree required), then refresh this report:
python -m benchmarks.tooling.freeze --src benchmarks/results/<run> --dst benchmarks/evidence/v0.3.1/raw
python -m benchmarks.tooling.render_evidence --evidence benchmarks/evidence/v0.3.1
```

Ephemeral outputs stay in `benchmarks/results/` (gitignored).
"""
    (evidence / "README.md").write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--evidence",
        type=Path,
        default=Path("benchmarks/evidence/v0.3.1"),
        help="Evidence directory containing raw/",
    )
    args = ap.parse_args(argv)
    evidence = args.evidence.resolve()
    raw = evidence / "raw"
    if not raw.is_dir():
        raise SystemExit(f"missing raw dir: {raw}")
    summary = _load_summary(raw)
    suites = summary.get("suites") or {}
    env = summary.get("environment") or {}
    fig_dir = evidence / "figures"
    written: list[Path] = []
    written += render_crossover(suites, fig_dir)
    written += render_budget(suites, fig_dir)
    written += render_qwen_memory(suites, env, fig_dir)
    written += render_fit_overhead(suites, fig_dir)
    pngs = sorted({p.name for p in written if p.suffix == ".png"})
    write_report(evidence, summary, pngs)
    for path in written:
        print(path)
    print(f"wrote {evidence / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
