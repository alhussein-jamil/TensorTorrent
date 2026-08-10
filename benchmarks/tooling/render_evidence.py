"""Render publication figures + polished report from frozen raw JSON.

Does not rerun benchmarks. Reads ``evidence/raw/`` and writes ``figures/``,
``README.md``, and ``REPORT.md``.

Usage::

    python -m benchmarks.tooling.render_evidence --evidence benchmarks/evidence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.tooling.report import render_markdown_tables

# Restrained palette (print-friendly).
_TT = "#1F4E79"
_EAGER = "#8B2942"
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


def _pick_tt(apps: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Prefer auto / primary user-facing approach keys in order."""
    for key in keys:
        run = apps.get(key)
        if isinstance(run, dict) and run.get("ok"):
            return run
    for key in keys:
        run = apps.get(key)
        if isinstance(run, dict):
            return run
    return {}


def _fmt_ms(run: Any) -> str:
    med = _ms(run)
    if med is None:
        return "—"
    if med >= 100:
        return f"{med:.0f}"
    if med >= 10:
        return f"{med:.1f}"
    return f"{med:.2f}"


def _fmt_gb(nbytes: Any) -> str:
    if nbytes is None:
        return "—"
    return f"{float(nbytes) / 1e9:.2f}"


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


def render_qwen_memory(suites: dict[str, Any], env: dict[str, Any], out: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    xf = suites.get("transformer_beyond_vram") or {}
    spec = xf.get("transformer_spec") or {}
    gib = 1024**3
    params = float(xf.get("params_bytes") or spec.get("param_bytes") or 0) / gib
    vram = float((env.get("gpu0") or {}).get("total_memory_bytes") or env.get("gpu_vram_bytes") or 0) / gib
    apps = xf.get("approaches") or {}
    tt = _pick_tt(apps, "tensortorrent_auto", "tensortorrent")
    peak = float(tt.get("peak_device_bytes") or 0) / gib
    if params <= 0 or vram <= 0 or peak <= 0:
        return []

    labels = ["Parameters\n(Qwen3-8B BF16)", "Physical\nGPU VRAM", "TensorTorrent auto\npeak allocated"]
    values = [params, vram, peak]
    colors = [_MUTED, _EAGER, _TT]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    bars = ax.bar(labels, values, color=colors, width=0.62, edgecolor="white")
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(values) * 0.02,
            f"{val:.2f} GiB",
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
        ylabel="GiB",
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


def write_report(evidence: Path, summary: dict[str, Any]) -> None:
    env = summary.get("environment") or {}
    suites = summary.get("suites") or {}
    gpu = (env.get("gpu0") or {}).get("name") or env.get("gpu") or "GPU"
    vram = (env.get("gpu0") or {}).get("total_memory_bytes") or env.get("gpu_vram_bytes")
    vram_gib = f"{float(vram) / (1024**3):.2f} GiB" if vram else "?"
    torch_v = env.get("torch") or "?"

    xf = suites.get("transformer_beyond_vram") or {}
    xf_apps = xf.get("approaches") or {}
    tt_auto = _pick_tt(xf_apps, "tensortorrent_auto", "tensortorrent")
    tt_forced = xf_apps.get("tensortorrent") or {}
    cpu = xf_apps.get("cpu_eager") or {}
    ge = xf_apps.get("gpu_eager") or {}
    acc = xf_apps.get("accelerate") or {}
    extras = tt_auto.get("extras") or {}
    peak_gb = float(tt_auto.get("peak_device_bytes") or 0) / 1e9
    params_gb = float(xf.get("params_bytes") or (xf.get("transformer_spec") or {}).get("param_bytes") or 0) / 1e9

    deep = suites.get("beyond_vram_deepmlp") or {}
    deep_apps = deep.get("approaches") or {}
    deep_tt = _pick_tt(deep_apps, "tensortorrent_auto", "tensortorrent")
    deep_gpu = deep_apps.get("tensortorrent_gpu_stream") or {}
    deep_cpu = deep_apps.get("cpu_eager") or {}
    deep_ge = deep_apps.get("gpu_eager") or {}
    deep_acc = deep_apps.get("accelerate") or {}
    deep_params_gb = float(deep.get("params_bytes") or 0) / 1e9
    deep_ratio = float(deep.get("params_over_vram") or deep.get("vram_multiple") or 0)

    cross_rows = (suites.get("model_size_crossover") or {}).get("results") or []
    cross_lines: list[str] = []
    for row in cross_rows:
        apps = row.get("approaches") or {}
        ttr = apps.get("tensortorrent") or {}
        ge_row = apps.get("gpu_eager") or {}
        strategy = (ttr.get("extras") or {}).get("execution_strategy") or "?"
        ge_s = "fits" if _eager_fits(ge_row) else "OOM"
        tt_s = f"{float(ttr.get('median_ms') or 0):.0f} ms" if ttr.get("ok") else "fail"
        cross_lines.append(f"| {float(row.get('vram_multiple') or 0):.2f}× | {ge_s} | {tt_s} | `{strategy}` |")

    fit_rows = (suites.get("fit") or {}).get("results") or []
    fit_lines: list[str] = []
    for row in fit_rows:
        apps = row.get("approaches") or {}
        eager = apps.get("eager") or {}
        tt = apps.get("tensortorrent") or {}
        name = str(row.get("workload") or row.get("name") or "?")
        peak_mb = float(tt.get("peak_device_bytes") or 0) / 1e6
        fit_lines.append(f"| {name} | {_fmt_ms(eager)} | {_fmt_ms(tt)} | {peak_mb:.0f} MB |")

    cos = extras.get("cosine")
    cos_s = f"{float(cos):.4f}" if cos is not None else "?"
    argmax = f"{extras.get('argmax_match')}/{extras.get('argmax_total')}"
    ge_status = "infeasible (params > VRAM)" if not ge.get("ok") else _fmt_ms(ge)
    acc_ms = _fmt_ms(acc) if acc.get("ok") else ("OOM" if "oom" in str(acc.get("note") or "").lower() else "fail")
    acc_peak = _fmt_gb(acc.get("peak_device_bytes")) if acc.get("ok") else "—"
    strategy = extras.get("execution_strategy") or "?"

    deep_ge_s = "GPU OOM" if not deep_ge.get("ok") else _fmt_ms(deep_ge)
    deep_acc_s = _fmt_ms(deep_acc) if deep_acc.get("ok") else "fail"
    deep_strategy = (deep_tt.get("extras") or {}).get("execution_strategy") or "?"
    deep_devices = (deep_tt.get("extras") or {}).get("devices_used") or []

    body = f"""# Benchmarks

Measured capacity and fit-in-VRAM results for TensorTorrent.

TensorTorrent targets models that approach or exceed accelerator memory. Native
PyTorch is expected to be faster for small models that fit comfortably on one
GPU — planning and runtime add overhead there. Beyond VRAM, TensorTorrent
provides capacity and can be competitive with host-offload runtimes.

Host for these numbers: {gpu} ({vram_gib}) · PyTorch {torch_v}.
Provenance (commit, packages, raw samples): [`raw/`](raw/).

## Qwen3-8B BF16 — fixed-shape logits forward (`seq_len=16`)

Not autoregressive generation. Parameters **{params_gb:.2f} GB** on **{vram_gib}** physical VRAM.

| Approach | Median ms | Peak VRAM | Notes |
| --- | ---: | ---: | --- |
| GPU eager | — | — | {ge_status} |
| CPU eager | {_fmt_ms(cpu)} | 0 | ok |
| **TensorTorrent auto** | **{_fmt_ms(tt_auto)}** | **{peak_gb:.2f} GB** | `{strategy}` · cosine {cos_s} · argmax {argmax} |
| TensorTorrent forced GPU | {_fmt_ms(tt_forced)} | {_fmt_gb(tt_forced.get("peak_device_bytes"))} GB | detailed; not the default UX |
| Accelerate (`device_map=auto`) | {acc_ms} | {acc_peak} GB | tested config only |

![Qwen memory footprint](figures/qwen_memory_footprint.svg)

## DeepMLP — {deep_ratio:.2f}× VRAM ({deep_params_gb:.2f} GB params)

| Approach | Median ms | Peak VRAM | Notes |
| --- | ---: | ---: | --- |
| GPU eager | — | — | {deep_ge_s} |
| CPU eager | {_fmt_ms(deep_cpu)} | {_fmt_gb(deep_cpu.get("peak_device_bytes"))} GB | ok |
| **TensorTorrent auto** | **{_fmt_ms(deep_tt)}** | **{_fmt_gb(deep_tt.get("peak_device_bytes"))} GB** | `{deep_strategy}` · devices `{deep_devices}` |
| TensorTorrent forced GPU stream | {_fmt_ms(deep_gpu)} | {_fmt_gb(deep_gpu.get("peak_device_bytes"))} GB | detailed |
| Accelerate (`device_map=auto`) | {deep_acc_s} | {_fmt_gb(deep_acc.get("peak_device_bytes"))} GB | tested config only |

## Model-size crossover (DeepMLP)

![Crossover latency](figures/crossover_latency.svg)

| Size × VRAM | GPU eager | TensorTorrent | Strategy |
| --- | --- | --- | --- |
{chr(10).join(cross_lines)}

## Fit-in-VRAM

When the model fits, native PyTorch is faster:

![Fit-in-VRAM overhead](figures/fit_overhead.svg)

| Workload | Eager ms | TensorTorrent ms | Peak VRAM |
| --- | ---: | ---: | ---: |
{chr(10).join(fit_lines)}

## Unmeasured here

Autoregressive generation · multi-GPU · other Accelerate configs · ROCm/XPU.

Full tabular dump: [`REPORT.md`](REPORT.md). Methodology: [`docs/product/benchmarks.md`](../../docs/product/benchmarks.md).

## Reproduce

```bash
uv sync --extra dev --extra bench
python -m benchmarks.public --suite all --out benchmarks/results/current
python -m benchmarks.tooling.freeze --src benchmarks/results/current --dst benchmarks/evidence/raw
python -m benchmarks.tooling.render_evidence --evidence benchmarks/evidence
```
"""
    (evidence / "README.md").write_text(body, encoding="utf-8")

    # Detailed tables (provenance pointer only — full env lives in raw/).
    report = render_markdown_tables(summary)
    report_lines = report.splitlines()
    if report_lines and report_lines[0].startswith("# "):
        report_lines[0] = "# Benchmark report"
    if len(report_lines) > 2 and report_lines[2].startswith("commit "):
        old = report_lines[2]
        report_lines[2:3] = [
            "Full environment and measured commit: [`raw/environment.json`](raw/environment.json).",
            "",
            f"_{old}_",
            "",
        ]
    (evidence / "REPORT.md").write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--evidence",
        type=Path,
        default=Path("benchmarks/evidence"),
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
    written += render_qwen_memory(suites, env, fig_dir)
    written += render_fit_overhead(suites, fig_dir)
    write_report(evidence, summary)
    for path in written:
        print(path)
    print(f"wrote {evidence / 'README.md'}")
    print(f"wrote {evidence / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
