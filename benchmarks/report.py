"""Markdown tables and optional plots from public benchmark summaries."""

from __future__ import annotations

from typing import Any


def _run_get(run: Any, key: str, default: Any = None) -> Any:
    if run is None:
        return default
    if isinstance(run, dict):
        return run.get(key, default)
    return getattr(run, key, default)


def _extras(run: Any) -> dict[str, Any]:
    ex = _run_get(run, "extras", {}) or {}
    if not isinstance(ex, dict):
        return {}
    inner = ex.get("extras")
    if isinstance(inner, dict):
        return inner
    return ex


def _instrumentation(run: Any) -> dict[str, Any]:
    inst = _extras(run).get("instrumentation")
    return inst if isinstance(inst, dict) else {}


def _format_status(run: Any) -> str:
    if run is None:
        return "NOT MEASURED"
    if _run_get(run, "ok"):
        return "ok"
    note = str(_run_get(run, "note", "") or "").strip()
    if not note:
        return "FAIL"
    low = note.lower()
    if "oom" in low or "out of memory" in low or "cuda oom" in low:
        return "OOM"
    if "infeasible by parameter footprint" in low:
        return "INFEASIBLE"
    if "not installed" in low or "unsupported" in low:
        return "UNSUPPORTED"
    if "no cuda" in low or "skipped" in low:
        return "NOT MEASURED"
    return note[:120]


def _format_ms(run: Any) -> str:
    if run is None:
        return "NOT MEASURED"
    if not _run_get(run, "ok"):
        return _format_status(run)
    med = float(_run_get(run, "median_ms", 0.0) or 0.0)
    note = str(_run_get(run, "note", "") or "")
    if med <= 0 and "fit" in note.lower():
        return "fits"
    if med <= 0:
        return "ok"
    return f"{med:.2f}"


def _format_gb(bytes_val: int | float | None) -> str:
    if bytes_val is None:
        return "NOT MEASURED"
    return f"{float(bytes_val) / 1e9:.2f}"


def _throughput_str(run: Any) -> str:
    if run is None or not _run_get(run, "ok"):
        return _format_status(run)
    med = float(_run_get(run, "median_ms", 0.0) or 0.0)
    if med <= 0:
        return "NOT MEASURED"
    return f"{1000.0 / med:.2f}"


def _gpu_fraction_str(run: Any) -> str:
    inst = _instrumentation(run)
    frac = inst.get("region_compute_fraction") or {}
    if not isinstance(frac, dict) or "gpu" not in frac:
        return "NOT MEASURED"
    return f"{100.0 * float(frac['gpu']):.1f}%"


def _transfer_gb_str(run: Any) -> str:
    inst = _instrumentation(run)
    h2d = int(inst.get("transfer_bytes_h2d") or 0)
    d2h = int(inst.get("transfer_bytes_d2h") or 0)
    total = h2d + d2h
    if total <= 0:
        return "NOT MEASURED"
    return f"{total / 1e9:.3f}"


def _section(title: str, body: str) -> str:
    if not body.strip():
        return ""
    return f"## {title}\n\n{body.strip()}\n"


def _fit_table(suites: dict[str, Any]) -> str:
    fit = suites.get("fit")
    if not fit or not fit.get("results"):
        return ""
    lines = [
        "| Workload | Eager ms | torch.compile ms | TT ms | rel | Peak VRAM MB | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in fit["results"]:
        apps = row.get("approaches") or {}
        eager = apps.get("eager")
        tc = apps.get("torch_compile")
        tt = apps.get("tensortorrent")
        rel = "NOT MEASURED"
        if _run_get(eager, "ok") and _run_get(tt, "ok"):
            em = float(_run_get(eager, "median_ms", 0.0) or 0.0)
            tm = float(_run_get(tt, "median_ms", 0.0) or 0.0)
            if em > 0 and tm > 0:
                rel = f"{tm / em:.2f}×"
        peak = "NOT MEASURED"
        if _run_get(tt, "ok"):
            peak = f"{int(_run_get(tt, 'peak_device_bytes', 0) or 0) / 1e6:.1f}"
        status = _format_status(tt)
        lines.append(
            f"| {row.get('workload', '?')} | {_format_ms(eager)} | {_format_ms(tc)} | "
            f"{_format_ms(tt)} | {rel} | {peak} | {status} |"
        )
    return "\n".join(lines)


def _baseline_table(payload: dict[str, Any] | None, *, title_note: str = "") -> str:
    if not payload:
        return ""
    approaches = payload.get("approaches") or {}
    if not approaches:
        return ""
    note = title_note or payload.get("suite", "baseline")
    lines = [
        f"*{note}* — params {_format_gb(payload.get('params_bytes'))} GB "
        f"({float(payload.get('params_over_vram', 0) or 0):.2f}× VRAM when applicable)",
        "",
        "| Approach | Median ms | Peak VRAM GB | Peak host GB | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, run in approaches.items():
        if _run_get(run, "ok"):
            lines.append(
                f"| {name} | {_format_ms(run)} | "
                f"{_format_gb(_run_get(run, 'peak_device_bytes'))} | "
                f"{_format_gb(_run_get(run, 'peak_host_bytes'))} | ok |"
            )
        else:
            lines.append(
                f"| {name} | {_format_status(run)} | {_format_status(run)} | "
                f"{_format_status(run)} | {_format_status(run)} |"
            )
    return "\n".join(lines)


def _budget_table(payload: dict[str, Any] | None) -> str:
    if not payload or not payload.get("results"):
        return ""
    lines = [
        "| Budget GiB | Median ms | Throughput iters/s | Transfer GB | GPU compute % | Status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["results"]:
        budget_gib = row.get("vram_budget_gib")
        if budget_gib is None:
            budget_gib = float(row.get("vram_budget_bytes", 0) or 0) / (1024**3)
        tt = row.get("tensortorrent")
        lines.append(
            f"| {budget_gib:.1f} | {_format_ms(tt)} | {_throughput_str(tt)} | "
            f"{_transfer_gb_str(tt)} | {_gpu_fraction_str(tt)} | {_format_status(tt)} |"
        )
    return "\n".join(lines)


def _crossover_table(payload: dict[str, Any] | None) -> str:
    if not payload or not payload.get("results"):
        return ""
    lines = [
        "| Size × VRAM | GPU eager ms | TensorTorrent ms | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in payload["results"]:
        mult = float(row.get("vram_multiple", 0) or 0)
        apps = row.get("approaches") or {}
        eg = apps.get("gpu_eager")
        tt = apps.get("tensortorrent")
        lines.append(
            f"| {mult:.2f} | {_format_ms(eg)} | {_format_ms(tt)} | TT:{_format_status(tt)} eager:{_format_status(eg)} |"
        )
    return "\n".join(lines)


def render_markdown_tables(summary: dict[str, Any]) -> str:
    """Render MEASURED markdown tables from a ``summary.json`` payload."""
    suites = summary.get("suites") or {}
    env = summary.get("environment") or {}
    parts: list[str] = [
        "# TensorTorrent benchmark report",
        "",
        f"commit `{str(env.get('commit', '?'))[:12]}` · torch {env.get('torch', '?')} · "
        f"CUDA available={env.get('cuda_available')} · smoke={summary.get('smoke')}",
    ]
    if env.get("cuda_driver_version"):
        parts[-1] += f" · driver {env['cuda_driver_version']}"

    fit_md = _fit_table(suites)
    if fit_md:
        parts.append(_section("Fit-in-VRAM workloads", fit_md))

    deep = suites.get("beyond_vram_deepmlp") or suites.get("beyond_vram")
    deep_md = _baseline_table(deep, title_note="DeepMLP beyond VRAM")
    if deep_md:
        parts.append(_section("Beyond VRAM — DeepMLP baselines", deep_md))

    xf = suites.get("transformer_beyond_vram")
    xf_md = _baseline_table(xf, title_note="HF transformer beyond VRAM")
    if xf_md:
        parts.append(_section("Beyond VRAM — transformer baselines", xf_md))

    budget = suites.get("memory_budget_curve") or suites.get("memory_pressure")
    budget_md = _budget_table(budget)
    if budget_md:
        parts.append(_section("Memory budget curve", budget_md))

    cross = suites.get("model_size_crossover") or suites.get("model_size_scaling")
    cross_md = _crossover_table(cross)
    if cross_md:
        parts.append(_section("Model size crossover", cross_md))

    hetero = suites.get("heterogeneous")
    if hetero and hetero.get("results"):
        lines = [
            "| Case | Evidence | Notes |",
            "| --- | --- | --- |",
        ]
        for row in hetero["results"]:
            case = row.get("case", "?")
            ev = row.get("evidence", "?")
            note = row.get("note") or ""
            tt = row.get("tensortorrent")
            if tt is not None:
                devices = (_extras(tt).get("devices_used") if _run_get(tt, "ok") else None) or []
                status = _format_status(tt)
                note = note or f"TT={status}; devices={devices}"
            lines.append(f"| {case} | {ev} | {note} |")
        parts.append(_section("Heterogeneous placement", "\n".join(lines)))

    return "\n".join(parts).strip() + "\n"


def _budget_points(suites: dict[str, Any]) -> list[dict[str, Any]]:
    payload = suites.get("memory_budget_curve") or suites.get("memory_pressure")
    if not payload:
        return []
    points: list[dict[str, Any]] = []
    for row in payload.get("results") or []:
        tt = row.get("tensortorrent")
        if tt is None:
            continue
        budget_gib = row.get("vram_budget_gib")
        if budget_gib is None:
            if "budget_fraction" in row:
                vram = float(payload.get("vram_bytes") or 0)
                budget_gib = (float(row["budget_fraction"]) * vram) / (1024**3) if vram else 0.0
            else:
                budget_gib = float(row.get("vram_budget_bytes") or 0) / (1024**3)
        med = float(_run_get(tt, "median_ms", 0.0) or 0.0)
        inst = _instrumentation(tt)
        transfer_gb = (int(inst.get("transfer_bytes_h2d") or 0) + int(inst.get("transfer_bytes_d2h") or 0)) / 1e9
        gpu_frac = float((inst.get("region_compute_fraction") or {}).get("gpu") or 0.0)
        points.append(
            {
                "budget_gib": float(budget_gib),
                "ok": bool(_run_get(tt, "ok")),
                "median_ms": med,
                "throughput": (1000.0 / med) if med > 0 and _run_get(tt, "ok") else None,
                "transfer_gb": transfer_gb if transfer_gb > 0 else None,
                "peak_vram_gb": float(_run_get(tt, "peak_device_bytes", 0) or 0) / 1e9,
                "gpu_frac": gpu_frac if gpu_frac > 0 else None,
            }
        )
    return sorted(points, key=lambda p: p["budget_gib"])


def _crossover_points(suites: dict[str, Any]) -> list[dict[str, Any]]:
    payload = suites.get("model_size_crossover") or suites.get("model_size_scaling")
    if not payload:
        return []
    points: list[dict[str, Any]] = []
    for row in payload.get("results") or []:
        mult = float(row.get("vram_multiple", 0) or 0)
        apps = row.get("approaches") or {}
        for label in ("gpu_eager", "tensortorrent"):
            run = apps.get(label)
            med = float(_run_get(run, "median_ms", 0.0) or 0.0)
            points.append(
                {
                    "multiple": mult,
                    "approach": label,
                    "ok": bool(_run_get(run, "ok")),
                    "throughput": (1000.0 / med) if med > 0 and _run_get(run, "ok") else None,
                }
            )
    return sorted(points, key=lambda p: (p["multiple"], p["approach"]))


def try_plot_all(out_dir: Any, summary: dict[str, Any]) -> list[str]:
    """Best-effort matplotlib plots; returns written file paths."""
    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return written

    from pathlib import Path

    root = Path(out_dir)
    suites = summary.get("suites") or {}
    budget_pts = [p for p in _budget_points(suites) if p["ok"]]
    if budget_pts:
        xs = [p["budget_gib"] for p in budget_pts]

        fig, ax = plt.subplots(figsize=(6, 4))
        ys = [p["throughput"] for p in budget_pts]
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("VRAM budget (GiB)")
        ax.set_ylabel("Throughput (iters/s)")
        ax.set_title("Throughput vs memory budget")
        path = root / "throughput_vs_budget.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

        fig, ax = plt.subplots(figsize=(6, 4))
        ys = [p["median_ms"] for p in budget_pts]
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("VRAM budget (GiB)")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency vs memory budget")
        path = root / "latency_vs_budget.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

        xfer = [p["transfer_gb"] for p in budget_pts]
        if any(v is not None for v in xfer):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xs, [v if v is not None else float("nan") for v in xfer], marker="o")
            ax.set_xlabel("VRAM budget (GiB)")
            ax.set_ylabel("Transfer volume (GB)")
            ax.set_title("Transfer vs memory budget")
            path = root / "transfer_vs_budget.png"
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
            written.append(str(path))

        peaks = [p["peak_vram_gb"] for p in budget_pts]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, peaks, marker="o")
        ax.set_xlabel("VRAM budget (GiB)")
        ax.set_ylabel("Peak VRAM (GB)")
        ax.set_title("Peak VRAM vs memory budget")
        path = root / "peak_vram_vs_budget.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    cross_pts = _crossover_points(suites)
    if cross_pts:
        multiples = sorted({p["multiple"] for p in cross_pts})
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, marker in (("gpu_eager", "o"), ("tensortorrent", "s")):
            ys = []
            for mult in multiples:
                match = [p for p in cross_pts if p["multiple"] == mult and p["approach"] == label]
                ys.append(match[0]["throughput"] if match and match[0]["throughput"] is not None else float("nan"))
            ax.plot(multiples, ys, marker=marker, label=label)
        ax.set_xlabel("Model size (× VRAM)")
        ax.set_ylabel("Throughput (iters/s)")
        ax.set_title("Throughput vs model size")
        ax.legend()
        path = root / "throughput_vs_model_size.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    return written
