"""Suite runners: fit, beyond-VRAM, pressure scaling, hetero markers."""

from __future__ import annotations

import contextlib
import time
from typing import Any

import torch
import torch.nn as nn

import tensortorrent as tt
from benchmarks.harness import (
    TimedRun,
    evidence_class,
    reset_peaks,
    summarize_samples,
    sync,
    timed_callable,
)
from benchmarks.workloads import (
    FIT_WORKLOADS,
    SMOKE_WORKLOADS,
    DeepMLP,
    deep_mlp_for_bytes,
    param_bytes,
)


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().cpu() - b.detach().cpu()).abs().max().item())


def _numerically_ok(a: torch.Tensor, b: torch.Tensor, *, atol: float = 1e-3, rtol: float = 1e-3) -> bool:
    return bool(torch.allclose(a.detach().cpu().float(), b.detach().cpu().float(), atol=atol, rtol=rtol))


def _gpu_eager_oom_probe(width: int, depth: int, batch: int) -> TimedRun:
    """Run GPU eager in a child process so OOM cannot fragment the parent allocator."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    worker = Path(__file__).with_name("_gpu_eager_worker.py")
    payload = json.dumps({"width": width, "depth": depth, "batch": batch})
    proc = subprocess.run(
        [sys.executable, str(worker)],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "gpu eager worker failed")[-200:]
        if "out of memory" in err.lower() or "oom" in err.lower():
            return TimedRun(ok=False, note=f"CUDA OOM (expected): {err[:120]}")
        return TimedRun(ok=False, note=err[:160])
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return TimedRun(ok=False, note="gpu eager worker produced no output")
    data = json.loads(lines[-1])
    if data.get("oom"):
        return TimedRun(ok=False, note=f"CUDA OOM (expected): {data.get('note', '')[:120]}")
    return TimedRun(ok=True, median_ms=float(data.get("median_ms", 0.0)), note=data.get("note", ""))


def _tt_plan_extras(compiled: Any) -> dict[str, Any]:
    plan = compiled.specialized.plan
    sched = compiled.specialized.schedule
    from tensortorrent.ir.graph import OpCode

    return {
        "devices_used": list(plan.devices_used),
        "prefetch_distance": int(plan.prefetch_distance),
        "predicted_latency_s": float(plan.predicted_latency_s),
        "notes": list(plan.notes)[:24],
        "schedule_notes": list(sched.notes)[:12],
        "n_transfer": sum(1 for i in sched.instructions if i.opcode == OpCode.TRANSFER),
        "n_load": sum(1 for i in sched.instructions if i.opcode == OpCode.LOAD),
        "n_evict": sum(1 for i in sched.instructions if i.opcode == OpCode.EVICT),
        "parameter_store": compiled.executor.parameter_store.stats(),
    }


def run_fit_suite(
    *,
    device: str = "cuda",
    iters: int = 30,
    warmup: int = 5,
    smoke: bool = False,
) -> dict[str, Any]:
    """Single-device fit: eager / torch.compile / TensorTorrent."""
    use_cuda = device == "cuda" and torch.cuda.is_available()
    workloads = SMOKE_WORKLOADS if smoke else FIT_WORKLOADS
    rows: list[dict[str, Any]] = []

    for name, (factory, shape) in workloads.items():
        torch.manual_seed(0)
        model = factory().eval()
        x = torch.randn(*shape)
        if use_cuda:
            x_ref = x.cuda()
            m_ref = model.cuda()
        else:
            x_ref = x.cpu()
            m_ref = model.cpu()

        with torch.no_grad():
            expected = m_ref(x_ref).detach().cpu()

        row: dict[str, Any] = {
            "workload": name,
            "params_bytes": param_bytes(model),
            "evidence": evidence_class("MEASURED"),
            "approaches": {},
        }

        # Eager
        reset_peaks()
        try:
            with torch.no_grad():
                samples = timed_callable(lambda m=m_ref, inp=x_ref: m(inp), iters=iters, warmup=warmup)
            err = _max_abs_err(m_ref(x_ref), expected)
            run = summarize_samples(samples, extras={"max_abs_err": err})
            row["approaches"]["eager"] = run
        except Exception as exc:  # noqa: BLE001
            row["approaches"]["eager"] = TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}")

        # torch.compile
        reset_peaks()
        try:
            compiled_pt = torch.compile(m_ref)
            with torch.no_grad():
                # compile happens on first call
                t0 = time.perf_counter()
                for _ in range(max(1, warmup)):
                    compiled_pt(x_ref)
                sync()
                compile_s = time.perf_counter() - t0
                samples = timed_callable(lambda fn=compiled_pt, inp=x_ref: fn(inp), iters=iters, warmup=0)
            err = _max_abs_err(compiled_pt(x_ref), expected)
            run = summarize_samples(samples, extras={"max_abs_err": err})
            run.compile_s = compile_s
            row["approaches"]["torch_compile"] = run
        except Exception as exc:  # noqa: BLE001
            row["approaches"]["torch_compile"] = TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}")

        # TensorTorrent
        reset_peaks()
        try:
            cfg = tt.CompileConfig(
                use_torch_compile=False,
                measure_regions=False,
                allow_gpu=use_cuda,
                allow_cpu=not use_cuda,
                prefer_direct_path=True,
            )
            t0 = time.perf_counter()
            compiled = tt.compile(model.cpu().eval(), example_inputs=(x.cpu(),), config=cfg)
            compile_s = time.perf_counter() - t0
            extras = _tt_plan_extras(compiled)
            with torch.no_grad():
                samples = timed_callable(lambda fn=compiled, inp=x: fn(inp.cpu()), iters=iters, warmup=warmup)
                out = compiled(x.cpu())
            err = _max_abs_err(out, expected)
            run = summarize_samples(samples, extras={"max_abs_err": err, **extras})
            run.compile_s = compile_s
            if err > 1e-3:
                run.ok = False
                run.note = f"numerical mismatch max_abs_err={err}"
            row["approaches"]["tensortorrent"] = run
            compiled.close()
        except Exception as exc:  # noqa: BLE001
            row["approaches"]["tensortorrent"] = TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}")

        rows.append(row)
        del model, m_ref
        reset_peaks()

    return {
        "suite": "fit",
        "device": "cuda" if use_cuda else "cpu",
        "iters": iters,
        "warmup": warmup,
        "results": rows,
    }


def run_beyond_vram_suite(
    *,
    vram_multiple: float = 1.5,
    iters: int = 5,
    warmup: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    """Model larger than VRAM: GPU eager OOM, TT GPU streaming, CPU eager, Accelerate."""
    if not torch.cuda.is_available():
        return {
            "suite": "beyond_vram",
            "evidence": evidence_class("SUPPORTED_BUT_UNMEASURED"),
            "note": "no CUDA device",
            "results": [],
        }

    vram = int(torch.cuda.get_device_properties(0).total_memory)
    multiple = 1.1 if smoke else vram_multiple
    width, depth = deep_mlp_for_bytes(int(vram * multiple), width=2048 if smoke else 4096)
    # Seed once: build reference weights, then inputs, then clone state for each approach.
    torch.manual_seed(0)
    ref = DeepMLP(width, depth).eval()
    state = {k: v.detach().cpu().clone() for k, v in ref.state_dict().items()}
    x = torch.randn(2 if smoke else 8, width)
    pbytes = param_bytes(ref)
    with torch.no_grad():
        expected = ref(x).clone()
    del ref

    approaches: dict[str, Any] = {}

    def _fresh() -> nn.Module:
        m = DeepMLP(width, depth).eval()
        m.load_state_dict(state)
        return m

    # GPU eager — expect OOM (isolated subprocess; must not fragment parent VRAM).
    approaches["gpu_eager"] = _gpu_eager_oom_probe(width, depth, int(x.shape[0]))
    reset_peaks()

    # TensorTorrent before Accelerate so a failed device_map cannot leave the
    # CUDA allocator fragmented for the TT measurement.
    reset_peaks()
    try:
        cfg = tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=True,
            allow_cpu=False,
            ram_budget_bytes=None,
            vram_budget_bytes=vram,
            max_region_nodes=16,
            prefetch_distance=1,
        )
        m = _fresh()
        t0 = time.perf_counter()
        compiled = tt.compile(m, example_inputs=(x.cpu(),), config=cfg)
        compile_s = time.perf_counter() - t0
        del m
        extras = _tt_plan_extras(compiled)
        with torch.no_grad():
            samples = timed_callable(lambda fn=compiled, inp=x: fn(inp.cpu()), iters=iters, warmup=warmup)
            out = compiled(x.cpu())
        err = _max_abs_err(out, expected)
        run = summarize_samples(samples, extras={"max_abs_err": err, **extras})
        run.compile_s = compile_s
        on_cuda = any(str(d).startswith("cuda_") for d in extras["devices_used"])
        if not on_cuda:
            run.ok = False
            run.note = f"expected CUDA placement, got {extras['devices_used']}"
        elif not _numerically_ok(out, expected):
            run.ok = False
            run.note = f"numerical mismatch max_abs_err={err}"
        approaches["tensortorrent"] = run
        with contextlib.suppress(Exception):
            compiled.close()
    except Exception as exc:  # noqa: BLE001
        approaches["tensortorrent"] = TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:200])
    reset_peaks()

    # CPU eager
    reset_peaks()
    try:
        m = _fresh()
        with torch.no_grad():
            samples = timed_callable(lambda model=m, inp=x: model(inp.cpu()), iters=iters, warmup=warmup)
            err = _max_abs_err(m(x.cpu()), expected)
        approaches["cpu_eager"] = summarize_samples(samples, extras={"max_abs_err": err})
        del m
    except Exception as exc:  # noqa: BLE001
        approaches["cpu_eager"] = TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:160])
    reset_peaks()

    # Accelerate last (may OOM / fragment VRAM)
    reset_peaks()
    try:
        from accelerate import dispatch_model, infer_auto_device_map

        m = _fresh()
        t0 = time.perf_counter()
        dmap = infer_auto_device_map(m, no_split_module_classes=["Linear"])
        m = dispatch_model(m, device_map=dmap)
        compile_s = time.perf_counter() - t0
        xd = x.cuda()
        with torch.no_grad():
            samples = timed_callable(lambda model=m, inp=xd: model(inp), iters=iters, warmup=warmup)
        run = summarize_samples(samples)
        run.compile_s = compile_s
        approaches["accelerate"] = run
        del m
    except ImportError:
        approaches["accelerate"] = TimedRun(ok=False, note="accelerate not installed")
    except Exception as exc:  # noqa: BLE001
        approaches["accelerate"] = TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:160])
    reset_peaks()

    return {
        "suite": "beyond_vram",
        "evidence": evidence_class("MEASURED"),
        "vram_bytes": vram,
        "vram_multiple": multiple,
        "width": width,
        "depth": depth,
        "params_bytes": pbytes,
        "params_over_vram": pbytes / vram,
        "approaches": approaches,
    }


def run_memory_pressure_suite(
    *,
    fractions: tuple[float, ...] = (1.0, 0.75, 0.5, 0.35, 0.25),
    iters: int = 5,
    warmup: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    """Same workload under artificial VRAM budgets."""
    if not torch.cuda.is_available():
        return {
            "suite": "memory_pressure",
            "evidence": evidence_class("SUPPORTED_BUT_UNMEASURED"),
            "note": "no CUDA device",
            "results": [],
        }

    vram = int(torch.cuda.get_device_properties(0).total_memory)
    # Model ~45% of physical VRAM so 100% budget fits, tighter budgets stream.
    width, depth = deep_mlp_for_bytes(int(vram * (0.25 if smoke else 0.45)), width=2048 if smoke else 4096)
    fracs = (1.0, 0.5, 0.25) if smoke else fractions
    torch.manual_seed(0)
    model = DeepMLP(width, depth).eval()
    x = torch.randn(2, width)
    pbytes = param_bytes(model)
    with torch.no_grad():
        expected = model(x).clone()

    rows: list[dict[str, Any]] = []
    for frac in fracs:
        budget = max(64 << 20, int(vram * frac))
        reset_peaks()
        row: dict[str, Any] = {
            "budget_fraction": frac,
            "vram_budget_bytes": budget,
            "evidence": evidence_class("MEASURED"),
        }
        try:
            cfg = tt.CompileConfig(
                use_torch_compile=False,
                measure_regions=False,
                allow_gpu=True,
                allow_cpu=False,
                vram_budget_bytes=budget,
                max_region_nodes=8 if smoke else 16,
                prefetch_distance=1,
            )
            t0 = time.perf_counter()
            compiled = tt.compile(model.cpu().eval(), example_inputs=(x.cpu(),), config=cfg)
            compile_s = time.perf_counter() - t0
            extras = _tt_plan_extras(compiled)
            with torch.no_grad():
                samples = timed_callable(lambda fn=compiled, inp=x: fn(inp.cpu()), iters=iters, warmup=warmup)
                out = compiled(x.cpu())
            err = _max_abs_err(out, expected)
            run = summarize_samples(samples, extras={"max_abs_err": err, **extras})
            run.compile_s = compile_s
            if err > 1e-3:
                run.ok = False
                run.note = f"numerical mismatch max_abs_err={err}"
            row["tensortorrent"] = run
            compiled.close()
        except Exception as exc:  # noqa: BLE001
            row["tensortorrent"] = TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:200])
        rows.append(row)
        reset_peaks()

    return {
        "suite": "memory_pressure",
        "evidence": evidence_class("MEASURED"),
        "vram_bytes": vram,
        "params_bytes": pbytes,
        "width": width,
        "depth": depth,
        "results": rows,
    }


def run_model_size_scaling_suite(
    *,
    iters: int = 3,
    warmup: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    """Sweep model sizes around the VRAM boundary."""
    if not torch.cuda.is_available():
        return {
            "suite": "model_size_scaling",
            "evidence": evidence_class("SUPPORTED_BUT_UNMEASURED"),
            "note": "no CUDA device",
            "results": [],
        }

    vram = int(torch.cuda.get_device_properties(0).total_memory)
    multiples = (0.3, 0.9, 1.1) if smoke else (0.25, 0.6, 0.95, 1.15, 1.5)
    rows: list[dict[str, Any]] = []
    for mult in multiples:
        width, depth = deep_mlp_for_bytes(int(vram * mult), width=2048 if smoke else 4096)
        torch.manual_seed(0)
        ref = DeepMLP(width, depth).eval()
        state = {k: v.detach().cpu().clone() for k, v in ref.state_dict().items()}
        x = torch.randn(2, width)
        pbytes = param_bytes(ref)
        with torch.no_grad():
            expected = ref(x).clone()
        del ref
        row: dict[str, Any] = {
            "vram_multiple": mult,
            "params_bytes": pbytes,
            "width": width,
            "depth": depth,
            "evidence": evidence_class("MEASURED"),
            "approaches": {},
        }

        # Eager GPU (may OOM) — isolate in a child process.
        row["approaches"]["gpu_eager"] = _gpu_eager_oom_probe(width, depth, int(x.shape[0]))
        # Probe returns ok=False on OOM; if it unexpectedly fits, re-time in-process.
        if row["approaches"]["gpu_eager"].ok and "unexpected success" in (row["approaches"]["gpu_eager"].note or ""):
            reset_peaks()
            try:
                m = DeepMLP(width, depth).eval()
                m.load_state_dict(state)
                m = m.cuda()
                xd = x.cuda()
                with torch.no_grad():
                    samples = timed_callable(lambda model=m, inp=xd: model(inp), iters=iters, warmup=warmup)
                row["approaches"]["gpu_eager"] = summarize_samples(samples)
                m.cpu()
                del m
            except Exception as exc:  # noqa: BLE001
                row["approaches"]["gpu_eager"] = TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:120])
        reset_peaks()

        # Near/over VRAM: budgeted CUDA Transfer/Evict (no CPU-only fallback).
        reset_peaks()
        near_or_over = pbytes >= int(vram * 0.85)
        try:
            cfg = tt.CompileConfig(
                use_torch_compile=False,
                measure_regions=False,
                allow_gpu=True,
                allow_cpu=not near_or_over,
                vram_budget_bytes=vram if near_or_over else None,
                max_region_nodes=16,
                prefetch_distance=1,
            )
            m = DeepMLP(width, depth).eval()
            m.load_state_dict(state)
            t0 = time.perf_counter()
            compiled = tt.compile(m, example_inputs=(x.cpu(),), config=cfg)
            compile_s = time.perf_counter() - t0
            del m
            extras = _tt_plan_extras(compiled)
            with torch.no_grad():
                samples = timed_callable(lambda fn=compiled, inp=x: fn(inp.cpu()), iters=iters, warmup=warmup)
                out = compiled(x.cpu())
            err = _max_abs_err(out, expected)
            run = summarize_samples(samples, extras={"max_abs_err": err, **extras})
            run.compile_s = compile_s
            if not _numerically_ok(out, expected):
                run.ok = False
                run.note = f"numerical mismatch max_abs_err={err}"
            row["approaches"]["tensortorrent"] = run
            compiled.close()
        except Exception as exc:  # noqa: BLE001
            row["approaches"]["tensortorrent"] = TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:160])
        rows.append(row)
        reset_peaks()

    return {
        "suite": "model_size_scaling",
        "evidence": evidence_class("MEASURED"),
        "vram_bytes": vram,
        "results": rows,
    }


def run_hetero_suite(*, smoke: bool = False) -> dict[str, Any]:
    """GPU+CPU and multi-GPU markers. Never fabricate multi-GPU numbers."""
    n_gpu = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    out: dict[str, Any] = {
        "suite": "heterogeneous",
        "cuda_device_count": n_gpu,
        "results": [],
    }

    if n_gpu < 1:
        out["evidence"] = evidence_class("SUPPORTED_BUT_UNMEASURED")
        out["note"] = "no CUDA device for GPU+CPU measurement"
        return out

    # GPU+CPU: compile with both allowed on a medium model; report placement.
    width, depth = (1024, 8) if smoke else (2048, 16)
    torch.manual_seed(0)
    model = DeepMLP(width, depth).eval()
    x = torch.randn(4, width)
    with torch.no_grad():
        expected = model(x).clone()
    reset_peaks()
    try:
        cfg = tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=True,
            allow_cpu=True,
            max_region_nodes=8,
        )
        t0 = time.perf_counter()
        compiled = tt.compile(model, example_inputs=(x,), config=cfg)
        compile_s = time.perf_counter() - t0
        extras = _tt_plan_extras(compiled)
        with torch.no_grad():
            samples = timed_callable(lambda fn=compiled, inp=x: fn(inp), iters=3 if smoke else 10, warmup=1)
            err = _max_abs_err(compiled(x), expected)
        run = summarize_samples(samples, extras={"max_abs_err": err, **extras})
        run.compile_s = compile_s
        out["results"].append(
            {
                "case": "gpu_plus_cpu_allowed",
                "evidence": evidence_class("MEASURED"),
                "tensortorrent": run,
            }
        )
        compiled.close()
    except Exception as exc:  # noqa: BLE001
        out["results"].append(
            {
                "case": "gpu_plus_cpu_allowed",
                "evidence": evidence_class("MEASURED"),
                "tensortorrent": TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:160]),
            }
        )

    if n_gpu < 2:
        out["results"].append(
            {
                "case": "two_gpu",
                "evidence": evidence_class("SUPPORTED_BUT_UNMEASURED"),
                "note": f"only {n_gpu} CUDA device(s) present; multi-GPU not measured",
            }
        )
    else:
        out["results"].append(
            {
                "case": "two_gpu",
                "evidence": evidence_class("PLANNED"),
                "note": "multi-GPU present but dedicated two-GPU harness not yet in this suite",
            }
        )

    out["evidence"] = evidence_class("MEASURED")
    return out


def try_plot(results_root: Any, payload: dict[str, Any]) -> list[str]:
    """Best-effort matplotlib plots; skip silently if unavailable."""
    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return written

    from pathlib import Path

    root = Path(results_root)
    # Pressure: throughput vs budget fraction
    if payload.get("suite") == "all":
        pressure = payload.get("suites", {}).get("memory_pressure")
        if pressure and pressure.get("results"):
            xs, ys = [], []
            for row in pressure["results"]:
                tt_run = row.get("tensortorrent")
                if not tt_run or not getattr(tt_run, "ok", tt_run.get("ok") if isinstance(tt_run, dict) else False):
                    continue
                med = tt_run.median_ms if hasattr(tt_run, "median_ms") else tt_run.get("median_ms", 0)
                if med <= 0:
                    continue
                xs.append(row["budget_fraction"] * 100)
                ys.append(1000.0 / med)
            if xs:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(xs, ys, marker="o")
                ax.set_xlabel("VRAM budget (% of device)")
                ax.set_ylabel("Throughput (iters/s)")
                ax.set_title("TensorTorrent throughput vs VRAM budget")
                path = root / "throughput_vs_vram_budget.png"
                fig.tight_layout()
                fig.savefig(path)
                plt.close(fig)
                written.append(str(path))

        scaling = payload.get("suites", {}).get("model_size_scaling")
        if scaling and scaling.get("results"):
            xs, eager_y, tt_y = [], [], []
            for row in scaling["results"]:
                xs.append(row["vram_multiple"])
                eg = row["approaches"].get("gpu_eager")
                tt_run = row["approaches"].get("tensortorrent")

                def _tps(run: Any) -> float | None:
                    if run is None:
                        return None
                    ok = run.ok if hasattr(run, "ok") else run.get("ok")
                    med = run.median_ms if hasattr(run, "median_ms") else run.get("median_ms", 0)
                    if not ok or not med:
                        return None
                    return 1000.0 / float(med)

                eager_y.append(_tps(eg))
                tt_y.append(_tps(tt_run))
            if xs:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(xs, [y if y is not None else float("nan") for y in eager_y], marker="o", label="GPU eager")
                ax.plot(xs, [y if y is not None else float("nan") for y in tt_y], marker="s", label="TensorTorrent")
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
