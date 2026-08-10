"""Hard GPU validation: forced-GPU vs auto vs baselines, no CPU cheat.

Labels every approach explicitly:
  forced_gpu | auto | cpu_eager | torch_compile | accelerate

Never special-cases workload/model names in compile config. Forced-GPU rows
assert every region binding targets CUDA.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

import torch

import tensortorrent as tt
from benchmarks.suites.memory_hygiene import (
    abort_if_host_tight,
    deepmlp_weight_file,
    load_deepmlp,
)
from benchmarks.suites.runners import _max_abs_err, _numerically_ok, _tt_plan_extras
from benchmarks.suites.workloads import deep_mlp_for_bytes
from benchmarks.tooling.harness import (
    TimedRun,
    evidence_class,
    release_host_memory,
    reset_peaks,
    summarize_samples,
    timed_callable,
)
from benchmarks.tooling.instrumentation import summarize_execution

FIT_FRACTIONS = (0.25, 0.50, 0.70, 0.90)
BEYOND_MULTIPLES = (1.05, 1.25, 1.50)
SMOKE_FIT_FRACTIONS = (0.25, 0.50)
SMOKE_BEYOND_MULTIPLES = (1.05,)


def _assert_forced_gpu_cuda(extras: dict[str, Any], run: TimedRun) -> TimedRun:
    """Fail the row if any selected region left CUDA."""
    devices = [str(d) for d in (extras.get("devices_used") or [])]
    on_cuda = bool(devices) and all(d.startswith("cuda_") for d in devices)
    if not on_cuda:
        run.ok = False
        run.note = f"forced GPU left CUDA: devices={devices}"
        return run
    # Region bindings must also be accelerator backends (no hidden CPU regions).
    region_kinds = (extras.get("instrumentation") or {}).get("regions_by_kind") or {}
    cpu_regions = int(region_kinds.get("cpu") or 0)
    if cpu_regions > 0:
        run.ok = False
        run.note = f"forced GPU hid {cpu_regions} CPU region(s): {region_kinds}"
    return run


def _compile_and_time(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    config: tt.CompileConfig,
    label: str,
    expected: torch.Tensor | None,
    iters: int,
    warmup: int,
    instrument: bool,
    require_cuda: bool,
) -> TimedRun:
    # Avoid CUDA context init before export-free CPU compile: synchronize/empty_cache
    # in reset_peaks permanently slows multi-GiB host GEMM in-process.
    if bool(getattr(config, "allow_gpu", False)) or bool(getattr(config, "allow_integrated_gpu", False)):
        if require_cuda or not bool(getattr(config, "allow_cpu", True)):
            reset_peaks()
        else:
            # Auto may pick export-free CPU — only collect host GC before compile.
            import gc

            gc.collect()
    else:
        import gc

        gc.collect()
    try:
        t0 = time.perf_counter()
        compiled = tt.compile(model, example_inputs=(x.cpu(),), config=config)
        compile_s = time.perf_counter() - t0
        extras = _tt_plan_extras(compiled)
        extras["mode_label"] = label
        on_cuda = any(str(d).startswith("cuda_") for d in extras["devices_used"])
        # Export-free gate times the eager module before returning; that warms the
        # package. Cool briefly so steady-state samples are not thermally skewed.
        if extras.get("export_free") and not on_cuda:
            time.sleep(6.0)
        with torch.no_grad():
            samples = timed_callable(
                lambda fn=compiled, inp=x: fn(inp.cpu()),
                iters=iters,
                warmup=warmup,
                synchronize=on_cuda,
            )
            out = compiled(x.cpu())
        if instrument:
            extras["instrumentation"] = summarize_execution(compiled)
        err = _max_abs_err(out, expected) if expected is not None else 0.0
        run = summarize_samples(
            samples,
            extras={
                "max_abs_err": err,
                "region_count": len(getattr(compiled._program, "regions", ()) or ()),  # noqa: SLF001
                **extras,
            },
        )
        run.compile_s = compile_s
        if expected is not None and not _numerically_ok(out, expected):
            run.ok = False
            run.note = f"numerical mismatch max_abs_err={err}"
        if require_cuda:
            run = _assert_forced_gpu_cuda(extras, run)
        with contextlib.suppress(Exception):
            compiled.close()
        return run
    except Exception as exc:  # noqa: BLE001
        return TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:220], extras={"mode_label": label})


def _gpu_eager_time(model: torch.nn.Module, x: torch.Tensor, *, iters: int, warmup: int) -> TimedRun:
    reset_peaks()
    try:
        m = model.cuda().eval()
        xd = x.cuda()
        with torch.no_grad():
            samples = timed_callable(lambda: m(xd), iters=iters, warmup=warmup)
        return summarize_samples(samples, extras={"mode_label": "gpu_eager"})
    except torch.cuda.OutOfMemoryError as exc:
        return TimedRun(ok=False, note=f"CUDA OOM: {exc}"[:160], extras={"mode_label": "gpu_eager"})
    except Exception as exc:  # noqa: BLE001
        return TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:160], extras={"mode_label": "gpu_eager"})
    finally:
        with contextlib.suppress(Exception):
            model.cpu()
        release_host_memory()


def _torch_compile_time(model: torch.nn.Module, x: torch.Tensor, *, iters: int, warmup: int) -> TimedRun:
    reset_peaks()
    try:
        m = model.cuda().eval()
        xd = x.cuda()
        compiled_pt = torch.compile(m)
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(max(1, warmup)):
                compiled_pt(xd)
            torch.cuda.synchronize()
            compile_s = time.perf_counter() - t0
            samples = timed_callable(lambda: compiled_pt(xd), iters=iters, warmup=0)
        run = summarize_samples(samples, extras={"mode_label": "torch_compile"})
        run.compile_s = compile_s
        return run
    except torch.cuda.OutOfMemoryError as exc:
        return TimedRun(ok=False, note=f"CUDA OOM: {exc}"[:160], extras={"mode_label": "torch_compile"})
    except Exception as exc:  # noqa: BLE001
        return TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:160], extras={"mode_label": "torch_compile"})
    finally:
        with contextlib.suppress(Exception):
            model.cpu()
        release_host_memory()


def _cpu_eager_time(model: torch.nn.Module, x: torch.Tensor, *, iters: int, warmup: int) -> TimedRun:
    reset_peaks()
    try:
        m = model.cpu().eval()
        with torch.no_grad():
            samples = timed_callable(
                lambda: m(x.cpu()),
                iters=iters,
                warmup=warmup,
                synchronize=False,
            )
        return summarize_samples(samples, extras={"mode_label": "cpu_eager"})
    except Exception as exc:  # noqa: BLE001
        return TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:160], extras={"mode_label": "cpu_eager"})


def run_forced_gpu_fit_suite(
    *,
    iters: int = 5,
    warmup: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    """Forced GPU fit: ~25/50/70/90% VRAM vs eager CUDA / torch.compile / TT."""
    if not torch.cuda.is_available():
        return {
            "suite": "hard_forced_gpu_fit",
            "evidence": evidence_class("SUPPORTED_BUT_UNMEASURED"),
            "note": "no CUDA device",
            "results": [],
        }

    vram = int(torch.cuda.get_device_properties(0).total_memory)
    fractions = SMOKE_FIT_FRACTIONS if smoke else FIT_FRACTIONS
    width = 2048 if smoke else 4096
    batch = 2 if smoke else 8
    rows: list[dict[str, Any]] = []

    for frac in fractions:
        target = int(vram * frac)
        w, depth = deep_mlp_for_bytes(target, width=width)
        torch.manual_seed(0)
        with deepmlp_weight_file(w, depth) as (wpath, pbytes):
            tight = abort_if_host_tight(pbytes, label=f"forced_gpu_fit_{frac}")
            if tight is not None:
                rows.append({"vram_fraction": frac, "params_bytes": pbytes, "approaches": {"skip": tight}})
                continue
            x = torch.randn(batch, w)
            ref = load_deepmlp(wpath, w, depth)
            with torch.no_grad():
                expected = ref(x).clone()
            del ref
            release_host_memory()

            approaches: dict[str, Any] = {}
            m = load_deepmlp(wpath, w, depth)
            approaches["gpu_eager"] = _gpu_eager_time(m, x, iters=iters, warmup=warmup)
            del m
            release_host_memory()

            m = load_deepmlp(wpath, w, depth)
            approaches["torch_compile"] = _torch_compile_time(m, x, iters=iters, warmup=warmup)
            del m
            release_host_memory()

            m = load_deepmlp(wpath, w, depth)
            approaches["forced_gpu"] = _compile_and_time(
                m,
                x,
                config=tt.CompileConfig(
                    use_torch_compile=False,
                    measure_regions=False,
                    allow_gpu=True,
                    allow_cpu=False,
                    vram_budget_bytes=vram,
                    max_region_nodes=16,
                    prefer_direct_path=True,
                ),
                label="forced_gpu",
                expected=expected,
                iters=iters,
                warmup=warmup,
                instrument=True,
                require_cuda=True,
            )
            del m
            release_host_memory()
            if torch.cuda.is_available():
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()

            rows.append(
                {
                    "vram_fraction": frac,
                    "width": w,
                    "depth": depth,
                    "params_bytes": pbytes,
                    "params_over_vram": pbytes / vram,
                    "approaches": approaches,
                }
            )

    return {
        "suite": "hard_forced_gpu_fit",
        "evidence": evidence_class("MEASURED"),
        "vram_bytes": vram,
        "fractions": list(fractions),
        "results": rows,
    }


def run_forced_gpu_beyond_suite(
    *,
    iters: int = 3,
    warmup: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    """Forced GPU beyond VRAM: must stream and complete with CPU disabled.

    Multiples ≥ 1.25× run in isolated subprocesses so earlier host-RAM retention
    cannot cause false skips. Approach order inside each child is cpu_eager →
    auto → forced_gpu so auto CPU is not poisoned by a prior GPU export.
    """
    if not torch.cuda.is_available():
        return {
            "suite": "hard_forced_gpu_beyond",
            "evidence": evidence_class("SUPPORTED_BUT_UNMEASURED"),
            "note": "no CUDA device",
            "results": [],
        }

    from benchmarks.suites.memory_hygiene import run_json_worker

    vram = int(torch.cuda.get_device_properties(0).total_memory)
    multiples = SMOKE_BEYOND_MULTIPLES if smoke else BEYOND_MULTIPLES
    width = 2048 if smoke else 4096
    batch = 2 if smoke else 8
    rows: list[dict[str, Any]] = []
    # Isolate every beyond multiple. Parent-held RSS from an in-process 1.05×
    # previously inflated later CPU timings and confused auto vs forced_gpu.
    isolate_multiples = {1.05, 1.25, 1.5}

    for mult in multiples:
        w, depth = deep_mlp_for_bytes(int(vram * mult), width=width)
        with deepmlp_weight_file(w, depth) as (wpath, pbytes):
            use_subprocess = float(mult) in isolate_multiples
            if use_subprocess:
                # Parent-side gate first: child abort alone is too late if the
                # fit suite already ate most of RAM/swap.
                tight = abort_if_host_tight(pbytes, label=f"forced_gpu_beyond_{mult}")
                if tight is not None:
                    rows.append(
                        {
                            "vram_multiple": mult,
                            "width": w,
                            "depth": depth,
                            "params_bytes": pbytes,
                            "params_over_vram": pbytes / vram,
                            "isolated_subprocess": True,
                            "approaches": {"skip": tight},
                        }
                    )
                    release_host_memory()
                    continue
                release_host_memory()
                if torch.cuda.is_available():
                    with contextlib.suppress(Exception):
                        torch.cuda.empty_cache()
                print(f"  beyond {mult:.2f}× → child process", flush=True)
                code, data, err = run_json_worker(
                    "benchmarks.suites._beyond_vram_worker",
                    {
                        "width": w,
                        "depth": depth,
                        "vram_multiple": mult,
                        "vram_bytes": vram,
                        "weight_path": wpath,
                        "params_bytes": pbytes,
                        "batch": batch,
                        "iters": iters,
                        "warmup": warmup,
                        "approaches": ["cpu_eager", "auto", "forced_gpu"],
                    },
                    timeout_s=1800,
                )
                if data is None:
                    rows.append(
                        {
                            "vram_multiple": mult,
                            "width": w,
                            "depth": depth,
                            "params_bytes": pbytes,
                            "params_over_vram": pbytes / vram,
                            "isolated_subprocess": True,
                            "approaches": {
                                "forced_gpu": TimedRun(
                                    ok=False,
                                    note=(err or f"beyond worker failed rc={code}")[-200:],
                                    extras={"mode_label": "forced_gpu"},
                                )
                            },
                        }
                    )
                else:
                    rows.append(data)
                release_host_memory()
                if torch.cuda.is_available():
                    with contextlib.suppress(Exception):
                        torch.cuda.empty_cache()
                continue

            tight = abort_if_host_tight(pbytes, label=f"forced_gpu_beyond_{mult}")
            if tight is not None:
                rows.append({"vram_multiple": mult, "params_bytes": pbytes, "approaches": {"skip": tight}})
                continue
            x = torch.randn(batch, w)
            ref = load_deepmlp(wpath, w, depth)
            with torch.no_grad():
                expected = ref(x).clone()
            del ref
            release_host_memory()

            approaches: dict[str, Any] = {}
            # cpu_eager first so auto/export-free path is measured against a clean host.
            m = load_deepmlp(wpath, w, depth)
            approaches["cpu_eager"] = _cpu_eager_time(m, x, iters=iters, warmup=warmup)
            del m
            release_host_memory()

            m = load_deepmlp(wpath, w, depth)
            approaches["auto"] = _compile_and_time(
                m,
                x,
                config=tt.CompileConfig(
                    use_torch_compile=False,
                    measure_regions=False,
                    allow_gpu=True,
                    allow_cpu=True,
                    vram_budget_bytes=vram,
                    max_region_nodes=16,
                    prefetch_distance=1,
                ),
                label="auto",
                expected=expected,
                iters=iters,
                warmup=warmup,
                instrument=True,
                require_cuda=False,
            )
            del m
            release_host_memory()

            m = load_deepmlp(wpath, w, depth)
            approaches["forced_gpu"] = _compile_and_time(
                m,
                x,
                config=tt.CompileConfig(
                    use_torch_compile=False,
                    measure_regions=False,
                    allow_gpu=True,
                    allow_cpu=False,
                    vram_budget_bytes=vram,
                    max_region_nodes=16,
                    prefetch_distance=1,
                ),
                label="forced_gpu",
                expected=expected,
                iters=iters,
                warmup=warmup,
                instrument=True,
                require_cuda=True,
            )
            del m
            release_host_memory()
            if torch.cuda.is_available():
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()

            rows.append(
                {
                    "vram_multiple": mult,
                    "width": w,
                    "depth": depth,
                    "params_bytes": pbytes,
                    "params_over_vram": pbytes / vram,
                    "approaches": approaches,
                }
            )

    return {
        "suite": "hard_forced_gpu_beyond",
        "evidence": evidence_class("MEASURED"),
        "vram_bytes": vram,
        "multiples": list(multiples),
        "results": rows,
    }


def run_auto_bakeoff_suite(
    *,
    iters: int = 3,
    warmup: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    """Auto vs forced-GPU vs CPU eager vs Accelerate on fit + beyond sizes."""
    if not torch.cuda.is_available():
        return {
            "suite": "hard_auto_bakeoff",
            "evidence": evidence_class("SUPPORTED_BUT_UNMEASURED"),
            "note": "no CUDA device",
            "results": [],
        }

    vram = int(torch.cuda.get_device_properties(0).total_memory)
    points = (
        [("fit", 0.50), ("beyond", 1.10)]
        if smoke
        else [("fit", 0.50), ("fit", 0.90), ("beyond", 1.05), ("beyond", 1.50)]
    )
    width = 2048 if smoke else 4096
    batch = 2 if smoke else 8
    rows: list[dict[str, Any]] = []

    for kind, scale in points:
        target = int(vram * scale)
        w, depth = deep_mlp_for_bytes(target, width=width)
        with deepmlp_weight_file(w, depth) as (wpath, pbytes):
            tight = abort_if_host_tight(pbytes, label=f"auto_{kind}_{scale}")
            if tight is not None:
                rows.append({"kind": kind, "scale": scale, "params_bytes": pbytes, "approaches": {"skip": tight}})
                continue
            x = torch.randn(batch, w)
            ref = load_deepmlp(wpath, w, depth)
            with torch.no_grad():
                expected = ref(x).clone()
            del ref
            release_host_memory()

            approaches: dict[str, Any] = {}
            # cpu_eager first — auto must stay near this, not a post-GPU-poisoned host.
            m = load_deepmlp(wpath, w, depth)
            approaches["cpu_eager"] = _cpu_eager_time(m, x, iters=iters, warmup=warmup)
            del m
            release_host_memory()

            m = load_deepmlp(wpath, w, depth)
            approaches["auto"] = _compile_and_time(
                m,
                x,
                config=tt.CompileConfig(
                    use_torch_compile=False,
                    measure_regions=False,
                    allow_gpu=True,
                    allow_cpu=True,
                    vram_budget_bytes=vram,
                    max_region_nodes=16,
                    prefetch_distance=1,
                ),
                label="auto",
                expected=expected,
                iters=iters,
                warmup=warmup,
                instrument=True,
                require_cuda=False,
            )
            del m
            release_host_memory()

            m = load_deepmlp(wpath, w, depth)
            approaches["forced_gpu"] = _compile_and_time(
                m,
                x,
                config=tt.CompileConfig(
                    use_torch_compile=False,
                    measure_regions=False,
                    allow_gpu=True,
                    allow_cpu=False,
                    vram_budget_bytes=vram,
                    max_region_nodes=16,
                    prefetch_distance=1,
                ),
                label="forced_gpu",
                expected=expected,
                iters=iters,
                warmup=warmup,
                instrument=True,
                require_cuda=True,
            )
            del m
            release_host_memory()

            # Accelerate offload (optional).
            accel_cfg = {
                "device_map": "auto",
                "max_memory": {0: f"{max(1, int(vram * 0.7 / (1024**3)))}GiB", "cpu": "48GiB"},
                "no_split_module_classes": ["Linear"],
            }
            try:
                from accelerate import dispatch_model, infer_auto_device_map

                m = load_deepmlp(wpath, w, depth)
                t0 = time.perf_counter()
                dmap = infer_auto_device_map(
                    m,
                    max_memory=accel_cfg["max_memory"],
                    no_split_module_classes=list(accel_cfg["no_split_module_classes"]),
                )
                m = dispatch_model(m, device_map=dmap)
                compile_s = time.perf_counter() - t0
                first = next(iter(dmap.values())) if dmap else "cpu"
                xd = x.cuda() if isinstance(first, str) and str(first).startswith("cuda") else x.cpu()
                with torch.no_grad():
                    samples = timed_callable(lambda model=m, inp=xd: model(inp), iters=iters, warmup=warmup)
                run = summarize_samples(
                    samples,
                    extras={"mode_label": "accelerate", "accelerate_config": dict(accel_cfg), "device_map": dict(dmap)},
                )
                run.compile_s = compile_s
                approaches["accelerate"] = run
                del m
            except ImportError:
                approaches["accelerate"] = TimedRun(
                    ok=False, note="accelerate not installed", extras={"mode_label": "accelerate"}
                )
            except Exception as exc:  # noqa: BLE001
                approaches["accelerate"] = TimedRun(
                    ok=False,
                    note=f"{type(exc).__name__}: {exc}"[:200],
                    extras={"mode_label": "accelerate", "accelerate_config": dict(accel_cfg)},
                )
            release_host_memory()

            auto_run = approaches["auto"]
            forced_run = approaches["forced_gpu"]
            cpu_run = approaches["cpu_eager"]
            selection_check = _verify_auto_selection(auto_run, forced_run, cpu_run)

            rows.append(
                {
                    "kind": kind,
                    "scale": scale,
                    "width": w,
                    "depth": depth,
                    "params_bytes": pbytes,
                    "params_over_vram": pbytes / vram,
                    "approaches": approaches,
                    "auto_selection_check": selection_check,
                }
            )

    return {
        "suite": "hard_auto_bakeoff",
        "evidence": evidence_class("MEASURED"),
        "vram_bytes": vram,
        "results": rows,
    }


def _median_ms(run: Any) -> float | None:
    if run is None:
        return None
    if isinstance(run, TimedRun):
        return float(run.median_ms) if run.ok else None
    if isinstance(run, dict):
        return float(run["median_ms"]) if run.get("ok") else None
    return None


def _devices(run: Any) -> list[str]:
    extras = run.extras if isinstance(run, TimedRun) else (run.get("extras") or {})
    return [str(d) for d in (extras.get("devices_used") or [])]


def _verify_auto_selection(auto_run: Any, forced_run: Any, cpu_run: Any) -> dict[str, Any]:
    """Check auto picked CPU only when measured CPU is genuinely faster."""
    auto_ms = _median_ms(auto_run)
    forced_ms = _median_ms(forced_run)
    cpu_ms = _median_ms(cpu_run)
    devices = _devices(auto_run)
    cpu_only = bool(devices) and all(d.startswith("cpu") for d in devices)
    ok = True
    note = "ok"
    if auto_ms is None:
        return {"ok": False, "note": "auto failed", "cpu_only": cpu_only}
    if cpu_only:
        # Auto chose CPU — must stay near cpu_eager, not merely beat forced GPU.
        if cpu_ms is not None and auto_ms > cpu_ms * 1.35:
            ok = False
            note = f"auto CPU path regresses vs cpu_eager: auto={auto_ms:.1f} cpu={cpu_ms:.1f}"
        elif forced_ms is not None and cpu_ms is not None and cpu_ms > forced_ms * 1.05:
            ok = False
            note = f"auto chose CPU but cpu_eager={cpu_ms:.1f}ms > forced_gpu={forced_ms:.1f}ms"
        elif forced_ms is not None and auto_ms > forced_ms * 1.10:
            ok = False
            note = f"auto CPU path slower than forced_gpu: auto={auto_ms:.1f} vs forced={forced_ms:.1f}"
    else:
        # Auto chose GPU/hetero — should not be much slower than CPU when CPU wins clearly.
        if cpu_ms is not None and forced_ms is not None and cpu_ms * 1.05 < forced_ms and auto_ms > cpu_ms * 1.10:
            ok = False
            note = f"auto stayed on GPU though CPU faster: auto={auto_ms:.1f} cpu={cpu_ms:.1f} forced={forced_ms:.1f}"
    return {
        "ok": ok,
        "note": note,
        "cpu_only": cpu_only,
        "auto_ms": auto_ms,
        "forced_gpu_ms": forced_ms,
        "cpu_eager_ms": cpu_ms,
        "devices": devices,
    }


def run_real_model_suite(
    *,
    model_id: str = "Qwen/Qwen3-8B",
    seq_len: int = 16,
    iters: int = 2,
    warmup: int = 1,
) -> dict[str, Any]:
    """Qwen3-8B BF16 fixed-shape logits: forced GPU vs CPU eager vs Accelerate."""
    # Reuse the public transformer runner (already labels approaches) and rename.
    from benchmarks.suites.runners import run_transformer_beyond_vram_suite

    payload = run_transformer_beyond_vram_suite(
        model_id=model_id,
        seq_len=seq_len,
        iters=iters,
        warmup=warmup,
    )
    approaches = dict(payload.get("approaches") or {})
    # Relabel for hard-validation clarity.
    remapped: dict[str, Any] = {}
    if "tensortorrent_auto" in approaches:
        remapped["auto"] = approaches["tensortorrent_auto"]
        if isinstance(remapped["auto"], dict):
            remapped["auto"].setdefault("extras", {})["mode_label"] = "auto"
        elif isinstance(remapped["auto"], TimedRun):
            remapped["auto"].extras["mode_label"] = "auto"
    if "tensortorrent" in approaches:
        remapped["forced_gpu"] = approaches["tensortorrent"]
        if isinstance(remapped["forced_gpu"], dict):
            remapped["forced_gpu"].setdefault("extras", {})["mode_label"] = "forced_gpu"
        elif isinstance(remapped["forced_gpu"], TimedRun):
            remapped["forced_gpu"].extras["mode_label"] = "forced_gpu"
    for key in ("cpu_eager", "accelerate", "gpu_eager"):
        if key in approaches:
            remapped[key] = approaches[key]
    payload = {
        **payload,
        "suite": "hard_real_model",
        "approaches": remapped,
        "source_suite": "transformer_beyond_vram",
    }
    return payload


def run_hard_validation_suite(
    *,
    smoke: bool = False,
    iters: int = 0,
    warmup: int = 0,
    include_transformer: bool = True,
    model_id: str = "Qwen/Qwen3-8B",
    seq_len: int = 16,
) -> dict[str, Any]:
    """Run all hard-validation subsections and return a combined payload."""
    heavy_iters = iters or (2 if smoke else 3)
    heavy_warmup = warmup or 1
    suites: dict[str, Any] = {}
    suites["forced_gpu_fit"] = run_forced_gpu_fit_suite(iters=heavy_iters, warmup=heavy_warmup, smoke=smoke)
    release_host_memory()
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    release_host_memory()
    suites["forced_gpu_beyond"] = run_forced_gpu_beyond_suite(iters=heavy_iters, warmup=heavy_warmup, smoke=smoke)
    release_host_memory()
    suites["auto_bakeoff"] = run_auto_bakeoff_suite(iters=heavy_iters, warmup=heavy_warmup, smoke=smoke)
    release_host_memory()
    if include_transformer and not smoke:
        suites["real_model"] = run_real_model_suite(
            model_id=model_id,
            seq_len=seq_len,
            iters=max(1, min(heavy_iters, 2)),
            warmup=heavy_warmup,
        )
        release_host_memory()
    return {
        "suite": "hard_validation",
        "evidence": evidence_class("MEASURED"),
        "smoke": smoke,
        "suites": suites,
    }


def render_hard_validation_table(payload: dict[str, Any]) -> str:
    """Concise markdown before/after-style table from a hard-validation payload."""
    lines = [
        "# Hard GPU validation",
        "",
        "| Case | Mode | median_ms | peak_VRAM | regions | devices | strategy | ok | note |",
        "|---|---|---:|---:|---:|---|---|---|---|",
    ]

    def _row(case: str, name: str, run: Any) -> None:
        if isinstance(run, TimedRun):
            d = {
                "ok": run.ok,
                "median_ms": run.median_ms,
                "peak_device_bytes": run.peak_device_bytes,
                "note": run.note,
                "extras": run.extras,
            }
        elif isinstance(run, dict):
            d = run
        else:
            return
        extras = d.get("extras") or {}
        peak = int(d.get("peak_device_bytes") or 0)
        lines.append(
            "| {case} | {mode} | {ms:.2f} | {peak:.2f} GiB | {regs} | `{devs}` | {strategy} | {ok} | {note} |".format(
                case=case,
                mode=name,
                ms=float(d.get("median_ms") or 0.0),
                peak=peak / (1024**3),
                regs=extras.get("region_count") or extras.get("n_regions") or "",
                devs=",".join(str(x) for x in (extras.get("devices_used") or [])[:3]) or "-",
                strategy=extras.get("execution_strategy") or "-",
                ok="✓" if d.get("ok") else "✗",
                note=(d.get("note") or "")[:60].replace("|", "/"),
            )
        )

    suites = payload.get("suites") or {}
    fit = suites.get("forced_gpu_fit") or {}
    for row in fit.get("results") or []:
        case = f"fit@{row.get('vram_fraction')}"
        for name, run in (row.get("approaches") or {}).items():
            _row(case, name, run)

    beyond = suites.get("forced_gpu_beyond") or {}
    for row in beyond.get("results") or []:
        case = f"beyond@{row.get('vram_multiple')}"
        for name, run in (row.get("approaches") or {}).items():
            _row(case, name, run)

    auto = suites.get("auto_bakeoff") or {}
    for row in auto.get("results") or []:
        case = f"auto-{row.get('kind')}@{row.get('scale')}"
        for name, run in (row.get("approaches") or {}).items():
            _row(case, name, run)
        check = row.get("auto_selection_check") or {}
        if check:
            lines.append(
                f"| {case} | selection_check |  |  |  |  |  | "
                f"{'✓' if check.get('ok') else '✗'} | {(check.get('note') or '')[:60]} |"
            )

    real = suites.get("real_model") or {}
    for name, run in (real.get("approaches") or {}).items():
        _row("qwen3-8b", name, run)

    lines.append("")
    return "\n".join(lines)
