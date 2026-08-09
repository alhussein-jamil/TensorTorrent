"""Suite runners: fit, beyond-VRAM, pressure scaling, hetero markers."""

from __future__ import annotations

import contextlib
import time
from typing import Any

import torch

import tensortorrent as tt
from benchmarks.harness import (
    TimedRun,
    evidence_class,
    release_host_memory,
    reset_peaks,
    summarize_samples,
    sync,
    timed_callable,
)
from benchmarks.instrumentation import summarize_execution
from benchmarks.memory_hygiene import (
    abort_if_host_tight,
    crossover_multiples,
    deepmlp_weight_file,
    load_deepmlp,
    run_json_worker,
)
from benchmarks.transformer_workload import load_causal_lm, release_model
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
    code, data, err = run_json_worker(
        "benchmarks._gpu_eager_worker",
        {"width": width, "depth": depth, "batch": batch},
        timeout_s=180,
    )
    if data is None:
        if "out of memory" in err.lower() or "oom" in err.lower():
            return TimedRun(ok=False, note=f"CUDA OOM (expected): {err[:120]}")
        return TimedRun(ok=False, note=(err or f"gpu eager worker failed rc={code}")[:160])
    if data.get("oom"):
        return TimedRun(ok=False, note=f"CUDA OOM (expected): {str(data.get('note', ''))[:120]}")
    note = str(data.get("note") or "fits in VRAM (probe; not timed)")
    # Feasibility probe: do not treat median_ms=0 as a timed measurement.
    return TimedRun(
        ok=True,
        median_ms=0.0,
        note=note,
        extras={"probe": "feasibility", "fits": bool(data.get("fits", True)), "timed": False},
    )


def _tt_plan_extras(compiled: Any) -> dict[str, Any]:
    plan = compiled.specialized.plan
    sched = compiled.specialized.schedule
    from tensortorrent.compile.fit import should_hoist_resident_parameters
    from tensortorrent.ir.graph import OpCode

    n_transfer = sum(1 for i in sched.instructions if i.opcode == OpCode.TRANSFER)
    n_load = sum(1 for i in sched.instructions if i.opcode == OpCode.LOAD)
    n_evict = sum(1 for i in sched.instructions if i.opcode == OpCode.EVICT)
    store_stats = compiled.executor.parameter_store.stats()
    store_kind = str(store_stats.get("kind") or "")
    state_bytes = 0
    program = getattr(getattr(compiled, "portable", None), "program", None) or getattr(
        compiled.specialized, "program", None
    )
    if program is not None:
        with contextlib.suppress(Exception):
            state_bytes = int(program.total_state_bytes())
    if state_bytes <= 0:
        state_bytes = int(store_stats.get("resident_bytes") or 0)
    machine = getattr(compiled.specialized, "machine", None)
    hoist = should_hoist_resident_parameters(
        compiled.config,
        state_bytes=state_bytes,
        machine=machine,
    )
    if "stream" in store_kind.lower() or n_load > 0:
        strategy = "streaming"
    elif not hoist:
        strategy = "transfer_evict"
    else:
        strategy = "resident"
    return {
        "devices_used": list(plan.devices_used),
        "prefetch_distance": int(plan.prefetch_distance),
        "predicted_latency_s": float(plan.predicted_latency_s),
        "notes": list(plan.notes)[:24],
        "schedule_notes": list(sched.notes)[:12],
        "n_transfer": n_transfer,
        "n_load": n_load,
        "n_evict": n_evict,
        "parameter_store": store_stats,
        "execution_strategy": strategy,
        "hoist_resident_parameters": bool(hoist),
        "state_bytes": state_bytes,
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

        reset_peaks()
        try:
            with torch.no_grad():
                samples = timed_callable(lambda m=m_ref, inp=x_ref: m(inp), iters=iters, warmup=warmup)
            err = _max_abs_err(m_ref(x_ref), expected)
            run = summarize_samples(samples, extras={"max_abs_err": err})
            row["approaches"]["eager"] = run
        except Exception as exc:  # noqa: BLE001
            row["approaches"]["eager"] = TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}")

        reset_peaks()
        try:
            compiled_pt = torch.compile(m_ref)
            with torch.no_grad():
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
    instrument: bool = False,
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
    batch = 2 if smoke else 8
    x = torch.randn(batch, width)
    approaches: dict[str, Any] = {}

    approaches["gpu_eager"] = _gpu_eager_oom_probe(width, depth, batch)
    release_host_memory()
    reset_peaks()

    with deepmlp_weight_file(width, depth) as (wpath, pbytes):
        tight = abort_if_host_tight(pbytes, label="beyond_vram")
        if tight is not None:
            approaches["tensortorrent"] = tight
            approaches["cpu_eager"] = TimedRun(ok=False, note="skipped: host RAM tight")
            approaches["accelerate"] = TimedRun(ok=False, note="skipped: host RAM tight")
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

        ref = load_deepmlp(wpath, width, depth)
        with torch.no_grad():
            expected = ref(x).clone()
        del ref
        release_host_memory()

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
            m = load_deepmlp(wpath, width, depth)
            t0 = time.perf_counter()
            compiled = tt.compile(m, example_inputs=(x.cpu(),), config=cfg)
            compile_s = time.perf_counter() - t0
            del m
            release_host_memory()
            with torch.no_grad():
                samples = timed_callable(lambda fn=compiled, inp=x: fn(inp.cpu()), iters=iters, warmup=warmup)
                out = compiled(x.cpu())
            extras = _tt_plan_extras(compiled)
            if instrument:
                extras["instrumentation"] = summarize_execution(compiled)
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
        release_host_memory()
        reset_peaks()

        try:
            m = load_deepmlp(wpath, width, depth)
            with torch.no_grad():
                samples = timed_callable(lambda model=m, inp=x: model(inp.cpu()), iters=iters, warmup=warmup)
                err = _max_abs_err(m(x.cpu()), expected)
            approaches["cpu_eager"] = summarize_samples(samples, extras={"max_abs_err": err})
            del m
        except Exception as exc:  # noqa: BLE001
            approaches["cpu_eager"] = TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:160])
        release_host_memory()
        reset_peaks()

        accel_cfg: dict[str, Any] = {
            "device_map": "auto",
            "max_memory": {0: f"{max(1, int(vram * 0.7 / (1024**3)))}GiB", "cpu": "48GiB"},
            "no_split_module_classes": ["Linear"],
        }
        try:
            from accelerate import dispatch_model, infer_auto_device_map

            m = load_deepmlp(wpath, width, depth)
            t0 = time.perf_counter()
            dmap = infer_auto_device_map(
                m,
                max_memory=accel_cfg["max_memory"],
                no_split_module_classes=list(accel_cfg["no_split_module_classes"]),
            )
            m = dispatch_model(m, device_map=dmap)
            compile_s = time.perf_counter() - t0
            first = next(iter(dmap.values())) if dmap else "cpu"
            xd = x.cuda() if isinstance(first, str) and first.startswith("cuda") else x.cpu()
            with torch.no_grad():
                samples = timed_callable(lambda model=m, inp=xd: model(inp), iters=iters, warmup=warmup)
            run = summarize_samples(samples, extras={"accelerate_config": dict(accel_cfg), "device_map": dict(dmap)})
            run.compile_s = compile_s
            approaches["accelerate"] = run
            del m
        except ImportError:
            approaches["accelerate"] = TimedRun(ok=False, note="accelerate not installed")
        except Exception as exc:  # noqa: BLE001
            approaches["accelerate"] = TimedRun(
                ok=False,
                note=f"{type(exc).__name__}: {exc}"[:200],
                extras={"accelerate_config": dict(accel_cfg)},
            )
        release_host_memory()
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


def run_transformer_beyond_vram_suite(
    *,
    model_id: str = "Qwen/Qwen3-8B",
    seq_len: int = 16,
    iters: int = 3,
    warmup: int = 1,
) -> dict[str, Any]:
    """Real HF causal LM beyond VRAM with fair baselines."""
    if not torch.cuda.is_available():
        return {
            "suite": "transformer_beyond_vram",
            "evidence": evidence_class("SUPPORTED_BUT_UNMEASURED"),
            "note": "no CUDA device",
            "approaches": {},
        }

    vram = int(torch.cuda.get_device_properties(0).total_memory)
    approaches: dict[str, Any] = {}
    wrap = None
    try:
        wrap, (input_ids, attention_mask), spec, meta = load_causal_lm(
            model_id=model_id,
            seq_len=seq_len,
        )
        pbytes = int(spec.param_bytes)
        with torch.no_grad():
            expected = wrap(input_ids, attention_mask).detach().cpu()

        reset_peaks()
        if pbytes > int(vram * 0.95):
            approaches["gpu_eager"] = TimedRun(
                ok=False,
                note=(
                    f"infeasible by parameter footprint: "
                    f"params={pbytes / 1e9:.2f}GB > VRAM={vram / 1e9:.2f}GB (not attempted)"
                ),
            )
        else:
            try:
                wrap_gpu = wrap.cuda()
                ids = input_ids.cuda()
                mask = attention_mask.cuda()
                with torch.no_grad():
                    samples = timed_callable(
                        lambda model=wrap_gpu, i=ids, a=mask: model(i, a),
                        iters=iters,
                        warmup=warmup,
                    )
                approaches["gpu_eager"] = summarize_samples(samples)
                wrap_gpu.cpu()
                del ids, mask, wrap_gpu
            except torch.cuda.OutOfMemoryError as exc:
                approaches["gpu_eager"] = TimedRun(ok=False, note=f"CUDA OOM (expected): {exc}"[:160])
            except Exception as exc:  # noqa: BLE001
                approaches["gpu_eager"] = TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:160])
            finally:
                wrap.cpu()
                release_host_memory()
        reset_peaks()

        reset_peaks()
        try:
            cpu_iters = 3
            with torch.no_grad():
                samples = timed_callable(
                    lambda model=wrap, i=input_ids, a=attention_mask: model(i, a),
                    iters=cpu_iters,
                    warmup=0,
                )
            approaches["cpu_eager"] = summarize_samples(samples, extras={"iters": cpu_iters})
        except Exception as exc:  # noqa: BLE001
            approaches["cpu_eager"] = TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:160])
        reset_peaks()

        reset_peaks()
        try:
            cfg = tt.CompileConfig(
                use_torch_compile=False,
                measure_regions=False,
                allow_gpu=True,
                allow_cpu=False,
                vram_budget_bytes=vram,
                max_region_nodes=16,
                prefetch_distance=1,
                enable_linear_sharding=True,
                validate_numerics=False,
            )
            t_cap = time.perf_counter()
            ep = tt.capture_module(wrap.cpu().eval(), (input_ids, attention_mask))
            capture_s = time.perf_counter() - t_cap
            release_model(wrap)
            wrap = None
            t0 = time.perf_counter()
            compiled = tt.compile_exported(ep, config=cfg)
            compile_s = time.perf_counter() - t0
            with torch.no_grad():
                samples = timed_callable(
                    lambda fn=compiled, i=input_ids, a=attention_mask: fn(i, a),
                    iters=iters,
                    warmup=warmup,
                )
                out = compiled(input_ids, attention_mask)
            extras = _tt_plan_extras(compiled)
            extras["instrumentation"] = summarize_execution(compiled)
            extras["capture_s"] = capture_s
            out_f = out.detach().float().cpu()
            exp_f = expected.float()
            err = float((out_f - exp_f).abs().max().item())
            cos = float(torch.nn.functional.cosine_similarity(out_f.reshape(1, -1), exp_f.reshape(1, -1)).item())
            argmax_match = int((out_f[0].argmax(-1) == exp_f[0].argmax(-1)).sum().item())
            run = summarize_samples(
                samples,
                extras={
                    "max_abs_err": err,
                    "cosine": cos,
                    "argmax_match": argmax_match,
                    "argmax_total": int(out_f.shape[1]),
                    **extras,
                },
            )
            run.compile_s = compile_s
            if cos < 0.99 or argmax_match < max(1, int(out_f.shape[1]) // 2):
                run.ok = False
                run.note = f"numerical mismatch cos={cos:.4f} argmax={argmax_match}/{out_f.shape[1]} max_abs_err={err}"
            approaches["tensortorrent"] = run
            with contextlib.suppress(Exception):
                compiled.close()
        except Exception as exc:  # noqa: BLE001
            approaches["tensortorrent"] = TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:200])
        reset_peaks()

        reset_peaks()
        accel_cfg: dict[str, Any] = {
            "device_map": "auto",
            "max_memory": {0: "6GiB", "cpu": "40GiB"},
            "dtype": "bfloat16",
        }
        offload_dir = None
        try:
            import tempfile

            from transformers import AutoModelForCausalLM

            offload_dir = tempfile.mkdtemp(prefix="tt_bench_accel_")
            accel_cfg["offload_folder"] = offload_dir
            t0 = time.perf_counter()
            accel_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map="auto",
                max_memory=accel_cfg["max_memory"],
                offload_folder=offload_dir,
                trust_remote_code=True,
            ).eval()
            compile_s = time.perf_counter() - t0
            ids = input_ids
            mask = attention_mask
            try:
                first_dev = next(accel_model.parameters()).device
                if first_dev.type != "meta":
                    ids = input_ids.to(first_dev)
                    mask = attention_mask.to(first_dev)
            except StopIteration:
                pass

            def _accel_fwd(model=accel_model, i=ids, a=mask) -> Any:
                return model(input_ids=i, attention_mask=a, use_cache=False).logits

            with torch.no_grad():
                samples = timed_callable(_accel_fwd, iters=iters, warmup=warmup)
            run = summarize_samples(
                samples,
                extras={"accelerate_config": dict(accel_cfg)},
            )
            run.compile_s = compile_s
            approaches["accelerate"] = run
            del accel_model
        except ImportError:
            approaches["accelerate"] = TimedRun(ok=False, note="accelerate/transformers not installed")
        except Exception as exc:  # noqa: BLE001
            approaches["accelerate"] = TimedRun(
                ok=False,
                note=(f"tested Accelerate auto-offload configuration OOM'd / failed: {type(exc).__name__}: {exc}")[
                    :220
                ],
                extras={"accelerate_config": dict(accel_cfg)},
            )
        finally:
            if offload_dir:
                import shutil

                with contextlib.suppress(Exception):
                    shutil.rmtree(offload_dir, ignore_errors=True)
        reset_peaks()

        return {
            "suite": "transformer_beyond_vram",
            "evidence": evidence_class("MEASURED"),
            "model_id": model_id,
            "seq_len": seq_len,
            "vram_bytes": vram,
            "params_bytes": pbytes,
            "params_over_vram": pbytes / vram,
            "transformer_spec": {
                "model_id": spec.model_id,
                "revision": spec.revision,
                "dtype": spec.dtype,
                "seq_len": spec.seq_len,
                "batch_size": spec.batch_size,
                "param_count": spec.param_count,
                "param_bytes": spec.param_bytes,
                "input_shapes": spec.input_shapes,
                "notes": spec.notes,
            },
            "load_meta": meta,
            "approaches": approaches,
        }
    finally:
        release_model(wrap)
        reset_peaks()


def run_memory_budget_curve_suite(
    *,
    iters: int = 5,
    warmup: int = 1,
    smoke: bool = False,
    instrument: bool = True,
) -> dict[str, Any]:
    """DeepMLP under absolute VRAM budgets (GiB), with transfer/GPU observability."""
    if not torch.cuda.is_available():
        return {
            "suite": "memory_budget_curve",
            "evidence": evidence_class("SUPPORTED_BUT_UNMEASURED"),
            "note": "no CUDA device",
            "results": [],
        }

    gib = 1024**3
    budgets_gib = (8.0, 4.0, 2.0) if smoke else (8.0, 6.0, 4.0, 3.0, 2.0)
    vram = int(torch.cuda.get_device_properties(0).total_memory)
    width, depth = deep_mlp_for_bytes(int(vram * (0.25 if smoke else 0.45)), width=2048 if smoke else 4096)
    torch.manual_seed(0)
    model = DeepMLP(width, depth).eval()
    x = torch.randn(2, width)
    pbytes = param_bytes(model)
    with torch.no_grad():
        expected = model(x).clone()

    rows: list[dict[str, Any]] = []
    for budget_gib in budgets_gib:
        budget = int(budget_gib * gib)
        reset_peaks()
        row: dict[str, Any] = {
            "vram_budget_gib": budget_gib,
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
            with torch.no_grad():
                samples = timed_callable(lambda fn=compiled, inp=x: fn(inp.cpu()), iters=iters, warmup=warmup)
                out = compiled(x.cpu())
            extras = _tt_plan_extras(compiled)
            if instrument:
                extras["instrumentation"] = summarize_execution(compiled)
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
        "suite": "memory_budget_curve",
        "evidence": evidence_class("MEASURED"),
        "vram_bytes": vram,
        "params_bytes": pbytes,
        "width": width,
        "depth": depth,
        "budgets_gib": list(budgets_gib),
        "results": rows,
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
    full_crossover: bool = False,
) -> dict[str, Any]:
    """Sweep model sizes around the VRAM boundary (one child process per size)."""
    if not torch.cuda.is_available():
        return {
            "suite": "model_size_crossover",
            "evidence": evidence_class("SUPPORTED_BUT_UNMEASURED"),
            "note": "no CUDA device",
            "results": [],
        }

    vram = int(torch.cuda.get_device_properties(0).total_memory)
    multiples = crossover_multiples(smoke=smoke, full=full_crossover)
    width = 2048 if smoke else 4096
    return _run_model_size_scaling_subprocess(
        vram=vram,
        multiples=multiples,
        iters=iters,
        warmup=warmup,
        width=width,
    )


def measure_one_crossover_point(
    *,
    width: int,
    depth: int,
    vram_multiple: float,
    vram_bytes: int,
    iters: int = 1,
    warmup: int = 0,
) -> dict[str, Any]:
    """Measure one crossover size in the current process (intended for a child)."""
    from dataclasses import asdict

    batch = 2
    x = torch.randn(batch, width)
    approaches: dict[str, Any] = {}

    approaches["gpu_eager"] = asdict(_gpu_eager_oom_probe(width, depth, batch))
    release_host_memory()

    with deepmlp_weight_file(width, depth) as (wpath, pbytes):
        tight = abort_if_host_tight(pbytes, label=f"crossover_{vram_multiple:.2f}x")
        if tight is not None:
            approaches["tensortorrent"] = asdict(tight)
            return {
                "vram_multiple": vram_multiple,
                "params_bytes": pbytes,
                "width": width,
                "depth": depth,
                "approaches": approaches,
            }

        ref = load_deepmlp(wpath, width, depth)
        with torch.no_grad():
            expected = ref(x).clone()
        del ref
        release_host_memory()

        reset_peaks()
        near_or_over = pbytes >= int(vram_bytes * 0.70)
        try:
            cfg = tt.CompileConfig(
                use_torch_compile=False,
                measure_regions=False,
                allow_gpu=True,
                allow_cpu=not near_or_over,
                # Always pass physical VRAM so hoist uses ACCELERATOR_REGION_STATE_FRACTION
                # headroom (0.70×) instead of treating unset budget as "infinite residency".
                vram_budget_bytes=vram_bytes,
                max_region_nodes=16,
                prefetch_distance=1,
            )
            m = load_deepmlp(wpath, width, depth)
            t0 = time.perf_counter()
            compiled = tt.compile(m, example_inputs=(x.cpu(),), config=cfg)
            compile_s = time.perf_counter() - t0
            del m
            release_host_memory()
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
            approaches["tensortorrent"] = asdict(run)
            with contextlib.suppress(Exception):
                compiled.close()
        except Exception as exc:  # noqa: BLE001
            approaches["tensortorrent"] = asdict(TimedRun(ok=False, note=f"{type(exc).__name__}: {exc}"[:160]))

    return {
        "vram_multiple": vram_multiple,
        "params_bytes": pbytes,
        "width": width,
        "depth": depth,
        "approaches": approaches,
    }


def _run_model_size_scaling_subprocess(
    *,
    vram: int,
    multiples: tuple[float, ...],
    iters: int,
    warmup: int,
    width: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for mult in multiples:
        w, depth = deep_mlp_for_bytes(int(vram * mult), width=width)
        print(f"  crossover {mult:.2f}× → child process", flush=True)
        code, data, err = run_json_worker(
            "benchmarks._crossover_worker",
            {
                "width": w,
                "depth": depth,
                "vram_multiple": mult,
                "vram_bytes": vram,
                "iters": iters,
                "warmup": warmup,
            },
            timeout_s=900,
        )
        if data is None:
            rows.append(
                {
                    "vram_multiple": mult,
                    "width": w,
                    "depth": depth,
                    "evidence": evidence_class("MEASURED"),
                    "approaches": {
                        "tensortorrent": TimedRun(
                            ok=False,
                            note=(err or f"crossover worker failed rc={code}")[-200:],
                        )
                    },
                }
            )
        else:
            data["evidence"] = evidence_class("MEASURED")
            rows.append(data)
        release_host_memory()

    return {
        "suite": "model_size_crossover",
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
    """Compat shim — plotting lives in ``benchmarks.report``."""
    from benchmarks.report import try_plot_all

    return try_plot_all(results_root, payload)
