"""Evidence README render: Qwen extras must survive crossover rows."""

from __future__ import annotations

from pathlib import Path

from benchmarks.tooling.render_evidence import write_report


def _run(*, ok: bool, median_ms: float = 0.0, extras: dict | None = None, note: str = "") -> dict:
    return {
        "ok": ok,
        "median_ms": median_ms,
        "peak_device_bytes": 1e9 if ok else 0,
        "note": note,
        "extras": extras or {},
    }


def test_write_report_keeps_qwen_cosine_and_labels_transfer_fail(tmp_path: Path) -> None:
    evidence = tmp_path
    summary = {
        "environment": {"gpu": "test-gpu", "gpu_vram_bytes": 8 * (1024**3), "torch": "2.13"},
        "suites": {
            "transformer_beyond_vram": {
                "params_bytes": 16e9,
                "approaches": {
                    "gpu_eager": _run(ok=False, note="infeasible"),
                    "cpu_eager": _run(ok=True, median_ms=3000),
                    "tensortorrent_auto": _run(
                        ok=True,
                        median_ms=1203,
                        extras={
                            "cosine": 0.99967,
                            "argmax_match": 15,
                            "argmax_total": 16,
                            "execution_strategy": "transfer_evict",
                        },
                    ),
                    "tensortorrent": _run(ok=True, median_ms=1229),
                    "accelerate": _run(ok=True, median_ms=1625, extras={}),
                },
            },
            "beyond_vram_deepmlp": {
                "params_bytes": 12e9,
                "vram_multiple": 1.5,
                "approaches": {
                    "gpu_eager": _run(ok=False),
                    "cpu_eager": _run(ok=True, median_ms=429),
                    "tensortorrent": _run(
                        ok=True, median_ms=434, extras={"execution_strategy": "direct_export_free", "devices_used": []}
                    ),
                    "tensortorrent_gpu_stream": _run(ok=True, median_ms=554),
                    "accelerate": _run(ok=True, median_ms=768),
                },
            },
            "model_size_crossover": {
                "results": [
                    {
                        "vram_multiple": 1.0,
                        "approaches": {
                            "gpu_eager": _run(ok=False, note="OOM"),
                            "tensortorrent": _run(
                                ok=False,
                                note="RuntimePlanError: native schedule execution failed: instruction transfer::->region_0:x opcode Transfer",
                            ),
                        },
                    }
                ]
            },
            "fit": {"results": []},
        },
        "smoke": False,
    }
    write_report(evidence, summary)
    body = (evidence / "README.md").read_text(encoding="utf-8")
    assert "cosine 0.9997" in body
    assert "argmax 15/16" in body
    assert "`Transfer fail`" in body
    assert "cosine ?" not in body
    assert "argmax None/None" not in body
