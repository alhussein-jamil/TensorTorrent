"""Static padded generate helpers and smoke suite."""

from __future__ import annotations

import torch
from benchmarks.suites.generate_workload import TinyCausalLM, greedy_padded_decode

from tensortorrent.config import CompileConfig


def test_greedy_padded_decode_grows_mask() -> None:
    model = TinyCausalLM(vocab=32, width=16, depth=1, max_len=8).eval()
    ids = torch.randint(0, 32, (2, 3))
    mask = torch.ones(2, 3, dtype=torch.long)
    with torch.no_grad():
        out = greedy_padded_decode(model, ids, mask, new_tokens=2)
    assert out.shape == (2, 5)
    assert torch.equal(out[:, :3], ids)


def test_tiny_generate_compile_matches_eager() -> None:
    import tensortorrent as tt

    model = TinyCausalLM(vocab=32, width=16, depth=1, max_len=8).eval()
    prompt = 3
    tokens = 2
    ids = torch.randint(0, 32, (1, prompt))
    mask = torch.ones(1, prompt, dtype=torch.long)
    example_ids = torch.zeros(1, prompt + tokens, dtype=torch.long)
    example_mask = torch.zeros(1, prompt + tokens, dtype=torch.long)
    example_ids[:, :prompt] = ids
    example_mask[:, :prompt] = 1
    with torch.no_grad():
        expected = greedy_padded_decode(model, ids, mask, new_tokens=tokens)
    compiled = tt.compile(
        model,
        (example_ids, example_mask),
        config=CompileConfig(use_torch_compile=False, measure_regions=False, allow_gpu=False),
    )
    try:

        def _fwd(i: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            return compiled(i, a)

        with torch.no_grad():
            got = greedy_padded_decode(_fwd, ids, mask, new_tokens=tokens)
        assert torch.equal(got.cpu(), expected.cpu())
    finally:
        compiled.close()


def test_generate_suite_smoke_skips_hf() -> None:
    from benchmarks.suites.runners import run_generate_suite

    payload = run_generate_suite(smoke=True, iters=1, warmup=0, new_tokens=2)
    assert payload["suite"] == "generate"
    assert payload["model"] == "TinyCausalLM"
    tt_run = payload["approaches"]["tensortorrent_padded"]
    assert tt_run.ok is True
    assert payload["approaches"]["accelerate_generate"].ok is False
