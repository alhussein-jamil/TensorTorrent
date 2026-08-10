# GPU hard-validation — Qwen remeasure (branch artifact)

Frozen from `benchmarks/results/qwen_partial_20260810T113053Z/` after partial-residency work on `agent/gpu-hard-validation`.

**Not** a replacement for published `v0.3.1` evidence. Tree was dirty at measure time (`441a932` + WIP); finished code landed as `9d8547d`.

## Headline (Qwen3-8B BF16 logits, seq=16)

| Approach | Median ms | Peak VRAM | Status |
| --- | ---: | ---: | --- |
| gpu_eager | — | — | infeasible (16.38 GB params > 8.22 GB VRAM) |
| cpu_eager | 2783 | 0 | ok |
| tensortorrent_auto | 1625 | 6.83 GB | ok · GPU `transfer_evict` · cosine 0.9997 · argmax 15/16 |
| tensortorrent (forced GPU) | 1587 | 6.84 GB | ok · H2D ≈ 9.87 GB/forward |
| accelerate | 1582 | 6.44 GB | ok |

Auto ≈ forced GPU and beats CPU (~1.7×). Raw JSON: `raw/`.
