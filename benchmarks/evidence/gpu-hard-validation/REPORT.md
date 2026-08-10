# TensorTorrent benchmark report

commit `441a932aade1` · torch 2.13.0+cu130 · CUDA available=True · smoke=False · driver 595.84
## Beyond VRAM — transformer baselines

*HF transformer beyond VRAM* — params 16.38 GB (1.99× VRAM when applicable)

| Approach | Median ms | Peak VRAM GB | Peak host GB | Status |
| --- | ---: | ---: | ---: | --- |
| gpu_eager | INFEASIBLE | INFEASIBLE | INFEASIBLE | INFEASIBLE |
| cpu_eager | 2782.88 | 0.00 | 16.05 | ok |
| tensortorrent_auto | 1624.97 | 6.83 | 33.82 | ok |
| tensortorrent | 1586.70 | 6.84 | 36.16 | ok |
| accelerate | 1581.65 | 6.44 | 36.16 | ok |
