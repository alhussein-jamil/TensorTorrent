# Benchmark report

Full environment and measured commit: [`raw/environment.json`](raw/environment.json).

_commit `32f09ac47b1d` · torch 2.13.0+cu130 · CUDA available=True · smoke=False · driver 595.84_

## Fit-in-VRAM workloads

| Workload | Eager ms | torch.compile ms | TT ms | rel | Peak VRAM MB | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mlp_512x8 | 0.23 | 0.27 | 0.28 | 1.18× | 17.2 | ok |
| transformer_256 | 0.26 | 0.29 | 0.34 | 1.34× | 20.4 | ok |
| mlp_2048x8 | 0.70 | 0.82 | 0.75 | 1.06× | 146.4 | ok |

## Beyond VRAM — DeepMLP baselines

*DeepMLP beyond VRAM* — params 12.35 GB (1.50× VRAM when applicable)

| Approach | Median ms | Peak VRAM GB | Peak host GB | Status |
| --- | ---: | ---: | ---: | --- |
| gpu_eager | OOM | OOM | OOM | OOM |
| tensortorrent | 444.21 | 0.00 | 25.37 | ok |
| tensortorrent_gpu_stream | 740.30 | 6.72 | 38.06 | ok |
| cpu_eager | 446.12 | 0.08 | 38.07 | ok |
| accelerate | 806.74 | 5.38 | 38.07 | ok |

## Beyond VRAM — transformer baselines

*HF transformer beyond VRAM* — params 16.38 GB (1.99× VRAM when applicable)

| Approach | Median ms | Peak VRAM GB | Peak host GB | Status |
| --- | ---: | ---: | ---: | --- |
| gpu_eager | INFEASIBLE | INFEASIBLE | INFEASIBLE | INFEASIBLE |
| cpu_eager | 4130.58 | 0.00 | 16.07 | ok |
| tensortorrent_auto | 1678.44 | 6.83 | 36.75 | ok |
| tensortorrent | 1699.69 | 6.53 | 37.75 | ok |
| accelerate | 1616.10 | 6.44 | 37.75 | ok |

## Memory budget curve

| Budget GiB | Median ms | Throughput iters/s | Transfer GB | GPU compute % | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| 8.0 | 13.21 | 75.70 | NOT MEASURED | 100.0% | ok |
| 6.0 | 14.94 | 66.95 | 0.000 | 100.0% | ok |
| 4.0 | 14.73 | 67.88 | 0.000 | 100.0% | ok |
| 3.0 | 142.71 | 7.01 | 1.410 | 100.0% | ok |
| 2.0 | 238.71 | 4.19 | 2.484 | 100.0% | ok |

## Model size crossover

| Size × VRAM | GPU eager ms | TensorTorrent ms | Status |
| --- | ---: | ---: | --- |
| 0.50 | fits | 14.73 | TT:ok eager:ok |
| 0.75 | fits | 24.55 | TT:ok eager:ok |
| 0.90 | OOM | 677.23 | TT:ok eager:OOM |
| 1.00 | OOM | 206.69 | TT:ok eager:OOM |
| 1.10 | OOM | 278.41 | TT:ok eager:OOM |
| 1.25 | OOM | 478.37 | TT:ok eager:OOM |
| 1.50 | OOM | 694.86 | TT:ok eager:OOM |

## Heterogeneous placement

| Case | Evidence | Notes |
| --- | --- | --- |
| gpu_plus_cpu_allowed | MEASURED | TT=ok; devices=['cuda_gpu_0'] |
| two_gpu | SUPPORTED_BUT_UNMEASURED | only 1 CUDA device(s) present; multi-GPU not measured |
