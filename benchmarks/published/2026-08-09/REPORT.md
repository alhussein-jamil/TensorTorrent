# TensorTorrent benchmark report

commit `b554d4cd43a7` · torch 2.13.0+cu130 · CUDA available=True · smoke=False · driver 595.84
## Fit-in-VRAM workloads

| Workload | Eager ms | torch.compile ms | TT ms | rel | Peak VRAM MB | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mlp_512x8 | 0.23 | 0.28 | 1.12 | 4.83× | 17.2 | ok |
| transformer_256 | 0.26 | 0.29 | 1.02 | 3.88× | 14.6 | ok |
| mlp_2048x8 | 0.71 | 0.84 | 1.22 | 1.71× | 143.4 | ok |

## Beyond VRAM — DeepMLP baselines

*DeepMLP beyond VRAM* — params 12.35 GB (1.50× VRAM when applicable)

| Approach | Median ms | Peak VRAM GB | Peak host GB | Status |
| --- | ---: | ---: | ---: | --- |
| gpu_eager | OOM | OOM | OOM | OOM |
| tensortorrent | 1633.71 | 0.61 | 25.47 | ok |
| cpu_eager | 756.08 | 0.08 | 37.85 | ok |
| accelerate | 940.28 | 5.38 | 37.85 | ok |

## Beyond VRAM — transformer baselines

*HF transformer beyond VRAM* — params 16.38 GB (1.99× VRAM when applicable)

| Approach | Median ms | Peak VRAM GB | Peak host GB | Status |
| --- | ---: | ---: | ---: | --- |
| gpu_eager | INFEASIBLE | INFEASIBLE | INFEASIBLE | INFEASIBLE |
| cpu_eager | 3433.29 | 0.00 | 16.15 | ok |
| tensortorrent | 2325.03 | 1.33 | 36.89 | ok |
| accelerate | OOM | OOM | OOM | OOM |

## Memory budget curve

| Budget GiB | Median ms | Throughput iters/s | Transfer GB | GPU compute % | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| 8.0 | 19.44 | 51.43 | 3.759 | 100.0% | ok |
| 6.0 | 18.49 | 54.10 | 3.759 | 100.0% | ok |
| 4.0 | 385.65 | 2.59 | 7.518 | 100.0% | ok |
| 3.0 | 379.91 | 2.63 | 7.518 | 100.0% | ok |
| 2.0 | 382.09 | 2.62 | 7.518 | 100.0% | ok |

## Model size crossover

| Size × VRAM | GPU eager ms | TensorTorrent ms | Status |
| --- | ---: | ---: | --- |
| 0.50 | fits | 21.62 | TT:ok eager:ok |
| 0.75 | fits | 621.93 | TT:ok eager:ok |
| 0.90 | OOM | 933.31 | TT:ok eager:OOM |
| 1.00 | OOM | 1033.45 | TT:ok eager:OOM |
| 1.10 | OOM | 1096.42 | TT:ok eager:OOM |
| 1.25 | OOM | 1283.44 | TT:ok eager:OOM |
| 1.50 | OOM | 1543.58 | TT:ok eager:OOM |

## Heterogeneous placement

| Case | Evidence | Notes |
| --- | --- | --- |
| gpu_plus_cpu_allowed | MEASURED | TT=ok; devices=['cuda_gpu_0'] |
| two_gpu | SUPPORTED_BUT_UNMEASURED | only 1 CUDA device(s) present; multi-GPU not measured |
