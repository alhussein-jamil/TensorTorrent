# TensorTorrent benchmark report

commit `fb503e5dfb1f` · torch 2.13.0+cu130 · CUDA available=True · smoke=False · driver 595.84
## Fit-in-VRAM workloads

| Workload | Eager ms | torch.compile ms | TT ms | rel | Peak VRAM MB | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mlp_512x8 | 0.23 | 0.27 | 1.09 | 4.67× | 17.2 | ok |
| transformer_256 | 0.33 | 0.29 | 1.07 | 3.20× | 14.6 | ok |
| mlp_2048x8 | 0.71 | 0.83 | 1.24 | 1.73× | 143.4 | ok |

## Beyond VRAM — DeepMLP baselines

*DeepMLP beyond VRAM* — params 12.35 GB (1.50× VRAM when applicable)

| Approach | Median ms | Peak VRAM GB | Peak host GB | Status |
| --- | ---: | ---: | ---: | --- |
| gpu_eager | OOM | OOM | OOM | OOM |
| tensortorrent | 1580.22 | 0.61 | 25.71 | ok |
| cpu_eager | 733.90 | 0.08 | 38.07 | ok |
| accelerate | 915.73 | 5.38 | 38.07 | ok |

## Beyond VRAM — transformer baselines

*HF transformer beyond VRAM* — params 16.38 GB (1.99× VRAM when applicable)

| Approach | Median ms | Peak VRAM GB | Peak host GB | Status |
| --- | ---: | ---: | ---: | --- |
| gpu_eager | INFEASIBLE | INFEASIBLE | INFEASIBLE | INFEASIBLE |
| cpu_eager | 2119.25 | 0.00 | 16.17 | ok |
| tensortorrent | 2521.76 | 1.33 | 36.91 | ok |
| accelerate | OOM | OOM | OOM | OOM |

## Memory budget curve

| Budget GiB | Median ms | Throughput iters/s | Transfer GB | GPU compute % | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| 8.0 | 19.12 | 52.30 | 3.759 | 100.0% | ok |
| 6.0 | 19.80 | 50.51 | 3.759 | 100.0% | ok |
| 4.0 | 389.35 | 2.57 | 7.518 | 100.0% | ok |
| 3.0 | 387.87 | 2.58 | 7.518 | 100.0% | ok |
| 2.0 | 385.84 | 2.59 | 7.518 | 100.0% | ok |

## Model size crossover

| Size × VRAM | GPU eager ms | TensorTorrent ms | Status |
| --- | ---: | ---: | --- |
| 0.50 | fits | 20.34 | TT:ok eager:ok |
| 0.75 | fits | 613.62 | TT:ok eager:ok |
| 0.90 | OOM | 738.31 | TT:ok eager:OOM |
| 1.00 | OOM | 1033.49 | TT:ok eager:OOM |
| 1.10 | OOM | 1125.93 | TT:ok eager:OOM |
| 1.25 | OOM | 1274.01 | TT:ok eager:OOM |
| 1.50 | OOM | 1511.98 | TT:ok eager:OOM |

## Heterogeneous placement

| Case | Evidence | Notes |
| --- | --- | --- |
| gpu_plus_cpu_allowed | MEASURED | TT=ok; devices=['cuda_gpu_0'] |
| two_gpu | SUPPORTED_BUT_UNMEASURED | only 1 CUDA device(s) present; multi-GPU not measured |
