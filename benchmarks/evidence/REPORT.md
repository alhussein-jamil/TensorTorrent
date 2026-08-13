# Benchmark report

Full environment and measured commit: [`raw/environment.json`](raw/environment.json).

_commit `fdbe974e8d69` · torch 2.13.0+cu130 · CUDA available=True · smoke=False · driver 595.84_

## Fit-in-VRAM workloads

| Workload | Eager ms | torch.compile ms | TT ms | rel | Peak VRAM MB | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mlp_512x8 | 0.23 | 0.27 | 0.23 | 0.97× | 25.8 | ok |
| transformer_256 | 0.26 | 0.29 | 0.29 | 1.13× | 24.1 | ok |
| mlp_2048x8 | 0.70 | 0.83 | 0.69 | 0.98× | 152.0 | ok |

## Beyond VRAM — DeepMLP baselines

*DeepMLP beyond VRAM* — params 12.35 GB (1.50× VRAM when applicable)

| Approach | Median ms | Peak VRAM GB | Peak host GB | Status |
| --- | ---: | ---: | ---: | --- |
| gpu_eager | OOM | OOM | OOM | OOM |
| tensortorrent | 433.88 | 0.00 | 25.37 | ok |
| tensortorrent_gpu_stream | 553.56 | 7.26 | 38.06 | ok |
| cpu_eager | 428.91 | 0.08 | 43.37 | ok |
| accelerate | 768.38 | 5.38 | 43.37 | ok |

## Beyond VRAM — transformer baselines

*HF transformer beyond VRAM* — params 16.38 GB (1.99× VRAM when applicable)

| Approach | Median ms | Peak VRAM GB | Peak host GB | Status |
| --- | ---: | ---: | ---: | --- |
| gpu_eager | INFEASIBLE | INFEASIBLE | INFEASIBLE | INFEASIBLE |
| cpu_eager | 3152.63 | 0.00 | 16.19 | ok |
| tensortorrent_auto | 1203.34 | 7.39 | 36.90 | ok |
| tensortorrent | 1229.43 | 7.26 | 49.28 | ok |
| accelerate | 1625.45 | 6.64 | 49.28 | ok |

## Memory budget curve

| Budget GiB | Median ms | Throughput iters/s | Transfer GB | GPU compute % | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| 8.0 | 11.75 | 85.08 | NOT MEASURED | 100.0% | ok |
| 6.0 | 11.76 | 85.02 | NOT MEASURED | 100.0% | ok |
| 4.0 | RuntimePlanError: native schedule execution failed: instruction transfer::->region_0:x opcode Transfer region=None tenso | RuntimePlanError: native schedule execution failed: instruction transfer::->region_0:x opcode Transfer region=None tenso | NOT MEASURED | NOT MEASURED | RuntimePlanError: native schedule execution failed: instruction transfer::->region_0:x opcode Transfer region=None tenso |
| 3.0 | 109.13 | 9.16 | NOT MEASURED | 100.0% | ok |
| 2.0 | RuntimePlanError: native schedule execution failed: instruction transfer::->region_0:x opcode Transfer region=None tenso | RuntimePlanError: native schedule execution failed: instruction transfer::->region_0:x opcode Transfer region=None tenso | NOT MEASURED | NOT MEASURED | RuntimePlanError: native schedule execution failed: instruction transfer::->region_0:x opcode Transfer region=None tenso |

## Model size crossover

| Size × VRAM | GPU eager ms | TensorTorrent ms | Status |
| --- | ---: | ---: | --- |
| 0.50 | fits | 13.02 | TT:ok eager:ok |
| 0.75 | fits | 19.27 | TT:ok eager:ok |
| 0.90 | fits | 49.08 | TT:ok eager:ok |
| 1.00 | OOM | RuntimePlanError: native schedule execution failed: instruction transfer::->region_0:x opcode Transfer region=None tenso | TT:RuntimePlanError: native schedule execution failed: instruction transfer::->region_0:x opcode Transfer region=None tenso eager:OOM |
| 1.10 | OOM | RuntimePlanError: native schedule execution failed: instruction transfer::->region_0:x opcode Transfer region=None tenso | TT:RuntimePlanError: native schedule execution failed: instruction transfer::->region_0:x opcode Transfer region=None tenso eager:OOM |
| 1.25 | OOM | 363.72 | TT:ok eager:OOM |
| 1.50 | OOM | 547.37 | TT:ok eager:OOM |

## Heterogeneous placement

| Case | Evidence | Notes |
| --- | --- | --- |
| gpu_plus_cpu_allowed | MEASURED | TT=ok; devices=['cuda_gpu_0'] |
| two_gpu | SUPPORTED_BUT_UNMEASURED | only 1 CUDA device(s) present; multi-GPU not measured |
