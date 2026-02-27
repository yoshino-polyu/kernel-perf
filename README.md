# cupti-perf

CUPTI-based GPU kernel benchmarking utility. Measures accurate GPU kernel execution time using NVIDIA's CUPTI activity tracing, free of CPU launch overhead and host-device synchronization noise.

Extracted from [FlashInfer](https://github.com/flashinfer-ai/flashinfer)'s testing infrastructure.

## Installation

```bash
pip install -e .
```

Requires CUDA 13+ and a CUDA-capable GPU.

## Usage

```python
import torch
from cupti_perf import bench_gpu_time

# Simplest form — pass a zero-arg callable
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")
times = bench_gpu_time(fn=lambda: torch.matmul(a, b))

import numpy as np
print(f"median {np.median(times):.3f} ms  std {np.std(times):.3f} ms")

# With explicit input tensors (needed for cold-L2 flush device inference)
def my_kernel(x, w):
    return x @ w.T

times = bench_gpu_time(
    fn=my_kernel,
    input_args=(a, b),
    cold_l2_cache=True,      # flush L2 before each iteration (default)
    dry_run_time_ms=25,      # target warmup duration
    repeat_iters=20,         # num of measured iterations
)
```

See `cupti_bench_gpu_time_tutorial.md` for detailed documentation on the CUPTI call path and L2 cache flushing strategy.
