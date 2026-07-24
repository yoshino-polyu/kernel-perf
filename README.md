# cupti-perf

CUPTI-based GPU kernel benchmarking utility with tensor dump/load tools for performance unit testing. Measures accurate GPU kernel execution time using NVIDIA's CUPTI activity tracing, free of CPU launch overhead and host-device synchronization noise.

Extracted from [FlashInfer](https://github.com/flashinfer-ai/flashinfer)'s testing infrastructure. Tensor dumper/loader adapted from [SGLang](https://github.com/sgl-project/sglang)'s debug utilities.

## Installation

> **Agent instructions** — follow these steps in order. Each step has a **check** command and an **install** command. Only run the install command if the check fails or returns nothing.

**Prerequisites:** CUDA 13+ and a CUDA-capable GPU. Verify with `nvidia-smi` — if it fails, stop and report the missing driver to the user.

### Step 1 — check for `cupti-python == 13.0.1`

Check:
```bash
python -c "import cupti; print(cupti.__version__)"
```

- If it prints `13.0.1` → skip to Step 2.
- If it prints a different version, or raises `ModuleNotFoundError` → run the install:

```bash
pip install "cupti-python==13.0.1"
```

Then re-run the check to confirm.

### Step 2 — check for `cupti_perf` (this package)

Check:
```bash
python -c "import cupti_perf; print(cupti_perf.__file__)"
```

- If it prints a path inside this repo (e.g. `.../kernel-perf/cupti_perf/__init__.py`) → installation complete.
- If it raises `ModuleNotFoundError`, or points somewhere outside this repo → run the install from the repo root:

```bash
pip install -e .
```

Then re-run the check to confirm.

### Step 3 — smoke test

```bash
python -c "import torch; from cupti_perf import bench_gpu_time; \
a=torch.randn(1024,1024,device='cuda'); b=torch.randn(1024,1024,device='cuda'); \
print(bench_gpu_time(fn=lambda: torch.matmul(a,b))[:3])"
```

Should print a short list of floats (kernel times in ms). If it errors on CUPTI activity registration, the CUDA toolkit version is likely < 13 — report this to the user rather than downgrading `cupti-python`.

## Usage

### GPU Kernel Benchmarking

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

### Tensor Dumper — Capture Kernel Inputs

```python
from cupti_perf import TensorDumper

dumper = TensorDumper("/tmp/kernel_dumps")

# Dump all inputs to a kernel call
dumper.dump_kernel_inputs(
    kernel_name="Mgemm_mxfp8",
    args={
        "x_fp8": x_fp8,         # [M, K] float8_e4m3fn
        "x_scale": x_scale,     # [M, K//32] uint8
        "w_fp8": w_fp8,         # [E, N, K] float8_e4m3fn
        "w_scale": w_scale,     # [E, N, K//32] uint8
        "cnt": slice_offs,      # [E+1] int32
    },
    scalars={"max_M_per_E": max_tokens_per_expert},
    tag="fwd_w13",
)
```

### Tensor Loader — Replay Kernel Inputs

```python
from cupti_perf import TensorLoader

loader = TensorLoader("/tmp/kernel_dumps")

# List available dumps
for entry in loader.list_dumps():
    print(entry)

# Load a specific dump
inputs = loader.load_kernel_inputs(
    "Mgemm_mxfp8__tag=fwd_w13__idx=0001",
    device="cuda",
)
print(inputs.summary())

# Use loaded tensors for benchmarking
from cupti_perf import bench_gpu_time
times = bench_gpu_time(
    fn=Mgemm_mxfp8,
    input_args=(
        inputs.tensors["x_fp8"],
        inputs.tensors["x_scale"],
        inputs.tensors["w_fp8"],
        inputs.tensors["w_scale"],
        inputs.tensors["cnt"],
        inputs.scalars["max_M_per_E"],
    ),
)
```

## Documentation

- `cupti_bench_gpu_time_tutorial.md` — CUPTI benchmarking internals
- `tensor_dump_load_tutorial.md` — Tensor dumper/loader usage guide with MoE kernel examples
