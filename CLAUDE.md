# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Standalone CUPTI GPU kernel benchmarking utility extracted from FlashInfer's `flashinfer/testing/utils.py`. It provides accurate GPU kernel timing using NVIDIA's CUPTI (CUDA Profiling Tools Interface) activity tracing — no CUDA-event or CUDA-graph fallback paths.

Also includes tensor dumper/loader tools (adapted from SGLang's `sglang.srt.debug_utils.dumper` and `dump_loader`) for capturing and replaying kernel inputs in isolated performance unit tests.

Package: `cupti_perf/` with three public modules:
- `bench_gpu_time()` — CUPTI-based GPU kernel benchmarking
- `TensorDumper` — dump kernel input tensors and scalars to disk
- `TensorLoader` — load dumped tensors, restoring original shape/dtype/strides

Tutorial docs:
- `cupti_bench_gpu_time_tutorial.md` — CUPTI benchmarking deep-dive
- `tensor_dump_load_tutorial.md` — dumper/loader usage and MoE kernel examples

## Dependencies

- `cupti-python >= 13.0.0` (requires CUDA 13+)
- `torch` (PyTorch with CUDA)
- `numpy`

## Install & Run

```bash
pip install cupti-python>=13.0.0
pip install -e .
```

```python
from cupti_perf import bench_gpu_time
times = bench_gpu_time(fn=lambda: torch.matmul(a, b))
```

```python
from cupti_perf import TensorDumper, TensorLoader
dumper = TensorDumper("/tmp/kernel_dumps")
dumper.dump_kernel_inputs("my_kernel", args={"x": x, "w": w}, scalars={"transpose_B": True})

loader = TensorLoader("/tmp/kernel_dumps")
inputs = loader.load_kernel_inputs(loader.list_dumps()[0].name, device="cuda")
```

No test suite exists.

## Architecture

### bench_gpu_time()

Executes 8 sequential phases:

1. **L2 flush setup** — Allocates `2 × L2_cache_size` buffer for cold-cache benchmarking (when `cold_l2_cache=True`)
2. **Call wrapper** — Wraps `fn` to handle both zero-arg and parametrized calling conventions
3. **Auto-calibration** — Runs 5 iterations to estimate kernel time, then computes `dry_run_iters` and `repeat_iters` from time budgets (`dry_run_time_ms`, `repeat_time_ms`)
4. **Warmup** — Executes `dry_run_iters` unmeasured iterations
5. **CUPTI setup** — Registers buffer callbacks and enables activity tracing for RUNTIME, CONCURRENT_KERNEL, DRIVER, MEMCPY, MEMSET
6. **Measurement loop** — Records CPU timestamps per iteration via `cupti.get_timestamp()`, collects GPU activity records via callbacks
7. **Post-processing** — Binary search to find CPU launches within each iteration's time window, then correlates to GPU kernels via `correlation_id`
8. **Validation & result** — Checks kernel consistency across iterations, computes GPU span `(max_end - min_start) / 1e6` ns→ms

Key design: CPU timestamps bracket iterations only for windowing; actual timing comes from GPU hardware counters in CUPTI activity records. The correlation model links CPU RUNTIME/DRIVER launches to GPU CONCURRENT_KERNEL/MEMCPY/MEMSET records via `correlation_id`.

### TensorDumper

Saves kernel inputs to a structured directory:
- Each `dump_kernel_inputs()` call creates a subdirectory
- One `.pt` file per tensor (via `torch.save`), preserving the full storage
- A `manifest.pt` with per-tensor metadata (shape, dtype, stride, device, contiguity) and scalar arguments
- View tensors are cloned before saving to ensure complete storage is captured

### TensorLoader

Restores dumped kernel inputs:
- `list_dumps()` scans the dump directory and returns metadata entries
- `load_kernel_inputs()` loads all tensors and scalars, optionally restoring original strides via `torch.as_strided` and moving to a target device
- Handles dtype conversion for FP8 and other exotic types

## File Layout

```
cupti_perf/
  __init__.py          — public API: bench_gpu_time, TensorDumper, TensorLoader
  tensor_dumper.py     — TensorDumper, TensorMeta, KernelDumpManifest
  tensor_loader.py     — TensorLoader, LoadedKernelInputs, DumpEntry
```

## Provenance

- `bench_gpu_time` maps to `flashinfer/testing/utils.py`. The tutorial's Source Provenance table provides line-by-line traceability.
- `TensorDumper` / `TensorLoader` are adapted from SGLang's `sglang.srt.debug_utils.dumper` and `sglang.srt.debug_utils.dump_loader`. The original tools include distributed-aware features (multi-rank, HTTP control, ZMQ RPC) that are stripped in this standalone version, keeping only the core dump/load logic needed for single-process performance testing.
