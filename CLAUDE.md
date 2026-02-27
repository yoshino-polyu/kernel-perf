# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Standalone CUPTI GPU kernel benchmarking utility extracted from FlashInfer's `flashinfer/testing/utils.py`. It provides accurate GPU kernel timing using NVIDIA's CUPTI (CUDA Profiling Tools Interface) activity tracing — no CUDA-event or CUDA-graph fallback paths.

Single package: `cupti_perf/` (public API: `bench_gpu_time()`). Tutorial docs in `cupti_bench_gpu_time_tutorial.md`.

## Dependencies

- `cupti-python >= 13.0.0` (requires CUDA 13+)
- `torch` (PyTorch with CUDA)
- `numpy`

## Install & Run

```bash
pip install -e .
```

```python
from cupti_perf import bench_gpu_time
times = bench_gpu_time(fn=lambda: torch.matmul(a, b))
```

No test suite exists.

## Architecture

`bench_gpu_time()` executes 8 sequential phases:

1. **L2 flush setup** — Allocates `2 × L2_cache_size` buffer for cold-cache benchmarking (when `cold_l2_cache=True`)
2. **Call wrapper** — Wraps `fn` to handle both zero-arg and parametrized calling conventions
3. **Auto-calibration** — Runs 5 iterations to estimate kernel time, then computes `dry_run_iters` and `repeat_iters` from time budgets (`dry_run_time_ms`, `repeat_time_ms`)
4. **Warmup** — Executes `dry_run_iters` unmeasured iterations
5. **CUPTI setup** — Registers buffer callbacks and enables activity tracing for RUNTIME, CONCURRENT_KERNEL, DRIVER, MEMCPY, MEMSET
6. **Measurement loop** — Records CPU timestamps per iteration via `cupti.get_timestamp()`, collects GPU activity records via callbacks
7. **Post-processing** — Binary search to find CPU launches within each iteration's time window, then correlates to GPU kernels via `correlation_id`
8. **Validation & result** — Checks kernel consistency across iterations, computes GPU span `(max_end - min_start) / 1e6` ns→ms

Key design: CPU timestamps bracket iterations only for windowing; actual timing comes from GPU hardware counters in CUPTI activity records. The correlation model links CPU RUNTIME/DRIVER launches to GPU CONCURRENT_KERNEL/MEMCPY/MEMSET records via `correlation_id`.

## Provenance

Every function maps to an original in `flashinfer/testing/utils.py`. The tutorial's Source Provenance table (`cupti_bench_gpu_time_tutorial.md`) provides line-by-line traceability. Intentional omissions: deprecated `l2_flush` params, CUDA-event fallback, CUDA-graph capture.
