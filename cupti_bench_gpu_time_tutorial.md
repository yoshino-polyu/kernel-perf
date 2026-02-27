# CUPTI Path in `bench_gpu_time`: Call Stack Trace and Internals

This document traces the CUPTI profiling path when calling `bench_gpu_time(fn, enable_cupti=True, ...)` from `flashinfer/testing/utils.py`.

## Call Stack Overview

```
bench_gpu_time(fn, enable_cupti=True, ...)
│
├── [1] Resolve cold_l2_cache (handle deprecated l2_flush params)
│
└── [2] bench_gpu_time_with_cupti(fn, ...)
     │
     ├── [2.1] Determine L2 flush configuration
     │    ├── _infer_device_from_tensors(input_args, input_kwargs)
     │    └── get_l2_cache_size(device)  →  props.L2_cache_size
     │
     ├── [2.2] Import & validate CUPTI
     │    ├── from cupti import cupti
     │    └── Check cupti-python version >= 13.0.0
     │    └── (On failure: fallback to bench_gpu_time_with_cuda_event)
     │
     ├── [2.3] Define CUPTI buffer callbacks
     │    ├── func_buffer_requested()        →  returns (8MB buffer, 0 max records)
     │    └── func_buffer_completed(...)     →  populates `launches` and `kernels` lists
     │         ├── CONCURRENT_KERNEL / MEMCPY / MEMSET  →  kernels[]
     │         └── RUNTIME / DRIVER                     →  launches[]
     │
     ├── [2.4] Prepare runner
     │    ├── Allocate L2 flush buffer (if cold_l2_cache=True)
     │    └── (Optional) Capture CUDA graph if use_cuda_graph=True
     │
     ├── [2.5] Calibrate iteration counts
     │    ├── Run fn() once to exclude first-call overhead
     │    ├── Time 5 iterations with CUDA events
     │    ├── estimated_kernel_execution_time = total / 5
     │    ├── dry_run_iters  = dry_run_time_ms  / estimated_time  (if not provided)
     │    └── repeat_iters   = repeat_time_ms   / estimated_time  (if not provided)
     │
     ├── [2.6] Dry-run warmup
     │    └── for _ in range(dry_run_iters): [L2 flush] + runner()
     │
     ├── [2.7] CUPTI measurement loop
     │    ├── cupti.activity_enable(RUNTIME, CONCURRENT_KERNEL, DRIVER, MEMCPY, MEMSET)
     │    ├── cupti.activity_register_callbacks(buffer_requested, buffer_completed)
     │    ├── for _ in range(repeat_iters):
     │    │    ├── [L2 flush via buffer.zero_()]
     │    │    ├── start_cpu = cupti.get_timestamp()
     │    │    ├── runner()
     │    │    ├── end_cpu = cupti.get_timestamp()
     │    │    ├── torch.cuda.synchronize()
     │    │    └── record (start_cpu, end_cpu) into iter_timestamps[]
     │    ├── cupti.activity_flush_all(0)
     │    └── cupti.activity_disable(all kinds) + cupti.finalize()
     │
     └── [2.8] Post-process: correlate CPU launches to GPU kernels
          ├── Sort launches by start timestamp
          ├── Build correlation_id → kernels mapping
          ├── For each iteration:
          │    ├── Binary search launches within [start_cpu, end_cpu]
          │    ├── Collect correlation_ids → find matching GPU kernels
          │    ├── Validate kernel names are consistent across iterations
          │    └── measured_time = (max_end - min_start) / 1e6  (ns → ms)
          └── Return List[float]  (per-iteration GPU times in ms)
```

## Key Data Structures

### CUPTI Activity Records

Each CUPTI activity record has different fields depending on its `kind`:

| Kind | Fields Used | Description |
|------|------------|-------------|
| `CONCURRENT_KERNEL` | `name`, `start`, `end`, `correlation_id` | GPU kernel execution |
| `MEMCPY` | `start`, `end`, `correlation_id`, `copy_kind`, `bytes` | Device memory copy |
| `MEMSET` | `start`, `end`, `correlation_id`, `value`, `bytes` | Device memory set |
| `RUNTIME` | `start`, `end`, `correlation_id`, `cbid` | CUDA runtime API call (CPU side) |
| `DRIVER` | `start`, `end`, `correlation_id`, `cbid` | CUDA driver API call (CPU side) |

### Correlation Model

CUPTI links CPU-side API calls to GPU-side kernel executions via `correlation_id`:

```
CPU timeline:  ──[RUNTIME call, corr_id=42]────────────
                         │
                         ▼ (launches)
GPU timeline:  ──────────[CONCURRENT_KERNEL, corr_id=42]──
```

The post-processing uses this to attribute GPU kernels to specific benchmark iterations:

1. Each iteration records a CPU time window `[start_cpu, end_cpu]`
2. Binary search finds RUNTIME/DRIVER launches within that window
3. Their `correlation_id`s are used to look up the corresponding GPU kernel activities
4. The GPU span = `max(end) - min(start)` across all kernels in that iteration

### L2 Cache Flush

When `cold_l2_cache=True`:
- A `torch.int8` buffer of size `2 × L2_cache_size` is allocated
- Before each iteration, `buffer.zero_()` writes zeros to the entire buffer, evicting kernel data from L2
- The CUPTI path uses this L2-flush strategy (not rotating buffers, which is the CUDA-graph-only strategy)

#### Pros of `cold_l2_cache=True`

- **Realistic for memory-bound kernels.** In production LLM serving, decode attention, small GEMMs, and element-wise ops almost never find their inputs already sitting in L2 from a previous invocation. Flushing L2 between iterations reproduces this real-world scenario, so the measured bandwidth and latency numbers are representative of actual deployment.
- **Reproducible.** Without flushing, whether data happens to be in L2 depends on iteration order, problem size relative to L2, and other concurrent work. Flushing removes this variable, giving lower variance across runs.
- **Prevents misleadingly fast numbers.** A kernel that fits entirely in L2 (e.g., small batch decode with short KV) can appear 2-5x faster with a warm cache than it ever will in practice, because in production other kernels run between invocations and evict the data.

#### Cons of `cold_l2_cache=True`

- **Overly pessimistic for compute-bound kernels.** For large matrix multiplications (e.g., 4096x4096 FP16 GEMM), the arithmetic intensity is high enough that L2 hit rate barely affects the total time. Flushing adds a pointless `buffer.zero_()` memset before each iteration — it doesn't change the measured kernel time but wastes wall-clock benchmarking time.
- **Overly pessimistic when L2 reuse is real.** Some fused operators (e.g., attention + residual add, or back-to-back GEMMs in MoE) are designed so that the second kernel reads data the first kernel just wrote, and L2 naturally caches it. Flushing destroys this inter-kernel locality and underestimates the fused operator's actual throughput.
- **Extra memory allocation.** The flush buffer is `2 × L2_cache_size` (e.g., 100 MiB on H100 with 50 MiB L2). On a memory-constrained GPU this can matter.
- **Adds overhead to each iteration.** The `buffer.zero_()` call itself takes GPU time. CUPTI brackets the measurement *after* the flush, so the flush time is not included in the reported kernel time — but it does extend the total benchmarking wall-clock time proportionally to `repeat_iters`.

#### Rule of thumb

| Kernel type | Recommendation |
|---|---|
| Memory-bound (decode attention, small batch GEMV, elementwise) | `cold_l2_cache=True` |
| Compute-bound (large GEMM, prefill attention with long seq) | `cold_l2_cache=False` |
| Fused pipelines where inter-kernel L2 reuse is intentional | `cold_l2_cache=False` |

## Timing Accuracy Notes

- **CUPTI timestamps** are in nanoseconds, sourced from GPU hardware counters
- **`cupti.get_timestamp()`** returns a CPU-side timestamp used only to bracket which RUNTIME launches belong to which iteration — the actual kernel time comes from GPU-side `activity.start` / `activity.end`
- The measured time is the **wall-clock GPU span** (from first kernel start to last kernel end within one iteration), which captures any gaps between consecutive sub-kernels

## Source Provenance of `cupti_bench.py`

Every function and code block in `cupti_bench.py` is extracted from
`flashinfer/testing/utils.py`. The table below maps each piece to its origin.
"Renamed" means the logic is identical but the function/variable name was
shortened for the standalone file. "Inlined" means several small original
helpers were merged into one function with the same logic.

| `cupti_bench.py` | Origin in `flashinfer/testing/utils.py` | Change |
|---|---|---|
| `get_l2_cache_size()` (L71-75) | `get_l2_cache_size()` (L38-51) | Condensed docstring |
| `_extract_gpu_tensors()` (L78-89) | `_extract_gpu_tensors()` (L72-92) | Condensed docstring |
| `_infer_device()` (L92-97) | `_infer_device_from_tensors()` (L213-230) | Renamed |
| `_sleep_after_kernel()` (L100-105) | `sleep_after_kernel_run()` (L438-453) | Renamed; `np.min([x,y])` → `min(x,y)` |
| CUPTI import & version check (L49-62) | `bench_gpu_time_with_cupti()` (L1010-1018) | Moved to module level; raises `ImportError` instead of falling back |
| `_buf_requested()` (L241-243) | `func_buffer_requested()` (L1062-1065) | Renamed |
| `_collect_kernel_info()` (L245-255) | `set_kernel_name()` (L1067-1073) + `get_bytes()` (L1075-1079) + `get_copy_kind()` (L1081-1085) + `get_value()` (L1087-1091) + `collect_kernel_info()` (L1093-1103) | Inlined four helpers into one function; identical field access and tuple layout |
| `_buf_completed()` (L257-272) | `func_buffer_completed()` (L1105-1131) | Renamed |
| L2 flush setup (L191-199) | L2 flush in `bench_gpu_time_with_cupti()` (L1002-1007, L1142-1145) | Computed inline using `get_l2_cache_size()` directly in bytes instead of converting through `_l2_flush_size_mb`; equivalent: both allocate `2 × L2_cache_size` bytes |
| `call_fn()` (L204-208) | `call_fn()` (L1133-1140) | Identical |
| Auto-calibrate iteration counts (L210-230) | Calibration block (L1166-1188) | `runner()` → `call_fn()`; `start_event`/`end_event` → `start_ev`/`end_ev`; `estimated_kernel_execution_time` → `est_time_ms` |
| Warmup dry-run loop (L232-238) | Dry-run loop (L1190-1196) | `_do_l2_flush` / `buffer` → `flush_buffer is not None`; `runner()` → `call_fn()` |
| CUPTI enable/register/loop (L274-297) | CUPTI measurement (L1198-1219) | `runner()` → `call_fn()`; `start_cpu`/`end_cpu` → `t0`/`t1` |
| CUPTI disable/finalize (L299-305) | CUPTI teardown (L1220-1226) | Identical |
| `_kernel_signature()` (L318-320) | `generate_kernel_string()` (L1228-1230) | Renamed |
| Post-process binary search loop (L322-351) | Post-process block (L1232-1280) | `corr_id_to_kernels` → `corr_to_kernels`; `kernel_names` → `expected_sigs`; `left_idx`/`right_idx` → `lo`/`hi`; `import bisect` moved to top-level |
| **Omitted**: deprecated `l2_flush` / `l2_flush_size_mb` / `l2_flush_device` params | (L916-918, L990-1000) | Removed — deprecated code not carried over |
| **Omitted**: CUPTI-unavailable fallback to `bench_gpu_time_with_cuda_event` / `bench_gpu_time_with_cudagraph` | (L1020-1059) | Removed — module-level import raises `ImportError` instead |
| **Omitted**: `use_cuda_graph` param and CUDA graph capture block | (L920, L1147-1164) | Removed per user request |
