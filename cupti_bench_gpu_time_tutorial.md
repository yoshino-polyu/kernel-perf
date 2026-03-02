# CUPTI Path in `bench_gpu_time`: Call Stack Trace and Internals

This document traces the CUPTI profiling path when calling `bench_gpu_time(fn, ...)` from `cupti_perf`.

## Call Stack Overview

```
bench_gpu_time(fn, ...)
│
├── [1] L2 flush setup
│    ├── _infer_device(input_args, input_kwargs)
│    └── get_l2_cache_size(device)  →  allocate 2× L2 flush buffer (if cold_l2_cache=True)
│
├── [2] Wrap fn into call_fn()
│    └── Zero-arg vs parametrized dispatch based on input_args/input_kwargs
│
├── [3] Calibrate iteration counts
│    ├── Run fn() once to exclude first-call overhead
│    ├── Time 5 iterations with CUDA events
│    ├── est_time_ms = total / 5
│    ├── dry_run_iters  = dry_run_time_ms  / est_time_ms  (if not provided)
│    └── repeat_iters   = repeat_time_ms   / est_time_ms  (if not provided)
│
├── [4] Dry-run warmup
│    └── for _ in range(dry_run_iters): [L2 flush] + call_fn()
│
├── [5] Define CUPTI buffer callbacks
│    ├── _buf_requested()          →  returns (8 MiB buffer, 0 max records)
│    └── _buf_completed(...)       →  populates `launches` and `kernels` lists
│         ├── CONCURRENT_KERNEL / MEMCPY / MEMSET  →  kernels[]
│         └── RUNTIME / DRIVER                     →  launches[]
│
├── [6] CUPTI measurement loop
│    ├── cupti.activity_enable(RUNTIME, CONCURRENT_KERNEL, DRIVER, MEMCPY, MEMSET)
│    ├── cupti.activity_register_callbacks(_buf_requested, _buf_completed)
│    ├── for _ in range(repeat_iters):
│    │    ├── [L2 flush via flush_buffer.zero_()]
│    │    ├── t0 = cupti.get_timestamp()
│    │    ├── call_fn()
│    │    ├── t1 = cupti.get_timestamp()
│    │    ├── torch.cuda.synchronize()
│    │    ├── record (t0, t1) into iter_timestamps[]
│    │    └── [sleep if sleep_after_run=True]
│    ├── cupti.activity_flush_all(0)
│    └── cupti.activity_disable(all kinds) + cupti.finalize()
│
└── [7] Post-process: correlate CPU launches to GPU kernels
     ├── Sort launches by start timestamp
     ├── Build correlation_id → kernels mapping
     ├── For each iteration:
     │    ├── Binary search launches within [t0, t1]
     │    ├── Collect correlation_ids → find matching GPU kernels
     │    ├── Validate kernel signatures are consistent across iterations
     │    └── measured_time = (max_end - min_start) / 1e6  (ns → ms)
     └── Return List[float]  (per-iteration GPU times in ms)
```

## Parameters

```python
bench_gpu_time(
    fn,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    dry_run_iters: Optional[int] = None,
    repeat_iters: Optional[int] = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    cold_l2_cache: bool = True,
    sleep_after_run: bool = False,
) -> List[float]
```

### `fn` (required)

The callable to benchmark. Two calling conventions are supported:

- **Zero-arg closure** (most common) — `fn` takes no arguments; all tensors are captured
  from the enclosing scope via Python's closure mechanism. `bench_gpu_time` calls it as `fn()`.
- **Parametrized** — `fn` takes arguments, which are supplied via `input_args`/`input_kwargs`.
  `bench_gpu_time` calls it as `fn(*input_args, **input_kwargs)`.

The dispatch logic is:

```python
# inside bench_gpu_time — simplified
has_args = bool(input_args) or bool(input_kwargs)

def call_fn():
    if has_args:
        fn(*input_args, **input_kwargs)   # parametrized
    else:
        fn()                               # zero-arg closure
```

`fn` may launch one or more GPU kernels per call. CUPTI records all of them and reports the
total GPU span (first kernel start to last kernel end) as the per-iteration time.

#### Closure style (recommended for most cases)

The most common pattern is to define a zero-arg wrapper function (or lambda) that captures
all tensors from the surrounding scope. This is how real benchmarking scripts typically look:

```python
# Real-world example — from bench_mla_sparse.py
# The closure captures Q_nope_abs_s, Q_pe_s, KV_cache, etc. from
# the enclosing scope.  bench_gpu_time receives a zero-arg fn and
# calls it as fn() — no input_args or input_kwargs needed.

def stream_kern_fn():
    return mla_attn_triton(
        Q_nope_abs_s, Q_pe_s, KV_cache, PE_cache,
        cu_q, cu_k, sm_scale,
        n_streaming_heads=n_streaming,
        window_size=W, n_sink=n_sink,
    )

stream_ms = float(np.median(
    bench_gpu_time(stream_kern_fn, dry_run_iters=20, repeat_iters=20)
))
```

The same pattern works with a lambda for simple one-liners:

```python
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")
times = bench_gpu_time(fn=lambda: torch.matmul(a, b))
```

In both cases, `input_args` and `input_kwargs` are left at their defaults (empty), so
`bench_gpu_time` just calls `fn()` directly.

#### Parametrized style

Alternatively, you can pass a function that takes arguments and supply them via `input_args`
/ `input_kwargs`. This is functionally equivalent to the closure style — it's syntactic sugar
for separating the function from its data:

```python
def my_kernel(x, w):
    return x @ w.T

# These two calls are equivalent:
times = bench_gpu_time(fn=my_kernel, input_args=(a, b))
times = bench_gpu_time(fn=lambda: my_kernel(a, b))
```

The parametrized style has one practical advantage: `bench_gpu_time` can inspect
`input_args`/`input_kwargs` to find CUDA tensors and infer which GPU device to allocate the
L2 flush buffer on (see `input_args` / `input_kwargs` below). With the closure style, the
tensors are invisible to `bench_gpu_time`, so it falls back to `"cuda"` (the current default
device) — which is fine for single-GPU setups.

### `input_args` / `input_kwargs`

| | Type | Default |
|---|---|---|
| `input_args` | `Tuple` | `()` |
| `input_kwargs` | `Optional[dict]` | `None` |

Positional and keyword arguments forwarded to `fn` on every invocation. **These are only
used with the parametrized calling style.** When using the closure style (zero-arg `fn`),
leave them at their defaults — `bench_gpu_time` will call `fn()` with no arguments.

#### Positional vs keyword arguments — quick primer

In Python, function arguments can be passed by **position** or by **name** (keyword):

```python
def attention(q, k, v, sm_scale=1.0, causal=False):
    ...

# Positional: matched left-to-right by position
attention(q_tensor, k_tensor, v_tensor, 0.5, True)
#         ↑ q      ↑ k      ↑ v      ↑ sm_scale  ↑ causal

# Keyword: matched by name (order doesn't matter)
attention(q=q_tensor, k=k_tensor, v=v_tensor, causal=True, sm_scale=0.5)

# Mixed: positional first, then keyword
attention(q_tensor, k_tensor, v_tensor, causal=True)
```

`input_args` maps to the positional arguments (a tuple), and `input_kwargs` maps to the
keyword arguments (a dict). Here is a concrete side-by-side showing how the same kernel
call can be expressed with each:

```python
def attention(q, k, v, sm_scale=1.0, causal=False):
    ...

q = torch.randn(32, 64, 128, device="cuda")
k = torch.randn(32, 64, 128, device="cuda")
v = torch.randn(32, 64, 128, device="cuda")

# ── Style 1: input_args only (all positional) ──────────────────────
bench_gpu_time(
    fn=attention,
    input_args=(q, k, v, 0.5, True),
)
# bench_gpu_time calls:  attention(q, k, v, 0.5, True)

# ── Style 2: input_kwargs only (all keyword) ───────────────────────
bench_gpu_time(
    fn=attention,
    input_kwargs={"q": q, "k": k, "v": v, "sm_scale": 0.5, "causal": True},
)
# bench_gpu_time calls:  attention(q=q, k=k, v=v, sm_scale=0.5, causal=True)

# ── Style 3: mixed (positional + keyword) ──────────────────────────
bench_gpu_time(
    fn=attention,
    input_args=(q, k, v),
    input_kwargs={"sm_scale": 0.5, "causal": True},
)
# bench_gpu_time calls:  attention(q, k, v, sm_scale=0.5, causal=True)

# ── Style 4: closure (most common — no input_args/input_kwargs) ────
bench_gpu_time(
    fn=lambda: attention(q, k, v, sm_scale=0.5, causal=True),
)
# bench_gpu_time calls:  (lambda)()
#   which internally calls:  attention(q, k, v, sm_scale=0.5, causal=True)
```

All four styles produce the same kernel execution. The dispatch inside `bench_gpu_time` is:

```
Closure style (input_args/input_kwargs empty):
  bench_gpu_time(fn=stream_kern_fn)
       │
       └──  call_fn()  →  stream_kern_fn()
                           ╰── tensors captured from enclosing scope

Parametrized style (input_args/input_kwargs provided):
  bench_gpu_time(fn=my_kernel, input_args=(a, b))
       │
       └──  call_fn()  →  my_kernel(*input_args, **input_kwargs)
                                     ╰── tensors passed explicitly
```

**Secondary role — device inference:** When `cold_l2_cache=True`, the L2 flush buffer must be
allocated on the same GPU as the profiled kernel's data. The implementation recursively scans
`input_args` and `input_kwargs` for CUDA tensors (`_extract_gpu_tensors()`) and uses the first
one's `.device` to place the flush buffer. If no GPU tensors are found (e.g. zero-arg closure),
it falls back to `"cuda"` (current default device).

This matters for multi-GPU setups. If your kernel runs on `cuda:1` and you use the closure
style, the flush buffer would be allocated on the wrong GPU. Two fixes:

```python
# Fix 1: use input_args so device inference works
times = bench_gpu_time(fn=my_kernel, input_args=(a_on_gpu1, b_on_gpu1))

# Fix 2: set the default device before calling bench_gpu_time
with torch.cuda.device(1):
    times = bench_gpu_time(fn=stream_kern_fn)
```

For single-GPU machines (the common case), the fallback to `"cuda"` is always correct and
you don't need to worry about this.

### `dry_run_iters` / `dry_run_time_ms`

| | Type | Default |
|---|---|---|
| `dry_run_iters` | `Optional[int]` | `None` |
| `dry_run_time_ms` | `int` | `25` |

Control the **warmup phase** — unmeasured iterations that run before CUPTI tracing begins.
Warmup is important because CUDA kernels often have higher latency on the first few invocations
(JIT compilation, memory pool initialization, cache warm-up for compute-bound kernels, etc.).

**How they interact:**

- If `dry_run_iters` is set explicitly, it is used directly and `dry_run_time_ms` is ignored.
- If `dry_run_iters` is `None`, it is auto-calibrated:

```
est_time_ms = (time for 5 calibration runs) / 5
dry_run_iters = max(1, int(dry_run_time_ms / est_time_ms))
```

For example, with the default `dry_run_time_ms=25`:
- A 0.1 ms kernel → ~250 warmup iterations
- A 5 ms kernel → ~5 warmup iterations
- A 50 ms kernel → 1 warmup iteration

**When to set explicitly:** If you know a kernel needs many warmups (e.g., torch.compile with
first-call tracing), set `dry_run_iters` directly to a high value. If warmup time doesn't
matter and you just want sufficient stabilization, leave both at their defaults.

### `repeat_iters` / `repeat_time_ms`

| | Type | Default |
|---|---|---|
| `repeat_iters` | `Optional[int]` | `None` |
| `repeat_time_ms` | `int` | `100` |

Control the **measurement phase** — the number of timed iterations whose GPU kernel spans
are recorded and returned.

**How they interact:**

- If `repeat_iters` is set explicitly, it is used directly and `repeat_time_ms` is ignored.
- If `repeat_iters` is `None`, it is auto-calibrated:

```
est_time_ms = (time for 5 calibration runs) / 5
repeat_iters = max(1, int(repeat_time_ms / est_time_ms))
```

For example, with the default `repeat_time_ms=100`:
- A 0.5 ms kernel → ~200 measured iterations
- A 10 ms kernel → ~10 measured iterations
- A 200 ms kernel → 1 measured iteration

This auto-scaling ensures fast kernels get enough samples for statistical significance
while slow kernels don't waste time running hundreds of iterations.

**Tradeoff — more iterations vs. wall-clock time:**

| Scenario | Recommendation |
|---|---|
| Quick A/B comparison of two kernels | Default `repeat_time_ms=100` (~0.1 s measurement) |
| Publication-quality numbers, low variance | `repeat_time_ms=1000` or `repeat_iters=500+` |
| Very slow kernel (>50 ms), just need a ballpark | `repeat_iters=5` |

The return value is always a `List[float]` of length `repeat_iters` — one GPU time (ms) per
measured iteration. Compute statistics yourself:

```python
import numpy as np
times = bench_gpu_time(fn=my_kernel, repeat_time_ms=500)
print(f"median={np.median(times):.3f} ms, std={np.std(times):.3f} ms, n={len(times)}")
```

### `cold_l2_cache`

| Type | Default |
|---|---|
| `bool` | `True` |

If `True`, allocate a scratch buffer of `2 × L2_cache_size` bytes and zero it on the GPU
before every measured iteration (and during warmup and calibration). This evicts all prior
data from L2, so the profiled kernel starts with a cold cache and must fetch everything
from DRAM.

See [L2 Cache Flush](#l2-cache-flush) below for a full explanation of the mechanism,
pros/cons, and when to use `True` vs `False`.

### `sleep_after_run`

| Type | Default |
|---|---|
| `bool` | `False` |

If `True`, insert a short CPU-side sleep after each measured iteration. The sleep duration
scales with the estimated kernel execution time:

```python
sleep_seconds = min(est_time_ms / 200, 1.0)   # capped at 1 second
```

For a 1 ms kernel this is ~5 microseconds; for a 100 ms kernel it is ~0.5 seconds.

**Why this exists:** When running hundreds of consecutive GPU iterations with no pauses,
the GPU has zero idle time and can heat up, triggering thermal throttling. The resulting
clock frequency drop causes later iterations to measure slower than earlier ones, adding
systematic drift to the timing data. A short sleep between iterations gives the GPU a
moment to cool, reducing this effect.

**When to use:**

| Scenario | Recommendation |
|---|---|
| Normal benchmarking, moderate iteration count | `False` (default) — throttling is unlikely |
| Long-running sweeps (many kernels × many configs) | `True` — prevents gradual heat buildup |
| Kernel close to GPU TDP (large GEMM, long-running) | `True` — these are most throttle-prone |
| Latency-sensitive: minimize total wall-clock bench time | `False` — sleep adds overhead |

### Return Value

`List[float]` — per-iteration GPU kernel execution times in **milliseconds**. Length equals
`repeat_iters` (whether auto-calibrated or explicit).

Each entry is the GPU span for one iteration: `(max_kernel_end - min_kernel_start) / 1e6`
nanoseconds → milliseconds.

#### Single-kernel case

When `fn` launches exactly one GPU kernel, the span is simply that kernel's execution time:

```
GPU timeline:  ──────[  kernel A  ]──────────────────
                     ^            ^
                   start         end

span = end - start
```

#### Multi-kernel case

When `fn` launches multiple GPU kernels per call (e.g. a fused operator that decomposes into
several sub-kernels, or a function that calls `matmul` followed by `add`), the span is
measured from the **earliest kernel start** to the **latest kernel end** across all kernels
in that iteration:

```
GPU timeline:  ──[kernel A]───gap───[kernel B]──[kernel C]──
                 ^                                        ^
              min(start)                             max(end)

span = max(end) - min(start)
       ╰────── includes kernel execution + inter-kernel gaps ──────╯
```

The idle gaps between consecutive kernels **are included** in the reported time. This is
intentional: in real workloads, these gaps (caused by kernel launch latency, memory
dependencies, or stream synchronization) are part of the end-to-end cost of calling `fn`.

For comparison, an alternative metric would sum each kernel's individual duration and exclude
the gaps — but that would undercount the true wall-clock GPU cost:

```
GPU timeline:  ──[kernel A]───gap───[kernel B]──[kernel C]──

Reported (span):          |◄──────── 1.2 ms ────────►|
Sum-of-kernels (not used): 0.3 ms + 0.4 ms + 0.3 ms = 1.0 ms
                           (ignores 0.2 ms of gap time)
```

Concretely, this is implemented in post-processing as:

```python
# iter_kernels: all CUPTI activity records attributed to this iteration
# k[1] = activity.start, k[2] = activity.end  (nanoseconds)
span_ms = (max(k[2] for k in iter_kernels) - min(k[1] for k in iter_kernels)) / 1e6
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

#### Why `buffer.zero_()` flushes L2 — even though it's unrelated to the profiled kernel

The GPU has a single, shared L2 cache sitting between all SMs and device DRAM.
Every GPU memory access — from *any* kernel — goes through this same L2. The
cache has a fixed capacity (e.g. 50 MiB on H100) and a replacement policy: when
new data arrives and the cache is full, the oldest/least-recently-used lines are
evicted.

`buffer.zero_()` launches a GPU-side memset kernel that writes to `2 × L2_cache_size`
bytes of device memory. Because the write footprint exceeds the entire L2 capacity,
every cache line in L2 gets replaced with flush-buffer data by the time the memset
finishes. After this point, none of the profiled kernel's input or output data
remains in L2 — the next call to `fn()` starts with an all-miss ("cold") L2
cache and must fetch everything from DRAM.

The flush is intentionally **unrelated** to the profiled kernel: the goal is to
pollute L2 with irrelevant data so that the kernel's own data is guaranteed to be
evicted. It does not matter what the flush buffer contains — only that the write
is large enough to cycle through all L2 lines.

```
Before flush:
  L2 cache: [kernel input A] [kernel output B] [other data ...]
                                                     ← L2 capacity →

buffer.zero_()  writes 2× L2 capacity of zeros:
  L2 cache: [flush buf page N] [flush buf page N+1] [flush buf ...]
                                                     ← L2 capacity →

Next fn() call:
  Every load/store → L2 miss → fetch from DRAM
```

Concretely, the implementation is:

```python
flush_buffer = torch.empty(l2_bytes * 2, device=device, dtype=torch.int8)
# ...
for _ in range(repeat_iters):
    flush_buffer.zero_()          # GPU memset — evicts all L2 lines
    # ... then measure fn() ...
```

The flush runs on the same CUDA stream before `fn()`, so it is guaranteed to
complete before the profiled kernel starts. CUPTI timestamps bracket `fn()` after
the flush, so the memset time is **not** included in the reported kernel time.

When `cold_l2_cache=True`:
- A `torch.int8` buffer of size `2 × L2_cache_size` is allocated once
- Before each iteration, `buffer.zero_()` writes zeros to the entire buffer on GPU, evicting all prior L2 contents
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
