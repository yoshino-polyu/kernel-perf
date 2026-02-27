"""
Standalone CUPTI GPU kernel benchmarking utility.

Extracted from flashinfer/testing/utils.py — contains only the CUPTI profiling
path with no CUDA-event or CUDA-graph fallback code.

Requirements:
    pip install cupti-python>=13.0.0   (needs CUDA 13+)
    pip install torch numpy

Usage:
    import torch
    from cupti_bench import bench_gpu_time

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

"""

import bisect
import math
import time
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# CUPTI import — fail fast so callers get a clear error
# ---------------------------------------------------------------------------
try:
    from cupti import cupti
    from importlib.metadata import version as _pkg_version

    _cupti_ver = _pkg_version("cupti-python")
    if int(_cupti_ver.split(".")[0]) < 13:
        raise ImportError(
            f"cupti-python version {_cupti_ver} is too old. "
            "Install >= 13.0.0: pip install -U cupti-python"
        )
except ModuleNotFoundError:
    raise ImportError(
        "cupti-python is not installed. Install it with: pip install -U cupti-python"
    )

from functools import partial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_l2_cache_size(device=None) -> int:
    """Return L2 cache size in bytes for *device* (default: current CUDA device)."""
    if device is None:
        device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).L2_cache_size


def _extract_gpu_tensors(obj) -> List[torch.Tensor]:
    """Recursively collect all CUDA tensors from a nested list/tuple/dict."""
    tensors: list[torch.Tensor] = []
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        tensors.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(_extract_gpu_tensors(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            tensors.extend(_extract_gpu_tensors(v))
    return tensors


def _infer_device(input_args, input_kwargs, default="cuda"):
    """Infer CUDA device from GPU tensors in the provided arguments."""
    if input_kwargs is None:
        input_kwargs = {}
    gpu_tensors = _extract_gpu_tensors(input_args) + _extract_gpu_tensors(input_kwargs)
    return gpu_tensors[0].device if gpu_tensors else default


def _sleep_after_kernel(execution_time_ms: float):
    """Short dynamic sleep (up to 1 s) proportional to kernel execution time."""
    if not math.isinf(execution_time_ms):
        time.sleep(min(execution_time_ms / 200, 1.0))
    else:
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bench_gpu_time(
    fn,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    dry_run_iters: Optional[int] = None,
    repeat_iters: Optional[int] = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    cold_l2_cache: bool = True,
    sleep_after_run: bool = False,
) -> List[float]:
    """Benchmark *fn* on the GPU using CUPTI activity tracing.

    CUPTI (CUDA Profiling Tools Interface) records hardware-level start/end
    timestamps for every GPU kernel, giving the most accurate measurement of
    pure GPU execution time — free of CPU launch overhead and host-device
    synchronization noise.

    Args:
        fn:
            The callable to benchmark.
            * If *input_args*/*input_kwargs* are provided, called as
              ``fn(*input_args, **input_kwargs)``.
            * Otherwise called as ``fn()`` (useful for lambdas that capture
              their own tensors).
        input_args:
            Positional arguments forwarded to *fn*.  Also used to infer the
            CUDA device for L2-flush buffer allocation.
        input_kwargs:
            Keyword arguments forwarded to *fn*.
        dry_run_iters:
            Exact number of warmup iterations (not timed).  When ``None``
            (default), the count is auto-calibrated from *dry_run_time_ms*.
        repeat_iters:
            Exact number of measured iterations.  When ``None`` (default),
            auto-calibrated from *repeat_time_ms*.
        dry_run_time_ms:
            Target wall-clock warmup duration in milliseconds (default 25).
            Ignored when *dry_run_iters* is set explicitly.
        repeat_time_ms:
            Target total wall-clock duration for the measurement phase, in
            milliseconds (default 100).  This is NOT the time of a single
            iteration — it is the budget for ALL measured iterations combined.

            How it works: before the measurement phase, the function runs
            *fn* a few times to estimate a single-iteration execution time
            (call it ``est``).  Then it computes:

                repeat_iters = max(1, int(repeat_time_ms / est))

            So if your kernel takes ~0.5 ms per call and repeat_time_ms=100,
            you get ~200 measured iterations.  If it takes ~10 ms per call,
            you get ~10 iterations.  This auto-scaling ensures:

            - Fast kernels get enough samples for statistical significance.
            - Slow kernels don't waste time running hundreds of iterations.

            If you pass *repeat_iters* explicitly, *repeat_time_ms* is
            ignored entirely.
        cold_l2_cache:
            If ``True`` (default), allocate a scratch buffer of
            ``2 × L2_cache_size`` and zero it before every iteration to evict
            cached data from L2, producing cold-cache timings.
        sleep_after_run:
            If ``True``, insert a short sleep after each measured iteration
            (can reduce thermal throttling on long runs).

    Returns:
        A list of per-iteration GPU kernel execution times in **milliseconds**.
        The length equals *repeat_iters* (auto-calibrated or explicit).

    Raises:
        ImportError: If ``cupti-python >= 13`` is not installed.
        ValueError:  If no GPU kernel activity is recorded for an iteration or
                     if kernel names are inconsistent across iterations.
    """
    if input_kwargs is None:
        input_kwargs = {}

    # ---- L2 flush setup ---------------------------------------------------
    device = _infer_device(input_args, input_kwargs)
    l2_bytes = get_l2_cache_size(device)
    flush_size = (l2_bytes * 2)  # 2× L2 size to ensure full eviction
    flush_buffer = (
        torch.empty(flush_size, device=device, dtype=torch.int8)
        if cold_l2_cache
        else None
    )

    # ---- Wrapper for calling fn -------------------------------------------
    has_args = bool(input_args) or bool(input_kwargs)

    def call_fn():
        if has_args:
            fn(*input_args, **input_kwargs)
        else:
            fn()

    # ---- Auto-calibrate iteration counts ----------------------------------
    measurement_iters = 5
    torch.cuda.synchronize()
    call_fn()  # exclude one-time init overhead
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(measurement_iters):
        if flush_buffer is not None:
            flush_buffer.zero_()
        call_fn()
    end_ev.record()
    torch.cuda.synchronize()
    est_time_ms = start_ev.elapsed_time(end_ev) / measurement_iters

    if dry_run_iters is None:
        dry_run_iters = max(1, int(dry_run_time_ms / est_time_ms))
    if repeat_iters is None:
        repeat_iters = max(1, int(repeat_time_ms / est_time_ms))

    # ---- Warmup (dry run) -------------------------------------------------
    torch.cuda.synchronize()
    for _ in range(dry_run_iters):
        if flush_buffer is not None:
            flush_buffer.zero_()
        call_fn()
    torch.cuda.synchronize()

    # ---- CUPTI buffer callbacks -------------------------------------------
    def _buf_requested():
        """Called by CUPTI when it needs a new activity buffer."""
        return 8 * 1024 * 1024, 0  # 8 MiB buffer, unlimited records

    def _collect_kernel_info(activity):
        copy_kind = activity.copy_kind if activity.kind == cupti.ActivityKind.MEMCPY else 0
        nbytes = activity.bytes if activity.kind in (cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET) else 0
        value = activity.value if activity.kind == cupti.ActivityKind.MEMSET else 0
        name = (
            activity.name
            if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL
            else ("MEMCPY" if activity.kind == cupti.ActivityKind.MEMCPY else "MEMSET")
        )
        return (name, activity.start, activity.end, activity.correlation_id,
                copy_kind, nbytes, value, activity.kind)

    def _buf_completed(launches, kernels, activities):
        """Called by CUPTI when an activity buffer is full or flushed."""
        for act in activities:
            if act.kind in (
                cupti.ActivityKind.CONCURRENT_KERNEL,
                cupti.ActivityKind.MEMCPY,
                cupti.ActivityKind.MEMSET,
            ):
                kernels.append(_collect_kernel_info(act))
            elif act.kind in (
                cupti.ActivityKind.RUNTIME,
                cupti.ActivityKind.DRIVER,
            ):
                launches.append((
                    act.start, act.end, act.correlation_id, act.cbid, act.kind
                ))

    # ---- CUPTI measurement loop ------------------------------------------
    launches: list = []
    kernels: list = []
    iter_timestamps: list[tuple[int, int]] = []

    cupti.activity_enable(cupti.ActivityKind.RUNTIME)
    cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_enable(cupti.ActivityKind.DRIVER)
    cupti.activity_enable(cupti.ActivityKind.MEMCPY)
    cupti.activity_enable(cupti.ActivityKind.MEMSET)
    cupti.activity_register_callbacks(
        _buf_requested, partial(_buf_completed, launches, kernels)
    )

    for _ in range(repeat_iters):
        if flush_buffer is not None:
            flush_buffer.zero_()
        t0 = cupti.get_timestamp()
        call_fn()
        t1 = cupti.get_timestamp()
        torch.cuda.synchronize()
        iter_timestamps.append((t0, t1))
        if sleep_after_run:
            _sleep_after_kernel(est_time_ms)

    cupti.activity_flush_all(0)
    cupti.activity_disable(cupti.ActivityKind.RUNTIME)
    cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_disable(cupti.ActivityKind.DRIVER)
    cupti.activity_disable(cupti.ActivityKind.MEMCPY)
    cupti.activity_disable(cupti.ActivityKind.MEMSET)
    cupti.finalize()

    # ---- Post-process: correlate CPU launches → GPU kernels ---------------
    # Sort launches by start timestamp for binary search
    sorted_launches = sorted(launches, key=lambda l: l[0])
    launch_starts = [l[0] for l in sorted_launches]

    # Build correlation_id → kernel records mapping
    corr_to_kernels: dict[int, list] = {}
    for k in kernels:
        cid = k[3]  # correlation_id
        corr_to_kernels.setdefault(cid, []).append(k)

    def _kernel_signature(k):
        """Stable identity string (excludes per-invocation start/end/corr_id)."""
        return f"{k[0]}_{k[4]}_{k[5]}_{k[6]}_{k[7]}"

    measured_times: list[float] = []
    expected_sigs: set[str] | None = None

    for idx, (t0, t1) in enumerate(iter_timestamps):
        # Binary search for launches in [t0, t1]
        lo = bisect.bisect_left(launch_starts, t0)
        hi = bisect.bisect_right(launch_starts, t1)
        corr_ids = {sorted_launches[i][2] for i in range(lo, hi)}

        # Resolve to GPU kernel records
        iter_kernels = []
        for cid in corr_ids:
            if cid in corr_to_kernels:
                iter_kernels.extend(corr_to_kernels[cid])

        if not iter_kernels:
            raise ValueError(f"No kernel activities recorded for iteration {idx}")

        # Sanity check: kernel set must be identical every iteration
        sigs = {_kernel_signature(k) for k in iter_kernels}
        if expected_sigs is None:
            expected_sigs = sigs
        elif sigs != expected_sigs:
            raise ValueError(
                f"Inconsistent kernels across iterations: {expected_sigs} vs {sigs}"
            )

        # GPU span: first kernel start → last kernel end (nanoseconds → ms)
        span_ms = (max(k[2] for k in iter_kernels) - min(k[1] for k in iter_kernels)) / 1e6
        measured_times.append(span_ms)

    return measured_times
