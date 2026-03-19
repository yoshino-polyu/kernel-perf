# Tensor Dump/Load Tutorial

This tutorial covers the **TensorDumper** and **TensorLoader** tools in `cupti_perf`. These tools let you capture the exact inputs to GPU kernel functions, save them to disk, and reload them later for isolated performance benchmarking — without needing to run the full model.

## Motivation

When benchmarking GPU kernels like `Mgemm_mxfp8` or `_Mgemm` inside a MoE (Mixture-of-Experts) forward pass, you need realistic input tensors. Running the full model just to feed a single kernel is expensive and introduces noise. The dump/load workflow solves this:

1. **Dump** — Instrument one model run to capture kernel inputs to disk.
2. **Load** — In a separate script, reload those tensors and benchmark the kernel in isolation.

This gives you production-realistic inputs with zero model overhead during benchmarking.

## Provenance

These tools are adapted from SGLang's debug utilities:
- `TensorDumper` ← `sglang.srt.debug_utils.dumper` (`_Dumper` class)
- `TensorLoader` ← `sglang.srt.debug_utils.dump_loader` (`DumpLoader`, `ValueWithMeta`)

The SGLang originals include distributed-aware features (multi-rank support, HTTP control server, ZMQ RPC, Polars-based metadata indexing). This standalone version strips those, keeping only the core dump/load logic needed for single-process performance testing.

## Quick Start

### Install

```bash
cd kernel-perf
pip install -e .
```

### Dump

```python
from cupti_perf import TensorDumper
import torch

dumper = TensorDumper("/tmp/kernel_dumps")

# Simulate some tensors
x = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.randn(8, 8192, 4096, device="cuda", dtype=torch.bfloat16)
cnt = torch.tensor([0, 128, 256, 384, 512, 640, 768, 896, 1024], dtype=torch.int32, device="cuda")

dumper.dump_kernel_inputs(
    kernel_name="_Mgemm",
    args={"x": x, "w": w, "cnt": cnt},
    scalars={"max_M_per_E": 128, "transpose_B": True},
    tag="fwd_w13",
)
```

### Load

```python
from cupti_perf import TensorLoader

loader = TensorLoader("/tmp/kernel_dumps")

# List what's available
for entry in loader.list_dumps():
    print(f"{entry.name}: {entry.kernel_name} ({entry.num_tensors} tensors, {entry.num_scalars} scalars)")
    for tname, tsummary in entry.tensor_summaries.items():
        print(f"    {tname}: {tsummary}")

# Load a specific dump to GPU
inputs = loader.load_kernel_inputs(
    "_Mgemm__tag=fwd_w13__idx=0001",
    device="cuda",
)
print(inputs.summary())
```

### Benchmark

```python
from cupti_perf import bench_gpu_time
import numpy as np

# Reconstruct the kernel call from loaded inputs
times = bench_gpu_time(
    fn=_Mgemm,
    input_args=(
        inputs.tensors["x"],
        inputs.tensors["w"],
        inputs.tensors["cnt"],
        inputs.scalars["max_M_per_E"],
    ),
    input_kwargs={"transpose_B": inputs.scalars["transpose_B"]},
    repeat_iters=50,
)

print(f"_Mgemm: median={np.median(times):.3f} ms  std={np.std(times):.3f} ms")
```

---

## API Reference

### TensorDumper

```python
from cupti_perf import TensorDumper

dumper = TensorDumper(dump_dir="/path/to/dumps")
```

#### `dump_kernel_inputs(kernel_name, args, scalars=None, tag="default") -> Path`

Dump all input tensors and scalar arguments for a kernel call.

| Parameter      | Type                   | Description                                               |
|---------------|------------------------|-----------------------------------------------------------|
| `kernel_name` | `str`                  | Name of the kernel (e.g. `"Mgemm_mxfp8"`)                |
| `args`        | `dict[str, Tensor]`    | Named tensor arguments. Each tensor is saved separately   |
| `scalars`     | `dict[str, Any]`       | Non-tensor arguments (ints, bools, floats)                |
| `tag`         | `str`                  | Distinguishes multiple dumps of the same kernel           |

Returns the `Path` to the dump subdirectory.

**What gets saved:**

For each tensor, the dumper records:
- The raw tensor data (via `torch.save`)
- Shape, dtype, stride, device, contiguity, numel, element_size, storage_offset

View tensors are automatically cloned before saving to ensure the full storage is captured.

#### `dump_tensor(name, tensor, tag="default") -> Path`

Convenience method to dump a single tensor. Equivalent to `dump_kernel_inputs(name, {name: tensor}, tag=tag)`.

### TensorLoader

```python
from cupti_perf import TensorLoader

loader = TensorLoader(dump_dir="/path/to/dumps")
```

#### `list_dumps(kernel_name=None) -> list[DumpEntry]`

List available dumps, optionally filtered by kernel name.

Each `DumpEntry` contains:

| Field               | Type              | Description                          |
|--------------------|-------------------|--------------------------------------|
| `name`             | `str`             | Directory name (pass to `load_kernel_inputs`) |
| `path`             | `Path`            | Full path to the dump directory      |
| `kernel_name`      | `str`             | Name of the kernel                   |
| `tag`              | `str`             | User-defined tag                     |
| `num_tensors`      | `int`             | Number of tensor arguments           |
| `num_scalars`      | `int`             | Number of scalar arguments           |
| `tensor_summaries` | `dict[str, str]`  | Per-tensor shape/dtype/stride summary |

#### `load_kernel_inputs(dump_name, device=None, restore_strides=True) -> LoadedKernelInputs`

Load all tensors and scalars from a named dump.

| Parameter         | Type                        | Description                                      |
|------------------|-----------------------------|--------------------------------------------------|
| `dump_name`      | `str`                       | Name of the dump subdirectory                    |
| `device`         | `str` or `torch.device`     | Move tensors to this device (`None` keeps CPU)   |
| `restore_strides`| `bool`                      | Restore original strides via `torch.as_strided`  |

Returns `LoadedKernelInputs` with:
- `.tensors` — `dict[str, Tensor]`
- `.scalars` — `dict[str, Any]`
- `.tensor_metas` — `dict[str, dict]` (raw metadata per tensor)
- `.kernel_name` — `str`
- `.tag` — `str`
- `.summary()` — human-readable string

#### `load_single_tensor(dump_name, tensor_name, device=None, restore_strides=True) -> Tensor`

Convenience method to load a single tensor by name from a dump.

---

## On-Disk Format

Each `dump_kernel_inputs()` call creates a subdirectory:

```
/tmp/kernel_dumps/
  Mgemm_mxfp8__tag=fwd_w13__idx=0001/
    manifest.pt          # metadata + scalar args
    x_fp8.pt             # tensor data
    x_scale.pt
    w_fp8.pt
    w_scale.pt
    cnt.pt
```

### manifest.pt

A dict saved via `torch.save`, containing:

```python
{
    "kernel_name": "Mgemm_mxfp8",
    "timestamp": 1710765432.123,
    "tag": "fwd_w13",
    "tensor_args": {
        "x_fp8": {
            "shape": [1024, 4096],
            "dtype": "torch.float8_e4m3fn",
            "stride": [4096, 1],
            "device": "cuda:0",
            "requires_grad": False,
            "is_contiguous": True,
            "numel": 4194304,
            "element_size": 1,
            "storage_offset": 0,
        },
        ...
    },
    "scalar_args": {
        "max_M_per_E": 128,
    },
}
```

### Tensor .pt files

Raw `torch.save(tensor)` output. View tensors are cloned to contiguous before saving to ensure the complete underlying storage is captured.

---

## Metadata Preservation

| Property          | Preserved? | Notes                                                    |
|------------------|------------|----------------------------------------------------------|
| Shape            | Yes        | Stored in manifest and in the tensor itself              |
| Dtype            | Yes        | Including FP8 types (float8_e4m3fn, etc.)                |
| Strides          | Yes*       | Restored via `torch.as_strided` on load. Views are cloned on dump (become contiguous), original strides recorded in manifest |
| Device           | Yes*       | Recorded in manifest. Tensors load to CPU by default; pass `device="cuda"` to place on GPU |
| requires_grad    | Yes        | Recorded in manifest (not re-applied on load)            |
| storage_offset   | Yes        | Recorded in manifest                                      |
| Contiguity       | Yes*       | Recorded. Non-contiguous strides restored if `restore_strides=True` |

---

## End-to-End Example: MoE Kernel Benchmarking

This section shows how to dump inputs from `triton-moe`'s `MoEFFNFunction.forward` and replay them for kernel benchmarking.

### Target Kernels

In `triton-moe/llm/kernel/moe_ffn.py` (lines 594–600), three GEMM backends compute `Y13 = X13 @ W13.T`:

| Condition           | Kernel Called    | Arguments                                                                  |
|--------------------|------------------|----------------------------------------------------------------------------|
| `use_triton_mxfp8` | `Mgemm_mxfp8`   | `x13_fp8, x13_scale, w13_fp8, w13_scale, slice_offs, max_tokens_per_expert` |
| `HAVE_RAGGED_TMA`  | `Mgemm` (via `_Mgemm`) | `x13, w13, slice_offs, max_tokens_per_expert, transpose_B=True`     |
| Fallback           | `groupedM` (via `_Mgemm`) | `x13, w13, cnt_off, num_local_tokens, 2*ffn_dim, hidden_dim, num_local_experts, max_tokens_per_expert, transpose_B=True` |

### Step 1: Instrument the Forward Pass

Add dump calls in `moe_ffn.py` right before the kernel calls:

```python
from cupti_perf import TensorDumper
_dumper = TensorDumper("/tmp/moe_kernel_dumps")

# Inside MoEFFNFunction.forward, before the Y13 GEMM block:

if use_triton_mxfp8:
    _dumper.dump_kernel_inputs(
        kernel_name="Mgemm_mxfp8",
        args={
            "x_fp8": x13_fp8,           # [M, K] float8_e4m3fn
            "x_scale": x13_scale,       # [M, K//32] uint8
            "w_fp8": w13_fp8,           # [E, N, K] float8_e4m3fn
            "w_scale": w13_scale,       # [E, N, K//32] uint8
            "cnt": slice_offs,          # [E+1] int32
        },
        scalars={"max_M_per_E": int(max_tokens_per_expert)},
        tag="fwd_w13",
    )
    y13 = Mgemm_mxfp8(x13_fp8, x13_scale, w13_fp8, w13_scale, slice_offs, max_tokens_per_expert)

elif HAVE_RAGGED_TMA:
    _dumper.dump_kernel_inputs(
        kernel_name="Mgemm",
        args={
            "x": x13,                  # [M, K] bfloat16
            "w": w13,                  # [E, N, K] bfloat16
            "cnt": slice_offs,         # [E+1] int32
        },
        scalars={
            "max_M_per_E": int(max_tokens_per_expert),
            "transpose_B": True,
        },
        tag="fwd_w13",
    )
    y13 = _Mgemm(x13, w13, slice_offs, max_tokens_per_expert, transpose_B=True)
```

### Step 2: Run the Model Once

Execute one forward pass (e.g. a single inference request) to capture real-world tensor shapes and values:

```bash
# Your normal model launch command
python run_model.py --model DeepSeek-V3 --input "Hello world"
```

Check the dump directory:

```bash
ls /tmp/moe_kernel_dumps/
# Mgemm_mxfp8__tag=fwd_w13__idx=0001/
# Mgemm_mxfp8__tag=fwd_w13__idx=0002/
# ...
```

### Step 3: Standalone Benchmark Script

```python
#!/usr/bin/env python3
"""Benchmark Mgemm_mxfp8 using dumped inputs."""

import numpy as np
import torch
from cupti_perf import TensorLoader, bench_gpu_time

# Import the kernel under test
from triton_moe.llm.kernel.gemm import Mgemm_mxfp8

loader = TensorLoader("/tmp/moe_kernel_dumps")

# Pick a dump
dumps = loader.list_dumps(kernel_name="Mgemm_mxfp8")
print(f"Found {len(dumps)} Mgemm_mxfp8 dumps")

for entry in dumps[:3]:  # benchmark first 3
    inputs = loader.load_kernel_inputs(entry.name, device="cuda")
    print(f"\n--- {entry.name} ---")
    print(inputs.summary())

    t = inputs.tensors
    s = inputs.scalars

    times = bench_gpu_time(
        fn=Mgemm_mxfp8,
        input_args=(
            t["x_fp8"], t["x_scale"],
            t["w_fp8"], t["w_scale"],
            t["cnt"], s["max_M_per_E"],
        ),
        cold_l2_cache=True,
        repeat_iters=50,
    )

    print(f"  median: {np.median(times):.3f} ms")
    print(f"  std:    {np.std(times):.3f} ms")
    print(f"  min:    {np.min(times):.3f} ms")
    print(f"  max:    {np.max(times):.3f} ms")
```

### Step 4: Compare Kernels

Use the same dumped inputs to compare different kernel implementations:

```python
from triton_moe.llm.kernel.gemm import Mgemm_mxfp8, Mgemm

loader = TensorLoader("/tmp/moe_kernel_dumps")

# Load MXFP8 inputs
mxfp8_inputs = loader.load_kernel_inputs(
    "Mgemm_mxfp8__tag=fwd_w13__idx=0001", device="cuda"
)

# Load bf16 inputs (if you dumped both)
bf16_inputs = loader.load_kernel_inputs(
    "Mgemm__tag=fwd_w13__idx=0001", device="cuda"
)

# Benchmark MXFP8 path
t = mxfp8_inputs.tensors
s = mxfp8_inputs.scalars
mxfp8_times = bench_gpu_time(
    fn=Mgemm_mxfp8,
    input_args=(t["x_fp8"], t["x_scale"], t["w_fp8"], t["w_scale"], t["cnt"], s["max_M_per_E"]),
    repeat_iters=50,
)

# Benchmark bf16 ragged-TMA path
t = bf16_inputs.tensors
s = bf16_inputs.scalars
bf16_times = bench_gpu_time(
    fn=Mgemm,
    input_args=(t["x"], t["w"], t["cnt"], s["max_M_per_E"]),
    input_kwargs={"transpose_B": s["transpose_B"]},
    repeat_iters=50,
)

print(f"Mgemm_mxfp8: median={np.median(mxfp8_times):.3f} ms")
print(f"Mgemm (bf16): median={np.median(bf16_times):.3f} ms")
print(f"Speedup: {np.median(bf16_times) / np.median(mxfp8_times):.2f}x")
```

---

## Kernel Input Reference

Below are the tensor signatures for the three GEMM kernels in `moe_ffn.py`:

### Mgemm_mxfp8 (Triton MXFP8 grouped GEMM)

| Argument    | Shape          | Dtype              | Description                              |
|------------|----------------|--------------------|------------------------------------------|
| `x_fp8`    | `[M, K]`      | `float8_e4m3fn`    | Activations, MXFP8 quantized             |
| `x_scale`  | `[M, K//32]`  | `uint8`            | E8M0 scales for activations              |
| `w_fp8`    | `[E, N, K]`   | `float8_e4m3fn`    | Weights, MXFP8 quantized                 |
| `w_scale`  | `[E, N, K//32]`| `uint8`           | E8M0 scales for weights                  |
| `cnt`      | `[E+1]`       | `int32`            | Cumulative prefix sum (`slice_offs`)      |
| `max_M_per_E` | scalar     | `int`              | Max tokens per expert                     |

Constraints: `N % 128 == 0`, `K % 128 == 0`.

### Mgemm (ragged TMA path, via `_Mgemm`)

| Argument       | Shape      | Dtype              | Description                              |
|---------------|------------|--------------------|------------------------------------------|
| `x`           | `[M, K]`  | `bfloat16`/`float16` | Activations                            |
| `w`           | `[E, N, K]`| `bfloat16`/`float16` | Weights                               |
| `cnt`         | `[E+1]`   | `int32`            | Cumulative prefix sum (`slice_offs`)      |
| `max_M_per_E` | scalar    | `int`              | Max tokens per expert                     |
| `transpose_B` | scalar    | `bool`             | Use `w.T` for matmul                     |

Constraints: `N % 256 == 0`, `K % 64 == 0`.

### groupedM (fallback path, via `_Mgemm`)

| Argument       | Shape      | Dtype              | Description                              |
|---------------|------------|--------------------|------------------------------------------|
| `A`           | `[M, K]`  | `bfloat16`/`float16` | Activations                            |
| `B`           | `[E, K, N]`| `bfloat16`/`float16` | Weights                               |
| `cnt`         | `[E]`     | `int32`            | Cumulative end offsets (`cnt_off`)        |
| `M`           | scalar    | `int`              | Total rows (`num_local_tokens`)           |
| `N`           | scalar    | `int`              | Output dim (`2 * ffn_dim`)                |
| `K`           | scalar    | `int`              | Inner dim (`hidden_dim`)                  |
| `E`           | scalar    | `int`              | Number of experts (`num_local_experts`)   |
| `max_M_per_E` | scalar    | `int`              | Max tokens per expert                     |
| `transpose_B` | scalar    | `bool`             | Transpose B for matmul                    |

---

## Tips

- **Dump once, benchmark many times.** The dump directory is persistent — you only need to instrument and run the model once.
- **Use tags** to distinguish different call sites (e.g. `fwd_w13` vs `fwd_w2`).
- **FP8 dtypes** (`float8_e4m3fn`, etc.) are preserved through save/load. Make sure your PyTorch version supports them.
- **Large tensors** can produce large dump files. For a typical MoE layer with E=8, N=8192, K=4096, the weight tensor alone is ~256 MB in bf16. Dumps will be proportionally sized.
- **`restore_strides=True`** (default) ensures non-contiguous layouts are faithfully reproduced. Disable it if you only care about values and want contiguous tensors.
- **Combine with `bench_gpu_time`** for the full workflow: dump → load → benchmark with CUPTI for hardware-accurate GPU timing.
