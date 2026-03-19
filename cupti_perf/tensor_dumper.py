"""
Tensor dumper for GPU kernel performance testing.

Saves PyTorch tensors and scalar arguments to disk, preserving shape, dtype,
strides, and device metadata.  Designed for capturing the exact inputs to
kernel functions (e.g. Mgemm_mxfp8, _Mgemm) so they can be replayed later
in isolation for benchmarking.

Adapted from SGLang's ``sglang.srt.debug_utils.dumper``.

Usage:

    from cupti_perf.tensor_dumper import TensorDumper

    dumper = TensorDumper("/tmp/kernel_dumps")
    dumper.dump_kernel_inputs(
        kernel_name="Mgemm_mxfp8",
        args={"x_fp8": x_fp8, "x_scale": x_scale, ...},
        scalars={"max_M_per_E": max_M_per_E},
    )
"""

import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class TensorMeta:
    """Metadata for a single tensor, stored alongside the data."""

    shape: List[int]
    dtype: str
    stride: List[int]
    device: str
    requires_grad: bool
    is_contiguous: bool
    numel: int
    element_size: int
    storage_offset: int

    @staticmethod
    def from_tensor(t: torch.Tensor) -> "TensorMeta":
        return TensorMeta(
            shape=list(t.shape),
            dtype=str(t.dtype),
            stride=list(t.stride()),
            device=str(t.device),
            requires_grad=t.requires_grad,
            is_contiguous=t.is_contiguous(),
            numel=t.numel(),
            element_size=t.element_size(),
            storage_offset=t.storage_offset(),
        )


@dataclass
class KernelDumpManifest:
    """Manifest describing a full kernel input snapshot."""

    kernel_name: str
    timestamp: float
    tag: str
    tensor_args: Dict[str, Dict[str, Any]]
    scalar_args: Dict[str, Any]

    def summary(self) -> str:
        lines = [f"Kernel: {self.kernel_name}  tag: {self.tag}"]
        for name, meta in self.tensor_args.items():
            lines.append(
                f"  {name}: shape={meta['shape']} dtype={meta['dtype']} "
                f"stride={meta['stride']} device={meta['device']}"
            )
        for name, val in self.scalar_args.items():
            lines.append(f"  {name}: {val!r}")
        return "\n".join(lines)


def _clone_if_view(t: torch.Tensor) -> torch.Tensor:
    if t.untyped_storage().nbytes() > t.nelement() * t.element_size():
        return t.clone()
    return t


_DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
    "torch.float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
    "torch.float8_e5m2": getattr(torch, "float8_e5m2", None),
    "torch.float8_e4m3fnuz": getattr(torch, "float8_e4m3fnuz", None),
    "torch.float8_e5m2fnuz": getattr(torch, "float8_e5m2fnuz", None),
}


def _str_to_dtype(s: str) -> torch.dtype:
    dt = _DTYPE_MAP.get(s)
    if dt is None:
        raise ValueError(f"Unsupported dtype string: {s!r}")
    return dt


class TensorDumper:
    """Dump kernel input tensors and scalars to a directory for later replay.

    Each call to :meth:`dump_kernel_inputs` creates a subdirectory containing:
      - One ``.pt`` file per tensor argument (via ``torch.save``)
      - A ``manifest.pt`` with full metadata and scalar arguments

    Parameters
    ----------
    dump_dir : str or Path
        Root directory for all dumps.
    cleanup_on_init : bool
        If ``True``, remove all existing dump subdirectories under *dump_dir*
        when the dumper is created.  Mirrors SGLang's ``DUMPER_CLEANUP_PREVIOUS``
        behaviour.  Default ``False``.
    """

    def __init__(self, dump_dir: Union[str, Path], cleanup_on_init: bool = False):
        self.dump_dir = Path(dump_dir)
        if cleanup_on_init and self.dump_dir.exists():
            self.cleanup()
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self._dump_count = 0

    def cleanup(self) -> None:
        """Remove all dump subdirectories (those containing a ``manifest.pt``)."""
        if not self.dump_dir.exists():
            return
        removed = 0
        for d in list(self.dump_dir.iterdir()):
            if d.is_dir() and (d / "manifest.pt").exists():
                shutil.rmtree(d)
                removed += 1
        if removed:
            print(f"[TensorDumper] Cleaned up {removed} dump(s) in {self.dump_dir}")

    def dump_kernel_inputs(
        self,
        kernel_name: str,
        args: Dict[str, torch.Tensor],
        scalars: Optional[Dict[str, Any]] = None,
        tag: str = "default",
    ) -> Path:
        """Dump all input tensors and scalar arguments for a kernel call.

        Parameters
        ----------
        kernel_name : str
            Name of the kernel (e.g. ``"Mgemm_mxfp8"``).
        args : dict[str, Tensor]
            Named tensor arguments.  Each tensor is saved as a separate file.
        scalars : dict[str, Any], optional
            Non-tensor arguments (ints, bools, floats).
        tag : str
            A user-defined tag for distinguishing multiple dumps of the same
            kernel (e.g. ``"fwd_w13"``, ``"fwd_w2"``).

        Returns
        -------
        Path
            The directory where this dump was written.
        """
        if scalars is None:
            scalars = {}

        ts = time.time()
        self._dump_count += 1
        dirname = f"{kernel_name}__tag={tag}__idx={self._dump_count:04d}"
        snap_dir = self.dump_dir / dirname
        snap_dir.mkdir(parents=True, exist_ok=True)

        tensor_metas: Dict[str, Dict[str, Any]] = {}

        for name, tensor in args.items():
            if not isinstance(tensor, torch.Tensor):
                scalars[name] = tensor
                continue
            meta = TensorMeta.from_tensor(tensor)
            tensor_metas[name] = asdict(meta)
            saved_tensor = _clone_if_view(tensor)
            torch.save(saved_tensor, snap_dir / f"{name}.pt")

        manifest = KernelDumpManifest(
            kernel_name=kernel_name,
            timestamp=ts,
            tag=tag,
            tensor_args=tensor_metas,
            scalar_args=scalars,
        )
        torch.save(asdict(manifest), snap_dir / "manifest.pt")

        print(
            f"[TensorDumper] Saved {kernel_name} (tag={tag}) → {snap_dir} "
            f"({len(tensor_metas)} tensors, {len(scalars)} scalars)"
        )
        return snap_dir

    def dump_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        tag: str = "default",
    ) -> Path:
        """Convenience: dump a single tensor with metadata.

        Returns
        -------
        Path
            Path to the saved ``.pt`` file.
        """
        return self.dump_kernel_inputs(
            kernel_name=name,
            args={name: tensor},
            tag=tag,
        )
