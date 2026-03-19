"""
Tensor loader for GPU kernel performance testing.

Loads tensors previously saved by :class:`TensorDumper`, restoring their
original shape, dtype, strides, and (optionally) device placement.  This
enables faithful replay of kernel inputs for isolated benchmarking.

Adapted from SGLang's ``sglang.srt.debug_utils.dump_loader``.

Usage:

    from cupti_perf.tensor_loader import TensorLoader

    loader = TensorLoader("/tmp/kernel_dumps")
    loader.list_dumps()

    inputs = loader.load_kernel_inputs(
        "Mgemm_mxfp8__tag=fwd_w13__idx=0001",
        device="cuda",
    )
    tensors = inputs.tensors   # dict[str, Tensor]
    scalars = inputs.scalars   # dict[str, Any]
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from cupti_perf.tensor_dumper import TensorMeta, _str_to_dtype


@dataclass
class LoadedKernelInputs:
    """Container returned by :meth:`TensorLoader.load_kernel_inputs`."""

    kernel_name: str
    tag: str
    tensors: Dict[str, torch.Tensor]
    scalars: Dict[str, Any]
    tensor_metas: Dict[str, Dict[str, Any]]

    def summary(self) -> str:
        lines = [f"Kernel: {self.kernel_name}  tag: {self.tag}"]
        for name, t in self.tensors.items():
            meta = self.tensor_metas.get(name, {})
            orig_device = meta.get("device", "?")
            lines.append(
                f"  {name}: shape={list(t.shape)} dtype={t.dtype} "
                f"stride={list(t.stride())} device={t.device} "
                f"(original_device={orig_device})"
            )
        for name, val in self.scalars.items():
            lines.append(f"  {name}: {val!r}")
        return "\n".join(lines)


@dataclass
class DumpEntry:
    """Metadata about a single dump directory, for listing."""

    name: str
    path: Path
    kernel_name: str
    tag: str
    num_tensors: int
    num_scalars: int
    tensor_summaries: Dict[str, str]

    def __repr__(self) -> str:
        return (
            f"DumpEntry({self.kernel_name!r}, tag={self.tag!r}, "
            f"tensors={self.num_tensors}, scalars={self.num_scalars})"
        )


def _restore_strides(
    tensor: torch.Tensor,
    target_shape: List[int],
    target_stride: List[int],
    target_dtype_str: str,
) -> torch.Tensor:
    """Reshape and restride a loaded tensor to match the original layout.

    If the original tensor was non-contiguous (e.g. a transposed view), we
    recreate that layout with ``torch.as_strided``.
    """
    target_dtype = _str_to_dtype(target_dtype_str)
    if tensor.dtype != target_dtype:
        tensor = tensor.to(target_dtype)

    if list(tensor.shape) != target_shape:
        tensor = tensor.reshape(target_shape)

    if list(tensor.stride()) != target_stride:
        tensor = torch.as_strided(tensor, target_shape, target_stride)

    return tensor


class TensorLoader:
    """Load kernel input dumps produced by :class:`TensorDumper`.

    Parameters
    ----------
    dump_dir : str or Path
        Root directory containing dump subdirectories.
    """

    def __init__(self, dump_dir: Union[str, Path]):
        self.dump_dir = Path(dump_dir)
        if not self.dump_dir.exists():
            raise FileNotFoundError(f"Dump directory does not exist: {self.dump_dir}")

    def list_dumps(self, kernel_name: Optional[str] = None) -> List[DumpEntry]:
        """List available dumps, optionally filtered by kernel name.

        Returns a list of :class:`DumpEntry` objects sorted by directory name.
        """
        entries: List[DumpEntry] = []
        for d in sorted(self.dump_dir.iterdir()):
            if not d.is_dir():
                continue
            manifest_path = d / "manifest.pt"
            if not manifest_path.exists():
                continue

            manifest = torch.load(manifest_path, weights_only=False, map_location="cpu")
            kname = manifest.get("kernel_name", "unknown")
            tag = manifest.get("tag", "")

            if kernel_name is not None and kname != kernel_name:
                continue

            tensor_metas = manifest.get("tensor_args", {})
            scalar_args = manifest.get("scalar_args", {})

            summaries = {}
            for tname, meta in tensor_metas.items():
                summaries[tname] = (
                    f"shape={meta['shape']} dtype={meta['dtype']} "
                    f"stride={meta['stride']}"
                )

            entries.append(
                DumpEntry(
                    name=d.name,
                    path=d,
                    kernel_name=kname,
                    tag=tag,
                    num_tensors=len(tensor_metas),
                    num_scalars=len(scalar_args),
                    tensor_summaries=summaries,
                )
            )

        return entries

    def load_kernel_inputs(
        self,
        dump_name: str,
        device: Optional[Union[str, torch.device]] = None,
        restore_strides: bool = True,
    ) -> LoadedKernelInputs:
        """Load all tensors and scalars from a named dump.

        Parameters
        ----------
        dump_name : str
            Name of the dump subdirectory (as returned by :meth:`list_dumps`).
        device : str or torch.device, optional
            Move loaded tensors to this device.  ``None`` keeps the saved
            device (typically ``"cpu"`` after ``torch.load``).  Use
            ``"cuda"`` to place tensors on the default GPU.
        restore_strides : bool
            If ``True`` (default), restore original strides via
            ``torch.as_strided`` when possible.

        Returns
        -------
        LoadedKernelInputs
            Container with ``.tensors`` (dict), ``.scalars`` (dict), and
            per-tensor metadata.
        """
        snap_dir = self.dump_dir / dump_name
        if not snap_dir.exists():
            raise FileNotFoundError(f"Dump not found: {snap_dir}")

        manifest_path = snap_dir / "manifest.pt"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found in {snap_dir}")

        manifest = torch.load(manifest_path, weights_only=False, map_location="cpu")
        kernel_name = manifest["kernel_name"]
        tag = manifest["tag"]
        tensor_metas = manifest.get("tensor_args", {})
        scalar_args = manifest.get("scalar_args", {})

        tensors: Dict[str, torch.Tensor] = {}
        for name, meta in tensor_metas.items():
            pt_path = snap_dir / f"{name}.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"Tensor file missing: {pt_path}")

            t = torch.load(pt_path, weights_only=False, map_location="cpu")

            if restore_strides:
                t = _restore_strides(
                    t,
                    target_shape=meta["shape"],
                    target_stride=meta["stride"],
                    target_dtype_str=meta["dtype"],
                )

            if device is not None:
                t = t.to(device)

            tensors[name] = t

        print(
            f"[TensorLoader] Loaded {kernel_name} (tag={tag}) ← {snap_dir} "
            f"({len(tensors)} tensors, {len(scalar_args)} scalars)"
        )

        return LoadedKernelInputs(
            kernel_name=kernel_name,
            tag=tag,
            tensors=tensors,
            scalars=scalar_args,
            tensor_metas=tensor_metas,
        )

    def load_single_tensor(
        self,
        dump_name: str,
        tensor_name: str,
        device: Optional[Union[str, torch.device]] = None,
        restore_strides: bool = True,
    ) -> torch.Tensor:
        """Convenience: load one tensor by name from a dump."""
        inputs = self.load_kernel_inputs(
            dump_name, device=device, restore_strides=restore_strides
        )
        if tensor_name not in inputs.tensors:
            available = list(inputs.tensors.keys())
            raise KeyError(
                f"Tensor {tensor_name!r} not found. Available: {available}"
            )
        return inputs.tensors[tensor_name]
