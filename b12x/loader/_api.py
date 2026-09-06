"""Buffered raw-file reads for allocation qualification; not the O_DIRECT adapter."""

from __future__ import annotations

import math
import operator

from ._native import load


def capabilities(device: int = 0) -> dict[str, int]:
    """Query the CUDA device before selecting a shared allocation type."""
    return load().capabilities(operator.index(device))


def storage_stats() -> dict[str, int]:
    """Return live allocation counts and requested bytes owned by this loader.

    These counters exclude page cache, CUDA context overhead, and page rounding.
    They are not a system memory budget.
    """
    return load().storage_stats()


def read_tensor(
    path, *, shape, dtype, offset: int = 0, allocation: str, device: int = 0
):
    """Read a contiguous file range into final, shared CUDA tensor storage.

    This experimental primitive uses buffered pread (or mmap for ``file``),
    independently of the vLLM adapter's O_DIRECT transport. It performs no dtype
    conversion or H2D copy.
    Reads finish before publishing the tensor. Its storage owns the allocation
    through tensor views and aliases; the final release waits for device work.
    Retain the tensor while replaying any graph that references its address.
    Checkpoint files must remain immutable during loading. With ``file``
    storage, they must remain immutable for the entire tensor lifetime.
    Unregistered ``system``/``file`` pointers can be rejected by consumers
    such as Triton's launcher; use ``registered`` for those consumers.

    Args:
        path: Local file containing the raw tensor bytes.
        shape: Contiguous tensor shape, including scalar and empty shapes.
        dtype: PyTorch dtype of the stored bytes.
        offset: Absolute byte offset in the file.
        allocation: ``system``, ``pinned``, ``pinned_wc``, ``registered``, ``managed``, or
            ``file``. System/file allocations require GPU host page tables.
            File mappings may not meet an inference kernel's alignment needs.
        device: CUDA device index that will consume the tensor.

    Returns:
        A CUDA tensor owning the loaded bytes, independent of the file handle.
    """
    import torch

    shape = tuple(operator.index(dimension) for dimension in shape)
    offset = operator.index(offset)
    if any(dimension < 0 for dimension in shape) or offset < 0:
        raise ValueError("shape and offset must be nonnegative")
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")
    if dtype.is_complex or dtype in (torch.qint8, torch.quint8, torch.qint32):
        raise ValueError(f"unsupported checkpoint dtype: {dtype}")
    nbytes = math.prod(shape) * dtype.itemsize
    if nbytes > 2**63 - 1 or offset > 2**63 - 1 - nbytes:
        raise OverflowError("tensor file range exceeds signed 64-bit addressing")
    if allocation == "file" and offset % dtype.itemsize:
        raise ValueError("file mapping offset must be aligned to the dtype")
    torch.cuda.init()
    native = load()
    with open(path, "rb", buffering=0) as stream:
        capsule = native.read(
            stream.fileno(), offset, nbytes, allocation, operator.index(device)
        )
    raw = torch.utils.dlpack.from_dlpack(capsule)
    return raw.view(dtype).reshape(shape)
