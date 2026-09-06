"""Shared final tensor allocations using PyTorch's CUDA caching allocator."""

from __future__ import annotations

import contextlib
import threading

from ._native import load

_allocators = {}
_lock = threading.RLock()


class WeightPool:
    """Activate shared storage only around an explicit weight factory."""

    def __init__(self, pool, device):
        self.pool = pool
        self.device = device
        self.depth = 0

    @contextlib.contextmanager
    def __call__(self):
        import torch

        if self.pool is None:
            raise RuntimeError("weight allocation pool is closed")
        if self.depth:
            yield
            return
        with (
            torch.cuda.device(self.device),
            torch.cuda.use_mem_pool(self.pool, device=self.device),
        ):
            self.depth += 1
            try:
                yield
            finally:
                self.depth -= 1


@contextlib.contextmanager
def weight_pool(*, allocation="registered", device=0):
    """Own a shared pool without changing the default CUDA allocation policy.

    PyTorch retains its normal tensor types, suballocator and stream tracking.
    Only use during exclusive model construction/loading, before graph capture.
    Unused segments retire on scope exit; live tensor storage remains valid.
    Allocator backends are process-owned because Torch keeps raw pointers to
    them even after the pool scope ends.
    """
    import torch

    if allocation not in ("registered", "pinned", "pinned_wc", "managed"):
        raise ValueError(
            "shared pools support registered, pinned, pinned_wc or managed storage"
        )
    torch.cuda.init()
    native = load()
    caps = native.capabilities(device)
    if not (caps["pageable_memory_access"] and caps["host_page_tables"]):
        raise RuntimeError("the initial b12x loader requires GPU host page tables")
    if torch.cuda.get_allocator_backend() != "native":
        raise RuntimeError("b12x shared pools require PyTorch's native CUDA allocator")
    with _lock:
        key = (device, allocation)
        if key not in _allocators:
            allocator = torch.cuda.memory.CUDAPluggableAllocator(
                native.__file__, f"b12x_{allocation}_alloc", "b12x_pool_free"
            )
            native.keep_allocator(allocator)
            _allocators[key] = allocator
        settings = torch.cuda.memory._snapshot()["allocator_settings"]
        expandable = settings["expandable_segments"]
        if expandable:
            torch.cuda.memory._set_allocator_settings("expandable_segments:False")
        allocator = WeightPool(torch.cuda.MemPool(_allocators[key].allocator()), device)
        try:
            yield allocator
        finally:
            allocator.pool = None
            torch.cuda.empty_cache()
            if expandable:
                torch.cuda.memory._set_allocator_settings("expandable_segments:True")


@contextlib.contextmanager
def shared_pool(*, allocation="registered", device=0):
    """Allocate explicitly scoped weights from CPU-addressable final storage."""
    with weight_pool(allocation=allocation, device=device) as allocator, allocator():
        yield


def owns_tensor(tensor):
    """Whether the contiguous tensor lies wholly in a live b12x pool segment."""
    return (
        tensor.is_cuda
        and tensor.is_contiguous()
        and load().pool_contains(tensor.data_ptr(), tensor.nbytes)
    )


def owns_storage(tensor):
    """Whether a tensor's backing storage belongs to the shared weight pool."""
    if not tensor.is_cuda:
        return False
    storage = tensor.untyped_storage()
    return load().pool_contains(storage.data_ptr(), storage.nbytes())


class HostWeightWriter:
    """Copy CPU checkpoint views into final shared storage without GPU staging.

    Callers retain both tensors through the synchronous write and must provide
    exclusive access to the destination on its current stream. Transforms and
    unsupported layouts use the model's ordinary Torch copy path.
    """

    def __init__(self):
        self.native = load()
        self.host_bytes = 0
        self.host_copies = 0
        self.torch_bytes = 0

    def __call__(self, destination, source):
        import torch

        if (
            source.device.type == "cpu"
            and destination.is_cuda
            and source.dtype == destination.dtype
            and (
                source.shape == destination.shape
                or source.numel() == destination.numel() == 1
            )
            and source.is_contiguous()
            and destination.is_contiguous()
            and not source.is_conj()
            and not source.is_neg()
            and not destination.is_conj()
            and not destination.is_neg()
        ):
            with torch.cuda.device(destination.device):
                copied = self.native.pool_copy(
                    destination.data_ptr(),
                    source.data_ptr(),
                    source.nbytes,
                    torch.cuda.current_stream(destination.device).cuda_stream,
                )
            if copied:
                torch.autograd.graph.increment_version(destination)
                self.host_bytes += source.nbytes
                self.host_copies += 1
                return True
        self.torch_bytes += destination.nbytes
        return False
