"""Push-based nine-rank island all-reduce runtime (two PCIe islands + root rank).

Serves the mid-size bf16 all-reduce class (a few KB to a few hundred KB) on
the nine-GPU PCIe topology where ranks 0-3 and 4-7 sit behind two switch
clusters and rank 8 on the CPU root complex. Rank ``island * 4 + lane`` owns
quarter ``lane`` of the vector; rank 8 contributes through island 0. Every
remote transfer is a posted write into a peer inbox, each rank maps at most
five peers, and the reduction accumulates in fp32 with one bf16 rounding
(island partials stay in fp32).

The public surface mirrors :class:`PCIeTwoShotBF16` where vLLM's PCIe
crossover dispatcher needs it: ``accepts``, ``all_reduce(inp, out=)``,
``prepare_graph``, ``capture``, ``close``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ._island9_cute import (
    HEADER_BYTES,
    ISLAND_SIZE,
    MAX_BLOCKS,
    WORLD_SIZE,
    get_island9_launcher,
    island9_peers,
)
from ._cuda_ipc import CudaRTLibrary
from .pcie_oneshot import (
    PCIeOneshotAllReduce,
    _finish_collective_runtime_setup,
    _is_current_stream_capturing,
    _normalize_device,
)

SUPPORTED_WORLD_SIZES = (WORLD_SIZE,)
SUPPORTED_BLOCKS = (1, 2, 4, 8, 16, 32)
_ALIGNMENT = 256


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _threads_from_env() -> int:
    raw = os.getenv("B12X_PCIE_ISLAND9_THREADS", "256")
    threads = int(raw)
    if not 32 <= threads <= 1024 or threads % 32:
        raise ValueError(
            "B12X_PCIE_ISLAND9_THREADS must be a multiple of 32 in [32, 1024]"
        )
    return threads


def _wait_nanosleep_cycles_from_env() -> int:
    cycles = int(os.getenv("B12X_PCIE_ISLAND9_NANOSLEEP_CYCLES", "24"))
    if not 0 <= cycles <= 1024:
        raise ValueError("B12X_PCIE_ISLAND9_NANOSLEEP_CYCLES must be in [0, 1024]")
    return cycles


def _pick_blocks(elements: int) -> int:
    """Launch geometry by element count: one warp-row of blocks per ~4 KB."""
    if elements <= 8192:
        return 8
    if elements <= 65536:
        return 16
    return 32


class PCIeIsland9AllReduce:
    """Equal-quarter bf16 all-reduce for TP9 on two PCIe islands plus a root rank."""

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_elements: int,
        blocks: Optional[int] = None,
    ) -> None:
        self.group = exchange_group
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self.device = _normalize_device(device)
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"island9 all-reduce requires TP9, got TP{self.world_size}")
        if self.device.type != "cuda":
            raise ValueError("island9 all-reduce requires a CUDA device")
        if blocks is not None and blocks not in SUPPORTED_BLOCKS:
            raise ValueError(f"blocks must be one of {SUPPORTED_BLOCKS}")
        if max_elements <= 0 or max_elements % 2:
            raise ValueError("max_elements must be a positive even count")

        self.max_elements = int(max_elements)
        self.blocks = None if blocks is None else int(blocks)
        self.threads = _threads_from_env()
        self.wait_nanosleep_cycles = _wait_nanosleep_cycles_from_env()
        self.all_reduce_mode = "push"

        max_pairs = self.max_elements // 2
        # Quarter stride in bf16x2 words; every inbox has two generation slots.
        self.quarter_capacity = _align_up((max_pairs + ISLAND_SIZE - 1) // ISLAND_SIZE, 8)
        quarter_bytes = self.quarter_capacity * 4
        stage_bytes = 5 * quarter_bytes  # four island lanes + rank 8
        part_bytes = 2 * quarter_bytes  # fp32 pairs
        final_bytes = ISLAND_SIZE * quarter_bytes
        self.stage_offset = _align_up(HEADER_BYTES)
        self.part_self_offset = _align_up(self.stage_offset + 2 * stage_bytes)
        self.part_inbox_offset = _align_up(self.part_self_offset + 2 * part_bytes)
        self.final_offset = _align_up(self.part_inbox_offset + 2 * part_bytes)
        self.slab_bytes = _align_up(self.final_offset + 2 * final_bytes)

        self._ipc = CudaRTLibrary()
        self._ipc.cudaSetDevice(self.device.index or 0)
        self._slab_ptrs: tuple[int, ...] = ()
        self._local_ptr = 0
        self._remote_ptrs: list[int] = []
        self._closed = False
        self._launcher = None
        self._mapped_peers = island9_peers(self.rank)
        self._capture_depth = 0

        shared = PCIeOneshotAllReduce._allocate_shared_buffer(
            exchange_group,
            self.slab_bytes,
            zero_fill=True,
            ipc=self._ipc,
            peer_ranks=self._mapped_peers,
        )
        self._local_ptr = shared.local_ptr
        self._remote_ptrs = list(shared.remote_ptrs)
        self._slab_ptrs = shared.peer_ptrs

        init_error: BaseException | None = None
        try:
            with torch.cuda.device(self.device):
                self._launcher = get_island9_launcher(
                    self.rank,
                    self.device.index or 0,
                    threads=self.threads,
                    wait_nanosleep_cycles=self.wait_nanosleep_cycles,
                )
        except Exception as exc:
            init_error = exc

        def detach_shared_ownership() -> None:
            self._slab_ptrs = ()
            self._remote_ptrs.clear()
            self._local_ptr = 0

        _finish_collective_runtime_setup(
            owner="PCIe island9 all-reduce",
            exchange_group=exchange_group,
            ipc=self._ipc,
            shared=shared,
            local_error=init_error,
            detach_shared_ownership=detach_shared_ownership,
        )

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_rows: int,
        row_elems: int,
    ) -> "PCIeIsland9AllReduce":
        """Construct with the two-shot runtime's capacity vocabulary."""
        elements = int(max_rows) * int(row_elems)
        return cls(
            exchange_group=exchange_group,
            device=device,
            max_elements=elements - (elements % 2),
        )

    @property
    def mapped_peers(self) -> tuple[int, ...]:
        return self._mapped_peers

    def accepts(self, inp: torch.Tensor) -> bool:
        return (
            not self._closed
            and inp.device == self.device
            and inp.dtype == torch.bfloat16
            and inp.is_contiguous()
            and 0 < inp.numel() <= self.max_elements
            and inp.numel() % 2 == 0
            and inp.data_ptr() % 4 == 0
        )

    should_allreduce = accepts

    def all_reduce(
        self,
        inp: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        blocks: Optional[int] = None,
        stream: object = None,
        channel_id: Optional[str] = None,
        threads: Optional[int] = None,
        block_limit: Optional[int] = None,
    ) -> torch.Tensor:
        del channel_id, threads, block_limit
        if not self.accepts(inp):
            raise ValueError(
                "input does not satisfy island9 all-reduce requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if out is None:
            if _is_current_stream_capturing(self.device):
                raise RuntimeError(
                    "PCIeIsland9AllReduce.all_reduce CUDA graph capture requires "
                    "a caller-owned preallocated output"
                )
            out = torch.empty_like(inp)
        if (
            out.dtype != inp.dtype
            or out.device != inp.device
            or out.shape != inp.shape
            or not out.is_contiguous()
            or out.data_ptr() % 4 != 0
        ):
            raise ValueError("output must match input and be 4-byte aligned")
        if blocks is not None:
            selected = int(blocks)
        elif self.blocks is not None:
            selected = self.blocks
        else:
            selected = _pick_blocks(inp.numel())
        if selected not in SUPPORTED_BLOCKS or selected > MAX_BLOCKS:
            raise ValueError(f"blocks must be one of {SUPPORTED_BLOCKS}")
        with torch.cuda.device(self.device):
            self._launcher(
                self._slab_ptrs,
                inp.data_ptr(),
                out.data_ptr(),
                self.stage_offset,
                self.part_self_offset,
                self.part_inbox_offset,
                self.final_offset,
                self.quarter_capacity,
                inp.numel(),
                selected,
                stream,
            )
        return out

    def prepare_graph(self, **kwargs: object) -> None:
        """The launcher is resolved at construction; nothing to stage before capture."""
        del kwargs
        if self._closed:
            raise RuntimeError("PCIeIsland9AllReduce is closed")

    @contextmanager
    def capture(self, **kwargs: object):
        del kwargs
        self.prepare_graph()
        self._capture_depth += 1
        try:
            yield self
        finally:
            self._capture_depth -= 1

    def prepare_channels(self, channel_ids: Sequence[str]) -> None:
        del channel_ids

    def for_stream(self, stream: object = None, *, channel_id: Optional[str] = None):
        del stream, channel_id
        return self

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self._slab_ptrs = ()
        for ptr in self._remote_ptrs:
            self._ipc.cudaIpcCloseMemHandle(ptr)
        self._remote_ptrs.clear()
        dist.barrier(group=self.group)
        if self._local_ptr:
            self._ipc.cudaFree(self._local_ptr)
            self._local_ptr = 0
        dist.barrier(group=self.group)


__all__ = ["PCIeIsland9AllReduce", "SUPPORTED_BLOCKS", "SUPPORTED_WORLD_SIZES"]
