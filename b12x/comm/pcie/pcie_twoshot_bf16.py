"""Single-rounding BF16 PCIe two-shot collective runtime.

Host-side twin of :mod:`pcie_twoshot` without the fp8 wire codec: payloads
travel as bf16 packs, are accumulated in fp32 in a fixed rank order and
rounded once.  Intended for TP decode all-reduces above the one-shot
ceiling (tens of KB) and below the DMA ring floor (MB), where NCCL ring is
the incumbent.  Graph capture follows the two-shot contract: enter
``runtime.capture()`` around ``torch.cuda.graph``. All eager launches and graph
replays from one instance must be serialized. CUDA graph capture requires a
caller-owned preallocated output for every operation. Callers must stop
submitting work before closing that instance.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ._cuda_ipc import CudaRTLibrary
from ._twoshot_bf16_cute import (
    get_twoshot_bf16_allreduce_launcher,
    get_twoshot_bf16_launcher,
    is_twoshot_bf16_allreduce_launcher_prepared,
    is_twoshot_bf16_launcher_prepared,
)
from .pcie_oneshot import (
    _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    IPC_SLAB_ALIGNMENT,
    PCIeOneshotAllReduce,
    _finish_collective_runtime_setup,
    _raise_local_cleanup_errors,
    _align_up,
    _coordinated_close_channels,
    _cuda_device_index,
    _device_guard,
    _is_current_stream_capturing,
    _normalize_device,
    _OwnedSharedBuffer,
    _require_collective_contract,
    _require_full_grid_residency,
    _run_collective_preallocation_setup,
)
from .pcie_twoshot import (
    TWOSHOT_REQUIRED_SMS,
    _MAX_BLOCKS,
    _SIGNAL_BYTES,
)

# Topologies with retained qualification evidence: TP4 (GLM-5.3 DCP4,
# docs/evidence/pcie_twoshot_bf16_sm120.md) and the Kimi-K3 TP8/TP9 PCIe rings
# (tests/comm/test_pcie_tp9_physical.py, nine RTX PRO 6000).
SUPPORTED_WORLD_SIZES = (2, 4, 8, 9)
_PACK_ELEMS = 8


def _pad_scalar_peer_ptrs(pointers, *, rank, world_size):
    """Retain every peer in the nine-slot BF16 launch ABI."""
    live = tuple(int(pointer) for pointer in pointers)
    if world_size not in SUPPORTED_WORLD_SIZES or len(live) != world_size:
        raise ValueError("invalid BF16 two-shot peer set")
    if not 0 <= rank < world_size:
        raise ValueError("BF16 two-shot rank is outside its peer set")
    return live + (live[rank],) * (9 - world_size)


@dataclass(frozen=True)
class _TwoShotBf16Layout:
    signal_bytes: int
    pack_stride: int
    reduced_offset: int
    slot_bytes: int
    slab_bytes: int


def _make_layout(max_rows: int, row_elems: int, world_size: int) -> _TwoShotBf16Layout:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"PCIeTwoShotBF16 supports world sizes {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    if max_rows <= 0 or max_rows % world_size != 0:
        raise ValueError("max_rows must be positive and divisible by world size")
    if row_elems <= 0 or row_elems % _PACK_ELEMS != 0:
        raise ValueError("row_elems must be a positive multiple of 8")
    max_rows_per_rank = max_rows // world_size
    packs_per_row = row_elems // _PACK_ELEMS
    pack_stride = _align_up(max_rows_per_rank * packs_per_row, 16)
    payload_bytes = world_size * pack_stride * 16
    # After the staged payload each slot keeps a reduced region with one
    # shard per source rank: the pull all-reduce publishes its own shard in
    # the first entry, the push all-reduce receives every peer's shard.
    reduced_offset = _align_up(payload_bytes, IPC_SLAB_ALIGNMENT)
    slot_bytes = _align_up(
        reduced_offset + world_size * pack_stride * 16, IPC_SLAB_ALIGNMENT
    )
    signal_bytes = _align_up(_SIGNAL_BYTES, IPC_SLAB_ALIGNMENT)
    return _TwoShotBf16Layout(
        signal_bytes=signal_bytes,
        pack_stride=pack_stride,
        reduced_offset=reduced_offset,
        slot_bytes=slot_bytes,
        slab_bytes=signal_bytes + 2 * slot_bytes,
    )


def _contiguous_storage_interval(tensor: torch.Tensor) -> tuple[int, int]:
    """Return the occupied byte interval of a validated contiguous tensor."""
    start = int(tensor.data_ptr())
    return start, start + int(tensor.numel()) * int(tensor.element_size())


def _require_disjoint(
    output: torch.Tensor,
    source: torch.Tensor,
    *,
    source_name: str,
) -> None:
    """Reject aliases that violate the non-coherent pull-kernel contract."""
    if output.device != source.device:
        return
    output_start, output_end = _contiguous_storage_interval(output)
    source_start, source_end = _contiguous_storage_interval(source)
    if max(output_start, source_start) < min(output_end, source_end):
        raise ValueError(f"output must not overlap {source_name}")


class PCIeTwoShotBF16:
    """Serialized single-rounding BF16 collective runtime (world sizes 2/4/8/9)."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("use PCIeTwoShotBF16.from_exchange_group()")

    @classmethod
    def _from_prepared_factory(
        cls,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        signal_ptrs: Sequence[int],
        staging_ptrs: Sequence[Sequence[int]],
        owned_buffers: Sequence[_OwnedSharedBuffer],
        ipc: CudaRTLibrary,
        exchange_group: ProcessGroup,
        max_rows: int,
        row_elems: int,
        pack_stride: int,
        reduced_offset: int,
        slot_bytes: int,
    ) -> "PCIeTwoShotBF16":
        self = object.__new__(cls)
        self.rank = rank
        self.world_size = world_size
        self.device = _normalize_device(device)
        self.exchange_group = exchange_group
        self._signal_ptrs = tuple(int(pointer) for pointer in signal_ptrs)
        self._staging_ptrs = tuple(
            tuple(int(pointer) for pointer in slot) for slot in staging_ptrs
        )
        if len(self._signal_ptrs) != 9 or (
            len(self._staging_ptrs) != 2
            or any(len(slot) != 9 for slot in self._staging_ptrs)
        ):
            raise ValueError(
                "two-shot scalar pointer ABI requires 9 peer slots and 2 slots"
            )
        self._owned_buffers = list(owned_buffers)
        self._ipc = ipc
        self.max_rows = max_rows
        self.row_elems = row_elems
        self._pack_stride = int(pack_stride)
        self._reduced_offset = int(reduced_offset)
        self._slot_bytes = int(slot_bytes)
        self._slot = 0
        self._device_slot_selection = False
        self._device_slot_bias = 0
        self._capture_context_depth = 0
        self._closed = False
        self._ipc_imports_closed = False
        self._ipc_exports_freed = False
        self._coordinated_close_complete = False
        self._wire_input = None
        self._wire_output = None
        if world_size == 9 and row_elems != _PACK_ELEMS:
            # Multi-pack rows cannot be split at pack granularity, so payloads
            # whose row count does not divide nine are padded on the wire.
            # These fixed owners keep that padding allocation-free during
            # eager launches and graph replay. Single-pack rows use the
            # balanced pack partition of the pull all-reduce instead.
            self._wire_input = torch.empty(
                (max_rows, row_elems), dtype=torch.bfloat16, device=self.device
            )
            self._wire_output = torch.empty_like(self._wire_input)
        self._closed_ipc_import_indices: set[tuple[int, int]] = set()
        # "pull": remote reads (the original single-launch kernel);
        # "push": posted remote writes. Same partition and reduction order.
        self.all_reduce_mode = "pull"
        return self

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_rows: int,
        row_elems: int,
    ) -> "PCIeTwoShotBF16":
        rank = dist.get_rank(group=exchange_group)
        world_size = dist.get_world_size(group=exchange_group)

        def validate_factory_arguments():
            device_obj = _normalize_device(device)
            normalized_max_rows = int(max_rows)
            normalized_row_elems = int(row_elems)
            if world_size not in SUPPORTED_WORLD_SIZES:
                raise ValueError(
                    f"PCIeTwoShotBF16 supports world sizes {SUPPORTED_WORLD_SIZES}, got {world_size}"
                )
            if device_obj.type != "cuda":
                raise ValueError("PCIe twoshot requires a CUDA device")
            if normalized_max_rows <= 0:
                raise ValueError("max_rows must be positive")
            if normalized_row_elems <= 0 or normalized_row_elems % _PACK_ELEMS != 0:
                raise ValueError("row_elems must be a positive multiple of 8")
            if normalized_max_rows % world_size != 0:
                raise ValueError("max_rows must be divisible by world size")
            return device_obj, normalized_max_rows, normalized_row_elems

        device_obj, max_rows, row_elems = _run_collective_preallocation_setup(
            owner="PCIe twoshot-bf16 argument validation",
            exchange_group=exchange_group,
            setup=validate_factory_arguments,
        )
        _require_full_grid_residency(
            owner="PCIe twoshot-bf16",
            required_sms=TWOSHOT_REQUIRED_SMS,
            device=device_obj,
            exchange_group=exchange_group,
        )

        def prepare():
            prepared_ipc = CudaRTLibrary()
            prepared_ipc.cudaSetDevice(_cuda_device_index(device_obj))
            return prepared_ipc, _make_layout(max_rows, row_elems, world_size)

        ipc, layout = _run_collective_preallocation_setup(
            owner="PCIe twoshot-bf16",
            exchange_group=exchange_group,
            setup=prepare,
        )
        _require_collective_contract(
            owner="PCIe twoshot-bf16 channel layout",
            exchange_group=exchange_group,
            contract=(int(max_rows), int(row_elems), layout),
        )
        shared = PCIeOneshotAllReduce._allocate_shared_buffer(
            exchange_group,
            layout.slab_bytes,
            zero_fill=True,
            ipc=ipc,
        )
        peer_ptrs = list(shared.peer_ptrs)
        signal_ptrs = _pad_scalar_peer_ptrs(peer_ptrs, rank=rank, world_size=world_size)
        staging_ptrs = (
            _pad_scalar_peer_ptrs(
                [p + layout.signal_bytes for p in peer_ptrs],
                rank=rank,
                world_size=world_size,
            ),
            _pad_scalar_peer_ptrs(
                [p + layout.signal_bytes + layout.slot_bytes for p in peer_ptrs],
                rank=rank,
                world_size=world_size,
            ),
        )
        runtime: Optional[PCIeTwoShotBF16] = None
        init_error: BaseException | None = None
        try:
            runtime = cls._from_prepared_factory(
                rank=rank,
                world_size=world_size,
                device=device_obj,
                signal_ptrs=signal_ptrs,
                staging_ptrs=staging_ptrs,
                owned_buffers=[shared],
                ipc=ipc,
                exchange_group=exchange_group,
                max_rows=max_rows,
                row_elems=row_elems,
                pack_stride=layout.pack_stride,
                reduced_offset=layout.reduced_offset,
                slot_bytes=layout.slot_bytes,
            )
        except Exception as exc:
            init_error = exc

        def detach_shared_ownership() -> None:
            if runtime is not None:
                runtime._owned_buffers.clear()

        _finish_collective_runtime_setup(
            owner="PCIe twoshot-bf16",
            exchange_group=exchange_group,
            ipc=ipc,
            shared=shared,
            local_error=init_error,
            detach_shared_ownership=detach_shared_ownership,
        )
        assert runtime is not None
        return runtime

    # ---- checks ---------------------------------------------------------

    def _check_tensor(
        self,
        tensor: torch.Tensor,
        *,
        shape: tuple[int, ...],
        name: str,
    ) -> None:
        if tensor.shape != shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != {shape}")
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on the runtime CUDA device")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bfloat16")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} must be 16-byte aligned")

    def _check(self, payload: torch.Tensor, rows: int) -> None:
        if self._closed:
            raise RuntimeError("PCIeTwoShotBF16 is closed")
        if rows > self.max_rows:
            raise ValueError("pcie_twoshot_bf16 staging capacity exceeded")
        self._check_tensor(
            payload,
            shape=(rows, self.row_elems),
            name="payload",
        )

    def _device_index(self) -> int:
        return (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )

    def accepts(self, inp: torch.Tensor) -> bool:
        """True when ``all_reduce`` can serve this tensor."""
        if (
            self._closed
            or inp.dtype != torch.bfloat16
            or not inp.is_contiguous()
            or inp.data_ptr() % 16 != 0
        ):
            return False
        if inp.device != self.device:
            return False
        numel = inp.numel()
        if numel == 0 or numel % _PACK_ELEMS != 0:
            return False
        if self._balanced_partition:
            # Pack-granular shards: the largest shard must fit the per-rank
            # reduced region, and the whole payload the staged payload region.
            # The region size follows the layout of the configured capacity
            # (a runtime built without a slab derives it the same way).
            pack_stride = getattr(self, "_pack_stride", None)
            if pack_stride is None:
                pack_stride = _make_layout(
                    self.max_rows, self.row_elems, self.world_size
                ).pack_stride
            packs = numel // _PACK_ELEMS
            largest_shard = -(-packs // self.world_size)
            return largest_shard <= pack_stride
        alignment = self.row_elems * self.world_size
        if self.world_size == 9:
            return _align_up(numel, alignment) <= self.max_rows * self.row_elems
        return numel % alignment == 0 and numel // self.row_elems <= self.max_rows

    @property
    def _balanced_partition(self) -> bool:
        """True when ``all_reduce`` splits payloads at pack granularity.

        Single-pack rows let the pull all-reduce hand each rank a contiguous
        pack range whose lengths differ by at most one, so no payload needs
        wire padding. Wider rows keep the row-aligned equal shards.
        """
        return self.row_elems == _PACK_ELEMS

    # ---- graph plumbing ---------------------------------------------------

    def prepare_graph(
        self,
        *,
        operations: Sequence[str] = ("reduce_scatter", "all_gather"),
        threads: int = 512,
    ) -> None:
        if self._closed:
            raise RuntimeError("PCIeTwoShotBF16 is closed")
        if _is_current_stream_capturing(self.device):
            raise RuntimeError(
                "prepare_graph() must be called before CUDA graph capture"
            )
        threads = int(threads)
        if threads <= 0 or threads > 512 or threads % 32 != 0:
            raise ValueError("threads must be a warp-aligned value in [32, 512]")
        requested = tuple(str(operation) for operation in operations)
        device_index = self._device_index()
        with torch.cuda.device(self.device):
            for operation in dict.fromkeys(requested):
                for slot_bias in (0, 1):
                    get_twoshot_bf16_launcher(
                        operation,
                        self.world_size,
                        self.rank,
                        True,
                        slot_bias,
                        threads,
                        self.row_elems,
                        device_index,
                    )
            for slot_bias in (0, 1):
                get_twoshot_bf16_allreduce_launcher(
                    self.world_size,
                    self.rank,
                    True,
                    slot_bias,
                    threads,
                    self.row_elems,
                    device_index,
                    self.all_reduce_mode,
                )

    @contextmanager
    def capture(
        self,
        *,
        operations: Sequence[str] = ("reduce_scatter", "all_gather"),
        threads: int = 512,
    ):
        if self._capture_context_depth:
            raise RuntimeError(
                "overlapping PCIe twoshot-bf16 capture contexts are not allowed"
            )
        requested = tuple(dict.fromkeys(str(operation) for operation in operations))
        threads = int(threads)
        self.prepare_graph(operations=requested, threads=threads)
        pending_slot_bias = (
            self._device_slot_bias if self._device_slot_selection else self._slot & 1
        )
        _require_collective_contract(
            owner="PCIe twoshot-bf16 graph slot selection",
            exchange_group=self.exchange_group,
            contract=(
                requested,
                threads,
                self._device_slot_selection,
                pending_slot_bias,
            ),
        )
        if not self._device_slot_selection:
            self._device_slot_bias = pending_slot_bias
            self._device_slot_selection = True
        self._capture_context_depth = 1
        try:
            yield self
        finally:
            self._capture_context_depth = 0

    # ---- launch -----------------------------------------------------------

    def _resolve_launch_parameters(
        self,
        operation: str,
        *,
        rows_per_rank: int,
        threads: int,
        block_limit: int,
        remainder_packs: int = 0,
    ) -> tuple[int, int, int]:
        threads = int(threads)
        if threads <= 0 or threads > 512 or threads % 32 != 0:
            raise ValueError("threads must be a warp-aligned value in [32, 512]")
        if not 0 <= int(remainder_packs) < self.world_size:
            raise ValueError("remainder_packs must be below the world size")
        shard_packs = rows_per_rank * (self.row_elems // _PACK_ELEMS)
        if remainder_packs:
            shard_packs += 1
        if shard_packs > self._pack_stride:
            raise ValueError("pcie_twoshot_bf16 staging capacity exceeded")
        if block_limit <= 0 or block_limit > _MAX_BLOCKS:
            raise ValueError(f"block_limit must be in [1, {_MAX_BLOCKS}]")
        blocks = max(
            1,
            min(int(block_limit), (shard_packs + threads - 1) // threads),
        )
        capturing = _is_current_stream_capturing(self.device)
        device_index = self._device_index()
        if capturing:
            if self._capture_context_depth <= 0:
                raise RuntimeError(
                    "cold PCIe twoshot-bf16 CUDA graph capture is not allowed; "
                    "enter runtime.capture() before torch.cuda.graph()"
                )
            if not self._device_slot_selection:
                raise RuntimeError(
                    "PCIe twoshot-bf16 graph capture has no rank-synchronized "
                    "slot selection; enter runtime.capture() on every rank"
                )
            if operation == "all_reduce":
                prepared = is_twoshot_bf16_allreduce_launcher_prepared(
                    self.world_size,
                    self.rank,
                    True,
                    self._device_slot_bias,
                    threads,
                    self.row_elems,
                    device_index,
                    self.all_reduce_mode,
                )
            else:
                prepared = is_twoshot_bf16_launcher_prepared(
                    operation,
                    self.world_size,
                    self.rank,
                    True,
                    self._device_slot_bias,
                    threads,
                    self.row_elems,
                    device_index,
                )
            if not prepared:
                raise RuntimeError(
                    "cold PCIe twoshot-bf16 CUDA graph capture is not allowed; "
                    "enter runtime.capture() before torch.cuda.graph()"
                )
        if self._device_slot_selection:
            slot = 0
        else:
            slot = self._slot % 2
            self._slot += 1
        return blocks, slot, device_index

    def _launch(
        self,
        operation: str,
        payload: torch.Tensor,
        out: torch.Tensor,
        *,
        rows_per_rank: int,
        threads: int,
        block_limit: int,
    ) -> None:
        blocks, slot, device_index = self._resolve_launch_parameters(
            operation,
            rows_per_rank=rows_per_rank,
            threads=threads,
            block_limit=block_limit,
        )
        with torch.cuda.device(self.device):
            launcher = get_twoshot_bf16_launcher(
                operation,
                self.world_size,
                self.rank,
                self._device_slot_selection,
                self._device_slot_bias,
                threads,
                self.row_elems,
                device_index,
            )
            launcher(
                payload.data_ptr(),
                self._staging_ptrs[slot],
                self._signal_ptrs,
                out.data_ptr(),
                self.rank,
                self._pack_stride,
                self._slot_bytes,
                rows_per_rank,
                blocks,
            )

    # ---- public collectives ---------------------------------------------

    def reduce_scatter(
        self,
        payload: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        threads: int = 512,
        block_limit: int = 64,
    ) -> torch.Tensor:
        with _device_guard(self.device):
            rows = payload.shape[0]
            self._check(payload, rows)
            if rows % self.world_size != 0:
                raise ValueError("rows must be divisible by world size")
            if out is None:
                if _is_current_stream_capturing(self.device):
                    raise RuntimeError(
                        "PCIeTwoShotBF16.reduce_scatter CUDA graph capture "
                        "requires a caller-owned preallocated output"
                    )
                out = torch.empty(
                    rows // self.world_size,
                    self.row_elems,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
            self._check_tensor(
                out,
                shape=(rows // self.world_size, self.row_elems),
                name="output",
            )
            _require_disjoint(out, payload, source_name="payload")
            self._launch(
                "reduce_scatter",
                payload,
                out,
                rows_per_rank=rows // self.world_size,
                threads=threads,
                block_limit=block_limit,
            )
            return out

    def all_gather(
        self,
        payload: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        threads: int = 512,
        block_limit: int = 64,
    ) -> torch.Tensor:
        with _device_guard(self.device):
            rows = payload.shape[0]
            self._check(payload, rows)
            if out is None:
                if _is_current_stream_capturing(self.device):
                    raise RuntimeError(
                        "PCIeTwoShotBF16.all_gather CUDA graph capture requires "
                        "a caller-owned preallocated output"
                    )
                out = torch.empty(
                    rows * self.world_size,
                    self.row_elems,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
            self._check_tensor(
                out,
                shape=(rows * self.world_size, self.row_elems),
                name="output",
            )
            _require_disjoint(out, payload, source_name="payload")
            self._launch(
                "all_gather",
                payload,
                out,
                rows_per_rank=rows,
                threads=threads,
                block_limit=block_limit,
            )
            return out

    def _launch_pull_all_reduce(
        self,
        payload: torch.Tensor,
        out: torch.Tensor,
        *,
        rows_per_rank: int,
        threads: int,
        block_limit: int,
        remainder_packs: int = 0,
    ) -> None:
        blocks, slot, device_index = self._resolve_launch_parameters(
            "all_reduce",
            rows_per_rank=rows_per_rank,
            threads=threads,
            block_limit=block_limit,
            remainder_packs=remainder_packs,
        )
        with torch.cuda.device(self.device):
            launcher = get_twoshot_bf16_allreduce_launcher(
                self.world_size,
                self.rank,
                self._device_slot_selection,
                self._device_slot_bias,
                threads,
                self.row_elems,
                device_index,
                self.all_reduce_mode,
            )
            launcher(
                payload.data_ptr(),
                self._staging_ptrs[slot],
                self._signal_ptrs,
                out.data_ptr(),
                self.rank,
                self._reduced_offset,
                self._slot_bytes,
                rows_per_rank,
                remainder_packs,
                blocks,
                pack_stride=self._pack_stride,
            )

    def all_reduce(
        self,
        inp: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        threads: int = 512,
        block_limit: int = 64,
    ) -> torch.Tensor:
        """FP32-accumulating BF16 all-reduce with one BF16 rounding.

        ``all_reduce_mode`` selects remote reads (``pull``) or posted remote
        writes (``push``); both give bit-identical results.
        """
        if not self.accepts(inp):
            raise ValueError("input not accepted by PCIeTwoShotBF16.all_reduce")
        logical_numel = inp.numel()
        with _device_guard(self.device):
            if out is None:
                if _is_current_stream_capturing(self.device):
                    raise RuntimeError(
                        "PCIeTwoShotBF16.all_reduce CUDA graph capture requires "
                        "a caller-owned preallocated output"
                    )
                out = torch.empty_like(inp)
            self._check_tensor(
                out,
                shape=tuple(inp.shape),
                name="output",
            )
            _require_disjoint(out, inp, source_name="input")
            if self._balanced_partition:
                packs = logical_numel // _PACK_ELEMS
                self._launch_pull_all_reduce(
                    inp.view(-1),
                    out.view(-1),
                    rows_per_rank=packs // self.world_size,
                    remainder_packs=packs % self.world_size,
                    threads=threads,
                    block_limit=block_limit,
                )
                return out
            wire_numel = _align_up(logical_numel, self.row_elems * self.world_size)
            rows = wire_numel // self.row_elems
            padded = wire_numel != logical_numel
            if padded:
                assert self._wire_input is not None and self._wire_output is not None
                payload = self._wire_input[:rows]
                payload.view(-1)[:logical_numel].copy_(inp.view(-1))
                payload.view(-1)[logical_numel:].zero_()
                out_view = self._wire_output[:rows]
            else:
                payload = inp.view(rows, self.row_elems)
                out_view = out.view(rows, self.row_elems)
            self._launch_pull_all_reduce(
                payload,
                out_view,
                rows_per_rank=rows // self.world_size,
                threads=threads,
                block_limit=block_limit,
            )
            if padded:
                out.view(-1).copy_(out_view.view(-1)[:logical_numel])
        return out

    # ---- teardown (mirrors pcie_twoshot) -----------------------------------

    def _closed_import_indices(self) -> set[tuple[int, int]]:
        closed = getattr(self, "_closed_ipc_import_indices", None)
        if closed is None:
            closed = set()
            self._closed_ipc_import_indices = closed
        return closed

    def _all_python_ipc_imports_closed(self, closed: set[tuple[int, int]]) -> bool:
        return all(
            (buffer_index, remote_index) in closed
            for buffer_index, shared in enumerate(self._owned_buffers)
            for remote_index, _ in enumerate(shared.remote_ptrs)
        )

    def _close_ipc_imports_strict(self) -> None:
        if self._ipc_imports_closed:
            return
        self._closed = True
        failures: list[tuple[str, Exception]] = []
        closed = self._closed_import_indices()
        for buffer_index, shared in enumerate(self._owned_buffers):
            for remote_index, ptr in enumerate(shared.remote_ptrs):
                key = (buffer_index, remote_index)
                if key in closed:
                    continue
                try:
                    self._ipc.cudaIpcCloseMemHandle(ptr)
                except Exception as exc:
                    failures.append((f"CUDA IPC import {ptr}", exc))
                else:
                    closed.add(key)
        if not failures and self._all_python_ipc_imports_closed(closed):
            self._ipc_imports_closed = True
        if failures:
            _raise_local_cleanup_errors(
                "PCIe twoshot-bf16", "IPC import close", failures
            )

    def _free_ipc_exports_strict(self) -> None:
        if self._ipc_exports_freed:
            return
        self._close_ipc_imports_strict()
        failures: list[tuple[str, Exception]] = []
        remaining = []
        for shared in self._owned_buffers:
            try:
                self._ipc.cudaFree(shared.local_ptr)
            except Exception as exc:
                remaining.append(shared)
                failures.append((f"CUDA IPC export {shared.local_ptr}", exc))
        self._owned_buffers = remaining
        if not remaining:
            self._ipc_exports_freed = True
        if failures:
            _raise_local_cleanup_errors(
                "PCIe twoshot-bf16", "IPC export free", failures
            )

    def close(self) -> None:
        """Synchronize submitted work and release channels on every rank.

        The caller must prevent any eager launch or graph replay from being
        submitted concurrently with or after this collective close.
        """
        if getattr(self, "_coordinated_close_complete", False):
            return
        _coordinated_close_channels(
            (self,),
            exchange_group=self.exchange_group,
            device=self.device,
        )

    def __enter__(self) -> "PCIeTwoShotBF16":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(
        self,
        _quarantine: dict[int, object] = _ABANDONED_PCIE_RUNTIME_QUARANTINE,
    ) -> None:
        if getattr(self, "_coordinated_close_complete", False):
            return
        if getattr(self, "_owned_buffers", ()):
            _quarantine[id(self)] = self


__all__ = ["PCIeTwoShotBF16"]
