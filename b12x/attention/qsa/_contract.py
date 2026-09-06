"""Capacity, binding, and fail-closed runtime contract for QSA."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor
from ...policy import PolicyContext, PolicyResolution, get_auto_policy
from ._policy import QSA_POLICY, QsaConfig, QsaQuery
from ._sparse_gqa_cute_config import MAX_SPLIT_ROWS as _MAX_SPLIT_ROWS

_ALIGN_BYTES = 256
_SCORE_WORKSPACE_LIMIT_BYTES = 128 * 1024 * 1024
_MIN_TOPK_WORKSPACE_BYTES = 1024 * 1024
_STABLE_TOPK_BLOCK = 512
_MAX_SCORE_CHUNK_GROUPS = 65536


def _align_up(value: int, alignment: int = _ALIGN_BYTES) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


def _is_power_of_two(value: int) -> bool:
    value = int(value)
    return value > 0 and value & (value - 1) == 0


def _canonical_device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and result.index is None and torch.cuda.is_available():
        result = torch.device("cuda", torch.cuda.current_device())
    return result


@dataclass(frozen=True, kw_only=True)
class CacheRequirements:
    """Pure per-page shapes and byte layout required by QSA cache storage."""

    dtype: torch.dtype
    kv_dtype: torch.dtype
    main_k_page_shape: tuple[int, int, int]
    main_v_page_shape: tuple[int, int, int]
    compressed_page_shape: tuple[int, int]
    raw_k_ring_shape: tuple[int, int]
    raw_logical_positions_shape: tuple[int]
    raw_rope_positions_shape: tuple[int, int]
    raw_interval_start_positions_shape: tuple[int]
    main_k_page_nbytes: int
    main_v_page_nbytes: int
    main_kv_page_nbytes: int
    compressed_page_nbytes: int
    raw_k_ring_offset_bytes: int
    raw_logical_positions_offset_bytes: int
    raw_rope_positions_offset_bytes: int
    raw_interval_start_positions_offset_bytes: int
    raw_page_nbytes: int
    raw_ring_capacity: int
    selection_width: int
    alignment_bytes: int
    shared_compressed_raw_storage_legal: bool


def cache_requirements(
    *,
    main_page_size: int,
    kv_heads: int = 2,
    head_dim: int = 256,
    index_head_dim: int = 128,
    compress_ratio: int = 4,
    budget: int = 2048,
    position_axes: int = 1,
    max_speculative_tokens: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    kv_dtype: torch.dtype = torch.bfloat16,
) -> CacheRequirements:
    """Describe QSA cache storage without requiring a device or pool capacity.

    The compressed page size is derived as ``main_page_size / compress_ratio``.
    Raw selector payload and metadata offsets describe their byte layout within
    a compressed/raw shared physical page.  A false
    ``shared_compressed_raw_storage_legal`` result lets an allocator reject or
    separately place a geometry before page counts and sequence limits exist.
    """

    positive = {
        "main_page_size": main_page_size,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "index_head_dim": index_head_dim,
        "compress_ratio": compress_ratio,
        "budget": budget,
    }
    for name, value in positive.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if dtype != torch.bfloat16:
        raise TypeError("QSA query and selector-cache dtype must be torch.bfloat16")
    if kv_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("QSA main KV cache dtype must be BF16 or FP8 E4M3FN")
    if not _is_power_of_two(head_dim) or int(head_dim) < 16:
        raise ValueError("head_dim must be a power of two at least 16")
    if not _is_power_of_two(index_head_dim):
        raise ValueError("index_head_dim must be a power of two")
    if int(main_page_size) % int(compress_ratio):
        raise ValueError("main_page_size must be divisible by compress_ratio")
    if int(budget) % int(compress_ratio):
        raise ValueError("budget must be divisible by compress_ratio")
    if int(budget) // int(compress_ratio) not in (512, 2048):
        raise ValueError("budget / compress_ratio must be 512 or 2048")
    if int(position_axes) not in (1, 3):
        raise ValueError("position_axes must be 1 or 3")
    if int(max_speculative_tokens) < 0:
        raise ValueError("max_speculative_tokens must be nonnegative")

    ratio = int(compress_ratio)
    raw_ring_capacity = ratio * math.ceil((ratio + int(max_speculative_tokens)) / ratio)
    if int(main_page_size) % raw_ring_capacity:
        raise ValueError("raw_ring_capacity must divide main_page_size")

    element_nbytes = dtype.itemsize
    kv_element_nbytes = kv_dtype.itemsize
    int64_nbytes = torch.int64.itemsize
    compressed_page_size = int(main_page_size) // ratio
    main_page_shape = (int(main_page_size), int(kv_heads), int(head_dim))
    main_page_nbytes = math.prod(main_page_shape) * kv_element_nbytes
    compressed_page_shape = (compressed_page_size, int(index_head_dim))
    compressed_page_nbytes = math.prod(compressed_page_shape) * element_nbytes
    raw_k_ring_shape = (raw_ring_capacity, int(index_head_dim))
    raw_k_ring_nbytes = math.prod(raw_k_ring_shape) * element_nbytes
    raw_logical_positions_shape = (raw_ring_capacity,)
    raw_logical_positions_offset_bytes = raw_k_ring_nbytes
    raw_rope_positions_shape = (raw_ring_capacity, int(position_axes))
    raw_rope_positions_offset_bytes = (
        raw_logical_positions_offset_bytes
        + math.prod(raw_logical_positions_shape) * int64_nbytes
    )
    raw_interval_start_positions_offset_bytes = (
        raw_rope_positions_offset_bytes
        + math.prod(raw_rope_positions_shape) * int64_nbytes
    )
    raw_page_nbytes = raw_interval_start_positions_offset_bytes + int64_nbytes

    return CacheRequirements(
        dtype=dtype,
        kv_dtype=kv_dtype,
        main_k_page_shape=main_page_shape,
        main_v_page_shape=main_page_shape,
        compressed_page_shape=compressed_page_shape,
        raw_k_ring_shape=raw_k_ring_shape,
        raw_logical_positions_shape=raw_logical_positions_shape,
        raw_rope_positions_shape=raw_rope_positions_shape,
        raw_interval_start_positions_shape=(1,),
        main_k_page_nbytes=main_page_nbytes,
        main_v_page_nbytes=main_page_nbytes,
        main_kv_page_nbytes=2 * main_page_nbytes,
        compressed_page_nbytes=compressed_page_nbytes,
        raw_k_ring_offset_bytes=0,
        raw_logical_positions_offset_bytes=raw_logical_positions_offset_bytes,
        raw_rope_positions_offset_bytes=raw_rope_positions_offset_bytes,
        raw_interval_start_positions_offset_bytes=(
            raw_interval_start_positions_offset_bytes
        ),
        raw_page_nbytes=raw_page_nbytes,
        raw_ring_capacity=raw_ring_capacity,
        selection_width=int(budget) + ratio - 1,
        alignment_bytes=_ALIGN_BYTES,
        shared_compressed_raw_storage_legal=(raw_page_nbytes <= compressed_page_nbytes),
    )


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Static model geometry and serving capacities for one QSA plan."""

    device: torch.device | str
    max_batch: int
    max_raw_state_slots: int
    max_q_rows: int
    max_seq_len: int
    num_main_cache_pages: int
    num_compressed_cache_pages: int
    main_page_size: int
    compressed_page_size: int
    max_speculative_tokens: int = 0
    q_heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    index_heads: int = 4
    index_kv_heads: int = 1
    index_head_dim: int = 128
    index_rotary_dim: int = 64
    compress_ratio: int = 4
    budget: int = 2048
    position_axes: int = 1
    mrope_sections: tuple[int, int, int] | None = None
    mrope_interleaved: bool = False
    rms_norm_eps: float = 1e-6
    dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _canonical_device(self.device))
        if self.device.type != "cuda":
            raise ValueError(f"QSA decode requires a CUDA device, got {self.device}")
        positive = {
            "max_batch": self.max_batch,
            "max_raw_state_slots": self.max_raw_state_slots,
            "max_q_rows": self.max_q_rows,
            "max_seq_len": self.max_seq_len,
            "num_main_cache_pages": self.num_main_cache_pages,
            "num_compressed_cache_pages": self.num_compressed_cache_pages,
            "main_page_size": self.main_page_size,
            "compressed_page_size": self.compressed_page_size,
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "index_heads": self.index_heads,
            "index_kv_heads": self.index_kv_heads,
            "index_head_dim": self.index_head_dim,
            "index_rotary_dim": self.index_rotary_dim,
            "compress_ratio": self.compress_ratio,
            "budget": self.budget,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.max_speculative_tokens) < 0:
            raise ValueError("max_speculative_tokens must be nonnegative")
        if not math.isfinite(float(self.rms_norm_eps)) or float(self.rms_norm_eps) <= 0:
            raise ValueError("rms_norm_eps must be finite and positive")
        if int(self.max_raw_state_slots) < int(self.max_batch):
            raise ValueError("max_raw_state_slots must be at least max_batch")
        if int(self.max_q_rows) < int(self.max_batch):
            raise ValueError("max_q_rows must be at least max_batch")
        if int(self.max_seq_len) < int(self.compress_ratio):
            raise ValueError("max_seq_len must be at least compress_ratio")
        if int(self.max_seq_len) > torch.iinfo(torch.int32).max:
            raise ValueError("max_seq_len must fit in positive int32 positions")
        max_page_count = torch.iinfo(torch.int32).max + 1
        if int(self.num_main_cache_pages) > max_page_count:
            raise ValueError("num_main_cache_pages must fit in nonnegative int32 IDs")
        if int(self.num_compressed_cache_pages) > max_page_count:
            raise ValueError(
                "num_compressed_cache_pages must fit in nonnegative int32 IDs"
            )
        if self.dtype != torch.bfloat16:
            raise TypeError("QSA query and selector-cache dtype must be torch.bfloat16")
        if self.kv_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise TypeError("QSA main KV cache dtype must be BF16 or FP8 E4M3FN")
        if int(self.index_kv_heads) != 1:
            raise ValueError("QSA requires exactly one index KV head")
        if int(self.q_heads) % int(self.kv_heads):
            raise ValueError("q_heads must be divisible by kv_heads")
        from ._sparse_gqa_cute_config import (
            BLOCK_N as QWEN_CUTE_BLOCK_N,
            NUM_SPLITS as QWEN_CUTE_NUM_SPLITS,
            is_qwen_geometry,
        )

        if not is_qwen_geometry(
            q_heads=int(self.q_heads),
            kv_heads=int(self.kv_heads),
            head_dim=int(self.head_dim),
            selection_width=int(self.selection_width),
            block_n=QWEN_CUTE_BLOCK_N,
            splits=QWEN_CUTE_NUM_SPLITS,
        ):
            raise NotImplementedError(
                "QSA requires the CuTe Qwen sparse-GQA geometry "
                "with q_heads divisible by kv_heads, head_dim=256, and "
                "selection_width>=2051"
            )
        if not _is_power_of_two(self.head_dim) or int(self.head_dim) < 16:
            raise ValueError("head_dim must be a power of two at least 16")
        if not _is_power_of_two(self.index_head_dim):
            raise ValueError("index_head_dim must be a power of two")
        if int(self.index_rotary_dim) > int(self.index_head_dim):
            raise ValueError("index_rotary_dim cannot exceed index_head_dim")
        if int(self.index_rotary_dim) % 2:
            raise ValueError("index_rotary_dim must be even")
        if int(self.budget) % int(self.compress_ratio):
            raise ValueError("budget must be divisible by compress_ratio")
        if self.group_budget not in (512, 2048):
            raise ValueError("budget / compress_ratio must be 512 or 2048")
        if int(self.main_page_size) % int(self.compress_ratio):
            raise ValueError("main_page_size must be divisible by compress_ratio")
        if int(self.compressed_page_size) != (
            int(self.main_page_size) // int(self.compress_ratio)
        ):
            raise ValueError(
                "compressed_page_size must equal main_page_size / compress_ratio"
            )
        if int(self.main_page_size) % int(self.raw_ring_capacity):
            raise ValueError("raw_ring_capacity must divide main_page_size")
        if int(self.num_main_cache_pages) * int(self.main_page_size) < int(
            self.max_seq_len
        ):
            raise ValueError("main cache capacity cannot cover max_seq_len")
        if (
            int(self.num_compressed_cache_pages) * int(self.compressed_page_size)
            < self.max_groups
        ):
            raise ValueError("compressed cache capacity cannot cover max_seq_len")
        if int(self.position_axes) not in (1, 3):
            raise ValueError("position_axes must be 1 or 3")
        if int(self.position_axes) == 1:
            if self.mrope_sections is not None:
                raise ValueError("mrope_sections must be None for scalar positions")
            if self.mrope_interleaved:
                raise ValueError("mrope_interleaved must be false for scalar positions")
        else:
            if self.mrope_sections is None or len(self.mrope_sections) != 3:
                raise ValueError("three-axis positions require three mrope_sections")
            if any(int(section) <= 0 for section in self.mrope_sections):
                raise ValueError("mrope_sections must be positive")
            if sum(map(int, self.mrope_sections)) != int(self.index_rotary_dim) // 2:
                raise ValueError("mrope_sections must sum to half of index_rotary_dim")
            if self.mrope_interleaved:
                half = int(self.index_rotary_dim) // 2
                expected_sections = ((half + 2) // 3, (half + 1) // 3, half // 3)
                if tuple(map(int, self.mrope_sections)) != expected_sections:
                    raise ValueError(
                        "interleaved mrope_sections must match the round-robin "
                        f"axis counts {expected_sections}"
                    )

    @property
    def group_budget(self) -> int:
        return int(self.budget) // int(self.compress_ratio)

    @property
    def cache_requirements(self) -> CacheRequirements:
        return cache_requirements(
            main_page_size=int(self.main_page_size),
            kv_heads=int(self.kv_heads),
            head_dim=int(self.head_dim),
            index_head_dim=int(self.index_head_dim),
            compress_ratio=int(self.compress_ratio),
            budget=int(self.budget),
            position_axes=int(self.position_axes),
            max_speculative_tokens=int(self.max_speculative_tokens),
            dtype=self.dtype,
            kv_dtype=self.kv_dtype,
        )

    @property
    def selection_width(self) -> int:
        return self.cache_requirements.selection_width

    @property
    def raw_ring_capacity(self) -> int:
        return self.cache_requirements.raw_ring_capacity

    @property
    def compressed_page_nbytes(self) -> int:
        return self.cache_requirements.compressed_page_nbytes

    @property
    def raw_page_nbytes(self) -> int:
        return self.cache_requirements.raw_page_nbytes

    @property
    def max_groups(self) -> int:
        return int(self.max_seq_len) // int(self.compress_ratio)

    @property
    def main_table_width(self) -> int:
        return math.ceil(int(self.max_seq_len) / int(self.main_page_size))

    @property
    def compressed_table_width(self) -> int:
        return math.ceil(self.max_groups / int(self.compressed_page_size))


@dataclass(frozen=True)
class _ScratchLayout:
    prepared_query_offset_bytes: int
    prepared_query_nbytes: int
    score_offset_bytes: int
    score_nbytes: int
    eligible_counts_offset_bytes: int
    eligible_counts_nbytes: int
    merge_lengths_offset_bytes: int
    merge_lengths_nbytes: int
    topk_values_offset_bytes: int
    topk_values_nbytes: int
    topk_indices_offset_bytes: int
    topk_indices_nbytes: int
    topk_values_b_offset_bytes: int
    topk_values_b_nbytes: int
    topk_indices_b_offset_bytes: int
    topk_indices_b_nbytes: int
    state_errors_offset_bytes: int
    state_errors_nbytes: int
    request_errors_offset_bytes: int
    request_errors_nbytes: int
    topk_offset_bytes: int
    topk_nbytes: int
    partial_output_offset_bytes: int
    partial_output_nbytes: int
    partial_lse_offset_bytes: int
    partial_lse_nbytes: int
    work_metadata_offset_bytes: int
    work_metadata_nbytes: int
    total_nbytes: int


@dataclass(frozen=True)
class Plan:
    """Fixed-capacity QSA policy and caller-allocated scratch contract."""

    caps: Caps
    workspace_q_rows: int
    score_chunk_groups: int
    score_workspace_width: int
    num_score_chunks: int
    max_split_row_product: int
    policy_resolution: PolicyResolution[QsaConfig]
    _layout: _ScratchLayout
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(self, **kwargs: object) -> Binding:
        return bind(self, **kwargs)


@dataclass(frozen=True)
class Binding:
    """Caller-owned QSA cache, state, output, and scratch views.

    The main K/V cache and its block table are read-only. Compressed-cache and
    raw-ring tensors are mutable decode state; ``output`` and
    ``selected_positions`` are caller-owned result buffers.
    """

    plan: Plan
    shared_compressed_raw_pool: bool
    scratch: torch.Tensor
    main_k_cache: torch.Tensor
    main_v_cache: torch.Tensor
    k_descale: torch.Tensor | None
    v_descale: torch.Tensor | None
    main_block_table: torch.Tensor
    compressed_k_cache: torch.Tensor
    compressed_block_table: torch.Tensor
    raw_k_ring: torch.Tensor
    raw_logical_positions: torch.Tensor
    raw_rope_positions: torch.Tensor
    raw_interval_start_positions: torch.Tensor
    raw_state_slot_ids: torch.Tensor
    index_q_norm_weight: torch.Tensor
    index_k_norm_weight: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    output: torch.Tensor
    selected_positions: torch.Tensor
    prepared_index_query: torch.Tensor
    scores: torch.Tensor
    eligible_group_counts: torch.Tensor
    merge_lengths: torch.Tensor
    topk_values: torch.Tensor
    topk_group_ids: torch.Tensor
    topk_values_b: torch.Tensor
    topk_group_ids_b: torch.Tensor
    state_errors: torch.Tensor
    request_errors: torch.Tensor
    partial_output: torch.Tensor
    partial_lse: torch.Tensor
    selection_stream: torch.cuda.Stream | None = None
    _selection_done: torch.cuda.Event | None = None


@dataclass(frozen=True)
class _KernelCaps:
    """Scalar-only launch contract reconstructed inside the opaque op."""

    max_batch: int
    max_raw_state_slots: int
    max_seq_len: int
    main_page_size: int
    compressed_page_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    index_heads: int
    index_head_dim: int
    index_rotary_dim: int
    compress_ratio: int
    budget: int
    position_axes: int
    mrope_sections: tuple[int, int, int] | None
    mrope_interleaved: bool
    rms_norm_eps: float
    raw_ring_capacity: int
    max_speculative_tokens: int

    @property
    def group_budget(self) -> int:
        return int(self.budget) // int(self.compress_ratio)

    @property
    def selection_width(self) -> int:
        return int(self.budget) + int(self.compress_ratio) - 1

    @property
    def max_groups(self) -> int:
        return int(self.max_seq_len) // int(self.compress_ratio)


def _qwen_row_splits(rows: int) -> int:
    return (
        64 if rows == 1 else 32 if rows <= 4 else 16 if rows <= _MAX_SPLIT_ROWS else 1
    )


def _target_splits(caps: Caps, rows: int) -> tuple[int, int]:
    from ._sparse_gqa_cute_config import (
        BLOCK_N as QWEN_CUTE_BLOCK_N,
        NUM_SPLITS as QWEN_CUTE_NUM_SPLITS,
        is_qwen_geometry,
    )

    if is_qwen_geometry(
        q_heads=int(caps.q_heads),
        kv_heads=int(caps.kv_heads),
        head_dim=int(caps.head_dim),
        selection_width=int(caps.selection_width),
        block_n=QWEN_CUTE_BLOCK_N,
        splits=QWEN_CUTE_NUM_SPLITS,
    ):
        return QWEN_CUTE_BLOCK_N, _qwen_row_splits(int(rows))
    raise NotImplementedError(
        "QSA requires the CuTe Qwen sparse-GQA geometry: q_heads divisible by "
        "kv_heads, head_dim=256, selection_width>=2051; "
        f"got q_heads={caps.q_heads}, "
        f"kv_heads={caps.kv_heads}, head_dim={caps.head_dim}, "
        f"main_page_size={caps.main_page_size}, "
        f"selection_width={caps.selection_width}"
    )


def _scratch_layout(
    caps: Caps,
) -> tuple[_ScratchLayout, int, int, int, int, int]:
    workspace_q_rows = int(caps.max_q_rows)
    score_width_limit = max(
        1,
        _SCORE_WORKSPACE_LIMIT_BYTES // (workspace_q_rows * torch.float32.itemsize),
    )
    score_width_limit = min(
        score_width_limit,
        int(caps.group_budget) + _MAX_SCORE_CHUNK_GROUPS,
    )
    if int(caps.max_groups) <= score_width_limit:
        score_chunk_groups = int(caps.max_groups)
        score_workspace_width = score_chunk_groups
    else:
        score_chunk_groups = score_width_limit - int(caps.group_budget)
        if score_chunk_groups <= 0:
            raise ValueError(
                "QSA 128 MiB score workspace cannot hold one top-k carry row"
            )
        score_workspace_width = int(caps.group_budget) + score_chunk_groups
    num_score_chunks = math.ceil(int(caps.max_groups) / score_chunk_groups)
    prepared_query_nbytes = (
        workspace_q_rows
        * int(caps.index_heads)
        * int(caps.index_head_dim)
        * torch.bfloat16.itemsize
    )
    score_nbytes = workspace_q_rows * score_workspace_width * torch.float32.itemsize
    from ._sparse_gqa_cute_config import MAX_SPLIT_ROWS

    max_split_row_product = max(
        rows * _target_splits(caps, rows)[1]
        for rows in range(1, min(workspace_q_rows, MAX_SPLIT_ROWS) + 1)
    )
    partial_output_nbytes = (
        max_split_row_product
        * int(caps.q_heads)
        * int(caps.head_dim)
        * torch.float32.itemsize
    )
    partial_lse_nbytes = (
        max_split_row_product * int(caps.q_heads) * torch.float32.itemsize
    )
    work_metadata_nbytes = max(
        workspace_q_rows * 8 * torch.int64.itemsize,
        (int(caps.num_compressed_cache_pages) + 1) * torch.int32.itemsize,
    )
    eligible_counts_nbytes = workspace_q_rows * torch.int32.itemsize
    merge_lengths_nbytes = workspace_q_rows * torch.int32.itemsize
    topk_values_nbytes = (
        workspace_q_rows * int(caps.group_budget) * torch.float32.itemsize
    )
    topk_indices_nbytes = (
        workspace_q_rows * int(caps.group_budget) * torch.int32.itemsize
    )
    state_errors_nbytes = int(caps.max_q_rows) * torch.int32.itemsize
    # One error word per request plus one global packed-boundary word.
    request_errors_nbytes = (int(caps.max_batch) + 1) * torch.int32.itemsize
    stable_topk_blocks = math.ceil(score_workspace_width / _STABLE_TOPK_BLOCK)
    stable_topk_nbytes = (
        2 * workspace_q_rows * stable_topk_blocks * torch.int32.itemsize
        + workspace_q_rows
        * int(caps.group_budget)
        * (torch.float32.itemsize + torch.int32.itemsize)
        + workspace_q_rows * (torch.float32.itemsize + torch.int32.itemsize)
    )
    topk_workspace_nbytes = _align_up(
        max(_MIN_TOPK_WORKSPACE_BYTES, stable_topk_nbytes)
    )

    offset = 0
    prepared_query_offset = _align_up(offset)
    offset = prepared_query_offset + prepared_query_nbytes
    score_offset = _align_up(offset)
    offset = score_offset + score_nbytes
    eligible_counts_offset = _align_up(offset)
    offset = eligible_counts_offset + eligible_counts_nbytes
    merge_lengths_offset = _align_up(offset)
    offset = merge_lengths_offset + merge_lengths_nbytes
    topk_values_offset = _align_up(offset)
    offset = topk_values_offset + topk_values_nbytes
    topk_indices_offset = _align_up(offset)
    offset = topk_indices_offset + topk_indices_nbytes
    topk_values_b_offset = _align_up(offset)
    offset = topk_values_b_offset + topk_values_nbytes
    topk_indices_b_offset = _align_up(offset)
    offset = topk_indices_b_offset + topk_indices_nbytes
    state_errors_offset = _align_up(offset)
    offset = state_errors_offset + state_errors_nbytes
    request_errors_offset = _align_up(offset)
    offset = request_errors_offset + request_errors_nbytes
    topk_offset = _align_up(offset)
    offset = topk_offset + topk_workspace_nbytes
    partial_output_offset = _align_up(offset)
    offset = partial_output_offset + partial_output_nbytes
    partial_lse_offset = _align_up(offset)
    offset = partial_lse_offset + partial_lse_nbytes
    work_metadata_offset = _align_up(offset)
    offset = work_metadata_offset + work_metadata_nbytes
    total_nbytes = _align_up(offset)
    return (
        _ScratchLayout(
            prepared_query_offset_bytes=prepared_query_offset,
            prepared_query_nbytes=prepared_query_nbytes,
            score_offset_bytes=score_offset,
            score_nbytes=score_nbytes,
            eligible_counts_offset_bytes=eligible_counts_offset,
            eligible_counts_nbytes=eligible_counts_nbytes,
            merge_lengths_offset_bytes=merge_lengths_offset,
            merge_lengths_nbytes=merge_lengths_nbytes,
            topk_values_offset_bytes=topk_values_offset,
            topk_values_nbytes=topk_values_nbytes,
            topk_indices_offset_bytes=topk_indices_offset,
            topk_indices_nbytes=topk_indices_nbytes,
            topk_values_b_offset_bytes=topk_values_b_offset,
            topk_values_b_nbytes=topk_values_nbytes,
            topk_indices_b_offset_bytes=topk_indices_b_offset,
            topk_indices_b_nbytes=topk_indices_nbytes,
            state_errors_offset_bytes=state_errors_offset,
            state_errors_nbytes=state_errors_nbytes,
            request_errors_offset_bytes=request_errors_offset,
            request_errors_nbytes=request_errors_nbytes,
            topk_offset_bytes=topk_offset,
            topk_nbytes=topk_workspace_nbytes,
            partial_output_offset_bytes=partial_output_offset,
            partial_output_nbytes=partial_output_nbytes,
            partial_lse_offset_bytes=partial_lse_offset,
            partial_lse_nbytes=partial_lse_nbytes,
            work_metadata_offset_bytes=work_metadata_offset,
            work_metadata_nbytes=work_metadata_nbytes,
            total_nbytes=total_nbytes,
        ),
        score_chunk_groups,
        score_workspace_width,
        num_score_chunks,
        max_split_row_product,
        workspace_q_rows,
    )


def plan(caps: Caps, *, policy: PolicyContext | None = None) -> Plan:
    """Plan QSA policy and one caller-owned scratch allocation."""
    if not isinstance(caps, Caps):
        raise TypeError("caps must be qsa.Caps")
    policy = policy or get_auto_policy(caps.device)
    if not isinstance(policy, PolicyContext):
        raise TypeError("policy must be a PolicyContext")
    policy.require_device(caps.device)
    resolution = policy.resolve(
        QSA_POLICY,
        QsaQuery(
            q_dtype=str(caps.dtype).removeprefix("torch."),
            kv_dtype=str(caps.kv_dtype).removeprefix("torch."),
            q_heads=caps.q_heads,
            kv_heads=caps.kv_heads,
            head_dim=caps.head_dim,
            index_heads=caps.index_heads,
            index_kv_heads=caps.index_kv_heads,
            index_head_dim=caps.index_head_dim,
            index_rotary_dim=caps.index_rotary_dim,
            main_page_size=caps.main_page_size,
            max_batch=caps.max_batch,
            max_q_rows=caps.max_q_rows,
            max_seq_len=caps.max_seq_len,
            max_speculative_tokens=caps.max_speculative_tokens,
            compress_ratio=caps.compress_ratio,
            budget=caps.budget,
            position_axes=caps.position_axes,
            mrope_interleaved=caps.mrope_interleaved,
        ),
    )
    (
        layout,
        score_chunk_groups,
        score_workspace_width,
        num_score_chunks,
        max_split_row_product,
        workspace_q_rows,
    ) = _scratch_layout(caps)
    return Plan(
        caps=caps,
        workspace_q_rows=workspace_q_rows,
        score_chunk_groups=score_chunk_groups,
        score_workspace_width=score_workspace_width,
        num_score_chunks=num_score_chunks,
        max_split_row_product=max_split_row_product,
        policy_resolution=resolution,
        _layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "qsa.scratch",
                nbytes=layout.total_nbytes,
                device=caps.device,
            ),
        ),
    )


def _check_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | tuple[torch.dtype, ...] | None = None,
    contiguous: bool = False,
    unit_inner_stride: bool = False,
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} device {tensor.device} does not match {device}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if dtype is not None:
        dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
        if tensor.dtype not in dtypes:
            raise TypeError(f"{name} must have dtype in {dtypes}, got {tensor.dtype}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if unit_inner_stride and (tensor.ndim == 0 or int(tensor.stride(-1)) != 1):
        raise ValueError(f"{name} must have unit inner stride")


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    """Return the bounding byte interval of a validated non-overlapping view."""
    start = int(tensor.untyped_storage().data_ptr()) + int(
        tensor.storage_offset()
    ) * int(tensor.element_size())
    extent = 0
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        if int(size) > 1:
            extent += (int(size) - 1) * int(stride)
    return start, start + (extent + 1) * int(tensor.element_size())


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.device != right.device:
        return False
    left_start, left_end = _byte_interval(left)
    right_start, right_end = _byte_interval(right)
    return left_start < right_end and right_start < left_end


def _slot_views_are_disjoint(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Prove disjoint live regions for two views embedded in shared pages."""
    if not _overlaps(left, right):
        return True
    if left.ndim == 0 or right.ndim == 0:
        return False
    left_stride = int(left.stride(0)) * int(left.element_size())
    right_stride = int(right.stride(0)) * int(right.element_size())
    if left_stride <= 0 or left_stride != right_stride:
        return False

    def slot_interval(tensor: torch.Tensor) -> tuple[int, int]:
        start = int(tensor.untyped_storage().data_ptr()) + int(
            tensor.storage_offset()
        ) * int(tensor.element_size())
        extent = 0
        for size, stride in zip(tensor.shape[1:], tensor.stride()[1:], strict=True):
            if int(size) > 1:
                extent += (int(size) - 1) * int(stride)
        return start, start + (extent + 1) * int(tensor.element_size())

    left_start, left_end = slot_interval(left)
    right_start, right_end = slot_interval(right)
    if left_end - left_start > left_stride or right_end - right_start > left_stride:
        return False
    delta = right_start - left_start
    center = (-delta) // left_stride
    left_count = int(left.shape[0])
    right_count = int(right.shape[0])
    for difference in range(center - 2, center + 3):
        if difference < -(left_count - 1) or difference > right_count - 1:
            continue
        shifted_start = right_start + difference * left_stride
        shifted_end = right_end + difference * left_stride
        if left_start < shifted_end and shifted_start < left_end:
            return False
    return True


def _require_non_overlapping_layout(name: str, tensor: torch.Tensor) -> None:
    """Reject internal aliases while permitting ordinary padded layouts."""
    span = 1
    dimensions = sorted(
        (int(stride), int(size))
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        if int(size) > 1
    )
    for stride, size in dimensions:
        if stride < span:
            raise ValueError(f"{name} must not have internal storage overlap")
        span += (size - 1) * stride


def _require_mutation_alias_contract(
    *,
    mutable: tuple[tuple[str, torch.Tensor], ...],
    read_only: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    for name, tensor in (*mutable, *read_only):
        _require_non_overlapping_layout(name, tensor)
    for index, (left_name, left) in enumerate(mutable):
        for right_name, right in mutable[index + 1 :]:
            if _overlaps(left, right):
                names = {left_name, right_name}
                raw_names = {
                    "raw_k_ring",
                    "raw_logical_positions",
                    "raw_rope_positions",
                    "raw_interval_start_positions",
                }
                if "compressed_k_cache" in names and names & raw_names:
                    continue
                if names <= raw_names and _slot_views_are_disjoint(left, right):
                    continue
                raise ValueError(
                    f"mutable buffers {left_name} and {right_name} must not overlap"
                )
        for right_name, right in read_only:
            if _overlaps(left, right):
                if _slot_views_are_disjoint(left, right):
                    continue
                raise ValueError(
                    f"mutable buffer {left_name} must not overlap read-only "
                    f"tensor {right_name}"
                )


def _validate_shared_compressed_raw_layout(
    *,
    caps: Caps,
    compressed_k_cache: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
) -> None:
    raw_tensors = (
        raw_k_ring,
        raw_logical_positions,
        raw_rope_positions,
        raw_interval_start_positions,
    )
    shared = tuple(_overlaps(compressed_k_cache, tensor) for tensor in raw_tensors)
    if not any(shared):
        return
    if not all(shared):
        raise ValueError(
            "shared compressed/raw storage must include key, logical-position, "
            "RoPE-position, and interval-start-position views"
        )
    page_stride = int(compressed_k_cache.stride(0)) * int(
        compressed_k_cache.element_size()
    )
    if int(caps.raw_page_nbytes) > page_stride:
        raise ValueError(
            "raw ring page bytes, including int64 logical and RoPE metadata, "
            "must fit inside one aliased compressed cache page"
        )
    expected_compressed_strides = (
        int(caps.compressed_page_size) * int(caps.index_head_dim),
        int(caps.index_head_dim),
        1,
    )
    if (
        page_stride != int(caps.compressed_page_nbytes)
        or tuple(compressed_k_cache.stride()) != expected_compressed_strides
    ):
        raise ValueError(
            "shared compressed/raw storage requires dense contiguous compressed pages"
        )
    if int(raw_k_ring.shape[0]) > int(compressed_k_cache.shape[0]):
        raise ValueError(
            "shared raw state slots cannot exceed compressed physical pages"
        )
    for name, tensor in zip(
        (
            "raw_k_ring",
            "raw_logical_positions",
            "raw_rope_positions",
            "raw_interval_start_positions",
        ),
        raw_tensors,
        strict=True,
    ):
        tensor_page_stride = int(tensor.stride(0)) * int(tensor.element_size())
        if tensor_page_stride != page_stride:
            raise ValueError(
                f"{name} page stride must match compressed page stride in bytes"
            )

    expected_raw_strides = (
        page_stride // torch.bfloat16.itemsize,
        int(caps.index_head_dim),
        1,
    )
    expected_tag_strides = (page_stride // torch.int64.itemsize, 1)
    expected_rope_strides = (
        page_stride // torch.int64.itemsize,
        int(caps.position_axes),
        1,
    )
    expected_interval_start_strides = (page_stride // torch.int64.itemsize,)
    if tuple(raw_k_ring.stride()) != expected_raw_strides:
        raise ValueError("shared raw_k_ring must use the packed raw-page layout")
    if tuple(raw_logical_positions.stride()) != expected_tag_strides:
        raise ValueError(
            "shared raw_logical_positions must use the packed raw-page tail layout"
        )
    if tuple(raw_rope_positions.stride()) != expected_rope_strides:
        raise ValueError(
            "shared raw_rope_positions must use the packed raw-page tail layout"
        )
    if tuple(raw_interval_start_positions.stride()) != expected_interval_start_strides:
        raise ValueError(
            "shared raw_interval_start_positions must use the packed raw-page "
            "tail layout"
        )

    compressed_start, _ = _byte_interval(compressed_k_cache[:1])
    raw_start, _ = _byte_interval(raw_k_ring[:1])
    tags_start, _ = _byte_interval(raw_logical_positions[:1])
    rope_start, _ = _byte_interval(raw_rope_positions[:1])
    interval_start, _ = _byte_interval(raw_interval_start_positions[:1])
    payload_nbytes = (
        int(caps.raw_ring_capacity) * int(caps.index_head_dim) * torch.bfloat16.itemsize
    )
    tags_nbytes = int(caps.raw_ring_capacity) * torch.int64.itemsize
    rope_nbytes = (
        int(caps.raw_ring_capacity) * int(caps.position_axes) * torch.int64.itemsize
    )
    expected_starts = (
        compressed_start,
        compressed_start + payload_nbytes,
        compressed_start + payload_nbytes + tags_nbytes,
        compressed_start + payload_nbytes + tags_nbytes + rope_nbytes,
    )
    if (raw_start, tags_start, rope_start, interval_start) != expected_starts:
        raise ValueError(
            "shared raw key and int64 state views must use their named "
            "page-tail offsets"
        )


def _scratch_view(
    storage: torch.Tensor,
    *,
    offset_bytes: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    elements = math.prod(shape)
    nbytes = elements * dtype.itemsize
    return storage.narrow(0, int(offset_bytes), int(nbytes)).view(dtype).view(shape)


def bind(
    plan: Plan,
    *,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    main_block_table: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    output: torch.Tensor,
    selected_positions: torch.Tensor,
    selection_stream: torch.cuda.Stream | None = None,
) -> Binding:
    """Bind runtime storage and optional selection-stream resources.

    ``rope_cos`` and ``rope_sin`` may be positive-row-stride views with unit
    inner stride.  This admits zero-copy slices of a combined cosine/sine
    table such as ``cos_sin_cache[:, :rotary_half]`` and
    ``cos_sin_cache[:, rotary_half:]``.

    ``output`` may have fewer rows than the plan's capacity. Every execution
    must fit its live query rows in that caller-owned contiguous output.

    With ``selection_stream``, :func:`run` requires an ``index_ready`` event
    covering all selector inputs, metadata, and bound state. Bind and prewarm
    before graph capture. Main Q/K/V producers execute on the calling stream;
    run joins selection before attention. Calls sharing state must be ordered.
    """
    if not isinstance(plan, Plan):
        raise TypeError("plan must be a qsa.Plan")
    caps = plan.caps
    scratch_storage = scratch_tensor(
        scratch,
        plan.scratch_specs(),
        owner="qsa",
    )
    if main_k_cache.ndim != 4 or tuple(main_k_cache.shape[1:]) != (
        int(caps.main_page_size),
        int(caps.kv_heads),
        int(caps.head_dim),
    ):
        raise ValueError(
            "main_k_cache must have shape "
            f"[pages, {caps.main_page_size}, {caps.kv_heads}, {caps.head_dim}]"
        )
    if not 0 < int(main_k_cache.shape[0]) <= int(caps.num_main_cache_pages):
        raise ValueError("main_k_cache page count exceeds planned capacity")
    _check_tensor(
        main_k_cache,
        name="main_k_cache",
        device=caps.device,
        dtype=caps.kv_dtype,
        unit_inner_stride=True,
    )
    _check_tensor(
        main_v_cache,
        name="main_v_cache",
        device=caps.device,
        shape=tuple(main_k_cache.shape),
        dtype=caps.kv_dtype,
        unit_inner_stride=True,
    )
    fp8_kv = caps.kv_dtype == torch.float8_e4m3fn
    if fp8_kv and (k_descale is None or v_descale is None):
        raise ValueError("FP8 QSA main caches require k_descale and v_descale")
    for descale, name in ((k_descale, "k_descale"), (v_descale, "v_descale")):
        if descale is None:
            continue
        _check_tensor(
            descale,
            name=name,
            device=caps.device,
            dtype=torch.float32,
            contiguous=True,
        )
        if descale.numel() != 1:
            raise ValueError(f"{name} must contain exactly one per-layer scale")
    if main_block_table.ndim != 2 or tuple(main_block_table.shape) != (
        int(caps.max_batch),
        int(caps.main_table_width),
    ):
        raise ValueError(
            "main_block_table must have shape "
            f"({caps.max_batch}, {caps.main_table_width})"
        )
    _check_tensor(
        main_block_table,
        name="main_block_table",
        device=caps.device,
        dtype=torch.int32,
        contiguous=True,
    )
    expected_compressed_tail = (
        int(caps.compressed_page_size),
        int(caps.index_head_dim),
    )
    if compressed_k_cache.ndim != 3 or tuple(compressed_k_cache.shape[1:]) != (
        expected_compressed_tail
    ):
        raise ValueError(
            "compressed_k_cache must have shape "
            f"[pages, {caps.compressed_page_size}, {caps.index_head_dim}]"
        )
    if not 0 < int(compressed_k_cache.shape[0]) <= int(caps.num_compressed_cache_pages):
        raise ValueError("compressed_k_cache page count exceeds planned capacity")
    _check_tensor(
        compressed_k_cache,
        name="compressed_k_cache",
        device=caps.device,
        dtype=caps.dtype,
        unit_inner_stride=True,
    )
    _check_tensor(
        compressed_block_table,
        name="compressed_block_table",
        device=caps.device,
        shape=(int(caps.max_batch), int(caps.compressed_table_width)),
        dtype=torch.int32,
        contiguous=True,
    )
    raw_shape = (
        int(caps.max_raw_state_slots),
        int(caps.raw_ring_capacity),
    )
    _check_tensor(
        raw_k_ring,
        name="raw_k_ring",
        device=caps.device,
        shape=(*raw_shape, int(caps.index_head_dim)),
        dtype=caps.dtype,
        unit_inner_stride=True,
    )
    _check_tensor(
        raw_logical_positions,
        name="raw_logical_positions",
        device=caps.device,
        shape=raw_shape,
        dtype=torch.int64,
        unit_inner_stride=True,
    )
    _check_tensor(
        raw_rope_positions,
        name="raw_rope_positions",
        device=caps.device,
        shape=(*raw_shape, int(caps.position_axes)),
        dtype=torch.int64,
        unit_inner_stride=True,
    )
    _check_tensor(
        raw_interval_start_positions,
        name="raw_interval_start_positions",
        device=caps.device,
        shape=(int(caps.max_raw_state_slots),),
        dtype=torch.int64,
    )
    _check_tensor(
        raw_state_slot_ids,
        name="raw_state_slot_ids",
        device=caps.device,
        shape=(int(caps.max_batch),),
        dtype=(torch.int32, torch.int64),
    )
    if int(raw_state_slot_ids.stride(0)) <= 0:
        raise ValueError("raw_state_slot_ids must have a positive stride")
    norm_dtype = (torch.bfloat16, torch.float32)
    _check_tensor(
        index_q_norm_weight,
        name="index_q_norm_weight",
        device=caps.device,
        shape=(int(caps.index_head_dim),),
        dtype=norm_dtype,
        contiguous=True,
    )
    _check_tensor(
        index_k_norm_weight,
        name="index_k_norm_weight",
        device=caps.device,
        shape=(int(caps.index_head_dim),),
        dtype=norm_dtype,
        contiguous=True,
    )
    rope_width = int(caps.index_rotary_dim) // 2
    if (
        rope_cos.ndim != 2
        or int(rope_cos.shape[0]) <= 0
        or int(rope_cos.shape[1]) != rope_width
    ):
        raise ValueError(f"rope_cos must have shape [positions, {rope_width}]")
    _check_tensor(
        rope_cos,
        name="rope_cos",
        device=caps.device,
        dtype=(torch.bfloat16, torch.float32),
        unit_inner_stride=True,
    )
    if int(rope_cos.stride(0)) <= 0:
        raise ValueError("rope_cos must have a positive row stride")
    _check_tensor(
        rope_sin,
        name="rope_sin",
        device=caps.device,
        shape=tuple(rope_cos.shape),
        dtype=rope_cos.dtype,
        unit_inner_stride=True,
    )
    if int(rope_sin.stride(0)) <= 0:
        raise ValueError("rope_sin must have a positive row stride")
    if output.ndim != 3 or not 0 < int(output.shape[0]) <= int(caps.max_q_rows):
        raise ValueError("output rows must be within the planned QSA capacity")
    _check_tensor(
        output,
        name="output",
        device=caps.device,
        shape=(int(output.shape[0]), int(caps.q_heads), int(caps.head_dim)),
        dtype=caps.dtype,
        contiguous=True,
    )
    _check_tensor(
        selected_positions,
        name="selected_positions",
        device=caps.device,
        shape=(int(caps.max_q_rows), int(caps.selection_width)),
        dtype=torch.int32,
        contiguous=True,
    )
    _validate_shared_compressed_raw_layout(
        caps=caps,
        compressed_k_cache=compressed_k_cache,
        raw_k_ring=raw_k_ring,
        raw_logical_positions=raw_logical_positions,
        raw_rope_positions=raw_rope_positions,
        raw_interval_start_positions=raw_interval_start_positions,
    )
    _require_mutation_alias_contract(
        mutable=(
            ("scratch", scratch_storage),
            ("compressed_k_cache", compressed_k_cache),
            ("raw_k_ring", raw_k_ring),
            ("raw_logical_positions", raw_logical_positions),
            ("raw_rope_positions", raw_rope_positions),
            ("raw_interval_start_positions", raw_interval_start_positions),
            ("output", output),
            ("selected_positions", selected_positions),
        ),
        read_only=tuple(
            (name, tensor)
            for name, tensor in (
                ("main_k_cache", main_k_cache),
                ("main_v_cache", main_v_cache),
                ("k_descale", k_descale),
                ("v_descale", v_descale),
                ("main_block_table", main_block_table),
                ("compressed_block_table", compressed_block_table),
                ("raw_state_slot_ids", raw_state_slot_ids),
                ("index_q_norm_weight", index_q_norm_weight),
                ("index_k_norm_weight", index_k_norm_weight),
                ("rope_cos", rope_cos),
                ("rope_sin", rope_sin),
            )
            if tensor is not None
        ),
    )
    layout = plan._layout
    workspace_q_rows = int(plan.workspace_q_rows)
    prepared_index_query = _scratch_view(
        scratch_storage,
        offset_bytes=layout.prepared_query_offset_bytes,
        shape=(
            workspace_q_rows,
            int(caps.index_heads),
            int(caps.index_head_dim),
        ),
        dtype=torch.bfloat16,
    )
    scores = _scratch_view(
        scratch_storage,
        offset_bytes=layout.score_offset_bytes,
        shape=(workspace_q_rows, int(plan.score_workspace_width)),
        dtype=torch.float32,
    )
    eligible_group_counts = _scratch_view(
        scratch_storage,
        offset_bytes=layout.eligible_counts_offset_bytes,
        shape=(workspace_q_rows,),
        dtype=torch.int32,
    )
    merge_lengths = _scratch_view(
        scratch_storage,
        offset_bytes=layout.merge_lengths_offset_bytes,
        shape=(workspace_q_rows,),
        dtype=torch.int32,
    )
    topk_values = _scratch_view(
        scratch_storage,
        offset_bytes=layout.topk_values_offset_bytes,
        shape=(workspace_q_rows, int(caps.group_budget)),
        dtype=torch.float32,
    )
    topk_group_ids = _scratch_view(
        scratch_storage,
        offset_bytes=layout.topk_indices_offset_bytes,
        shape=(workspace_q_rows, int(caps.group_budget)),
        dtype=torch.int32,
    )
    topk_values_b = _scratch_view(
        scratch_storage,
        offset_bytes=layout.topk_values_b_offset_bytes,
        shape=(workspace_q_rows, int(caps.group_budget)),
        dtype=torch.float32,
    )
    topk_group_ids_b = _scratch_view(
        scratch_storage,
        offset_bytes=layout.topk_indices_b_offset_bytes,
        shape=(workspace_q_rows, int(caps.group_budget)),
        dtype=torch.int32,
    )
    state_errors = _scratch_view(
        scratch_storage,
        offset_bytes=layout.state_errors_offset_bytes,
        shape=(int(caps.max_q_rows),),
        dtype=torch.int32,
    )
    partial_output = _scratch_view(
        scratch_storage,
        offset_bytes=layout.partial_output_offset_bytes,
        shape=(
            int(plan.max_split_row_product),
            int(caps.q_heads),
            int(caps.head_dim),
        ),
        dtype=torch.float32,
    )
    partial_lse = _scratch_view(
        scratch_storage,
        offset_bytes=layout.partial_lse_offset_bytes,
        shape=(int(plan.max_split_row_product), int(caps.q_heads)),
        dtype=torch.float32,
    )
    selection_done = None
    if selection_stream is not None:
        if not isinstance(selection_stream, torch.cuda.Stream):
            raise TypeError("selection_stream must be a CUDA stream")
        if selection_stream.device != caps.device:
            raise ValueError("selection_stream must match the QSA plan device")
        with torch.cuda.device(caps.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("bind selection-stream resources before graph capture")
            selection_done = torch.cuda.Event()
            selection_done.record(selection_stream)
    return Binding(
        selection_stream=selection_stream,
        _selection_done=selection_done,
        plan=plan,
        shared_compressed_raw_pool=_overlaps(compressed_k_cache, raw_k_ring),
        scratch=scratch_storage,
        main_k_cache=main_k_cache,
        main_v_cache=main_v_cache,
        k_descale=k_descale,
        v_descale=v_descale,
        main_block_table=main_block_table,
        compressed_k_cache=compressed_k_cache,
        compressed_block_table=compressed_block_table,
        raw_k_ring=raw_k_ring,
        raw_logical_positions=raw_logical_positions,
        raw_rope_positions=raw_rope_positions,
        raw_interval_start_positions=raw_interval_start_positions,
        raw_state_slot_ids=raw_state_slot_ids,
        index_q_norm_weight=index_q_norm_weight,
        index_k_norm_weight=index_k_norm_weight,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        output=output,
        selected_positions=selected_positions,
        prepared_index_query=prepared_index_query,
        scores=scores,
        eligible_group_counts=eligible_group_counts,
        merge_lengths=merge_lengths,
        topk_values=topk_values,
        topk_group_ids=topk_group_ids,
        topk_values_b=topk_values_b,
        topk_group_ids_b=topk_group_ids_b,
        state_errors=state_errors,
        request_errors=_scratch_view(
            scratch_storage,
            offset_bytes=layout.request_errors_offset_bytes,
            shape=(int(caps.max_batch) + 1,),
            dtype=torch.int32,
        ),
        partial_output=partial_output,
        partial_lse=partial_lse,
    )


def _qsa_decode_impl(
    query: torch.Tensor | None,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    is_prefilling: torch.Tensor,
    scratch: torch.Tensor,
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    main_block_table: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    output: torch.Tensor,
    selected_positions: torch.Tensor,
    sparse_gqa_direct_kv_warps: int,
    max_seq_len: int,
    max_speculative_tokens: int,
    compress_ratio: int,
    budget: int,
    index_rotary_dim: int,
    mrope_section_0: int,
    mrope_section_1: int,
    mrope_section_2: int,
    mrope_interleaved: bool,
    rms_norm_eps: float,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    max_split_row_product: int,
    workspace_q_rows: int,
    prepared_query_offset_bytes: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    state_errors_offset_bytes: int,
    request_errors_offset_bytes: int,
    topk_offset_bytes: int,
    partial_output_offset_bytes: int,
    partial_lse_offset_bytes: int,
    work_metadata_offset_bytes: int,
) -> None:
    """Launch selector state updates, optionally followed by sparse attention."""
    rows = int(index_query.shape[0])
    q_heads = int(output.shape[1])
    head_dim = int(output.shape[2])
    index_heads = int(index_query.shape[1])
    index_head_dim = int(index_query.shape[2])
    position_axes = int(rope_positions.shape[1])
    sections = (
        (int(mrope_section_0), int(mrope_section_1), int(mrope_section_2))
        if position_axes == 3
        else None
    )
    caps = _KernelCaps(
        max_batch=int(main_block_table.shape[0]),
        max_raw_state_slots=int(raw_k_ring.shape[0]),
        max_seq_len=int(max_seq_len),
        main_page_size=int(main_k_cache.shape[1]),
        compressed_page_size=int(compressed_k_cache.shape[1]),
        q_heads=q_heads,
        kv_heads=int(main_k_cache.shape[2]),
        head_dim=head_dim,
        index_heads=index_heads,
        index_head_dim=index_head_dim,
        index_rotary_dim=int(index_rotary_dim),
        compress_ratio=int(compress_ratio),
        budget=int(budget),
        position_axes=position_axes,
        mrope_sections=sections,
        mrope_interleaved=bool(mrope_interleaved),
        rms_norm_eps=float(rms_norm_eps),
        raw_ring_capacity=int(raw_k_ring.shape[1]),
        max_speculative_tokens=int(max_speculative_tokens),
    )

    max_q_rows = int(selected_positions.shape[0])
    work_rows = int(workspace_q_rows)
    if not 0 < work_rows <= max_q_rows or rows > int(output.shape[0]):
        raise RuntimeError("invalid planned QSA workspace row capacity")
    group_budget = int(budget) // int(compress_ratio)
    prepared_query = _scratch_view(
        scratch,
        offset_bytes=int(prepared_query_offset_bytes),
        shape=(work_rows, index_heads, index_head_dim),
        dtype=torch.bfloat16,
    )
    scores = _scratch_view(
        scratch,
        offset_bytes=int(score_offset_bytes),
        shape=(work_rows, int(score_workspace_width)),
        dtype=torch.float32,
    )
    eligible_counts = _scratch_view(
        scratch,
        offset_bytes=int(eligible_counts_offset_bytes),
        shape=(work_rows,),
        dtype=torch.int32,
    )
    merge_lengths = _scratch_view(
        scratch,
        offset_bytes=int(merge_lengths_offset_bytes),
        shape=(work_rows,),
        dtype=torch.int32,
    )
    topk_values = _scratch_view(
        scratch,
        offset_bytes=int(topk_values_offset_bytes),
        shape=(work_rows, group_budget),
        dtype=torch.float32,
    )
    topk_ids = _scratch_view(
        scratch,
        offset_bytes=int(topk_indices_offset_bytes),
        shape=(work_rows, group_budget),
        dtype=torch.int32,
    )
    topk_values_b = _scratch_view(
        scratch,
        offset_bytes=int(topk_values_b_offset_bytes),
        shape=(work_rows, group_budget),
        dtype=torch.float32,
    )
    topk_ids_b = _scratch_view(
        scratch,
        offset_bytes=int(topk_indices_b_offset_bytes),
        shape=(work_rows, group_budget),
        dtype=torch.int32,
    )
    state_errors = _scratch_view(
        scratch,
        offset_bytes=int(state_errors_offset_bytes),
        shape=(max_q_rows,),
        dtype=torch.int32,
    )[:rows]
    request_errors = _scratch_view(
        scratch,
        offset_bytes=int(request_errors_offset_bytes),
        shape=(int(caps.max_batch) + 1,),
        dtype=torch.int32,
    )
    stable_topk_blocks = math.ceil(int(score_workspace_width) / _STABLE_TOPK_BLOCK)
    stable_offset = int(topk_offset_bytes)
    stable_count_nbytes = work_rows * stable_topk_blocks * torch.int32.itemsize
    tie_counts = _scratch_view(
        scratch,
        offset_bytes=stable_offset,
        shape=(work_rows, stable_topk_blocks),
        dtype=torch.int32,
    )
    stable_offset += stable_count_nbytes
    greater_counts = _scratch_view(
        scratch,
        offset_bytes=stable_offset,
        shape=(work_rows, stable_topk_blocks),
        dtype=torch.int32,
    )
    stable_offset += stable_count_nbytes
    stable_values = _scratch_view(
        scratch,
        offset_bytes=stable_offset,
        shape=(work_rows, group_budget),
        dtype=torch.float32,
    )
    stable_offset += work_rows * group_budget * torch.float32.itemsize
    stable_ids = _scratch_view(
        scratch,
        offset_bytes=stable_offset,
        shape=(work_rows, group_budget),
        dtype=torch.int32,
    )
    stable_offset += work_rows * group_budget * torch.int32.itemsize
    thresholds = _scratch_view(
        scratch,
        offset_bytes=stable_offset,
        shape=(work_rows,),
        dtype=torch.float32,
    )
    stable_offset += work_rows * torch.float32.itemsize
    greater_totals = _scratch_view(
        scratch,
        offset_bytes=stable_offset,
        shape=(work_rows,),
        dtype=torch.int32,
    )
    partial_output_storage = _scratch_view(
        scratch,
        offset_bytes=int(partial_output_offset_bytes),
        shape=(int(max_split_row_product), q_heads, head_dim),
        dtype=torch.float32,
    )
    partial_lse_storage = _scratch_view(
        scratch,
        offset_bytes=int(partial_lse_offset_bytes),
        shape=(int(max_split_row_product), q_heads),
        dtype=torch.float32,
    )
    shared_compressed_raw_pool = any(
        _overlaps(compressed_k_cache, raw_tensor)
        for raw_tensor in (
            raw_k_ring,
            raw_logical_positions,
            raw_rope_positions,
            raw_interval_start_positions,
        )
    )
    shared_page_occupancy = _scratch_view(
        scratch,
        offset_bytes=int(work_metadata_offset_bytes),
        shape=(int(compressed_k_cache.shape[0]) + 1,),
        dtype=torch.int32,
    )

    from ._kernels import (
        launch_compress_completed_groups,
        launch_expand_selected_groups,
        launch_poison_failed_rows,
        launch_prepare_index_query,
        launch_propagate_request_errors,
        launch_remap_topk_group_ids,
        launch_stabilize_topk,
        launch_stage_topk_carry,
        launch_topk_groups,
        launch_commit_raw_ring,
        launch_validate_completed_groups,
        launch_validate_rows,
        launch_validate_page_tables,
        launch_validate_shared_pool_ownership,
    )
    from ._score_cute import launch_score_representatives

    launch_validate_rows(
        request_ids=request_ids,
        query_positions=query_positions,
        rope_positions=rope_positions,
        sequence_lengths=sequence_lengths,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted_tokens,
        is_prefilling=is_prefilling,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_interval_start_positions=raw_interval_start_positions,
        request_errors=request_errors,
        state_errors=state_errors,
        rope_position_rows=int(rope_cos.shape[0]),
        caps=caps,
    )
    launch_validate_page_tables(
        request_ids=request_ids,
        sequence_lengths=sequence_lengths,
        main_block_table=main_block_table,
        compressed_block_table=compressed_block_table,
        raw_state_slot_ids=raw_state_slot_ids,
        state_errors=state_errors,
        num_main_pages=int(main_k_cache.shape[0]),
        num_compressed_pages=int(compressed_k_cache.shape[0]),
        shared_compressed_raw_pool=shared_compressed_raw_pool,
        caps=caps,
    )
    if shared_compressed_raw_pool:
        launch_validate_shared_pool_ownership(
            request_ids=request_ids,
            sequence_lengths=sequence_lengths,
            compressed_block_table=compressed_block_table,
            raw_state_slot_ids=raw_state_slot_ids,
            state_errors=state_errors,
            occupancy=shared_page_occupancy,
            num_compressed_pages=int(compressed_k_cache.shape[0]),
            caps=caps,
        )
    launch_validate_completed_groups(
        query_positions=query_positions,
        rope_positions=rope_positions,
        request_ids=request_ids,
        query_start_loc=query_start_loc,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_logical_positions=raw_logical_positions,
        raw_rope_positions=raw_rope_positions,
        state_errors=state_errors,
        rope_position_rows=int(rope_cos.shape[0]),
        caps=caps,
    )
    launch_propagate_request_errors(
        request_ids=request_ids,
        request_errors=request_errors,
        state_errors=state_errors,
        caps=caps,
    )
    # Completion consumes the old ring before the current suffix can wrap it.
    launch_compress_completed_groups(
        raw_index_key=raw_index_key,
        query_positions=query_positions,
        rope_positions=rope_positions,
        request_ids=request_ids,
        query_start_loc=query_start_loc,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_k_ring=raw_k_ring,
        raw_logical_positions=raw_logical_positions,
        raw_rope_positions=raw_rope_positions,
        key_norm_weight=index_k_norm_weight,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        compressed_cache=compressed_k_cache,
        compressed_block_table=compressed_block_table,
        state_errors=state_errors,
        caps=caps,
    )
    launch_commit_raw_ring(
        raw_index_key=raw_index_key,
        query_positions=query_positions,
        rope_positions=rope_positions,
        request_ids=request_ids,
        query_start_loc=query_start_loc,
        sequence_lengths=sequence_lengths,
        is_prefilling=is_prefilling,
        raw_state_slot_ids=raw_state_slot_ids,
        raw_k_ring=raw_k_ring,
        raw_logical_positions=raw_logical_positions,
        raw_rope_positions=raw_rope_positions,
        raw_interval_start_positions=raw_interval_start_positions,
        state_errors=state_errors,
        caps=caps,
    )

    from ._sparse_gqa import launch_sparse_paged_gqa

    for row_offset in range(0, rows, work_rows):
        chunk_rows = min(work_rows, rows - row_offset)
        row_slice = slice(row_offset, row_offset + chunk_rows)
        chunk_request_ids = request_ids[row_slice]
        chunk_positions = query_positions[row_slice]
        chunk_rope = rope_positions[row_slice]
        chunk_errors = state_errors[row_slice]
        chunk_prepared = prepared_query[:chunk_rows]
        chunk_scores = scores[:chunk_rows]
        chunk_eligible = eligible_counts[:chunk_rows]
        chunk_merge_lengths = merge_lengths[:chunk_rows]
        chunk_topk_values = topk_values[:chunk_rows]
        chunk_topk_ids = topk_ids[:chunk_rows]
        chunk_topk_values_b = topk_values_b[:chunk_rows]
        chunk_topk_ids_b = topk_ids_b[:chunk_rows]
        chunk_tie_counts = tie_counts[:chunk_rows]
        chunk_greater_counts = greater_counts[:chunk_rows]
        chunk_stable_values = stable_values[:chunk_rows]
        chunk_stable_ids = stable_ids[:chunk_rows]
        chunk_thresholds = thresholds[:chunk_rows]
        chunk_greater_totals = greater_totals[:chunk_rows]

        launch_prepare_index_query(
            index_query=index_query[row_slice],
            request_ids=chunk_request_ids,
            norm_weight=index_q_norm_weight,
            rope_positions=chunk_rope,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            state_errors=chunk_errors,
            prepared_query=chunk_prepared,
            caps=caps,
        )
        prior_values = chunk_topk_values_b
        prior_ids = chunk_topk_ids_b
        final_ids = prior_ids
        for score_chunk in range(int(num_score_chunks)):
            group_offset = score_chunk * int(score_chunk_groups)
            group_count = min(
                int(score_chunk_groups), int(caps.max_groups) - group_offset
            )
            output_values = (
                chunk_topk_values if score_chunk % 2 == 0 else chunk_topk_values_b
            )
            output_ids = chunk_topk_ids if score_chunk % 2 == 0 else chunk_topk_ids_b
            if group_offset:
                launch_stage_topk_carry(
                    prior_values=prior_values,
                    eligible_counts=chunk_eligible,
                    scores=chunk_scores,
                    group_offset=group_offset,
                    group_budget=group_budget,
                )
            launch_score_representatives(
                prepared_query=chunk_prepared,
                query_positions=chunk_positions,
                request_ids=chunk_request_ids,
                sequence_lengths=sequence_lengths,
                compressed_cache=compressed_k_cache,
                compressed_block_table=compressed_block_table,
                state_errors=chunk_errors,
                scores=chunk_scores,
                eligible_counts=chunk_eligible,
                merge_lengths=chunk_merge_lengths,
                group_offset=group_offset,
                group_count=group_count,
                caps=caps,
            )
            launch_topk_groups(
                scores=chunk_scores,
                eligible_counts=chunk_merge_lengths,
                topk_values=output_values,
                topk_group_ids=output_ids,
                group_budget=group_budget,
            )
            launch_remap_topk_group_ids(
                local_ids=output_ids,
                prior_ids=prior_ids,
                eligible_counts=chunk_eligible,
                merge_lengths=chunk_merge_lengths,
                group_offset=group_offset,
                group_budget=group_budget,
            )
            launch_stabilize_topk(
                scores=chunk_scores,
                merge_lengths=chunk_merge_lengths,
                prior_ids=prior_ids,
                eligible_counts=chunk_eligible,
                topk_values=output_values,
                topk_group_ids=output_ids,
                tie_counts=chunk_tie_counts,
                greater_counts=chunk_greater_counts,
                stable_values=chunk_stable_values,
                stable_ids=chunk_stable_ids,
                thresholds=chunk_thresholds,
                greater_totals=chunk_greater_totals,
                group_offset=group_offset,
                group_budget=group_budget,
            )
            prior_values, prior_ids = output_values, output_ids
            final_ids = output_ids

        selected = selected_positions[row_slice]
        launch_expand_selected_groups(
            topk_group_ids=final_ids,
            eligible_counts=chunk_eligible,
            query_positions=chunk_positions,
            state_errors=chunk_errors,
            selected_positions=selected,
            caps=caps,
        )

        # Selection has no dependency on main Q/K/V contents. Its output and
        # error mask remain live until the attention stage consumes them.
        if query is None:
            continue

        block_n, splits = _target_splits(caps, chunk_rows)
        split_output = None
        split_lse = None
        if splits > 1:
            split_output = partial_output_storage[: chunk_rows * splits].view(
                chunk_rows, splits, q_heads, head_dim
            )
            split_lse = partial_lse_storage[: chunk_rows * splits].view(
                chunk_rows, splits, q_heads
            )
        active_output = output[row_slice]
        launch_sparse_paged_gqa(
            query=query[row_slice],
            key_cache=main_k_cache,
            value_cache=main_v_cache,
            k_descale=k_descale,
            v_descale=v_descale,
            block_table=main_block_table,
            request_ids=chunk_request_ids,
            selected_positions=selected,
            query_positions=chunk_positions,
            output=active_output,
            partial_output=split_output,
            partial_lse=split_lse,
            softmax_scale=1.0 / math.sqrt(head_dim),
            block_n=block_n,
            splits=splits,
            direct_kv_warps=int(sparse_gqa_direct_kv_warps),
        )
        launch_poison_failed_rows(output=active_output, state_errors=chunk_errors)


_QSA_MUTATED_ARGUMENTS = (
    "scratch",
    "compressed_k_cache",
    "raw_k_ring",
    "raw_logical_positions",
    "raw_rope_positions",
    "raw_interval_start_positions",
    "output",
    "selected_positions",
)


@torch.library.custom_op(
    "b12x::qsa_decode",
    mutates_args=_QSA_MUTATED_ARGUMENTS,
)
def _qsa_decode_op(
    query: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    is_prefilling: torch.Tensor,
    scratch: torch.Tensor,
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    main_block_table: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    output: torch.Tensor,
    selected_positions: torch.Tensor,
    sparse_gqa_direct_kv_warps: int,
    max_seq_len: int,
    max_speculative_tokens: int,
    compress_ratio: int,
    budget: int,
    index_rotary_dim: int,
    mrope_section_0: int,
    mrope_section_1: int,
    mrope_section_2: int,
    mrope_interleaved: bool,
    rms_norm_eps: float,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    max_split_row_product: int,
    workspace_q_rows: int,
    prepared_query_offset_bytes: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    state_errors_offset_bytes: int,
    request_errors_offset_bytes: int,
    topk_offset_bytes: int,
    partial_output_offset_bytes: int,
    partial_lse_offset_bytes: int,
    work_metadata_offset_bytes: int,
    selection_only: bool,
) -> None:
    _require_mutation_alias_contract(
        mutable=(
            ("scratch", scratch),
            ("compressed_k_cache", compressed_k_cache),
            ("raw_k_ring", raw_k_ring),
            ("raw_logical_positions", raw_logical_positions),
            ("raw_rope_positions", raw_rope_positions),
            ("raw_interval_start_positions", raw_interval_start_positions),
            ("output", output),
            ("selected_positions", selected_positions),
        ),
        read_only=(
            ("query", query),
            ("index_query", index_query),
            ("raw_index_key", raw_index_key),
            ("request_ids", request_ids),
            ("query_positions", query_positions),
            ("rope_positions", rope_positions),
            ("sequence_lengths", sequence_lengths),
            ("query_start_loc", query_start_loc),
            ("num_accepted_tokens", num_accepted_tokens),
            ("is_prefilling", is_prefilling),
        ),
    )
    _qsa_decode_impl(
        None if selection_only else query,
        index_query,
        raw_index_key,
        request_ids,
        query_positions,
        rope_positions,
        sequence_lengths,
        query_start_loc,
        num_accepted_tokens,
        is_prefilling,
        scratch,
        main_k_cache,
        main_v_cache,
        k_descale,
        v_descale,
        main_block_table,
        compressed_k_cache,
        compressed_block_table,
        raw_k_ring,
        raw_logical_positions,
        raw_rope_positions,
        raw_interval_start_positions,
        raw_state_slot_ids,
        index_q_norm_weight,
        index_k_norm_weight,
        rope_cos,
        rope_sin,
        output,
        selected_positions,
        sparse_gqa_direct_kv_warps,
        max_seq_len,
        max_speculative_tokens,
        compress_ratio,
        budget,
        index_rotary_dim,
        mrope_section_0,
        mrope_section_1,
        mrope_section_2,
        mrope_interleaved,
        rms_norm_eps,
        score_chunk_groups,
        score_workspace_width,
        num_score_chunks,
        max_split_row_product,
        workspace_q_rows,
        prepared_query_offset_bytes,
        score_offset_bytes,
        eligible_counts_offset_bytes,
        merge_lengths_offset_bytes,
        topk_values_offset_bytes,
        topk_indices_offset_bytes,
        topk_values_b_offset_bytes,
        topk_indices_b_offset_bytes,
        state_errors_offset_bytes,
        request_errors_offset_bytes,
        topk_offset_bytes,
        partial_output_offset_bytes,
        partial_lse_offset_bytes,
        work_metadata_offset_bytes,
    )


@_qsa_decode_op.register_fake
def _qsa_decode_fake(
    query: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    is_prefilling: torch.Tensor,
    scratch: torch.Tensor,
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    main_block_table: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_k_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    raw_interval_start_positions: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    output: torch.Tensor,
    selected_positions: torch.Tensor,
    sparse_gqa_direct_kv_warps: int,
    max_seq_len: int,
    max_speculative_tokens: int,
    compress_ratio: int,
    budget: int,
    index_rotary_dim: int,
    mrope_section_0: int,
    mrope_section_1: int,
    mrope_section_2: int,
    mrope_interleaved: bool,
    rms_norm_eps: float,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    max_split_row_product: int,
    workspace_q_rows: int,
    prepared_query_offset_bytes: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    state_errors_offset_bytes: int,
    request_errors_offset_bytes: int,
    topk_offset_bytes: int,
    partial_output_offset_bytes: int,
    partial_lse_offset_bytes: int,
    work_metadata_offset_bytes: int,
    selection_only: bool,
) -> None:
    return None


def _raw_views_from_compressed_pool(
    compressed_k_cache: torch.Tensor,
    *,
    max_raw_state_slots: int,
    raw_ring_capacity: int,
    index_head_dim: int,
    position_axes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    page_elements = int(compressed_k_cache.shape[1]) * int(compressed_k_cache.shape[2])
    raw_k_ring = compressed_k_cache.as_strided(
        (int(max_raw_state_slots), int(raw_ring_capacity), int(index_head_dim)),
        (page_elements, int(index_head_dim), 1),
        storage_offset=int(compressed_k_cache.storage_offset()),
    )
    pool_i64 = compressed_k_cache.view(torch.uint8).reshape(-1).view(torch.int64)
    page_i64 = (
        int(compressed_k_cache.stride(0))
        * int(compressed_k_cache.element_size())
        // torch.int64.itemsize
    )
    payload_i64 = (
        int(raw_ring_capacity)
        * int(index_head_dim)
        * torch.bfloat16.itemsize
        // torch.int64.itemsize
    )
    base_i64 = int(pool_i64.storage_offset())
    raw_logical_positions = pool_i64.as_strided(
        (int(max_raw_state_slots), int(raw_ring_capacity)),
        (page_i64, 1),
        storage_offset=base_i64 + payload_i64,
    )
    raw_rope_positions = pool_i64.as_strided(
        (
            int(max_raw_state_slots),
            int(raw_ring_capacity),
            int(position_axes),
        ),
        (page_i64, int(position_axes), 1),
        storage_offset=base_i64 + payload_i64 + int(raw_ring_capacity),
    )
    raw_interval_start_positions = pool_i64.as_strided(
        (int(max_raw_state_slots),),
        (page_i64,),
        storage_offset=(
            base_i64 + payload_i64 + int(raw_ring_capacity) * (1 + int(position_axes))
        ),
    )
    return (
        raw_k_ring,
        raw_logical_positions,
        raw_rope_positions,
        raw_interval_start_positions,
    )


_QSA_SHARED_MUTATED_ARGUMENTS = (
    "scratch",
    "compressed_raw_pool",
    "output",
    "selected_positions",
)


@torch.library.custom_op(
    "b12x::qsa_decode_shared",
    mutates_args=_QSA_SHARED_MUTATED_ARGUMENTS,
)
def _qsa_decode_shared_op(
    query: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    is_prefilling: torch.Tensor,
    scratch: torch.Tensor,
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    main_block_table: torch.Tensor,
    compressed_raw_pool: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    output: torch.Tensor,
    selected_positions: torch.Tensor,
    sparse_gqa_direct_kv_warps: int,
    max_raw_state_slots: int,
    raw_ring_capacity: int,
    max_seq_len: int,
    max_speculative_tokens: int,
    compress_ratio: int,
    budget: int,
    index_rotary_dim: int,
    mrope_section_0: int,
    mrope_section_1: int,
    mrope_section_2: int,
    mrope_interleaved: bool,
    rms_norm_eps: float,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    max_split_row_product: int,
    workspace_q_rows: int,
    prepared_query_offset_bytes: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    state_errors_offset_bytes: int,
    request_errors_offset_bytes: int,
    topk_offset_bytes: int,
    partial_output_offset_bytes: int,
    partial_lse_offset_bytes: int,
    work_metadata_offset_bytes: int,
    selection_only: bool,
) -> None:
    _require_mutation_alias_contract(
        mutable=(
            ("scratch", scratch),
            ("compressed_raw_pool", compressed_raw_pool),
            ("output", output),
            ("selected_positions", selected_positions),
        ),
        read_only=(
            ("query", query),
            ("index_query", index_query),
            ("raw_index_key", raw_index_key),
            ("request_ids", request_ids),
            ("query_positions", query_positions),
            ("rope_positions", rope_positions),
            ("sequence_lengths", sequence_lengths),
            ("query_start_loc", query_start_loc),
            ("num_accepted_tokens", num_accepted_tokens),
            ("is_prefilling", is_prefilling),
        ),
    )
    (
        raw_k_ring,
        raw_logical_positions,
        raw_rope_positions,
        raw_interval_start_positions,
    ) = _raw_views_from_compressed_pool(
        compressed_raw_pool,
        max_raw_state_slots=int(max_raw_state_slots),
        raw_ring_capacity=int(raw_ring_capacity),
        index_head_dim=int(index_query.shape[2]),
        position_axes=int(rope_positions.shape[1]),
    )
    _qsa_decode_impl(
        None if selection_only else query,
        index_query,
        raw_index_key,
        request_ids,
        query_positions,
        rope_positions,
        sequence_lengths,
        query_start_loc,
        num_accepted_tokens,
        is_prefilling,
        scratch,
        main_k_cache,
        main_v_cache,
        k_descale,
        v_descale,
        main_block_table,
        compressed_raw_pool,
        compressed_block_table,
        raw_k_ring,
        raw_logical_positions,
        raw_rope_positions,
        raw_interval_start_positions,
        raw_state_slot_ids,
        index_q_norm_weight,
        index_k_norm_weight,
        rope_cos,
        rope_sin,
        output,
        selected_positions,
        sparse_gqa_direct_kv_warps,
        max_seq_len,
        max_speculative_tokens,
        compress_ratio,
        budget,
        index_rotary_dim,
        mrope_section_0,
        mrope_section_1,
        mrope_section_2,
        mrope_interleaved,
        rms_norm_eps,
        score_chunk_groups,
        score_workspace_width,
        num_score_chunks,
        max_split_row_product,
        workspace_q_rows,
        prepared_query_offset_bytes,
        score_offset_bytes,
        eligible_counts_offset_bytes,
        merge_lengths_offset_bytes,
        topk_values_offset_bytes,
        topk_indices_offset_bytes,
        topk_values_b_offset_bytes,
        topk_indices_b_offset_bytes,
        state_errors_offset_bytes,
        request_errors_offset_bytes,
        topk_offset_bytes,
        partial_output_offset_bytes,
        partial_lse_offset_bytes,
        work_metadata_offset_bytes,
    )


@_qsa_decode_shared_op.register_fake
def _qsa_decode_shared_fake(
    query: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    is_prefilling: torch.Tensor,
    scratch: torch.Tensor,
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    main_block_table: torch.Tensor,
    compressed_raw_pool: torch.Tensor,
    compressed_block_table: torch.Tensor,
    raw_state_slot_ids: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    output: torch.Tensor,
    selected_positions: torch.Tensor,
    sparse_gqa_direct_kv_warps: int,
    max_raw_state_slots: int,
    raw_ring_capacity: int,
    max_seq_len: int,
    max_speculative_tokens: int,
    compress_ratio: int,
    budget: int,
    index_rotary_dim: int,
    mrope_section_0: int,
    mrope_section_1: int,
    mrope_section_2: int,
    mrope_interleaved: bool,
    rms_norm_eps: float,
    score_chunk_groups: int,
    score_workspace_width: int,
    num_score_chunks: int,
    max_split_row_product: int,
    workspace_q_rows: int,
    prepared_query_offset_bytes: int,
    score_offset_bytes: int,
    eligible_counts_offset_bytes: int,
    merge_lengths_offset_bytes: int,
    topk_values_offset_bytes: int,
    topk_indices_offset_bytes: int,
    topk_values_b_offset_bytes: int,
    topk_indices_b_offset_bytes: int,
    state_errors_offset_bytes: int,
    request_errors_offset_bytes: int,
    topk_offset_bytes: int,
    partial_output_offset_bytes: int,
    partial_lse_offset_bytes: int,
    work_metadata_offset_bytes: int,
    selection_only: bool,
) -> None:
    return None


def _run(
    binding: Binding,
    *,
    query: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    is_prefilling: torch.Tensor,
    selection_only: bool = False,
) -> torch.Tensor:
    """Validate a complete transaction and launch its combined or selector stage."""
    if not isinstance(binding, Binding):
        raise TypeError("binding must be a qsa.Binding")
    caps = binding.plan.caps
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a tensor")
    if index_query.device.type != "cuda":
        raise RuntimeError("QSA GPU decode requires a CUDA device")
    rows = int(index_query.shape[0])
    if not 0 < rows <= int(caps.max_q_rows):
        raise ValueError("query rows must be within the planned decode capacity")
    if rows > int(binding.output.shape[0]):
        raise ValueError("query rows exceed the bound QSA output capacity")
    dynamic_specs = (
        (
            index_query,
            "index_query",
            (rows, int(caps.index_heads), int(caps.index_head_dim)),
            (torch.bfloat16,),
        ),
        (
            raw_index_key,
            "raw_index_key",
            (rows, int(caps.index_head_dim)),
            (torch.bfloat16,),
        ),
        (request_ids, "request_ids", (rows,), (torch.int32, torch.int64)),
        (query_positions, "query_positions", (rows,), (torch.int64,)),
        (
            rope_positions,
            "rope_positions",
            (rows, int(caps.position_axes)),
            (torch.int64,),
        ),
        (
            sequence_lengths,
            "sequence_lengths",
            (int(caps.max_batch),),
            (torch.int32,),
        ),
        (
            query_start_loc,
            "query_start_loc",
            (int(caps.max_batch) + 1,),
            (torch.int32,),
        ),
        (
            num_accepted_tokens,
            "num_accepted_tokens",
            (int(caps.max_batch),),
            (torch.int32,),
        ),
        (
            is_prefilling,
            "is_prefilling",
            (int(caps.max_batch),),
            (torch.bool,),
        ),
    )
    dynamic_specs = (
        (
            query,
            "query",
            (rows, int(caps.q_heads), int(caps.head_dim)),
            (torch.bfloat16,),
        ),
        *dynamic_specs,
    )
    for tensor, name, shape, dtypes in dynamic_specs:
        _check_tensor(
            tensor,
            name=name,
            device=caps.device,
            shape=shape,
            dtype=dtypes,
            contiguous=name not in ("rope_positions", "index_query", "raw_index_key"),
        )
    for tensor, name in (
        (index_query, "index_query"),
        (raw_index_key, "raw_index_key"),
    ):
        tail_elements = math.prod(tensor.shape[1:])
        if (
            int(tensor.stride(0)) < tail_elements
            or int(tensor.stride(-1)) != 1
            or (tensor.ndim == 3 and int(tensor.stride(1)) != int(tensor.shape[2]))
        ):
            raise ValueError(f"{name} requires non-overlapping dense rows")
    if int(rope_positions.stride(0)) <= 0 or int(rope_positions.stride(1)) <= 0:
        raise ValueError("rope_positions must have positive row and axis strides")
    _require_non_overlapping_layout("rope_positions", rope_positions)
    if not torch.compiler.is_compiling():
        _require_mutation_alias_contract(
            mutable=(
                ("scratch", binding.scratch),
                ("compressed_k_cache", binding.compressed_k_cache),
                ("raw_k_ring", binding.raw_k_ring),
                ("raw_logical_positions", binding.raw_logical_positions),
                ("raw_rope_positions", binding.raw_rope_positions),
                (
                    "raw_interval_start_positions",
                    binding.raw_interval_start_positions,
                ),
                ("output", binding.output),
                ("selected_positions", binding.selected_positions),
            ),
            read_only=tuple(
                (name, tensor) for tensor, name, _shape, _dtypes in dynamic_specs
            ),
        )

    sections = caps.mrope_sections or (0, 0, 0)
    layout = binding.plan._layout
    if binding.shared_compressed_raw_pool:
        _qsa_decode_shared_op(
            query,
            index_query,
            raw_index_key,
            request_ids,
            query_positions,
            rope_positions,
            sequence_lengths,
            query_start_loc,
            num_accepted_tokens,
            is_prefilling,
            binding.scratch,
            binding.main_k_cache,
            binding.main_v_cache,
            binding.k_descale,
            binding.v_descale,
            binding.main_block_table,
            binding.compressed_k_cache,
            binding.compressed_block_table,
            binding.raw_state_slot_ids,
            binding.index_q_norm_weight,
            binding.index_k_norm_weight,
            binding.rope_cos,
            binding.rope_sin,
            binding.output,
            binding.selected_positions,
            int(binding.plan.policy_resolution.config.sparse_gqa_direct_kv_warps),
            int(caps.max_raw_state_slots),
            int(caps.raw_ring_capacity),
            int(caps.max_seq_len),
            int(caps.max_speculative_tokens),
            int(caps.compress_ratio),
            int(caps.budget),
            int(caps.index_rotary_dim),
            int(sections[0]),
            int(sections[1]),
            int(sections[2]),
            bool(caps.mrope_interleaved),
            float(caps.rms_norm_eps),
            int(binding.plan.score_chunk_groups),
            int(binding.plan.score_workspace_width),
            int(binding.plan.num_score_chunks),
            int(binding.plan.max_split_row_product),
            int(binding.plan.workspace_q_rows),
            int(layout.prepared_query_offset_bytes),
            int(layout.score_offset_bytes),
            int(layout.eligible_counts_offset_bytes),
            int(layout.merge_lengths_offset_bytes),
            int(layout.topk_values_offset_bytes),
            int(layout.topk_indices_offset_bytes),
            int(layout.topk_values_b_offset_bytes),
            int(layout.topk_indices_b_offset_bytes),
            int(layout.state_errors_offset_bytes),
            int(layout.request_errors_offset_bytes),
            int(layout.topk_offset_bytes),
            int(layout.partial_output_offset_bytes),
            int(layout.partial_lse_offset_bytes),
            int(layout.work_metadata_offset_bytes),
            selection_only,
        )
        return binding.output[:rows]
    _qsa_decode_op(
        query,
        index_query,
        raw_index_key,
        request_ids,
        query_positions,
        rope_positions,
        sequence_lengths,
        query_start_loc,
        num_accepted_tokens,
        is_prefilling,
        binding.scratch,
        binding.main_k_cache,
        binding.main_v_cache,
        binding.k_descale,
        binding.v_descale,
        binding.main_block_table,
        binding.compressed_k_cache,
        binding.compressed_block_table,
        binding.raw_k_ring,
        binding.raw_logical_positions,
        binding.raw_rope_positions,
        binding.raw_interval_start_positions,
        binding.raw_state_slot_ids,
        binding.index_q_norm_weight,
        binding.index_k_norm_weight,
        binding.rope_cos,
        binding.rope_sin,
        binding.output,
        binding.selected_positions,
        int(binding.plan.policy_resolution.config.sparse_gqa_direct_kv_warps),
        int(caps.max_seq_len),
        int(caps.max_speculative_tokens),
        int(caps.compress_ratio),
        int(caps.budget),
        int(caps.index_rotary_dim),
        int(sections[0]),
        int(sections[1]),
        int(sections[2]),
        bool(caps.mrope_interleaved),
        float(caps.rms_norm_eps),
        int(binding.plan.score_chunk_groups),
        int(binding.plan.score_workspace_width),
        int(binding.plan.num_score_chunks),
        int(binding.plan.max_split_row_product),
        int(binding.plan.workspace_q_rows),
        int(layout.prepared_query_offset_bytes),
        int(layout.score_offset_bytes),
        int(layout.eligible_counts_offset_bytes),
        int(layout.merge_lengths_offset_bytes),
        int(layout.topk_values_offset_bytes),
        int(layout.topk_indices_offset_bytes),
        int(layout.topk_values_b_offset_bytes),
        int(layout.topk_indices_b_offset_bytes),
        int(layout.state_errors_offset_bytes),
        int(layout.request_errors_offset_bytes),
        int(layout.topk_offset_bytes),
        int(layout.partial_output_offset_bytes),
        int(layout.partial_lse_offset_bytes),
        int(layout.work_metadata_offset_bytes),
        selection_only,
    )
    return binding.output[:rows]


def run(
    binding: Binding,
    *,
    query: torch.Tensor,
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    is_prefilling: torch.Tensor,
    index_ready: torch.cuda.Event | None = None,
) -> torch.Tensor:
    """Run one packed QSA decode transaction after main K/V writes are enqueued.

    Main Q/K/V writes precede this call on the calling stream. By default all
    QSA work follows them on that stream. A binding with ``selection_stream``
    instead requires ``index_ready``, recorded after all selector inputs,
    metadata and prior state writes, before independent main Q/K/V work.
    Selection waits for that event and runs on the bound stream; attention
    joins it on the calling stream. Intermediate positions/errors remain an
    internal handoff within this call. Bind and prewarm before graph capture,
    and record the readiness event inside the graph for captured producers.

    Active request intervals are the dense prefix encoded by
    ``query_start_loc``.  Remaining boundaries repeat the live-row count and
    remaining query rows contain ``-1`` metadata.  ``sequence_lengths`` includes
    every row in the corresponding current interval.  The read-only main K/V
    cache must contain every selected original-token K/V row before attention
    executes on the calling stream.

    ``num_accepted_tokens`` commits the preceding verification interval:
    each active value is in ``[1, 1 + max_speculative_tokens]`` and counts the
    accepted prefix including its guaranteed or recovered token.  The mutable
    ``raw_interval_start_positions`` entry for the request's persistent state
    slot records the preceding interval's first row before the call and the
    current interval's first row after it.  Candidate rows may
    remain physically resident after rejection, but exact logical tags prevent
    them from becoming eligible.  Invalid dynamic metadata is detected on the
    device, suppresses persistent mutation for the whole request, and poisons
    its output rows with NaNs.

    With shared compressed/raw backing, requests that have no rows in the
    current packed call remain live page owners until eviction.  Their
    ``sequence_lengths``, ``compressed_block_table`` entries, and
    ``raw_state_slot_ids`` mappings must remain valid.  Zero sequence lengths
    and ``-1`` table or slot entries describe only unused or evicted capacity.

    For the first decode interval beginning at logical position ``N``, seed
    the persistent interval anchor to ``N - num_accepted_tokens``.  If the
    interval can complete a group whose prefix came from prefill, the raw ring
    must already contain exact logical tags, raw index keys, and RoPE positions
    for the trailing open group.  An anchor of ``-1`` is valid only for the
    position-zero initialization with one accepted token.  ``rope_positions``
    may be any non-overlapping 2-D view with positive row and axis strides,
    including ``positions_by_axis.T``.
    Selector projections may be disjoint dense-row views of a fused Q/K
    projection; their positive row strides are consumed without staging copies.
    """
    if not isinstance(binding, Binding):
        raise TypeError("binding must be a qsa.Binding")
    inputs = dict(
        query=query,
        index_query=index_query,
        raw_index_key=raw_index_key,
        request_ids=request_ids,
        query_positions=query_positions,
        rope_positions=rope_positions,
        sequence_lengths=sequence_lengths,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted_tokens,
        is_prefilling=is_prefilling,
    )
    stream = binding.selection_stream
    if stream is None:
        if index_ready is not None:
            raise ValueError("index_ready requires a binding with selection_stream")
        return _run(binding, **inputs)
    if index_ready is None:
        raise ValueError("selection_stream requires an index_ready event")
    if not torch.compiler.is_compiling():
        if not isinstance(index_ready, torch.cuda.Event):
            raise TypeError("index_ready must be a recorded CUDA event")
        if index_ready.device != binding.plan.caps.device:
            raise ValueError("index_ready must be recorded on the QSA plan device")
    with torch.cuda.device(binding.plan.caps.device):
        main_stream = torch.cuda.current_stream()
        stream.wait_event(index_ready)
        with torch.cuda.stream(stream):
            _run(binding, **inputs, selection_only=True)
            binding._selection_done.record(stream)
        main_stream.wait_event(binding._selection_done)
        return _run_attention(
            binding,
            query=query,
            request_ids=request_ids,
            query_positions=query_positions,
        )


@torch.library.custom_op("b12x::qsa_attention", mutates_args=("scratch", "output"))
def _qsa_attention_op(
    query: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    main_block_table: torch.Tensor,
    selected_positions: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor,
    work_rows: int,
    max_split_row_product: int,
    state_errors_offset: int,
    partial_output_offset: int,
    partial_lse_offset: int,
    direct_kv_warps: int,
) -> None:
    from ._kernels import launch_poison_failed_rows
    from ._sparse_gqa import launch_sparse_paged_gqa
    from ._sparse_gqa_cute_config import BLOCK_N

    # Storage addresses exist at the custom-op runtime boundary, not in tracing.
    _require_mutation_alias_contract(
        mutable=(("scratch", scratch), ("output", output)),
        read_only=(
            ("query", query),
            ("request_ids", request_ids),
            ("query_positions", query_positions),
            ("selected_positions", selected_positions),
        ),
    )
    rows, q_heads, head_dim = map(int, query.shape)
    errors = _scratch_view(
        scratch,
        offset_bytes=state_errors_offset,
        shape=(int(output.shape[0]),),
        dtype=torch.int32,
    )
    partial_output = _scratch_view(
        scratch,
        offset_bytes=partial_output_offset,
        shape=(max_split_row_product, q_heads, head_dim),
        dtype=torch.float32,
    )
    partial_lse = _scratch_view(
        scratch,
        offset_bytes=partial_lse_offset,
        shape=(max_split_row_product, q_heads),
        dtype=torch.float32,
    )
    # Use the same row-chunk and native CuTe split policy as the combined call.
    for start in range(0, rows, work_rows):
        count = min(work_rows, rows - start)
        row_slice = slice(start, start + count)
        splits = _qwen_row_splits(count)
        launch_sparse_paged_gqa(
            query=query[row_slice],
            key_cache=main_k_cache,
            value_cache=main_v_cache,
            k_descale=k_descale,
            v_descale=v_descale,
            block_table=main_block_table,
            request_ids=request_ids[row_slice],
            selected_positions=selected_positions[row_slice],
            query_positions=query_positions[row_slice],
            output=output[row_slice],
            partial_output=(
                partial_output[: count * splits].view(count, splits, q_heads, head_dim)
                if splits > 1
                else None
            ),
            partial_lse=(
                partial_lse[: count * splits].view(count, splits, q_heads)
                if splits > 1
                else None
            ),
            softmax_scale=1.0 / math.sqrt(head_dim),
            block_n=BLOCK_N,
            splits=splits,
            direct_kv_warps=direct_kv_warps,
        )
        launch_poison_failed_rows(
            output=output[row_slice], state_errors=errors[row_slice]
        )


@_qsa_attention_op.register_fake
def _qsa_attention_fake(
    query: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    main_k_cache: torch.Tensor,
    main_v_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    main_block_table: torch.Tensor,
    selected_positions: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor,
    work_rows: int,
    max_split_row_product: int,
    state_errors_offset: int,
    partial_output_offset: int,
    partial_lse_offset: int,
    direct_kv_warps: int,
) -> None:
    return None


def _run_attention(
    binding: Binding,
    *,
    query: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
) -> torch.Tensor:
    """Consume the bound selection and error mask within one run call."""

    if not isinstance(binding, Binding):
        raise TypeError("binding must be a qsa.Binding")
    caps = binding.plan.caps
    rows = int(query.shape[0])
    if not 0 < rows <= int(caps.max_q_rows):
        raise ValueError("query rows must be within the planned decode capacity")
    if rows > int(binding.output.shape[0]):
        raise ValueError("query rows exceed the bound QSA output capacity")
    for tensor, name, shape, dtype in (
        (query, "query", (rows, caps.q_heads, caps.head_dim), torch.bfloat16),
        (request_ids, "request_ids", (rows,), (torch.int32, torch.int64)),
        (query_positions, "query_positions", (rows,), torch.int64),
    ):
        _check_tensor(
            tensor,
            name=name,
            device=caps.device,
            shape=shape,
            dtype=dtype,
            contiguous=True,
        )
    layout = binding.plan._layout
    _qsa_attention_op(
        query,
        request_ids,
        query_positions,
        binding.main_k_cache,
        binding.main_v_cache,
        binding.k_descale,
        binding.v_descale,
        binding.main_block_table,
        binding.selected_positions,
        binding.scratch,
        binding.output,
        binding.plan.workspace_q_rows,
        binding.plan.max_split_row_product,
        layout.state_errors_offset_bytes,
        layout.partial_output_offset_bytes,
        layout.partial_lse_offset_bytes,
        binding.plan.policy_resolution.config.sparse_gqa_direct_kv_warps,
    )
    return binding.output[:rows]


def prewarm(binding: Binding, *, rows: int | None = None) -> None:
    """Compile a bound QSA transaction without mutating persistent state.

    Every synthetic row has an invalid request ID and position. The launch
    therefore warms the bound output capacity, cache-table stride, selector
    workspace, selection, and attention specializations. Cache accesses
    and persistent selector-state writes remain masked. Scratch, output, and
    selected-position buffers are transient and have unspecified contents
    after this call.
    """
    if not isinstance(binding, Binding):
        raise TypeError("binding must be a qsa.Binding")

    caps = binding.plan.caps
    output_capacity = int(binding.output.shape[0])
    requested_rows = output_capacity if rows is None else int(rows)
    if not 0 < requested_rows <= output_capacity:
        raise ValueError("prewarm rows must fit the bound QSA output capacity")
    device = caps.device
    ready = torch.cuda.Event() if binding.selection_stream is not None else None
    sequence_lengths = torch.zeros(
        int(caps.max_batch), dtype=torch.int32, device=device
    )
    query_start_loc = torch.zeros(
        int(caps.max_batch) + 1, dtype=torch.int32, device=device
    )
    num_accepted_tokens = torch.ones(
        int(caps.max_batch), dtype=torch.int32, device=device
    )
    is_prefilling = torch.ones(int(caps.max_batch), dtype=torch.bool, device=device)
    warm_rows = {requested_rows}
    if output_capacity > _MAX_SPLIT_ROWS:
        warm_rows.add(_MAX_SPLIT_ROWS)
        warm_rows.add(_MAX_SPLIT_ROWS + 1)
    for warm_row_count in sorted(warm_rows):
        query = torch.empty(
            (warm_row_count, int(caps.q_heads), int(caps.head_dim)),
            dtype=caps.dtype,
            device=device,
        )
        index_query = torch.empty(
            (warm_row_count, int(caps.index_heads), int(caps.index_head_dim)),
            dtype=caps.dtype,
            device=device,
        )
        raw_index_key = torch.empty(
            (warm_row_count, int(caps.index_head_dim)),
            dtype=caps.dtype,
            device=device,
        )
        query_positions = torch.full(
            (warm_row_count,), -1, dtype=torch.int64, device=device
        )
        rope_positions = torch.full(
            (warm_row_count, int(caps.position_axes)),
            -1,
            dtype=torch.int64,
            device=device,
        )
        for request_id_dtype in (torch.int32, torch.int64):
            request_ids = torch.full(
                (warm_row_count,),
                -1,
                dtype=request_id_dtype,
                device=device,
            )
            if ready is not None:
                ready.record(torch.cuda.current_stream(device))
            run(
                binding,
                query=query,
                index_query=index_query,
                raw_index_key=raw_index_key,
                request_ids=request_ids,
                query_positions=query_positions,
                rope_positions=rope_positions,
                sequence_lengths=sequence_lengths,
                query_start_loc=query_start_loc,
                num_accepted_tokens=num_accepted_tokens,
                is_prefilling=is_prefilling,
                index_ready=ready,
            )


def is_supported(device: torch.device | str | None = None) -> bool:
    """Return whether the mandatory CuTe QSA dependencies are available."""
    from ..._lib.gating import has_cutlass_dsl, has_triton

    del device
    return has_cutlass_dsl() and has_triton()


__all__ = [
    "CacheRequirements",
    "Caps",
    "Plan",
    "Binding",
    "cache_requirements",
    "plan",
    "bind",
    "prewarm",
    "run",
    "is_supported",
]
