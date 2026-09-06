"""Indexed sparse paged causal GQA for Qwen3.8 Flash Next.

This private stage reads BF16 or globally scaled FP8 E4M3 main-cache K/V at
caller-selected logical token positions. It never writes either cache. Small
batches use split CuTe kernels to expose enough parallel work. Large prefill
batches use the direct selected-position entry of the paged CuTe engine. Every
unsupported geometry, layout, or device fails closed.
"""

from __future__ import annotations

import math

import torch

from ._sparse_gqa_cute_config import (
    MAX_SPLIT_ROWS as _MAX_SPLIT_ROWS,
    _is_page_token_head_layout,
    is_candidate as _cute_is_candidate,
    is_qwen_geometry as _is_qwen_geometry,
)


def _require_unit_inner_stride(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim == 0 or int(tensor.stride(-1)) != 1:
        raise ValueError(f"{name} must have unit inner stride")


def _validate_launch(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    partial_output: torch.Tensor | None,
    partial_lse: torch.Tensor | None,
    softmax_scale: float,
    block_n: int,
    splits: int,
) -> tuple[int, int, int]:
    if not query.is_cuda:
        raise ValueError("QSA sparse GQA requires CUDA tensors")
    device = query.device
    if query.ndim != 3:
        raise ValueError("query must have shape [rows, q_heads, head_dim]")
    rows, q_heads, head_dim = map(int, query.shape)
    if rows <= 0:
        raise ValueError("query must contain at least one active row")
    if query.dtype != torch.bfloat16:
        raise TypeError(f"query must be torch.bfloat16, got {query.dtype}")
    if head_dim < 16 or head_dim & (head_dim - 1):
        raise ValueError("head_dim must be a power of two at least 16")
    _require_unit_inner_stride(query, "query")

    if key_cache.ndim != 4:
        raise ValueError("key_cache must have shape [pages, page, kv_heads, dim]")
    pages, page_size, kv_heads, cache_dim = map(int, key_cache.shape)
    if pages <= 0 or page_size <= 0 or kv_heads <= 0:
        raise ValueError("key_cache dimensions must be positive")
    if cache_dim != head_dim:
        raise ValueError("key_cache head dimension must match query")
    if q_heads % kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads")
    if key_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("key_cache must be BF16 or FP8 E4M3FN")
    if value_cache.shape != key_cache.shape or value_cache.dtype != key_cache.dtype:
        raise ValueError("value_cache must match the key_cache shape and dtype")
    _require_unit_inner_stride(key_cache, "key_cache")
    _require_unit_inner_stride(value_cache, "value_cache")
    if key_cache.dtype == torch.float8_e4m3fn:
        if k_descale is None or v_descale is None:
            raise ValueError("FP8 QSA caches require k_descale and v_descale")
        for descale, name in ((k_descale, "k_descale"), (v_descale, "v_descale")):
            if (
                descale.device != device
                or descale.dtype != torch.float32
                or descale.numel() != 1
                or not descale.is_contiguous()
            ):
                raise ValueError(f"{name} must be one contiguous CUDA float32 value")

    tensors = (
        block_table,
        request_ids,
        selected_positions,
        query_positions,
        output,
        key_cache,
        value_cache,
    )
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("all QSA sparse GQA tensors must share one device")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise TypeError("block_table must be rank-2 torch.int32")
    if not block_table.is_contiguous():
        raise ValueError("block_table must be contiguous")
    if request_ids.shape != (rows,) or request_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("request_ids must be contiguous int32/int64 shape [rows]")
    if not request_ids.is_contiguous():
        raise ValueError("request_ids must be contiguous")
    if query_positions.shape != (rows,) or query_positions.dtype != torch.int64:
        raise TypeError("query_positions must be contiguous int64 shape [rows]")
    if not query_positions.is_contiguous():
        raise ValueError("query_positions must be contiguous")
    if (
        selected_positions.ndim != 2
        or int(selected_positions.shape[0]) < rows
        or int(selected_positions.shape[1]) <= 0
        or selected_positions.dtype != torch.int32
    ):
        raise TypeError(
            "selected_positions must be int32 [capacity_rows, selection_width]"
        )
    if not selected_positions.is_contiguous():
        raise ValueError("selected_positions must be contiguous")
    if (
        output.ndim != 3
        or int(output.shape[0]) < rows
        or tuple(output.shape[1:]) != (q_heads, head_dim)
        or output.dtype != torch.bfloat16
    ):
        raise ValueError("output must be BF16 [capacity_rows, q_heads, head_dim]")
    _require_unit_inner_stride(output, "output")

    if rows > _MAX_SPLIT_ROWS:
        if not query.is_contiguous():
            raise ValueError("query must be contiguous for selected-position paging")
        if not (
            _is_page_token_head_layout(key_cache)
            and _is_page_token_head_layout(value_cache)
        ):
            raise ValueError(
                "selected-position paged K/V caches must have non-overlapping rows"
            )
        if not output.is_contiguous():
            raise ValueError("output must be contiguous for selected-position paging")

    block_n = int(block_n)
    splits = int(splits)
    if block_n not in (16, 64):
        raise ValueError("block_n must be 16 or 64")
    key_tiles = math.ceil(int(selected_positions.shape[1]) / block_n)
    if splits <= 0 or splits & (splits - 1) or splits > key_tiles:
        raise ValueError(
            "splits must be a positive power of two not exceeding key tiles"
        )
    if not math.isfinite(float(softmax_scale)) or float(softmax_scale) <= 0:
        raise ValueError("softmax_scale must be finite and positive")

    if splits > 1:
        if partial_output is None or partial_lse is None:
            raise ValueError("split sparse GQA requires both partial tensors")
        if partial_output.device != device or partial_lse.device != device:
            raise ValueError("partial tensors must share the query device")
        if (
            partial_output.ndim != 4
            or int(partial_output.shape[0]) < rows
            or int(partial_output.shape[1]) < splits
            or tuple(partial_output.shape[2:]) != (q_heads, head_dim)
            or partial_output.dtype != torch.float32
        ):
            raise ValueError(
                "partial_output must be float32 [capacity_rows, >=splits, "
                "q_heads, head_dim]"
            )
        if (
            partial_lse.ndim != 3
            or int(partial_lse.shape[0]) < rows
            or int(partial_lse.shape[1]) < splits
            or int(partial_lse.shape[2]) != q_heads
            or partial_lse.dtype != torch.float32
        ):
            raise ValueError(
                "partial_lse must be float32 [capacity_rows, >=splits, q_heads]"
            )
        _require_unit_inner_stride(partial_output, "partial_output")
        _require_unit_inner_stride(partial_lse, "partial_lse")
    return rows, q_heads, head_dim


def launch_sparse_paged_gqa(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    partial_output: torch.Tensor | None,
    partial_lse: torch.Tensor | None,
    softmax_scale: float,
    block_n: int,
    splits: int,
    direct_kv_warps: int = 2,
) -> torch.Tensor:
    """Launch the allocation-free CuTe Qwen sparse GQA into ``output``."""
    rows, _, _ = _validate_launch(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        k_descale=k_descale,
        v_descale=v_descale,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        output=output,
        partial_output=partial_output,
        partial_lse=partial_lse,
        softmax_scale=softmax_scale,
        block_n=block_n,
        splits=splits,
    )
    if not _is_qwen_geometry(
        q_heads=int(query.shape[1]),
        kv_heads=int(key_cache.shape[2]),
        head_dim=int(query.shape[2]),
        selection_width=int(selected_positions.shape[1]),
        block_n=block_n,
        splits=splits,
    ):
        raise NotImplementedError(
            "QSA sparse GQA has no CuTe implementation for q_heads="
            f"{int(query.shape[1])}, kv_heads={int(key_cache.shape[2])}, "
            f"head_dim={int(query.shape[2])}, page_size={int(key_cache.shape[1])}, "
            f"selection_width={int(selected_positions.shape[1])}, "
            f"block_n={int(block_n)}, splits={int(splits)}"
        )
    if rows > _MAX_SPLIT_ROWS:
        from ..paged._selected_forward import launch_selected_paged_gqa_direct

        launch_selected_paged_gqa_direct(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            k_descale=k_descale,
            v_descale=v_descale,
            block_table=block_table,
            request_ids=request_ids,
            selected_positions=selected_positions,
            query_positions=query_positions,
            output=output,
            softmax_scale=softmax_scale,
            kv_warps=int(direct_kv_warps),
        )
        return output[:rows]
    assert partial_output is not None and partial_lse is not None
    if not _cute_is_candidate(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        partial_output=partial_output,
        partial_lse=partial_lse,
        block_n=block_n,
        splits=splits,
    ):
        raise RuntimeError(
            "Qwen sparse GQA requires its CuTe layout and capacity contract"
        )

    from ._sparse_gqa_cute import (
        launch_sparse_gqa_merge,
        launch_sparse_gqa_split,
    )

    launch_sparse_gqa_split(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        k_descale=k_descale,
        v_descale=v_descale,
        block_table=block_table,
        request_ids=request_ids,
        selected_positions=selected_positions,
        query_positions=query_positions,
        partial_output=partial_output,
        partial_lse=partial_lse,
        softmax_scale=softmax_scale,
        splits=splits,
    )
    launch_sparse_gqa_merge(
        partial_output=partial_output,
        partial_lse=partial_lse,
        output=output,
        rows=rows,
        splits=splits,
    )
    return output[:rows]


__all__ = ["launch_sparse_paged_gqa"]
