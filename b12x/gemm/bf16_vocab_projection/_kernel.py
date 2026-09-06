"""Triton kernels for single-row BF16 vocabulary projection."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _row_kernel(
    source,
    weight,
    output,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    values = tl.load(source + offsets, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(
        weight + row * K + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(output + row, tl.sum(values * weights, axis=0))


@triton.jit
def _row_loop_kernel(
    source,
    weight,
    output,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((), tl.float32)
    for start in range(0, K, BLOCK_K):
        positions = start + offsets
        mask = positions < K
        values = tl.load(source + positions, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(
            weight + row * K + positions,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(values * weights, axis=0)
    tl.store(output + row, accumulator)


def _can_run(source: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        source.ndim == 2
        and source.shape[0] == 1
        and weight.ndim == 2
        and source.shape[1] == weight.shape[1]
        and source.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and source.is_cuda
        and weight.is_cuda
        and source.is_contiguous()
        and weight.is_contiguous()
    )


@torch.library.custom_op("b12x::bf16_vocab_projection", mutates_args=())
def bf16_vocab_projection(
    source: torch.Tensor,
    weight: torch.Tensor,
    algorithm: int,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    """Project one BF16 row through a contiguous vocabulary matrix."""
    if not _can_run(source, weight):
        return torch.nn.functional.linear(source, weight)
    out_features, in_features = weight.shape
    output = torch.empty(
        (1, out_features),
        dtype=torch.bfloat16,
        device=source.device,
    )
    kernel = _row_kernel if algorithm == 0 else _row_loop_kernel
    kernel[(out_features,)](
        source,
        weight,
        output,
        K=in_features,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return output


@bf16_vocab_projection.register_fake
def _bf16_vocab_projection_fake(
    source: torch.Tensor,
    weight: torch.Tensor,
    algorithm: int,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    del algorithm, block_k, num_warps
    return source.new_empty((source.shape[0], weight.shape[0]))


__all__ = ["bf16_vocab_projection"]
