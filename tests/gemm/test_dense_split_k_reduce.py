"""The split-K partial reduction rounds once and does not depend on the
scratch layout: the Triton kernels (two slices and any slice count) and the
torch fallback for other layouts add the FP32 slices in index order and round
to the output dtype at the store, so their results are bit-identical."""

from __future__ import annotations

import pytest
import torch

from b12x._lib.dense_gemm import _reduce_split_k2_bf16

from ..conftest import require_b12x


def _ordered_reference(partials: torch.Tensor) -> torch.Tensor:
    accum = partials[:, :, 0].to(torch.float32).clone()
    for s in range(1, partials.shape[2]):
        accum += partials[:, :, s].to(torch.float32)
    return accum.to(torch.bfloat16)


@pytest.mark.parametrize("slices", [2, 3, 4, 8])
def test_kernel_and_fallback_reduce_slices_in_order(slices: int) -> None:
    device = require_b12x()
    torch.manual_seed(20260903)
    m, n = 8, 1536
    # Partial magnitudes spanning several binades so that reassociation
    # changes low bits.
    scratch = torch.randn(slices, m, n, device=device, dtype=torch.float32)
    scratch *= torch.logspace(-3, 3, slices, device=device).view(slices, 1, 1)
    kernel_view = scratch.permute(1, 2, 0)  # [m, n, slices], stride (n, 1, m*n)
    assert kernel_view.stride() == (n, 1, m * n)
    kernel_out = torch.empty(m, n, 1, device=device, dtype=torch.bfloat16)
    _reduce_split_k2_bf16(kernel_view, kernel_out, m=m, n=n)

    # A strided copy of the same partials takes the torch fallback.
    padded = torch.zeros(slices, m, n + 64, device=device, dtype=torch.float32)
    padded[:, :, :n].copy_(scratch)
    fallback_view = padded[:, :, :n].permute(1, 2, 0)
    assert fallback_view.stride() != (n, 1, m * n)
    fallback_out = torch.empty(m, n, 1, device=device, dtype=torch.bfloat16)
    _reduce_split_k2_bf16(fallback_view, fallback_out, m=m, n=n)

    expected = _ordered_reference(kernel_view)
    assert torch.count_nonzero(expected).item() == expected.numel()
    assert torch.equal(kernel_out[:, :, 0], expected)
    assert torch.equal(fallback_out[:, :, 0], expected)
