"""Small-M MXFP8 linears quantize their BF16 activations inside the GEMM.

For up to eight tokens, an unpadded K and N up to
B12X_MXFP8_LINEAR_FUSED_QUANT_A_MAX_N, `mxfp8_linear` runs one GEMM that
quantizes A in each CTA instead of the two scale-buffer fills, the row
quantization kernel and the GEMM. The in-CTA quantization computes the same
UE8M0 scales and E4M3 values, so the output must be bit-identical to the
separate path, eagerly and under CUDA-graph capture.

Bit identity holds when the separate path reduces split-K partials in FP32
(B12X_DENSE_SPLITK_TURBO=0, the production setting); the bf16 atomic
split-K accumulation is itself order-dependent, so the module constant is
pinned before the GEMM module is imported.
"""
from __future__ import annotations

import os

os.environ["B12X_DENSE_SPLITK_TURBO"] = "0"

import pytest
import torch

pytest.importorskip("cutlass")

from b12x.gemm.blockscaled import _linear as kernel_module
from b12x.gemm.blockscaled._linear import mxfp8_linear, pack_mxfp8_linear_weight

from ..conftest import require_b12x

SHAPES = [(7168, 1536), (7168, 576), (7168, 3584), (4096, 4096), (2048, 512), (128, 64)]


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 12


def _packed(n: int, k: int, gen: torch.Generator, device: torch.device):
    weight = (torch.randn((n, k), generator=gen, device=device) * 0.02).to(
        torch.float8_e4m3fn
    )
    scale = (
        127 + torch.randint(-3, 4, (n, k // 32), generator=gen, device=device)
    ).to(torch.uint8)
    return pack_mxfp8_linear_weight(weight, scale)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize(("k", "n"), SHAPES)
@pytest.mark.parametrize("m", [1, 3, 8])
def test_fused_quant_a_matches_separate_quantizer(
    monkeypatch: pytest.MonkeyPatch, k: int, n: int, m: int
) -> None:
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(20260902 + k + n)
    packed = _packed(n, k, gen, device)
    x = (torch.randn((m, k), generator=gen, device=device) * 0.5).to(torch.bfloat16)

    monkeypatch.setenv("B12X_MXFP8_LINEAR_FUSED_QUANT_A_MAX_N", "0")
    kernel_module._fused_quant_a_max_n.cache_clear()
    assert not kernel_module._use_fused_quant_a(m, x.dtype, k, k, n)
    separate = mxfp8_linear(x, packed).clone()

    monkeypatch.setenv("B12X_MXFP8_LINEAR_FUSED_QUANT_A_MAX_N", str(n))
    kernel_module._fused_quant_a_max_n.cache_clear()
    assert kernel_module._use_fused_quant_a(m, x.dtype, k, k, n)
    fused = mxfp8_linear(x, packed).clone()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mxfp8_linear(x, packed)
    graph.replay()
    torch.cuda.synchronize(device)

    assert torch.isfinite(separate).all()
    assert torch.count_nonzero(separate).item() > 0
    assert torch.equal(separate, fused)
    assert torch.equal(separate, captured)


def test_fused_quant_a_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("B12X_MXFP8_LINEAR_FUSED_QUANT_A_MAX_N", "4096")
    kernel_module._fused_quant_a_max_n.cache_clear()
    use = kernel_module._use_fused_quant_a
    assert use(8, torch.bfloat16, 7168, 7168, 4096)
    # More than eight tokens, FP16 activations, padded K or a wider N stay on
    # the separate quantizer.
    assert not use(9, torch.bfloat16, 7168, 7168, 4096)
    assert not use(8, torch.float16, 7168, 7168, 4096)
    assert not use(8, torch.bfloat16, 7136, 7168, 4096)
    assert not use(8, torch.bfloat16, 7168, 7168, 7168)
    # N tails (not a multiple of 64) have no fused handling.
    assert not use(8, torch.bfloat16, 7168, 7168, 132)
    assert not use(8, torch.bfloat16, 7168, 7168, 32)
    monkeypatch.setenv("B12X_MXFP8_LINEAR_FUSED_QUANT_A_MAX_N", "0")
    kernel_module._fused_quant_a_max_n.cache_clear()
    assert not use(1, torch.bfloat16, 7168, 7168, 64)
    # The limit is process configuration: it is read once, and it must be a
    # non-negative integer.
    monkeypatch.setenv("B12X_MXFP8_LINEAR_FUSED_QUANT_A_MAX_N", "4096")
    assert not use(1, torch.bfloat16, 7168, 7168, 64)
    kernel_module._fused_quant_a_max_n.cache_clear()
    assert use(1, torch.bfloat16, 7168, 7168, 64)
    for bad in ("-1", "many"):
        monkeypatch.setenv("B12X_MXFP8_LINEAR_FUSED_QUANT_A_MAX_N", bad)
        kernel_module._fused_quant_a_max_n.cache_clear()
        with pytest.raises(ValueError, match="B12X_MXFP8_LINEAR_FUSED_QUANT_A_MAX_N"):
            use(1, torch.bfloat16, 7168, 7168, 64)
    kernel_module._fused_quant_a_max_n.cache_clear()
