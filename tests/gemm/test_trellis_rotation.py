"""Dense CuTe rotations: numerical boundaries and graph-safe live row counts."""

import pytest
import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.gemm.trellis_linear import _rotation


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _reference(source, pre, post):
    value = source if pre is None else (source * pre).half()
    matrix = torch.ones((1, 1), device=source.device, dtype=torch.float64)
    for _ in range(7):
        matrix = torch.cat((torch.cat((matrix, matrix), dim=1),
                            torch.cat((matrix, -matrix), dim=1)), dim=0)
    result = (value.double().reshape(-1, 128) @ (matrix / 128**0.5)).half().reshape_as(source)
    return result if post is None else (result * post).half()


@pytest.mark.parametrize("columns", (128, 384, 4096))
@pytest.mark.parametrize("mode", ("none", "pre", "post", "both"))
def test_hadamard_scale_rounding_and_inplace(columns, mode):
    rng = torch.Generator(device="cuda").manual_seed(712)
    source = torch.randn((7, columns), device="cuda", dtype=torch.float16, generator=rng)
    pre = torch.randn(columns, device="cuda", dtype=torch.float16, generator=rng) if mode in ("pre", "both") else None
    post = torch.randn(columns, device="cuda", dtype=torch.float16, generator=rng) if mode in ("post", "both") else None
    expected = _reference(source, pre, post)
    output = torch.full_like(source, float("nan"))
    _rotation.hadamard_128(source, output, pre, post)
    torch.testing.assert_close(output, expected, rtol=1e-3, atol=2e-3)
    _rotation.hadamard_128(source, source, pre, post)
    assert torch.equal(source, output)


@pytest.mark.parametrize("mode", ("pre", "post"))
def test_hadamard_live_rows_reuse_callable_under_frozen_capture(mode):
    source = torch.randn((65, 384), device="cuda", dtype=torch.float16)
    output = torch.empty_like(source)
    scales = torch.randn((384,), device="cuda", dtype=torch.float16)
    pre, post = (scales, None) if mode == "pre" else (None, scales)
    _rotation.hadamard_128(source[:1], output[:1], pre, post)
    compiled = _rotation._resolve(384, pre is not None, post is not None)
    expected = _reference(source, pre, post)
    freeze_kernel_resolution("Hadamard live rows must reuse one callable")
    try:
        for rows in (1, 3, 8, 33, 65):
            assert _rotation._resolve(384, pre is not None, post is not None) is compiled
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _rotation.hadamard_128(source[:rows], output[:rows], pre, post)
            pointers = (source.data_ptr(), output.data_ptr())
            allocated = torch.cuda.memory_allocated()
            for _ in range(3):
                output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == allocated
                assert pointers == (source.data_ptr(), output.data_ptr())
                torch.testing.assert_close(output[:rows], expected[:rows], rtol=1e-3, atol=2e-3)
                assert torch.isnan(output[rows:]).all()
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.parametrize("mode", ("pre", "post"))
def test_hadamard_basis_and_fp16_boundaries(mode):
    source = torch.eye(128, device="cuda", dtype=torch.float16)
    signs = torch.where(torch.arange(128, device="cuda") % 3 == 0, -0.75, 1.125).half()
    pre, post = (signs, None) if mode == "pre" else (None, signs)
    output = torch.empty_like(source)
    _rotation.hadamard_128(source, output, pre, post)
    assert torch.equal(output, _reference(source, pre, post))
