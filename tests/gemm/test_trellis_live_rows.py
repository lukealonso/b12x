import pytest
import torch

import b12x
from b12x.gemm import trellis_linear
from b12x.moe._shared.kernels.w4a16 import kernel
from b12x.moe._shared.kernels.w4a16.host import packed_gemm_scratch_elements
from b12x.policy.generation.providers.trellis import _hadamard_reference
from b12x.policy.generation.trellis_reference import _reconstruct_native
from tests.conftest import require_b12x


@pytest.mark.parametrize("columns", (128, 8192))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("block_rows", (16, 64))
def test_dense_trellis_live_rows_reuse_warmed_callable(columns, dtype, block_rows):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    capacity, width = 385, 128
    payload = torch.randint(-32768, 32767, (width // 16, columns // 16, 48),
                            device=device, dtype=torch.int16)
    scales = [torch.full((size,), .5, device=device, dtype=torch.float16) for size in (width, columns)]
    weight = trellis_linear.prepare_weight(payload, *scales, codebook="mcg", params_dtype=dtype)
    source = torch.randn((capacity, width), device=device).mul_(.01).to(dtype)
    decoded = _reconstruct_native(payload, codebook="mcg").to(device=device, dtype=dtype)
    rotated = _hadamard_reference(source.to(torch.float16) * scales[0]).to(dtype)
    projected = (rotated.float() @ decoded.float()).to(dtype).to(torch.float16)
    expected = (_hadamard_reference(projected) * scales[1]).to(dtype)
    sizes = {"output": (columns, dtype), "gemm_output": (columns, dtype),
             "input_f16": (width, torch.float16), "rotated_f16": (width, torch.float16),
             "rotated_compute": (width, dtype), "gemm_output_f16": (columns, torch.float16),
             "output_f16": (columns, torch.float16)}
    buffers = {name: torch.empty((capacity, size), device=device, dtype=dt) for name, (size, dt) in sizes.items()}
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    scratch = torch.empty(packed_gemm_scratch_elements(size_n=columns, route_slots=448,
        moe_block_size=block_rows, sms=sms), device=device, dtype=torch.float32)
    def run(rows):
        return trellis_linear.run(source[:rows], weight, c_tmp=scratch,
                                  _moe_block_size=block_rows,
                                  **{name: value[:rows] for name, value in buffers.items()})
    trellis_linear.clear_caches()
    run(1)
    torch.cuda.synchronize(device)
    warmed = {key: value.compiled for key, value in kernel._CACHE.items()}
    addresses = tuple(value.data_ptr() for value in (*buffers.values(), scratch))
    b12x.freeze_kernel_resolution("dense Trellis live rows must reuse one prepared callable")
    try:
        for rows in (1, 3, 65, 384, 385):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(rows)
            for _ in range(3):
                buffers["output"].fill_(float("nan"))
                allocated = torch.cuda.memory_allocated(device)
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == allocated
                actual = buffers["output"][:rows]
                assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
                torch.testing.assert_close(actual.float(), expected[:rows].float(), rtol=.02, atol=.001)
                assert torch.isnan(buffers["output"][rows:]).all()
                assert addresses == tuple(value.data_ptr() for value in (*buffers.values(), scratch))
            graph.reset()
        assert warmed == {key: value.compiled for key, value in kernel._CACHE.items()}
    finally:
        b12x.unfreeze_kernel_resolution()
