"""Contracts for the fused SM120 unpaired K6/MCG dense linear."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.gemm import trellis_linear
from b12x.gemm.trellis_linear import _k6_mcg_cute


def _weight_descriptor(**overrides):
    descriptor = {
        "params_dtype": torch.float16,
        "weight_layout": "trellis_t256",
        "num_experts": 1,
        "trellis_bits": 6,
        "trellis_codebook": "mcg",
        "trellis_pair_kind": None,
        "trellis_rate_axis": None,
        "in_features": 2048,
        "out_features": 4096,
    }
    descriptor.update(overrides)
    return SimpleNamespace(**descriptor)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"params_dtype": torch.bfloat16}, False),
        ({"weight_layout": "mxfp4"}, False),
        ({"num_experts": 2}, False),
        ({"trellis_bits": 5}, False),
        ({"trellis_codebook": "sqg_e4m3"}, False),
        ({"trellis_pair_kind": "P24", "trellis_rate_axis": "n"}, False),
        ({"trellis_pair_kind": "P33", "trellis_rate_axis": "k"}, False),
        ({"trellis_pair_kind": "P24"}, False),
        ({"trellis_rate_axis": "n"}, False),
        ({"in_features": 2112}, False),
        ({"out_features": 4032}, False),
    ],
)
def test_unpaired_weight_gate_excludes_qsrt_pairs(
    overrides: dict[str, object], expected: bool
) -> None:
    assert (
        _k6_mcg_cute._is_unpaired_k6_mcg_weight(_weight_descriptor(**overrides))
        is expected
    )


@pytest.mark.parametrize(
    ("size_k", "size_n", "expected"),
    [
        (2048, 4096, 131072),
        (6144, 1024, 65536),
        (512, 6144, 98304),
        (128, 128, 2048),
    ],
)
def test_scratch_contract_covers_split_k_owners(
    size_k: int, size_n: int, expected: int
) -> None:
    assert trellis_linear.k6_mcg_small_m_scratch_elements(size_k, size_n) == expected


@pytest.mark.parametrize("shape", [(0, 128), (128, 0), (192, 128), (128, 192)])
def test_scratch_contract_rejects_unaligned_shapes(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="positive multiples of 128"):
        trellis_linear.k6_mcg_small_m_scratch_elements(*shape)


def test_rotation_grid_and_gemm_grid_have_independent_capacity() -> None:
    launch = _k6_mcg_cute.K6McgSmallMCompileResult(
        compiled=None,
        device_index=0,
        size_k=6144,
        size_n=1024,
        grid_x=32,
        cta_threads=256,
        resident_ctas=376,
        blocks_per_sm=2,
        shared_memory_bytes=0,
        registers_per_thread=0,
        local_memory_bytes=0,
    )

    assert launch.launch_grid_x(1) == 32
    assert launch.launch_grid_x(4) == 32
    assert launch.launch_grid_x(8) == 49
    assert launch.launch_grid_x(16) == 97


def test_bound_launch_rejects_non_cuda_runtime_inputs() -> None:
    launch = _k6_mcg_cute.K6McgSmallMCompileResult(
        compiled=None,
        device_index=0,
        size_k=128,
        size_n=128,
        grid_x=1,
        cta_threads=256,
        resident_ctas=1,
        blocks_per_sm=1,
        shared_memory_bytes=0,
        registers_per_thread=0,
        local_memory_bytes=0,
    )

    assert not launch.accepts_input(torch.empty((1, 128), dtype=torch.float16))


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return int(props.major) == 12


def _buffers(
    rows: int, size_k: int, size_n: int, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "output": torch.empty((rows, size_n), dtype=torch.float16, device=device),
        "rotated_f16": torch.empty((rows, size_k), dtype=torch.float16, device=device),
        "c_tmp": torch.empty(
            (trellis_linear.k6_mcg_small_m_scratch_elements(size_k, size_n),),
            dtype=torch.float32,
            device=device,
        ),
    }


def _hadamard_128_reference(
    source: torch.Tensor,
    output: torch.Tensor,
    pre_scale: torch.Tensor | None,
    post_scale: torch.Tensor | None,
    scale: float,
) -> None:
    """Apply normalized Hadamard-128 from its mathematical definition."""
    values = source.float()
    if pre_scale is not None:
        values = values * pre_scale.float()

    blocks = values.reshape(*values.shape[:-1], -1, 128)
    stride = 1
    while stride < 128:
        stages = blocks.reshape(*blocks.shape[:-1], -1, 2, stride)
        lower = stages[..., 0, :]
        upper = stages[..., 1, :]
        blocks = torch.stack((lower + upper, lower - upper), dim=-2).flatten(-3, -1)
        stride *= 2
    values = blocks.reshape_as(values) * (float(scale) / (128.0**0.5))

    if post_scale is not None:
        values = values * post_scale.float()
    output.copy_(values.to(output.dtype))


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize(
    ("rows", "size_k", "size_n"),
    [
        (1, 128, 128),
        (4, 128, 128),
        (8, 128, 128),
        (11, 128, 128),
        (16, 128, 128),
        # Exercise split-K reduction and the multi-CTA input rotation on a
        # production GLM-5.2 projection geometry.
        (8, 6144, 1024),
    ],
)
def test_fused_k6_mcg_matches_separate_rotation_pipeline(
    rows: int, size_k: int, size_n: int
) -> None:
    torch.manual_seed(0x4B364D43 + rows)
    device = torch.device("cuda", torch.cuda.current_device())
    trellis = torch.randint(
        -32768,
        32767,
        (size_k // 16, size_n // 16, 96),
        dtype=torch.int16,
        device=device,
    )
    suh = torch.where(
        torch.rand(size_k, device=device) > 0.5,
        torch.ones(size_k, device=device),
        -torch.ones(size_k, device=device),
    ).to(torch.float16)
    svh = torch.where(
        torch.rand(size_n, device=device) > 0.5,
        torch.ones(size_n, device=device),
        -torch.ones(size_n, device=device),
    ).to(torch.float16)
    weight = trellis_linear.prepare_weight(
        trellis,
        suh,
        svh,
        codebook="mcg",
        params_dtype=torch.float16,
    )
    launch = weight.k6_mcg_small_m_launch
    assert isinstance(launch, _k6_mcg_cute.K6McgSmallMCompileResult)
    if (size_k, size_n) == (6144, 1024):
        assert launch.grid_x > 1
        assert launch.launch_grid_x(rows) > launch.grid_x
    source = (torch.randn((rows, size_k), device=device) * 0.05).to(torch.float16)
    noncontiguous = torch.empty((rows, size_k * 2), dtype=torch.float16, device=device)[
        :, ::2
    ]
    assert not noncontiguous.is_contiguous()
    assert not _k6_mcg_cute.is_k6_mcg_small_m_eligible(noncontiguous, weight)

    reference_buffers = _buffers(rows, size_k, size_n, device)
    reference = trellis_linear.run(
        source,
        weight,
        hadamard_128=_hadamard_128_reference,
        **reference_buffers,
    ).clone()
    weight.workspace.zero_()
    fused_buffers = _buffers(rows, size_k, size_n, device)
    actual = trellis_linear.run(source, weight, **fused_buffers).clone()
    torch.cuda.synchronize(device)

    delta = actual.float() - reference.float()
    relative_l2 = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(
        reference.float()
    ).clamp_min(1.0e-12)
    assert float(relative_l2) <= 2.0e-3
    assert float(delta.abs().max()) <= 1.0


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_fused_k6_mcg_cuda_graph_replay_is_stable() -> None:
    torch.manual_seed(0x4B364752)
    device = torch.device("cuda", torch.cuda.current_device())
    rows = 8
    size_k = size_n = 128
    trellis = torch.randint(
        -32768,
        32767,
        (size_k // 16, size_n // 16, 96),
        dtype=torch.int16,
        device=device,
    )
    signs = torch.ones(size_k, dtype=torch.float16, device=device)
    weight = trellis_linear.prepare_weight(
        trellis,
        signs,
        signs.clone(),
        codebook="mcg",
        params_dtype=torch.float16,
    )
    assert isinstance(
        weight.k6_mcg_small_m_launch, _k6_mcg_cute.K6McgSmallMCompileResult
    )
    source = (torch.randn((rows, size_k), device=device) * 0.05).to(torch.float16)
    buffers = _buffers(rows, size_k, size_n, device)
    _k6_mcg_cute.clear_k6_mcg_small_m_cache()
    freeze_kernel_resolution("bound K6/MCG launch must survive cache clearing")
    try:
        expected = trellis_linear.run(source, weight, **buffers).clone()
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = trellis_linear.run(source, weight, **buffers)
        for _ in range(5):
            buffers["output"].fill_(float("nan"))
            graph.replay()
        torch.cuda.synchronize(device)
    finally:
        unfreeze_kernel_resolution()

    assert torch.equal(captured, expected)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("missing", ["output", "rotated_f16", "c_tmp"])
def test_fused_k6_mcg_capture_requires_caller_owned_buffers(
    missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    rows = 8
    size_k = size_n = 128
    trellis = torch.zeros(
        (size_k // 16, size_n // 16, 96), dtype=torch.int16, device=device
    )
    signs = torch.ones(size_k, dtype=torch.float16, device=device)
    weight = trellis_linear.prepare_weight(
        trellis,
        signs,
        signs.clone(),
        codebook="mcg",
        params_dtype=torch.float16,
    )
    source = torch.zeros((rows, size_k), dtype=torch.float16, device=device)
    buffers = _buffers(rows, size_k, size_n, device)

    buffers.pop(missing)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        trellis_linear.run(source, weight, **buffers)
