from __future__ import annotations

import pytest
import torch

from b12x.sequence import ple_embedding
from b12x.sequence.ple_embedding import reference
from b12x.sequence.ple_hash.reference import (
    nth_prime_after,
    ple_hash_packed_reference,
)

from ..conftest import require_b12x


def _small_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([5, 7, 11, 13], dtype=torch.int64),
        torch.tensor([0, 5, 12, 23], dtype=torch.int64),
        torch.tensor([11, 13, 17], dtype=torch.int64),
    )


def _small_plan(
    device: torch.device | str,
    *,
    tp_rank: int = 1,
    tp_size: int = 2,
    max_tokens: int = 5,
    quant_mode: str = "fp8_e4m3_per_tensor",
) -> ple_embedding.Plan:
    prime_sizes, table_offsets, multipliers = _small_geometry()
    return ple_embedding.plan(
        ple_embedding.Caps(
            device=device,
            max_tokens=max_tokens,
            max_seqs=2,
            vocab_size=100,
            eos_token_id=99,
            max_order=3,
            heads_per_order=2,
            dense_layer_ordinal=0,
            base_table_size=5,
            embedding_dim=64,
            tp_size=tp_size,
            tp_rank=tp_rank,
            table_alignment=8,
            quant_mode=quant_mode,
        ),
        prime_sizes=prime_sizes,
        table_offsets=table_offsets,
        multipliers=multipliers,
    )


def _fp8_weight(shape: tuple[int, int], device: torch.device | str) -> torch.Tensor:
    values = torch.arange(shape[0] * shape[1], dtype=torch.float32, device=device)
    values = values.remainder(15).sub(7).view(shape)
    return values.to(torch.float8_e4m3fn).contiguous()


def _nvfp4_weight(shape: tuple[int, int], device: torch.device | str) -> torch.Tensor:
    codes = torch.arange(
        shape[0] * shape[1] * 2, dtype=torch.uint8, device=device
    ).remainder_(16)
    codes = codes.view(shape[0], shape[1], 2)
    return (codes[..., 0] | (codes[..., 1] << 4)).contiguous()


def _storage(
    plan: ple_embedding.Plan,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    mode = plan.caps.quant_mode
    device = plan.caps.device
    if mode == "bf16":
        values = torch.arange(
            plan.weight_shape[0] * plan.weight_shape[1],
            dtype=torch.float32,
            device=device,
        )
        weight = values.remainder(17).sub(8).view(plan.weight_shape).to(torch.bfloat16)
        return weight.contiguous(), None, None
    if mode == "fp8_e4m3_per_tensor":
        return (
            _fp8_weight(plan.weight_shape, device),
            torch.tensor([0.25], dtype=torch.bfloat16, device=device),
            None,
        )
    assert mode == "nvfp4_group16"
    scale_values = torch.arange(
        plan.weight_scale_shape[0] * plan.weight_scale_shape[1],
        dtype=torch.float32,
        device=device,
    )
    scales = scale_values.remainder(4).add(1).mul(0.5)
    return (
        _nvfp4_weight(plan.weight_shape, device),
        scales.view(plan.weight_scale_shape).to(torch.float8_e4m3fn).contiguous(),
        torch.tensor([0.25], dtype=torch.float32, device=device),
    )


def _bind_small(
    plan: ple_embedding.Plan,
    *,
    num_tokens: int = 4,
    num_seqs: int = 2,
) -> ple_embedding.Binding:
    device = plan.caps.device
    scratch_spec = plan.scratch_specs()[0]
    weight, weight_scale, weight_scale_2 = _storage(plan)
    token_ids = torch.tensor([3, 4, 5, 6, 0], dtype=torch.int64, device=device)
    if plan.caps.max_tokens != 5:
        token_ids = torch.arange(
            plan.caps.max_tokens, dtype=torch.int64, device=device
        ).add_(3)
    return plan.bind(
        scratch=torch.empty(
            scratch_spec.shape, dtype=scratch_spec.dtype, device=device
        ),
        weight=weight,
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2,
        token_ids=token_ids,
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32, device=device),
        committed_history=torch.tensor(
            [[99, 99], [7, 8]], dtype=torch.int64, device=device
        ),
        num_seqs=torch.tensor([num_seqs], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([num_tokens], dtype=torch.int32, device=device),
        out=torch.full(plan.output_shape, 37, dtype=plan.output_dtype, device=device),
    )


def _reference(binding: ple_embedding.Binding) -> torch.Tensor:
    plan = binding.plan
    return reference.fused(
        binding.weight,
        binding.weight_scale,
        binding.token_ids,
        binding.query_start_loc,
        binding.committed_history,
        quant_mode=plan.caps.quant_mode,
        weight_scale_2=binding.weight_scale_2,
        num_seqs=int(binding.num_seqs.item()),
        num_tokens=int(binding.num_tokens.item()),
        eos_token_id=plan.caps.eos_token_id,
        multipliers=plan.multipliers,
        prime_sizes=plan.prime_sizes,
        table_offsets=plan.table_offsets,
        heads_per_order=plan.caps.heads_per_order,
        shard_start=plan.shard_start,
        embedding_dim=plan.caps.embedding_dim,
        output_dtype=plan.output_dtype,
    )


@pytest.mark.parametrize(
    (
        "quant_mode",
        "weight_shape",
        "weight_dtype",
        "scale_shape",
        "scale_dtype",
        "scale_2_shape",
        "scale_2_dtype",
    ),
    [
        ("bf16", (20, 16), torch.bfloat16, None, None, None, None),
        (
            "fp8_e4m3_per_tensor",
            (20, 16),
            torch.float8_e4m3fn,
            (1,),
            torch.bfloat16,
            None,
            None,
        ),
        (
            "nvfp4_group16",
            (20, 8),
            torch.uint8,
            (20, 1),
            torch.float8_e4m3fn,
            (1,),
            torch.float32,
        ),
    ],
)
def test_plan_owns_hash_geometry_tp_partition_and_storage_contract(
    quant_mode: str,
    weight_shape: tuple[int, int],
    weight_dtype: torch.dtype,
    scale_shape: tuple[int, ...] | None,
    scale_dtype: torch.dtype | None,
    scale_2_shape: tuple[int, ...] | None,
    scale_2_dtype: torch.dtype | None,
) -> None:
    plan = _small_plan("cpu", quant_mode=quant_mode)

    assert plan.head_count == 4
    assert plan.head_dim == 16
    assert plan.table_vocab_size == 36
    assert plan.padded_vocab_size == 40
    assert (plan.shard_start, plan.shard_end) == (20, 40)
    assert plan.weight_shape == weight_shape
    assert plan.weight_dtype == weight_dtype
    assert plan.weight_scale_shape == plan.scale_shape == scale_shape
    assert plan.weight_scale_dtype == scale_dtype
    assert plan.weight_scale_2_shape == scale_2_shape
    assert plan.weight_scale_2_dtype == scale_2_dtype
    assert plan._ids_shape == (5, 4)
    assert plan.output_shape == (5, 64)
    assert plan.output_dtype == torch.bfloat16
    assert plan._layout.ids_offset_bytes % 1024 == 0
    assert plan.scratch_specs()[0].nbytes >= 5 * 4 * torch.int64.itemsize
    torch.testing.assert_close(
        plan.prime_sizes.cpu(), _small_geometry()[0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        plan.table_offsets.cpu(), _small_geometry()[1], rtol=0, atol=0
    )
    torch.testing.assert_close(
        plan.multipliers.cpu(), _small_geometry()[2], rtol=0, atol=0
    )


@pytest.mark.parametrize("tp_rank", range(4))
def test_plan_matches_16_head_320m_table_geometry(tp_rank: int) -> None:
    plan = ple_embedding.plan(
        ple_embedding.Caps(
            device="cpu",
            max_tokens=1,
            max_seqs=1,
            vocab_size=248_320,
            eos_token_id=248_044,
            max_order=3,
            heads_per_order=8,
            dense_layer_ordinal=0,
            base_table_size=20_000_000,
            embedding_dim=2_560,
            tp_size=4,
            tp_rank=tp_rank,
            table_alignment=128,
            quant_mode="bf16",
        )
    )

    assert plan.head_count == 16
    assert plan.head_dim == 160
    assert plan.table_vocab_size == 320_001_446
    assert plan.padded_vocab_size == 320_001_536
    assert plan.shard_start == tp_rank * 80_000_384
    assert plan.shard_end == (tp_rank + 1) * 80_000_384
    assert plan.weight_shape == (80_000_384, 160)


def test_caps_and_plan_reject_unsupported_storage_contracts() -> None:
    common = dict(
        device="cpu",
        max_tokens=2,
        max_seqs=1,
        vocab_size=100,
        eos_token_id=99,
        max_order=3,
        heads_per_order=2,
        dense_layer_ordinal=0,
        base_table_size=5,
        embedding_dim=64,
        tp_size=2,
        tp_rank=0,
        table_alignment=8,
    )
    with pytest.raises(ValueError, match="embedding_dim=.*head_count"):
        ple_embedding.Caps(**{**common, "embedding_dim": 65})
    with pytest.raises(ValueError, match="tp_rank"):
        ple_embedding.Caps(**{**common, "tp_rank": 2})
    with pytest.raises(ValueError, match="quant_mode"):
        ple_embedding.Caps(**{**common, "quant_mode": "int8"})
    with pytest.raises(ValueError, match="table_memory"):
        ple_embedding.Caps(**{**common, "table_memory": "managed"})
    with pytest.raises(ValueError, match="requires a CUDA device"):
        ple_embedding.Caps(**{**common, "table_memory": "mapped_host"})
    with pytest.raises(TypeError, match="BF16.*scale_dtype must be None"):
        ple_embedding.Caps(
            **{**common, "quant_mode": "bf16", "scale_dtype": torch.bfloat16}
        )
    with pytest.raises(TypeError, match="scale.*torch.bfloat16"):
        ple_embedding.Caps(
            **{
                **common,
                "quant_mode": "fp8_e4m3_per_tensor",
                "scale_dtype": torch.float32,
            }
        )
    with pytest.raises(TypeError, match="scale.*torch.float8_e4m3fn"):
        ple_embedding.Caps(
            **{
                **common,
                "quant_mode": "nvfp4_group16",
                "scale_dtype": torch.bfloat16,
            }
        )
    with pytest.raises(ValueError, match="head_dim=.*divisible by 16"):
        ple_embedding.Caps(
            **{**common, "quant_mode": "nvfp4_group16", "embedding_dim": 12}
        )
    with pytest.raises(TypeError, match="output.*torch.bfloat16"):
        ple_embedding.Caps(**{**common, "output_dtype": torch.float16})

    prime_sizes, table_offsets, multipliers = _small_geometry()
    caps = ple_embedding.Caps(**{**common, "tp_size": 3})
    with pytest.raises(ValueError, match="padded_vocab_size=.*divisible"):
        ple_embedding.plan(
            caps,
            prime_sizes=prime_sizes,
            table_offsets=table_offsets,
            multipliers=multipliers,
        )


@pytest.mark.parametrize("quant_mode", ["bf16", "fp8_e4m3_per_tensor", "nvfp4_group16"])
def test_bind_uses_caller_storage_and_rejects_bad_quant_tensors(
    quant_mode: str,
) -> None:
    plan = _small_plan("cpu", quant_mode=quant_mode)
    binding = _bind_small(plan)
    scratch_start = binding.scratch.data_ptr()
    scratch_end = scratch_start + binding.scratch.numel()

    assert scratch_start <= binding._hash_scratch.data_ptr() < scratch_end
    assert scratch_start <= binding._ids.data_ptr() < scratch_end
    assert binding.out.shape == plan.output_shape
    if quant_mode == "bf16":
        assert binding.weight_scale is None
    else:
        assert binding.weight_scale is not None
    if quant_mode == "nvfp4_group16":
        assert binding.weight_scale_2 is not None
    else:
        assert binding.weight_scale_2 is None
    assert binding.error_code is binding._hash_binding.error_code

    kwargs = dict(
        scratch=binding.scratch,
        weight=binding.weight,
        weight_scale=binding.weight_scale,
        weight_scale_2=binding.weight_scale_2,
        token_ids=binding.token_ids,
        query_start_loc=binding.query_start_loc,
        committed_history=binding.committed_history,
        num_seqs=binding.num_seqs,
        num_tokens=binding.num_tokens,
        out=binding.out,
    )
    with pytest.raises(TypeError, match="weight must have dtype"):
        plan.bind(**{**kwargs, "weight": binding.weight.float()})
    with pytest.raises(ValueError, match="weight must have shape"):
        plan.bind(**{**kwargs, "weight": binding.weight[:-1]})
    if quant_mode == "bf16":
        with pytest.raises(ValueError, match="weight_scale must be None"):
            plan.bind(
                **{
                    **kwargs,
                    "weight_scale": torch.ones(1, dtype=torch.bfloat16),
                }
            )
        with pytest.raises(ValueError, match="weight_scale_2 must be None"):
            plan.bind(
                **{
                    **kwargs,
                    "weight_scale_2": torch.ones(1, dtype=torch.float32),
                }
            )
    elif quant_mode == "fp8_e4m3_per_tensor":
        assert binding.weight_scale is not None
        with pytest.raises(TypeError, match="weight_scale must have dtype"):
            plan.bind(**{**kwargs, "weight_scale": binding.weight_scale.float()})
        with pytest.raises(ValueError, match="weight_scale_2 must be None"):
            plan.bind(
                **{
                    **kwargs,
                    "weight_scale_2": torch.ones(1, dtype=torch.float32),
                }
            )
    else:
        assert binding.weight_scale is not None
        assert binding.weight_scale_2 is not None
        with pytest.raises(TypeError, match="weight_scale must have dtype"):
            plan.bind(**{**kwargs, "weight_scale": binding.weight_scale.float()})
        with pytest.raises(ValueError, match="weight_scale_2 is required"):
            plan.bind(**{**kwargs, "weight_scale_2": None})
        with pytest.raises(TypeError, match="weight_scale_2 must have dtype"):
            plan.bind(
                **{
                    **kwargs,
                    "weight_scale_2": binding.weight_scale_2.to(torch.bfloat16),
                }
            )
    with pytest.raises(ValueError, match="scratch"):
        plan.bind(**{**kwargs, "scratch": binding.scratch[:-1]})
    with pytest.raises(ValueError, match="out must have shape"):
        plan.bind(**{**kwargs, "out": binding.out[:-1]})

    with pytest.raises(ValueError, match="GPU run requires CUDA"):
        ple_embedding.run(binding)


@pytest.mark.parametrize("quant_mode", ["bf16", "fp8_e4m3_per_tensor", "nvfp4_group16"])
def test_reference_composes_hash_tp_lookup_and_dequantization(
    quant_mode: str,
) -> None:
    plan0 = _small_plan("cpu", tp_rank=0, quant_mode=quant_mode)
    plan1 = _small_plan("cpu", tp_rank=1, quant_mode=quant_mode)
    full_plan = _small_plan("cpu", tp_rank=0, tp_size=1, quant_mode=quant_mode)
    full_weight, full_scale, full_scale_2 = _storage(full_plan)
    token_ids = torch.tensor([3, 4, 5, 6, 0], dtype=torch.int64)
    starts = torch.tensor([0, 2, 4], dtype=torch.int32)
    history = torch.tensor([[99, 99], [7, 8]], dtype=torch.int64)

    rank_outputs = []
    for plan in (plan0, plan1):
        local_scale = full_scale
        if quant_mode == "nvfp4_group16":
            assert full_scale is not None
            local_scale = full_scale[plan.shard_start : plan.shard_end]
        rank_outputs.append(
            reference.fused(
                full_weight[plan.shard_start : plan.shard_end],
                local_scale,
                token_ids,
                starts,
                history,
                quant_mode=quant_mode,
                weight_scale_2=full_scale_2,
                num_seqs=2,
                num_tokens=4,
                eos_token_id=plan.caps.eos_token_id,
                multipliers=plan.multipliers,
                prime_sizes=plan.prime_sizes,
                table_offsets=plan.table_offsets,
                heads_per_order=plan.caps.heads_per_order,
                shard_start=plan.shard_start,
                embedding_dim=plan.caps.embedding_dim,
            )
        )

    ids = ple_hash_packed_reference(
        token_ids[:4],
        starts,
        history,
        eos_token_id=99,
        multipliers=plan0.multipliers,
        prime_sizes=plan0.prime_sizes,
        table_offsets=plan0.table_offsets,
        heads_per_order=2,
    )
    selected = torch.index_select(full_weight, 0, ids.flatten())
    if quant_mode == "bf16":
        gathered = selected.reshape(4, 4, 16).float()
    elif quant_mode == "fp8_e4m3_per_tensor":
        assert full_scale is not None
        gathered = selected.reshape(4, 4, 16).float() * full_scale.float()
    else:
        assert full_scale is not None and full_scale_2 is not None
        low = selected & 0xF
        high = (selected >> 4) & 0xF
        codes = torch.stack((low, high), dim=-1).reshape(4, 4, 16)
        lut = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
        )
        selected_scales = torch.index_select(full_scale, 0, ids.flatten())
        gathered = lut[codes.long()] * selected_scales.float().reshape(4, 4, 1)
        gathered = gathered * full_scale_2.float()
    expected = torch.zeros((5, 64), dtype=torch.bfloat16)
    expected[:4].copy_(gathered.to(torch.bfloat16).flatten(-2))

    actual = rank_outputs[0] + rank_outputs[1]
    assert torch.count_nonzero(actual[:4]).item() > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual[4]).item() == 0


def test_nvfp4_reference_decodes_low_nibble_first_and_applies_both_scales() -> None:
    row0_codes = torch.arange(16, dtype=torch.uint8)
    row1_codes = torch.arange(15, -1, -1, dtype=torch.uint8)
    codes = torch.stack((row0_codes, row1_codes))
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    scales = torch.tensor([[2.0], [4.0]], dtype=torch.float8_e4m3fn)
    scale_2 = torch.tensor([0.25], dtype=torch.float32)
    ids = torch.tensor([[10, 11, 9, 12]], dtype=torch.int64)

    actual = reference.lookup(
        packed,
        scales,
        ids,
        quant_mode="nvfp4_group16",
        weight_scale_2=scale_2,
        shard_start=10,
        embedding_dim=64,
    )

    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
    )
    expected = torch.zeros((1, 4, 16), dtype=torch.bfloat16)
    expected[0, 0].copy_((lut * 2.0 * 0.25).to(torch.bfloat16))
    expected[0, 1].copy_((lut.flip(0) * 4.0 * 0.25).to(torch.bfloat16))
    torch.testing.assert_close(actual, expected.flatten(-2), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("quant_mode", "weight", "weight_scale", "weight_scale_2"),
    [
        (
            "bf16",
            torch.arange(32, dtype=torch.bfloat16).reshape(2, 16),
            None,
            None,
        ),
        (
            "fp8_e4m3_per_tensor",
            torch.arange(32, dtype=torch.float32)
            .reshape(2, 16)
            .to(torch.float8_e4m3fn),
            torch.tensor([0.5], dtype=torch.bfloat16),
            None,
        ),
    ],
)
def test_reference_gathers_only_selected_rows_and_zeros_nonlocal_heads(
    quant_mode: str,
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    weight_scale_2: torch.Tensor | None,
) -> None:
    ids = torch.tensor([[101, 99], [100, 102]], dtype=torch.int64)
    actual = reference.lookup(
        weight,
        weight_scale,
        ids,
        quant_mode=quant_mode,
        weight_scale_2=weight_scale_2,
        shard_start=100,
        embedding_dim=32,
        num_tokens=1,
    )
    scale = 1.0 if weight_scale is None else float(weight_scale.item())
    expected = torch.zeros((2, 32), dtype=torch.bfloat16)
    expected[0, :16].copy_((weight[1].float() * scale).to(torch.bfloat16))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("quant_mode", ["bf16", "fp8_e4m3_per_tensor", "nvfp4_group16"])
def test_cuda_matches_reference_and_preserves_read_only_tensors(
    quant_mode: str,
) -> None:
    device = require_b12x()
    plan = _small_plan(device, quant_mode=quant_mode)
    binding = _bind_small(plan)
    expected = _reference(binding)
    read_names = [
        "weight",
        "token_ids",
        "query_start_loc",
        "committed_history",
        "num_seqs",
        "num_tokens",
    ]
    if binding.weight_scale is not None:
        read_names.append("weight_scale")
    if binding.weight_scale_2 is not None:
        read_names.append("weight_scale_2")
    read_only = {name: getattr(binding, name).clone() for name in read_names}

    actual = ple_embedding.run(binding)
    torch.cuda.synchronize(device)

    assert actual.data_ptr() == binding.out.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual[4]).item() == 0
    for name, before in read_only.items():
        torch.testing.assert_close(getattr(binding, name), before, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize(
    ("quant_mode", "custom_op"),
    [
        ("bf16", "ple_embedding_bf16_pipeline"),
        ("fp8_e4m3_per_tensor", "ple_embedding_fp8_pipeline"),
        ("nvfp4_group16", "ple_embedding_nvfp4_pipeline"),
    ],
)
def test_cuda_public_run_exports_as_one_opaque_fullgraph_custom_op(
    quant_mode: str,
    custom_op: str,
) -> None:
    device = require_b12x()
    binding = _bind_small(_small_plan(device, quant_mode=quant_mode))
    expected = _reference(binding)

    graph, _ = torch._dynamo.export(lambda: ple_embedding.run(binding))()
    assert f"torch.ops.b12x.{custom_op}" in graph.code
    assert "ple_hash_pipeline" not in graph.code
    assert "ple_embedding_fp8_lookup" not in graph.code
    assert "ple_embedding_nvfp4_lookup" not in graph.code
    assert "triton" not in graph.code

    compiled = torch.compile(lambda: ple_embedding.run(binding), fullgraph=True)
    actual = compiled()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize("quant_mode", ["bf16", "fp8_e4m3_per_tensor", "nvfp4_group16"])
@pytest.mark.parametrize("token_count", [None, 3])
def test_cuda_graph_replay_uses_bound_storage_without_allocating(
    quant_mode: str,
    token_count: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from triton.runtime.jit import JITFunction

    from b12x.sequence.ple_embedding import _kernels
    from b12x.sequence.ple_hash import _kernels as hash_kernels

    device = require_b12x()
    binding = _bind_small(_small_plan(device, quant_mode=quant_mode))
    ple_embedding.run(binding)

    def reject_resolution(*args, **kwargs):
        pytest.fail("Live launch bounds must reuse the warmed embedding kernels")

    for module in (_kernels, hash_kernels):
        for kernel in vars(module).values():
            if isinstance(kernel, JITFunction):
                monkeypatch.setattr(kernel, "_do_compile", reject_resolution)

    binding.num_seqs.zero_()
    binding.num_tokens.zero_()
    for bound in (0, 1, 3, 5):
        binding.out.fill_(37)
        actual = ple_embedding.run(binding, token_count=bound)
        assert bool((actual == 0).all().item())
        assert bool((binding.out[bound:] == 37).all().item())
    for bound in (-1, 6):
        with pytest.raises(ValueError, match="planned token capacity"):
            ple_embedding.run(binding, token_count=bound)

    binding.num_seqs.fill_(2)
    binding.num_tokens.fill_(4)
    binding.out.fill_(37)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ple_embedding.run(binding, token_count=token_count)
    output_address = captured.data_ptr()
    scratch_address = binding.scratch.data_ptr()
    if token_count is not None:
        # Device metadata cannot exceed the launch bound even if it fits capacity.
        graph.replay()
        assert binding.error_code.item() & 1
        assert bool((captured == 0).all().item())

    if quant_mode == "bf16":
        binding.weight.neg_()
    elif quant_mode == "fp8_e4m3_per_tensor":
        replacement = (
            _fp8_weight(binding.plan.weight_shape, device)
            .float()
            .neg_()
            .to(torch.float8_e4m3fn)
        )
        binding.weight.copy_(replacement)
        assert binding.weight_scale is not None
        binding.weight_scale.fill_(0.5)
    else:
        binding.weight.bitwise_xor_(0x88)
        assert binding.weight_scale is not None
        assert binding.weight_scale_2 is not None
        binding.weight_scale.fill_(0.5)
        binding.weight_scale_2.fill_(0.75)
    binding.token_ids[:2].copy_(torch.tensor([8, 9], dtype=torch.int64, device=device))
    binding.query_start_loc.copy_(
        torch.tensor([0, 2, 0], dtype=torch.int32, device=device)
    )
    binding.num_seqs.fill_(1)
    binding.num_tokens.fill_(2)
    expected = _reference(binding)
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)

    assert captured.data_ptr() == output_address == binding.out.data_ptr()
    assert binding.scratch.data_ptr() == scratch_address
    assert allocated_after == allocated_before
    torch.testing.assert_close(captured, expected[:token_count], rtol=0, atol=0)
    assert binding.error_code.item() == 0
    if token_count is not None:
        assert bool((binding.out[token_count:] == 37).all().item())


@torch.inference_mode()
def test_cuda_large_local_row_uses_int64_scaled_addressing() -> None:
    device = require_b12x()
    prime = nth_prime_after(1 << 24, 1)
    target_id = prime - 1
    plan = ple_embedding.plan(
        ple_embedding.Caps(
            device=device,
            max_tokens=1,
            max_seqs=1,
            vocab_size=prime + 2,
            eos_token_id=prime + 1,
            max_order=2,
            heads_per_order=1,
            dense_layer_ordinal=0,
            base_table_size=prime,
            embedding_dim=160,
            tp_size=1,
            tp_rank=0,
            table_alignment=128,
        ),
        prime_sizes=torch.tensor([prime], dtype=torch.int64),
        table_offsets=torch.tensor([0], dtype=torch.int64),
        multipliers=torch.tensor([1, 1], dtype=torch.int64),
    )
    assert target_id * plan.head_dim > 2**31
    weight = torch.empty(plan.weight_shape, dtype=torch.float8_e4m3fn, device=device)
    weight[target_id].fill_(2.0)
    scratch_spec = plan.scratch_specs()[0]
    binding = plan.bind(
        scratch=torch.empty(
            scratch_spec.shape, dtype=scratch_spec.dtype, device=device
        ),
        weight=weight,
        weight_scale=torch.tensor([0.25], dtype=torch.bfloat16, device=device),
        weight_scale_2=None,
        token_ids=torch.tensor([target_id], dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        committed_history=torch.tensor([[0]], dtype=torch.int64, device=device),
        num_seqs=torch.tensor([1], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([1], dtype=torch.int32, device=device),
        out=torch.empty(plan.output_shape, dtype=torch.bfloat16, device=device),
    )

    actual = ple_embedding.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(
        actual,
        torch.full_like(actual, 0.5),
        rtol=0,
        atol=0,
    )
