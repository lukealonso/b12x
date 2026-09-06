from __future__ import annotations

from dataclasses import replace
import gc
import weakref

import pytest
import torch

import b12x
from b12x._lib.intrinsics import swizzle_block_scale
from b12x.moe import fused_moe
from b12x.moe._shared.kernels.reference import moe_reference_nvfp4
from tests._reference.w4a16_reference import moe_reference_w4a16
from b12x.policy import MOE_DECODE, PolicyContext, PolicySource, get_auto_policy
from benchmarks.benchmark_w4a16_nvfp4_layouts import _check


def weight_plan(**kwargs):
    return fused_moe.plan_weights(
        source=kwargs.pop("source", fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat.MODELOPT_NVFP4,
            w13_layout=fused_moe.W13Layout.W13,
        )),
        geometry=fused_moe.MoEGeometry(num_experts=8, hidden_size=128, intermediate_size=128),
        activation=kwargs.pop("activation", fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.AUTO, nonlinearity="silu", io_dtype=torch.bfloat16,
        )), **kwargs,
    )


@pytest.mark.parametrize("field,value", [
    ("io_dtype", torch.float16), ("nonlinearity", "relu2"),
])
def test_auto_rejects_unsupported_activation(field, value):
    activation = replace(weight_plan().activation, **{field: value})
    with pytest.raises(ValueError, match="automatic MoE precision"):
        weight_plan(activation=activation)


def test_auto_requires_nvfp4_native_storage():
    plan = weight_plan()
    assert plan._impl.quant_modes == {"nvfp4", "w4a16"}
    assert plan.prepared_format.packing is fused_moe.WeightPacking.SOURCE_NATIVE
    with pytest.raises(ValueError, match="up/gate"):
        weight_plan(source=replace(plan.source, w13_layout=fused_moe.W13Layout.W31))
    with pytest.raises(ValueError, match="source-native"):
        weight_plan(constraints=fused_moe.WeightPlanConstraints(
            required_packing=fused_moe.WeightPacking.MMA_PACKED,
        ))
    with pytest.raises(ValueError, match="ModelOpt NVFP4"):
        weight_plan(source=fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat.MXFP4_E8M0_K32,
        ))


def test_uniform_nvfp4_a16_requires_only_mma_packing():
    activation = replace(weight_plan().activation, mode=fused_moe.ActivationMode.A16)
    plan = weight_plan(activation=activation)
    assert plan.prepared_format.available_packings == {fused_moe.WeightPacking.MMA_PACKED}
    with pytest.raises(ValueError, match="uniform NVFP4 W4A16 requires mma_packed"):
        weight_plan(activation=activation, constraints=fused_moe.WeightPlanConstraints(
            required_packing=fused_moe.WeightPacking.SOURCE_NATIVE,
        ))


@pytest.mark.parametrize("name", ["w13_blockscale", "w2_blockscale"])
@pytest.mark.parametrize("invalid", ["truncated", "strided", "dtype"])
def test_native_nvfp4_preparation_validates_scale_storage(name, invalid):
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_w4a16_modelopt_native_weights
    inputs = dict(
        w13_fp4=torch.empty((1, 256, 64), dtype=torch.uint8),
        w2_fp4=torch.empty((1, 128, 64), dtype=torch.uint8),
        w13_global_scale=torch.ones(1), w2_global_scale=torch.ones(1),
        w13_blockscale=torch.empty((1, 256, 8), dtype=torch.uint8),
        w2_blockscale=torch.empty((1, 128, 8), dtype=torch.uint8),
        activation="silu",
    )
    scales = inputs[name]
    if invalid == "truncated":
        inputs[name] = scales[:, :-1].contiguous()
    elif invalid == "strided":
        inputs[name] = scales.transpose(1, 2)
    else:
        inputs[name] = scales.float()
    with pytest.raises((ValueError, TypeError), match=name):
        prepare_w4a16_modelopt_native_weights(**inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_uniform_nvfp4_a16_does_not_retain_source_scales():
    plan = weight_plan(activation=replace(weight_plan().activation, mode=fused_moe.ActivationMode.A16))
    weights = fused_moe.PackedWeights(
        w13=torch.zeros((8, 256, 64), dtype=torch.uint8, device="cuda"),
        w2=torch.zeros((8, 128, 64), dtype=torch.uint8, device="cuda"),
        w13_block_scales=swizzle_block_scale(torch.ones((8, 256, 8), device="cuda").to(torch.float8_e4m3fn)),
        w2_block_scales=swizzle_block_scale(torch.ones((8, 128, 8), device="cuda").to(torch.float8_e4m3fn)),
        w13_global_scales=torch.ones(8, device="cuda"),
        w2_global_scales=torch.ones(8, device="cuda"),
    )
    source_scales = (weakref.ref(weights.w13_block_scales), weakref.ref(weights.w2_block_scales))
    experts = fused_moe.prepare_weights(plan=plan, weights=weights)
    prepared = experts._impl.representation_for("w4a16")
    assert prepared.weight_layout == "packed"
    assert prepared.w13.data_ptr() == weights.w13.data_ptr()
    assert prepared.w2.data_ptr() == weights.w2.data_ptr()
    assert prepared.w13_scale is experts._impl.w1_blockscale
    assert prepared.w2_scale is experts._impl.w2_blockscale
    del weights
    gc.collect()
    assert all(ref() is None for ref in source_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_auto_shared_weights_precision_plans_replay_without_resolution(monkeypatch):
    torch.manual_seed(927)
    device = torch.device("cuda", torch.cuda.current_device())
    def scales(shape):
        return swizzle_block_scale((torch.rand(shape, device=device) * 0.1 + 0.03).to(torch.float8_e4m3fn))
    packed = fused_moe.PackedWeights(
        w13=torch.randint(0, 256, (8, 256, 64), device=device, dtype=torch.uint8),
        w2=torch.randint(0, 256, (8, 128, 64), device=device, dtype=torch.uint8),
        w13_block_scales=scales((8, 256, 8)), w2_block_scales=scales((8, 128, 8)),
        w13_global_scales=torch.rand(8, device=device) * 0.1 + 0.1,
        w2_global_scales=torch.rand(8, device=device) * 0.1 + 0.1,
        input_scale=torch.full((8,), 128.0, device=device),
        intermediate_scale=torch.full((8,), 512.0, device=device),
    )
    originals = [t.clone() for t in (packed.w13, packed.w2, packed.w13_block_scales, packed.w2_block_scales)]
    import b12x.moe._shared.kernels.w4a16.prepare as preparation
    def no_conversion(*args, **kwargs):
        raise AssertionError("native preparation attempted a second scale layout")
    monkeypatch.setattr(preparation, "unswizzle_expert_scales", no_conversion)
    monkeypatch.setattr(preparation, "_permute_nvfp4_scales", no_conversion)
    torch.cuda.reset_peak_memory_stats(device)
    allocated = torch.cuda.memory_allocated(device)
    experts = fused_moe.prepare_weights(plan=weight_plan(), weights=packed)
    native = experts._impl.representation_for("w4a16")
    assert torch.cuda.max_memory_allocated(device) - allocated <= native.workspace.nbytes + 8192
    assert native.w13_scale.data_ptr() == packed.w13_block_scales.data_ptr()
    assert native.w2_scale.data_ptr() == packed.w2_block_scales.data_ptr()
    assert native.w13_global_scale.data_ptr() == packed.w13_global_scales.data_ptr()
    assert native.w2_global_scale.data_ptr() == packed.w2_global_scales.data_ptr()
    for source, a4, a16 in (
        (packed.w13, experts._impl.w1_fp4, native.w13),
        (packed.w2, experts._impl.w2_fp4, native.w2),
        (packed.w13_block_scales, experts._impl.w1_blockscale, native.micro_w13_scale),
        (packed.w2_block_scales, experts._impl.w2_blockscale, native.micro_w2_scale),
    ):
        assert source.data_ptr() == a4.data_ptr() == a16.data_ptr()
    plans = []
    for mode, config in (
        (fused_moe.ActivationMode.A4, fused_moe.MoeDecodeConfig(
            backend="micro", route_planner="internal", max_active_clusters=None,
        )),
        (fused_moe.ActivationMode.A16, fused_moe.MoeDecodeConfig(
            backend="w4a16", route_planner="internal", max_active_clusters=None,
            w4a16_route_mode="direct",
        )),
        (fused_moe.ActivationMode.A16, fused_moe.MoeDecodeConfig(
            backend="w4a16", route_planner="internal", max_active_clusters=None,
            w4a16_route_mode="packed",
        )),
    ):
        plan = fused_moe.plan_execution(
            experts=experts,
            capacity=fused_moe.ExecutionCapacity(max_tokens=8, top_k=2, warmup_token_counts=(1, 3, 8)),
            policy=get_auto_policy(device).with_override(MOE_DECODE, config),
        )
        assert plan.activation_mode is mode
        assert plan.precision_resolution.source is PolicySource.OVERRIDE
        fused_moe.prewarm(plan)
        spec, = plan.scratch_specs()
        plans.append((mode, plan, torch.empty(spec.shape, device=device, dtype=spec.dtype)))

    def no_resolution(*args, **kwargs):
        raise AssertionError("bind or run performed policy resolution")
    monkeypatch.setattr(PolicyContext, "resolve", no_resolution)
    b12x.freeze_kernel_resolution("shared native NVFP4 precision graph test")
    try:
        for m in (1, 3, 8):
            x = (torch.randn(m, 128, device=device) * 0.25).to(torch.bfloat16)
            ids = torch.stack([torch.randperm(8, device=device)[:2] for _ in range(m)]).to(torch.int32)
            routes = torch.softmax(torch.randn(m, 2, device=device), dim=-1)
            for mode, plan, scratch in plans:
                output = torch.empty_like(x)
                binding = fused_moe.bind(plan, scratch=scratch, a=x, experts=experts,
                                         topk_ids=ids, topk_weights=routes, output=output,
                                         input_scales_static=True)
                assert binding.unit_scale_contract == (mode is fused_moe.ActivationMode.A16)
                args = (x, packed.w13, packed.w13_block_scales, packed.w13_global_scales,
                        packed.w2, packed.w2_block_scales, packed.w2_global_scales)
                if mode is fused_moe.ActivationMode.A16:
                    expected = moe_reference_w4a16(*args, ids, routes, 8, 128, 128)
                    oracle_mode = "w4a16"
                else:
                    expected = moe_reference_nvfp4(
                        x, packed.w13, packed.w13_block_scales, experts._impl.w1_alphas,
                        packed.w2, packed.w2_block_scales, experts._impl.w2_alphas,
                        packed.input_scale, packed.intermediate_scale, ids, routes, 8, 128, 128,
                        quant_scale_math="reciprocal_multiply",
                    )
                    oracle_mode = "nvfp4"
                fused_moe.run(binding=binding)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fused_moe.run(binding=binding)
                output.fill_(float("nan"))
                allocated = torch.cuda.memory_allocated(device)
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == allocated
                _check(output, expected, name=mode.value, m=m, oracle_mode=oracle_mode)
                graph.reset()
    finally:
        b12x.unfreeze_kernel_resolution()
    for original, actual in zip(originals, (packed.w13, packed.w2, packed.w13_block_scales, packed.w2_block_scales)):
        assert torch.equal(original.view(torch.uint8), actual.view(torch.uint8))
