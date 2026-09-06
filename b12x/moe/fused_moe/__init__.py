"""Fused tensor-parallel MoE for SM12x.

The canonical API keeps checkpoint encoding, prepared representation, and
activation precision independent:

    import torch
    from b12x.moe import fused_moe

    source = fused_moe.PackedSource(
        format=fused_moe.PackedSourceFormat.MXFP4_E8M0_K32,
        w13_layout=fused_moe.W13Layout.W31,
    )
    activation = fused_moe.ActivationSpec(
        mode=fused_moe.ActivationMode.A8,
        nonlinearity="silu",
        io_dtype=torch.bfloat16,
    )
    geometry = fused_moe.MoEGeometry(
        num_experts=128,
        hidden_size=4096,
        intermediate_size=14336,
    )
    weight_plan = fused_moe.plan_weights(
        source=source,
        activation=activation,
        geometry=geometry,
    )
    experts = fused_moe.prepare_weights(plan=weight_plan, weights=weights)
    execution = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=4096, top_k=8),
    )
    fused_moe.prewarm(execution)
    spec = execution.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = fused_moe.bind(
        execution,
        scratch=scratch,
        a=x,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    output = fused_moe.run(binding=binding)

``quant_modes`` planning, flat tensor preparation, ``Caps``/``plan``, and the
lightweight ``plan_execution`` call remain available as the compatibility
contract consumed by official vLLM releases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="fused_moe",
    group="moe",
    api_style="planned",
    entry_points=(
        "ActivationMode",
        "ActivationSpec",
        "Binding",
        "Caps",
        "ExecutionCapacity",
        "ExecutionPlan",
        "ExecutionVariant",
        "ExpertWeights",
        "MoEGeometry",
        "MoeDecodeConfig",
        "MoeDecodeQuery",
        "PackedSource",
        "PackedSourceFormat",
        "PackedWeights",
        "Plan",
        "PreparedExperts",
        "PreparedWeightFormat",
        "RouteBinding",
        "Routing",
        "RoutingSpec",
        "ScaleEncoding",
        "ScaleFactors",
        "ScratchRequirement",
        "SparseBinding",
        "TrellisConfig",
        "TrellisWeights",
        "W13Layout",
        "WeightEncoding",
        "WeightPacking",
        "WeightPlan",
        "WeightPlanConstraints",
        "WeightSource",
        "WeightsPlan",
        "bind",
        "bind_route",
        "bind_sparse",
        "clear_caches",
        "is_supported",
        "plan",
        "plan_execution",
        "plan_weights",
        "prepare_fc2_weights",
        "prepare_weights",
        "prewarm",
        "prewarm_fc2",
        "required_nbytes",
        "route",
        "route_topk",
        "run",
        "run_fc2",
        "run_sparse",
    ),
    dtypes=("bf16", "fp16"),
    recipes=(
        "nvfp4",
        "w4a8_mx",
        "w4a8_nvfp4",
        "w6a8_mx",
        "w4a16",
        "b12x_trellis",
    ),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/integration/tp_moe.py", "b12x/moe/"),
    ),
    test_path="tests/moe/test_fused_moe.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        ActivationMode,
        ActivationSpec,
        Binding,
        Caps,
        ExecutionCapacity,
        ExecutionPlan,
        ExecutionVariant,
        ExpertWeights,
        MoEGeometry,
        MoeDecodeConfig,
        MoeDecodeQuery,
        PackedSource,
        PackedSourceFormat,
        PackedWeights,
        Plan,
        PreparedExperts,
        PreparedWeightFormat,
        RouteBinding,
        Routing,
        RoutingSpec,
        ScaleEncoding,
        ScaleFactors,
        ScratchRequirement,
        SparseBinding,
        TrellisConfig,
        TrellisWeights,
        W13Layout,
        WeightEncoding,
        WeightPacking,
        WeightPlan,
        WeightPlanConstraints,
        WeightSource,
        WeightsPlan,
        bind,
        bind_route,
        bind_sparse,
        clear_caches,
        is_supported,
        plan,
        plan_execution,
        plan_weights,
        prepare_fc2_weights,
        prepare_weights,
        prewarm,
        prewarm_fc2,
        required_nbytes,
        route,
        route_topk,
        run,
        run_fc2,
        run_sparse,
    )

install_lazy_api(globals(), META)
