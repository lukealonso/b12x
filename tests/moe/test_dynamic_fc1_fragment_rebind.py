"""FC1 must not consume stale fragment aliases after a prior task's FC2.

The normal tiny corpus has fewer work items than resident CTAs, so each CTA
can exit after one task. Force one resident CTA to exercise repeated tasks
without changing the production route/compute implementation.
"""

from __future__ import annotations

import pytest
import torch

from b12x.moe import fused_moe
from b12x.policy import MOE_DECODE, get_auto_policy
from tests._reference.helpers import prepare_tp_moe_fp4_experts, require_b12x
from tests.moe.test_cute_migration_moe_standard_corpus import (
    _BoundCase,
    _make_inputs,
    _make_nvfp4_weights,
    _nvfp4_oracle,
    _reset_dispatch_environment,
    _run_live_graph_check,
)


@pytest.mark.parametrize("work_source", ["persistent_grid", "materialized_queue"])
@pytest.mark.parametrize(
    ("tile_m", "deterministic"),
    [(16, False), (32, False), (64, False), (128, False), (64, True)],
)
def test_dynamic_fc1_fragments_follow_current_task(
    monkeypatch: pytest.MonkeyPatch,
    work_source: str,
    tile_m: int,
    deterministic: bool,
) -> None:
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    monkeypatch.setenv("B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS", "1")
    monkeypatch.setenv("B12X_DYNAMIC_WORK_SOURCE", work_source)
    monkeypatch.setenv("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", str(int(deterministic)))
    # Two intermediate slices also exercise the inner grouped-slice loop in
    # deterministic mode, not just transitions between unrelated tasks.
    intermediate_size = 256 if deterministic else 128
    weights = _make_nvfp4_weights(device, seed=201, intermediate_size=intermediate_size)
    initial = _make_inputs(device, m=128, seed=202, route_shift=0)
    changed = _make_inputs(device, m=128, seed=203, route_shift=2)
    initial_reference = _nvfp4_oracle(
        weights, initial, intermediate_size=intermediate_size
    )
    changed_reference = _nvfp4_oracle(
        weights, changed, intermediate_size=intermediate_size
    )
    experts = prepare_tp_moe_fp4_experts(
        a=initial.a,
        a1_gscale=weights.a1_scale,
        w1_fp4=weights.w1_fp4,
        w1_blockscale=weights.w1_scale,
        w1_alphas=weights.w1_alpha,
        a2_gscale=weights.a2_scale,
        w2_fp4=weights.w2_fp4,
        w2_blockscale=weights.w2_scale,
        w2_alphas=weights.w2_alpha,
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
    )
    policy = get_auto_policy(initial.a.device).with_override(
        MOE_DECODE,
        fused_moe.MoeDecodeConfig(
            backend="dynamic",
            route_planner="internal",
            max_active_clusters=None,
            dynamic_tile_m=tile_m,
            dynamic_route_mode="grouped",
        ),
    )
    plan = fused_moe.plan(
        fused_moe.Caps(
            max_tokens=128,
            num_topk=2,
            device=initial.a.device,
            weight_plan=experts.plan,
            quant_mode="nvfp4",
            core_token_counts=(128,),
            frozen=True,
            policy_context=policy,
            deterministic_output=deterministic,
        )
    )
    scratch = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in plan.scratch_specs()
    )
    output = torch.empty_like(initial.a)
    binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=initial.a,
        experts=experts,
        topk_ids=initial.topk_ids,
        topk_weights=initial.topk_weights,
        output=output,
        input_scales_static=True,
        # The inherited Torch oracle models precise quantization. Native
        # fast-math coverage is retained separately on real GLM weights.
        fast_math=False,
    )
    case = _BoundCase(weights, experts, plan, scratch, binding)
    assert case.binding.implementation == "dynamic"
    assert case.scratch_plan.launch_plan.execution.tile_m == tile_m
    assert (
        case.scratch_plan.launch_plan.policy_resolution.config.dynamic_tile_m == tile_m
    )
    assert case.scratch_plan.launch_plan.deterministic_output == deterministic
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_reference,
        changed_reference=changed_reference,
        context=f"fc1-rebind-m{tile_m}-{work_source}-det{deterministic}",
        min_cos=0.999,
        max_normalized_rmse=0.03,
        replay_count=3,
        assert_no_replay_allocations=True,
    )
