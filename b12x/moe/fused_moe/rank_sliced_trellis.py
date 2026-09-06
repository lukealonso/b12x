"""Adopt rank-sliced Trellis checkpoint tensors into fused-MoE plans."""

from __future__ import annotations

import torch

from .._shared.execution import MoEWeightPreparationPlan, PreparedWeightLayout
from .._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights
from ._impl import B12XFP4ExpertWeights, _PreparedWeightRepresentation


def prepare_rank_sliced_trellis_weights(
    *,
    plan: MoEWeightPreparationPlan,
    w13: torch.Tensor,
    w2: torch.Tensor,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate_rotations: torch.Tensor,
    down_svh: torch.Tensor,
) -> B12XFP4ExpertWeights:
    """Validate and wrap one uniform-rate, rank-sliced Trellis layer.

    The checkpoint tensors already use B12X's native projection-major tile
    layout, so preparation reuses their storage through flat views instead of
    copying model-sized payloads. This adapter deliberately accepts only the
    uniform-rate contract; mixed-rate checkpoints use the canonical
    atom-container path instead.
    """

    if not isinstance(plan, MoEWeightPreparationPlan):
        raise TypeError("plan must be a MoEWeightPreparationPlan")
    if plan.source_format != "b12x_trellis":
        raise ValueError(
            "rank-sliced Trellis preparation requires source_format='b12x_trellis'"
        )
    if plan.quant_modes != frozenset({"w4a16"}):
        raise ValueError("rank-sliced Trellis preparation requires w4a16")
    if plan.trellis_rate_granularity != "uniform":
        raise ValueError("rank-sliced Trellis preparation requires uniform rates")
    if plan.trellis_pair_kinds is not None or plan.coupled_hadamard:
        raise ValueError(
            "rank-sliced Trellis preparation does not accept paired-rate or "
            "coupled-Hadamard plans"
        )
    if plan.trellis_tile_config is None or plan.trellis_bits is None:
        raise ValueError("rank-sliced Trellis plan is missing tile or bitrate data")
    if plan.required_weight_layout("w4a16") is not PreparedWeightLayout.TRELLIS_NATIVE:
        raise ValueError("rank-sliced Trellis plan must select trellis_native")

    tile_config = plan.trellis_tile_config
    prepared = prepare_trellis256_moe_weights(
        w13=w13,
        w2=w2,
        hidden_size=plan.hidden_size,
        intermediate_size=plan.intermediate_size,
        num_experts=plan.num_experts,
        activation=plan.activation,
        fc1_tile_n=tile_config[1],
        fc2_tile_n=tile_config[3],
        # Full-rotation Trellis kernels use FP16 prepared scratch independently
        # of the model I/O dtype carried by the execution plan.
        params_dtype=torch.float16,
        w13_layout="trellis_t256_proj",
        trellis_bits=plan.trellis_bits,
        codebook=plan.trellis_codebook or "mcg",
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        tile_config=tile_config,
    )
    representation = _PreparedWeightRepresentation(
        quant_mode="w4a16",
        layout=PreparedWeightLayout.TRELLIS_NATIVE,
        value=prepared,
    )
    unit_input = torch.ones((), dtype=torch.float32, device=prepared.w13.device)
    return B12XFP4ExpertWeights(
        plan=plan,
        a1_gscale=unit_input,
        w1_fp4=prepared.w13,
        w1_blockscale=prepared.w13_scale,
        w1_alphas=prepared.w13_global_scale,
        a2_gscale=unit_input,
        w2_fp4=prepared.w2,
        w2_blockscale=prepared.w2_scale,
        w2_alphas=prepared.w2_global_scale,
        representation=representation,
    )


__all__ = ["prepare_rank_sliced_trellis_weights"]
