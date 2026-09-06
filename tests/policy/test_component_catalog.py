from __future__ import annotations

import b12x
from b12x.attention.qsa._policy import QSA_POLICY, QsaQuery
from b12x.gemm.bf16_vocab_projection._policy import (
    BF16_VOCAB_PROJECTION_POLICY,
    Bf16VocabProjectionQuery,
)
from b12x.moe.fused_moe._policy import MOE_DECODE_POLICY, MoeDecodeQuery
from b12x.policy import (
    EMBEDDED_REGISTRY,
    PlanningPolicyMode,
    PolicyContext,
    PolicyMode,
    PolicySource,
    list_planning_components,
    list_profiled_components,
    validate_component_profile_contract,
)
from b12x.policy.generation import ComponentGeneratorRegistry
from b12x.policy.generation.moe_corpus import COMMON_PREFILL_TOKEN_CAPACITIES
from b12x.policy.generation.providers import register_builtin_generators


def test_every_planned_op_has_exactly_one_policy_registration() -> None:
    planned_ops = {
        meta.qualname for meta in b12x.list_ops() if meta.api_style == "planned"
    }
    registrations = list_planning_components()

    assert len({item.op_qualname for item in registrations}) == len(registrations)
    assert {item.op_qualname for item in registrations} == planned_ops
    assert all(item.mode is PlanningPolicyMode.PROFILED for item in registrations)


def test_profiled_policies_generators_and_embedded_profile_stay_in_lockstep() -> None:
    registrations = list_profiled_components()
    expected_ids = tuple(str(item.component_id) for item in registrations)
    generator_registry = ComponentGeneratorRegistry()
    register_builtin_generators(generator_registry)
    profile = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm")

    assert generator_registry.component_ids() == expected_ids
    assert tuple(component.component_id for component in profile.components) == (
        expected_ids
    )
    for registration in registrations:
        policy = registration.load_policy()
        generator = registration.create_generator()
        component = profile.component(str(registration.component_id))
        assert component is not None
        assert generator.component_id == policy.component_id
        assert generator.query_schema_version == policy.query_schema_version
        assert generator.config_schema_version == policy.config_schema_version
        validate_component_profile_contract(policy, component)


def test_qwen_flash_next_qsa_serving_shape_resolves_from_gb10_profile() -> None:
    profile = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm")
    context = PolicyContext.for_identity(
        profile.targets[0],
        mode=PolicyMode.PREPLANNED_ONLY,
    )

    resolution = context.resolve(
        QSA_POLICY,
        QsaQuery(
            q_dtype="bfloat16",
            kv_dtype="float8_e4m3fn",
            q_heads=24,
            kv_heads=2,
            head_dim=256,
            index_heads=4,
            index_kv_heads=1,
            index_head_dim=128,
            index_rotary_dim=64,
            main_page_size=16,
            max_batch=1,
            max_q_rows=4,
            max_seq_len=65_536,
            max_speculative_tokens=3,
            compress_ratio=4,
            budget=2048,
            position_axes=3,
            mrope_interleaved=True,
        ),
    )

    assert resolution.source is PolicySource.PREPLANNED
    assert resolution.rule_name == "qualified-split-sparse-gqa"
    assert resolution.config.backend == "cutedsl"
    assert resolution.config.sparse_gqa_direct_kv_warps == 2


def test_qsa_unmeasured_direct_prefill_uses_cute_heuristic() -> None:
    profile = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm")
    context = PolicyContext.for_identity(profile.targets[0])
    resolution = context.resolve(
        QSA_POLICY,
        QsaQuery(
            q_dtype="bfloat16",
            kv_dtype="float8_e4m3fn",
            q_heads=24,
            kv_heads=2,
            head_dim=256,
            index_heads=4,
            index_kv_heads=1,
            index_head_dim=128,
            index_rotary_dim=64,
            main_page_size=3_008,
            max_batch=4,
            max_q_rows=6_016,
            max_seq_len=32_768,
            max_speculative_tokens=3,
            compress_ratio=4,
            budget=2_048,
            position_axes=3,
            mrope_interleaved=True,
        ),
    )

    assert resolution.source is PolicySource.HEURISTIC
    assert resolution.config.backend == "cutedsl"
    assert resolution.config.sparse_gqa_direct_kv_warps == 2


def test_qwen_flash_next_vocab_projection_resolves_from_gb10_profile() -> None:
    profile = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm")
    context = PolicyContext.for_identity(
        profile.targets[0],
        mode=PolicyMode.PREPLANNED_ONLY,
    )

    resolution = context.resolve(
        BF16_VOCAB_PROJECTION_POLICY,
        Bf16VocabProjectionQuery(
            dtype="bfloat16",
            max_tokens=1,
            in_features=2_560,
            out_features=248_320,
        ),
    )

    assert resolution.source is PolicySource.PREPLANNED
    assert resolution.config.backend == "triton"
    assert resolution.config.algorithm == "row"
    assert resolution.config.num_warps == 8


def test_qwen_flash_next_prefill_moe_resolves_from_embedded_profiles() -> None:
    for profile_id in (
        "nvidia.gb10.48sm",
        "nvidia.rtx.pro.6000.blackwell",
    ):
        profile = EMBEDDED_REGISTRY.get(profile_id)
        context = PolicyContext.for_identity(
            profile.targets[0],
            mode=PolicyMode.PREPLANNED_ONLY,
        )

        for num_tokens in COMMON_PREFILL_TOKEN_CAPACITIES:
            resolution = context.resolve(
                MOE_DECODE_POLICY,
                MoeDecodeQuery(
                    activation="silu",
                    hidden_size=2_560,
                    intermediate_size=640,
                    num_experts=512,
                    num_tokens=num_tokens,
                    quant_mode="nvfp4",
                    routed_rows=num_tokens * 10,
                    source_format="modelopt_nvfp4",
                    top_k=10,
                ),
            )

            assert resolution.source is PolicySource.PREPLANNED
            assert resolution.config.backend == "dynamic"
            assert resolution.config.route_planner == "internal"


def test_qwen_flash_next_128_token_components_resolve_from_gb10_profile() -> None:
    from b12x.norm.hyperconnection._policy import (
        HYPERCONNECTION_POLICY,
        HyperConnectionQuery,
    )
    from b12x.sequence.mtp_feedback._policy import (
        MTP_FEEDBACK_POLICY,
        MtpFeedbackQuery,
    )

    profile = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm")
    context = PolicyContext.for_identity(
        profile.targets[0],
        mode=PolicyMode.PREPLANNED_ONLY,
    )

    hyperconnection = context.resolve(
        HYPERCONNECTION_POLICY,
        HyperConnectionQuery(
            dtype="bfloat16",
            max_tokens=128,
            hidden_size=2_560,
            streams=4,
            lowrank=320,
        ),
    )
    mtp_feedback = context.resolve(
        MTP_FEEDBACK_POLICY,
        MtpFeedbackQuery(
            dtype="bfloat16",
            max_tokens=128,
            hidden_size=2_560,
            streams=4,
        ),
    )

    assert hyperconnection.source is PolicySource.PREPLANNED
    assert mtp_feedback.source is PolicySource.PREPLANNED


def test_qwen_flash_next_planners_are_profile_backed() -> None:
    registrations = {item.op_qualname: item for item in list_planning_components()}

    for op_qualname in (
        "norm.hyperconnection",
        "sequence.mtp_feedback",
        "sequence.ple",
        "sequence.ple_embedding",
        "sequence.ple_hash",
    ):
        registration = registrations[op_qualname]
        assert registration.mode is PlanningPolicyMode.PROFILED
        assert registration.component_id is not None


def test_unknown_gpu_uses_the_component_heuristic() -> None:
    from b12x.norm.hyperconnection._policy import (
        HYPERCONNECTION_POLICY,
        HyperConnectionQuery,
    )
    from b12x.policy import DeviceIdentity

    context = PolicyContext.for_identity(
        DeviceIdentity(
            vendor="nvidia",
            compute_capability=(12, 1),
            sm_count=47,
            product_name="unknown synthetic gpu",
        )
    )
    resolution = context.resolve(
        HYPERCONNECTION_POLICY,
        HyperConnectionQuery(
            dtype="bfloat16",
            max_tokens=4,
            hidden_size=2560,
            streams=4,
            lowrank=320,
        ),
    )

    assert resolution.source is PolicySource.HEURISTIC
    assert resolution.config.reduction_block_h == 4096


def test_every_profile_generator_has_real_gpu_measurement_work() -> None:
    for registration in list_profiled_components():
        generator = registration.create_generator()
        assert generator.component_id == registration.component_id
        assert "precomputed" not in type(generator).__module__
