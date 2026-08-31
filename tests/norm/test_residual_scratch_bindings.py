from __future__ import annotations

import pytest
import torch

import b12x.norm.mhc._impl as residual_impl
from b12x.norm.mhc._impl import B12XMHCBinding, B12XMHCScratchCaps, plan_mhc_scratch
from b12x.norm.mhc._impl import MHC_DEFAULT_BLOCK_K, MHC_DEFAULT_SPLIT_K
from b12x.norm.mhc._policy import MHC_POLICY, MhcConfig, MhcQuery
from b12x.policy import (
    MHC,
    DeviceIdentity,
    PolicyContext,
    PolicyMode,
    PolicySource,
)


def _medium_tf32_config(*, stages: int) -> MhcConfig:
    return MhcConfig(
        backend="tf32_tma",
        decode_partials_schedule="default",
        projection_tile_m=64,
        projection_tile_n=24,
        projection_tile_k=64,
        projection_num_stages=stages,
        projection_num_m_warps=4,
        projection_num_n_warps=1,
        projection_k_splits=8,
    )


def test_mhc_policy_owns_medium_prefill_projection_geometry() -> None:
    config = _medium_tf32_config(stages=2)
    policy = PolicyContext.for_identity(
        None,
        mode=PolicyMode.HEURISTIC_ONLY,
    ).with_override(MHC, config)
    plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device="cpu",
            max_tokens=3_072,
            hidden_size=4_096,
            split_k=64,
        ),
        policy=policy,
    )

    assert plan.config is config
    assert plan.config.projection_tile_m == 64
    assert plan.config.projection_tile_n == 24
    assert plan.config.projection_tile_k == 64
    assert plan.config.projection_num_stages == 2
    assert plan.config.projection_k_splits == 8


@pytest.mark.parametrize(
    ("device", "tile_m", "k_splits"),
    (
        (
            DeviceIdentity(
                vendor="nvidia",
                product_name="nvidia gb10",
                compute_capability=(12, 1),
                sm_count=48,
            ),
            128,
            4,
        ),
        (
            DeviceIdentity(
                vendor="nvidia",
                product_name=(
                    "nvidia rtx pro 6000 blackwell max-q workstation edition"
                ),
                compute_capability=(12, 0),
                sm_count=188,
            ),
            64,
            8,
        ),
    ),
)
def test_embedded_mhc_profiles_resolve_measured_medium_prefill_geometry(
    device: DeviceIdentity,
    tile_m: int,
    k_splits: int,
) -> None:
    resolution = PolicyContext.for_identity(device).resolve(
        MHC_POLICY,
        MhcQuery(
            dtype="bfloat16",
            max_tokens=3_072,
            hidden_size=4_096,
            split_k=64,
        ),
    )

    assert resolution.source is PolicySource.PREPLANNED
    assert resolution.config.backend == "tf32_tma"
    assert resolution.config.projection_tile_m == tile_m
    assert resolution.config.projection_k_splits == k_splits


def test_rtx_profile_scopes_mhc_decode_schedule_to_measured_capacity() -> None:
    device = DeviceIdentity(
        vendor="nvidia",
        product_name=(
            "nvidia rtx pro 6000 blackwell max-q workstation edition"
        ),
        compute_capability=(12, 0),
        sm_count=188,
    )
    policy = PolicyContext.for_identity(
        device,
        mode=PolicyMode.PREPLANNED_ONLY,
    )

    resolutions = {
        tokens: policy.resolve(
            MHC_POLICY,
            MhcQuery(
                dtype="bfloat16",
                max_tokens=tokens,
                hidden_size=4_096,
                split_k=64,
            ),
        )
        for tokens in (127, 128, 129)
    }

    assert all(
        resolution.source is PolicySource.PREPLANNED
        for resolution in resolutions.values()
    )
    assert resolutions[127].config.decode_partials_schedule == "default"
    assert (
        resolutions[128].config.decode_partials_schedule
        == "hidden4096_m128_v1"
    )
    assert resolutions[129].config.decode_partials_schedule == "default"


def test_mhc_plan_owns_profiled_decode_schedule() -> None:
    base = MHC_POLICY.heuristic(
        MhcQuery(
            dtype="bfloat16",
            max_tokens=128,
            hidden_size=4_096,
            split_k=64,
        ),
        None,
    )
    config = MhcConfig(
        **{
            **base.to_dict(),
            "decode_partials_schedule": "hidden4096_m128_v1",
        }
    )
    policy = PolicyContext.for_identity(
        None,
        mode=PolicyMode.HEURISTIC_ONLY,
    ).with_override(MHC, config)

    plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device="cpu",
            max_tokens=128,
            hidden_size=4_096,
            split_k=64,
        ),
        policy=policy,
    )

    assert plan.config is config
    assert plan.config.decode_partials_schedule == "hidden4096_m128_v1"


def test_mhc_profiled_decode_schedule_rejects_unmeasured_queries() -> None:
    config = MhcConfig(
        **{
            **MHC_POLICY.heuristic(
                MhcQuery(
                    dtype="bfloat16",
                    max_tokens=128,
                    hidden_size=4_096,
                    split_k=64,
                ),
                None,
            ).to_dict(),
            "decode_partials_schedule": "hidden4096_m128_v1",
        }
    )

    with pytest.raises(ValueError, match="requires native mHC"):
        MHC_POLICY.validate_config(
            MhcQuery(
                dtype="bfloat16",
                max_tokens=64,
                hidden_size=4_096,
                split_k=64,
            ),
            config,
            None,
        )


@pytest.mark.parametrize(
    ("tokens", "tile_m", "stages", "k_splits"),
    (
        (2_303, 16, 1, 1),
        (2_304, 64, 3, 8),
        (3_072, 64, 2, 8),
        (3_584, 192, 2, 8),
        (8_192, 128, 2, 4),
    ),
)
def test_mhc_heuristic_has_explicit_prefill_capacity_regimes(
    tokens: int,
    tile_m: int,
    stages: int,
    k_splits: int,
) -> None:
    query = MhcQuery(
        dtype="bfloat16",
        max_tokens=tokens,
        hidden_size=4_096,
        split_k=64,
    )

    config = MHC_POLICY.heuristic(query, None)
    MHC_POLICY.validate_config(query, config, None)

    assert config.backend == "tf32_tma"
    assert config.projection_tile_m == tile_m
    assert config.projection_num_stages == stages
    assert config.projection_k_splits == k_splits


def test_mhc_scratch_plan_exposes_one_component_scratch_spec() -> None:
    plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device="cpu",
            max_tokens=4,
            hidden_size=16,
            split_k=8,
        )
    )

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "mhc.scratch"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert specs[0].nbytes == plan.layout.nbytes
    assert plan.caps.max_tokens == 4
    assert plan.caps.hidden_size == 16
    assert plan.caps.split_k == 8


def test_mhc_scratch_plan_binds_caller_owned_scratch(monkeypatch) -> None:
    plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device="cpu",
            max_tokens=4,
            hidden_size=16,
            split_k=8,
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)

    binding = plan.bind(scratch=scratch)

    assert isinstance(binding, B12XMHCBinding)
    assert not hasattr(binding, "workspace")
    assert binding.partials.shape == (4, 8, residual_impl.MHC_PARTIALS)
    assert binding.y is None
    assert binding.post_buffer is None
    assert binding.comb_buffer is None
    assert binding.out is None
    assert binding.split_k == 8
    assert binding.partials.device == scratch.device


def test_mhc_scratch_plan_binds_live_token_shape() -> None:
    plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device="cpu",
            max_tokens=4,
            hidden_size=16,
            split_k=8,
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    y = torch.empty((2, 16), dtype=torch.bfloat16)
    post = torch.empty((2, 4), dtype=torch.float32)
    comb = torch.empty((2, 4, 4), dtype=torch.float32)
    out = torch.empty((2, 4, 16), dtype=torch.bfloat16)

    binding = plan.bind(
        scratch=scratch,
        tokens=2,
        y=y,
        post=post,
        comb=comb,
        out=out,
    )

    assert binding.partials.shape == (2, 8, residual_impl.MHC_PARTIALS)
    assert binding.y is y
    assert binding.post_buffer is post
    assert binding.comb_buffer is comb
    assert binding.out is out


def test_mhc_plan_binding_maps_caller_owned_outputs() -> None:
    plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device="cpu",
            max_tokens=4,
            hidden_size=16,
            split_k=8,
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    y = torch.empty((4, 16), dtype=torch.bfloat16)
    post = torch.empty((4, 4), dtype=torch.float32)
    comb = torch.empty((4, 4, 4), dtype=torch.float32)
    out = torch.empty((4, 4, 16), dtype=torch.bfloat16)

    binding = plan.bind(scratch=scratch, y=y, post=post, comb=comb, out=out)

    assert isinstance(binding, B12XMHCBinding)
    assert not hasattr(binding, "workspace")
    assert binding.partials.data_ptr() == scratch.data_ptr()
    assert binding.y is y
    assert binding.post_buffer is post
    assert binding.comb_buffer is comb
    assert binding.out is out


def test_mhc_prefill_bf16_project_policy_defaults_to_large_expected_m(monkeypatch) -> None:
    monkeypatch.delenv("B12X_MHC_PREFILL_BF16_MMA", raising=False)
    monkeypatch.delenv("B12X_MHC_PREFILL_BF16_MIN_TOKENS", raising=False)
    norm_weight = torch.empty((16,), dtype=torch.bfloat16)
    fn_bf16 = torch.empty((24, 64), dtype=torch.bfloat16)

    assert not residual_impl._use_mhc_prefill_bf16_project(
        norm_weight=norm_weight,
        policy_m=352,
        fn_bf16=fn_bf16,
    )
    assert residual_impl._use_mhc_prefill_bf16_project(
        norm_weight=norm_weight,
        policy_m=384,
        fn_bf16=fn_bf16,
    )
    assert not residual_impl._use_mhc_prefill_bf16_project(
        norm_weight=None,
        policy_m=4096,
        fn_bf16=fn_bf16,
    )
    assert not residual_impl._use_mhc_prefill_bf16_project(
        norm_weight=norm_weight,
        policy_m=4096,
        fn_bf16=None,
    )


def test_mhc_prefill_bf16_project_policy_can_be_overridden(monkeypatch) -> None:
    norm_weight = torch.empty((16,), dtype=torch.bfloat16)
    fn_bf16 = torch.empty((24, 64), dtype=torch.bfloat16)

    monkeypatch.setenv("B12X_MHC_PREFILL_BF16_MMA", "0")
    assert not residual_impl._use_mhc_prefill_bf16_project(
        norm_weight=norm_weight,
        policy_m=4096,
        fn_bf16=fn_bf16,
    )

    monkeypatch.setenv("B12X_MHC_PREFILL_BF16_MMA", "1")
    monkeypatch.setenv("B12X_MHC_PREFILL_BF16_MIN_TOKENS", "256")
    assert residual_impl._use_mhc_prefill_bf16_project(
        norm_weight=norm_weight,
        policy_m=256,
        fn_bf16=fn_bf16,
    )


def test_mhc_prefill_tf32_project_policy_defaults_to_bf16_regime(monkeypatch) -> None:
    monkeypatch.delenv("B12X_MHC_PREFILL_TF32_MMA", raising=False)
    monkeypatch.delenv("B12X_MHC_PREFILL_TF32_MIN_TOKENS", raising=False)
    monkeypatch.delenv("B12X_MHC_PREFILL_BF16_MMA", raising=False)
    monkeypatch.delenv("B12X_MHC_PREFILL_BF16_MIN_TOKENS", raising=False)
    norm_weight = torch.empty((16,), dtype=torch.bfloat16)

    assert not residual_impl._use_mhc_prefill_tf32_project(
        norm_weight=norm_weight,
        policy_m=352,
    )
    assert residual_impl._use_mhc_prefill_tf32_project(
        norm_weight=norm_weight,
        policy_m=384,
    )
    assert not residual_impl._use_mhc_prefill_tf32_project(
        norm_weight=None,
        policy_m=4096,
    )


def test_mhc_prefill_tf32_project_policy_can_be_overridden(monkeypatch) -> None:
    norm_weight = torch.empty((16,), dtype=torch.bfloat16)

    monkeypatch.setenv("B12X_MHC_PREFILL_TF32_MMA", "0")
    assert not residual_impl._use_mhc_prefill_tf32_project(
        norm_weight=norm_weight,
        policy_m=4096,
    )

    monkeypatch.delenv("B12X_MHC_PREFILL_TF32_MMA", raising=False)
    monkeypatch.setenv("B12X_MHC_PREFILL_BF16_MMA", "0")
    assert not residual_impl._use_mhc_prefill_tf32_project(
        norm_weight=norm_weight,
        policy_m=4096,
    )

    monkeypatch.setenv("B12X_MHC_PREFILL_TF32_MMA", "1")
    assert residual_impl._use_mhc_prefill_tf32_project(
        norm_weight=norm_weight,
        policy_m=4096,
    )

    monkeypatch.delenv("B12X_MHC_PREFILL_TF32_MMA", raising=False)
    monkeypatch.setenv("B12X_MHC_PREFILL_BF16_MMA", "1")
    monkeypatch.setenv("B12X_MHC_PREFILL_BF16_MIN_TOKENS", "256")
    assert residual_impl._use_mhc_prefill_tf32_project(
        norm_weight=norm_weight,
        policy_m=256,
    )

    monkeypatch.setenv("B12X_MHC_PREFILL_TF32_MIN_TOKENS", "512")
    assert not residual_impl._use_mhc_prefill_tf32_project(
        norm_weight=norm_weight,
        policy_m=384,
    )
    assert residual_impl._use_mhc_prefill_tf32_project(
        norm_weight=norm_weight,
        policy_m=512,
    )


def test_mhc_pre_binding_supplies_bound_outputs(monkeypatch) -> None:
    plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device="cpu",
            max_tokens=4,
            hidden_size=16,
            split_k=MHC_DEFAULT_SPLIT_K,
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    y_storage = torch.empty((4, 16), dtype=torch.bfloat16)
    post_storage = torch.empty((4, 4), dtype=torch.float32)
    comb_storage = torch.empty((4, 4, 4), dtype=torch.float32)
    residual_storage = torch.empty((4, 4, 16), dtype=torch.bfloat16)
    binding = plan.bind(
        scratch=scratch,
        y=y_storage,
        post=post_storage,
        comb=comb_storage,
        out=residual_storage,
    )
    residual = torch.empty((0, 16), dtype=torch.bfloat16)
    fn = torch.empty((24, 16), dtype=torch.float32)
    hc_scale = torch.empty((3,), dtype=torch.float32)
    hc_base = torch.empty((24,), dtype=torch.float32)

    def fake_validate_pre_inputs(*args):
        return 0, 16, MHC_DEFAULT_SPLIT_K * MHC_DEFAULT_BLOCK_K

    monkeypatch.setattr(residual_impl, "_validate_pre_inputs", fake_validate_pre_inputs)

    residual_out, post, comb, y = residual_impl.b12x_mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=2,
        binding=binding,
    )

    assert residual_out.shape == (0, 4, 16)
    assert y.shape == (0, 16)
    assert post.shape == (0, 4)
    assert comb.shape == (0, 4, 4)
    assert y.untyped_storage().data_ptr() == y_storage.untyped_storage().data_ptr()
    assert post.untyped_storage().data_ptr() == post_storage.untyped_storage().data_ptr()
    assert comb.untyped_storage().data_ptr() == comb_storage.untyped_storage().data_ptr()
    assert residual_out.untyped_storage().data_ptr() == residual_storage.untyped_storage().data_ptr()


def test_mhc_post_pre_binding_supplies_bound_outputs(monkeypatch) -> None:
    plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device="cpu",
            max_tokens=4,
            hidden_size=16,
            split_k=MHC_DEFAULT_SPLIT_K,
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    residual_storage = torch.empty((4, 4, 16), dtype=torch.bfloat16)
    y_storage = torch.empty((4, 16), dtype=torch.bfloat16)
    post_storage = torch.empty((4, 4), dtype=torch.float32)
    comb_storage = torch.empty((4, 4, 4), dtype=torch.float32)
    binding = plan.bind(
        scratch=scratch,
        y=y_storage,
        post=post_storage,
        comb=comb_storage,
        out=residual_storage,
    )
    residual = torch.empty((0, 4, 16), dtype=torch.bfloat16)
    x = torch.empty((0, 16), dtype=torch.bfloat16)
    prev_post = torch.empty((0, 4, 1), dtype=torch.float32)
    prev_comb = torch.empty((0, 4, 4), dtype=torch.float32)
    fn = torch.empty((24, 64), dtype=torch.float32)
    hc_scale = torch.empty((3,), dtype=torch.float32)
    hc_base = torch.empty((24,), dtype=torch.float32)

    def fake_validate_post_pre_inputs(*args):
        return 0, 16, MHC_DEFAULT_SPLIT_K * MHC_DEFAULT_BLOCK_K

    monkeypatch.setattr(
        residual_impl,
        "_validate_post_pre_inputs",
        fake_validate_post_pre_inputs,
    )

    residual_cur, post, comb, y = residual_impl.b12x_mhc_post_pre(
        x,
        residual,
        prev_post,
        prev_comb,
        fn,
        hc_scale,
        hc_base,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=2,
        binding=binding,
    )

    assert residual_cur.shape == (0, 4, 16)
    assert post.shape == (0, 4)
    assert comb.shape == (0, 4, 4)
    assert y.shape == (0, 16)
    assert residual_cur.untyped_storage().data_ptr() == residual_storage.untyped_storage().data_ptr()
    assert post.untyped_storage().data_ptr() == post_storage.untyped_storage().data_ptr()
    assert comb.untyped_storage().data_ptr() == comb_storage.untyped_storage().data_ptr()
    assert y.untyped_storage().data_ptr() == y_storage.untyped_storage().data_ptr()


def test_mhc_pre_binding_owns_outputs() -> None:
    plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device="cpu",
            max_tokens=4,
            hidden_size=16,
            split_k=8,
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = plan.bind(scratch=scratch)
    y_out = torch.empty((0, 16), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="binding owns scratch and output buffers"):
        residual_impl.b12x_mhc_pre(
            torch.empty((0, 16), dtype=torch.bfloat16),
            torch.empty((24, 16), dtype=torch.float32),
            torch.empty((3,), dtype=torch.float32),
            torch.empty((24,), dtype=torch.float32),
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=2,
            binding=binding,
            y_out=y_out,
        )
