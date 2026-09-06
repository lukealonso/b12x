"""Measured launch-policy providers for norm and sequence fusions."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from contextlib import AbstractContextManager
from b12x.policy.generation.replay import PreparedCandidate, capture_warmed_graph, measure_prepared_candidates


from b12x.policy.components import HYPERCONNECTION, MHC, MTP_FEEDBACK
from b12x.policy.generation.attention_corpus import (
    COMMON_PREFILL_TOKEN_CAPACITIES,
    COMMON_SEQUENCE_CAPACITIES,
)
from b12x.policy.generation.sweep import (
    DiscreteSweepGenerator,
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
)

from .gpu_workers import (
    _l2_flush_fn,
)

_NORM_SEQUENCE_TOKEN_CAPACITIES = (
    *COMMON_SEQUENCE_CAPACITIES,
    512,
    *COMMON_PREFILL_TOKEN_CAPACITIES,
)


def _hyperconnection_cases() -> tuple[SweepCase, ...]:
    return tuple(
        SweepCase.create(
            group_id=f"qwen-flash-next-m{tokens}",
            query={
                "dtype": "bfloat16",
                "max_tokens": tokens,
                "hidden_size": 2_560,
                "streams": 4,
                "lowrank": 320,
            },
        )
        for tokens in _NORM_SEQUENCE_TOKEN_CAPACITIES
    )


def _mtp_feedback_cases() -> tuple[SweepCase, ...]:
    return tuple(
        SweepCase.create(
            group_id=f"qwen-flash-next-m{tokens}",
            query={
                "dtype": "bfloat16",
                "max_tokens": tokens,
                "hidden_size": 2_560,
                "streams": 4,
            },
        )
        for tokens in _NORM_SEQUENCE_TOKEN_CAPACITIES
    )


def _mhc_cases() -> tuple[SweepCase, ...]:
    capacities = tuple(
        sorted(
            {
                *COMMON_SEQUENCE_CAPACITIES,
                384,
                512,
                *COMMON_PREFILL_TOKEN_CAPACITIES,
                2_304,
                3_072,
                3_584,
            }
        )
    )
    return tuple(
        SweepCase.create(
            group_id=f"mhc-h{hidden_size}-m{tokens}",
            query={
                "dtype": "bfloat16",
                "max_tokens": tokens,
                "hidden_size": hidden_size,
                "split_k": split_k,
            },
        )
        for hidden_size, split_k in ((4_096, 64), (7_168, 112))
        for tokens in capacities
    )


def _mhc_config(
    *,
    backend: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stages: int,
    m_warps: int,
    n_warps: int,
    k_splits: int,
) -> dict[str, object]:
    return {
        "backend": backend,
        "projection_tile_m": tile_m,
        "projection_tile_n": tile_n,
        "projection_tile_k": tile_k,
        "projection_num_stages": stages,
        "projection_num_m_warps": m_warps,
        "projection_num_n_warps": n_warps,
        "projection_k_splits": k_splits,
    }


_MHC_NATIVE_CANDIDATE = SweepCandidate.create(
    _mhc_config(
        backend="native",
        tile_m=16,
        tile_n=8,
        tile_k=256,
        stages=1,
        m_warps=1,
        n_warps=1,
        k_splits=1,
    )
)
_MHC_TF32_CANDIDATES = tuple(
    SweepCandidate.create(
        _mhc_config(
            backend="tf32_tma",
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            stages=stages,
            m_warps=m_warps,
            n_warps=n_warps,
            k_splits=k_splits,
        )
    )
    for tile_m, tile_n, tile_k, stages, m_warps, n_warps, k_splits in (
        (16, 8, 256, 1, 1, 1, 1),
        (32, 8, 256, 1, 2, 1, 1),
        (64, 24, 64, 3, 4, 1, 8),
        (64, 24, 64, 2, 4, 1, 8),
        (128, 24, 64, 2, 8, 1, 4),
        (192, 24, 64, 2, 12, 1, 8),
    )
)


class _GpuSession(AbstractContextManager):
    def __init__(self, context) -> None:
        self._context = context

    def __enter__(self):
        return self

    def __exit__(self, *_exc: object) -> None:
        import torch

        gc.collect()
        torch.cuda.synchronize(self._context.device_ordinal)
        torch.cuda.empty_cache()
        return None


class _HyperConnectionSession(_GpuSession):
    _CANDIDATES = tuple(
        SweepCandidate.create(
            {
                "backend": "cutedsl",
                "reduction_block_h": 4_096,
                "pointwise_block": pointwise_block,
                "reduction_num_warps": num_warps,
            }
        )
        for pointwise_block in (128, 256, 512)
        for num_warps in (4, 8)
    )

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        del case
        return self._CANDIDATES

    def measure(self, case, candidates):
        import torch
        from dataclasses import replace
        from b12x.norm.hyperconnection._policy import HyperConnectionConfig
        from b12x.policy import PolicyContext, PolicyMode
        from benchmarks.benchmark_hyperconnection import Profile, _make_case, build_plan_binding, _validate_outputs

        device = torch.device("cuda", self._context.device_ordinal)
        settings = self._context.settings
        profile = Profile(tokens=int(case.query["max_tokens"]))
        base_policy = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)
        fixture = _make_case(profile, seed=settings.seed + int(case.case_id[-8:], 16),
                             device=device, policy=base_policy)
        expected = fixture.reference("full_chain")
        prepared = []
        for candidate in candidates:
            try:
                config = HyperConnectionConfig.from_profile(candidate.config)
                plan, binding = build_plan_binding(device=device, tokens=profile.tokens,
                    policy=base_policy.with_override(HYPERCONNECTION, config))
                active = replace(fixture, plan=plan, binding=binding)
                def run(active=active):
                    return active.launch("full_chain")
                for _ in range(settings.warmup):
                    run()
                torch.cuda.synchronize(device)
                graph, outputs = capture_warmed_graph(run, device=device)
                addresses = tuple(t.data_ptr() for t in outputs)
                for _ in range(3):
                    for output in outputs:
                        output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    _validate_outputs(operator="full_chain", actual=outputs, expected=expected)
                stable = addresses == tuple(t.data_ptr() for t in outputs)
                prepared.append(PreparedCandidate(candidate=candidate, graph=graph, correct=stable,
                    owners=(active, outputs), metrics={"operator": "full_chain", "stable_addresses": stable,
                        "frozen_resolution_capture": True, "poison_replays": 3}))
            except Exception as exc:
                prepared.append(SweepMeasurement(candidate=candidate, latency_us=None, correct=False,
                                                 error=f"{type(exc).__name__}: {exc}"))
        return measure_prepared_candidates(prepared, settings=settings, device=device,
                                           flush=_l2_flush_fn(device, enabled=settings.cold_l2))


class _MtpFeedbackSession(_GpuSession):
    _CANDIDATES = tuple(
        SweepCandidate.create(
            {
                "backend": "cutedsl",
                "norm_block_h": 4_096,
                "norm_block_s": 4,
                "norm_num_warps": num_warps,
            }
        )
        for num_warps in (4, 8)
    )

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        del case
        return self._CANDIDATES

    def measure(self, case, candidates):
        import torch
        from b12x.sequence import mtp_feedback as mtp
        from b12x.sequence.mtp_feedback._policy import MtpFeedbackConfig
        from b12x.policy import PolicyContext, PolicyMode
        from benchmarks.benchmark_mtp_feedback import Profile, _make_binding, _comparison_metrics, _require_correct

        device = torch.device("cuda", self._context.device_ordinal)
        settings = self._context.settings
        tokens = int(case.query["max_tokens"])
        profile = Profile(name=f"profile-m{tokens}", phase="mixed", tokens=tokens)
        base_policy = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)
        fixture = _make_binding(profile, seed=settings.seed + int(case.case_id[-8:], 16), device=device,
                                capacity_tokens=tokens, policy=base_policy)
        inputs = {name: getattr(fixture, name) for name in ("token_embedding", "multi_state", "token_norm_weight",
                   "state_norm_weight", "embedding_fc_weight", "hidden_fc_weight")}
        reference = mtp.reference.feedback(**inputs, eps=1e-6)
        if not bool(torch.isfinite(reference).all()) or not bool(torch.count_nonzero(reference)):
            raise ValueError("MTP reference must be finite and nonzero")
        prepared = []
        for candidate in candidates:
            try:
                config = MtpFeedbackConfig.from_profile(candidate.config)
                plan = mtp.plan(fixture.plan.caps, policy=base_policy.with_override(MTP_FEEDBACK, config))
                spec, = plan.scratch_specs()
                scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
                output = torch.empty_like(fixture.output)
                binding = mtp.bind(plan, scratch=scratch, output=output, tokens=tokens, **inputs)
                def run(binding=binding):
                    return mtp.run(binding, eps=1e-6)
                for _ in range(settings.warmup):
                    run()
                torch.cuda.synchronize(device)
                graph, _ = capture_warmed_graph(run, device=device)
                pointers = (output.data_ptr(), scratch.data_ptr())
                for _ in range(3):
                    output.fill_(float("nan"))
                    scratch.fill_(0xFF)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    metrics = _comparison_metrics(output, reference)
                    _require_correct("MTP production graph", metrics)
                stable = pointers == (output.data_ptr(), scratch.data_ptr())
                prepared.append(PreparedCandidate(candidate=candidate, graph=graph, correct=stable,
                    owners=(binding, output, scratch), aggregation="median",
                    metrics={"cosine": metrics["cosine"], "stable_addresses": stable,
                             "frozen_resolution_capture": True, "poison_replays": 3}))
            except Exception as exc:
                prepared.append(SweepMeasurement(candidate=candidate, latency_us=None, correct=False,
                                                 error=f"{type(exc).__name__}: {exc}"))
        return measure_prepared_candidates(prepared, settings=settings, device=device,
                                           flush=_l2_flush_fn(device, enabled=settings.cold_l2))


class _MhcSession(_GpuSession):
    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        from b12x.norm.mhc._policy import MHC_POLICY, MhcConfig, MhcQuery

        query = MhcQuery(**case.query.to_dict())
        candidates = (
            (_MHC_NATIVE_CANDIDATE,)
            if query.max_tokens < 384
            else (_MHC_NATIVE_CANDIDATE, *_MHC_TF32_CANDIDATES)
        )
        valid = []
        for candidate in candidates:
            config = MhcConfig.from_profile(candidate.config)
            try:
                MHC_POLICY.validate_config(query, config, self._context.device)
            except ValueError:
                continue
            valid.append(candidate)
        return tuple(valid)

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch
        import torch.nn.functional as torch_functional

        from b12x.norm import mhc
        from b12x.norm.mhc._policy import MhcConfig
        from b12x.policy import PolicyContext, PolicyMode
        from benchmarks.benchmark_residual import (
            _make_inputs,
            _mhc_pre_reference,
            _post_pre_reference,
        )

        device = torch.device("cuda", self._context.device_ordinal)
        settings = self._context.settings
        tokens = int(case.query["max_tokens"])
        hidden_size = int(case.query["hidden_size"])
        split_k = int(case.query["split_k"])
        residual, x, fn, scale, bias = _make_inputs(
            tokens=tokens,
            hidden_size=hidden_size,
            seed=settings.seed,
            device=device,
        )
        _, prev_post, prev_comb = _mhc_pre_reference(
            residual,
            fn,
            scale,
            bias,
            rms_eps=1.0e-6,
            hc_eps=1.0e-6,
            sinkhorn_iters=20,
        )
        prev_post = prev_post.contiguous()
        prev_comb = prev_comb.contiguous()
        generator = torch.Generator(device="cpu").manual_seed(settings.seed + 17)
        norm_weight = (
            torch.randn(
                (hidden_size,),
                generator=generator,
                dtype=torch.float32,
            )
            .to(device=device, dtype=torch.bfloat16)
            .contiguous()
        )
        expected = _post_pre_reference(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            scale,
            bias,
            rms_eps=1.0e-6,
            hc_eps=1.0e-6,
            sinkhorn_iters=20,
            norm_weight=norm_weight,
            norm_eps=1.0e-6,
        )
        base_policy = PolicyContext.for_device(
            device,
            mode=PolicyMode.HEURISTIC_ONLY,
        )
        flush = _l2_flush_fn(device, enabled=settings.cold_l2)
        prepared: dict[str, PreparedCandidate] = {}
        failures: dict[str, SweepMeasurement] = {}
        for candidate in candidates:
            try:
                config = MhcConfig.from_profile(candidate.config)
                policy = base_policy.with_override(MHC, config)
                plan = mhc.plan(
                    mhc.Caps(
                        device=device,
                        max_tokens=tokens,
                        hidden_size=hidden_size,
                        split_k=split_k,
                    ),
                    policy=policy,
                )
                scratch = tuple(
                    torch.empty(shape, dtype=dtype, device=device)
                    for shape, dtype in plan.shapes_and_dtypes()
                )
                output = torch.empty(
                    (tokens, 4, hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                )
                y = torch.empty(
                    (tokens, hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                )
                post = torch.empty(
                    (tokens, 4),
                    dtype=torch.float32,
                    device=device,
                )
                comb = torch.empty(
                    (tokens, 4, 4),
                    dtype=torch.float32,
                    device=device,
                )
                binding = mhc.bind(
                    plan,
                    scratch=scratch,
                    tokens=tokens,
                    y=y,
                    post=post,
                    comb=comb,
                    out=output,
                )

                def run() -> None:
                    mhc.run_post_pre(
                        x,
                        residual,
                        prev_post,
                        prev_comb,
                        fn,
                        scale,
                        bias,
                        rms_eps=1.0e-6,
                        hc_eps=1.0e-6,
                        sinkhorn_iters=20,
                        norm_weight=norm_weight,
                        norm_eps=1.0e-6,
                        binding=binding,
                    )

                for _ in range(settings.warmup):
                    run()
                torch.cuda.synchronize(device)
                graph, _ = capture_warmed_graph(run, device=device)
                for actual in (output, y, post, comb):
                    actual.fill_(float("nan"))
                allocated_before = torch.cuda.memory_allocated(device)
                graph.replay()
                torch.cuda.synchronize(device)
                allocated_after = torch.cuda.memory_allocated(device)
                cosines = tuple(
                    float(
                        torch_functional.cosine_similarity(
                            actual.float().reshape(1, -1),
                            reference.float().reshape(1, -1),
                        ).item()
                    )
                    for actual, reference in zip(
                        (output, y, post, comb),
                        expected,
                        strict=True,
                    )
                )
                finite = all(
                    bool(torch.isfinite(actual).all().item())
                    for actual in (output, y, post, comb)
                )
                nonzero = all(
                    bool(torch.count_nonzero(actual).item())
                    for actual in (output, y, post, comb)
                )
                prepared[candidate.candidate_id] = PreparedCandidate(
                    candidate=candidate,
                    graph=graph,
                    owners=(scratch, output, y, post, comb, binding),
                    correct=(
                        finite
                        and nonzero
                        and min(cosines) >= settings.minimum_cosine
                        and allocated_after <= allocated_before
                    ),
                    metrics={
                        "output_cosine": cosines[0],
                        "y_cosine": cosines[1],
                        "post_cosine": cosines[2],
                        "comb_cosine": cosines[3],
                        "finite": finite,
                        "nonzero": nonzero,
                        "replay_allocation_bytes": (
                            allocated_after - allocated_before
                        ),
                    },
                )
            except Exception as exc:  # noqa: BLE001 - failed configs survive
                failures[candidate.candidate_id] = SweepMeasurement(
                    candidate=candidate,
                    latency_us=None,
                    correct=False,
                    error=f"{type(exc).__name__}: {exc}",
                )

        return measure_prepared_candidates(
            (failures[c.candidate_id] if c.candidate_id in failures else prepared[c.candidate_id] for c in candidates),
            settings=settings, device=device, flush=flush)


class _OneCaseFactory:
    def __init__(self, session_type) -> None:
        self._session_type = session_type

    def __call__(self, group_id, cases, context):
        del group_id
        if len(cases) != 1:
            raise ValueError("norm/sequence allocation groups contain one case")
        return self._session_type(context)


class HyperConnectionGenerator(DiscreteSweepGenerator):
    """Race production HyperConnection launch geometry."""

    @staticmethod
    def validate_region_decision(inputs, device, decision):
        from b12x.norm.hyperconnection._policy import TUNING_PROBLEM

        query = TUNING_PROBLEM.query_from_inputs(inputs)
        if ((query.dtype, query.hidden_size, query.streams, query.lowrank) != ("bfloat16", 2560, 4, 320)
                or query.max_tokens <= 0):
            raise ValueError("HyperConnection fixtures require positive capacity and BF16 geometry H=2560, streams=4, lowrank=320")
        TUNING_PROBLEM.lower(query, device, decision)

    @staticmethod
    def cases_for_tuning_queries(queries):
        from b12x.policy.problem import stable_identity

        for query in queries:
            HyperConnectionGenerator.validate_region_decision(query, None, _HyperConnectionSession._CANDIDATES[0].config)
            yield SweepCase.create(group_id=f"shape-{stable_identity(query)[:24]}", query=query)

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=HYPERCONNECTION,
            query_schema_version=1,
            config_schema_version=1,
            candidate_contract_version=2,
            query_fields=(
                "dtype",
                "max_tokens",
                "hidden_size",
                "streams",
                "lowrank",
            ),
            range_fields=frozenset({"max_tokens"}),
            cases=_hyperconnection_cases() if cases is None else cases,
            benchmark_factory=_OneCaseFactory(_HyperConnectionSession),
            coverage={
                "token_capacities": list(_NORM_SEQUENCE_TOKEN_CAPACITIES),
            },
            nearest_range_bounds={"max_tokens": (1, 8_192)},
        )


class MtpFeedbackGenerator(DiscreteSweepGenerator):
    """Race production MTP feedback normalization launch geometry."""

    @staticmethod
    def validate_region_decision(inputs, device, decision):
        from b12x.sequence.mtp_feedback._policy import TUNING_PROBLEM

        query = TUNING_PROBLEM.query_from_inputs(inputs)
        if ((query.dtype, query.hidden_size, query.streams) != ("bfloat16", 2560, 4)
                or query.max_tokens <= 0):
            raise ValueError("MTP feedback fixtures require positive capacity and BF16 geometry H=2560, streams=4")
        TUNING_PROBLEM.lower(query, device, decision)

    @staticmethod
    def cases_for_tuning_queries(queries):
        from b12x.policy.problem import stable_identity

        for query in queries:
            MtpFeedbackGenerator.validate_region_decision(query, None, _MtpFeedbackSession._CANDIDATES[0].config)
            yield SweepCase.create(group_id=f"shape-{stable_identity(query)[:24]}", query=query)

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=MTP_FEEDBACK,
            query_schema_version=1,
            config_schema_version=1,
            candidate_contract_version=2,
            query_fields=("dtype", "max_tokens", "hidden_size", "streams"),
            range_fields=frozenset({"max_tokens"}),
            cases=_mtp_feedback_cases() if cases is None else cases,
            benchmark_factory=_OneCaseFactory(_MtpFeedbackSession),
            coverage={
                "token_capacities": list(_NORM_SEQUENCE_TOKEN_CAPACITIES),
            },
            nearest_range_bounds={"max_tokens": (1, 8_192)},
        )


class MhcGenerator(DiscreteSweepGenerator):
    """Race production mHC post/pre backends and TF32 projection geometry."""

    @staticmethod
    def validate_region_decision(inputs, device, decision):
        from b12x.norm.mhc._policy import TUNING_PROBLEM

        query = TUNING_PROBLEM.query_from_inputs(inputs)
        if ((query.hidden_size, query.split_k) not in ((4096, 64), (7168, 112))
                or query.dtype != "bfloat16" or query.max_tokens <= 0):
            raise ValueError("mHC fixtures require positive BF16 capacity and supported hidden-width/split-K geometry")
        config = TUNING_PROBLEM.lower(query, device, decision)
        if query.max_tokens < 384 and config.backend != "native":
            raise ValueError("mHC production candidates use the native backend below 384 tokens")

    @staticmethod
    def cases_for_tuning_queries(queries):
        from b12x.policy.problem import stable_identity

        for query in queries:
            MhcGenerator.validate_region_decision(query, None, _MHC_NATIVE_CANDIDATE.config)
            yield SweepCase.create(group_id=f"shape-{stable_identity(query)[:24]}", query=query)

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=MHC,
            query_schema_version=1,
            config_schema_version=2,
            query_fields=("dtype", "max_tokens", "hidden_size", "split_k"),
            range_fields=frozenset({"max_tokens"}),
            cases=_mhc_cases() if cases is None else cases,
            benchmark_factory=_OneCaseFactory(_MhcSession),
            coverage={
                "hidden_sizes": [4_096, 7_168],
                "prefill_capacities": list(COMMON_PREFILL_TOKEN_CAPACITIES),
                "medium_prefill_anchors": [2_304, 3_072, 3_584],
            },
            candidate_contract_version=2,
            nearest_range_bounds={"max_tokens": (1, 8_192)},
        )

    def reviewed_queries(self):
        from b12x.norm.mhc._policy import MhcQuery

        return tuple(MhcQuery(**case.query.to_dict()) for case in self._cases)


__all__ = ["HyperConnectionGenerator", "MhcGenerator", "MtpFeedbackGenerator"]
