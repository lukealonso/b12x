"""Measured launch-policy providers for norm and sequence fusions."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

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
    _median_of_group_medians,
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

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch

        from b12x.norm.hyperconnection._policy import HyperConnectionConfig
        from b12x.policy import PolicyContext, PolicyMode
        from benchmarks.benchmark_hyperconnection import (
            Profile,
            _graph_samples_us,
            _make_case,
        )

        device = torch.device("cuda", self._context.device_ordinal)
        settings = self._context.settings
        profile = Profile(tokens=int(case.query["max_tokens"]))
        base_policy = PolicyContext.for_device(
            device,
            mode=PolicyMode.HEURISTIC_ONLY,
        )
        flush = _l2_flush_fn(device, enabled=settings.cold_l2)
        measurements = []
        for candidate in candidates:
            try:
                config = HyperConnectionConfig.from_profile(candidate.config)
                policy = base_policy.with_override(HYPERCONNECTION, config)
                active = _make_case(
                    profile,
                    seed=settings.seed + int(candidate.candidate_id[-8:], 16),
                    device=device,
                    policy=policy,
                )
                samples, graph_contract, correctness = _graph_samples_us(
                    active,
                    "full_chain",
                    warmup=settings.warmup,
                    samples=settings.groups * settings.repetitions,
                    l2_flush=flush,
                )
                measurements.append(
                    SweepMeasurement(
                        candidate=candidate,
                        latency_us=_median_of_group_medians(
                            tuple(samples),
                            groups=settings.groups,
                            repetitions=settings.repetitions,
                        ),
                        correct=(
                            correctness.get("status") == "passed"
                            and graph_contract.get(
                                "replay_allocation_delta_bytes"
                            )
                            == 0
                        ),
                        metrics={
                            "operator": "full_chain",
                            "replay_allocation_bytes": graph_contract[
                                "replay_allocation_delta_bytes"
                            ],
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001 - failed configs survive
                measurements.append(
                    SweepMeasurement(
                        candidate=candidate,
                        latency_us=None,
                        correct=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return tuple(measurements)


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

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch

        from b12x.policy import PolicyContext, PolicyMode
        from b12x.sequence.mtp_feedback._policy import MtpFeedbackConfig
        from benchmarks.benchmark_mtp_feedback import Profile, _benchmark_profile

        device = torch.device("cuda", self._context.device_ordinal)
        settings = self._context.settings
        tokens = int(case.query["max_tokens"])
        profile = Profile(name=f"profile-m{tokens}", phase="mixed", tokens=tokens)
        base_policy = PolicyContext.for_device(
            device,
            mode=PolicyMode.HEURISTIC_ONLY,
        )
        flush = _l2_flush_fn(device, enabled=settings.cold_l2)
        measurements = []
        for candidate in candidates:
            try:
                config = MtpFeedbackConfig.from_profile(candidate.config)
                policy = base_policy.with_override(MTP_FEEDBACK, config)
                result = _benchmark_profile(
                    profile,
                    seed=settings.seed + int(candidate.candidate_id[-8:], 16),
                    device=device,
                    eps=1.0e-6,
                    warmup=settings.warmup,
                    samples=settings.groups * settings.repetitions,
                    l2_flush=flush,
                    capacity_tokens=tokens,
                    policy=policy,
                )
                timings = result["timings"]
                correctness = result["correctness"]
                storage = result["storage"]
                measurements.append(
                    SweepMeasurement(
                        candidate=candidate,
                        latency_us=float(
                            timings["cuda_graph_replay"]["median_us"]
                        ),
                        correct=bool(
                            correctness["passed"]
                            and storage["graph_replay_allocation_delta_bytes"] == 0
                        ),
                        metrics={
                            "cosine": correctness[
                                "graph_replay_after_output_poison"
                            ]["cosine"],
                            "replay_allocation_bytes": storage[
                                "graph_replay_allocation_delta_bytes"
                            ],
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001 - failed configs survive
                measurements.append(
                    SweepMeasurement(
                        candidate=candidate,
                        latency_us=None,
                        correct=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return tuple(measurements)


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
        prepared: dict[str, _PreparedMhcCandidate] = {}
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
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
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
                prepared[candidate.candidate_id] = _PreparedMhcCandidate(
                    candidate=candidate,
                    graph=graph,
                    retained=(scratch, output, y, post, comb, binding),
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

        active = list(prepared.values())
        if not active:
            return tuple(failures[candidate.candidate_id] for candidate in candidates)
        recorded_events = []
        for group in range(settings.groups):
            offset = group % len(active)
            ordered = active[offset:] + active[:offset]
            if group % 2:
                ordered.reverse()
            for item in ordered:
                for _ in range(settings.repetitions):
                    if flush is not None:
                        flush()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    item.graph.replay()
                    end.record()
                    recorded_events.append((item, start, end))
        torch.cuda.synchronize(device)
        for item, start, end in recorded_events:
            item.samples.append(float(start.elapsed_time(end)) * 1_000.0)

        measurements = []
        for candidate in candidates:
            failed = failures.get(candidate.candidate_id)
            if failed is not None:
                measurements.append(failed)
                continue
            item = prepared[candidate.candidate_id]
            measurements.append(
                SweepMeasurement(
                    candidate=candidate,
                    latency_us=_median_of_group_medians(
                        tuple(item.samples),
                        groups=settings.groups,
                        repetitions=settings.repetitions,
                    ),
                    correct=item.correct,
                    metrics=item.metrics,
                )
            )
        return tuple(measurements)


@dataclass
class _PreparedMhcCandidate:
    candidate: SweepCandidate
    graph: object
    retained: tuple[object, ...]
    correct: bool
    metrics: dict[str, object]
    samples: list[float] = field(default_factory=list)


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

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=HYPERCONNECTION,
            query_schema_version=1,
            config_schema_version=1,
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

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=MTP_FEEDBACK,
            query_schema_version=1,
            config_schema_version=1,
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
