"""Measured providers for planned GEMM composites."""

from __future__ import annotations

import gc
import statistics
from collections.abc import Sequence
from contextlib import AbstractContextManager

from b12x.policy.components import BF16_VOCAB_PROJECTION, BLOCK_FP8_LINEAR
from b12x.policy.generation.contracts import GenerationContext
from b12x.policy.generation.measured import (
    GpuProbeMeasurement,
    MeasuredPolicyGenerator,
)
from b12x.policy.generation.sweep import (
    DiscreteSweepGenerator,
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
)

from .gpu_workers import (
    _cuda_event_samples_us,
    _l2_flush_fn,
    _median_of_group_medians,
)


_BLOCK_FP8_TILES = (
    (16, 64),
    (16, 128),
    (32, 64),
    (32, 128),
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
)

_VOCAB_PROJECTION_GEOMETRIES = (
    ("qwen3.8-flash-next-180b", 2_560, (248_320,)),
    ("qwen3.8-27b", 5_120, (248_320,)),
    ("glm-5.3", 6_144, (154_880,)),
    ("glm-5.3-flash", 4_096, (154_880,)),
    ("glm-5.2", 6_144, (163_840, 163_968)),
)
_VOCAB_PROJECTION_TP_SIZES = (1, 2, 4, 8, 16)


def _bf16_vocab_projection_cases() -> tuple[SweepCase, ...]:
    cases = []
    for model_id, in_features, global_vocab_sizes in _VOCAB_PROJECTION_GEOMETRIES:
        for global_vocab_size in global_vocab_sizes:
            for tp_size in _VOCAB_PROJECTION_TP_SIZES:
                if global_vocab_size % tp_size:
                    continue
                out_features = global_vocab_size // tp_size
                cases.append(
                    SweepCase.create(
                        group_id=(f"{model_id}-v{global_vocab_size}-tp{tp_size}"),
                        query={
                            "dtype": "bfloat16",
                            "max_tokens": 1,
                            "in_features": in_features,
                            "out_features": out_features,
                        },
                        scenario=f"{model_id}-tp{tp_size}",
                        metadata={
                            "model_id": model_id,
                            "global_vocab_size": global_vocab_size,
                            "tp_size": tp_size,
                        },
                    )
                )
    return tuple(cases)


class _Bf16VocabProjectionSession(
    AbstractContextManager["_Bf16VocabProjectionSession"]
):
    def __init__(self, context) -> None:
        self._context = context

    def __enter__(self) -> "_Bf16VocabProjectionSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        import torch

        gc.collect()
        torch.cuda.synchronize(self._context.device_ordinal)
        torch.cuda.empty_cache()
        return None

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        in_features = int(case.query["in_features"])
        direct_block = 1 << (in_features - 1).bit_length()
        configs = [
            {
                "backend": "torch",
                "algorithm": "torch",
                "block_k": 0,
                "num_warps": 0,
            }
        ]
        configs.extend(
            {
                "backend": "triton",
                "algorithm": "row",
                "block_k": direct_block,
                "num_warps": num_warps,
            }
            for num_warps in (1, 2, 4, 8)
        )
        configs.extend(
            {
                "backend": "triton",
                "algorithm": "loop",
                "block_k": block_k,
                "num_warps": num_warps,
            }
            for block_k in (256, 512, 1_024)
            for num_warps in (4, 8)
        )
        return tuple(SweepCandidate.create(config) for config in configs)

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch
        import torch.nn.functional as torch_functional

        from b12x.gemm import bf16_vocab_projection as projection
        from b12x.gemm.bf16_vocab_projection._policy import (
            Bf16VocabProjectionConfig,
        )
        from b12x.policy import PolicyContext, PolicyMode

        settings = self._context.settings
        device = torch.device("cuda", self._context.device_ordinal)
        in_features = int(case.query["in_features"])
        out_features = int(case.query["out_features"])
        generator = torch.Generator(device=device).manual_seed(
            settings.seed + int(case.case_id[-8:], 16)
        )
        with torch.cuda.device(self._context.device_ordinal):
            source = torch.randn(
                (1, in_features),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            weight = torch.randn(
                (out_features, in_features),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.125)
            expected = torch_functional.linear(source, weight)
            flush = _l2_flush_fn(device, enabled=settings.cold_l2)
            base_policy = PolicyContext.for_device(
                device,
                mode=PolicyMode.HEURISTIC_ONLY,
            )
            measurements = []
            for candidate in candidates:
                try:
                    config = Bf16VocabProjectionConfig.from_profile(candidate.config)
                    policy = base_policy.with_override(
                        BF16_VOCAB_PROJECTION,
                        config,
                    )
                    planned = projection.plan(
                        projection.Caps(
                            device=device,
                            max_tokens=1,
                            in_features=in_features,
                            out_features=out_features,
                        ),
                        policy=policy,
                    )
                    binding = projection.bind(
                        planned,
                        source=source,
                        weight=weight,
                    )

                    def run():
                        return projection.run(binding)

                    for _ in range(settings.warmup):
                        run()
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = run()
                    actual.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    cosine = float(
                        torch_functional.cosine_similarity(
                            actual.float(),
                            expected.float(),
                        ).item()
                    )
                    finite = bool(torch.isfinite(actual).all().item())
                    allocated_before = torch.cuda.memory_allocated(device)
                    samples = _cuda_event_samples_us(
                        graph.replay,
                        count=settings.groups * settings.repetitions,
                        device=device,
                        flush=flush,
                    )
                    allocated_after = torch.cuda.memory_allocated(device)
                    latency_us = _median_of_group_medians(
                        samples,
                        groups=settings.groups,
                        repetitions=settings.repetitions,
                    )
                    transferred_bytes = 2 * (
                        out_features * in_features + in_features + out_features
                    )
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=latency_us,
                            correct=(
                                finite
                                and cosine >= settings.minimum_cosine
                                and allocated_after <= allocated_before
                            ),
                            metrics={
                                "cosine": cosine,
                                "finite": finite,
                                "replay_allocation_bytes": (
                                    allocated_after - allocated_before
                                ),
                                "effective_bandwidth_gbps": (
                                    transferred_bytes / latency_us / 1_000.0
                                ),
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


class _Bf16VocabProjectionFactory:
    def __call__(self, group_id, cases, context):
        del group_id
        if len(cases) != 1:
            raise ValueError("vocabulary projection allocation groups contain one case")
        return _Bf16VocabProjectionSession(context)


class Bf16VocabProjectionGenerator(DiscreteSweepGenerator):
    """Race production BF16 vocabulary projection paths over common models."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=BF16_VOCAB_PROJECTION,
            query_schema_version=1,
            config_schema_version=1,
            query_fields=(
                "dtype",
                "max_tokens",
                "in_features",
                "out_features",
            ),
            range_fields=frozenset({"out_features"}),
            cases=(_bf16_vocab_projection_cases() if cases is None else cases),
            benchmark_factory=_Bf16VocabProjectionFactory(),
            coverage={},
            candidate_contract_version=1,
            nearest_range_bounds={"out_features": (1, 248_320)},
        )


def _block_fp8_cases() -> tuple[SweepCase, ...]:
    return tuple(
        SweepCase.create(
            group_id=f"m{tokens}-k{in_features}-n{out_features}",
            query={
                "max_tokens": tokens,
                "in_features": in_features,
                "out_features": out_features,
                "output_dtype": "bfloat16",
            },
        )
        for tokens in (4, 32)
        for in_features, out_features in (
            (2_560, 2_560),
            (2_560, 10_240),
        )
    )


class _BlockFp8Session(AbstractContextManager["_BlockFp8Session"]):
    _CANDIDATES = tuple(
        SweepCandidate.create(
            {"backend": "mxfp8", "tile_m": tile_m, "tile_n": tile_n}
        )
        for tile_m, tile_n in _BLOCK_FP8_TILES
    )

    def __init__(self, context) -> None:
        self._context = context

    def __enter__(self) -> "_BlockFp8Session":
        return self

    def __exit__(self, *_exc: object) -> None:
        import torch

        gc.collect()
        torch.cuda.synchronize(self._context.device_ordinal)
        torch.cuda.empty_cache()
        return None

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        del case
        return self._CANDIDATES

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch
        import torch.nn.functional as torch_functional

        from b12x.gemm import block_fp8_linear as block_fp8
        from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch
        from b12x.gemm.block_fp8_linear._policy import BlockFp8LinearConfig
        from b12x.policy import PolicyContext, PolicyMode

        settings = self._context.settings
        device = torch.device("cuda", self._context.device_ordinal)
        tokens = int(case.query["max_tokens"])
        in_features = int(case.query["in_features"])
        out_features = int(case.query["out_features"])
        generator = torch.Generator(device=device).manual_seed(
            settings.seed + int(case.case_id[-8:], 16)
        )
        with torch.cuda.device(self._context.device_ordinal):
            source = torch.randn(
                (tokens, in_features),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            weight = torch.randn(
                (out_features, in_features),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.125).to(torch.float8_e4m3fn)
            scale = torch.ones(
                (out_features // 128, in_features // 128),
                dtype=torch.float8_e8m0fnu,
                device=device,
            )
            packed = block_fp8.pack_weight(weight, scale)
            source_q = block_fp8.quantize_input(source)
            source_dequantized = dequantize_mxfp8_rows_torch(
                source_q.values,
                source_q.scale_rows,
            )
            weight_dequantized = dequantize_mxfp8_rows_torch(
                packed.weight.values,
                packed.weight.scale_rows,
            )
            expected = source_dequantized @ weight_dequantized.T
            del source_q, source_dequantized, weight_dequantized
            flush = _l2_flush_fn(device, enabled=settings.cold_l2)
            base_policy = PolicyContext.for_device(
                device,
                mode=PolicyMode.HEURISTIC_ONLY,
            )
            measurements = []
            for candidate in candidates:
                try:
                    config = BlockFp8LinearConfig.from_profile(candidate.config)
                    policy = base_policy.with_override(BLOCK_FP8_LINEAR, config)
                    plan = block_fp8.plan(
                        block_fp8.Caps(
                            device=device,
                            max_tokens=tokens,
                            in_features=in_features,
                            out_features=out_features,
                            output_dtype=torch.bfloat16,
                        ),
                        policy=policy,
                    )
                    (scratch_spec,) = plan.scratch_specs()
                    scratch = torch.empty(
                        scratch_spec.shape,
                        dtype=scratch_spec.dtype,
                        device=scratch_spec.device,
                    )
                    output = torch.empty(
                        (tokens, out_features, 1),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    binding = block_fp8.bind(
                        plan,
                        scratch=scratch,
                        source=source,
                        packed_weight=packed,
                        output=output,
                        expected_m=tokens,
                    )

                    def run() -> None:
                        block_fp8.run(binding=binding)

                    for _ in range(settings.warmup):
                        run()
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    binding.output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    actual = binding.output[:, :, 0]
                    cosine = float(
                        torch_functional.cosine_similarity(
                            actual.float().reshape(1, -1),
                            expected.float().reshape(1, -1),
                        ).item()
                    )
                    finite = bool(torch.isfinite(actual).all().item())
                    allocated_before = torch.cuda.memory_allocated(device)
                    samples = _cuda_event_samples_us(
                        graph.replay,
                        count=settings.groups * settings.repetitions,
                        device=device,
                        flush=flush,
                    )
                    allocated_after = torch.cuda.memory_allocated(device)
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=_median_of_group_medians(
                                samples,
                                groups=settings.groups,
                                repetitions=settings.repetitions,
                            ),
                            correct=(
                                finite
                                and cosine >= settings.minimum_cosine
                                and allocated_after <= allocated_before
                            ),
                            metrics={
                                "cosine": cosine,
                                "finite": finite,
                                "replay_allocation_bytes": (
                                    allocated_after - allocated_before
                                ),
                            },
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - failed tiles survive
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=None,
                            correct=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            return tuple(measurements)


class _BlockFp8Factory:
    def __call__(self, group_id, cases, context):
        del group_id
        if len(cases) != 1:
            raise ValueError("block-FP8 allocation groups contain one case")
        return _BlockFp8Session(context)


class BlockFp8LinearGenerator(DiscreteSweepGenerator):
    """Race the production block-FP8 linear MMA tiles."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=BLOCK_FP8_LINEAR,
            query_schema_version=1,
            config_schema_version=2,
            query_fields=(
                "max_tokens",
                "in_features",
                "out_features",
                "output_dtype",
            ),
            range_fields=frozenset(
                {"max_tokens", "in_features", "out_features"}
            ),
            cases=_block_fp8_cases() if cases is None else cases,
            benchmark_factory=_BlockFp8Factory(),
            coverage={},
        )


class _WoProjectionProbe:
    _CASES = ((1, 1), (4, 2), (32, 4), (32, 8))

    @property
    def case_count(self) -> int:
        return len(self._CASES)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(
            f"m{tokens}-tp{tp_size}"
            for tokens, tp_size in self._CASES
        )

    @property
    def description(self) -> str:
        return "production W_o two-GEMM graph qualification over TP slices"

    def __call__(
        self,
        context: GenerationContext,
    ) -> tuple[GpuProbeMeasurement, ...]:
        import torch

        from benchmarks.benchmark_wo_projection import bench_one

        flush = _l2_flush_fn(
            torch.device("cuda", context.device_ordinal),
            enabled=context.settings.cold_l2,
        )
        if flush is None:
            def flush() -> None:
                return None
        measurements = []
        for index, (tokens, tp_size) in enumerate(self._CASES):
            result = bench_one(
                tokens,
                groups=24 // tp_size,
                group_width=512,
                rank=512,
                hidden=2_560,
                warmup=context.settings.warmup,
                iters=context.settings.groups * context.settings.repetitions,
                check=True,
                l2_flush=flush,
                seed=context.settings.seed + 10_007 * index,
                inv_rope=False,
                context_length=16_384,
                nope_dim=448,
                rope_dim=64,
            )
            samples = result.get("b12x")
            if not isinstance(samples, list) or not samples:
                raise RuntimeError("W_o benchmark did not produce b12x samples")
            measurements.append(
                GpuProbeMeasurement(
                    label=f"m{tokens}-tp{tp_size}",
                    latency_us=statistics.median(samples) * 1_000.0,
                    correct=True,
                    metrics={"tokens": tokens, "tp_size": tp_size},
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
        return tuple(measurements)


class WoProjectionGenerator(MeasuredPolicyGenerator):
    """Generate a measured policy for the production W_o composite."""

    def __init__(self) -> None:
        from b12x.gemm.wo_projection._policy import (
            WO_PROJECTION_POLICY,
            WoProjectionQuery,
        )

        queries = tuple(
            WoProjectionQuery(
                dtype="bfloat16",
                max_tokens=tokens,
                groups=24 // tp_size,
                group_width=512,
                rank=512,
                hidden=2_560,
            )
            for tokens in (4, 32)
            for tp_size in (1, 2, 4, 8)
        )
        super().__init__(
            policy=WO_PROJECTION_POLICY,
            queries=queries,
            encode_config=lambda config: config.to_dict(),
            probe=_WoProjectionProbe(),
        )


__all__ = [
    "Bf16VocabProjectionGenerator",
    "BlockFp8LinearGenerator",
    "WoProjectionGenerator",
]
