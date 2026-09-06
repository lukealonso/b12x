"""Measured providers for the Qwen PLE component family."""

from __future__ import annotations

from collections.abc import Mapping

from b12x.policy.generation.contracts import GenerationContext
from b12x.policy.generation.measured import (
    GpuProbeMeasurement,
    MeasuredPolicyGenerator,
)


def _graph_latency(timing: Mapping[str, object]) -> float:
    kernel = timing.get("kernel")
    if not isinstance(kernel, Mapping):
        raise TypeError("PLE graph timing must contain a kernel summary")
    return float(kernel["median"])


def _sample_count(context: GenerationContext) -> int:
    return max(1, context.settings.groups * context.settings.repetitions)


class _PleLayerProbe:
    @property
    def case_count(self) -> int:
        from benchmarks.benchmark_ple import LAYER_PROFILES

        return len(LAYER_PROFILES)

    @property
    def case_ids(self) -> tuple[str, ...]:
        from benchmarks.benchmark_ple import LAYER_PROFILES

        return tuple(profile.name for profile in LAYER_PROFILES)

    @property
    def description(self) -> str:
        return "production PLE layer graph qualification for every execution mode"

    def __call__(
        self,
        context: GenerationContext,
    ) -> tuple[GpuProbeMeasurement, ...]:
        import torch

        from benchmarks.benchmark_ple import (
            LAYER_PROFILES,
            _address_tensors,
            _time_case,
            _validate_layer,
            build_layer_case,
        )

        from .gpu_workers import _l2_flush_fn

        device = torch.device("cuda", context.device_ordinal)
        flush = _l2_flush_fn(device, enabled=context.settings.cold_l2)
        measurements = []
        for index, profile in enumerate(LAYER_PROFILES):
            case = build_layer_case(
                profile,
                device=device,
                seed=context.settings.seed + 10_007 * index,
            )
            timing, correctness, graph_contract = _time_case(
                launch=case.launch,
                validate=lambda active=case: _validate_layer(active),
                address_tensors=_address_tensors(case.binding, stateful=True),
                prepare=case.restore,
                mode="graph",
                warmup=context.settings.warmup,
                samples=_sample_count(context),
                l2_flush=flush,
                device=device,
            )
            measurements.append(
                GpuProbeMeasurement(
                    label=profile.name,
                    latency_us=_graph_latency(timing),
                    correct=(
                        correctness.get("status") == "passed"
                        and graph_contract is not None
                    ),
                    metrics={"mode": profile.mode, "tokens": profile.tokens},
                )
            )
        return tuple(measurements)


class _PleHashProbe:
    @property
    def case_count(self) -> int:
        from benchmarks.benchmark_ple import PACKED_PROFILES

        return len(PACKED_PROFILES)

    @property
    def case_ids(self) -> tuple[str, ...]:
        from benchmarks.benchmark_ple import PACKED_PROFILES

        return tuple(profile.name for profile in PACKED_PROFILES)

    @property
    def description(self) -> str:
        return "production PLE hash graph qualification over packed token phases"

    def __call__(
        self,
        context: GenerationContext,
    ) -> tuple[GpuProbeMeasurement, ...]:
        import torch

        from benchmarks.benchmark_ple import (
            PACKED_PROFILES,
            _address_tensors,
            _time_case,
            _validate_hash,
            build_hash_case,
        )

        from .gpu_workers import _l2_flush_fn

        device = torch.device("cuda", context.device_ordinal)
        flush = _l2_flush_fn(device, enabled=context.settings.cold_l2)
        measurements = []
        for index, profile in enumerate(PACKED_PROFILES):
            case = build_hash_case(
                profile,
                device=device,
                seed=context.settings.seed + 10_007 * index,
            )
            timing, correctness, graph_contract = _time_case(
                launch=case.launch,
                validate=lambda active=case: _validate_hash(active),
                address_tensors=_address_tensors(case.binding, stateful=False),
                prepare=None,
                mode="graph",
                warmup=context.settings.warmup,
                samples=_sample_count(context),
                l2_flush=flush,
                device=device,
            )
            measurements.append(
                GpuProbeMeasurement(
                    label=profile.name,
                    latency_us=_graph_latency(timing),
                    correct=(
                        correctness.get("status") == "passed"
                        and graph_contract is not None
                    ),
                    metrics={"phase": profile.phase, "tokens": profile.tokens},
                )
            )
        return tuple(measurements)


class _PleEmbeddingProbe:
    _PROFILES = ("decode-t1-bs1", "prefill-t512-bs4")
    _QUANT_MODES = ("bf16", "fp8_e4m3_per_tensor", "nvfp4_group16")
    _TABLE_MEMORIES = ("device", "mapped_host")

    @property
    def case_count(self) -> int:
        return len(self._PROFILES) * len(self._QUANT_MODES) * len(
            self._TABLE_MEMORIES
        )

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{profile}-{quant_mode}-{table_memory}"
            for profile in self._PROFILES
            for quant_mode in self._QUANT_MODES
            for table_memory in self._TABLE_MEMORIES
        )

    @property
    def description(self) -> str:
        return "production PLE embedding graph qualification by quant and storage"

    def __call__(
        self,
        context: GenerationContext,
    ) -> tuple[GpuProbeMeasurement, ...]:
        import torch

        from benchmarks.benchmark_ple import (
            EMBEDDING_GEOMETRIES,
            PACKED_PROFILES,
            _address_tensors,
            _time_case,
            _validate_embedding,
            build_embedding_case,
        )

        from .gpu_workers import _l2_flush_fn

        device = torch.device("cuda", context.device_ordinal)
        flush = _l2_flush_fn(device, enabled=context.settings.cold_l2)
        by_name = {profile.name: profile for profile in PACKED_PROFILES}
        geometry = EMBEDDING_GEOMETRIES["storage-scaled"]
        measurements = []
        case_index = 0
        for profile_name in self._PROFILES:
            for quant_mode in self._QUANT_MODES:
                for table_memory in self._TABLE_MEMORIES:
                    profile = by_name[profile_name]
                    case = build_embedding_case(
                        profile,
                        device=device,
                        seed=context.settings.seed + 10_007 * case_index,
                        quant_mode=quant_mode,
                        geometry=geometry,
                        table_memory=table_memory,
                    )
                    try:
                        timing, correctness, graph_contract = _time_case(
                            launch=case.launch,
                            validate=(
                                lambda active=case: _validate_embedding(active)
                            ),
                            address_tensors=_address_tensors(
                                case.binding,
                                stateful=False,
                            ),
                            prepare=None,
                            mode="graph",
                            warmup=context.settings.warmup,
                            samples=_sample_count(context),
                            l2_flush=flush,
                            device=device,
                        )
                        label = f"{profile.name}-{quant_mode}-{table_memory}"
                        measurements.append(
                            GpuProbeMeasurement(
                                label=label,
                                latency_us=_graph_latency(timing),
                                correct=(
                                    correctness.get("status") == "passed"
                                    and graph_contract is not None
                                ),
                                metrics={
                                    "quant_mode": quant_mode,
                                    "table_memory": table_memory,
                                    "tokens": profile.tokens,
                                },
                            )
                        )
                    finally:
                        case.close()
                    case_index += 1
        return tuple(measurements)


class PleGenerator(MeasuredPolicyGenerator):
    """Generate a measured PLE layer policy."""

    def __init__(self) -> None:
        from b12x.sequence.ple._policy import PLE_POLICY, PleQuery

        queries = tuple(
            PleQuery(
                mode=mode,
                dtype="bfloat16",
                max_tokens=tokens,
                max_seqs=16,
                max_speculative_tokens=3,
                streams=4,
                hidden_size=2_560,
                kernel_size=4,
                dilation=1,
            )
            for mode in ("decode", "prefill", "mixed")
            for tokens in (4, 64)
        )
        super().__init__(
            policy=PLE_POLICY,
            queries=queries,
            encode_config=lambda config: config.to_dict(),
            probe=_PleLayerProbe(),
        )


class PleHashGenerator(MeasuredPolicyGenerator):
    """Generate a measured PLE hash policy."""

    def __init__(self) -> None:
        from b12x.sequence.ple_hash._policy import PLE_HASH_POLICY, PleHashQuery

        queries = tuple(
            PleHashQuery(
                max_tokens=tokens,
                max_seqs=16,
                vocab_size=152_064,
                max_order=order,
                heads_per_order=4,
                base_table_size=1_009,
            )
            for order in (3, 5)
            for tokens in (4, 64)
        )
        super().__init__(
            policy=PLE_HASH_POLICY,
            queries=queries,
            encode_config=lambda config: config.to_dict(),
            probe=_PleHashProbe(),
        )


class PleEmbeddingGenerator(MeasuredPolicyGenerator):
    """Generate a measured PLE embedding policy."""

    def __init__(self) -> None:
        from b12x.sequence.ple_embedding._policy import (
            PLE_EMBEDDING_POLICY,
            PleEmbeddingQuery,
        )

        queries = tuple(
            PleEmbeddingQuery(
                quant_mode=quant_mode,
                table_memory=table_memory,
                output_dtype="bfloat16",
                max_tokens=64,
                max_seqs=16,
                vocab_size=152_064,
                max_order=5,
                heads_per_order=4,
                base_table_size=1_009,
                embedding_dim=2_560,
                tp_size=tp_size,
            )
            for quant_mode in (
                "bf16",
                "fp8_e4m3_per_tensor",
                "nvfp4_group16",
            )
            for table_memory in ("device", "mapped_host")
            for tp_size in (1, 2, 4, 8, 16)
        )
        super().__init__(
            policy=PLE_EMBEDDING_POLICY,
            queries=queries,
            encode_config=lambda config: config.to_dict(),
            probe=_PleEmbeddingProbe(),
        )


__all__ = ["PleEmbeddingGenerator", "PleGenerator", "PleHashGenerator"]
