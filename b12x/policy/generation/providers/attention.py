"""Built-in attention component generators and reviewed corpora."""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager

from b12x.policy.components import (
    COMPRESSED_SPARSE_MLA_ATTENTION,
    GDN_ATTENTION,
    GQA_ATTENTION,
    MLA_ATTENTION,
    QSA_ATTENTION,
)
from b12x.policy.generation.attention_corpus import (
    COMMON_SEQUENCE_CAPACITIES,
    GDN_GEOMETRIES,
    GDN_STATE_INDEX_COLUMNS,
    GQA_GEOMETRIES,
    MLA_GEOMETRIES,
    SPARSE_MLA_GEOMETRIES,
    gdn_cases,
    gqa_cases,
    mla_cases,
    qsa_cases,
    sparse_mla_cases,
)
from b12x.policy.generation.contracts import (
    GenerationContext,
)
from b12x.policy.generation.sweep import (
    DiscreteSweepGenerator,
    SweepBenchmarkFactory,
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
)

from .gpu_workers import (
    GdnBenchmarkFactory,
    GqaBenchmarkFactory,
    MlaBenchmarkFactory,
    SparseMlaBenchmarkFactory,
)


class _MissingAttentionBenchmarkFactory:
    def __init__(self, component_id: str) -> None:
        self._component_id = component_id

    def __call__(self, group_id, cases, context):
        del group_id, cases, context
        raise RuntimeError(
            f"{self._component_id} has a reviewed corpus and reducer, but its "
            "production GPU measurement worker is not registered"
        )


class _AttentionGenerator(DiscreteSweepGenerator):
    def __init__(
        self,
        *,
        component_id: str,
        query_fields: tuple[str, ...],
        range_fields: frozenset[str],
        cases: Sequence[SweepCase],
        corpus_name: str,
        geometry_count: int,
        benchmark_factory: SweepBenchmarkFactory | None,
        query_schema_version: int = 1,
        config_schema_version: int = 1,
        nearest_range_bounds: Mapping[str, tuple[int, int]] | None = None,
    ) -> None:
        del corpus_name
        super().__init__(
            component_id=component_id,
            query_schema_version=query_schema_version,
            config_schema_version=config_schema_version,
            query_fields=query_fields,
            range_fields=range_fields,
            cases=cases,
            benchmark_factory=(
                benchmark_factory
                if benchmark_factory is not None
                else _MissingAttentionBenchmarkFactory(component_id)
            ),
            coverage={
                "model_geometries": geometry_count,
            },
            nearest_range_bounds=nearest_range_bounds,
        )


class GdnAttentionGenerator(_AttentionGenerator):
    """Generate the recurrent Qwen GDN attention component profile."""

    def __init__(
        self,
        *,
        benchmark_factory: SweepBenchmarkFactory | None = None,
        cases: Sequence[SweepCase] | None = None,
    ) -> None:
        super().__init__(
            component_id=GDN_ATTENTION,
            query_fields=(
                "gate_activation",
                "qk_l2norm",
                "state_dtype",
                "key_heads",
                "value_heads",
                "max_seqs",
                "max_tokens",
                "state_index_columns",
            ),
            range_fields=frozenset(
                {
                    "max_seqs",
                    "max_tokens",
                    "state_index_columns",
                }
            ),
            cases=gdn_cases() if cases is None else cases,
            corpus_name="gdn",
            geometry_count=len(GDN_GEOMETRIES),
            benchmark_factory=benchmark_factory or GdnBenchmarkFactory(),
            config_schema_version=2,
            nearest_range_bounds={
                "max_seqs": (1, max(COMMON_SEQUENCE_CAPACITIES)),
                "max_tokens": (
                    1,
                    max(COMMON_SEQUENCE_CAPACITIES) * max(GDN_STATE_INDEX_COLUMNS),
                ),
                "state_index_columns": (1, max(GDN_STATE_INDEX_COLUMNS)),
            },
        )


class GqaAttentionGenerator(_AttentionGenerator):
    """Generate the paged GQA attention component profile."""

    def __init__(
        self,
        *,
        benchmark_factory: SweepBenchmarkFactory | None = None,
        cases: Sequence[SweepCase] | None = None,
    ) -> None:
        super().__init__(
            component_id=GQA_ATTENTION,
            query_fields=(
                "mode",
                "q_dtype",
                "kv_dtype",
                "q_heads",
                "kv_heads",
                "head_dim_qk",
                "head_dim_vo",
                "page_size",
                "kv_cache_layout",
                "batch_size",
                "query_len",
                "cache_tokens",
                "window_left",
                "requested_graph_ctas_per_sm",
                "force_split_kv",
            ),
            range_fields=frozenset({"batch_size", "query_len", "cache_tokens"}),
            cases=gqa_cases() if cases is None else cases,
            corpus_name="gqa",
            geometry_count=len(GQA_GEOMETRIES),
            benchmark_factory=benchmark_factory or GqaBenchmarkFactory(),
            query_schema_version=2,
            config_schema_version=2,
        )


class _QsaSession(AbstractContextManager["_QsaSession"]):
    _CANDIDATES = tuple(
        SweepCandidate.create(
            {
                "backend": "cutedsl",
                "sparse_gqa_direct_kv_warps": kv_warps,
            }
        )
        for kv_warps in (2, 1, 4)
    )

    def __init__(self, context: GenerationContext) -> None:
        self._context = context

    def __enter__(self) -> "_QsaSession":
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
        import argparse

        import torch

        from benchmarks.benchmark_qsa import BenchmarkCase, PROFILES, _run_case
        from b12x.attention.qsa._policy import QsaConfig
        from b12x.policy import PolicyContext, PolicyMode

        from .gpu_workers import _l2_flush_fn

        metadata = case.metadata
        tp_size = int(metadata["tensor_parallel_size"])
        profile_name = f"tp{tp_size}"
        kv_dtype = str(metadata["kv_dtype"])
        benchmark_dtype = "fp8_e4m3" if kv_dtype == "float8_e4m3fn" else "bf16"
        benchmark_case = BenchmarkCase(
            PROFILES[profile_name],
            int(metadata["rows"]),
            int(metadata["context"]),
            kind=str(metadata["kind"]),
            main_page_size=int(metadata["main_page_size"]),
            planned_max_batch=int(case.query["max_batch"]),
            planned_max_q_rows=int(case.query["max_q_rows"]),
            planned_max_speculative_tokens=int(case.query["max_speculative_tokens"]),
        )
        device = torch.device("cuda", self._context.device_ordinal)
        settings = self._context.settings
        flush = _l2_flush_fn(device, enabled=settings.cold_l2)
        base_policy = PolicyContext.for_device(
            device,
            mode=PolicyMode.HEURISTIC_ONLY,
        )
        args = argparse.Namespace(
            seed=settings.seed,
            main_cache_layout="interleaved",
            kv_cache_dtype=benchmark_dtype,
            warmup=settings.warmup,
            eager_replays=1,
            graph_replays=max(1, settings.groups * settings.repetitions),
        )
        case_index = int(case.case_id[-8:], 16)
        measurements = []
        for candidate in candidates:
            try:
                config = QsaConfig.from_profile(candidate.config)
                policy = base_policy.with_override(QSA_ATTENTION, config)
                result = _run_case(
                    benchmark_case,
                    args=args,
                    device=device,
                    l2_flush=flush,
                    case_index=case_index,
                    policy=policy,
                )
                timing = result["timing"]
                graph_contract = result["graph_contract"]
                correctness = result["correctness"]
                if not all(
                    isinstance(item, Mapping)
                    for item in (timing, graph_contract, correctness)
                ):
                    raise TypeError("QSA benchmark result sections must be objects")
                graph_timing = timing["cuda_graph"]
                if not isinstance(graph_timing, Mapping):
                    raise TypeError("QSA CUDA graph timing must be an object")
                summary = graph_timing["replay_summary"]
                if not isinstance(summary, Mapping):
                    raise TypeError("QSA graph replay summary must be an object")
                correct = bool(
                    correctness["graph_finite"]
                    and correctness["graph_nonzero_elements"]
                    and correctness["eager_graph_exact"]
                    and correctness["graph_persistent_state_exact"]
                    and correctness["graph_main_kv_read_only"]
                    and graph_contract["stable_bound_addresses"]
                    and graph_contract["replay_allocation_delta_bytes"] == 0
                )
                measurements.append(
                    SweepMeasurement(
                        candidate=candidate,
                        latency_us=float(summary["median_us"]),
                        correct=correct,
                        metrics={
                            "graph_replay_samples_us": list(
                                graph_timing["replay_samples_us"]
                            ),
                            "page_size": benchmark_case.main_page_size,
                            "rows": benchmark_case.rows,
                            "context": benchmark_case.context,
                            "kv_dtype": kv_dtype,
                            "tensor_parallel_size": tp_size,
                        },
                    )
                )
            except Exception as error:
                measurements.append(
                    SweepMeasurement(
                        candidate=candidate,
                        latency_us=None,
                        correct=False,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        return tuple(measurements)


class _QsaBenchmarkFactory:
    def __call__(self, group_id, cases, context):
        del group_id, cases
        return _QsaSession(context)


def _qsa_candidate_tie_rank(candidate: SweepCandidate) -> int:
    return {2: 0, 1: 1, 4: 2}[int(candidate.config["sparse_gqa_direct_kv_warps"])]


class QsaAttentionGenerator(DiscreteSweepGenerator):
    """Race selected-position QSA launch geometry on public graph transactions."""

    def __init__(
        self,
        *,
        benchmark_factory: SweepBenchmarkFactory | None = None,
        cases: Sequence[SweepCase] | None = None,
    ) -> None:
        from b12x.attention.qsa._policy import QSA_POLICY, QsaQuery

        super().__init__(
            component_id=QSA_POLICY.component_id,
            query_schema_version=QSA_POLICY.query_schema_version,
            config_schema_version=QSA_POLICY.config_schema_version,
            query_fields=tuple(QsaQuery.__dataclass_fields__),
            range_fields=frozenset(),
            cases=qsa_cases() if cases is None else cases,
            benchmark_factory=benchmark_factory or _QsaBenchmarkFactory(),
            coverage={
                "profile_cases": len(qsa_cases() if cases is None else cases),
                "candidate_kv_warps": [2, 1, 4],
                "unmeasured_queries": "heuristic",
            },
            candidate_contract_version=2,
            candidate_tie_breaker=_qsa_candidate_tie_rank,
        )


class MlaAttentionGenerator(_AttentionGenerator):
    """Generate the dense MLA attention component profile."""

    def __init__(
        self,
        *,
        benchmark_factory: SweepBenchmarkFactory | None = None,
        cases: Sequence[SweepCase] | None = None,
    ) -> None:
        super().__init__(
            component_id=MLA_ATTENTION,
            query_fields=(
                "mode",
                "q_dtype",
                "kv_dtype",
                "num_q_heads",
                "qk_head_dim",
                "v_head_dim",
                "page_size",
                "query_rows",
                "max_batch",
                "cache_tokens",
                "physical_record_width",
                "window_size",
                "use_cuda_graph",
            ),
            range_fields=frozenset({"query_rows", "cache_tokens"}),
            cases=mla_cases() if cases is None else cases,
            corpus_name="mla",
            geometry_count=len(MLA_GEOMETRIES),
            benchmark_factory=benchmark_factory or MlaBenchmarkFactory(),
            query_schema_version=2,
        )


class CompressedSparseMlaAttentionGenerator(_AttentionGenerator):
    """Generate the compressed sparse-MLA component profile."""

    def __init__(
        self,
        *,
        benchmark_factory: SweepBenchmarkFactory | None = None,
        cases: Sequence[SweepCase] | None = None,
    ) -> None:
        super().__init__(
            component_id=COMPRESSED_SPARSE_MLA_ATTENTION,
            query_fields=(
                "layout",
                "mode",
                "q_dtype",
                "kv_dtype",
                "num_q_heads",
                "qk_head_dim",
                "v_head_dim",
                "swa_width",
                "swa_page_size",
                "indexed_width",
                "indexed_page_size",
                "query_rows",
            ),
            range_fields=frozenset({"swa_width", "indexed_width", "query_rows"}),
            cases=sparse_mla_cases() if cases is None else cases,
            corpus_name="sparse_mla",
            geometry_count=len(SPARSE_MLA_GEOMETRIES),
            benchmark_factory=benchmark_factory or SparseMlaBenchmarkFactory(),
        )


__all__ = [
    "GdnAttentionGenerator",
    "GqaAttentionGenerator",
    "MlaAttentionGenerator",
    "QsaAttentionGenerator",
    "CompressedSparseMlaAttentionGenerator",
]
