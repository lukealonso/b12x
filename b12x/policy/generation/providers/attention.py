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
        candidate_contract_version: int = 1,
        subset_reuse_contract_versions: Sequence[int] = (),
        nearest_range_bounds: Mapping[str, tuple[int, int]] | None = None,
    ) -> None:
        del corpus_name
        super().__init__(
            component_id=component_id,
            query_schema_version=query_schema_version,
            config_schema_version=config_schema_version,
            candidate_contract_version=candidate_contract_version,
            subset_reuse_contract_versions=subset_reuse_contract_versions,
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

    @staticmethod
    def validate_region_decision(inputs, device, decision):
        from b12x.sequence.gdn_decode._policy import TUNING_PROBLEM

        query = TUNING_PROBLEM.query_from_inputs(inputs)
        kda = query.value_heads == query.key_heads
        if (min(query.key_heads, query.value_heads, query.max_seqs, query.max_tokens, query.state_index_columns) <= 0
                or query.state_index_columns > 8 or query.max_tokens > query.max_seqs * query.state_index_columns
                or query.value_heads not in (query.key_heads, 3 * query.key_heads)
                or query.state_dtype not in ("bfloat16", "float32")
                or query.gate_activation not in (("sigmoid",) if kda else ("sigmoid", "silu"))):
            raise ValueError("GDN fixtures require Qwen 3:1 or KDA 1:1 heads, supported gates/state dtype, and tokens within sequence/column capacity")
        config = TUNING_PROBLEM.lower(query, device, decision)
        if config.backend != ("triton" if kda else "cutedsl"):
            raise ValueError("GDN decision must use the production backend for its decay recipe")

    @staticmethod
    def cases_for_tuning_queries(queries):
        from b12x.policy.problem import stable_identity

        for query in queries:
            kda = query["key_heads"] == query["value_heads"]
            GdnAttentionGenerator.validate_region_decision(query, None, {"backend": "triton" if kda else "cutedsl"})
            group = f"shape-{stable_identity(query)[:24]}"
            for scenario, tokens in (("full", query["max_tokens"]), ("partial", max(1, query["max_tokens"] // 2))):
                sequences = min(query["max_seqs"], tokens)
                quotient, remainder = divmod(tokens, sequences)
                lengths = [quotient + (index < remainder) for index in range(sequences)]
                yield SweepCase.create(group_id=group, query=query, scenario=scenario,
                                       metadata={"decay_recipe": "kda" if kda else "gdn", "query_lengths": lengths})

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
            candidate_contract_version=2,
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

    def tuning_inputs(self, query):
        from b12x.policy.types import FrozenMapping

        return FrozenMapping({**query, "requested_max_work_items": query.get("requested_max_work_items"),
                              "requested_max_partial_rows": query.get("requested_max_partial_rows")})

    @staticmethod
    def validate_region_decision(inputs, device, decision):
        from b12x.attention.paged._policy import TUNING_PROBLEM

        query = TUNING_PROBLEM.query_from_inputs(inputs)
        if (query.q_dtype != "bfloat16" or query.kv_dtype not in ("bfloat16", "float8_e4m3fn")
                or query.head_dim_qk != query.head_dim_vo or query.head_dim_qk not in (128, 256)
                or query.page_size not in (64, 128) or query.cache_tokens % query.page_size
                or query.batch_size <= 0):
            raise ValueError("GQA search requires BF16 decode, matched 128/256 head dimensions, and whole-page capacity")
        TUNING_PROBLEM.lower(query, device, decision)

    @staticmethod
    def cases_for_tuning_queries(queries):
        from b12x.policy.problem import stable_identity

        for query in queries:
            group = {name: value for name, value in query.items() if name not in {"batch_size", "kv_cache_layout"}}
            yield SweepCase.create(group_id=f"shape-{stable_identity(group)[:24]}", query=query)

    def profile_config(self, query, candidate):
        if candidate.decision is None:
            raise ValueError("GQA profiles require independent decision metadata")
        return candidate.decision

    def build_planner(self, records, *, device=None):
        """Cover nonbinding storage limits with the identical measured schedule."""
        from dataclasses import replace
        from b12x.attention.paged._policy import TUNING_PROBLEM, WORKSPACE_LIMITS
        from b12x.policy.generation.reducer import build_axis_tree
        from b12x.policy.types import ExactDecisionNode, MatchRange, ProfileLeaf, RangeDecisionNode

        limits = tuple(field for field, _ in WORKSPACE_LIMITS)

        def terminal(group):
            def on_miss(node, fallback):
                if isinstance(node, ProfileLeaf):
                    return node
                return replace(node, branches=tuple((key, on_miss(child, fallback)) for key, child in node.branches),
                               default=fallback if node.default is None else on_miss(node.default, fallback))

            ordered = sorted(group, key=lambda record: (sum(record.query[field] is not None for field in limits),
                                                         repr(record.query)))
            result = None
            for record in ordered:
                query = TUNING_PROBLEM.query_from_inputs(record.query)
                config = TUNING_PROBLEM.lower(query, device, record.config)
                guard = ProfileLeaf.create(name=SweepCandidate.create(record.config).candidate_id, config=record.config)
                for field, requirement in reversed(WORKSPACE_LIMITS):
                    value = record.query[field]
                    default = None if value is not None else RangeDecisionNode(
                        field=field, branches=((MatchRange(getattr(config, requirement), 2**63 - 1), guard),))
                    guard = ExactDecisionNode(field=field, branches=((value, guard),), default=default)
                result = guard if result is None else on_miss(guard, result)
            return result

        order = tuple(field for field in self._query_fields if field not in limits) + limits
        return build_axis_tree(records, field_order=order, range_fields=self._range_fields,
                               terminal_fields=frozenset(limits), terminal_factory=terminal)

    compile_constraint_coverage = build_planner

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
                "requested_max_work_items",
                "requested_max_partial_rows",
                "force_split_kv",
            ),
            range_fields=frozenset({"batch_size", "query_len", "cache_tokens"}),
            cases=gqa_cases() if cases is None else cases,
            corpus_name="gqa",
            geometry_count=len(GQA_GEOMETRIES),
            benchmark_factory=benchmark_factory or GqaBenchmarkFactory(),
            query_schema_version=3,
            config_schema_version=3,
            candidate_contract_version=5,
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

    def measure(self, case, candidates):
        from dataclasses import replace
        import torch
        from b12x.attention import qsa
        from b12x.attention.qsa._policy import QsaConfig, QsaQuery
        from b12x.policy import PolicyContext, PolicyMode
        from b12x.policy.generation.replay import PreparedCandidate, measure_prepared_candidates
        from benchmarks.benchmark_qsa import _make_caps, _prepare_case, _qualify_prepared_graph
        from .gpu_workers import _l2_flush_fn

        kv_dtype = str(case.metadata["kv_dtype"])
        benchmark_case = _qsa_benchmark_case(case)
        device = torch.device("cuda", self._context.device_ordinal)
        caps = _make_caps(benchmark_case, device, kv_dtype=getattr(torch, kv_dtype))
        if QsaQuery.from_caps(caps).profile_fields() != dict(case.query):
            raise ValueError("QSA production fixture does not preserve the complete tuning query")
        settings = self._context.settings
        base_policy = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)
        fixture = _prepare_case(benchmark_case, device=device,
            seed=settings.seed + 1009 * int(case.case_id[-8:], 16), main_cache_layout="interleaved",
            kv_cache_dtype="fp8_e4m3" if kv_dtype == "float8_e4m3fn" else "bf16", policy=base_policy)
        shared = {name: getattr(fixture.binding, name) for name in (
            "main_k_cache", "main_v_cache", "k_descale", "v_descale", "main_block_table", "compressed_k_cache",
            "compressed_block_table", "raw_k_ring", "raw_logical_positions", "raw_rope_positions",
            "raw_interval_start_positions", "raw_state_slot_ids", "index_q_norm_weight", "index_k_norm_weight",
            "rope_cos", "rope_sin")}
        prepared = []
        for candidate in candidates:
            try:
                config = QsaConfig.from_profile(candidate.config)
                plan = qsa.plan(caps, policy=base_policy.with_override(QSA_ATTENTION, config))
                spec, = plan.scratch_specs()
                scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
                binding = qsa.bind(plan, scratch=scratch, output=torch.empty_like(fixture.binding.output),
                                   selected_positions=torch.empty_like(fixture.binding.selected_positions), **shared)
                active = replace(fixture, binding=binding)
                active.state_restore.restore()
                graph, correctness, addresses, allocation = _qualify_prepared_graph(
                    active, warmup=settings.warmup, device=device)
                prepared.append(PreparedCandidate(candidate=candidate, graph=graph,
                    correct=bool(correctness["graph_finite"] and correctness["graph_nonzero_elements"] and allocation == 0),
                    owners=(active, binding, scratch), before_each=active.state_restore.restore, aggregation="median",
                    metrics={"correctness": correctness, "stable_addresses": True, "addresses": addresses,
                             "frozen_resolution_capture": True, "state_reset_before_each": True,
                             "page_size": benchmark_case.main_page_size, "rows": benchmark_case.rows,
                             "context": benchmark_case.context, "kv_dtype": kv_dtype,
                             "tensor_parallel_size": int(case.metadata["tensor_parallel_size"])}))
            except Exception as exc:
                prepared.append(SweepMeasurement(candidate=candidate, latency_us=None, correct=False,
                                                 error=f"{type(exc).__name__}: {exc}"))
        return measure_prepared_candidates(prepared, settings=settings, device=device,
                                           flush=_l2_flush_fn(device, enabled=settings.cold_l2))


class _QsaBenchmarkFactory:
    def __call__(self, group_id, cases, context):
        del group_id, cases
        return _QsaSession(context)


def _qsa_candidate_tie_rank(candidate: SweepCandidate) -> int:
    return {2: 0, 1: 1, 4: 2}[int(candidate.config["sparse_gqa_direct_kv_warps"])]


class QsaAttentionGenerator(DiscreteSweepGenerator):
    """Race selected-position QSA launch geometry on public graph transactions."""

    @staticmethod
    def validate_region_decision(inputs, device, decision):
        import torch
        from benchmarks.benchmark_qsa import _make_caps
        from b12x.attention.qsa._policy import TUNING_PROBLEM, QsaQuery

        query = TUNING_PROBLEM.query_from_inputs(inputs)
        TUNING_PROBLEM.lower(query, device, decision)
        if (query.max_seq_len < 8 or query.max_q_rows > query.max_seq_len
                or query.max_seq_len % 4):
            raise ValueError("QSA fixtures require compression-aligned context >= 8 and prefill rows within context")
        case = _qsa_shape_case(inputs, "full-prefill", query.max_q_rows, "prefill")
        caps = _make_caps(_qsa_benchmark_case(case), torch.device("cuda:0"),
                          kv_dtype=getattr(torch, query.kv_dtype))
        if QsaQuery.from_caps(caps) != query:
            raise ValueError("QSA fixture geometry must match the production benchmark contract")

    @staticmethod
    def cases_for_tuning_queries(queries):
        for query in queries:
            QsaAttentionGenerator.validate_region_decision(
                query, None, {"backend": "cutedsl", "sparse_gqa_direct_kv_warps": 2})
            yield _qsa_shape_case(query, "full-prefill", query["max_q_rows"], "prefill")
            yield _qsa_shape_case(query, "partial-prefill", max(1, query["max_q_rows"] // 2), "prefill")
            yield _qsa_shape_case(query, "decode", query["max_batch"], "throughput")

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


def _qsa_benchmark_case(case):
    from benchmarks.benchmark_qsa import BenchmarkCase, PROFILES

    metadata = case.metadata
    return BenchmarkCase(
        PROFILES[f"tp{metadata['tensor_parallel_size']}"], metadata["rows"], metadata["context"],
        kind=metadata["kind"], main_page_size=metadata["main_page_size"],
        planned_max_batch=case.query["max_batch"], planned_max_q_rows=case.query["max_q_rows"],
        planned_max_speculative_tokens=case.query["max_speculative_tokens"],
    )


def _qsa_shape_case(query, scenario, rows, kind):
    from b12x.policy.problem import stable_identity

    tp = {(24, 2): 1, (12, 1): 2, (6, 1): 4}.get((query["q_heads"], query["kv_heads"]))
    if tp is None:
        raise ValueError("QSA fixtures require a supported tensor-parallel head layout")
    return SweepCase.create(
        group_id=f"shape-{stable_identity(query)[:24]}", query=query, scenario=scenario,
        metadata={"tensor_parallel_size": tp, "rows": rows, "context": query["max_seq_len"],
                  "main_page_size": query["main_page_size"], "kv_dtype": query["kv_dtype"], "kind": kind},
    )


def _mla_split_candidates(query, device):
    from b12x.attention.dense_mla._policy import DENSE_MLA_POLICY, DenseMlaQuery

    default = DENSE_MLA_POLICY.heuristic(DenseMlaQuery(**query), device).max_splits
    limits = {default}
    value = 1
    while value < default:
        limits.add(value)
        value *= 2
    return tuple(SweepCandidate.create({"max_splits": limit}) for limit in sorted(limits))


class MlaAttentionGenerator(_AttentionGenerator):
    """Generate the dense MLA attention component profile."""

    @staticmethod
    def validate_region_decision(inputs, device, decision):
        from b12x.attention.dense_mla._policy import TUNING_PROBLEM, _query_tile
        from b12x.attention.dense_mla._layout import make_smem_layout
        from b12x.attention._shared.mla.smem import SM120_SMEM_CARVEOUT_BYTES
        from .gpu_workers import _dense_mla_caps

        query = TUNING_PROBLEM.query_from_inputs(inputs)
        if (query.mode not in ("decode", "extend") or query.window_size is not None
                or not query.use_cuda_graph or query.physical_record_width != query.qk_head_dim
                or query.max_batch != (query.query_rows if query.mode == "decode" else 1)
                or (query.mode == "extend" and query.query_rows > query.cache_tokens)):
            raise ValueError("dense MLA fixtures require a full-window graph contract with one row per decode sequence or one extend sequence")
        config = TUNING_PROBLEM.lower(query, device, decision)
        _dense_mla_caps(SweepCase.create(group_id="legality", query=inputs), device="cpu", max_splits=config.max_splits)
        layout = make_smem_layout(query_tile=_query_tile(query), fp8=query.kv_dtype == "float8_e4m3fn",
                                  qk_dim=query.qk_head_dim)
        if layout.total_bytes > SM120_SMEM_CARVEOUT_BYTES:
            raise ValueError("dense MLA query exceeds the SM120/SM121 shared-memory capacity")
        if config.max_splits not in {candidate.config["max_splits"] for candidate in _mla_split_candidates(inputs, device)}:
            raise ValueError("dense MLA decision is outside the production split candidate set")

    @staticmethod
    def cases_for_tuning_queries(queries):
        from b12x.policy.problem import stable_identity

        for query in queries:
            MlaAttentionGenerator.validate_region_decision(query, None, {"max_splits": 1})
            yield SweepCase.create(group_id=f"shape-{stable_identity(query)[:24]}", query=query)

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
            candidate_contract_version=2,
        )


def _compressed_mla_candidates(query, device):
    from b12x.attention._shared.mla.compressed_config import (
        compressed_sparse_mla_split_config_for_contract,
        compressed_sparse_mla_uses_single_pass_decode,
    )

    single_pass = query["mode"] != "decode" or compressed_sparse_mla_uses_single_pass_decode(
        rows=query["query_rows"], heads=query["num_q_heads"], swa_width=query["swa_width"],
        indexed_width=query["indexed_width"], swa_page_size=query["swa_page_size"],
        indexed_page_size=query["indexed_page_size"],
        compute_capability=(0, 0) if device is None else device.compute_capability,
    )
    representatives = {}
    for cap in ((1,) if single_pass else (1, 2, 4, 8, 16, 32, 64, 256)):
        config = compressed_sparse_mla_split_config_for_contract(
            rows=query["query_rows"], width=query["swa_width"] + query["indexed_width"], max_chunks=cap,
        )
        representatives.setdefault((config.chunk_size, config.num_chunks), cap)
    return tuple(SweepCandidate.create({"max_chunks_per_row": cap}) for cap in representatives.values())


class CompressedSparseMlaAttentionGenerator(_AttentionGenerator):
    """Generate the compressed sparse-MLA component profile."""

    @staticmethod
    def validate_region_decision(inputs, device, decision):
        from b12x.attention.compressed_sparse_mla._policy import TUNING_PROBLEM
        from b12x.attention._shared.mla.compressed_config import compressed_sparse_mla_uses_single_pass_decode

        query = TUNING_PROBLEM.query_from_inputs(inputs)
        if (query.layout != "compressed_dsv4" or query.mode not in ("decode", "extend")
                or query.q_dtype != "bfloat16" or query.kv_dtype != "float8_e4m3fn"
                or query.qk_head_dim != 512 or query.v_head_dim != 512
                or query.num_q_heads not in (8, 16, 32, 64) or query.query_rows <= 0
                or query.swa_width < 0 or query.indexed_width < 0
                or query.swa_width + query.indexed_width <= 0
                or query.swa_page_size not in (64, 256) or query.indexed_page_size not in (2, 64)):
            raise ValueError("compressed MLA fixtures require the 512-wide BF16/DSV4 contract, supported head/page layouts, and positive row/total-width capacities")
        single_pass = query.mode == "extend" or compressed_sparse_mla_uses_single_pass_decode(
            rows=query.query_rows, heads=query.num_q_heads, swa_width=query.swa_width,
            indexed_width=query.indexed_width, swa_page_size=query.swa_page_size,
            indexed_page_size=query.indexed_page_size,
            compute_capability=(0, 0) if device is None else device.compute_capability,
        )
        if single_pass and query.swa_width != 128:
            raise ValueError("compressed MLA single-pass production kernels require swa_width=128")
        TUNING_PROBLEM.lower(query, device, decision)
        if decision["max_chunks_per_row"] not in {
            candidate.config["max_chunks_per_row"] for candidate in _compressed_mla_candidates(inputs, device)
        }:
            raise ValueError("compressed MLA decision is outside the production split candidate set")

    @staticmethod
    def cases_for_tuning_queries(queries):
        from b12x.policy.problem import stable_identity

        for query in queries:
            CompressedSparseMlaAttentionGenerator.validate_region_decision(query, None, {"max_chunks_per_row": 1})
            yield SweepCase.create(group_id=f"shape-{stable_identity(query)[:24]}", query=query)

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
            candidate_contract_version=2,
        )


__all__ = [
    "GdnAttentionGenerator",
    "GqaAttentionGenerator",
    "MlaAttentionGenerator",
    "QsaAttentionGenerator",
    "CompressedSparseMlaAttentionGenerator",
]
