"""Measured sweep providers for tunable component launch policies."""

from __future__ import annotations

import dataclasses
import gc
import math
from collections.abc import Sequence
from contextlib import AbstractContextManager

from b12x.policy.components import DSA_INDEXER, NVFP4_QUANTIZATION, VARLEN_ATTENTION
from b12x.policy.generation.contracts import (
    ComponentGenerationResult,
    GenerationContext,
    WorkEstimate,
)
from b12x.policy.generation.reducer import decision_node_to_dict
from b12x.policy.types import DecisionNode, ProfileLeaf
from b12x.policy.generation.sweep import (
    DiscreteSweepGenerator,
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
)

from .gpu_workers import (
    _bounded_repetitions,
    _cuda_event_samples_us,
    _l2_flush_fn,
    _median_of_group_medians,
)


def _nvfp4_cases() -> tuple[SweepCase, ...]:
    return tuple(
        SweepCase.create(
            group_id=f"m{rows}-k{columns}",
            query={
                "dtype": "bfloat16",
                "rows": rows,
                "columns": columns,
            },
        )
        for rows in (128, 512, 2_048)
        for columns in (2_560, 4_096, 5_120, 7_168, 10_240)
    )


def _varlen_attention_cases() -> tuple[SweepCase, ...]:
    return tuple(
        SweepCase.create(
            group_id=(
                f"{variant}-d{head_dim}-c{int(causal)}-"
                "b4-q128-k1024"
            ),
            query={
                "variant": variant,
                "dtype": "bfloat16",
                "causal": causal,
                "batch_size": 4,
                "q_heads": 16,
                "kv_heads": 4,
                "q_head_dim": head_dim,
                "v_head_dim": head_dim,
                "query_rows": 512,
                "kv_rows": 4_096,
                "max_seqlen_q": 128,
                "max_seqlen_k": 1_024,
            },
        )
        for variant in ("batched", "varlen")
        for head_dim in (64, 128, 256)
        for causal in (False, True)
    )


class _Nvfp4Session(AbstractContextManager["_Nvfp4Session"]):
    _CANDIDATES = tuple(
        SweepCandidate.create(
            {
                "backend": "cutedsl",
                "liveness_strategy": strategy,
            }
        )
        for strategy in ("retain", "packed")
    )

    def __init__(self, context) -> None:
        self._context = context

    def __enter__(self) -> "_Nvfp4Session":
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

        from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
        from b12x.quantization import nvfp4
        from b12x.quantization.nvfp4._policy import Nvfp4QuantizationConfig

        if candidates != self._CANDIDATES:
            raise ValueError("NVFP4 worker received an unknown candidate set")
        settings = self._context.settings
        device = torch.device("cuda", self._context.device_ordinal)
        rows = int(case.query["rows"])
        columns = int(case.query["columns"])
        generator = torch.Generator(device=device).manual_seed(
            settings.seed + int(case.case_id[-8:], 16)
        )
        with torch.cuda.device(self._context.device_ordinal):
            source = torch.randn(
                (rows, columns),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            global_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
            row_counts = torch.tensor([rows], dtype=torch.int32, device=device)
            packed_ref, scale_view_ref = quantize_grouped_nvfp4_torch(
                source.unsqueeze(0),
                row_counts,
                global_scale,
            )
            scale_ref = (
                scale_view_ref.permute(5, 2, 4, 0, 1, 3)
                .contiguous()
                .view(torch.uint8)
                .reshape(-1)
            )
            measurements = []
            flush = _l2_flush_fn(device, enabled=settings.cold_l2)
            for candidate in candidates:
                try:
                    config = Nvfp4QuantizationConfig.from_profile(candidate.config)
                    policy = self._context_policy(device).with_override(
                        NVFP4_QUANTIZATION,
                        config,
                    )
                    plan = nvfp4.plan(rows, columns, policy=policy)
                    outputs = nvfp4.allocate_outputs(plan, device=device)

                    def run() -> None:
                        nvfp4.run(
                            plan=plan,
                            x=source,
                            global_scale=global_scale,
                            outputs=outputs,
                        )

                    for _ in range(settings.warmup):
                        run()
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    graph.replay()
                    torch.cuda.synchronize(device)
                    packed_actual = outputs.packed_a_storage.permute(1, 2, 0)
                    packed_exact = bool(torch.equal(packed_actual, packed_ref))
                    scales_exact = bool(torch.equal(outputs.scale_flat, scale_ref))
                    nonzero = bool(
                        torch.count_nonzero(packed_actual).item()
                        and torch.count_nonzero(outputs.scale_flat).item()
                    )
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    repetitions = _bounded_repetitions(
                        settings,
                        pilot_us=float(start.elapsed_time(end)) * 1_000.0,
                    )
                    allocated_before = torch.cuda.memory_allocated(device)
                    samples = _cuda_event_samples_us(
                        graph.replay,
                        count=settings.groups * repetitions,
                        device=device,
                        flush=flush,
                    )
                    allocated_after = torch.cuda.memory_allocated(device)
                    latency = _median_of_group_medians(
                        samples,
                        groups=settings.groups,
                        repetitions=repetitions,
                    )
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=latency,
                            correct=(
                                packed_exact
                                and scales_exact
                                and nonzero
                                and allocated_after <= allocated_before
                            ),
                            metrics={
                                "packed_exact": packed_exact,
                                "scales_exact": scales_exact,
                                "nonzero": nonzero,
                                "replay_allocation_bytes": (
                                    allocated_after - allocated_before
                                ),
                            },
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - failed candidates survive
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=None,
                            correct=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            return tuple(measurements)

    @staticmethod
    def _context_policy(device):
        from b12x.policy import PolicyContext, PolicyMode

        return PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)


class _Nvfp4BenchmarkFactory:
    def __call__(self, group_id, cases, context):
        del group_id
        if len(cases) != 1:
            raise ValueError("NVFP4 allocation groups contain one case")
        return _Nvfp4Session(context)


class Nvfp4QuantizationGenerator(DiscreteSweepGenerator):
    """Race both real NVFP4 register-liveness schedules."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=NVFP4_QUANTIZATION,
            query_schema_version=1,
            config_schema_version=2,
            query_fields=("dtype", "rows", "columns"),
            range_fields=frozenset({"rows", "columns"}),
            cases=_nvfp4_cases() if cases is None else cases,
            benchmark_factory=_Nvfp4BenchmarkFactory(),
            coverage={},
        )


def _attention_candidates(head_dim: int) -> tuple[SweepCandidate, ...]:
    if head_dim <= 128:
        tiles = ((64, 64), (64, 128), (128, 64), (128, 128))
    elif head_dim == 256:
        tiles = ((32, 32), (64, 32), (64, 48), (64, 64), (128, 32))
    else:
        raise ValueError(f"unsupported contiguous attention head dim {head_dim}")
    return tuple(
        SweepCandidate.create({"tile_m": tile_m, "tile_n": tile_n})
        for tile_m, tile_n in tiles
    )


def _attention_reference(case: SweepCase, q, k, v, cu_q, cu_k):
    import torch

    from b12x.attention.paged.reference import attention_reference

    causal = bool(case.query["causal"])
    if str(case.query["variant"]) == "batched":
        return attention_reference(q, k, v, causal=causal)
    outputs = []
    lses = []
    batch_size = int(case.query["batch_size"])
    for batch_idx in range(batch_size):
        q_start = int(cu_q[batch_idx].item())
        q_end = int(cu_q[batch_idx + 1].item())
        k_start = int(cu_k[batch_idx].item())
        k_end = int(cu_k[batch_idx + 1].item())
        output, lse = attention_reference(
            q[q_start:q_end],
            k[k_start:k_end],
            v[k_start:k_end],
            causal=causal,
        )
        outputs.append(output)
        lses.append(lse)
    return torch.cat(outputs, dim=0), torch.cat(lses, dim=1)


class _VarlenAttentionSession(
    AbstractContextManager["_VarlenAttentionSession"]
):
    def __init__(self, context) -> None:
        self._context = context

    def __enter__(self) -> "_VarlenAttentionSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        import torch

        gc.collect()
        torch.cuda.synchronize(self._context.device_ordinal)
        torch.cuda.empty_cache()
        return None

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        return _attention_candidates(int(case.query["q_head_dim"]))

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch
        import torch.nn.functional as torch_functional

        from b12x.attention import varlen
        from b12x.attention.varlen._policy import VarlenAttentionConfig
        from b12x.policy import PolicyContext, PolicyMode

        settings = self._context.settings
        device = torch.device("cuda", self._context.device_ordinal)
        variant = str(case.query["variant"])
        batch_size = int(case.query["batch_size"])
        q_heads = int(case.query["q_heads"])
        kv_heads = int(case.query["kv_heads"])
        q_head_dim = int(case.query["q_head_dim"])
        v_head_dim = int(case.query["v_head_dim"])
        max_q = int(case.query["max_seqlen_q"])
        max_k = int(case.query["max_seqlen_k"])
        causal = bool(case.query["causal"])
        generator = torch.Generator(device=device).manual_seed(
            settings.seed + int(case.case_id[-8:], 16)
        )
        with torch.cuda.device(self._context.device_ordinal):
            if variant == "batched":
                q_shape = (batch_size, max_q, q_heads, q_head_dim)
                k_shape = (batch_size, max_k, kv_heads, q_head_dim)
                v_shape = (batch_size, max_k, kv_heads, v_head_dim)
                cu_q = None
                cu_k = None
            elif variant == "varlen":
                q_shape = (batch_size * max_q, q_heads, q_head_dim)
                k_shape = (batch_size * max_k, kv_heads, q_head_dim)
                v_shape = (batch_size * max_k, kv_heads, v_head_dim)
                cu_q = torch.arange(
                    batch_size + 1,
                    dtype=torch.int32,
                    device=device,
                ).mul_(max_q)
                cu_k = torch.arange(
                    batch_size + 1,
                    dtype=torch.int32,
                    device=device,
                ).mul_(max_k)
            else:
                raise ValueError(f"unsupported attention variant {variant!r}")
            q = torch.randn(
                q_shape,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            k = torch.randn(
                k_shape,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            v = torch.randn(
                v_shape,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            expected, _expected_lse = _attention_reference(
                case, q, k, v, cu_q, cu_k
            )
            flush = _l2_flush_fn(device, enabled=settings.cold_l2)
            base_policy = PolicyContext.for_device(
                device,
                mode=PolicyMode.HEURISTIC_ONLY,
            )
            measurements = []
            for candidate in candidates:
                try:
                    config = VarlenAttentionConfig.from_profile(candidate.config)
                    policy = base_policy.with_override(VARLEN_ATTENTION, config)
                    if variant == "batched":
                        plan = varlen.create_plan_batched(
                            q,
                            k,
                            v,
                            causal=causal,
                            policy=policy,
                        )
                        scratch_plan = varlen.plan_batched(plan)
                    else:
                        plan = varlen.create_plan(
                            q,
                            k,
                            v,
                            cu_q,
                            cu_k,
                            max_seqlen_q=max_q,
                            max_seqlen_k=max_k,
                            causal=causal,
                            policy=policy,
                        )
                        scratch_plan = varlen.plan(plan)
                    (scratch_spec,) = scratch_plan.scratch_specs()
                    scratch = torch.empty(
                        scratch_spec.shape,
                        dtype=scratch_spec.dtype,
                        device=scratch_spec.device,
                    )
                    if variant == "batched":
                        binding = scratch_plan.bind(
                            scratch=scratch,
                            q=q,
                            k=k,
                            v=v,
                        )

                        def run() -> None:
                            varlen.run_batched(binding=binding)

                    else:
                        binding = scratch_plan.bind(
                            scratch=scratch,
                            q=q,
                            k=k,
                            v=v,
                            cu_seqlens_q=cu_q,
                            cu_seqlens_k=cu_k,
                            max_seqlen_q=max_q,
                            max_seqlen_k=max_k,
                            causal=causal,
                        )

                        def run() -> None:
                            varlen.run(binding=binding)

                    for _ in range(settings.warmup):
                        run()
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    binding.output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    actual = binding.output
                    finite = bool(torch.isfinite(actual).all().item())
                    cosine = float(
                        torch_functional.cosine_similarity(
                            actual.float().reshape(1, -1),
                            expected.float().reshape(1, -1),
                        ).item()
                    )
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    repetitions = _bounded_repetitions(
                        settings,
                        pilot_us=float(start.elapsed_time(end)) * 1_000.0,
                    )
                    allocated_before = torch.cuda.memory_allocated(device)
                    samples = _cuda_event_samples_us(
                        graph.replay,
                        count=settings.groups * repetitions,
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
                                repetitions=repetitions,
                            ),
                            correct=(
                                finite
                                and cosine >= 0.999
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


class _VarlenAttentionBenchmarkFactory:
    def __call__(self, group_id, cases, context):
        del group_id
        if len(cases) != 1:
            raise ValueError("attention allocation groups contain one case")
        return _VarlenAttentionSession(context)


class VarlenAttentionGenerator(DiscreteSweepGenerator):
    """Race production contiguous-attention tiles on reviewed shapes."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=VARLEN_ATTENTION,
            query_schema_version=1,
            config_schema_version=1,
            query_fields=(
                "variant",
                "dtype",
                "causal",
                "batch_size",
                "q_heads",
                "kv_heads",
                "q_head_dim",
                "v_head_dim",
                "query_rows",
                "kv_rows",
                "max_seqlen_q",
                "max_seqlen_k",
            ),
            range_fields=frozenset(
                {
                    "batch_size",
                    "query_rows",
                    "kv_rows",
                    "max_seqlen_q",
                    "max_seqlen_k",
                }
            ),
            cases=_varlen_attention_cases() if cases is None else cases,
            benchmark_factory=_VarlenAttentionBenchmarkFactory(),
            coverage={},
        )


# --- DSA indexer: fused paged route, cross-CTA merge arm race -----------------
# Query fields in DsaIndexerQuery declaration order.
_DSA_INDEXER_QUERY_FIELDS = (
    "source_layout",
    "mode",
    "dtype",
    "kv_dtype",
    "num_q_heads",
    "num_idx_heads",
    "max_q_rows",
    "max_k_rows",
    "top_k",
    "page_size",
    "score_mode",
    "shared_page_table",
)
_DSA_INDEXER_MERGE_PAGE_TABLE_WIDTH = 512
# (heads, top_k, decode row counts) of the serving contracts that take the fused
# route: GLM-5.2/5.3 (32 heads, top-k 2048), DeepSeek V4 (64 heads, top-k 512)
# and GLM-5.3 Flash (32 heads, top-k 512).
_DSA_INDEXER_MERGE_SHAPES = (
    (32, 2_048, (1, 2, 4, 8, 16)),
    (64, 512, (1, 2, 4, 8)),
    (32, 512, (1, 4, 8)),
)
# Live row lengths raced per query point; the winner must be robust across both.
_DSA_INDEXER_MERGE_SCENARIOS = (("ctx4k", 4_096), ("ctx32k", 32_768))


def _dsa_indexer_merge_cases() -> tuple[SweepCase, ...]:
    cases = []
    for heads, topk, rows_list in _DSA_INDEXER_MERGE_SHAPES:
        for rows in rows_list:
            for scenario, seq_len in _DSA_INDEXER_MERGE_SCENARIOS:
                cases.append(
                    SweepCase.create(
                        group_id=f"fused-h{heads}-k{topk}-r{rows}",
                        scenario=scenario,
                        query={
                            "source_layout": "paged",
                            "mode": "decode",
                            "dtype": "bfloat16",
                            "kv_dtype": "uint8",
                            "num_q_heads": heads,
                            "num_idx_heads": 1,
                            "max_q_rows": rows,
                            "max_k_rows": 0,
                            "top_k": topk,
                            "page_size": 64,
                            "score_mode": "dsa",
                            "shared_page_table": False,
                        },
                        metadata={
                            "seq_len": seq_len,
                            "page_table_width": _DSA_INDEXER_MERGE_PAGE_TABLE_WIDTH,
                        },
                    )
                )
    return tuple(cases)


def _dsa_indexer_merge_candidates() -> tuple[SweepCandidate, ...]:
    from b12x.attention.dsa_indexer._policy import (
        FUSED_MERGE_COOPERATIVE,
        FUSED_MERGE_SERIAL,
    )

    return tuple(
        SweepCandidate.create({"backend": "native", "fused_merge": choice})
        for choice in (FUSED_MERGE_COOPERATIVE, FUSED_MERGE_SERIAL)
    )


# Kernel time one merge-race sample must span. CUDA-event timestamps advance in
# coarse steps on some devices (2.048 us observed on SM120 parts), so a single
# replay of a 10-60 us indexer launch is quantized to a few steps. Each sample
# averages enough independently event-bracketed replays, cold-flushed before
# every replay, that the quantization error stays near one percent.
_MERGE_SAMPLE_SPAN_US = 128.0
_MERGE_MAX_REPLAYS_PER_SAMPLE = 16
_MERGE_PILOT_REPLAYS = 4


def _merge_replays_per_sample(pilot_us: float) -> int:
    return max(
        1,
        min(
            _MERGE_MAX_REPLAYS_PER_SAMPLE,
            math.ceil(_MERGE_SAMPLE_SPAN_US / max(float(pilot_us), 1.0)),
        ),
    )


def _mean_of_consecutive(samples: Sequence[float], *, width: int) -> tuple[float, ...]:
    if len(samples) % width:
        raise ValueError("sample count must be a multiple of the aggregation width")
    return tuple(
        sum(samples[index : index + width]) / width
        for index in range(0, len(samples), width)
    )


class _DsaIndexerMergeSession(AbstractContextManager["_DsaIndexerMergeSession"]):
    """Race the fused indexer's merge arms on one decode shape.

    Each candidate runs the production fused kernel with the merge threshold
    the planner derives from its ``fused_merge`` choice
    (:func:`b12x.attention.dsa_indexer.fused_indexer.resolve_fused_merge_threshold`),
    is gated on an exact top-k match against a torch reference, and is timed
    as CUDA-graph replays with a zero-allocation check.
    """

    def __init__(self, context) -> None:
        self._context = context

    def __enter__(self) -> "_DsaIndexerMergeSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        import torch

        gc.collect()
        torch.cuda.synchronize(self._context.device_ordinal)
        torch.cuda.empty_cache()
        return None

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        del case
        return _dsa_indexer_merge_candidates()

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch

        from b12x.attention.dsa_indexer._policy import DsaIndexerConfig
        from b12x.attention.dsa_indexer.fused_indexer import (
            _resolve_default_ctas_per_group,
            fused_indexer_scratch_capacity,
            resolve_fused_merge_threshold,
            run_fused_paged_indexer,
        )

        settings = self._context.settings
        device = torch.device("cuda", self._context.device_ordinal)
        heads = int(case.query["num_q_heads"])
        rows = int(case.query["max_q_rows"])
        topk = int(case.query["top_k"])
        seq_len = int(case.metadata["seq_len"])
        width = int(case.metadata["page_table_width"])
        pages_per_row = (seq_len + 63) // 64
        if pages_per_row > width:
            raise ValueError("sweep case seq_len exceeds its page table width")
        generator = torch.Generator(device=device).manual_seed(
            settings.seed + int(case.case_id[-8:], 16)
        )
        with torch.cuda.device(self._context.device_ordinal):
            num_pages = rows * pages_per_row
            q_fp8 = (
                torch.randn((rows, heads, 128), device=device, generator=generator)
                / 3
            ).to(torch.float8_e4m3fn)
            weights = torch.randn(
                (rows, heads), device=device, generator=generator, dtype=torch.float32
            )
            k_fp8 = (
                torch.randn((num_pages, 64, 128), device=device, generator=generator)
                / 3
            ).to(torch.float8_e4m3fn)
            k_scales = (
                torch.rand(
                    (num_pages, 64), device=device, generator=generator, dtype=torch.float32
                )
                + 0.1
            )
            page_table = torch.full((rows, width), -1, dtype=torch.int32, device=device)
            page_table[:, :pages_per_row] = torch.arange(
                num_pages, dtype=torch.int32, device=device
            ).view(rows, pages_per_row)
            seqlens = torch.full((rows,), seq_len, dtype=torch.int32, device=device)
            # Exact reference: relu(q . k) weighted per head, scaled per token.
            expected_sets = []
            expected_values = []
            qf = q_fp8.float()
            kf = k_fp8.float().reshape(num_pages * 64, 128)
            sf = k_scales.reshape(-1)
            for row in range(rows):
                start = row * pages_per_row * 64
                logits = (
                    torch.relu(qf[row] @ kf[start : start + seq_len].T)
                    * weights[row].unsqueeze(1)
                ).sum(0) * sf[start : start + seq_len]
                top = torch.topk(logits, topk, largest=True, sorted=True)
                expected_sets.append(set(top.indices.tolist()))
                expected_values.append(top.values)
            expected_sorted = torch.stack(expected_values)
            num_sms = torch.cuda.get_device_properties(device).multi_processor_count
            ctas_per_group = _resolve_default_ctas_per_group(
                num_rows=rows, max_pages=width, device=device
            )
            pack_capacity, state_words = fused_indexer_scratch_capacity(
                rows, topk, num_sms
            )
            pack_values = torch.empty(pack_capacity, dtype=torch.float32, device=device)
            pack_indices = torch.empty(pack_capacity, dtype=torch.int32, device=device)
            merge_state = torch.zeros(state_words, dtype=torch.int32, device=device)
            out_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
            out_values = torch.empty((rows, topk), dtype=torch.float32, device=device)
            flush = _l2_flush_fn(device, enabled=settings.cold_l2)
            measurements = []
            for candidate in candidates:
                try:
                    config = DsaIndexerConfig.from_profile(candidate.config)
                    merge_threshold = resolve_fused_merge_threshold(
                        config.fused_merge,
                        ctas_per_group=ctas_per_group,
                        num_heads=heads,
                        topk=topk,
                    )

                    def run() -> None:
                        run_fused_paged_indexer(
                            q_bytes=q_fp8.view(torch.uint8),
                            weights=weights,
                            k_quant_bytes=k_fp8.view(torch.uint8),
                            k_scales=k_scales,
                            real_page_table=page_table,
                            seqlens=seqlens,
                            num_heads=heads,
                            topk=topk,
                            out_indices=out_indices,
                            out_values=out_values,
                            ctas_per_group=ctas_per_group,
                            merge_threshold=merge_threshold,
                            pack_values=pack_values,
                            pack_indices=pack_indices,
                            merge_state=merge_state,
                            merge_state_preinitialized=True,
                        )

                    for _ in range(settings.warmup):
                        run()
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    out_indices.fill_(-1)
                    out_values.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    actual_sorted = torch.sort(out_values, dim=1, descending=True).values
                    correct = bool(torch.isfinite(out_values).all().item()) and all(
                        set(out_indices[row].tolist()) == expected_sets[row]
                        for row in range(rows)
                    ) and bool(
                        torch.allclose(actual_sorted, expected_sorted, atol=1e-2, rtol=0)
                    )
                    # The first replays after capture carry graph upload cost, so
                    # the pilot is the fastest of a few bracketed replays.
                    pilot_us = min(
                        _cuda_event_samples_us(
                            graph.replay,
                            count=_MERGE_PILOT_REPLAYS,
                            device=device,
                            flush=flush,
                        )
                    )
                    replays_per_sample = _merge_replays_per_sample(pilot_us)
                    repetitions = _bounded_repetitions(
                        settings,
                        pilot_us=pilot_us * replays_per_sample,
                    )
                    allocated_before = torch.cuda.memory_allocated(device)
                    inner_samples = _cuda_event_samples_us(
                        graph.replay,
                        count=settings.groups * repetitions * replays_per_sample,
                        device=device,
                        flush=flush,
                    )
                    allocation_delta = torch.cuda.memory_allocated(device) - allocated_before
                    latency = _median_of_group_medians(
                        _mean_of_consecutive(inner_samples, width=replays_per_sample),
                        groups=settings.groups,
                        repetitions=repetitions,
                    )
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=latency,
                            correct=correct and allocation_delta == 0,
                            metrics={
                                "merge_threshold": int(merge_threshold),
                                "ctas_per_group": int(ctas_per_group),
                                "seq_len": seq_len,
                                "repetitions": int(repetitions),
                                "replays_per_sample": int(replays_per_sample),
                                "inner_samples_us": [
                                    round(value, 3) for value in inner_samples
                                ],
                                "replay_allocation_bytes": int(allocation_delta),
                            },
                        )
                    )
                    del graph
                except Exception as exc:  # noqa: BLE001 - recorded as a measurement failure
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=None,
                            correct=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
        return tuple(measurements)


class _DsaIndexerMergeBenchmarkFactory:
    def __call__(self, group_id, cases, context):
        del group_id, cases
        return _DsaIndexerMergeSession(context)


class DsaIndexerMergeGenerator(DiscreteSweepGenerator):
    """Race the fused paged indexer's cross-CTA merge arms per decode shape."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=DSA_INDEXER,
            query_schema_version=1,
            config_schema_version=1,
            query_fields=_DSA_INDEXER_QUERY_FIELDS,
            range_fields=frozenset({"max_q_rows"}),
            cases=_dsa_indexer_merge_cases() if cases is None else cases,
            benchmark_factory=_DsaIndexerMergeBenchmarkFactory(),
            coverage={"fused_merge_candidates": ["cooperative", "serial"]},
        )


# Name of the planner leaf that carries the qualified production config, shared
# with the single-leaf planners that qualification-only generators emit.
_QUALIFIED_LEAF_NAME = "measured-production-implementation"


def _with_default_leaf(node: DecisionNode, default: ProfileLeaf) -> DecisionNode:
    """Attach ``default`` to every branching node of a raced planner.

    A lookup consults a node's default only when none of that node's branch
    values match, so a default at the root alone would not cover a query that
    matches the root but misses deeper (for example a row count between two
    raced anchors). Attaching the same leaf at every level makes every query
    outside the raced points resolve to the qualified production config.
    """
    if isinstance(node, ProfileLeaf):
        return node
    if node.default is not None:
        raise ValueError("raced planner already carries a default leaf")
    return dataclasses.replace(
        node,
        branches=tuple(
            (value, _with_default_leaf(child, default)) for value, child in node.branches
        ),
        default=default,
    )


class DsaIndexerProfileGenerator:
    """Profile generator for the DSA indexer component.

    Qualifies every production indexer path through
    :class:`~b12x.policy.generation.providers.qualification.DsaIndexerGenerator`
    and races the fused route's cross-CTA merge arm through
    :class:`DsaIndexerMergeGenerator`. The emitted planner pins the raced
    ``fused_merge`` winner at each raced decode query point and carries the
    qualified production config as its default leaf, so every other query the
    profile receives resolves preplanned to the qualified config exactly as a
    single-leaf qualification planner would.
    """

    def __init__(self) -> None:
        from .qualification import DsaIndexerGenerator

        self._qualification = DsaIndexerGenerator()
        self._race = DsaIndexerMergeGenerator()
        self.component_id = DSA_INDEXER
        self.query_schema_version = self._qualification.query_schema_version
        self.config_schema_version = self._qualification.config_schema_version
        if (
            self._race.query_schema_version != self.query_schema_version
            or self._race.config_schema_version != self.config_schema_version
        ):
            raise ValueError("DSA indexer qualification and race schemas differ")

    def reviewed_queries(self):
        return self._qualification.reviewed_queries()

    def estimate(self, context: GenerationContext) -> WorkEstimate:
        qualification = self._qualification.estimate(context)
        race = self._race.estimate(context)
        return WorkEstimate(
            component_id=self.component_id,
            work_units=qualification.work_units + race.work_units,
            case_count=qualification.case_count + race.case_count,
            description=(
                f"{qualification.description}; fused merge-arm race over "
                f"{race.case_count} decode cases"
            ),
            dimensions={
                "qualification": qualification.dimensions,
                "fused_merge_race": race.dimensions,
            },
        )

    def generate(
        self,
        context: GenerationContext,
        *,
        progress,
        checkpoints,
    ) -> ComponentGenerationResult:
        qualification = self._qualification.qualify(
            context, progress=progress, checkpoints=checkpoints
        )
        race = self._race.race(context, progress=progress, checkpoints=checkpoints)
        records = tuple(race.records)
        if not records:
            raise ValueError("the fused merge race produced no decision records")
        default_leaf = ProfileLeaf.create(
            name=_QUALIFIED_LEAF_NAME,
            config=qualification.encoded_config,
        )
        coverage = dict(race.coverage)
        coverage.update(
            {
                "qualified_runtime_queries": len(self._qualification.reviewed_queries()),
                "raced_query_points": len(records),
                "default": _QUALIFIED_LEAF_NAME,
            }
        )
        return ComponentGenerationResult(
            component={
                "component_id": self.component_id,
                "query_schema_version": self.query_schema_version,
                "config_schema_version": self.config_schema_version,
                "coverage": coverage,
                "planner": decision_node_to_dict(
                    _with_default_leaf(self._race.build_planner(records), default_leaf)
                ),
            },
            evidence={
                "selection": "production_qualification_plus_fused_merge_race",
                "gpu_measurement_cases": (
                    int(qualification.evidence["gpu_measurement_cases"])
                    + int(race.evidence["gpu_measurement_cases"])
                ),
                "qualification": qualification.evidence,
                "fused_merge_race": race.evidence,
            },
            completed_work_units=(
                qualification.completed_work_units + race.completed_work_units
            ),
        )


__all__ = [
    "DsaIndexerMergeGenerator",
    "DsaIndexerProfileGenerator",
    "Nvfp4QuantizationGenerator",
    "VarlenAttentionGenerator",
]
