"""Measured qualification providers for single-implementation components."""

from __future__ import annotations

import gc
import io
import json
import math
from contextlib import redirect_stdout

from b12x.policy.generation.contracts import GenerationContext
from b12x.policy.generation.attention_corpus import (
    COMMON_PREFILL_TOKEN_CAPACITIES,
    COMMON_SEQUENCE_CAPACITIES,
)
from b12x.policy.generation.measured import (
    GpuProbeMeasurement,
    MeasuredPolicyGenerator,
)

from .gpu_workers import (
    _cuda_event_samples_us,
    _l2_flush_fn,
    _median_of_group_medians,
)


def _timed_graph_measurement(
    *,
    context: GenerationContext,
    label: str,
    run,
    output,
    expected,
    flush,
) -> GpuProbeMeasurement:
    import torch
    import torch.nn.functional as torch_functional

    device = torch.device("cuda", context.device_ordinal)
    settings = context.settings
    for _ in range(settings.warmup):
        run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    if output.is_floating_point():
        output.fill_(float("nan"))
    else:
        output.zero_()
    graph.replay()
    torch.cuda.synchronize(device)
    finite = bool(torch.isfinite(output.float()).all().item())
    nonzero = bool(torch.count_nonzero(output).item())
    cosine = float(
        torch_functional.cosine_similarity(
            output.float().reshape(1, -1),
            expected.float().reshape(1, -1),
        ).item()
    )
    allocated_before = torch.cuda.memory_allocated(device)
    samples = _cuda_event_samples_us(
        graph.replay,
        count=settings.groups * settings.repetitions,
        device=device,
        flush=flush,
    )
    allocated_after = torch.cuda.memory_allocated(device)
    return GpuProbeMeasurement(
        label=label,
        latency_us=_median_of_group_medians(
            samples,
            groups=settings.groups,
            repetitions=settings.repetitions,
        ),
        correct=(
            finite
            and nonzero
            and cosine >= settings.minimum_cosine
            and allocated_after <= allocated_before
        ),
        metrics={
            "cosine": cosine,
            "finite": finite,
            "nonzero": nonzero,
            "replay_allocation_bytes": allocated_after - allocated_before,
        },
    )


def _timed_exact_graph_measurement(
    *,
    context: GenerationContext,
    label: str,
    run,
    output,
    expected,
    flush,
) -> GpuProbeMeasurement:
    import torch

    device = torch.device("cuda", context.device_ordinal)
    settings = context.settings
    for _ in range(settings.warmup):
        run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    output.fill_(-2)
    graph.replay()
    torch.cuda.synchronize(device)
    exact = bool(torch.equal(output, expected))
    nonzero = bool(torch.count_nonzero(output).item())
    allocated_before = torch.cuda.memory_allocated(device)
    samples = _cuda_event_samples_us(
        graph.replay,
        count=settings.groups * settings.repetitions,
        device=device,
        flush=flush,
    )
    allocated_after = torch.cuda.memory_allocated(device)
    return GpuProbeMeasurement(
        label=label,
        latency_us=_median_of_group_medians(
            samples,
            groups=settings.groups,
            repetitions=settings.repetitions,
        ),
        correct=exact and nonzero and allocated_after <= allocated_before,
        metrics={
            "exact": exact,
            "nonzero": nonzero,
            "replay_allocation_bytes": allocated_after - allocated_before,
        },
    )


class _DsaIndexerProbe:
    _CASES = (
        ("glm52-decode-spec4", "decode", 4, 4_096, 32, 2_048),
        ("glm53-pooled-spec6", "decode", 6, 4_096, 32, 512),
        ("glm52-decode-bucket16", "decode", 16, 4_096, 32, 2_048),
        ("glm52-extend", "extend", 1, 4_096, 32, 2_048),
        *(
            (
                f"glm52-extend-m{tokens}",
                "extend",
                tokens // 128,
                16_384,
                32,
                2_048,
            )
            for tokens in COMMON_PREFILL_TOKEN_CAPACITIES
        ),
    )
    _MSA_CASES = (
        ("minimax-m3-msa-decode", "decode", 4, 4, 8_192),
        ("minimax-m3-msa-prefill", "prefill", 16, 4, 8_192),
        *(
            (f"minimax-m3-msa-prefill-m{tokens}", "prefill", tokens, 4, 8_192)
            for tokens in COMMON_PREFILL_TOKEN_CAPACITIES
        ),
    )

    @property
    def case_count(self) -> int:
        return len(self._CASES) + len(self._MSA_CASES)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case[0] for case in (*self._CASES, *self._MSA_CASES))

    @property
    def description(self) -> str:
        return "production DSA and MSA indexer qualification"

    def __call__(
        self,
        context: GenerationContext,
    ) -> tuple[GpuProbeMeasurement, ...]:
        import torch

        from benchmarks.benchmark_dsa_indexer import (
            GLMNSAConfig,
            _run_decode_case,
            _run_extend_case,
        )
        from benchmarks.benchmark_msa_indexer import (
            _make_decode_case,
            _make_prefill_case,
        )
        from b12x.attention.dsa_indexer._impl import (
            msa_q2k_indices_decode,
            msa_q2k_indices_prefill,
        )
        from b12x.attention.dsa_indexer.msa_reference import (
            MSA_BLOCK_TOKENS,
            MSA_TOPK_BLOCKS,
            msa_q2k_indices_reference,
        )

        device = torch.device("cuda", context.device_ordinal)
        settings = context.settings
        flush = _l2_flush_fn(device, enabled=settings.cold_l2)
        replays = settings.groups * settings.repetitions
        measurements = []
        for index, (label, mode, rows, cache_len, heads, top_k) in enumerate(
            self._CASES
        ):
            query_rows = rows if mode == "decode" else rows * 128
            cfg = GLMNSAConfig(num_heads=heads)
            captured = io.StringIO()
            with redirect_stdout(captured):
                if mode == "decode":
                    _run_decode_case(
                        cfg=cfg,
                        q_rows=rows,
                        cache_len=cache_len,
                        width=cache_len,
                        topk=top_k,
                        warmup=settings.warmup,
                        replays=replays,
                        seed=settings.seed + 17 * index,
                        device=device,
                        pool_factor=2,
                        l2_flush=flush,
                    )
                else:
                    _run_extend_case(
                        cfg=cfg,
                        batch=rows,
                        q_len=128,
                        cache_len=cache_len,
                        width=cache_len,
                        topk=top_k,
                        warmup=settings.warmup,
                        replays=replays,
                        seed=settings.seed + 17 * index,
                        device=device,
                        pool_factor=2,
                        l2_flush=flush,
                    )
            records = [
                json.loads(line)
                for line in captured.getvalue().splitlines()
                if line.strip().startswith("{")
            ]
            if len(records) != 1:
                raise RuntimeError("DSA benchmark did not emit one timing record")
            record = records[0]
            latency = (
                record["replay_median_us"]
                if mode == "decode"
                else record["median_us"]
            )
            measurements.append(
                GpuProbeMeasurement(
                    label=label,
                    latency_us=float(latency),
                    correct=True,
                    metrics={
                        "mode": mode,
                        "query_rows": query_rows,
                        "num_heads": heads,
                        "top_k": top_k,
                    },
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
        first_msa_index = len(self._CASES)
        for offset, (label, mode, rows, heads, width) in enumerate(
            self._MSA_CASES
        ):
            output = torch.empty(
                (heads, rows, MSA_TOPK_BLOCKS),
                dtype=torch.int32,
                device=device,
            )
            seed = settings.seed + 17 * (first_msa_index + offset)
            if mode == "decode":
                q_fp8, q_scale, index_k_cache, metadata = _make_decode_case(
                    rows=rows,
                    heads=heads,
                    ctx_tokens=width,
                    seed=seed,
                    device=device,
                )
                expected = msa_q2k_indices_reference(
                    q_fp8=q_fp8,
                    q_scale=q_scale,
                    index_k_cache=index_k_cache,
                    real_page_table=metadata.real_page_table,
                    cache_seqlens_int32=metadata.cache_seqlens_int32,
                    query_positions=metadata.cache_seqlens_int32 - 1,
                )

                def run_msa_decode() -> None:
                    msa_q2k_indices_decode(
                        q_fp8=q_fp8,
                        q_scale=q_scale,
                        index_k_cache=index_k_cache,
                        metadata=metadata,
                        out_indices=output,
                    )

                run = run_msa_decode
            else:
                q_fp8, q_scale, kv_fp8, metadata = _make_prefill_case(
                    rows=rows,
                    heads=heads,
                    k_rows=width,
                    seed=seed,
                    device=device,
                )
                expected = msa_q2k_indices_reference(
                    q_fp8=q_fp8,
                    q_scale=q_scale,
                    kv_fp8=kv_fp8,
                    k_start=metadata.k_start,
                    k_end=metadata.k_end,
                    query_positions=metadata.k_end - 1,
                    block_base=torch.div(
                        metadata.k_start,
                        MSA_BLOCK_TOKENS,
                        rounding_mode="floor",
                    ),
                )

                def run_msa_prefill() -> None:
                    msa_q2k_indices_prefill(
                        q_fp8=q_fp8,
                        q_scale=q_scale,
                        kv_fp8=kv_fp8,
                        metadata=metadata,
                        out_indices=output,
                    )

                run = run_msa_prefill
            measurements.append(
                _timed_exact_graph_measurement(
                    context=context,
                    label=label,
                    run=run,
                    output=output,
                    expected=expected,
                    flush=flush,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
        return tuple(measurements)


class _SparseMlaProbe:
    _GLM52_CASES = (
        ("glm52-tp1-decode", 1, 64, 512),
        ("glm52-tp8-spec4", 4, 8, 2_048),
    )
    _GLM53_CASES = (
        ("glm53-tp1-decode", 1, 64, 512),
        ("glm53-tp4-spec6", 6, 16, 2_051),
    )

    @property
    def case_count(self) -> int:
        return len(self._GLM52_CASES) + len(self._GLM53_CASES)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(
            case[0] for case in (*self._GLM52_CASES, *self._GLM53_CASES)
        )

    @property
    def description(self) -> str:
        return "production sparse-MLA plan/bind/decode graph qualification"

    def __call__(
        self,
        context: GenerationContext,
    ) -> tuple[GpuProbeMeasurement, ...]:
        import torch

        from b12x.attention import sparse_mla
        from b12x.attention._shared.mla.reference import (
            pack_mla_kv_cache_reference,
        )
        from b12x.attention._shared.mla.traits import ModelType
        from b12x.attention.sparse_mla.reference import sparse_mla_reference
        from b12x.policy import PolicyContext, PolicyMode
        from benchmarks.benchmark_unified_mla_sm120 import _make_glm_inputs

        device = torch.device("cuda", context.device_ordinal)
        flush = _l2_flush_fn(device, enabled=context.settings.cold_l2)
        policy = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)
        measurements = []
        for index, (label, rows, heads, topk) in enumerate(self._GLM52_CASES):
            sm_scale = 1.0 / math.sqrt(576)
            q, kv_cache, selected, cache_seqlens, _kv_bytes = _make_glm_inputs(
                rows=rows,
                num_heads=heads,
                topk=topk,
                device=device,
                seed=context.settings.seed + 17 * index,
            )
            plan = sparse_mla.plan(
                sparse_mla.Caps(
                    device=device,
                    num_q_heads=heads,
                    max_q_rows=rows,
                    max_width=topk,
                    softmax_scale=sm_scale,
                    dtype=torch.bfloat16,
                    kv_dtype=torch.uint8,
                    head_dim=576,
                    v_head_dim=512,
                    mode="decode",
                    max_batch=rows,
                    max_kv_rows=int(kv_cache.shape[0]),
                    max_chunks_per_row=max(1, math.ceil(topk / 64)),
                    page_size=1,
                ),
                policy=policy,
            )
            (scratch_spec,) = plan.scratch_specs()
            scratch = torch.empty(
                scratch_spec.shape,
                dtype=scratch_spec.dtype,
                device=scratch_spec.device,
            )
            binding = sparse_mla.bind(
                plan,
                scratch=scratch,
                q=q,
                kv_cache=kv_cache,
                selected_indices=selected,
                cache_lengths=cache_seqlens,
                selected_lengths=cache_seqlens,
            )
            expected = sparse_mla_reference(
                q_all=q,
                kv_cache=kv_cache,
                page_table_1=selected,
                active_token_counts=cache_seqlens,
                sm_scale=sm_scale,
                v_head_dim=512,
            )

            def run():
                return sparse_mla.run(binding)

            output = run()
            torch.cuda.synchronize(device)
            measurements.append(
                _timed_graph_measurement(
                    context=context,
                    label=label,
                    run=run,
                    output=output,
                    expected=expected,
                    flush=flush,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()

        first_glm53_index = len(self._GLM52_CASES)
        for offset, (label, rows, heads, width) in enumerate(self._GLM53_CASES):
            index = first_glm53_index + offset
            generator = torch.Generator(device="cpu").manual_seed(
                context.settings.seed + 17 * index
            )
            page_size = 256
            num_records = math.ceil(width / page_size) * page_size
            latent = (
                torch.randn(
                    (num_records, 512),
                    generator=generator,
                    dtype=torch.float32,
                )
                .div_(4.0)
                .to(device=device, dtype=torch.bfloat16)
            )
            unpacked_cache = latent.view(num_records, 1, 512)
            kv_cache = pack_mla_kv_cache_reference(unpacked_cache).view(
                num_records // page_size,
                page_size,
                528,
            )
            q = (
                torch.randn(
                    (rows, heads, 512),
                    generator=generator,
                    dtype=torch.float32,
                )
                .div_(4.0)
                .to(device=device, dtype=torch.bfloat16)
            )
            selected = torch.randint(
                0,
                num_records,
                (rows, width),
                generator=generator,
                dtype=torch.int32,
            ).to(device)
            active = torch.full((rows,), width, dtype=torch.int32, device=device)
            cache_seqlens = torch.full(
                (rows,), num_records, dtype=torch.int32, device=device
            )
            sm_scale = 256**-0.5
            plan = sparse_mla.plan(
                sparse_mla.Caps(
                    device=device,
                    num_q_heads=heads,
                    max_q_rows=rows,
                    max_width=width,
                    softmax_scale=sm_scale,
                    dtype=torch.bfloat16,
                    kv_dtype=torch.uint8,
                    head_dim=512,
                    v_head_dim=512,
                    mode="decode",
                    max_batch=rows,
                    max_kv_rows=num_records,
                    max_chunks_per_row=math.ceil(width / 64),
                    page_size=page_size,
                    model_type=ModelType.GLM_NEXT,
                ),
                policy=policy,
            )
            (scratch_spec,) = plan.scratch_specs()
            binding = sparse_mla.bind(
                plan,
                scratch=torch.empty(
                    scratch_spec.shape,
                    dtype=scratch_spec.dtype,
                    device=scratch_spec.device,
                ),
                q=q,
                kv_cache=kv_cache,
                selected_indices=selected,
                cache_lengths=cache_seqlens,
                selected_lengths=active,
            )
            expected = sparse_mla_reference(
                q_all=q,
                kv_cache=kv_cache.view(num_records, 1, 528),
                page_table_1=selected,
                active_token_counts=active,
                sm_scale=sm_scale,
                v_head_dim=512,
            )

            def run_glm53():
                return sparse_mla.run(binding)

            output = run_glm53()
            torch.cuda.synchronize(device)
            measurements.append(
                _timed_graph_measurement(
                    context=context,
                    label=label,
                    run=run_glm53,
                    output=output,
                    expected=expected,
                    flush=flush,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
        return tuple(measurements)


class _EpMoeProbe:
    _CASES = ((4, 4), (16, 8))

    @property
    def case_count(self) -> int:
        return len(self._CASES)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(
            f"m{tokens}-topk{top_k}" for tokens, top_k in self._CASES
        )

    @property
    def description(self) -> str:
        return "production W4A16 EP-MoE graph qualification against fused MoE"

    def __call__(
        self,
        context: GenerationContext,
    ) -> tuple[GpuProbeMeasurement, ...]:
        import torch

        from b12x.moe import ep_moe, fused_moe
        from b12x.policy import PolicyContext, PolicyMode
        from b12x.policy.generation.moe_corpus import (
            MOE_RECIPES,
            MoePhysicalGeometry,
        )

        from .moe_gpu_worker import _packed_weights

        device = torch.device("cuda", context.device_ordinal)
        recipe = next(
            item for item in MOE_RECIPES if item.recipe_id == "modelopt-w4a16"
        )
        geometry = MoePhysicalGeometry(
            recipe=recipe,
            activation="silu",
            num_experts=8,
            hidden_size=2_560,
            intermediate_size=640,
            aliases=(),
        )
        experts = _packed_weights(geometry, device=device)
        expert_map_tensor = torch.full(
            (512,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        expert_map_tensor[:8] = torch.arange(8, dtype=torch.int32, device=device)
        expert_map = ep_moe.prepare_expert_map(
            expert_map_tensor,
            local_num_experts=8,
            global_num_experts=512,
            device=device,
        )
        policy = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)
        flush = _l2_flush_fn(device, enabled=context.settings.cold_l2)
        measurements = []
        for index, (tokens, top_k) in enumerate(self._CASES):
            generator = torch.Generator(device=device).manual_seed(
                context.settings.seed + 17 * index
            )
            activations = torch.randn(
                (tokens, 2_560),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            topk_ids = (
                torch.arange(top_k, dtype=torch.int32, device=device)
                .view(1, -1)
                .expand(tokens, -1)
                .contiguous()
            )
            topk_weights = torch.softmax(
                torch.randn(
                    (tokens, top_k),
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                ),
                dim=-1,
            ).contiguous()
            fused_plan = fused_moe.plan_execution(
                experts=experts,
                capacity=fused_moe.ExecutionCapacity(
                    max_tokens=tokens,
                    top_k=top_k,
                ),
                policy=policy,
            )
            fused_moe.prewarm(fused_plan)
            fused_scratch = {
                spec.name: torch.empty(
                    spec.shape,
                    dtype=spec.dtype,
                    device=spec.device,
                )
                for spec in fused_plan.scratch_specs()
            }
            fused_binding = fused_moe.bind(
                fused_plan,
                scratch=fused_scratch,
                a=activations,
                experts=experts,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output=torch.empty_like(activations),
            )
            expected = fused_moe.run(binding=fused_binding).clone()
            ep_plan = ep_moe.plan(
                ep_moe.Caps(
                    max_tokens=tokens,
                    num_topk=top_k,
                    global_num_experts=512,
                    device=device,
                    weight_plan=experts.plan._impl,
                ),
                policy=policy,
            )
            (scratch_spec,) = ep_plan.scratch_specs()
            scratch = torch.empty(
                scratch_spec.shape,
                dtype=scratch_spec.dtype,
                device=scratch_spec.device,
            )
            output = torch.empty_like(activations)
            binding = ep_moe.bind(
                ep_plan,
                scratch=scratch,
                a=activations,
                experts=experts._impl,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=expert_map,
                output=output,
            )

            def run() -> None:
                ep_moe.run(binding=binding)

            measurements.append(
                _timed_graph_measurement(
                    context=context,
                    label=f"m{tokens}-topk{top_k}",
                    run=run,
                    output=output,
                    expected=expected,
                    flush=flush,
                )
            )
        return tuple(measurements)


class DsaIndexerGenerator(MeasuredPolicyGenerator):
    """Generate a measured policy for DSA indexer production paths."""

    def __init__(self) -> None:
        from b12x.attention.dsa_indexer._policy import (
            DSA_INDEXER_POLICY,
            DsaIndexerQuery,
        )

        queries = (
            DsaIndexerQuery(
                source_layout="paged",
                mode="decode",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=32,
                num_idx_heads=1,
                max_q_rows=4,
                max_k_rows=0,
                top_k=2_048,
                page_size=64,
                score_mode="dsa",
                shared_page_table=False,
            ),
            DsaIndexerQuery(
                source_layout="paged",
                mode="decode",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=32,
                num_idx_heads=1,
                max_q_rows=6,
                max_k_rows=0,
                top_k=512,
                page_size=64,
                score_mode="dsa",
                shared_page_table=False,
            ),
            DsaIndexerQuery(
                source_layout="contiguous",
                mode="prefill",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=32,
                num_idx_heads=1,
                max_q_rows=4_096,
                max_k_rows=16_384,
                top_k=2_048,
                page_size=64,
                score_mode="dsa",
                shared_page_table=False,
            ),
            DsaIndexerQuery(
                source_layout="contiguous",
                mode="prefill",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=32,
                num_idx_heads=1,
                max_q_rows=16_384,
                max_k_rows=131_072,
                top_k=2_048,
                page_size=64,
                score_mode="dsa",
                shared_page_table=False,
            ),
            DsaIndexerQuery(
                source_layout="paged",
                mode="decode",
                dtype="float8_e4m3fn",
                kv_dtype="uint8",
                num_q_heads=1,
                num_idx_heads=4,
                max_q_rows=4,
                max_k_rows=0,
                top_k=16,
                page_size=64,
                score_mode="msa",
                shared_page_table=False,
            ),
            DsaIndexerQuery(
                source_layout="contiguous",
                mode="prefill",
                dtype="float8_e4m3fn",
                kv_dtype="uint8",
                num_q_heads=1,
                num_idx_heads=4,
                max_q_rows=16,
                max_k_rows=8_192,
                top_k=16,
                page_size=64,
                score_mode="msa",
                shared_page_table=False,
            ),
        )
        queries += tuple(
            DsaIndexerQuery(
                source_layout="contiguous",
                mode="prefill",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=32,
                num_idx_heads=1,
                max_q_rows=rows,
                max_k_rows=16_384,
                top_k=2_048,
                page_size=64,
                score_mode="dsa",
                shared_page_table=False,
            )
            for rows in COMMON_PREFILL_TOKEN_CAPACITIES
            if rows != 4_096
        )
        queries += tuple(
            DsaIndexerQuery(
                source_layout="contiguous",
                mode="prefill",
                dtype="float8_e4m3fn",
                kv_dtype="uint8",
                num_q_heads=1,
                num_idx_heads=4,
                max_q_rows=rows,
                max_k_rows=8_192,
                top_k=16,
                page_size=64,
                score_mode="msa",
                shared_page_table=False,
            )
            for rows in COMMON_PREFILL_TOKEN_CAPACITIES
        )
        queries += tuple(
            DsaIndexerQuery(
                source_layout="paged",
                mode="decode",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=heads,
                num_idx_heads=1,
                max_q_rows=1,
                max_k_rows=0,
                top_k=512,
                page_size=64,
                score_mode="dsa",
                shared_page_table=False,
            )
            for heads in (64, 16, 8)
        )
        super().__init__(
            policy=DSA_INDEXER_POLICY,
            queries=queries,
            encode_config=lambda config: config.to_dict(),
            probe=_DsaIndexerProbe(),
        )


class SparseMlaGenerator(MeasuredPolicyGenerator):
    """Generate a measured policy for sparse MLA."""

    def __init__(self) -> None:
        from b12x.attention.sparse_mla._policy import (
            SPARSE_MLA_POLICY,
            SparseMlaQuery,
        )

        from b12x.attention._shared.mla.traits import ModelType

        glm52_queries = tuple(
            SparseMlaQuery(
                mode="decode",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=heads,
                qk_head_dim=576,
                v_head_dim=512,
                max_q_rows=rows,
                max_width=2_048,
                page_size=64,
                model_type=None,
                head_major_output=False,
            )
            for heads in (64, 32, 16, 8)
            for rows in COMMON_SEQUENCE_CAPACITIES
        )
        glm53_queries = tuple(
            SparseMlaQuery(
                mode="decode",
                dtype="bfloat16",
                kv_dtype="uint8",
                num_q_heads=heads,
                qk_head_dim=512,
                v_head_dim=512,
                max_q_rows=rows,
                max_width=2_051,
                page_size=256,
                model_type=ModelType.GLM_NEXT,
                head_major_output=False,
            )
            for heads in (64, 32, 16, 8)
            for rows in COMMON_SEQUENCE_CAPACITIES
        )
        super().__init__(
            policy=SPARSE_MLA_POLICY,
            queries=(*glm52_queries, *glm53_queries),
            encode_config=lambda config: config.to_dict(),
            probe=_SparseMlaProbe(),
        )


class EpMoeGenerator(MeasuredPolicyGenerator):
    """Generate a measured policy for replicated-input EP MoE."""

    def __init__(self) -> None:
        from b12x.moe.ep_moe._policy import EP_MOE_POLICY, EpMoeQuery

        queries = tuple(
            EpMoeQuery(
                max_tokens=tokens,
                top_k=top_k,
                num_experts=512,
                hidden_size=2_560,
                intermediate_size=640,
                activation="silu",
            )
            for tokens in (4, 16)
            for top_k in (4, 8)
        )
        super().__init__(
            policy=EP_MOE_POLICY,
            queries=queries,
            encode_config=lambda config: config.to_dict(),
            probe=_EpMoeProbe(),
        )


__all__ = [
    "DsaIndexerGenerator",
    "EpMoeGenerator",
    "SparseMlaGenerator",
]
