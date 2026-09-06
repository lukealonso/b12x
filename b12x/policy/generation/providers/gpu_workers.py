"""Production GPU measurement workers for built-in component generators."""

from __future__ import annotations

import gc
import statistics
from contextlib import AbstractContextManager
from dataclasses import dataclass

from b12x.policy.generation.contracts import GenerationContext
from b12x.policy.generation.sweep import (
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
)


@dataclass(frozen=True, kw_only=True)
class _GdnBuffers:
    binding: object
    initial_state: object
    decay_recipe: str


def _gdn_random_tensor(
    shape: tuple[int, ...],
    *,
    device: object,
    generator: object,
    dtype: object,
    scale: float = 0.25,
):
    import torch

    return (
        torch.randn(shape, generator=generator, dtype=torch.float32)
        .mul_(scale)
        .to(device=device, dtype=dtype)
        .contiguous()
    )


def _build_gdn_buffers(
    case: SweepCase,
    *,
    device: object,
    seed: int,
) -> _GdnBuffers:
    import torch

    from b12x.policy import PolicyContext, PolicyMode
    from b12x.sequence import gdn_decode as gdn

    raw_lengths = case.metadata["query_lengths"]
    if not isinstance(raw_lengths, tuple):
        raise TypeError("GDN query_lengths metadata must be an array")
    query_lengths = tuple(int(value) for value in raw_lengths)
    key_heads = int(case.query["key_heads"])
    value_heads = int(case.query["value_heads"])
    decay_recipe = str(case.metadata.get("decay_recipe", "gdn"))
    if decay_recipe == "gdn" and value_heads != 3 * key_heads:
        raise ValueError("Qwen GDN requires value_heads=3*key_heads")
    if decay_recipe == "kda" and value_heads != key_heads:
        raise ValueError("GLM KDA requires value_heads=key_heads")
    if decay_recipe not in {"gdn", "kda"}:
        raise ValueError(f"unknown GDN decay recipe {decay_recipe!r}")
    state_dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(str(case.query["state_dtype"]))
    if state_dtype is None:
        raise ValueError(f"unsupported GDN state dtype {case.query['state_dtype']!r}")

    live_tokens = sum(query_lengths)
    live_sequences = len(query_lengths)
    max_sequences = int(case.query["max_seqs"])
    columns = int(case.query["state_index_columns"])
    max_tokens = int(case.query["max_tokens"])
    if live_sequences > max_sequences or max(query_lengths) > columns:
        raise ValueError("GDN live shape exceeds the profiled capacity")
    if max_tokens != max_sequences * columns:
        raise ValueError("GDN token capacity must equal sequences times columns")
    active_state_cells = live_sequences * columns
    state_slots = max(active_state_cells, live_tokens) + 1
    caps = gdn.Caps(
        device=device,
        max_tokens=max_tokens,
        max_seqs=max_sequences,
        max_state_slots=state_slots,
        key_heads=key_heads,
        value_heads=value_heads,
        state_index_columns=columns,
        state_dtype=state_dtype,
        gate_activation="sigmoid",
        qk_l2norm=True,
    )
    planned = gdn.plan(
        caps,
        policy=PolicyContext.for_device(
            device,
            mode=PolicyMode.HEURISTIC_ONLY,
        ),
    )
    (scratch_spec,) = planned.scratch_specs()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query_start_loc = torch.full(
        (max_sequences + 1,),
        live_tokens,
        dtype=torch.int32,
        device=device,
    )
    query_start_loc[0] = 0
    query_start_loc[1 : live_sequences + 1].copy_(
        torch.tensor(query_lengths, dtype=torch.int32, device=device).cumsum(0)
    )
    state_indices = torch.arange(
        max_sequences * columns,
        dtype=torch.int32,
        device=device,
    ).view(max_sequences, columns)
    shared_tensors = {
        "scratch": torch.empty(
            scratch_spec.shape,
            dtype=scratch_spec.dtype,
            device=device,
        ),
        "mixed_qkv": _gdn_random_tensor(
            (max_tokens, caps.packed_qkv_width),
            device=device,
            generator=generator,
            dtype=torch.bfloat16,
        ),
        "z": _gdn_random_tensor(
            (max_tokens, value_heads, 128),
            device=device,
            generator=generator,
            dtype=torch.bfloat16,
        ),
        "norm_weight": (
            1.0
            + _gdn_random_tensor(
                (128,),
                device=device,
                generator=generator,
                dtype=torch.bfloat16,
                scale=0.05,
            )
        ).contiguous(),
        "recurrent_state": _gdn_random_tensor(
            (state_slots, value_heads, 128, 128),
            device=device,
            generator=generator,
            dtype=state_dtype,
            scale=0.1,
        ),
        "query_start_loc": query_start_loc,
        "num_accepted_tokens": torch.ones(
            max_sequences,
            dtype=torch.int32,
            device=device,
        ),
        "state_indices": state_indices,
        "num_seqs": torch.tensor([live_sequences], dtype=torch.int32, device=device),
        "num_tokens": torch.tensor([live_tokens], dtype=torch.int32, device=device),
        "output": torch.empty(
            (max_tokens, value_heads, 128),
            dtype=torch.bfloat16,
            device=device,
        ),
    }
    if decay_recipe == "kda":
        binding = gdn.bind_kda(
            planned,
            **shared_tensors,
            raw_g=_gdn_random_tensor(
                (max_tokens, key_heads, 128),
                device=device,
                generator=generator,
                dtype=torch.bfloat16,
            ),
            raw_beta=_gdn_random_tensor(
                (max_tokens, key_heads),
                device=device,
                generator=generator,
                dtype=torch.bfloat16,
            ),
            A_log=_gdn_random_tensor(
                (key_heads,),
                device=device,
                generator=generator,
                dtype=torch.float32,
                scale=0.1,
            ),
            dt_bias=_gdn_random_tensor(
                (key_heads, 128),
                device=device,
                generator=generator,
                dtype=torch.float32,
                scale=0.1,
            ),
        )
    else:
        binding = gdn.bind(
            planned,
            **shared_tensors,
            a=_gdn_random_tensor(
                (max_tokens, value_heads),
                device=device,
                generator=generator,
                dtype=torch.bfloat16,
            ),
            b=_gdn_random_tensor(
                (max_tokens, value_heads),
                device=device,
                generator=generator,
                dtype=torch.bfloat16,
            ),
            A_log=_gdn_random_tensor(
                (value_heads,),
                device=device,
                generator=generator,
                dtype=torch.float32,
                scale=0.1,
            ),
            dt_bias=_gdn_random_tensor(
                (value_heads,),
                device=device,
                generator=generator,
                dtype=torch.float32,
                scale=0.1,
            ),
        )
    return _GdnBuffers(
        binding=binding,
        initial_state=binding.recurrent_state.clone(),
        decay_recipe=decay_recipe,
    )


def _gdn_reference(buffers: _GdnBuffers):
    from b12x.sequence import gdn_decode as gdn

    binding = buffers.binding
    caps = binding.plan.caps
    state = buffers.initial_state.clone()
    if buffers.decay_recipe == "kda":
        output = gdn.reference.decode_kda(
            binding.mixed_qkv,
            binding.raw_g,
            binding.raw_beta,
            binding.z,
            binding.A_log,
            binding.dt_bias,
            binding.norm_weight,
            state,
            binding.query_start_loc,
            binding.num_accepted_tokens,
            binding.state_indices,
            binding.num_seqs,
            binding.num_tokens,
            heads=caps.key_heads,
            qk_l2norm=True,
        )
    else:
        output = gdn.reference.decode(
            binding.mixed_qkv,
            binding.a,
            binding.b,
            binding.z,
            binding.A_log,
            binding.dt_bias,
            binding.norm_weight,
            state,
            binding.query_start_loc,
            binding.num_accepted_tokens,
            binding.state_indices,
            binding.num_seqs,
            binding.num_tokens,
            key_heads=caps.key_heads,
            value_heads=caps.value_heads,
            gate_activation="sigmoid",
            qk_l2norm=True,
        )
    return output, state


def _l2_flush_fn(device: object, *, enabled: bool):
    if not enabled:
        return None
    import torch

    properties = torch.cuda.get_device_properties(device)
    flush_bytes = max(2 * int(properties.L2_cache_size), 64 << 20)
    buffer = torch.ones(
        (flush_bytes + 3) // 4,
        dtype=torch.float32,
        device=device,
    )
    reduction = torch.empty((), dtype=torch.float32, device=device)

    def flush() -> None:
        torch.sum(buffer, dim=0, out=reduction)

    return flush


def _median_of_group_medians(
    samples: tuple[float, ...],
    *,
    groups: int,
    repetitions: int,
) -> float:
    expected = int(groups) * int(repetitions)
    if len(samples) != expected:
        raise ValueError(f"expected {expected} timing samples, received {len(samples)}")
    medians = [
        statistics.median(samples[start : start + repetitions])
        for start in range(0, expected, repetitions)
    ]
    return float(statistics.median(medians))


def _bounded_repetitions(settings, *, pilot_us: float) -> int:
    budget_us = float(settings.max_candidate_seconds) * 1_000_000.0
    budgeted = int(budget_us / (max(float(pilot_us), 1.0) * settings.groups))
    return max(1, min(settings.repetitions, budgeted))


def _cuda_event_samples_us(
    run,
    *,
    count: int,
    device: object,
    flush=None,
    before_each=None,
) -> tuple[float, ...]:
    import torch

    starts = tuple(torch.cuda.Event(enable_timing=True) for _ in range(count))
    ends = tuple(torch.cuda.Event(enable_timing=True) for _ in range(count))
    for start, end in zip(starts, ends, strict=True):
        if before_each is not None:
            before_each()
        if flush is not None:
            flush()
        start.record()
        run()
        end.record()
    torch.cuda.synchronize(device)
    return tuple(
        float(start.elapsed_time(end)) * 1_000.0
        for start, end in zip(starts, ends, strict=True)
    )


class _GdnSession(AbstractContextManager["_GdnSession"]):
    _CANDIDATES = {
        "gdn": (SweepCandidate.create({"backend": "cutedsl"}),),
        "kda": (SweepCandidate.create({"backend": "triton"}),),
    }

    def __init__(self, context: GenerationContext) -> None:
        self._context = context

    def __enter__(self) -> "_GdnSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        try:
            import torch

            torch.cuda.synchronize(self._context.device_ordinal)
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must preserve the root error
            pass
        return None

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        return self._CANDIDATES[str(case.metadata.get("decay_recipe", "gdn"))]

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch

        from b12x.sequence import gdn_decode as gdn

        expected_candidates = self.candidates(case)
        if candidates != expected_candidates:
            raise ValueError("GDN worker received an unknown candidate set")
        settings = self._context.settings
        device = torch.device("cuda", self._context.device_ordinal)
        with torch.cuda.device(self._context.device_ordinal):
            buffers = _build_gdn_buffers(
                case,
                device=device,
                seed=settings.seed,
            )
            expected_output, expected_state = _gdn_reference(buffers)
            binding = buffers.binding

            def restore() -> None:
                binding.recurrent_state.copy_(buffers.initial_state)

            def run() -> None:
                if buffers.decay_recipe == "kda":
                    gdn.run_kda(binding)
                else:
                    gdn.run(binding)

            for _ in range(settings.warmup):
                restore()
                run()
            torch.cuda.synchronize(device)
            restore()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            torch.cuda.synchronize(device)

            restore()
            binding.output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            actual_output = binding.output
            actual_state = binding.recurrent_state
            finite = bool(torch.isfinite(actual_output).all().item())
            output_nonzero = int(torch.count_nonzero(actual_output).item())
            torch.testing.assert_close(
                actual_output,
                expected_output,
                rtol=1e-2,
                atol=2e-2,
            )
            state_rtol = 1e-2 if actual_state.dtype == torch.bfloat16 else 1e-5
            state_atol = 8e-3 if actual_state.dtype == torch.bfloat16 else 2e-5
            torch.testing.assert_close(
                actual_state,
                expected_state,
                rtol=state_rtol,
                atol=state_atol,
            )
            output_max_abs = float(
                (actual_output.float() - expected_output.float()).abs().max()
            )
            state_max_abs = float(
                (actual_state.float() - expected_state.float()).abs().max()
            )
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    actual_output.float().flatten(),
                    expected_output.float().flatten(),
                    dim=0,
                )
            )
            flush = _l2_flush_fn(device, enabled=settings.cold_l2)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            restore()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            timed_repetitions = _bounded_repetitions(
                settings,
                pilot_us=float(start.elapsed_time(end)) * 1_000.0,
            )
            gc.collect()
            allocated_before = torch.cuda.memory_allocated(device)
            samples_us = _cuda_event_samples_us(
                graph.replay,
                count=settings.groups * timed_repetitions,
                device=device,
                flush=flush,
                before_each=restore,
            )
            allocated_after = torch.cuda.memory_allocated(device)
        latency_us = _median_of_group_medians(
            tuple(samples_us),
            groups=settings.groups,
            repetitions=timed_repetitions,
        )
        return (
            SweepMeasurement(
                candidate=candidates[0],
                latency_us=latency_us,
                correct=(
                    finite
                    and output_nonzero > 0
                    and cosine >= 0.999
                    and allocated_after <= allocated_before
                ),
                metrics={
                    "output_max_abs": output_max_abs,
                    "state_max_abs": state_max_abs,
                    "output_cosine": cosine,
                    "output_nonzero": output_nonzero,
                    "stable_addresses": True,
                    "replay_allocation_bytes": allocated_after - allocated_before,
                },
            ),
        )


class GdnBenchmarkFactory:
    """Measure GDN cases independently within each corpus group."""

    def __call__(self, group_id, cases, context):
        del group_id, cases
        return _GdnSession(context)


def _torch_dtype(name: str):
    import torch

    try:
        return {
            "bfloat16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
        }[name]
    except KeyError as exc:
        raise ValueError(f"unsupported profile dtype {name!r}") from exc


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


class _GqaSession(AbstractContextManager["_GqaSession"]):
    def __init__(
        self,
        cases: tuple[SweepCase, ...],
        context: GenerationContext,
    ) -> None:
        self._cases = cases
        self._context = context
        page_sizes = {int(case.query["page_size"]) for case in cases}
        if len(page_sizes) != 1:
            raise ValueError("GQA allocation groups must use one page size")
        self._page_size = next(iter(page_sizes))
        maximum_context = max(int(case.query["cache_tokens"]) for case in cases)
        self._capture_page_count = _ceil_div(maximum_context, self._page_size)
        self._candidate_cache: dict[str, tuple[SweepCandidate, ...]] = {}
        self._device = None
        self._flush = None

    def __enter__(self) -> "_GqaSession":
        import torch

        self._device = torch.device("cuda", self._context.device_ordinal)
        self._flush = _l2_flush_fn(
            self._device,
            enabled=self._context.settings.cold_l2,
        )
        return self

    def __exit__(self, *_exc: object) -> None:
        try:
            import torch

            gc.collect()
            torch.cuda.synchronize(self._context.device_ordinal)
            self._flush = None
            self._device = None
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must preserve the root error
            pass
        return None

    def _capacity(
        self,
        case: SweepCase,
        *,
        graph_ctas_per_sm: int | None = None,
        force_split_kv: bool | None = None,
    ):
        import torch

        from b12x.attention import paged
        from b12x.policy import PolicyContext, PolicyMode

        query = case.query
        return paged.decode_graph_capacity(
            device=torch.device("cuda", self._context.device_ordinal),
            q_dtype=_torch_dtype(str(query["q_dtype"])),
            kv_dtype=_torch_dtype(str(query["kv_dtype"])),
            num_q_heads=int(query["q_heads"]),
            num_kv_heads=int(query["kv_heads"]),
            head_dim_qk=int(query["head_dim_qk"]),
            head_dim_vo=int(query["head_dim_vo"]),
            page_size=self._page_size,
            batch=int(query["batch_size"]),
            max_cache_page_count=self._capture_page_count,
            window_left=int(query["window_left"]),
            graph_ctas_per_sm=graph_ctas_per_sm,
            force_split_kv=force_split_kv,
            kv_cache_layout=str(query["kv_cache_layout"]),
            policy=PolicyContext.for_device(
                torch.device("cuda", self._context.device_ordinal),
                mode=PolicyMode.HEURISTIC_ONLY,
            ),
        )

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        cached = self._candidate_cache.get(case.case_id)
        if cached is not None:
            return cached
        from b12x.attention.paged._policy import GqaConfig

        query = case.query
        requested_ctas = query["requested_graph_ctas_per_sm"]
        graph_options = (
            (int(requested_ctas),)
            if requested_ctas is not None
            else (None, 1, 2, 3, 4, 6)
        )
        requested_split = query["force_split_kv"]
        split_options = (
            (bool(requested_split),)
            if requested_split is not None
            else (None, False, True)
        )
        candidates_by_id: dict[str, SweepCandidate] = {}
        for split in split_options:
            for graph_ctas in graph_options:
                capacity = self._capacity(
                    case,
                    graph_ctas_per_sm=graph_ctas,
                    force_split_kv=split,
                )
                candidate = SweepCandidate.create(
                    GqaConfig.from_capacity(capacity).profile_dict()
                )
                candidates_by_id.setdefault(candidate.candidate_id, candidate)
        candidates = tuple(candidates_by_id.values())
        self._candidate_cache[case.case_id] = candidates
        return candidates

    def _make_cache(
        self,
        *,
        pages: int,
        page_size: int,
        kv_heads: int,
        head_dim: int,
        batch: int,
        kv_dtype: object,
        combined: bool,
        device: object,
        generator: object,
    ):
        import torch

        shape = (pages, page_size, kv_heads, head_dim)
        if kv_dtype == torch.bfloat16:
            if combined:
                storage = torch.randn(
                    (pages, 2, page_size, kv_heads, head_dim),
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                ).mul_(0.25)
                return storage[:, 0], storage[:, 1], None, None
            k_cache = torch.randn(
                shape,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            v_cache = torch.randn(
                shape,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            return k_cache, v_cache, None, None

        float_storage = torch.randn(
            (pages, 2, page_size, kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).mul_(0.25)
        finfo = torch.finfo(torch.float8_e4m3fn)
        scales = (
            float_storage.float().abs().amax(dim=(0, 2, 3, 4)) / finfo.max
        ).clamp_min_(torch.finfo(torch.float32).tiny)
        quantized = (
            float_storage.float()
            .div_(scales.view(1, 2, 1, 1, 1))
            .clamp_(min=finfo.min, max=finfo.max)
            .to(torch.float8_e4m3fn)
        )
        del float_storage
        k_cache = quantized[:, 0]
        v_cache = quantized[:, 1]
        if not combined:
            k_cache = k_cache.contiguous()
            v_cache = v_cache.contiguous()
            del quantized
        k_descale = torch.full(
            (batch, kv_heads),
            float(scales[0]),
            dtype=torch.float32,
            device=device,
        )
        v_descale = torch.full(
            (batch, kv_heads),
            float(scales[1]),
            dtype=torch.float32,
            device=device,
        )
        return k_cache, v_cache, k_descale, v_descale

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch

        from b12x.attention import paged
        from b12x.attention.paged._policy import GqaConfig
        from b12x.attention.paged.reference import paged_attention_reference
        from b12x.policy import GQA_ATTENTION, PolicyContext, PolicyMode

        if candidates != self.candidates(case):
            raise ValueError("GQA worker received an unknown candidate set")
        query = case.query
        settings = self._context.settings
        device = torch.device("cuda", self._context.device_ordinal)
        q_dtype = _torch_dtype(str(query["q_dtype"]))
        kv_dtype = _torch_dtype(str(query["kv_dtype"]))
        batch = int(query["batch_size"])
        q_heads = int(query["q_heads"])
        kv_heads = int(query["kv_heads"])
        head_dim_qk = int(query["head_dim_qk"])
        head_dim_vo = int(query["head_dim_vo"])
        if head_dim_qk != head_dim_vo:
            raise ValueError("GQA GPU worker requires matching QK and V dimensions")
        head_dim = head_dim_qk
        live_cache_tokens = int(query["cache_tokens"])
        combined = str(query["kv_cache_layout"]) == "combined"
        generator = torch.Generator(device=device).manual_seed(
            settings.seed + int(case.case_id[-8:], 16)
        )

        with torch.cuda.device(self._context.device_ordinal):
            q = torch.randn(
                (batch, q_heads, head_dim),
                dtype=q_dtype,
                device=device,
                generator=generator,
            ).mul_(0.25)
            k_cache, v_cache, k_descale, v_descale = self._make_cache(
                pages=self._capture_page_count,
                page_size=self._page_size,
                kv_heads=kv_heads,
                head_dim=head_dim,
                batch=batch,
                kv_dtype=kv_dtype,
                combined=combined,
                device=device,
                generator=generator,
            )
            page_ids = torch.arange(
                self._capture_page_count,
                dtype=torch.int32,
                device=device,
            )
            page_table = page_ids.unsqueeze(0).expand(batch, -1).contiguous()
            cache_seqlens = torch.full(
                (batch,),
                self._capture_page_count * self._page_size,
                dtype=torch.int32,
                device=device,
            )
            cu_seqlens_q = torch.arange(
                batch + 1,
                dtype=torch.int32,
                device=device,
            )
            cache_seqlens.fill_(live_cache_tokens)
            expected_output, _ = paged_attention_reference(
                q,
                k_cache,
                v_cache,
                page_table,
                cache_seqlens,
                cu_seqlens_q,
                k_descale=k_descale,
                v_descale=v_descale,
                causal=True,
            )
            base_policy = PolicyContext.for_device(
                device,
                mode=PolicyMode.HEURISTIC_ONLY,
            )
            measurements = []
            for candidate in candidates:
                config = GqaConfig.from_profile(candidate.config)
                policy = base_policy.with_override(GQA_ATTENTION, config)
                caps = paged.Caps(
                    device=device,
                    mode="decode",
                    dtype=q_dtype,
                    kv_dtype=kv_dtype,
                    num_q_heads=q_heads,
                    num_kv_heads=kv_heads,
                    head_dim_qk=head_dim,
                    head_dim_vo=head_dim,
                    page_size=self._page_size,
                    max_total_q=batch,
                    max_batch=batch,
                    max_page_table_width=self._capture_page_count,
                    max_work_items=config.max_work_items,
                    max_partial_rows=config.max_partial_rows,
                    num_cache_pages=self._capture_page_count,
                    use_cuda_graph=True,
                    copy_runtime_metadata=False,
                )
                plan = paged.plan(caps, policy=policy)
                plan.prepare_decode_graph_replay_state(
                    batch=batch,
                    total_q_capacity=batch,
                    max_page_table_width=self._capture_page_count,
                    max_cache_page_count=self._capture_page_count,
                    force_split_kv=query["force_split_kv"],
                )
                (scratch_spec,) = plan.scratch_specs()
                scratch = torch.empty(
                    scratch_spec.shape,
                    dtype=scratch_spec.dtype,
                    device=device,
                )
                output = torch.empty_like(q)
                binding = None

                def run() -> None:
                    nonlocal binding
                    if binding is None:
                        binding = paged.bind(
                            plan,
                            scratch=scratch,
                            q=q,
                            k_cache=k_cache,
                            v_cache=v_cache,
                            output=output,
                            page_table=page_table,
                            cache_seqlens=cache_seqlens,
                            cu_seqlens_q=cu_seqlens_q,
                            active_total_q=batch,
                            k_descale=k_descale,
                            v_descale=v_descale,
                        )
                    paged.run(binding=binding)

                for _ in range(settings.warmup):
                    run()
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                torch.cuda.synchronize(device)
                output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                finite = bool(torch.isfinite(output).all().item())
                output_nonzero = int(torch.count_nonzero(output).item())
                cosine = float(
                    torch.nn.functional.cosine_similarity(
                        output.float().flatten(),
                        expected_output.float().flatten(),
                        dim=0,
                    )
                )
                difference = output.float() - expected_output.float()
                relative_l2 = float(
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(expected_output.float()).clamp_min(
                        1e-12
                    )
                )
                maximum_absolute_error = float(difference.abs().max())
                allclose = bool(
                    torch.allclose(
                        output.float(),
                        expected_output.float(),
                        rtol=0.05,
                        atol=0.02,
                    )
                )
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                timed_repetitions = _bounded_repetitions(
                    settings,
                    pilot_us=float(start.elapsed_time(end)) * 1_000.0,
                )
                allocated_before = torch.cuda.memory_allocated(device)
                samples_us = _cuda_event_samples_us(
                    graph.replay,
                    count=settings.groups * timed_repetitions,
                    device=device,
                    flush=self._flush,
                )
                allocated_after = torch.cuda.memory_allocated(device)
                latency_us = _median_of_group_medians(
                    tuple(samples_us),
                    groups=settings.groups,
                    repetitions=timed_repetitions,
                )
                correct = (
                    finite
                    and output_nonzero > 0
                    and cosine >= settings.minimum_cosine
                    and relative_l2 <= 0.02
                    and allclose
                    and allocated_after <= allocated_before
                )
                measurements.append(
                    SweepMeasurement(
                        candidate=candidate,
                        latency_us=latency_us,
                        correct=correct,
                        metrics={
                            "cosine": cosine,
                            "relative_l2": relative_l2,
                            "maximum_absolute_error": maximum_absolute_error,
                            "allclose": allclose,
                            "output_nonzero": output_nonzero,
                            "replay_allocation_bytes": (
                                allocated_after - allocated_before
                            ),
                            "physical_cache_pages": self._capture_page_count,
                            "shared_pages_across_requests": True,
                            "graph_ctas_per_sm": config.graph_ctas_per_sm,
                            "max_chunks_per_request": (
                                config.max_chunks_per_request
                            ),
                        },
                    )
                )
                graph.reset()
                torch.cuda.synchronize(device)
        return tuple(measurements)


class GqaBenchmarkFactory:
    """Measure paged GQA with shared physical pages and graph replay."""

    def __call__(self, group_id, cases, context):
        del group_id
        return _GqaSession(cases, context)


def _dense_mla_caps(
    case: SweepCase,
    *,
    device: object,
    max_splits: int | None,
):
    import torch

    from b12x.attention import dense_mla

    query = case.query
    mode = str(query["mode"])
    query_rows = int(query["query_rows"])
    cache_tokens = int(query["cache_tokens"])
    page_size = int(query["page_size"])
    pages = _ceil_div(cache_tokens, page_size)
    batch = query_rows if mode == "decode" else 1
    budget = None
    if max_splits is not None:
        budget = dense_mla.Budget(max_splits=max_splits)
    return dense_mla.Caps(
        device=device,
        mode=mode,
        dtype=torch.bfloat16,
        q_dtype=_torch_dtype(str(query["q_dtype"])),
        kv_dtype=_torch_dtype(str(query["kv_dtype"])),
        num_q_heads=int(query["num_q_heads"]),
        head_dim=int(query["qk_head_dim"]),
        v_head_dim=int(query["v_head_dim"]),
        physical_record_width=int(query["qk_head_dim"]),
        page_size=page_size,
        max_total_q=query_rows,
        max_batch=batch,
        max_cache_tokens=cache_tokens,
        max_page_table_width=pages,
        num_cache_pages=pages,
        use_cuda_graph=True,
        budget=budget,
    )


def _fp8_tensor_and_scale(source):
    import torch

    finfo = torch.finfo(torch.float8_e4m3fn)
    scale = (source.float().abs().amax() / finfo.max).clamp_min_(
        torch.finfo(torch.float32).tiny
    )
    quantized = (
        source.float()
        .div_(scale)
        .clamp_(min=finfo.min, max=finfo.max)
        .to(torch.float8_e4m3fn)
    )
    return quantized, scale.reshape(1)


class _MlaSession(AbstractContextManager["_MlaSession"]):
    def __init__(self, context: GenerationContext) -> None:
        self._context = context
        self._candidate_cache: dict[str, tuple[SweepCandidate, ...]] = {}

    def __enter__(self) -> "_MlaSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        try:
            import torch

            gc.collect()
            torch.cuda.synchronize(self._context.device_ordinal)
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must preserve the root error
            pass
        return None

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        import torch

        from b12x.attention import dense_mla
        from b12x.policy import PolicyContext, PolicyMode

        cached = self._candidate_cache.get(case.case_id)
        if cached is not None:
            return cached
        device = torch.device("cuda", self._context.device_ordinal)
        default_plan = dense_mla.plan(
            _dense_mla_caps(case, device=device, max_splits=None),
            policy=PolicyContext.for_device(
                device,
                mode=PolicyMode.HEURISTIC_ONLY,
            ),
        )
        default_splits = int(default_plan.num_splits)
        split_limits = {default_splits}
        split_limit = 1
        while split_limit < default_splits:
            split_limits.add(split_limit)
            split_limit *= 2
        candidates = tuple(
            SweepCandidate.create({"max_splits": value})
            for value in sorted(split_limits)
        )
        self._candidate_cache[case.case_id] = candidates
        return candidates

    def _inputs(self, case: SweepCase, *, device: object):
        import torch

        query = case.query
        mode = str(query["mode"])
        query_rows = int(query["query_rows"])
        cache_tokens = int(query["cache_tokens"])
        page_size = int(query["page_size"])
        pages = _ceil_div(cache_tokens, page_size)
        heads = int(query["num_q_heads"])
        qk_dim = int(query["qk_head_dim"])
        q_dtype = _torch_dtype(str(query["q_dtype"]))
        kv_dtype = _torch_dtype(str(query["kv_dtype"]))
        generator = torch.Generator(device=device).manual_seed(
            self._context.settings.seed + int(case.case_id[-8:], 16)
        )
        q_source = torch.randn(
            (query_rows, heads, qk_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).mul_(0.1)
        cache_source = torch.randn(
            (pages, page_size, qk_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).mul_(0.1)
        q_scale = None
        kv_scale = None
        if q_dtype == torch.float8_e4m3fn:
            q, q_scale = _fp8_tensor_and_scale(q_source)
            del q_source
        else:
            q = q_source
        if kv_dtype == torch.float8_e4m3fn:
            cache, kv_scale = _fp8_tensor_and_scale(cache_source)
            del cache_source
        else:
            cache = cache_source
        batch = query_rows if mode == "decode" else 1
        page_ids = torch.arange(pages, dtype=torch.int32, device=device)
        page_table = page_ids.unsqueeze(0).expand(batch, -1).contiguous()
        cache_seqlens = torch.full(
            (batch,),
            cache_tokens,
            dtype=torch.int32,
            device=device,
        )
        if mode == "decode":
            cu_seqlens_q = torch.arange(
                batch + 1,
                dtype=torch.int32,
                device=device,
            )
        else:
            cu_seqlens_q = torch.tensor(
                [0, query_rows],
                dtype=torch.int32,
                device=device,
            )
        return (
            q,
            cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            q_scale,
            kv_scale,
        )

    def _measure_candidate(
        self,
        *,
        case: SweepCase,
        candidate: SweepCandidate,
        inputs: tuple[object, ...],
        expected_output: object,
        expected_lse: object,
        device: object,
    ) -> SweepMeasurement:
        import torch

        from b12x.attention import dense_mla

        settings = self._context.settings
        q, cache, page_table, cache_seqlens, cu_seqlens_q, q_scale, kv_scale = inputs
        max_splits = int(candidate.config["max_splits"])
        try:
            plan = dense_mla.plan(
                _dense_mla_caps(
                    case,
                    device=device,
                    max_splits=max_splits,
                )
            )
            (scratch_spec,) = plan.scratch_specs()
            scratch = torch.empty(
                scratch_spec.shape,
                dtype=scratch_spec.dtype,
                device=device,
            )
            output = torch.empty(
                (
                    int(case.query["query_rows"]),
                    int(case.query["num_q_heads"]),
                    int(case.query["v_head_dim"]),
                ),
                dtype=torch.bfloat16,
                device=device,
            )
            binding = dense_mla.bind(
                plan,
                scratch=scratch,
                q=q,
                kv_cache=cache,
                output=output,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                q_scale=q_scale,
                kv_scale=kv_scale,
            )
            dense_mla.compile(binding=binding)

            def run() -> None:
                dense_mla.run(binding=binding)

            for _ in range(settings.warmup):
                run()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            torch.cuda.synchronize(device)
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            actual_lse = binding.scratch.final_lse[: int(case.query["query_rows"])]
            finite = bool(
                torch.isfinite(output).all().item()
                and torch.isfinite(actual_lse).all().item()
            )
            output_nonzero = int(torch.count_nonzero(output).item())
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    output.float().flatten(),
                    expected_output.float().flatten(),
                    dim=0,
                )
            )
            output_max_abs = float(
                (output.float() - expected_output.float()).abs().max()
            )
            lse_max_abs = float((actual_lse - expected_lse).abs().max())
            flush = _l2_flush_fn(device, enabled=settings.cold_l2)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            timed_repetitions = _bounded_repetitions(
                settings,
                pilot_us=float(start.elapsed_time(end)) * 1_000.0,
            )
            gc.collect()
            allocated_before = torch.cuda.memory_allocated(device)
            samples_us = _cuda_event_samples_us(
                graph.replay,
                count=settings.groups * timed_repetitions,
                device=device,
                flush=flush,
            )
            allocated_after = torch.cuda.memory_allocated(device)
            latency_us = _median_of_group_medians(
                tuple(samples_us),
                groups=settings.groups,
                repetitions=timed_repetitions,
            )
            return SweepMeasurement(
                candidate=candidate,
                latency_us=latency_us,
                correct=(
                    finite
                    and output_nonzero > 0
                    and cosine >= 0.999
                    and lse_max_abs < 2e-5
                    and allocated_after <= allocated_before
                ),
                metrics={
                    "cosine": cosine,
                    "output_max_abs": output_max_abs,
                    "lse_max_abs": lse_max_abs,
                    "output_nonzero": output_nonzero,
                    "num_splits": int(plan.num_splits),
                    "query_tile": int(plan.query_tile),
                    "chunks_per_split": int(plan.chunks_per_split),
                    "replay_allocation_bytes": allocated_after - allocated_before,
                },
            )
        except Exception as exc:  # noqa: BLE001 - one candidate may fail closed
            return SweepMeasurement(
                candidate=candidate,
                latency_us=None,
                correct=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch

        from b12x.attention import dense_mla

        if candidates != self.candidates(case):
            raise ValueError("dense MLA worker received an unknown candidate set")
        device = torch.device("cuda", self._context.device_ordinal)
        with torch.cuda.device(self._context.device_ordinal):
            inputs = self._inputs(case, device=device)
            q, cache, page_table, cache_seqlens, cu_seqlens_q, q_scale, kv_scale = (
                inputs
            )
            expected_output, expected_lse = dense_mla.reference(
                q,
                cache,
                page_table,
                cache_seqlens,
                cu_seqlens_q,
                q_scale=q_scale,
                kv_scale=kv_scale,
            )
            measurements = tuple(
                self._measure_candidate(
                    case=case,
                    candidate=candidate,
                    inputs=inputs,
                    expected_output=expected_output,
                    expected_lse=expected_lse,
                    device=device,
                )
                for candidate in candidates
            )
        return measurements


class MlaBenchmarkFactory:
    """Race dense-MLA split budgets with bounded per-candidate scratch."""

    def __call__(self, group_id, cases, context):
        del group_id, cases
        return _MlaSession(context)


@dataclass(frozen=True, kw_only=True)
class _SparseMlaInputs:
    q: object
    swa_cache: object
    swa_indices: object
    swa_lengths: object
    indexed_cache: object | None
    indexed_indices: object | None
    indexed_lengths: object | None


def _compressed_cache(
    *,
    tokens: int,
    page_size: int,
    device: object,
    generator: object,
):
    import torch

    from b12x.attention._shared.mla.compressed_reference import (
        COMPRESSED_SPARSE_MLA_NOPE_DIM,
        COMPRESSED_SPARSE_MLA_ROPE_DIM,
        pack_compressed_sparse_mla_kv_cache_reference,
    )

    k_nope = torch.randn(
        (tokens, COMPRESSED_SPARSE_MLA_NOPE_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.05)
    k_rope = torch.randn(
        (tokens, COMPRESSED_SPARSE_MLA_ROPE_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.05)
    return pack_compressed_sparse_mla_kv_cache_reference(
        k_nope,
        k_rope,
        page_size=page_size,
    )


def _sparse_indices(
    *,
    rows: int,
    width: int,
    tokens: int,
    device: object,
):
    import torch

    if width == 0:
        return torch.empty((rows, 0), dtype=torch.int32, device=device)
    stride = max(1, tokens // max(1, rows))
    offsets = (torch.arange(rows, dtype=torch.int64, device=device) * stride)[:, None]
    columns = torch.arange(width, dtype=torch.int64, device=device)[None, :]
    return ((offsets + columns) % tokens).to(torch.int32)


class _SparseMlaSession(AbstractContextManager["_SparseMlaSession"]):
    def __init__(self, context: GenerationContext) -> None:
        self._context = context
        self._candidate_cache: dict[str, tuple[SweepCandidate, ...]] = {}

    def __enter__(self) -> "_SparseMlaSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        try:
            import torch

            gc.collect()
            torch.cuda.synchronize(self._context.device_ordinal)
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must preserve the root error
            pass
        return None

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        import torch

        from b12x.attention._shared.mla.compressed_config import (
            compressed_sparse_mla_split_config_for_contract,
        )

        cached = self._candidate_cache.get(case.case_id)
        if cached is not None:
            return cached
        query = case.query
        rows = int(query["query_rows"])
        width = int(query["swa_width"]) + int(query["indexed_width"])
        indexed_width = int(query["indexed_width"])
        indexed_page_size = int(query["indexed_page_size"])
        capability = tuple(
            torch.cuda.get_device_capability(self._context.device_ordinal)
        )
        uses_single_pass = str(query["mode"]) != "decode" or (
            capability == (12, 1)
            and rows >= 16
            and int(query["num_q_heads"]) == 32
            and int(query["swa_page_size"]) == 64
            and (indexed_width == 0 or indexed_page_size == 64)
        )
        caps = (1,) if uses_single_pass else (1, 2, 4, 8, 16, 32, 64, 256)
        representatives: dict[tuple[int, int], int] = {}
        for chunk_cap in caps:
            config = compressed_sparse_mla_split_config_for_contract(
                rows=rows,
                width=max(1, width),
                max_chunks=chunk_cap,
            )
            representatives.setdefault(
                (int(config.chunk_size), int(config.num_chunks)),
                chunk_cap,
            )
        candidates = tuple(
            SweepCandidate.create({"max_chunks_per_row": chunk_cap})
            for chunk_cap in sorted(representatives.values())
        )
        self._candidate_cache[case.case_id] = candidates
        return candidates

    def _inputs(self, case: SweepCase, *, device: object) -> _SparseMlaInputs:
        import torch

        from b12x.attention._shared.mla.compressed_reference import (
            COMPRESSED_SPARSE_MLA_HEAD_DIM,
        )

        query = case.query
        rows = int(query["query_rows"])
        heads = int(query["num_q_heads"])
        mode = str(query["mode"])
        swa_width = int(query["swa_width"])
        swa_page_size = int(query["swa_page_size"])
        indexed_width = int(query["indexed_width"])
        indexed_page_size = int(query["indexed_page_size"])
        shared_cache = mode != "decode"
        generator = torch.Generator(device=device).manual_seed(
            self._context.settings.seed + int(case.case_id[-8:], 16)
        )
        q = torch.randn(
            (rows, heads, COMPRESSED_SPARSE_MLA_HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).mul_(0.04)
        swa_tokens = max(swa_width if shared_cache else swa_width * rows, 1)
        swa_cache = _compressed_cache(
            tokens=swa_tokens,
            page_size=swa_page_size,
            device=device,
            generator=generator,
        )
        swa_indices = _sparse_indices(
            rows=rows,
            width=swa_width,
            tokens=swa_tokens,
            device=device,
        )
        swa_lengths = torch.full(
            (rows,),
            swa_width,
            dtype=torch.int32,
            device=device,
        )
        indexed_cache = None
        indexed_indices = None
        indexed_lengths = None
        if indexed_width:
            indexed_tokens = indexed_width if shared_cache else indexed_width * rows
            indexed_cache = _compressed_cache(
                tokens=indexed_tokens,
                page_size=indexed_page_size,
                device=device,
                generator=generator,
            )
            indexed_indices = _sparse_indices(
                rows=rows,
                width=indexed_width,
                tokens=indexed_tokens,
                device=device,
            )
            indexed_lengths = torch.full(
                (rows,),
                indexed_width,
                dtype=torch.int32,
                device=device,
            )
        return _SparseMlaInputs(
            q=q,
            swa_cache=swa_cache,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            indexed_cache=indexed_cache,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
        )

    def _measure_candidate(
        self,
        *,
        case: SweepCase,
        candidate: SweepCandidate,
        inputs: _SparseMlaInputs,
        expected_output: object,
        device: object,
    ) -> SweepMeasurement:
        import math

        import torch

        from b12x.attention import compressed_sparse_mla
        from b12x.attention._shared.mla.compressed_config import (
            compressed_sparse_mla_split_config_for_contract,
        )
        from b12x.attention._shared.mla.compressed_reference import (
            COMPRESSED_SPARSE_MLA_HEAD_DIM,
        )

        query = case.query
        settings = self._context.settings
        rows = int(query["query_rows"])
        heads = int(query["num_q_heads"])
        width = int(query["swa_width"]) + int(query["indexed_width"])
        chunk_cap = int(candidate.config["max_chunks_per_row"])
        split_config = compressed_sparse_mla_split_config_for_contract(
            rows=rows,
            width=max(1, width),
            max_chunks=chunk_cap,
        )
        try:
            plan = compressed_sparse_mla.plan(
                compressed_sparse_mla.Caps(
                    device=device,
                    num_q_heads=heads,
                    max_q_rows=rows,
                    max_width=max(1, width),
                    head_dim=COMPRESSED_SPARSE_MLA_HEAD_DIM,
                    v_head_dim=COMPRESSED_SPARSE_MLA_HEAD_DIM,
                    max_batch=rows,
                    page_size=int(query["swa_page_size"]),
                    layout=str(query["layout"]),
                    mode=str(query["mode"]),
                    swa_width=int(query["swa_width"]),
                    indexed_width=int(query["indexed_width"]),
                    swa_page_size=int(query["swa_page_size"]),
                    indexed_page_size=int(query["indexed_page_size"]),
                    use_cuda_graph=True,
                    max_chunks_per_row=chunk_cap,
                )
            )
            (scratch_spec,) = plan.scratch_specs()
            scratch = torch.empty(
                scratch_spec.shape,
                dtype=scratch_spec.dtype,
                device=device,
            )
            binding = compressed_sparse_mla.bind(
                plan,
                scratch=scratch,
                q=inputs.q,
                swa_indices=inputs.swa_indices,
                swa_lengths=inputs.swa_lengths,
                indexed_indices=inputs.indexed_indices,
                indexed_lengths=inputs.indexed_lengths,
            )
            output = torch.empty(
                (rows, heads, COMPRESSED_SPARSE_MLA_HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            )

            def run() -> None:
                compressed_sparse_mla.run(
                    binding=binding,
                    swa_k_cache=inputs.swa_cache,
                    swa_page_size=int(query["swa_page_size"]),
                    indexed_k_cache=inputs.indexed_cache,
                    indexed_page_size=(
                        int(query["indexed_page_size"])
                        if int(query["indexed_width"])
                        else None
                    ),
                    sm_scale=1.0 / math.sqrt(COMPRESSED_SPARSE_MLA_HEAD_DIM),
                    expected_num_q_heads=heads,
                    out=output,
                )

            for _ in range(settings.warmup):
                run()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            torch.cuda.synchronize(device)
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            finite = bool(torch.isfinite(output).all().item())
            output_nonzero = int(torch.count_nonzero(output).item())
            difference = output.float() - expected_output.float()
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    output.float().flatten(),
                    expected_output.float().flatten(),
                    dim=0,
                )
            )
            maximum_absolute_error = float(difference.abs().max())
            rmse = float(torch.sqrt(torch.mean(difference * difference)))
            flush = _l2_flush_fn(device, enabled=settings.cold_l2)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            timed_repetitions = _bounded_repetitions(
                settings,
                pilot_us=float(start.elapsed_time(end)) * 1_000.0,
            )
            gc.collect()
            allocated_before = torch.cuda.memory_allocated(device)
            samples_us = _cuda_event_samples_us(
                graph.replay,
                count=settings.groups * timed_repetitions,
                device=device,
                flush=flush,
            )
            allocated_after = torch.cuda.memory_allocated(device)
            return SweepMeasurement(
                candidate=candidate,
                latency_us=_median_of_group_medians(
                    tuple(samples_us),
                    groups=settings.groups,
                    repetitions=timed_repetitions,
                ),
                correct=(
                    finite
                    and output_nonzero > 0
                    and cosine >= 0.995
                    and allocated_after <= allocated_before
                ),
                metrics={
                    "cosine": cosine,
                    "maximum_absolute_error": maximum_absolute_error,
                    "rmse": rmse,
                    "output_nonzero": output_nonzero,
                    "chunk_size": int(split_config.chunk_size),
                    "num_chunks": int(split_config.num_chunks),
                    "scratch_bytes": int(scratch.numel() * scratch.element_size()),
                    "replay_allocation_bytes": allocated_after - allocated_before,
                },
            )
        except Exception as exc:  # noqa: BLE001 - one candidate may fail closed
            return SweepMeasurement(
                candidate=candidate,
                latency_us=None,
                correct=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import math

        import torch

        from b12x.attention._shared.mla.compressed_reference import (
            COMPRESSED_SPARSE_MLA_HEAD_DIM,
            compressed_sparse_mla_reference,
        )

        if candidates != self.candidates(case):
            raise ValueError("sparse MLA worker received an unknown candidate set")
        query = case.query
        device = torch.device("cuda", self._context.device_ordinal)
        with torch.cuda.device(self._context.device_ordinal):
            inputs = self._inputs(case, device=device)
            expected_output = compressed_sparse_mla_reference(
                inputs.q,
                inputs.swa_cache,
                inputs.swa_indices,
                inputs.swa_lengths,
                sm_scale=1.0 / math.sqrt(COMPRESSED_SPARSE_MLA_HEAD_DIM),
                extra_k_cache=inputs.indexed_cache,
                extra_indices=inputs.indexed_indices,
                extra_topk_lengths=inputs.indexed_lengths,
                swa_page_size=int(query["swa_page_size"]),
                extra_page_size=(
                    int(query["indexed_page_size"])
                    if int(query["indexed_width"])
                    else None
                ),
            )
            measurements = tuple(
                self._measure_candidate(
                    case=case,
                    candidate=candidate,
                    inputs=inputs,
                    expected_output=expected_output,
                    device=device,
                )
                for candidate in candidates
            )
        return measurements


class SparseMlaBenchmarkFactory:
    """Race bounded compressed sparse-MLA split contracts."""

    def __call__(self, group_id, cases, context):
        del group_id, cases
        return _SparseMlaSession(context)


__all__ = [
    "GdnBenchmarkFactory",
    "GqaBenchmarkFactory",
    "MlaBenchmarkFactory",
    "SparseMlaBenchmarkFactory",
]
