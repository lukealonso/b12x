"""Production KDA prefill races with an independent FP32 state oracle."""

from __future__ import annotations

import gc
import time
from contextlib import AbstractContextManager
from itertools import product


from b12x.policy.components import KDA_PREFILL
from b12x.policy.generation.sweep import DiscreteSweepGenerator, SweepCandidate, SweepCase, SweepMeasurement

from .gpu_workers import _l2_flush_fn


def kda_cases() -> tuple[SweepCase, ...]:
    cases = []
    for heads, tokens, seqs, normalize, checkpoint in product(
        (16, 64), (32, 256, 2048), (1, 4), (False, True), (False, True)
    ):
        query = dict(heads=heads, head_dim=128, model_dtype="bfloat16", state_dtype="float32",
                     qk_l2norm=normalize, checkpoint_export=checkpoint,
                     max_tokens=tokens, max_seqs=seqs)
        group = f"h{heads}-m{tokens}-b{seqs}-norm{int(normalize)}-cp{int(checkpoint)}"
        for scenario, live_tokens, gate in (("full", tokens, "random"),
                                             ("partial-long-memory", tokens // 2 + 1, "long_memory")):
            cases.append(SweepCase.create(group_id=group, query=query, scenario=scenario,
                                         metadata={"live_tokens": live_tokens, "gate": gate,
                                                   "input_contract": "unit-key-fp32-state-1"}))
    return tuple(cases)


def kda_candidates(query, device) -> tuple[SweepCandidate, ...]:
    from b12x.sequence.kda_prefill._policy import (
        KDA_PREFILL_POLICY, KdaPrefillConfig, V_SPLIT_CHOICES, K_SPLIT_CHOICES,
        STAGE_CHOICES, default_window_tiles, tiles_capacity,
    )

    capacity = tiles_capacity(query.max_tokens, query.max_seqs)
    windows = sorted({min(capacity, value) for value in (4, 16, 64, 256)} |
                     {default_window_tiles(query.heads, query.max_tokens, query.max_seqs)})
    candidates = []
    for value, key, stages, window in product(V_SPLIT_CHOICES, K_SPLIT_CHOICES, STAGE_CHOICES, windows):
        config = KdaPrefillConfig(v_split=value, k_split=key, stages=stages, window_tiles=window)
        try:
            KDA_PREFILL_POLICY.validate_config(query, config, device)
        except ValueError:
            continue
        candidates.append(SweepCandidate.create(config.to_dict()))
    return tuple(candidates)


def _fixture(case, context):
    import torch
    from b12x.sequence.kda_prefill.reference import prefill_kda

    query = case.query
    device = torch.device("cuda", context.device_ordinal)
    generator = torch.Generator(device="cpu").manual_seed(context.settings.seed)
    heads, capacity, seqs = (int(query[name]) for name in ("heads", "max_tokens", "max_seqs"))
    tokens = int(case.metadata["live_tokens"])
    lengths = [tokens // seqs + int(index < tokens % seqs) for index in range(seqs)]

    def random(shape, scale=0.25, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=generator) * scale).to(device=device, dtype=dtype)

    row_shape = (capacity, heads, 128)
    keys = random(row_shape).float()
    keys = (keys / keys.norm(dim=-1, keepdim=True)).to(torch.bfloat16)
    gate = random(row_shape, scale=1.)
    if case.metadata["gate"] == "long_memory":
        gate[:, :, :32] = -12.
    offsets = [16 if query["checkpoint_export"] and length >= 16 else 0 for length in lengths]
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)

    def integers(values):
        return torch.tensor(values, device=device, dtype=torch.int32)

    tensors = dict(
        q=random(row_shape), k=keys, v=random(row_shape), raw_g=gate,
        raw_beta=random((capacity, heads), scale=1.),
        A_log=random((heads,), scale=0.1, dtype=torch.float32),
        dt_bias=random((heads, 128), scale=0.1, dtype=torch.float32),
        recurrent_state=random((3 * seqs + 1, heads, 128, 128), scale=0.1, dtype=torch.float32),
        cu_seqlens=integers(cumulative), initial_state_indices=integers(range(seqs)),
        final_state_indices=integers(range(seqs, 2 * seqs)),
        checkpoint_state_indices=integers(range(2 * seqs, 3 * seqs)),
        checkpoint_offsets=integers(offsets), num_tokens=integers([tokens]), num_seqs=integers([seqs]),
        output=torch.empty(row_shape, device=device, dtype=torch.bfloat16),
    )
    initial = tensors["recurrent_state"].clone()
    expected_state = initial.clone()
    expected = prefill_kda(
        tensors["q"], tensors["k"], tensors["v"], tensors["raw_g"], tensors["raw_beta"],
        tensors["A_log"], tensors["dt_bias"], expected_state, tensors["cu_seqlens"],
        tensors["initial_state_indices"], tensors["final_state_indices"],
        tensors["checkpoint_state_indices"], tensors["checkpoint_offsets"], seqs, tokens,
        lower_bound=-5., qk_l2norm=bool(query["qk_l2norm"]),
    )
    written = list(range(seqs, 2 * seqs)) + [2 * seqs + i for i, offset in enumerate(offsets) if offset]
    untouched = [i for i in range(3 * seqs + 1) if i not in written]
    return tensors, initial, expected[:tokens], expected_state, written, untouched


def _error_metrics(actual, expected, *, tolerance, minimum_cosine):
    import torch
    import torch.nn.functional as functional

    actual, expected = actual.float(), expected.float()
    if not bool(torch.isfinite(expected).all()) or not bool(torch.count_nonzero(expected)):
        raise RuntimeError("KDA reference must be finite and nonzero")
    difference = (actual - expected).abs()
    rms = expected.square().mean().sqrt()
    relative = float(difference.square().mean().sqrt() / (rms + 1e-8))
    peak = float(difference.max())
    cosine = float(functional.cosine_similarity(actual.reshape(1, -1), expected.reshape(1, -1)))
    passed = (bool(torch.isfinite(actual).all()) and bool(torch.count_nonzero(actual))
              and relative < tolerance and cosine >= minimum_cosine
              and peak <= 0.04 * float(rms) + 2**-6 * float(expected.abs().max()))
    return passed, {"relative_rmse": relative, "max_error": peak, "cosine": cosine}


class _KdaSession(AbstractContextManager):
    def __init__(self, context):
        self.context = context

    def __enter__(self):
        import torch

        self._device_context = torch.cuda.device(self.context.device_ordinal)
        self._device_context.__enter__()
        return self

    def __exit__(self, *_exc):
        import torch

        try:
            gc.collect()
            torch.cuda.synchronize(self.context.device_ordinal)
            torch.cuda.empty_cache()
        finally:
            self._device_context.__exit__(*_exc)

    def candidates(self, case):
        from b12x.sequence.kda_prefill._policy import KdaPrefillQuery

        return kda_candidates(KdaPrefillQuery(**case.query.to_dict()), self.context.device)

    def measure(self, case, candidates):
        import torch
        from b12x.sequence import kda_prefill as kda
        from b12x.policy import PolicyContext, PolicyMode
        from b12x.policy.generation.replay import PreparedCandidate, capture_warmed_graph, measure_prepared_candidates

        context, settings = self.context, self.context.settings
        device = torch.device("cuda", context.device_ordinal)
        oracle_start = time.monotonic()
        tensors, initial, expected, expected_state, written, untouched = _fixture(case, context)
        torch.cuda.synchronize(device)
        oracle_seconds = time.monotonic() - oracle_start
        query = case.query
        caps = kda.Caps(device=device, heads=int(query["heads"]), max_tokens=int(query["max_tokens"]),
                        max_seqs=int(query["max_seqs"]), max_state_slots=initial.shape[0],
                        qk_l2norm=bool(query["qk_l2norm"]), checkpoint_export=bool(query["checkpoint_export"]))
        base_policy = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)
        flush = _l2_flush_fn(device, enabled=settings.cold_l2)
        prepared, failures = {}, {}
        for candidate in candidates:
            started = time.monotonic()
            try:
                config = kda.KdaPrefillConfig.from_profile(candidate.config)
                plan = kda.plan(caps, policy=base_policy.with_override(KDA_PREFILL, config))
                scratch = torch.empty(plan.scratch_specs()[0].shape, dtype=torch.uint8, device=device)
                binding = kda.bind(plan, scratch=scratch, **tensors)
                for _ in range(settings.warmup):
                    kda.run(binding, lower_bound=-5.)
                torch.cuda.synchronize(device)
                graph, _ = capture_warmed_graph(lambda: kda.run(binding, lower_bound=-5.), device=device)
                tensors["recurrent_state"].copy_(initial)
                tensors["output"].fill_(float("nan"))
                scratch.fill_(0xFF)
                before = torch.cuda.memory_allocated(device)
                graph.replay()
                torch.cuda.synchronize(device)
                allocation = torch.cuda.memory_allocated(device) - before
                out_pass, out_metrics = _error_metrics(tensors["output"][:expected.shape[0]], expected,
                                                      tolerance=1e-2, minimum_cosine=settings.minimum_cosine)
                state_pass, state_metrics = _error_metrics(tensors["recurrent_state"][written], expected_state[written],
                                                          tolerance=5e-3, minimum_cosine=settings.minimum_cosine)
                correct = (out_pass and state_pass and allocation == 0 and int(binding.error_code.item()) == 0
                           and torch.equal(tensors["recurrent_state"][untouched], initial[untouched])
                           and bool(torch.isnan(tensors["output"][expected.shape[0]:]).all()))
                metrics = {"output": out_metrics, "state": state_metrics,
                           "replay_allocation_bytes": allocation, "oracle_seconds": oracle_seconds,
                           "prepare_seconds": time.monotonic() - started}
                if not correct:
                    failures[candidate.candidate_id] = SweepMeasurement(
                        candidate=candidate, latency_us=None, correct=False,
                        metrics=metrics, error="KDA output, state, or replay contract failed")
                else:
                    prepared[candidate.candidate_id] = PreparedCandidate(
                        candidate=candidate, graph=graph, correct=True, owners=(binding, scratch, tensors, initial),
                        metrics={**metrics, "frozen_resolution_capture": True, "state_reset_before_each": True},
                        before_each=lambda: tensors["recurrent_state"].copy_(initial))
            except Exception as exc:
                failures[candidate.candidate_id] = SweepMeasurement(
                    candidate=candidate, latency_us=None, correct=False, error=f"{type(exc).__name__}: {exc}")

        return measure_prepared_candidates(
            (failures[c.candidate_id] if c.candidate_id in failures else prepared[c.candidate_id] for c in candidates),
            settings=settings, device=device, flush=flush)


class _KdaFactory:
    def __call__(self, group_id, cases, context):
        return _KdaSession(context)


class KdaPrefillGenerator(DiscreteSweepGenerator):
    @staticmethod
    def validate_region_decision(inputs, device, decision):
        from b12x.sequence.kda_prefill._policy import TUNING_PROBLEM

        query = TUNING_PROBLEM.query_from_inputs(inputs)
        TUNING_PROBLEM.lower(query, device, decision)
        if min(query.heads, query.max_tokens, query.max_seqs) <= 0:
            raise ValueError("KDA search requires positive head and capacity dimensions")

    @staticmethod
    def cases_for_tuning_queries(queries):
        from b12x.policy.problem import stable_identity
        from b12x.sequence.kda_prefill._policy import TUNING_PROBLEM

        for values in queries:
            query = TUNING_PROBLEM.query_from_inputs(values)
            KdaPrefillGenerator.validate_region_decision(
                values, None, TUNING_PROBLEM.policy.heuristic(query, None).to_dict())
            for scenario, live_tokens, gate in (("full", query.max_tokens, "random"),
                                               ("partial-long-memory", query.max_tokens // 2 + 1, "long_memory")):
                yield SweepCase.create(group_id=f"shape-{stable_identity(values)[:24]}", query=values,
                                       scenario=scenario, metadata={"live_tokens": live_tokens, "gate": gate,
                                                                  "input_contract": "unit-key-fp32-state-1"})

    def __init__(self, *, cases=None, benchmark_factory=None):
        from b12x.sequence.kda_prefill._policy import KDA_PREFILL_POLICY

        super().__init__(
            component_id=KDA_PREFILL, query_schema_version=1, config_schema_version=1,
            query_fields=tuple(sorted(KDA_PREFILL_POLICY.query_fields)), range_fields=frozenset(),
            cases=kda_cases() if cases is None else cases,
            benchmark_factory=_KdaFactory() if benchmark_factory is None else benchmark_factory,
            coverage={"oracle": "independent-fp32-recurrence", "input_contract": "unit-key-fp32-state-1"},
            candidate_contract_version=2,
        )

    def reviewed_queries(self):
        from b12x.sequence.kda_prefill._policy import KdaPrefillQuery

        return tuple(KdaPrefillQuery(**query.to_dict()) for query in dict.fromkeys(case.query for case in self._cases))


__all__ = ["KdaPrefillGenerator", "kda_candidates", "kda_cases"]
