"""Qualify activation precision candidates over shared native NVFP4 storage."""

from __future__ import annotations

import math
import statistics

from .moe import MoeMeasurement


def source_identity():
    import hashlib
    import importlib.metadata
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    paths = (
        *sorted((root / "b12x/moe").rglob("*.py")),
        *sorted((root / "b12x/_lib").rglob("*.py")),
        *sorted((root / "b12x/policy/generation").rglob("*.py")),
        root / "benchmarks/benchmark_blockscaled_precision.py",
        root / "benchmarks/benchmark_w4a16_nvfp4_layouts.py",
        root / "benchmarks/benchmark_moe.py",
        root / "tests/_reference/w4a16_reference.py",
    )
    payload = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    payload["toolchain"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "nvidia-cutlass-dsl", "triton")
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def qualifies(pairs, candidate, baseline):
    """Prefer A16 at median parity; retain the interval as diagnostic evidence."""
    from benchmarks.benchmark_blockscaled_precision import _ratio_interval

    ratio = statistics.median(p[candidate] for p in pairs) / statistics.median(
        p[baseline] for p in pairs
    )
    return ratio <= 1.0, ratio, _ratio_interval(pairs, candidate, baseline)


def select_winner(grouped, minimum_cosine):
    """Compare confirmed geometric means across the same route patterns."""
    from collections import defaultdict

    by_candidate = defaultdict(list)
    for _case, measurements in grouped:
        for measurement in measurements:
            if measurement.passes(minimum_cosine):
                by_candidate[measurement.candidate.candidate_id].append(measurement)
    scores = []
    for measurements in by_candidate.values():
        if len(measurements) != len(grouped):
            continue
        initial, confirmation = (
            math.exp(statistics.mean(math.log(item.metrics[field]) for item in measurements))
            for field in ("initial_latency_us", "confirmation_latency_us")
        )
        scores.append((initial, confirmation, measurements[0].candidate))
    baselines = [row for row in scores if row[2].config["backend"] != "w4a16"]
    if not baselines:
        raise RuntimeError("automatic MoE precision has no route-robust qualified A4 baseline")
    eligible = [row for row in scores if row[2].config["backend"] != "w4a16" or all(
        row[0] <= base[0] and row[1] <= base[1] for base in baselines
    )]
    return min(eligible, key=lambda row: (
        row[1], row[2].config["backend"] != "w4a16", row[2].candidate_id,
    ))[2]


def _reference(session, inputs, candidate):
    from b12x.moe._shared.kernels.reference import moe_reference_nvfp4
    from tests._reference.w4a16_reference import moe_reference_w4a16

    weights = session._experts._impl
    geometry = session._geometry
    common = (
        inputs.topk_ids, inputs.topk_weights,
        geometry.num_experts, geometry.hidden_size, geometry.intermediate_size,
    )
    if candidate.config["backend"] == "w4a16":
        return moe_reference_w4a16(
            inputs.x, weights.w1_fp4, weights.w1_blockscale,
            weights.representation_for("w4a16").micro_w13_global_scale,
            weights.w2_fp4, weights.w2_blockscale,
            weights.representation_for("w4a16").micro_w2_global_scale, *common,
            activation=geometry.activation,
        )
    return moe_reference_nvfp4(
        inputs.x, weights.w1_fp4, weights.w1_blockscale, weights.w1_alphas,
        weights.w2_fp4, weights.w2_blockscale, weights.w2_alphas,
        weights.a1_gscale, weights.a2_gscale, *common,
        activation=geometry.activation,
        quant_scale_math=(
            "reciprocal_multiply" if candidate.config["backend"] == "micro"
            else "direct_division"
        ),
    )


def measure_precision(session, case, candidates):
    import torch
    import b12x

    from benchmarks.benchmark_blockscaled_precision import _clock_checks, _paired, _snapshot
    from benchmarks.benchmark_w4a16_nvfp4_layouts import _check
    from benchmarks.moe_checkpoint_snapshot import tensor_digest
    from .moe_gpu_worker import _is_fatal_accelerator_error

    if any(candidate not in session.candidates for candidate in candidates):
        raise ValueError("MoE precision worker received an unknown candidate")
    eligible = session.eligible_candidates(case, candidates)
    inputs = session._stage_query_inputs(case)
    settings = session._context.settings
    experts = session._experts._impl
    native = experts.representation_for("w4a16")
    def weight_hashes():
        return {
            name: tensor_digest(getattr(experts, name))
            for name in ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale")
        }

    original_hashes = weight_hashes()
    for actual, original in (
        (native.w13, experts.w1_fp4), (native.w2, experts.w2_fp4),
        (native.micro_w13_scale, experts.w1_blockscale),
        (native.micro_w2_scale, experts.w2_blockscale),
    ):
        if actual.data_ptr() != original.data_ptr():
            raise AssertionError("precision candidates must share native NVFP4 storage")
    graphs, metrics, references, failures = {}, {}, {}, {}
    for candidate in candidates:
        if candidate not in eligible:
            failures[candidate.candidate_id] = "candidate does not support this geometry and capacity"
    for candidate in eligible:
        name = candidate.candidate_id
        mode = "w4a16" if candidate.config["backend"] == "w4a16" else "nvfp4"
        reference_key = (mode, candidate.config["backend"] == "micro")
        try:
            if reference_key not in references:
                references[reference_key] = _reference(session, inputs, candidate)
            expected = references[reference_key]
            prepared = session._prepare_candidate(case=case, candidate=candidate, inputs=inputs)
            prepared.graph.replay()
            metrics[name] = _check(
                prepared.output, expected, name=name, m=case.num_tokens, oracle_mode=mode,
            )
            prepared.output.fill_(float("nan"))
            prepared.graph.replay()
            _check(prepared.output, expected, name=name, m=case.num_tokens, oracle_mode=mode)
            graphs[name] = prepared.graph
        except Exception as error:
            if _is_fatal_accelerator_error(error):
                raise
            failures[name] = f"{type(error).__name__}: {error}"
    baselines = [
        c.candidate_id for c in eligible
        if c.config["backend"] != "w4a16" and c.candidate_id in graphs
    ]
    if not baselines:
        return tuple(MoeMeasurement(
            candidate=c, latency_us=None, cosine=None,
            error=failures.get(c.candidate_id, "no qualified A4 baseline"),
        ) for c in candidates)
    count = max(20, settings.groups * settings.repetitions)
    warmup = max(3, settings.warmup)

    def timed(selected):
        attempts = []
        for _ in range(3):
            _paired(selected, warmup, count, session._flush)
            before = _snapshot()
            allocated = torch.cuda.memory_allocated(session._device)
            pairs = _paired(selected, warmup, count, session._flush)
            allocation_delta = torch.cuda.memory_allocated(session._device) - allocated
            after = _snapshot()
            clocks = _clock_checks(before, after)
            attempts.append(dict(
                samples_us=pairs, snapshot_before=before, snapshot_after=after,
                clock_validation=clocks, replay_allocation_bytes=allocation_delta,
            ))
            if clocks["valid"] and allocation_delta == 0:
                break
        return pairs, clocks["valid"] and allocation_delta == 0, attempts

    b12x.freeze_kernel_resolution("MoE precision race and confirmation")
    try:
        pairs, valid, attempts = timed(graphs)
        confirmation, confirmation_valid, confirmation_attempts = timed(graphs)
        for candidate in eligible:
            name = candidate.candidate_id
            if name not in graphs:
                continue
            prepared = session._prepared_candidates[name]
            mode = "w4a16" if candidate.config["backend"] == "w4a16" else "nvfp4"
            expected = references[(mode, candidate.config["backend"] == "micro")]
            prepared.output.fill_(float("nan"))
            prepared.graph.replay()
            _check(prepared.output, expected, name=name, m=case.num_tokens, oracle_mode=mode)
    finally:
        b12x.unfreeze_kernel_resolution()
    if weight_hashes() != original_hashes:
        raise AssertionError("precision candidate mutated the shared native weights")
    results = []
    for candidate in candidates:
        name = candidate.candidate_id
        if name in failures:
            results.append(MoeMeasurement(candidate=candidate, latency_us=None, cosine=None, error=failures[name]))
            continue
        error = None
        if not valid or not confirmation_valid:
            error = "clock or replay-allocation qualification failed"
        initial_latency = statistics.median(p[name] for p in pairs)
        confirmation_latency = statistics.median(p[name] for p in confirmation)
        results.append(MoeMeasurement(
            candidate=candidate, latency_us=confirmation_latency,
            cosine=metrics[name]["cos"], error=error,
            metrics=dict(
                correctness=metrics[name],
                shared_native_storage=True,
                native_weight_digests=original_hashes,
                concrete_path=session._prepared_candidates[name].concrete_path,
                timing_attempts=attempts, confirmation_attempts=confirmation_attempts,
                initial_latency_us=initial_latency,
                confirmation_latency_us=confirmation_latency,
                promotion_ratio_limit=1.0,
                selection_scope="geometric mean over qualified route patterns",
                ratio_direction="A16 / A4; lower is faster",
                qualified_baselines=baselines,
                initial_ratios={base: qualifies(pairs, name, base)[1:] for base in baselines},
                confirmation_ratios={
                    base: qualifies(confirmation, name, base)[1:] for base in baselines
                } if confirmation and name in confirmation[0] else {},
            ),
        ))
    return tuple(results)
