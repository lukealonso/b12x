"""Measured precision routing for the one-shot dense GEMM specialization."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import statistics
from collections import defaultdict
from contextlib import AbstractContextManager
from pathlib import Path

from b12x.gemm.blockscaled._policy import A16_CONFIGS, BLOCKSCALED_PRECISION
from b12x.policy.generation.reducer import DecisionRecord, build_axis_tree
from b12x.policy.generation.sweep import DiscreteSweepGenerator, SweepCandidate, SweepCase, SweepMeasurement
from b12x.policy.types import FrozenMapping

GEOMETRIES = ((4096, 5376), (16384, 1024), (17408, 5120), (5120, 17408), (248320, 2560))
ROW_COUNTS = (*range(1, 17), 24, 32, 64, 128, 256, 512, 1024, 2048)
QUERY_FIELDS = ("recipe", "in_features", "out_features")


def input_seed(seed, recipe, n, k, m=None):
    identity = json.dumps((recipe, n, k, m), separators=(",", ":"))
    return seed + int(hashlib.sha256(identity.encode()).hexdigest()[-8:], 16)


def _source_identity():
    root = Path(__file__).resolve().parents[3]
    paths = (root / "_lib/dense_gemm.py", root / "_lib/intrinsics.py", Path(__file__),
             *sorted((root / "gemm/blockscaled").glob("*.py")),
             root.parent / "benchmarks/benchmark_blockscaled_precision.py")
    payload = {str(p.relative_to(root.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in paths}
    payload["toolchain"] = {name: importlib.metadata.version(name)
                            for name in ("torch", "nvidia-cutlass-dsl", "triton")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def precision_cases(*, geometries=GEOMETRIES, counts=ROW_COUNTS, recipes=("nvfp4", "mxfp8")):
    identity = _source_identity()
    return tuple(SweepCase.create(
        group_id=f"{recipe}-n{n}-k{k}",
        query=dict(recipe=recipe, in_features=k, out_features=n, measured_m=m),
        metadata={"source_toolchain_sha256": identity},
    ) for recipe in recipes for n, k in geometries for m in counts)


def qualifies(pairs, candidate, baseline):
    from benchmarks.benchmark_blockscaled_precision import _ratio_interval
    ratio = statistics.median(p[candidate] for p in pairs) / statistics.median(p[baseline] for p in pairs)
    interval = _ratio_interval(pairs, candidate, baseline)
    return ratio <= 1.0, ratio, interval


class _Session(AbstractContextManager):
    def __init__(self, context):
        self.context = context
        self.weight = None

    def __enter__(self):
        import torch
        self.device_context = torch.cuda.device(self.context.device_ordinal)
        self.device_context.__enter__()
        if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
            raise ValueError("blockscaled precision qualification requires SM120/SM121")
        return self

    def __exit__(self, *exc):
        import torch
        torch.cuda.synchronize()
        self.weight = None
        self.local = None
        self.device_context.__exit__(*exc)

    def candidates(self, case):
        m = int(case.query["measured_m"])
        return (SweepCandidate.create({"a16_rows": []}), *(
            SweepCandidate.create({"a16_rows": [[m, *config]]}) for config in A16_CONFIGS
        ))

    def measure(self, case, candidates):
        import torch
        from b12x.gemm import blockscaled
        from b12x._lib.dense_gemm import dense_gemm, dense_gemm_fused_quant_a
        from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
        from b12x.gemm._shared.wo_mxfp8 import quantize_mxfp8_rows_torch, dequantize_mxfp8_rows_torch
        from benchmarks.benchmark_blockscaled_precision import (
            _reference_weight, _capture, _check, _paired, _snapshot, _clock_checks,
        )
        from .gpu_workers import _l2_flush_fn

        settings = self.context.settings
        m, n, k = (int(case.query[field]) for field in ("measured_m", "out_features", "in_features"))
        recipe = str(case.query["recipe"])
        weight_seed = input_seed(settings.seed, recipe, n, k)
        source_seed = input_seed(settings.seed, recipe, n, k, m)
        if self.weight is None:
            torch.manual_seed(weight_seed)
            self.weight, self.local, self.multiplier = _reference_weight(recipe, n, k)
        weight, local, multiplier = self.weight, self.local, self.multiplier
        torch.manual_seed(source_seed)
        source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).mul_(0.25)
        options = (dict(activation_global_scale=(2688.0 / source.abs().amax().float()).reshape(1))
                   if recipe == "nvfp4" else {})
        reference = (source.float() @ local.to(torch.bfloat16).float().T) * multiplier
        if recipe == "nvfp4":
            aq, sf = quantize_grouped_nvfp4_torch(source[None], torch.tensor([m], device="cuda"), options["activation_global_scale"])
            qref = dense_gemm((aq, sf), (weight.values[:, :, None], weight.scale_mma),
                             ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn", c_dtype="bfloat16",
                             sf_vec_size=16, alpha=multiplier / options["activation_global_scale"], expected_m=m)[:, :, 0]
        else:
            aq = quantize_mxfp8_rows_torch(source)
            qref = dequantize_mxfp8_rows_torch(aq.values, aq.scale_rows).float() @ local.T
        scratch_bytes = max(blockscaled.workspace_size(
            weight, m, _config=tuple(candidate.config["a16_rows"][0][1:])
            if candidate.config["a16_rows"] else None,
        ) for candidate in candidates)
        scratch = torch.empty(scratch_bytes, device="cuda", dtype=torch.uint8)
        graphs, buffers, checks, failures = {}, [scratch], {}, {}
        for index, candidate in enumerate(candidates):
            rows = candidate.config["a16_rows"]
            config = tuple(rows[0][1:]) if rows else None
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            def call(out=out, scratch=scratch, config=config):
                return blockscaled.mm(source, weight, out=out, workspace=scratch,
                                      mode="a16" if config else "quantized", _config=config,
                                      expected_m=m, **({} if config else options))
            try:
                call()
                checks[str(index)] = _check(out, reference if config else qref, str(index))
                graph = _capture(call)
                out.fill_(float("nan"))
                scratch.fill_(255)
                graph.replay()
                _check(out, reference if config else qref, str(index))
                graphs[str(index)] = graph
                buffers.append(out)
            except Exception as exc:
                failures[str(index)] = f"{type(exc).__name__}: {exc}"
        if "0" not in graphs:
            raise RuntimeError(f"quantized baseline failed: {failures['0']}")
        if recipe == "mxfp8" and m <= 8:
            out = torch.empty(m, n, 1, device="cuda", dtype=torch.bfloat16)
            scratch = torch.empty(2 * m * n, device="cuda", dtype=torch.float32)
            def fused():
                return dense_gemm_fused_quant_a(source, weight.weight.values[:, :, None],
                    weight.weight.scale_mma, out=out, expected_m=m, _split_k_workspace=scratch)
            fused()
            checks["fused"] = _check(out[:, :, 0], qref, "fused")
            graphs["fused"] = _capture(fused)
            out.fill_(float("nan"))
            scratch.fill_(float("nan"))
            graphs["fused"].replay()
            _check(out[:, :, 0], qref, "fused")
            buffers.extend((out, scratch))
        flush = _l2_flush_fn(torch.device("cuda", self.context.device_ordinal), enabled=settings.cold_l2)
        warmup, count = max(3, settings.warmup), max(20, settings.groups * settings.repetitions)
        def timed_pass(selected):
            attempts = []
            for _ in range(3):
                _paired(selected, warmup, count, flush)
                before = _snapshot()
                allocated = torch.cuda.memory_allocated()
                pairs = _paired(selected, warmup, count, flush)
                allocation_delta = torch.cuda.memory_allocated() - allocated
                after = _snapshot()
                clocks = _clock_checks(before, after)
                attempts.append(dict(snapshot_before=before, snapshot_after=after,
                                     samples_us=pairs, clock_validation=clocks,
                                     replay_allocation_bytes=allocation_delta))
                if clocks["valid"] and allocation_delta == 0:
                    break
            return pairs, clocks, allocation_delta, attempts

        pairs, clocks, allocation_delta, attempts = timed_pass(graphs)
        baseline_names = [name for name in ("0", "fused") if name in graphs]
        baseline = min(baseline_names, key=lambda name: statistics.median(p[name] for p in pairs))
        eligible = []
        for name in graphs:
            if name not in baseline_names and all(qualifies(pairs, name, base)[0] for base in baseline_names):
                eligible.append(name)
        confirmation = None
        confirmation_clocks = None
        confirmation_attempts = []
        if eligible:
            confirm_graphs = {name: graphs[name] for name in (*baseline_names, *eligible)}
            confirmation, confirmation_clocks, confirm_allocation, confirmation_attempts = timed_pass(confirm_graphs)
            if confirm_allocation != 0:
                confirmation_clocks["valid"] = False
        result = []
        for index, candidate in enumerate(candidates):
            name = str(index)
            if name in failures:
                result.append(SweepMeasurement(candidate=candidate, latency_us=None,
                                               correct=False, error=failures[name]))
                continue
            error = None
            if not clocks["valid"] or allocation_delta != 0:
                error = "clock or replay-allocation qualification failed"
            if index and (name not in eligible or confirmation is None or
                          not confirmation_clocks["valid"] or
                          not all(qualifies(confirmation, name, base)[0] for base in baseline_names)):
                error = error or "A16 was slower than a quantized baseline or lacked valid independent confirmation"
            result.append(SweepMeasurement(
                candidate=candidate, correct=True,
                latency_us=statistics.median(p[name] for p in pairs), error=error,
                metrics=dict(correctness=checks[name], baseline=baseline,
                             baseline_correctness={base: checks[base] for base in baseline_names},
                             weight_seed=weight_seed, activation_seed=source_seed,
                             samples_us=[{key: p[key] for key in (*baseline_names, name)} for p in pairs],
                             confirmation_samples_us=confirmation or [], clock_validation=clocks,
                             confirmation_clock_validation=confirmation_clocks or {},
                             replay_allocation_bytes=allocation_delta,
                             timing_attempts=attempts,
                             confirmation_attempts=confirmation_attempts,
                             ratio_direction="a16 / quantized; lower is faster"),
            ))
        return tuple(result)


class _Factory:
    def __call__(self, group_id, cases, context):
        return _Session(context)


class BlockscaledPrecisionGenerator(DiscreteSweepGenerator):
    """Race static dense specializations and emit exact measured-M routes.

    measured_m exists only in the offline corpus. The emitted planner query
    contains model geometry and recipe; live M never enters runtime resolution.
    """
    def __init__(self, *, cases=None):
        super().__init__(component_id=BLOCKSCALED_PRECISION,
                         query_schema_version=1, config_schema_version=1,
                         query_fields=(*QUERY_FIELDS, "measured_m"), range_fields=frozenset(),
                         cases=precision_cases() if cases is None else cases,
                         benchmark_factory=_Factory(), coverage={}, candidate_contract_version=3,
                         candidate_tie_breaker=lambda candidate: int(not candidate.config["a16_rows"]))

    def estimate(self, context):
        from dataclasses import replace
        estimate = super().estimate(context)
        return replace(estimate,
                       description="17 candidates per row count; balanced graph timing, correctness, and independent latency-parity confirmation",
                       dimensions={**estimate.dimensions, "candidates_per_case": 17,
                                   "maximum_candidate_measurements": 17 * len(self._cases)})

    def build_planner(self, records):
        grouped = defaultdict(list)
        for record in records:
            key = tuple(record.query[field] for field in QUERY_FIELDS)
            grouped[key].extend(record.config["a16_rows"])
        merged = tuple(DecisionRecord(query=FrozenMapping(dict(zip(QUERY_FIELDS, key, strict=False))),
                                      config=FrozenMapping({"a16_rows": sorted(rows)}))
                       for key, rows in grouped.items())
        return build_axis_tree(merged, field_order=QUERY_FIELDS, range_fields=frozenset())

    def _load_checkpoint(self, **kwargs):
        cached = super()._load_checkpoint(**kwargs)
        if cached is not None and not any(item.passes() for item in cached.measurements):
            if any(item.error == "clock or replay-allocation qualification failed"
                   for item in cached.measurements):
                return None
        return cached

    def generate(self, context, *, progress, checkpoints):
        from dataclasses import replace

        result = super().generate(context, progress=progress, checkpoints=checkpoints)
        queries = {tuple(case.query[field] for field in QUERY_FIELDS) for case in self._cases}
        component = dict(result.component)
        component["coverage"] = {**component["coverage"],
                                 "runtime_query_points": len(queries),
                                 "measured_row_counts": sorted({case.query["measured_m"] for case in self._cases})}
        return replace(result, component=component)
