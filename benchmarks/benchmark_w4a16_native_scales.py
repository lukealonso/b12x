#!/usr/bin/env python3
"""Interleave public A16 plans before and after native NVFP4 scale loading.

The baseline preparation and kernel come from a specified git revision. Both
arms use identical checkpoint operands, routing, launch policy, and toolchain.
Baseline code is loaded only in this benchmark process, never into a service.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
from importlib import metadata as package_metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import b12x
from b12x._lib.compiler import clear_compile_cache
from b12x.moe import fused_moe
from b12x.moe.fused_moe import _impl
from b12x.policy import MOE_DECODE, get_auto_policy
from benchmarks.benchmark_blockscaled_precision import _clock_checks, _paired, _snapshot
from benchmarks.benchmark_moe import get_quant_mode_params
from benchmarks.benchmark_w4a16_nvfp4_layouts import _check
from benchmarks.common import make_l2_flush_fn
from benchmarks.moe_checkpoint_snapshot import load_snapshot, tensor_digest
from tests._reference.w4a16_reference import moe_reference_w4a16


@contextmanager
def baseline_modules(revision, directory):
    import b12x.moe._shared.kernels.w4a16 as package
    saved = {}
    hashes = {}
    try:
        for name in ("prepare", "kernel"):
            canonical = f"{package.__name__}.{name}"
            importlib.import_module(canonical)
            source = subprocess.check_output([
                "git", "show", f"{revision}:b12x/moe/_shared/kernels/w4a16/{name}.py",
            ], cwd=ROOT)
            hashes[name] = hashlib.sha256(source).hexdigest()
            # Isolate dispatcher names so importing the baseline cannot replace
            # the production CUDA operator implementations in this process.
            source = source.replace(b'b12x::w4a16_', b'b12x_scale_baseline::w4a16_')
            source = source.replace(b'torch.ops.b12x.w4a16_', b'torch.ops.b12x_scale_baseline.w4a16_')
            path = directory / f"{name}.py"
            path.write_bytes(source)
            alias = f"{package.__name__}.baseline_{name}"
            spec = importlib.util.spec_from_file_location(alias, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[alias] = module
            spec.loader.exec_module(module)
            saved[name] = (sys.modules[canonical], getattr(package, name))
            sys.modules[canonical] = module
            setattr(package, name, module)
        _impl.clear_tp_moe_caches()
        clear_compile_cache()
        yield hashes
    finally:
        for name, (module, attribute) in saved.items():
            sys.modules[f"{package.__name__}.{name}"] = module
            setattr(package, name, attribute)
        _impl.clear_tp_moe_caches()
        clear_compile_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-snapshot", type=Path, required=True)
    parser.add_argument("--baseline-revision", default="HEAD")
    parser.add_argument("--capacities", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--route-mode", choices=("direct", "packed"), required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_revision = subprocess.check_output(
        ["git", "rev-parse", args.baseline_revision], cwd=ROOT, text=True,
    ).strip()
    packet = torch.load(args.input_snapshot, map_location="cpu", weights_only=True)
    metadata = packet["metadata"]
    del packet
    weights, inputs, digests = load_snapshot(args.input_snapshot, device="cuda:0", expected_metadata=metadata)
    params = get_quant_mode_params(weights, "shared", "w4a16")
    spec = weights.spec
    bundle = fused_moe.PackedWeights(
        w13=weights.w13_weight, w2=weights.w2_weight,
        w13_block_scales=weights.w13_blockscale_swizzled,
        w2_block_scales=weights.w2_blockscale_swizzled,
        w13_global_scales=params.g1_alphas, w2_global_scales=params.g2_alphas,
        input_scale=params.a1_gscale, intermediate_scale=params.a2_gscale,
    )
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format=weights.source_format, w13_layout=weights.w13_layout),
        activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.AUTO,
            nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=fused_moe.MoEGeometry(num_experts=spec.num_experts,
            hidden_size=spec.hidden_size, intermediate_size=spec.I_tp),
    )
    policy = get_auto_policy("cuda:0").with_override(MOE_DECODE, fused_moe.MoeDecodeConfig(
        backend="w4a16", route_planner="internal", max_active_clusters=None,
        w4a16_route_mode=args.route_mode,
    ))
    report = dict(command=[sys.executable, *sys.argv], tensor_digests=digests,
        baseline_revision=baseline_revision, route_mode=args.route_mode,
        worktree=str(ROOT), versions={name: package_metadata.version(name) for name in
                                     ("torch", "nvidia-cutlass-dsl")},
        source_sha256={name: hashlib.sha256((ROOT / f"b12x/moe/_shared/kernels/w4a16/{name}.py").read_bytes()).hexdigest()
                       for name in ("kernel", "prepare")}, cases=[],
        ratio="native-scales latency / baseline latency; lower is better")

    def capture(m, expected):
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        experts = fused_moe.prepare_weights(plan=weight_plan, weights=bundle)
        peak = torch.cuda.max_memory_allocated() - allocated
        native = experts._impl.representation_for("w4a16")
        unique_scales = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                         for t in (native.w13_scale, native.w2_scale, native.micro_w13_scale, native.micro_w2_scale)}
        plan = fused_moe.plan_execution(experts=experts,
            capacity=fused_moe.ExecutionCapacity(max_tokens=m, top_k=spec.top_k, warmup_token_counts=(m,)),
            policy=policy)
        fused_moe.prewarm(plan)
        scratch_spec, = plan.scratch_specs()
        scratch = torch.empty(scratch_spec.shape, dtype=scratch_spec.dtype, device="cuda")
        x, ids, routes = inputs[m]
        output = torch.empty_like(x)
        binding = fused_moe.bind(plan, scratch=scratch, a=x, experts=experts,
            topk_ids=ids, topk_weights=routes, output=output, input_scales_static=True)
        fused_moe.run(binding=binding)
        _check(output, expected, name="eager", m=m)
        graph = torch.cuda.CUDAGraph()
        warm_graph = torch.cuda.CUDAGraph()
        b12x.freeze_kernel_resolution("native scale before/after graph capture")
        try:
            with torch.cuda.graph(graph):
                fused_moe.run(binding=binding)
            with torch.cuda.graph(warm_graph):
                for _ in range(32):
                    fused_moe.run(binding=binding)
        finally:
            b12x.unfreeze_kernel_resolution()
        for captured in (graph, warm_graph):
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            captured.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            metrics = _check(output, expected, name="replay", m=m)
        return dict(graph=graph, warm_graph=warm_graph, output=output, owner=(experts, plan, scratch, binding),
                    metrics=metrics, preparation_peak_bytes=peak, scale_storage_bytes=sum(unique_scales.values()))

    with tempfile.TemporaryDirectory(prefix="b12x-scale-baseline-") as temporary:
        for m in args.capacities:
            x, ids, routes = inputs[m]
            expected = moe_reference_w4a16(x, bundle.w13, bundle.w13_block_scales,
                bundle.w13_global_scales, bundle.w2, bundle.w2_block_scales,
                bundle.w2_global_scales, ids, routes, spec.num_experts, spec.hidden_size, spec.I_tp)
            with baseline_modules(baseline_revision, Path(temporary)) as hashes:
                baseline = capture(m, expected)
            current = capture(m, expected)
            assert current["scale_storage_bytes"] == bundle.w13_block_scales.nbytes + bundle.w2_block_scales.nbytes
            report["baseline_source_sha256"] = hashes
            case = dict(capacity=m, memory={name: {k: arm[k] for k in
                ("preparation_peak_bytes", "scale_storage_bytes", "metrics")}
                for name, arm in (("baseline", baseline), ("native", current))}, timings={})
            graphs = {"baseline": baseline["graph"], "native": current["graph"]}
            b12x.freeze_kernel_resolution("native scale paired timing")
            try:
                for cache in ("cold", "warm"):
                    flush = make_l2_flush_fn(cache == "cold", 0)
                    replays = 32 if cache == "warm" else 1
                    timed_graphs = ({"baseline": baseline["warm_graph"], "native": current["warm_graph"]}
                                    if cache == "warm" else graphs)
                    attempts = []
                    for _ in range(3):
                        _paired(timed_graphs, 10, args.samples, flush)
                        before = _snapshot()
                        allocated = torch.cuda.memory_allocated()
                        totals = _paired(timed_graphs, 10, args.samples, flush)
                        allocation_delta = torch.cuda.memory_allocated() - allocated
                        after = _snapshot()
                        clocks = _clock_checks(before, after)
                        attempts.append(dict(total_samples_us=totals, before=before, after=after,
                            clock_validation=clocks, replay_allocation_bytes=allocation_delta))
                        if clocks["valid"] and allocation_delta == 0:
                            break
                    samples = [{name: value / replays for name, value in pair.items()} for pair in totals]
                    if not clocks["valid"] or allocation_delta:
                        case["timings"][cache] = dict(attempts=attempts, valid=False)
                        report["cases"].append(case)
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        raise RuntimeError(f"invalid timing: {clocks}, allocation delta {allocation_delta}")
                    medians = {name: statistics.median(p[name] for p in samples) for name in graphs}
                    case["timings"][cache] = dict(samples_us=samples, median_us=medians,
                        ratio=medians["native"] / medians["baseline"], before=before, after=after,
                        clock_validation=clocks, replay_allocation_bytes=allocation_delta,
                        attempts=attempts, production_replays_per_timing_graph=replays)
                    print(m, cache, medians, flush=True)
            finally:
                b12x.unfreeze_kernel_resolution()
            for arm in (baseline, current):
                arm["output"].fill_(float("nan"))
                arm["graph"].replay()
                _check(arm["output"], expected, name="post-timing", m=m)
                arm["graph"].reset()
                arm["warm_graph"].reset()
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name in ("w13_weight", "w2_weight", "w13_blockscale_swizzled", "w2_blockscale_swizzled"):
        assert tensor_digest(getattr(weights, name)) == digests[name]
    report["qualified"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
