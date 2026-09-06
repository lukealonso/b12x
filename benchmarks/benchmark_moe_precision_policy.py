#!/usr/bin/env python3
"""Run the registered MoE precision generator for a bounded geometry corpus.

This uses the generator's deterministic synthetic weights and route patterns.
The output is a partial component profile; raw candidate gates and paired
CUDA graph samples remain in the work directory's resumable checkpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b12x.policy.device import detect_device
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.moe_corpus import expand_physical_geometries, expand_sweep_cases
from b12x.policy.generation.moe_corpus import MoeSweepCase
from b12x.policy.generation.progress import NullProgressReporter
from b12x.policy.generation.providers.moe import MoeDecodeGenerator
from b12x.policy.generation.providers.moe_precision import source_identity
from b12x.policy.generation.store import CheckpointStore


class Progress(NullProgressReporter):
    def advance(self, component_id, *, units=1, detail=None):
        if detail:
            print(detail, flush=True)


def snapshot_factory(weights, inputs):
    import torch
    from b12x.moe import fused_moe
    from b12x.policy.generation.providers.moe_gpu_worker import _MoeGeometrySession, _QueryInputs
    from benchmarks.benchmark_moe import get_quant_mode_params

    class Session(_MoeGeometrySession):
        def __init__(self, geometry, context):
            super().__init__(geometry, context)
            params = get_quant_mode_params(weights, "shared", "w4a16")
            plan = fused_moe.plan_weights(
                source=fused_moe.PackedSource(format=weights.source_format, w13_layout=weights.w13_layout),
                geometry=fused_moe.MoEGeometry(num_experts=geometry.num_experts,
                    hidden_size=geometry.hidden_size, intermediate_size=geometry.intermediate_size),
                activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.AUTO,
                    nonlinearity=geometry.activation, io_dtype=torch.bfloat16),
            )
            self._experts = fused_moe.prepare_weights(plan=plan, weights=fused_moe.PackedWeights(
                w13=weights.w13_weight, w2=weights.w2_weight,
                w13_block_scales=weights.w13_blockscale_swizzled, w2_block_scales=weights.w2_blockscale_swizzled,
                w13_global_scales=params.g1_alphas, w2_global_scales=params.g2_alphas,
                input_scale=params.a1_gscale, intermediate_scale=params.a2_gscale,
            ))

        def _stage_query_inputs(self, case):
            self._clear_query_state()
            x, ids, routes = inputs[case.num_tokens]
            self._query_inputs = _QueryInputs(x=x, topk_ids=ids, topk_weights=routes)
            return self._query_inputs

    return Session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--capacities", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--route-patterns", nargs="+", default=["balanced", "hot", "zipf", "disjoint"])
    parser.add_argument("--cache", choices=("cold", "warm"), default="cold")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--input-snapshot", type=Path,
                        help="race the exact checkpoint operands exported by benchmark_nvfp4_decode_precisions.py")
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        parser.error("set CUDA_VISIBLE_DEVICES to one assigned GPU")
    geometries = tuple(
        geometry for geometry in expand_physical_geometries()
        if geometry.recipe.quant_mode == "nvfp4_auto"
        and (geometry.hidden_size, geometry.intermediate_size, geometry.num_experts)
        == (args.hidden_size, args.intermediate_size, args.experts)
    )
    if len(geometries) != 1:
        parser.error("select one geometry from the reviewed MoE corpus")
    cases = expand_sweep_cases(
        geometries=geometries, token_counts=tuple(args.capacities),
        top_ks=(args.top_k,), route_patterns=tuple(args.route_patterns),
    )
    context = GenerationContext(
        device=detect_device("cuda:0").identity, device_ordinal=0,
        work_dir=args.work_dir,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        settings=GenerationSettings(warmup=args.warmup, groups=1, repetitions=args.samples,
                                    cold_l2=args.cache == "cold"),
    )
    benchmark_factory = None
    input_identity = {"kind": "deterministic synthetic generator weights and routes"}
    if args.input_snapshot:
        import torch
        from benchmarks.moe_checkpoint_snapshot import load_snapshot

        packet = torch.load(args.input_snapshot, map_location="cpu", weights_only=True)
        metadata = packet["metadata"]
        del packet
        weights, inputs, digests = load_snapshot(args.input_snapshot, device="cuda:0", expected_metadata=metadata)
        spec = weights.spec
        if (spec.hidden_size, spec.I_tp, spec.num_experts, spec.top_k) != (
            args.hidden_size, args.intermediate_size, args.experts, args.top_k,
        ) or not set(args.capacities) <= set(inputs):
            parser.error("snapshot geometry or token capacities do not match the requested corpus")
        identity = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
        cases = tuple(MoeSweepCase(
            case_id=f"checkpoint-{identity[:16]}-m{count}", geometry=geometries[0],
            top_k=args.top_k, num_tokens=count, route_pattern=f"checkpoint_{metadata['routing']}",
        ) for count in args.capacities)
        benchmark_factory = snapshot_factory(weights, inputs)
        input_identity = {"kind": "exact checkpoint benchmark snapshot", "metadata": metadata,
                          "tensor_digests": digests, "tensor_manifest_sha256": identity}
    generator = MoeDecodeGenerator(geometries=geometries, cases=cases, benchmark_factory=benchmark_factory)
    estimate = generator.estimate(context)
    print(json.dumps(asdict(estimate), indent=2), flush=True)
    if args.estimate_only:
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "command": [sys.executable, *sys.argv], "worktree": str(ROOT),
        "generation": context.checkpoint_metadata(), "estimate": asdict(estimate),
        "source_toolchain_sha256": source_identity(),
        "inputs": input_identity,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = generator.generate(context, progress=Progress(), checkpoints=CheckpointStore(args.work_dir / "checkpoints"))
    (args.work_dir / "result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    print(json.dumps(result.evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
