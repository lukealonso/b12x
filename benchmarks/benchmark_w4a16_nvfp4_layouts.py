#!/usr/bin/env python3
"""Compare native and repacked NVFP4 through planned W4A16 MoE graph replay.

Both arms use one checkpoint layer, identical routed inputs, and the existing
W4A16 oracle and tolerances. Preparation and compilation are outside timing.
The native arm must share both weight allocations with the A4 preparation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

import b12x
from b12x.moe import fused_moe
from b12x.policy import MOE_DECODE, get_auto_policy
from benchmarks.benchmark_moe import (
    MODEL_PROFILES,
    bench_events,
    build_model_spec,
    check_oracle_metrics,
    compare_to_reference,
    get_quant_mode_params,
    load_expert_weights,
    make_oracle_reference,
    make_profile_routed_inputs,
)
from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weight_hashes(bundle) -> dict[str, str]:
    return {
        name: hashlib.sha256(tensor.view(torch.uint8).cpu().numpy()).hexdigest()
        for name, tensor in (
            ("w13", bundle.w13), ("w2", bundle.w2),
            ("w13_block_scales", bundle.w13_block_scales),
            ("w2_block_scales", bundle.w2_block_scales),
        )
    }


def _gpu_snapshot() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "-i", os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-gpu=uuid,pstate,clocks.sm,clocks.mem,power.draw,power.limit,clocks_event_reasons.active",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def _check(actual, expected, *, name: str, m: int, oracle_mode: str = "w4a16") -> dict:
    for label, value in ((name, actual), ("oracle", expected)):
        if not bool(torch.isfinite(value).all()) or not bool(value.count_nonzero()):
            raise AssertionError(
                f"{label}, M={m}: output must be finite and nonzero; "
                f"finite={int(torch.isfinite(value).sum())}/{value.numel()}, "
                f"nonzero={int(value.count_nonzero())}, "
                f"max_abs={float(value.abs().max())}"
            )
    metrics = compare_to_reference(actual, expected)
    failures = check_oracle_metrics(
        name, metrics, m, activation="silu", oracle_mode=oracle_mode,
    )
    if failures:
        raise AssertionError("\n".join(failures))
    return asdict(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-profile", choices=sorted(MODEL_PROFILES), default="glm52")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--a4-prefill-tokens", type=int, default=64)
    parser.add_argument("--cache", choices=("warm", "cold"), default="cold")
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        parser.error("set CUDA_VISIBLE_DEVICES to the single assigned GPU")
    if min(*args.batch_sizes, args.a4_prefill_tokens, args.warmup, args.iterations, args.rounds) <= 0:
        parser.error("batch sizes, warmup, iterations, and rounds must be positive")
    if args.capacity < max(args.batch_sizes):
        parser.error("capacity must cover every batch size")
    profile = MODEL_PROFILES[args.model_profile]
    if profile.shape is not None or profile.default_activation != "silu":
        parser.error("select a checkpoint profile with SiLU activation")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    spec = build_model_spec(args.model_path, profile, tp_size_override=args.tp_size)
    report = {
        "command": [sys.executable, *sys.argv],
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree": str(ROOT),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {key: value for key, value in os.environ.items() if key.startswith(("B12X_", "CUTE_DSL_", "CUDA_VISIBLE_DEVICES"))},
        "versions": {name: metadata.version(name) for name in ("torch", "nvidia-cutlass-dsl", "triton")},
        "source_sha256": {
            str(path): _sha256(ROOT / path)
            for path in (
                Path(__file__).relative_to(ROOT),
                Path("b12x/moe/_shared/kernels/w4a16/kernel.py"),
                Path("b12x/moe/_shared/kernels/w4a16/prepare.py"),
                Path("b12x/moe/_shared/kernels/micro.py"),
                Path("b12x/_lib/intrinsics.py"),
            )
        },
        "checkpoint": str(args.model_path.resolve()),
        "checkpoint_metadata_sha256": {
            name: _sha256(args.model_path / name)
            for name in ("config.json", "model.safetensors.index.json")
        },
        "layer": args.layer_idx,
        "shape": asdict(spec),
        "routing": profile.default_routing,
        "capacity": args.capacity,
        "cache": args.cache,
        "l2_flush_bytes": resolve_l2_flush_bytes(args.l2_flush_bytes) if args.cache == "cold" else 0,
        "timing": "per-replay CUDA events; L2 eviction excluded" if args.cache == "cold" else "CUDA events around consecutive graph replays",
        "ratio": "native latency / packed latency; lower is better for native",
        "cases": [],
    }

    def save() -> None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(f"Loading {args.model_path}, layer={args.layer_idx}, {spec}", flush=True)
    weights = load_expert_weights(
        args.model_path, layer_idx=args.layer_idx, checkpoint_family=profile.checkpoint_family, spec=spec,
        activation="silu",
    )
    if weights.source_format != "modelopt_nvfp4":
        raise ValueError(f"expected ModelOpt NVFP4, got {weights.source_format}")
    params = get_quant_mode_params(weights, "shared", "w4a16")
    a4_params = get_quant_mode_params(weights, "shared", "nvfp4")
    bundle = fused_moe.PackedWeights(
        w13=weights.w13_weight, w2=weights.w2_weight,
        w13_block_scales=weights.w13_blockscale_swizzled,
        w2_block_scales=weights.w2_blockscale_swizzled,
        w13_global_scales=params.g1_alphas, w2_global_scales=params.g2_alphas,
        input_scale=params.a1_gscale, intermediate_scale=params.a2_gscale,
    )
    source = fused_moe.PackedSource(format=weights.source_format, w13_layout=weights.w13_layout)
    report["native_weight_sha256"] = _weight_hashes(bundle)
    geometry = fused_moe.MoEGeometry(
        num_experts=spec.num_experts, hidden_size=spec.hidden_size,
        intermediate_size=spec.I_tp,
    )
    arms = {}
    for name, mode, packing in (
        ("a4", fused_moe.ActivationMode.A4, None),
        ("native", fused_moe.ActivationMode.AUTO, fused_moe.WeightPacking.SOURCE_NATIVE),
        ("packed", fused_moe.ActivationMode.A16, fused_moe.WeightPacking.MMA_PACKED),
    ):
        plan = fused_moe.plan_weights(
            source=source, geometry=geometry,
            activation=fused_moe.ActivationSpec(mode=mode, nonlinearity="silu", io_dtype=torch.bfloat16),
            constraints=fused_moe.WeightPlanConstraints(required_packing=packing),
        )
        # Packed preparation consumes its input storage; each A/B arm needs
        # independent ownership of every allocation that preparation rewrites.
        arm_bundle = (
            replace(bundle, w13=bundle.w13.clone(), w2=bundle.w2.clone())
            if name == "packed" else bundle
        )
        experts = fused_moe.prepare_weights(plan=plan, weights=arm_bundle)
        arms[name] = {"experts": experts}
    assert _weight_hashes(bundle) == report["native_weight_sha256"]
    for attr, original in (("w1_fp4", bundle.w13), ("w2_fp4", bundle.w2)):
        for name in ("a4", "native"):
            assert getattr(arms[name]["experts"]._impl, attr).data_ptr() == original.data_ptr()
    report["a4_native_share_weight_allocations"] = True
    native = arms["native"]["experts"]._impl.representation_for("w4a16")
    for micro_attr, a4_attr, original in (
        ("micro_w13_scale", "w1_blockscale", bundle.w13_block_scales),
        ("micro_w2_scale", "w2_blockscale", bundle.w2_block_scales),
    ):
        assert getattr(native, micro_attr).data_ptr() == original.data_ptr()
        assert getattr(arms["a4"]["experts"]._impl, a4_attr).data_ptr() == original.data_ptr()
    report["a4_native_share_decode_block_scales"] = True
    assert native.w13_scale.data_ptr() == bundle.w13_block_scales.data_ptr()
    assert native.w2_scale.data_ptr() == bundle.w2_block_scales.data_ptr()
    for name in ("native", "packed"):
        arm = arms[name]
        plan = fused_moe.plan_execution(
            experts=arm["experts"],
            policy=(get_auto_policy(device).with_override(MOE_DECODE, fused_moe.MoeDecodeConfig(
                backend="w4a16", route_planner="internal", max_active_clusters=None,
                w4a16_route_mode="direct",
            )) if name == "native" else None),
            capacity=fused_moe.ExecutionCapacity(
                max_tokens=args.capacity, top_k=spec.top_k,
                warmup_token_counts=tuple(args.batch_sizes),
            ),
        )
        print(f"Prewarming {name}", flush=True)
        fused_moe.prewarm(plan)
        scratch_spec, = plan.scratch_specs()
        arm["plan"] = plan
        arm["scratch"] = torch.empty(scratch_spec.shape, device=device, dtype=scratch_spec.dtype)
    save()
    l2_flush = make_l2_flush_fn(args.cache == "cold", args.l2_flush_bytes)
    replay_cases = []
    for m in args.batch_sizes:
        x, ids, route_weights = make_profile_routed_inputs(profile, weights, spec, m, 42, device)
        expected = make_oracle_reference(
            "w4a16", "w4a16", x, weights, params, ids, route_weights, activation="silu",
        )
        case = {"tokens": m, "metrics": {}, "plans": {}, "samples_us": {name: [] for name in ("native", "packed")}, "replay_samples_us": {name: [] for name in ("native", "packed")}, "gpu": []}
        for name in ("native", "packed"):
            arm = arms[name]
            out = torch.empty_like(x)
            binding = fused_moe.bind(
                arm["plan"], scratch=arm["scratch"], a=x, experts=arm["experts"],
                topk_weights=route_weights, topk_ids=ids, output=out, input_scales_static=True,
            )
            case["plans"][name] = {
                "variant": asdict(arm["plan"].variant_for(m)._impl.policy_resolution.config),
                "bound": asdict(binding.execution_plan.policy_resolution.config),
                "bound_capacity": binding.execution_plan.routed_rows // spec.top_k,
            }
            for _ in range(args.warmup):
                with torch.cuda.nvtx.range(name):
                    fused_moe.run(binding=binding)
            torch.cuda.synchronize()
            case["metrics"][name] = _check(out, expected, name=name, m=m)
            graph = torch.cuda.CUDAGraph()
            b12x.freeze_kernel_resolution("NVFP4 A16 graph capture qualification")
            try:
                with torch.cuda.graph(graph):
                    fused_moe.run(binding=binding)
            finally:
                b12x.unfreeze_kernel_resolution()
            arm["graph"] = graph
            arm["binding"] = binding
            arm["output"] = out
            for _ in range(args.warmup):
                graph.replay()
            torch.cuda.synchronize()
            _check(out, expected, name=name + " graph", m=m)
        allocated = torch.cuda.memory_allocated()
        for round_idx in range(args.rounds):
            case["gpu"].append(_gpu_snapshot())
            order = ("native", "packed") if round_idx % 2 == 0 else ("packed", "native")
            for name in order:
                graph = arms[name]["graph"]
                if l2_flush is not None:
                    samples = [value * 1000 for value in bench_events(
                        graph.replay, warmup=args.warmup, iters=args.iterations,
                        l2_flush=l2_flush,
                    )]
                    case["replay_samples_us"][name].append(samples)
                    case["samples_us"][name].append(statistics.median(samples))
                    continue
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    graph.replay()
                end.record()
                end.synchronize()
                case["samples_us"][name].append(start.elapsed_time(end) * 1000 / args.iterations)
        assert torch.cuda.memory_allocated() == allocated
        for name in ("native", "packed"):
            _check(arms[name]["output"], expected, name=name + " replay", m=m)
        case["gpu"].append(_gpu_snapshot())
        medians = {name: statistics.median(values) for name, values in case["samples_us"].items()}
        case["median_us"] = medians
        case["native_over_packed"] = medians["native"] / medians["packed"]
        report["cases"].append(case)
        replay_cases.append((case, expected, {
            name: {key: arms[name][key] for key in ("binding", "graph", "output")}
            for name in ("native", "packed")
        }))
        save()
        print(f"M={m}: native={medians['native']:.2f} us, packed={medians['packed']:.2f} us, native/packed={case['native_over_packed']:.3f}", flush=True)
    assert _weight_hashes(bundle) == report["native_weight_sha256"]
    report["native_weights_unchanged"] = True
    report["a16_qualified"] = True
    save()
    a4 = arms["a4"]
    a4["plan"] = fused_moe.plan_execution(
        experts=a4["experts"],
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=args.a4_prefill_tokens, top_k=spec.top_k,
            warmup_token_counts=(args.a4_prefill_tokens,),
        ),
    )
    print("Prewarming A4 prefill", flush=True)
    fused_moe.prewarm(a4["plan"])
    a4_spec, = a4["plan"].scratch_specs()
    a4["scratch"] = torch.empty(a4_spec.shape, device=device, dtype=a4_spec.dtype)
    a4_x, a4_ids, a4_weights = make_profile_routed_inputs(
        profile, weights, spec, args.a4_prefill_tokens, 42, device,
    )
    a4["output"] = torch.empty_like(a4_x)
    a4["binding"] = fused_moe.bind(
        a4["plan"], scratch=a4["scratch"], a=a4_x, experts=a4["experts"],
        topk_weights=a4_weights, topk_ids=a4_ids, output=a4["output"],
        input_scales_static=True,
    )
    a4_expected = make_oracle_reference(
        "nvfp4", "nvfp4", a4_x, weights, a4_params, a4_ids, a4_weights, activation="silu",
    )
    for _ in range(args.warmup):
        fused_moe.run(binding=a4["binding"])
    report["a4_prefill"] = {
        "tokens": args.a4_prefill_tokens,
        "metrics": asdict(compare_to_reference(a4["output"], a4_expected)),
        "passed": False,
    }
    save()
    qualification_failures = []

    def check_a4(label: str) -> bool:
        try:
            _check(
                a4["output"], a4_expected, name=label,
                m=args.a4_prefill_tokens, oracle_mode="nvfp4",
            )
        except AssertionError as error:
            qualification_failures.append(str(error))
            return False
        return True

    report["a4_prefill"]["passed"] = check_a4("a4 prefill")
    a4["graph"] = torch.cuda.CUDAGraph()
    b12x.freeze_kernel_resolution("NVFP4 A4/A16 precision-switch qualification")
    try:
        with torch.cuda.graph(a4["graph"]):
            fused_moe.run(binding=a4["binding"])
        allocated = torch.cuda.memory_allocated()
        for case, expected, decode in replay_cases:
            switch_passed = True
            case["a4_switch_metrics"] = []
            for name in ("a4", "native", "a4", "native", "a4"):
                arm = a4 if name == "a4" else decode[name]
                arm["output"].fill_(float("nan"))
                arm["graph"].replay()
                if name == "a4":
                    switch_passed = check_a4("a4 precision switch") and switch_passed
                    case["a4_switch_metrics"].append(
                        asdict(compare_to_reference(arm["output"], a4_expected))
                    )
                else:
                    _check(
                        arm["output"], expected, name="a16 after a4",
                        m=case["tokens"],
                    )
            case["a16_after_a4_oracle_passed"] = True
            case["precision_switch_passed"] = switch_passed
        assert torch.cuda.memory_allocated() == allocated
    finally:
        b12x.unfreeze_kernel_resolution()
    assert _weight_hashes(bundle) == report["native_weight_sha256"]
    report["native_weights_unchanged"] = True
    report["qualification_failures"] = sorted(set(qualification_failures))
    save()
    if qualification_failures:
        raise AssertionError("\n".join(report["qualification_failures"]))


if __name__ == "__main__":
    main()
