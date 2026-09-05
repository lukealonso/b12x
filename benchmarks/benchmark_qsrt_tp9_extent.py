"""Compare a real QSRT extent at compact and zero-padded runtime widths.

This exercises the serving plan/prepare/bind/run path on one GPU. It does not
measure nine-GPU collectives or end-to-end serving latency. The checkpoint is
read-only and both arms use the identical source atoms, scales, and rotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from b12x.moe import fused_moe
from b12x.moe._shared.qsrt_sharding import plan_qsrt_tp9_rank
from vllm.model_executor.layers.quantization.kquant_qsrt_atoms_v2 import (
    open_qsrt_atom_v2_extent,
    read_qsrt_atom_v2_layer_metadata,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--block-m", type=int, default=8, choices=(8, 32, 48, 64, 96, 128))
    parser.add_argument(
        "--time-iters",
        type=int,
        default=0,
        help="when > 0, time the graph replay of each width (median of this many iterations)",
    )
    parser.add_argument(
        "--m-values",
        default="1,2,4,8,16,1536",
        help="comma-separated token counts; values above --max-tokens are skipped",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="TP9 rank whose extent is loaded (default: first rank with 256 channels)",
    )
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(71903)
    metadata = read_qsrt_atom_v2_layer_metadata(
        args.model / f"qsrt-layer-{args.layer:05d}.safetensors", layer=args.layer
    )
    if args.rank is None:
        rank = next(
            r
            for r in range(9)
            if plan_qsrt_tp9_rank(args.layer, r).intermediate_channels == 256
        )
    else:
        rank = args.rank
    m_values = tuple(int(value) for value in args.m_values.split(","))
    runtimes = {}
    for width in (384, 256):
        before = torch.cuda.memory_allocated()
        weight_plan = fused_moe.plan_weights(
            quant_modes="w4a16",
            source_format="qsrt_sqg_e4m3",
            activation="situ",
            params_dtype=torch.bfloat16,
            num_experts=896,
            hidden_size=3584,
            intermediate_size=width,
            w13_layout="w13",
            trellis_bits=2,
            trellis_tile_config=(128, 128, 128, 128),
            qsrt_storage_format="qsrt_atoms_v2",
            qsrt_profile=metadata.profile,
        )
        with open_qsrt_atom_v2_extent(
            metadata, shard_count=9, shard_index=rank, device=None
        ) as (first, atoms):
            weights = fused_moe.prepare_weights(
                plan=weight_plan,
                params_dtype=torch.bfloat16,
                qsrt_atom_payload=atoms,
                qsrt_first_atom_slot=first,
                qsrt_layer_index=args.layer,
                gate_suh=metadata.gate_suh.unsqueeze(0).cuda(),
                up_suh=metadata.up_suh.unsqueeze(0).cuda(),
                down_svh=metadata.down_svh.unsqueeze(0).cuda(),
                qsrt_rotation_draws=metadata.rotation_draws,
            )
        weight_bytes = torch.cuda.memory_allocated() - before
        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=args.max_tokens,
                num_topk=16,
                device=0,
                weight_plan=weights.plan,
                quant_mode="w4a16",
                route_num_experts=896,
                w4a16_block_size_m=args.block_m,
            )
        )
        spec = plan.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        output = torch.empty(
            (args.max_tokens, 3584), dtype=torch.float32, device="cuda"
        )
        runtimes[width] = (weights, plan, scratch, output, weight_bytes)
        print(
            f"width={width} source=[{first},{first + atoms.shape[0]}) "
            f"weight_bytes={weight_bytes}",
            flush=True,
        )
    expert_map = torch.arange(896, dtype=torch.int32, device="cuda")
    records = []
    for m in (value for value in m_values if value <= args.max_tokens):
        x = (torch.randn((m, 3584), device="cuda") * 0.1).to(torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(896, device="cuda")[:16] for _ in range(m)]
        ).to(torch.int32)
        routing = torch.softmax(torch.randn((m, 16), device="cuda"), dim=-1)
        outputs = {}
        timings: dict[int, float] = {}
        for width, (weights, plan, scratch, output, _) in runtimes.items():
            binding = fused_moe.bind(
                plan,
                scratch=scratch,
                a=x,
                experts=weights,
                topk_weights=routing,
                topk_ids=ids,
                route_expert_map=expert_map,
                output=output[:m],
            )
            eager = fused_moe.run(binding=binding).clone()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replay_output = fused_moe.run(binding=binding)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(replay_output, eager, rtol=0, atol=0)
            outputs[width] = eager
            if args.time_iters > 0:
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                samples = []
                for _ in range(args.time_iters):
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(begin.elapsed_time(end) * 1000.0)
                samples.sort()
                timings[width] = samples[len(samples) // 2]
            del graph
        reference = outputs[384]
        difference = outputs[256] - reference
        record = {
            "m": m,
            "finite": bool(torch.isfinite(outputs[256]).all().item()),
            "reference_norm": float(reference.norm().item()),
            "max_abs_error": float(difference.abs().max().item()),
            "relative_l2": float((difference.norm() / reference.norm()).item()),
            "cosine": float(
                torch.nn.functional.cosine_similarity(
                    outputs[256].flatten(), reference.flatten(), dim=0
                ).item()
            ),
            "graph_eager_exact": True,
        }
        if timings:
            record["graph_replay_us"] = {str(w): round(t, 1) for w, t in timings.items()}
            record["block_m"] = args.block_m
        # Digests let runs with different route blocks be compared for
        # bit-identical outputs without keeping the tensors.
        record["output_sha256"] = {
            str(width): hashlib.sha256(
                tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest()[:16]
            for width, tensor in outputs.items()
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    result = {
        "status": "research-only",
        "layer": args.layer,
        "tp9_rank": rank,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "runtime_weight_bytes": {str(w): values[-1] for w, values in runtimes.items()},
        "records": records,
        "limitation": "one actual source extent; no full-model or TP9 collective validation",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if any(
        not row["finite"] or row["reference_norm"] == 0 or row["relative_l2"] > 0.001
        for row in records
    ):
        raise RuntimeError("compact extent exceeded the declared 0.1% relative-L2 gate")


if __name__ == "__main__":
    main()
