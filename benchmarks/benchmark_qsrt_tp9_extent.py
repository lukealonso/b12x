"""Compare a real QSRT extent at compact and zero-padded runtime widths.

This exercises the serving plan/prepare/bind/run path on one GPU. It does not
measure nine-GPU collectives or end-to-end serving latency. The checkpoint is
read-only and both arms use the identical source atoms, scales, and rotations.

Routing comes from one of three sources: the default seeded uniform synthetic
routing (16 distinct experts per token, all experts equally likely), a
seeded Zipf-skewed synthetic routing (``--routing zipf:<s>``; expert
popularity proportional to ``rank**-s`` over a random expert permutation, a
stand-in for over-dispersed production traffic), or a captured routing
(``--topk-ids``, a ``[tokens, 16]`` int tensor saved with ``torch.save``,
optionally a dict holding ``topk_ids`` and ``topk_weights``). A captured
routing fixes the token count. Every record carries the per-expert routed-row
histogram and the route-block counts it implies, so a route block can be
chosen from the traffic it will serve, and the output digests let two runs
with different route blocks be compared for bit-identical outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

NUM_EXPERTS = 896
TOP_K = 16
HIDDEN = 3584
_HISTOGRAM_BLOCKS = (48, 64)


def _uniform_topk_ids(m: int, device: torch.device) -> torch.Tensor:
    return torch.stack(
        [torch.randperm(NUM_EXPERTS, device=device)[:TOP_K] for _ in range(m)]
    ).to(torch.int32)


def _zipf_topk_ids(m: int, exponent: float, device: torch.device) -> torch.Tensor:
    """Sample 16 distinct experts per token with Zipf(s) expert popularity."""
    if exponent <= 0:
        raise ValueError("zipf exponent must be positive")
    ranks = torch.arange(1, NUM_EXPERTS + 1, dtype=torch.float64)
    popularity = ranks.pow(-float(exponent))
    weights = torch.empty(NUM_EXPERTS, dtype=torch.float64)
    weights[torch.randperm(NUM_EXPERTS)] = popularity
    weights = weights.to(device=device, dtype=torch.float32)
    ids = torch.multinomial(weights.unsqueeze(0).repeat(m, 1), TOP_K, replacement=False)
    return ids.to(torch.int32)


def _load_topk_ids(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    payload = torch.load(path, map_location="cpu")
    weights = None
    if isinstance(payload, dict):
        weights = payload.get("topk_weights")
        payload = payload["topk_ids"]
    ids = torch.as_tensor(payload)
    if ids.ndim != 2 or ids.shape[1] != TOP_K:
        raise ValueError(
            f"{path}: expected a [tokens, {TOP_K}] expert-id tensor, got {tuple(ids.shape)}"
        )
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{path}: expert ids must be int32 or int64, got {ids.dtype}")
    if int(ids.min()) < 0 or int(ids.max()) >= NUM_EXPERTS:
        raise ValueError(f"{path}: expert ids must lie in [0, {NUM_EXPERTS})")
    ids = ids.to(device=device, dtype=torch.int32).contiguous()
    if weights is not None:
        weights = torch.as_tensor(weights)
        if tuple(weights.shape) != tuple(ids.shape):
            raise ValueError(
                f"{path}: topk_weights shape {tuple(weights.shape)} != ids {tuple(ids.shape)}"
            )
        weights = weights.to(device=device, dtype=torch.float32).contiguous()
    return ids, weights


def _routing_histogram(ids: torch.Tensor, chosen_block: int) -> dict:
    """Per-expert routed-row statistics and the route blocks they imply."""
    rows = torch.bincount(ids.reshape(-1).to(torch.int64), minlength=NUM_EXPERTS)
    rows = rows.to(torch.float64).cpu()
    quantiles = torch.quantile(rows, torch.tensor([0.5, 0.9], dtype=torch.float64))
    routes = int(rows.sum().item())
    summary = {
        "tokens": int(ids.shape[0]),
        "routes": routes,
        "rows_per_expert": {
            "mean": round(float(rows.mean().item()), 2),
            "p50": float(quantiles[0].item()),
            "p90": float(quantiles[1].item()),
            "max": int(rows.max().item()),
            "empty_experts": int((rows == 0).sum().item()),
        },
        "blocks": {},
    }
    for block in sorted({int(chosen_block), *_HISTOGRAM_BLOCKS}):
        needed = torch.ceil(rows / block)
        total_blocks = int(needed.sum().item())
        summary["blocks"][str(block)] = {
            "total_blocks": total_blocks,
            "padded_slots": total_blocks * block,
            "padding_fraction": round(total_blocks * block / max(routes, 1) - 1.0, 4),
            "experts_needing": {
                "0": round(float((needed == 0).double().mean().item()), 4),
                "1": round(float((needed == 1).double().mean().item()), 4),
                "2": round(float((needed == 2).double().mean().item()), 4),
                "3": round(float((needed == 3).double().mean().item()), 4),
                "4+": round(float((needed >= 4).double().mean().item()), 4),
            },
        }
    return summary


def _print_histogram(summary: dict, chosen_block: int) -> None:
    rows = summary["rows_per_expert"]
    print(
        f"routing: tokens={summary['tokens']} routes={summary['routes']} "
        f"rows/expert mean={rows['mean']} p50={rows['p50']} p90={rows['p90']} "
        f"max={rows['max']} empty={rows['empty_experts']}",
        flush=True,
    )
    for block, stats in summary["blocks"].items():
        marker = " (chosen)" if int(block) == int(chosen_block) else ""
        needing = " ".join(f"{k}:{v:.3f}" for k, v in stats["experts_needing"].items())
        print(
            f"  block {block}{marker}: total_blocks={stats['total_blocks']} "
            f"padding={stats['padding_fraction']:+.3f} experts_needing_blocks {needing}",
            flush=True,
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
    parser.add_argument(
        "--routing",
        default="uniform",
        help=(
            "synthetic routing: 'uniform' (16 distinct experts per token, seed "
            "71903) or 'zipf:<s>' (expert popularity ~ rank**-s, over-dispersed)"
        ),
    )
    parser.add_argument(
        "--topk-ids",
        type=Path,
        default=None,
        help=(
            "replay a captured routing: torch.save'd [tokens, 16] int32/int64 "
            "expert ids (or a dict with 'topk_ids' and optional 'topk_weights'); "
            "the token count comes from the file and overrides --m-values"
        ),
    )
    args = parser.parse_args()
    # The checkpoint reader lives in the vLLM fork; the routing helpers above
    # stay importable without it.
    from b12x.moe import fused_moe
    from b12x.moe._shared.qsrt_sharding import plan_qsrt_tp9_rank
    from vllm.model_executor.layers.quantization.kquant_qsrt_atoms_v2 import (
        open_qsrt_atom_v2_extent,
        read_qsrt_atom_v2_layer_metadata,
    )

    torch.cuda.set_device(0)
    torch.manual_seed(71903)
    device = torch.device("cuda")
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
    captured_ids = None
    captured_weights = None
    if args.topk_ids is not None:
        captured_ids, captured_weights = _load_topk_ids(args.topk_ids, device)
        m_values = (int(captured_ids.shape[0]),)
        if m_values[0] > args.max_tokens:
            print(
                f"--max-tokens {args.max_tokens} raised to the captured token "
                f"count {m_values[0]}",
                flush=True,
            )
            args.max_tokens = m_values[0]
    else:
        m_values = tuple(int(value) for value in args.m_values.split(","))
    zipf_exponent = None
    if args.routing != "uniform":
        kind, _, value = args.routing.partition(":")
        if kind != "zipf" or not value:
            raise ValueError("--routing must be 'uniform' or 'zipf:<s>'")
        zipf_exponent = float(value)
    runtimes = {}
    for width in (384, 256):
        before = torch.cuda.memory_allocated()
        weight_plan = fused_moe.plan_weights(
            quant_modes="w4a16",
            source_format="qsrt_sqg_e4m3",
            activation="situ",
            params_dtype=torch.bfloat16,
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN,
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
                num_topk=TOP_K,
                device=0,
                weight_plan=weights.plan,
                quant_mode="w4a16",
                route_num_experts=NUM_EXPERTS,
                w4a16_block_size_m=args.block_m,
            )
        )
        spec = plan.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        output = torch.empty(
            (args.max_tokens, HIDDEN), dtype=torch.float32, device="cuda"
        )
        runtimes[width] = (weights, plan, scratch, output, weight_bytes)
        print(
            f"width={width} source=[{first},{first + atoms.shape[0]}) "
            f"weight_bytes={weight_bytes} scratch_bytes={scratch.numel() * scratch.element_size()}",
            flush=True,
        )
    expert_map = torch.arange(NUM_EXPERTS, dtype=torch.int32, device="cuda")
    records = []
    for m in (value for value in m_values if value <= args.max_tokens):
        x = (torch.randn((m, HIDDEN), device="cuda") * 0.1).to(torch.bfloat16)
        if captured_ids is not None:
            ids = captured_ids
            routing_source = f"captured:{args.topk_ids}"
        elif zipf_exponent is not None:
            ids = _zipf_topk_ids(m, zipf_exponent, device)
            routing_source = f"zipf:{zipf_exponent}"
        else:
            ids = _uniform_topk_ids(m, device)
            routing_source = "uniform"
        if captured_weights is not None:
            routing = captured_weights
        else:
            routing = torch.softmax(torch.randn((m, TOP_K), device="cuda"), dim=-1)
        histogram = _routing_histogram(ids, args.block_m)
        _print_histogram(histogram, args.block_m)
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
            "routing": routing_source,
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
            "block_m": args.block_m,
            "routing_histogram": histogram,
        }
        if timings:
            record["graph_replay_us"] = {str(w): round(t, 1) for w, t in timings.items()}
        # Digests let runs with different route blocks be compared for
        # bit-identical outputs without keeping the tensors.
        record["output_sha256"] = {
            str(width): hashlib.sha256(
                tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest()[:16]
            for width, tensor in outputs.items()
        }
        record["topk_ids_sha256"] = hashlib.sha256(
            ids.contiguous().cpu().numpy().tobytes()
        ).hexdigest()[:16]
        records.append(record)
        print(json.dumps(record), flush=True)
    result = {
        "status": "research-only",
        "layer": args.layer,
        "tp9_rank": rank,
        "block_m": args.block_m,
        "routing": (
            f"captured:{args.topk_ids}"
            if captured_ids is not None
            else args.routing
        ),
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
