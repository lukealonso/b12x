"""Route-block cost model for the W4A16 fused MoE prefill launch.

The fused megakernel gives one CTA the whole ``[route block x tile_n]`` output
tile of one expert and walks the full K range, so a launch decodes each
expert's weight slab once per route block that expert owns:

    decoded weights = sum_e ceil(rows_e / block) * N * K

while the tensor-core work is the padded row count,
``sum_e ceil(rows_e / block) * block * N * K * 2`` FLOP. Raising the route block
therefore trades decode issue work (the measured bound at 4,608 tokens) against
MMA work and padding. Whether that trade pays depends entirely on the routed
rows-per-expert distribution: a block just above the typical row count collapses
two blocks per expert into one at unchanged padding, a block below it changes
nothing, and a block far above it only adds padding.

This model computes those counts for a routing histogram and, when measured
route-block timings are supplied, calibrates a three-parameter launch model

    T(block, width) = C + blocks(block) * Kt(width) * (alpha + beta * block)

    C      launch cost independent of the route block (input rotation,
           router-order sum, route packing)
    alpha  per k-tile cost that does not scale with the block (trellis decode
           issue, shared-memory weight loads, pipeline waits, the tile
           prologue and epilogue amortized over the tile's k-tiles)
    beta   per k-tile cost per routed row (MMA and A-fragment loads)

    Kt(width) = k-tile iterations one route block costs

and predicts unmeasured block sizes. Least squares over all supplied points;
the residuals are printed so an ill-fitting model is visible. Because a route
block that spills registers pays that cost per row, a beta fitted across a
spilling block over-states the row cost and the prediction for a larger block
is then a lower bound on the gain.

CPU only: no CUDA device, no kernel launch.

Examples:

    python benchmarks/model_w4a16_route_block_cost.py \
      --routing uniform --tokens 4608 --blocks 32,48,64,96,128

    python benchmarks/model_w4a16_route_block_cost.py \
      --routing uniform --tokens 4608 --blocks 48,64,96,128 \
      --measured uniform:48:384=7756.4 --measured uniform:48:256=6104.1 \
      --measured uniform:64:384=8371.2 --measured uniform:64:256=6548.5 \
      --measured-blocks uniform:48=1849 --measured-blocks uniform:64=1777
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

NUM_EXPERTS = 896
TOP_K = 16
HIDDEN = 3584
TILE_N = 128
TILE_K = 128


def rows_per_expert_uniform(tokens: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(NUM_EXPERTS, generator=generator)[:TOP_K] for _ in range(tokens)]
    )
    return torch.bincount(ids.reshape(-1), minlength=NUM_EXPERTS)


def rows_per_expert_zipf(tokens: int, exponent: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    ranks = torch.arange(1, NUM_EXPERTS + 1, dtype=torch.float64)
    popularity = ranks.pow(-float(exponent))
    weights = torch.empty(NUM_EXPERTS, dtype=torch.float64)
    weights[torch.randperm(NUM_EXPERTS, generator=generator)] = popularity
    ids = torch.multinomial(
        weights.to(torch.float32).unsqueeze(0).repeat(tokens, 1),
        TOP_K,
        replacement=False,
        generator=generator,
    )
    return torch.bincount(ids.reshape(-1), minlength=NUM_EXPERTS)


def rows_per_expert_poisson(tokens: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    mean = tokens * TOP_K / NUM_EXPERTS
    return torch.poisson(torch.full((NUM_EXPERTS,), mean), generator=generator).to(torch.int64)


def rows_per_expert_from_capture(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        payload = payload["topk_ids"]
    ids = torch.as_tensor(payload).reshape(-1).to(torch.int64)
    return torch.bincount(ids, minlength=NUM_EXPERTS)


def gemm_shapes(width: int) -> list[tuple[str, int, int]]:
    """(name, N, K) of the two GEMMs one fused launch runs at this rank width."""
    return [("fc1", 2 * width, HIDDEN), ("fc2", HIDDEN, width)]


def tiles_per_block(width: int) -> tuple[int, int]:
    """(CTA tiles, k-tile iterations) one route block costs at this width."""
    cta_tiles = 0
    k_tile_iters = 0
    for _, n, k in gemm_shapes(width):
        n_tiles = n // TILE_N
        k_tiles = k // TILE_K
        cta_tiles += n_tiles
        k_tile_iters += n_tiles * k_tiles
    return cta_tiles, k_tile_iters


def block_stats(rows: torch.Tensor, block: int, width: int) -> dict:
    rows64 = rows.to(torch.int64)
    routes = int(rows64.sum().item())
    needed = torch.div(rows64 + block - 1, block, rounding_mode="floor")
    total_blocks = int(needed.sum().item())
    non_empty = int((rows64 > 0).sum().item())
    cta_tiles, k_tile_iters = tiles_per_block(width)
    weights_per_expert = sum(n * k for _, n, k in gemm_shapes(width))
    return {
        "block_m": block,
        "total_blocks": total_blocks,
        "blocks_per_used_expert": round(total_blocks / max(non_empty, 1), 4),
        "padded_rows": total_blocks * block,
        "padding_fraction": round(total_blocks * block / max(routes, 1) - 1.0, 4),
        "decoded_weights": total_blocks * weights_per_expert,
        "padded_mma_flop": total_blocks * block * weights_per_expert * 2,
        "effective_flop": routes * weights_per_expert * 2,
        "cta_tiles": total_blocks * cta_tiles,
        "k_tile_iters": total_blocks * k_tile_iters,
    }


def calibrate(points: list[tuple[int, int, float, int]]) -> dict | None:
    """Least-squares fit of (C, alpha, beta) to (block, width, micros, blocks).

    The per-CTA-tile term is folded into ``alpha``: with only two distinct
    route blocks measured, a separate tile term makes the design matrix
    collinear and the fitted parameters meaningless even though the residuals
    vanish. Three parameters over four points leaves one degree of freedom, so
    the residuals reported here are a real check on the model.
    """
    if len(points) < 3:
        return None
    import numpy as np

    design = []
    target = []
    for block, width, micros, blocks_count in points:
        _, k_tile_iters = tiles_per_block(width)
        design.append(
            [1.0, blocks_count * k_tile_iters, blocks_count * k_tile_iters * block]
        )
        target.append(micros)
    matrix = np.asarray(design, dtype=np.float64)
    vector = np.asarray(target, dtype=np.float64)
    solution, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
    residual = matrix @ solution - vector
    return {
        "points": [
            {"block_m": b, "width": w, "measured_us": t, "total_blocks": n}
            for b, w, t, n in points
        ],
        "C_us": float(solution[0]),
        "alpha_us_per_k_tile": float(solution[1]),
        "beta_us_per_k_tile_per_row": float(solution[2]),
        "residual_us": [round(float(value), 2) for value in residual],
        "max_abs_residual_us": round(float(abs(residual).max()), 2),
    }


def predict(fit: dict, block: int, width: int, blocks_count: int) -> float:
    _, k_tile_iters = tiles_per_block(width)
    return (
        fit["C_us"]
        + blocks_count * k_tile_iters * fit["alpha_us_per_k_tile"]
        + blocks_count * k_tile_iters * block * fit["beta_us_per_k_tile_per_row"]
    )


def parse_measured(values: list[str]) -> dict[str, list[tuple[int, int, float]]]:
    """``routing:block:width=micros`` into {routing: [(block, width, micros)]}."""
    parsed: dict[str, list[tuple[int, int, float]]] = {}
    for value in values:
        key, _, micros = value.partition("=")
        routing, block, width = key.split(":")
        parsed.setdefault(routing, []).append((int(block), int(width), float(micros)))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--routing",
        default="uniform",
        help="uniform | poisson | zipf:<s> | capture (with --topk-ids)",
    )
    parser.add_argument("--topk-ids", type=Path, default=None)
    parser.add_argument("--tokens", type=int, default=4608)
    parser.add_argument("--seed", type=int, default=71903)
    parser.add_argument("--blocks", default="32,48,64,96,128")
    parser.add_argument("--widths", default="384,256")
    parser.add_argument(
        "--measured",
        action="append",
        default=[],
        help="routing:block:width=microseconds, repeatable; three or more "
        "points calibrate the launch model",
    )
    parser.add_argument(
        "--measured-blocks",
        action="append",
        default=[],
        help="routing:block=total_blocks, repeatable; the route-block count "
        "the measured run actually built (from the harness histogram), used "
        "instead of this model's synthetic count when calibrating",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.topk_ids is not None:
        rows = rows_per_expert_from_capture(args.topk_ids)
        routing_name = f"capture:{args.topk_ids.name}"
    elif args.routing.startswith("zipf"):
        exponent = float(args.routing.split(":", 1)[1]) if ":" in args.routing else 1.0
        rows = rows_per_expert_zipf(args.tokens, exponent, args.seed)
        routing_name = args.routing
    elif args.routing == "poisson":
        rows = rows_per_expert_poisson(args.tokens, args.seed)
        routing_name = "poisson"
    else:
        rows = rows_per_expert_uniform(args.tokens, args.seed)
        routing_name = "uniform"

    blocks = [int(value) for value in args.blocks.split(",") if value.strip()]
    widths = [int(value) for value in args.widths.split(",") if value.strip()]
    rows64 = rows.to(torch.int64)
    routes = int(rows64.sum().item())

    report = {
        "routing": routing_name,
        "tokens": int(args.tokens if args.topk_ids is None else routes // TOP_K),
        "routes": routes,
        "rows_per_expert": {
            "mean": round(routes / NUM_EXPERTS, 2),
            "p50": int(rows64.median().item()),
            "max": int(rows64.max().item()),
            "empty_experts": int((rows64 == 0).sum().item()),
        },
        "widths": {},
    }

    measured = parse_measured(args.measured)
    measured_blocks: dict[tuple[str, int], int] = {}
    for value in args.measured_blocks:
        key, _, count = value.partition("=")
        routing_key, block_key = key.split(":")
        measured_blocks[(routing_key, int(block_key))] = int(count)
    fits: dict[str, dict] = {}
    for routing, points in measured.items():
        annotated = [
            (
                block,
                width,
                micros,
                measured_blocks.get(
                    (routing, block), block_stats(rows, block, width)["total_blocks"]
                ),
            )
            for block, width, micros in points
        ]
        fit = calibrate(annotated)
        if fit is not None:
            fits[routing] = fit
    if fits:
        report["calibration"] = fits

    for width in widths:
        entries = []
        for block in blocks:
            stats = block_stats(rows, block, width)
            for routing, fit in fits.items():
                stats[f"predicted_us[{routing}]"] = round(
                    predict(fit, block, width, stats["total_blocks"]), 1
                )
            entries.append(stats)
        report["widths"][str(width)] = entries

    for width, entries in report["widths"].items():
        print(f"width {width}:")
        base = next((e for e in entries if e["block_m"] == 48), entries[0])
        for entry in entries:
            decode_ratio = entry["decoded_weights"] / base["decoded_weights"]
            mma_ratio = entry["padded_mma_flop"] / base["padded_mma_flop"]
            line = (
                f"  block {entry['block_m']:>3}: blocks={entry['total_blocks']:>6} "
                f"({entry['blocks_per_used_expert']:.3f}/expert) "
                f"padding={entry['padding_fraction']:+.4f} "
                f"decode={decode_ratio:.3f}x mma={mma_ratio:.3f}x "
                f"k_tiles={entry['k_tile_iters']:>8}"
            )
            for key, value in entry.items():
                if key.startswith("predicted_us"):
                    line += f" {key}={value}"
            print(line)
    for routing, fit in fits.items():
        row_cost = fit["beta_us_per_k_tile_per_row"] * 48
        share = row_cost / (fit["alpha_us_per_k_tile"] + row_cost)
        print(
            f"calibration[{routing}]: C={fit['C_us']:.1f}us "
            f"alpha={fit['alpha_us_per_k_tile'] * 1e3:.4f}ns/k-tile "
            f"beta={fit['beta_us_per_k_tile_per_row'] * 1e6:.4f}ps/k-tile/row "
            f"row-proportional share at block 48 = {share:.1%} "
            f"max_residual={fit['max_abs_residual_us']}us"
        )

    text = json.dumps(report, indent=1)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
