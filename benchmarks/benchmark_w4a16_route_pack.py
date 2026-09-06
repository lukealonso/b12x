#!/usr/bin/env python3
"""Benchmark the production W4A16 MoE route packer.

The default corpus covers the mapped decode geometries used by Kimi K3 and
GLM-5.2 hybrid checkpoints and the Kimi K3 TP9 prefill chunk (4,608 tokens,
top-16 over 896 experts, route blocks 48 and 64) under uniform routing, plus
one Zipf-weighted prefill chunk whose largest expert segments exceed the
stable packer's widest register sort and take its workspace-scan path.  Each
record reports ``segment_bands``, the histogram of live segment sizes over
those launch bands, so a report says which packing kernels it exercised.
Both eager launches and
CUDA graph replay are timed with caller-owned workspaces, matching the
serving contract.  The packer honours ``B12X_W4A16_STABLE_ROUTE_PACK`` (the
served setting is 1) and ``B12X_W4A16_STABLE_ROUTE_PACK_SCAN`` (1 selects the
sequential per-expert scan kernel instead of the atomic scatter + segment
sort); with the stable flag the packed layout is a pure function of the
routing, so ``packed_sha256`` of two runs must match.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmarks.common import bench_cuda_graph, capture_cuda_graph, require_sm120
from b12x.moe._shared.kernels.w4a16.host import (
    max_packed_route_slots,
    route_pack_numel_capacity,
)
from b12x.moe._shared.kernels.w4a16.kernel import (
    pack_topk_routes_by_expert,
)
from b12x.moe._shared.kernels.w4a16.route_pack import (
    _STABLE_SEGMENT_SORT_WIDTHS as STABLE_SEGMENT_SORT_WIDTHS,
)


@dataclass(frozen=True)
class RoutePackCase:
    name: str
    tokens: int
    topk: int
    num_experts: int
    local_experts: int
    block_size: int
    # "uniform" draws each token's experts with equal probability; "zipf"
    # weights expert ``i`` by ``(i + 1) ** -ZIPF_EXPONENT``, which spreads the
    # per-expert segment sizes over every register sort width and the
    # workspace-scan fallback of the stable packer.
    routing: str = "uniform"


ZIPF_EXPONENT = 1.2


CASES = {
    "k3": RoutePackCase(
        name="k3-tp12-hybrid-decode",
        tokens=1,
        topk=16,
        num_experts=896,
        local_experts=448,
        block_size=8,
    ),
    "glm52": RoutePackCase(
        name="glm52-tp4-hybrid-decode",
        tokens=4,
        topk=8,
        num_experts=256,
        local_experts=192,
        block_size=8,
    ),
    "k3-prefill-b48": RoutePackCase(
        name="k3-tp9-prefill-chunk-block48",
        tokens=4608,
        topk=16,
        num_experts=896,
        local_experts=896,
        block_size=48,
    ),
    "k3-prefill-b64": RoutePackCase(
        name="k3-tp9-prefill-chunk-block64",
        tokens=4608,
        topk=16,
        num_experts=896,
        local_experts=896,
        block_size=64,
    ),
    "k3-prefill-b48-zipf": RoutePackCase(
        name="k3-tp9-prefill-chunk-block48-zipf",
        tokens=4608,
        topk=16,
        num_experts=896,
        local_experts=896,
        block_size=48,
        routing="zipf",
    ),
}


def _git_revision() -> str:
    repo = pathlib.Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{revision}{'-dirty' if dirty else ''}"


def _make_expert_map(case: RoutePackCase, device: torch.device) -> torch.Tensor:
    expert_map = torch.full(
        (case.num_experts,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    global_ids = torch.arange(case.local_experts, device=device) * (
        case.num_experts // case.local_experts
    )
    expert_map[global_ids] = torch.arange(
        case.local_experts,
        dtype=torch.int32,
        device=device,
    )
    return expert_map


def _make_routes(
    case: RoutePackCase,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if case.routing == "zipf":
        weights = torch.arange(1, case.num_experts + 1, dtype=torch.float32).pow(
            -ZIPF_EXPONENT
        )
    elif case.routing == "uniform":
        weights = torch.ones(case.num_experts)
    else:
        raise ValueError(f"unknown routing {case.routing!r}")
    if case.tokens * case.topk <= case.num_experts and case.routing == "uniform":
        routes = torch.randperm(case.num_experts, generator=generator)[
            : case.tokens * case.topk
        ].reshape(case.tokens, case.topk)
    else:
        # Prefill: every token routes to topk distinct experts. Under uniform
        # weights the served router's marginal at 4,608 tokens is close to
        # uniform and every expert segment stays below a few hundred routes;
        # the Zipf weights concentrate traffic so that the largest segments
        # exceed the widest register sort and take the packer's workspace-scan
        # path, which uniform routing never reaches.
        routes = torch.multinomial(
            weights.unsqueeze(0).expand(case.tokens, case.num_experts),
            case.topk,
            replacement=False,
            generator=generator,
        )
    return routes.to(dtype=torch.int32, device=device)


def _stable_layout_requested(case: RoutePackCase) -> bool:
    flag = os.environ.get("B12X_W4A16_STABLE_ROUTE_PACK", "")
    return flag.strip().lower() not in {"", "0", "false", "no", "off"} and (
        case.tokens * case.topk >= 4096
    )


def _workspace(
    case: RoutePackCase,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    live_routes = case.tokens * case.topk
    route_capacity = route_pack_numel_capacity(live_routes, topk=case.topk)
    packed_capacity = max_packed_route_slots(
        route_capacity,
        case.block_size,
        case.num_experts,
    )
    block_capacity = (packed_capacity + case.block_size - 1) // case.block_size
    return {
        "packed_route_indices": torch.empty(
            packed_capacity,
            dtype=torch.int32,
            device=device,
        ),
        "block_expert_ids": torch.empty(
            block_capacity,
            dtype=torch.int32,
            device=device,
        ),
        "packed_route_count": torch.empty(1, dtype=torch.int32, device=device),
        "expert_offsets": torch.empty(
            case.num_experts + 1,
            dtype=torch.int32,
            device=device,
        ),
        "expert_counts": torch.empty(
            case.num_experts,
            dtype=torch.int32,
            device=device,
        ),
    }


def _segment_bands(counts: torch.Tensor) -> dict[str, int]:
    """Live expert segments per launch band of the stable packer.

    The stable path orders each expert's segment with one launch per register
    sort width and one workspace-scan launch above the widest width, so a
    routing whose segments all land in the narrowest band leaves the other
    kernels unmeasured. Reporting the histogram makes a record say which
    bands it covered.
    """
    live = counts[counts > 0]
    bands: dict[str, int] = {}
    lower = 0
    for width, _ in STABLE_SEGMENT_SORT_WIDTHS:
        bands[f"sort_{width}"] = int(((live > lower) & (live <= width)).sum())
        lower = width
    bands["scan"] = int((live > lower).sum())
    bands["max_segment"] = int(live.max()) if live.numel() else 0
    return bands


def _validate(
    case: RoutePackCase,
    routes: torch.Tensor,
    expert_map: torch.Tensor,
    workspace: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Check the packed layout and return the live route count per expert."""
    raw_ids = routes.cpu().reshape(-1).to(torch.int64)
    host_map = expert_map.cpu().to(torch.int64)
    mapped_ids = host_map[raw_ids]
    valid = mapped_ids >= 0
    counts = torch.bincount(
        mapped_ids[valid],
        minlength=case.num_experts,
    )
    padded = (
        (counts + case.block_size - 1) // case.block_size * case.block_size
    )
    expected_count = int(padded.sum().item())
    actual_count = int(workspace["packed_route_count"].cpu().item())
    if actual_count != expected_count:
        raise AssertionError(
            f"{case.name}: packed count {actual_count} != {expected_count}"
        )

    expected_block_experts = torch.repeat_interleave(
        torch.arange(case.num_experts),
        padded // case.block_size,
    ).to(torch.int32)
    actual_block_experts = workspace["block_expert_ids"][
        : expected_count // case.block_size
    ].cpu()
    if not torch.equal(actual_block_experts, expected_block_experts):
        raise AssertionError(f"{case.name}: block expert ids do not match")

    packed = workspace["packed_route_indices"][:expected_count].cpu().to(torch.int64)
    payload = packed[packed < raw_ids.numel()]
    expected_payload = torch.nonzero(valid, as_tuple=False).flatten()
    if not torch.equal(payload.sort().values, expected_payload.sort().values):
        raise AssertionError(f"{case.name}: packed route payload does not match")
    if payload.unique().numel() != payload.numel():
        raise AssertionError(f"{case.name}: packed route payload contains duplicates")

    for block, expert in enumerate(actual_block_experts.tolist()):
        block_routes = packed[
            block * case.block_size : (block + 1) * case.block_size
        ]
        block_payload = block_routes[block_routes < raw_ids.numel()]
        if block_payload.numel() and not torch.all(
            mapped_ids[block_payload] == expert
        ):
            raise AssertionError(
                f"{case.name}: block {block} contains another expert"
            )
    if _stable_layout_requested(case):
        # Stable packing: inside every expert segment the live routes ascend
        # and the padding slots hold the live route count.
        cursor = 0
        for expert, live in enumerate(counts.tolist()):
            padded_rows = int(padded[expert])
            segment = packed[cursor : cursor + padded_rows]
            live_routes = segment[:live]
            if not torch.all(live_routes[1:] > live_routes[:-1]):
                raise AssertionError(
                    f"{case.name}: expert {expert} segment is not ascending"
                )
            if not torch.all(segment[live:] == raw_ids.numel()):
                raise AssertionError(
                    f"{case.name}: expert {expert} padding is not the sentinel"
                )
            cursor += padded_rows
    return counts


def _packed_digest(case: RoutePackCase, workspace: dict[str, torch.Tensor]) -> str:
    count = int(workspace["packed_route_count"].cpu().item())
    packed = workspace["packed_route_indices"][:count].cpu().contiguous()
    blocks = workspace["block_expert_ids"][: count // case.block_size].cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(packed.numpy().tobytes())
    digest.update(blocks.numpy().tobytes())
    return digest.hexdigest()[:16]


def _eager_samples(
    run: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        run()
        end.record()
    torch.cuda.synchronize()
    return [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends, strict=True)
    ]


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[len(ordered) // 10],
        "p90_us": ordered[9 * len(ordered) // 10],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _run_case(
    case: RoutePackCase,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    routes = _make_routes(case, device, seed)
    expert_map = _make_expert_map(case, device)
    workspace = _workspace(case, device)

    def run() -> object:
        return pack_topk_routes_by_expert(
            routes,
            case.block_size,
            case.num_experts,
            expert_map=expert_map,
            **workspace,
        )

    run()
    torch.cuda.synchronize()
    _validate(case, routes, expert_map, workspace)
    eager = _eager_samples(run, warmup=warmup, iterations=iterations)
    graph = capture_cuda_graph(run, warmup=warmup)
    graph_samples = bench_cuda_graph(graph, replays=iterations)["replay_us"]
    counts = _validate(case, routes, expert_map, workspace)
    return {
        "case": case.__dict__,
        "stable_layout": _stable_layout_requested(case),
        "segment_bands": _segment_bands(counts),
        "packed_sha256": _packed_digest(case, workspace),
        "eager": _summary(eager),
        "graph": _summary(graph_samples),
        "raw_eager_us": eager,
        "raw_graph_us": graph_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        nargs="+",
        choices=(*CASES, "all"),
        default=["all"],
        help="one or more cases to run in a single report; 'all' runs every case",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()

    device = require_sm120()
    props = torch.cuda.get_device_properties(device)
    names = list(CASES) if "all" in args.case else list(dict.fromkeys(args.case))
    selected = [CASES[name] for name in names]
    report = {
        "commit": _git_revision(),
        "device": props.name,
        "device_uuid": str(getattr(props, "uuid", "")),
        "torch": torch.__version__,
        "cases": [
            _run_case(
                case,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
                seed=args.seed,
            )
            for case in selected
        ],
    }
    for result in report["cases"]:
        case = result["case"]
        eager = result["eager"]
        graph = result["graph"]
        bands = result["segment_bands"]
        print(
            f"{case['name']}: eager={eager['median_us']:.3f} us "
            f"(p10={eager['p10_us']:.3f}, p90={eager['p90_us']:.3f}) | "
            f"graph={graph['median_us']:.3f} us "
            f"(p10={graph['p10_us']:.3f}, p90={graph['p90_us']:.3f}) | "
            f"segments {bands}"
        )
        if not args.raw:
            result.pop("raw_eager_us")
            result.pop("raw_graph_us")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
