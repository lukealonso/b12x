#!/usr/bin/env python3
"""Benchmark the production W4A16 MoE route packer.

The default corpus covers the mapped decode geometries used by Kimi K3 and
GLM-5.2 hybrid checkpoints.  Both eager launches and CUDA graph replay are
timed with caller-owned workspaces, matching the serving contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
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


@dataclass(frozen=True)
class RoutePackCase:
    name: str
    tokens: int
    topk: int
    num_experts: int
    local_experts: int
    block_size: int
    use_expert_map: bool = True


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
    "qwen38-flash-next-m4": RoutePackCase(
        name="qwen38-flash-next-tp1-m4",
        tokens=4,
        topk=10,
        num_experts=512,
        local_experts=512,
        block_size=16,
        use_expert_map=False,
    ),
    "qwen38-flash-next-m5": RoutePackCase(
        name="qwen38-flash-next-tp1-m5",
        tokens=5,
        topk=10,
        num_experts=512,
        local_experts=512,
        block_size=16,
        use_expert_map=False,
    ),
    "qwen38-flash-next-m6": RoutePackCase(
        name="qwen38-flash-next-tp1-m6",
        tokens=6,
        topk=10,
        num_experts=512,
        local_experts=512,
        block_size=16,
        use_expert_map=False,
    ),
    "qwen38-flash-next-m7": RoutePackCase(
        name="qwen38-flash-next-tp1-m7",
        tokens=7,
        topk=10,
        num_experts=512,
        local_experts=512,
        block_size=16,
        use_expert_map=False,
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
    routes = torch.randperm(case.num_experts, generator=generator)[
        : case.tokens * case.topk
    ]
    return routes.reshape(case.tokens, case.topk).to(
        dtype=torch.int32,
        device=device,
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


def _validate(
    case: RoutePackCase,
    routes: torch.Tensor,
    expert_map: torch.Tensor | None,
    workspace: dict[str, torch.Tensor],
) -> None:
    raw_ids = routes.cpu().reshape(-1).to(torch.int64)
    mapped_ids = (
        raw_ids
        if expert_map is None
        else expert_map.cpu().to(torch.int64)[raw_ids]
    )
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
    expert_map = _make_expert_map(case, device) if case.use_expert_map else None
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
    _validate(case, routes, expert_map, workspace)
    return {
        "case": case.__dict__,
        "eager": _summary(eager),
        "graph": _summary(graph_samples),
        "raw_eager_us": eager,
        "raw_graph_us": graph_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=(*CASES, "all"),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()

    device = require_sm120()
    props = torch.cuda.get_device_properties(device)
    selected = CASES.values() if args.case == "all" else (CASES[args.case],)
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
        print(
            f"{case['name']}: eager={eager['median_us']:.3f} us "
            f"(p10={eager['p10_us']:.3f}, p90={eager['p90_us']:.3f}) | "
            f"graph={graph['median_us']:.3f} us "
            f"(p10={graph['p10_us']:.3f}, p90={graph['p90_us']:.3f})"
        )
        if not args.raw:
            result.pop("raw_eager_us")
            result.pop("raw_graph_us")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
