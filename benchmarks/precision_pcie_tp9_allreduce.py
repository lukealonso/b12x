#!/usr/bin/env python3
"""Numerical comparison of the nine-rank PCIe all-reduce kernels.

Every kernel accumulates the nine bf16 contributions in fp32 and rounds once
to bf16, but in different orders: the one-shot in ascending rank order, the
two-shot (pull/push) from the shard owner onwards, the island9 kernel as
(ranks 0-3 + 8) + (ranks 4-7). The script feeds heavy-tailed activations
(Gaussian bulk, sparse outliers up to 1e4, per-rank scale spread) through
each kernel and measures, against the correctly rounded fp64 sum:

* elements whose bf16 output differs from the correctly rounded sum,
* the largest error in bf16 ulps and the relative L2 error,
* elements on which the kernels differ from each other.

The result is a JSON record per kernel (rank 0 prints and writes it).

    B12X_RUN_PCIE_TP9_TEST=1 python benchmarks/precision_pcie_tp9_allreduce.py --output out.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 9


def _activations(rows: int, width: int, rank: int, seed: int, device: torch.device) -> torch.Tensor:
    """Heavy-tailed bf16 activations with per-rank scale and sparse outliers."""
    generator = torch.Generator(device="cpu").manual_seed(seed * 1000 + rank)
    bulk = torch.randn(rows, width, generator=generator, dtype=torch.float32)
    scale = 2.0 ** (rank - 4)  # per-rank magnitude spread of 2^-4 .. 2^4
    bulk = bulk * scale
    outlier_mask = torch.rand(rows, width, generator=generator) < 0.01
    outliers = torch.randn(rows, width, generator=generator, dtype=torch.float32) * 1.0e4
    bulk = torch.where(outlier_mask, outliers, bulk)
    return bulk.to(torch.bfloat16).to(device)


def _ulps(actual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Distance in bf16 ulps (of the reference) between two bf16 tensors."""
    exponent = torch.floor(torch.log2(reference.abs().float().clamp_min(2.0 ** -126)))
    ulp = torch.pow(2.0, exponent - 7)
    return (actual.float() - reference.float()).abs() / ulp


def _worker(rank: int, port: int, args: argparse.Namespace) -> None:
    from b12x.comm.pcie.pcie_island9 import PCIeIsland9AllReduce
    from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool
    from b12x.comm.pcie.pcie_twoshot_bf16 import PCIeTwoShotBF16

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=240),
        device_id=device,
    )
    group = dist.group.WORLD
    rows, width = args.rows, args.width
    results: dict[str, dict] = {}
    outputs: dict[str, list[torch.Tensor]] = {}

    kernels = {}
    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=group, device=device, max_input_bytes=rows * width * 2,
        max_concurrent_channels=1,
    )
    pool.prepare_channels(("eager",))
    kernels["oneshot"] = lambda inp, out: pool.all_reduce(inp, out=out, channel_id="eager")
    twoshot = PCIeTwoShotBF16.from_exchange_group(
        exchange_group=group, device=device, max_rows=49149, row_elems=8,
    )
    twoshot.all_reduce_mode = "push"
    kernels["twoshot-push"] = lambda inp, out: twoshot.all_reduce(inp, out=out)
    island9 = PCIeIsland9AllReduce.from_exchange_group(
        exchange_group=group, device=device, max_rows=49149, row_elems=8,
    )
    kernels["island9-push"] = lambda inp, out: island9.all_reduce(inp, out=out)

    for seed in range(args.samples):
        inp = _activations(rows, width, rank, seed, device)
        # Correctly rounded reference: fp64 sum of the nine bf16 inputs.
        gathered = [torch.empty_like(inp) for _ in range(WORLD)]
        dist.all_gather(gathered, inp, group=group)
        reference64 = sum(t.double() for t in gathered)
        reference = reference64.to(torch.bfloat16)
        for name, call in kernels.items():
            out = torch.empty_like(inp)
            call(inp, out)
            torch.cuda.synchronize(device)
            outputs.setdefault(name, []).append(out.clone())
            record = results.setdefault(name, {"mismatch_elements": 0, "elements": 0,
                                               "max_ulp": 0.0, "rel_l2_sum": 0.0, "samples": 0})
            diff = out != reference
            record["mismatch_elements"] += int(diff.sum().item())
            record["elements"] += out.numel()
            record["max_ulp"] = max(record["max_ulp"], float(_ulps(out, reference).max().item()))
            err = (out.double() - reference64).norm() / reference64.norm().clamp_min(1e-30)
            record["rel_l2_sum"] += float(err.item())
            record["samples"] += 1
    for name, record in results.items():
        record["rel_l2_mean"] = record["rel_l2_sum"] / max(record["samples"], 1)
        del record["rel_l2_sum"]
    names = list(kernels)
    cross = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            differing = sum(int((x != y).sum().item()) for x, y in zip(outputs[a], outputs[b]))
            cross[f"{a} vs {b}"] = differing
    # Every rank must hold the same output for a given kernel and sample.
    consistency = {}
    for name in names:
        mismatch = 0
        for out in outputs[name]:
            peers = [torch.empty_like(out) for _ in range(WORLD)]
            dist.all_gather(peers, out, group=group)
            mismatch += sum(int((p != peers[0]).sum().item()) for p in peers[1:])
        consistency[name] = mismatch
    twoshot.close()
    island9.close()
    if rank == 0:
        summary = {"rows": rows, "width": width, "samples": args.samples,
                   "per_kernel": results, "kernels_differ_on": cross,
                   "cross_rank_inconsistent_elements": consistency}
        print(json.dumps(summary, indent=2), flush=True)
        if args.output:
            with open(args.output, "w") as stream:
                json.dump(summary, stream, indent=2)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--width", type=int, default=7168)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if os.getenv("B12X_RUN_PCIE_TP9_TEST") != "1":
        raise SystemExit("requires nine idle GPUs and B12X_RUN_PCIE_TP9_TEST=1")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    mp.spawn(_worker, args=(port, args), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
