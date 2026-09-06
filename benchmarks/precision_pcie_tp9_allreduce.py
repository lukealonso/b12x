#!/usr/bin/env python3
"""Numerical comparison of the nine-rank PCIe all-reduce kernels.

The decode-size kernels accumulate the nine bf16 contributions in fp32 and
round once to bf16, but in different orders: the one-shot in ascending rank
order, the two-shot (pull/push) from the shard owner onwards, the island9
kernel as (ranks 0-3 + 8) + (ranks 4-7). The prefill-size DMA ring
(``dma-ring``) rounds to bf16 after every reduce-scatter hop (eight
roundings at nine ranks). The column reduce-scatter of the same ring
(``dma-rs-bf16``: eight roundings in column-block order; ``dma-rs-fp32``:
fp32 running sum on the wire, one rounding) returns one column block per
rank; the script gathers the blocks so every kernel is compared on the full
``[rows, width]`` sum. The script feeds heavy-tailed activations (Gaussian
bulk, sparse outliers up to 1e4, per-rank scale spread) through each kernel
and measures, against the correctly rounded fp64 sum:

* elements whose bf16 output differs from the correctly rounded sum,
* the largest error in bf16 ulps and the relative L2 error,
* elements on which the kernels differ from each other.

With ``--norm``, the script also measures the latent RMSNorm that consumes
the reduced tensor: the served arithmetic (fp32 variance over the ring's
bf16 latent, ``bf16(bf16(x * s) * w)``) against the column-block scheme
(variance from fp64 per-block partial sums combined across ranks with an
fp64 all-reduce, ``bf16(x * s * w)`` rounded once), both against the fp64
RMSNorm of the fp64 sum.

A selected kernel that cannot serve the requested ``(rows, width)`` - the
decode-size staging kernels reject prefill-size inputs - is dropped with its
reason under ``skipped`` instead of aborting the run, so one unsupported pair
in a ``--kernels`` list does not cost the measurements of the others. The
decision is taken collectively, so every rank drops the same kernels.

The result is a JSON record per kernel (rank 0 prints and writes it).

    B12X_RUN_PCIE_TP9_TEST=1 python benchmarks/precision_pcie_tp9_allreduce.py \
        --rows 4608 --width 3584 --samples 8 --norm \
        --kernels dma-ring,dma-rs-bf16,dma-rs-fp32 --output out.json
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
ALL_KERNELS = (
    "oneshot",
    "twoshot-push",
    "island9-push",
    "dma-ring",
    "dma-rs-bf16",
    "dma-rs-fp32",
)


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


def _record(results: dict, name: str, out: torch.Tensor, reference: torch.Tensor,
            reference64: torch.Tensor) -> None:
    record = results.setdefault(name, {"mismatch_elements": 0, "elements": 0,
                                       "max_ulp": 0.0, "rel_l2_sum": 0.0, "samples": 0})
    diff = out != reference
    record["mismatch_elements"] += int(diff.sum().item())
    record["elements"] += out.numel()
    record["max_ulp"] = max(record["max_ulp"], float(_ulps(out, reference).max().item()))
    err = (out.double() - reference64).norm() / reference64.norm().clamp_min(1e-30)
    record["rel_l2_sum"] += float(err.item())
    record["samples"] += 1


def _gather_column_blocks(block: torch.Tensor, width: int, group) -> torch.Tensor:
    """Assemble the full ``[rows, width]`` sum from every rank's column block."""
    blocks = [torch.empty_like(block) for _ in range(WORLD)]
    dist.all_gather(blocks, block, group=group)
    cols = block.shape[1]
    full = torch.empty((block.shape[0], width), dtype=block.dtype, device=block.device)
    for rank, part in enumerate(blocks):
        start = rank * cols
        valid = min(cols, width - start)
        if valid > 0:
            full[:, start : start + valid] = part[:, :valid]
    return full


def _served_rms_norm(latent: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The served in-place RMSNorm arithmetic on a bf16 latent: fp32 variance,
    ``bf16(bf16(x * s) * w)`` (two roundings per element)."""
    x = latent.float()
    variance = x.square().mean(dim=-1, keepdim=True)
    s = torch.rsqrt(variance + eps)
    return ((x * s).to(torch.bfloat16).float() * weight.float()).to(torch.bfloat16)


def _column_block_rms_norm(
    block: torch.Tensor, weight_block: torch.Tensor, width: int, eps: float, group
) -> torch.Tensor:
    """The column-block RMSNorm: fp64 per-block sum of squares combined with an
    fp64 all-reduce, ``bf16(x * s * w)`` rounded once (mirrors the vLLM
    ``KimiRoutedOutputTransform`` column-block path)."""
    sumsq = block.double().square().sum(dim=-1, keepdim=True)
    dist.all_reduce(sumsq, group=group)
    s = torch.rsqrt(sumsq / width + eps).float()
    return (block.float() * s * weight_block.float()).to(torch.bfloat16)


def _worker(rank: int, port: int, args: argparse.Namespace) -> None:
    from b12x.comm.pcie.pcie_dma import PCIeDmaAllReduce
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
    selected = [name for name in ALL_KERNELS if name in args.kernels]
    results: dict[str, dict] = {}
    outputs: dict[str, list[torch.Tensor]] = {}
    closers = []

    kernels = {}
    accepts = {}
    skipped: dict[str, str] = {}

    def build(name: str, construct):
        """Construct a runtime, or record why this size has no kernel."""
        if name not in selected:
            return None
        try:
            return construct()
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            skipped[name] = f"{type(exc).__name__} while building: {exc}"
            return None

    pool = build(
        "oneshot",
        lambda: PCIeOneshotAllReducePool.from_process_group(
            process_group=group, device=device, max_input_bytes=rows * width * 2,
            max_concurrent_channels=1,
        ),
    )
    if pool is not None:
        pool.prepare_channels(("eager",))
        kernels["oneshot"] = lambda inp, out: pool.all_reduce(inp, out=out, channel_id="eager")
        accepts["oneshot"] = lambda inp: (
            pool.for_stream(None, channel_id="eager").should_allreduce(inp),
            "input not accepted by the one-shot channel",
        )
    twoshot = build(
        "twoshot-push",
        lambda: PCIeTwoShotBF16.from_exchange_group(
            exchange_group=group, device=device, max_rows=49149, row_elems=8,
        ),
    )
    if twoshot is not None:
        twoshot.all_reduce_mode = "push"
        kernels["twoshot-push"] = lambda inp, out: twoshot.all_reduce(inp, out=out)
        accepts["twoshot-push"] = lambda inp: (
            twoshot.accepts(inp),
            "input not accepted by PCIeTwoShotBF16.all_reduce (a decode-size "
            f"kernel: staging holds {twoshot.max_rows} rows of "
            f"{twoshot.row_elems} elements)",
        )
        closers.append(twoshot.close)
    island9 = build(
        "island9-push",
        lambda: PCIeIsland9AllReduce.from_exchange_group(
            exchange_group=group, device=device, max_rows=49149, row_elems=8,
        ),
    )
    if island9 is not None:
        kernels["island9-push"] = lambda inp, out: island9.all_reduce(inp, out=out)
        accepts["island9-push"] = lambda inp: (
            island9.accepts(inp),
            "input not accepted by PCIeIsland9AllReduce.all_reduce (a "
            f"decode-size kernel: staging holds {island9.max_elements} elements)",
        )
        closers.append(island9.close)
    dma = None
    if any(name.startswith("dma-") for name in selected):
        dma = build(
            "dma",
            lambda: PCIeDmaAllReduce(
                exchange_group=group, device=device, max_bytes=rows * width * 2
            ),
        )
    if dma is not None:
        dma.min_bytes = 0
        closers.append(dma.close)
        if "dma-ring" in selected:
            kernels["dma-ring"] = lambda inp, out: dma.all_reduce(inp, out=out)
            accepts["dma-ring"] = lambda inp: (
                dma.should_allreduce(inp),
                "input not accepted by the PCIe DMA ring all-reduce",
            )
        for wire in ("bf16", "fp32"):
            name = f"dma-rs-{wire}"
            if name not in selected:
                continue
            kernels[name] = (
                lambda inp, out, wire=wire: out.copy_(
                    _gather_column_blocks(
                        dma.reduce_scatter_columns(inp, wire=wire), width, group
                    )
                )
            )
            accepts[name] = lambda inp: (
                dma.should_reduce_scatter_columns(inp),
                "input not accepted by the PCIe DMA ring column reduce-scatter",
            )
    elif any(name.startswith("dma-") for name in selected):
        reason = skipped.pop("dma", "the PCIe DMA ring could not be built")
        for name in selected:
            if name.startswith("dma-"):
                skipped[name] = reason

    # A kernel that cannot serve this (rows, width) is dropped with its reason
    # instead of aborting the run, so one unsupported pair in a --kernels list
    # does not cost the measurements of the others. The verdict is collective:
    # every rank must drop the same kernels or the survivors would deadlock.
    probe = torch.zeros(rows, width, dtype=torch.bfloat16, device=device)
    for name in list(kernels):
        check = accepts.get(name)
        if check is None:
            continue
        try:
            ok, reason = check(probe)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            ok, reason = False, f"{type(exc).__name__}: {exc}"
        verdict = torch.tensor([0 if ok else 1], device=device)
        dist.all_reduce(verdict, group=group)
        if int(verdict.item()):
            skipped[name] = reason
            del kernels[name]
    del probe
    if not kernels:
        raise SystemExit(
            f"no selected kernel serves rows={rows} width={width}: {skipped}"
        )

    norm_weight = None
    if args.norm:
        generator = torch.Generator(device="cpu").manual_seed(4242)
        norm_weight = (1.0 + 0.1 * torch.randn(width, generator=generator)).to(torch.bfloat16).to(device)
        cols = (width + WORLD - 1) // WORLD
        weight_block = torch.zeros(cols, dtype=torch.bfloat16, device=device)
        start = rank * cols
        valid = min(cols, width - start)
        if valid > 0:
            weight_block[:valid] = norm_weight[start : start + valid]

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
            _record(results, name, out, reference, reference64)
        if args.norm and norm_weight is not None:
            eps = args.eps
            s64 = torch.rsqrt(reference64.square().mean(dim=-1, keepdim=True) + eps)
            norm64 = reference64 * s64 * norm_weight.double()
            norm_ref = norm64.to(torch.bfloat16)
            if "dma-ring" in kernels:
                served = _served_rms_norm(outputs["dma-ring"][-1], norm_weight, eps)
                _record(results, "norm/dma-ring", served, norm_ref, norm64)
            for name in ("dma-rs-fp32", "dma-rs-bf16"):
                if name in kernels and dma is not None:
                    wire = name.rsplit("-", 1)[1]
                    block = dma.reduce_scatter_columns(inp, wire=wire)
                    normed_block = _column_block_rms_norm(block, weight_block, width, eps, group)
                    normed = _gather_column_blocks(normed_block, width, group)
                    _record(results, f"norm/{name}", normed, norm_ref, norm64)
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
    for close in closers:
        close()
    if rank == 0:
        summary = {"rows": rows, "width": width, "samples": args.samples,
                   "kernels": names, "skipped": skipped, "per_kernel": results,
                   "kernels_differ_on": cross,
                   "cross_rank_inconsistent_elements": consistency}
        print(json.dumps(summary, indent=2), flush=True)
        if args.output:
            with open(args.output, "w") as stream:
                json.dump(summary, stream, indent=2)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--width", type=int, default=7168)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--kernels", default=",".join(ALL_KERNELS),
                        help="comma-separated subset of " + ", ".join(ALL_KERNELS))
    parser.add_argument("--norm", action="store_true",
                        help="also measure the latent RMSNorm consuming the reduced tensor")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    args.kernels = {name.strip() for name in args.kernels.split(",") if name.strip()}
    unknown = args.kernels - set(ALL_KERNELS)
    if unknown:
        raise SystemExit(f"unknown kernels: {sorted(unknown)}")
    if os.getenv("B12X_RUN_PCIE_TP9_TEST") != "1":
        raise SystemExit("requires nine idle GPUs and B12X_RUN_PCIE_TP9_TEST=1")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    mp.spawn(_worker, args=(port, args), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
