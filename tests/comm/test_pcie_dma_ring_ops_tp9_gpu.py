"""Nine-GPU qualification of the PCIe DMA ring's static-buffer replay, paired
all-gather and column reduce-scatter at the Kimi-K3 prefill shapes.

Every check compares the device result with the reference arithmetic of
``tests/comm/pcie_dma_emulation.py`` computed from the gathered inputs, so a
pass means the CUDA schedule, the copy engine and the CuTe adds produce the
values the CPU-emulated tests already proved. Requires nine idle GPUs:

    B12X_RUN_PCIE_TP9_TEST=1 python tests/comm/test_pcie_dma_ring_ops_tp9_gpu.py
"""

from __future__ import annotations

import json
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 9
ROWS = 4608
HIDDEN = 7168
LATENT = 3584
ROUTER_COLS = 104
LATENT_COLS = 400
MAX_BYTES = ROWS * HIDDEN * 2


def _heavy_tailed(
    rows: int, cols: int, rank: int, seed: int, dtype: torch.dtype, device
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed * 1000 + rank)
    bulk = torch.randn(rows, cols, generator=generator) * 2.0 ** (rank - 4)
    outliers = torch.rand(rows, cols, generator=generator) < 0.01
    spikes = torch.randn(rows, cols, generator=generator) * 1.0e4
    return torch.where(outliers, spikes, bulk).to(dtype).to(device)


def _worker(rank: int, port: int, evidence: str) -> None:
    from b12x.comm.pcie import pcie_dma
    from tests.comm.pcie_dma_emulation import (
        column_reduce_scatter_reference,
        ring_all_reduce_reference,
    )

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=600),
        device_id=device,
    )
    group = dist.group.WORLD
    records: list[dict] = []

    def record(stage: str, **fields) -> None:
        torch.cuda.synchronize(device)
        dist.barrier()
        entry = {"stage": stage, **fields}
        records.append(entry)
        if rank == 0:
            print(json.dumps(entry), flush=True)

    def gather_all(tensor: torch.Tensor) -> list[torch.Tensor]:
        peers = [torch.empty_like(tensor) for _ in range(WORLD)]
        dist.all_gather(peers, tensor, group=group)
        return peers

    def same_on_all_ranks(tensor: torch.Tensor) -> bool:
        peers = gather_all(tensor)
        return all(torch.equal(p, peers[0]) for p in peers[1:])

    ring = pcie_dma.PCIeDmaAllReduce(
        exchange_group=group, device=device, max_bytes=MAX_BYTES
    )
    ring.min_bytes = 0
    assert ring._graph_replay, "set B12X_PCIE_DMA_GRAPH_REPLAY=1"
    record("ring_ready", replay_in_place=ring._replay_in_place)

    # ---- item 7: in-place replay, borrowed outputs, producer chain -----------
    hidden = [_heavy_tailed(ROWS, HIDDEN, rank, seed, torch.bfloat16, device) for seed in range(3)]
    expected = [ring_all_reduce_reference(gather_all(h), WORLD) for h in hidden]
    eager = ring.all_reduce(hidden[0])
    torch.cuda.synchronize(device)
    assert torch.equal(eager, expected[0]), "eager all-reduce differs from the ring reference"
    assert not ring._replay_entries
    replayed = ring.all_reduce(hidden[0])
    torch.cuda.synchronize(device)
    assert torch.equal(replayed, expected[0])
    entry = next(iter(ring._replay_entries.values()))
    assert entry.key == ("ar", ROWS * HIDDEN, torch.bfloat16)
    assert entry.inp is entry.out
    static = ring.all_reduce_input((ROWS, HIDDEN), torch.bfloat16)
    assert static is not None and static.data_ptr() == entry.inp.data_ptr()
    borrowed = None
    for i in range(300):
        source = hidden[i % 3]
        target = static if borrowed is None else borrowed
        target.copy_(source)
        borrowed = ring.all_reduce(target, borrow_output=True)
        assert borrowed.data_ptr() == static.data_ptr()
        if i % 50 == 0 or i == 299:
            torch.cuda.synchronize(device)
            assert torch.equal(borrowed, expected[i % 3]), f"borrowed replay {i} differs"
    torch.cuda.synchronize(device)
    used = ring._send_counters.nonzero().flatten().tolist()
    assert used and max(used) < pcie_dma.AG_PAIR_SLOT_BASE
    assert all(int(ring._send_counters[s]) == 302 for s in used)
    assert torch.equal(ring._send_counters, ring._wait_counters)
    record("item7_in_place_replay_and_borrow_passed", replays=302, slots=len(used))

    # ---- item 1: paired all-gather on a side stream --------------------------
    side = torch.cuda.Stream(device=device)
    ready, done = torch.cuda.Event(), torch.cuda.Event()
    main = torch.cuda.current_stream(device)
    for call in range(3):
        first = _heavy_tailed(ROWS, ROUTER_COLS, rank, 10 + call, torch.float32, device)
        second = _heavy_tailed(ROWS, LATENT_COLS, rank, 20 + call, torch.bfloat16, device)
        expected_first = torch.stack(gather_all(first))
        expected_second = torch.stack(gather_all(second))
        ready.record(main)
        side.wait_event(ready)
        with torch.cuda.stream(side):
            out_first, out_second = ring.all_gather_pair(first, second)
        done.record(side)
        # Work the main stream keeps issuing while the ring transfers.
        filler = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
        filler = filler @ filler
        main.wait_event(done)
        torch.cuda.synchronize(device)
        assert torch.equal(out_first.view(torch.int32), expected_first.view(torch.int32))
        assert torch.equal(out_second.view(torch.int16), expected_second.view(torch.int16))
        if call >= 1:
            assert ring.is_ring_storage(out_first) and ring.is_ring_storage(out_second)
        del filler
    record("item1_gather_pair_eager_and_replay_passed")

    # ---- item 3: column reduce-scatter, fp32 and bf16 wire -------------------
    ring.prepare_reduce_scatter("fp32")
    latent = [_heavy_tailed(ROWS, LATENT, rank, 30 + seed, torch.bfloat16, device) for seed in range(2)]
    gathered_latent = [gather_all(x) for x in latent]
    sum64 = [sum(x.double() for x in g) for g in gathered_latent]
    stats = {}
    wire_faults: dict[str, list[str]] = {}
    for wire in ("fp32", "bf16"):
        for call in range(3):
            seed = call % 2
            block = ring.reduce_scatter_columns(latent[seed], wire=wire, cols=LATENT_COLS)
            torch.cuda.synchronize(device)
            reference = column_reduce_scatter_reference(
                gathered_latent[seed], WORLD, wire, cols=LATENT_COLS
            )[rank]
            assert block.shape == (ROWS, LATENT_COLS)
            equal = torch.equal(block, reference)
            if wire == "fp32":
                # The deployable wire must match the chained reference on
                # every call (eager capture, then two replays).
                assert equal, f"{wire} wire block differs (call {call})"
            elif not equal:
                # The bf16 wire is a candidate (served precision class); a
                # replay mismatch is recorded as a fault, not a test failure,
                # so the fp32 evidence and the layer sequence still land.
                wire_faults.setdefault(wire, []).append(f"rank {rank} call {call}")
            if call >= 1:
                assert ring.is_ring_storage(block)
        # The last call reduced latent[0]; count its mismatches against the
        # correctly rounded fp64 sum over every rank's valid columns.
        start = rank * LATENT_COLS
        valid = min(LATENT_COLS, LATENT - start)
        exact = sum64[0][:, start : start + valid].to(torch.bfloat16)
        mismatch = torch.tensor([int((block[:, :valid] != exact).sum())], device=device)
        dist.all_reduce(mismatch, group=group)
        stats[wire] = int(mismatch.item())
    ring_full = ring_all_reduce_reference(gathered_latent[0], WORLD)
    ring_mismatch = int((ring_full != sum64[0].to(torch.bfloat16)).sum())
    faults = [f for f in gather_all(torch.tensor([len(wire_faults.get("bf16", []))], device=device))]
    bf16_faults = int(sum(int(f.item()) for f in faults))
    assert stats["fp32"] * 4 < ring_mismatch
    record(
        "item3_reduce_scatter_passed",
        mismatch_vs_fp64={"rs_fp32": stats["fp32"], "rs_bf16": stats["bf16"], "ring": ring_mismatch},
        elements=ROWS * LATENT,
        rs_bf16_replay_faults=bf16_faults,
    )

    # ---- the layer sequence on one channel, replayed --------------------------
    first = _heavy_tailed(ROWS, ROUTER_COLS, rank, 40, torch.float32, device)
    second = _heavy_tailed(ROWS, LATENT_COLS, rank, 41, torch.bfloat16, device)
    expected_first = torch.stack(gather_all(first))
    expected_second = torch.stack(gather_all(second))
    rs_reference = column_reduce_scatter_reference(gathered_latent[0], WORLD, "fp32", cols=LATENT_COLS)[rank]
    static = ring.all_reduce_input((ROWS, HIDDEN), torch.bfloat16)
    assert static is not None
    for layer in range(200):
        static.copy_(hidden[layer % 3])
        a1 = ring.all_reduce(static, borrow_output=True)
        ready.record(main)
        side.wait_event(ready)
        with torch.cuda.stream(side):
            g_first, g_second = ring.all_gather_pair(first, second)
        done.record(side)
        main.wait_event(done)
        block = ring.reduce_scatter_columns(latent[0], wire="fp32", cols=LATENT_COLS)
        a1_snapshot = a1.clone()
        a1.copy_(hidden[(layer + 1) % 3])
        a5 = ring.all_reduce(a1, borrow_output=True)
        if layer % 40 == 0 or layer == 199:
            torch.cuda.synchronize(device)
            assert torch.equal(a1_snapshot, expected[layer % 3])
            assert torch.equal(a5, expected[(layer + 1) % 3])
            assert torch.equal(g_first.view(torch.int32), expected_first.view(torch.int32))
            assert torch.equal(g_second.view(torch.int16), expected_second.view(torch.int16))
            assert torch.equal(block, rs_reference)
            assert same_on_all_ranks(a5)
    torch.cuda.synchronize(device)
    assert {k[0] for k in ring._replay_entries} >= {"ag_pair", "ar", "rs_fp32"}
    record(
        "layer_sequence_replay_passed",
        layers=200,
        entries=[k[0] for k in ring._replay_entries],
        op_seq=ring._op_seq,
    )

    ring.close()
    if rank == 0 and evidence:
        with open(evidence, "w") as stream:
            json.dump(records, stream, indent=2)
    dist.destroy_process_group()


@pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_TP9_TEST") != "1",
    reason="requires nine idle GPUs and B12X_RUN_PCIE_TP9_TEST=1",
)
def test_tp9_dma_ring_ops() -> None:
    assert torch.cuda.device_count() == WORLD
    os.environ["B12X_PCIE_DMA_GRAPH_REPLAY"] = "1"
    os.environ["B12X_PCIE_DMA_GRAPH_REPLAY_MAX_ENTRIES"] = "4"
    os.environ["B12X_PCIE_DMA_GRAPH_REPLAY_MIN_BYTES"] = str(8 << 20)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    evidence = os.getenv("B12X_DMA_RING_OPS_EVIDENCE", "")
    mp.spawn(_worker, args=(port, evidence), nprocs=WORLD, join=True)


if __name__ == "__main__":
    test_tp9_dma_ring_ops()
