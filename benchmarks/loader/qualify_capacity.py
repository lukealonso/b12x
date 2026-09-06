"""Hold a checkpoint's tensor-size distribution in shared storage concurrently.

This destructive-to-available-memory benchmark needs an otherwise unloaded GPU.
It reads a sparse synthetic file into every allocation, touching every byte,
then checks both ends of every tensor from a captured GPU graph. No allocation
failure is retried with another storage kind. File-backed lazy mappings are
deliberately excluded: they would not prove resident allocation capacity.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

import torch
import triton
import triton.language as tl

from b12x.loader import capabilities, read_tensor, storage_stats
from benchmarks.loader._utils import source_identity


@triton.jit
def _read_ends(Pointers, Sizes, Output, Count, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    pointer = tl.load(Pointers + row, row < Count, 0).to(tl.pointer_type(tl.uint8))
    size = tl.load(Sizes + row, row < Count, 0)
    first = tl.load(pointer, (row < Count) & (size > 0), 0)
    last = tl.load(pointer + tl.maximum(size - 1, 0), (row < Count) & (size > 0), 0)
    tl.store(Output + row * 2, first, row < Count)
    tl.store(Output + row * 2 + 1, last, row < Count)


def memory():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in (
            "MemAvailable",
            "MemFree",
            "Unevictable",
            "Mlocked",
            "SwapFree",
            "Cached",
        ):
            fields[key] = int(value.split()[0]) * 1024
    fields["process_max_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    fields["process_vmas"] = len(Path("/proc/self/maps").read_text().splitlines())
    return fields


def verify(tensors, offsets=None):
    pointers = torch.tensor(
        [tensor.data_ptr() for tensor in tensors], device="cuda", dtype=torch.int64
    )
    sizes = torch.tensor(
        [tensor.nbytes for tensor in tensors], device="cuda", dtype=torch.int64
    )
    output = torch.empty(len(tensors) * 2, device="cuda", dtype=torch.uint8)

    def run():
        _read_ends[(triton.cdiv(len(tensors), 256),)](
            pointers, sizes, output, len(tensors), BLOCK=256
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        output.fill_(0xFF)
        graph.replay()
    expected = torch.zeros((len(tensors), 2), dtype=torch.uint8)
    lengths = sizes.cpu()
    starts = (
        torch.zeros(len(tensors), dtype=torch.int64)
        if offsets is None
        else torch.tensor(offsets)
    )
    expected[(lengths > 0) & (starts == 0), 0] = 0xA5
    expected[(lengths == 1) & (starts == 0), 1] = 0xA5
    torch.testing.assert_close(output.cpu().reshape(-1, 2), expected, rtol=0, atol=0)
    graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON list of tensor entries with a byte count in 'bytes'",
    )
    parser.add_argument(
        "--allocation",
        choices=["pinned", "system", "registered", "managed"],
        default="pinned",
    )
    parser.add_argument("--target-gib", type=float, default=100)
    parser.add_argument(
        "--arena-mib",
        type=int,
        default=0,
        help="Pack tensor views into aligned backing allocations; 0 allocates each tensor separately",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())
    sizes = [int(entry["bytes"]) for entry in entries]
    if not sizes or min(sizes) < 0 or max(sizes) <= 0:
        raise ValueError("manifest must contain nonnegative, nonempty tensor ranges")
    target = int(args.target_gib * 2**30)
    if target < sum(sizes):
        raise ValueError("target must hold the entire manifest")
    while sum(sizes) < target:
        sizes.append(min(max(sizes), target - sum(sizes)))
    if args.arena_mib < 0:
        raise ValueError("arena size must be nonnegative")
    blocks = []
    capacity = args.arena_mib * 2**20
    current = []
    used = 0
    for size in sizes:
        offset = (used + 255) // 256 * 256
        if current and (not capacity or offset + size > capacity):
            blocks.append((used, current))
            current, used, offset = [], 0, 0
        current.append((offset, size))
        used = offset + size
    if current:
        blocks.append((used, current))
    torch.cuda.init()
    report = {
        **source_identity(),
        "command": sys.argv,
        "allocation": args.allocation,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "tensor_count": len(entries),
        "allocation_count": len(blocks),
        "logical_tensor_count": len(sizes),
        "arena_mib": args.arena_mib,
        "alignment_bytes": sum(size for size, _ in blocks) - target,
        "checkpoint_bytes": sum(int(entry["bytes"]) for entry in entries),
        "target_bytes": target,
        "capabilities": capabilities(),
        "memlock_limit": resource.getrlimit(resource.RLIMIT_MEMLOCK),
        "max_map_count": int(Path("/proc/sys/vm/max_map_count").read_text()),
        "before": memory(),
        "progress": [],
        "success": False,
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    }
    held = []
    offsets = []
    start = time.monotonic()
    next_report = start

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        with tempfile.TemporaryDirectory(prefix="b12x-capacity-") as directory:
            path = Path(directory) / "synthetic-weights"
            with path.open("wb") as stream:
                stream.write(b"\xa5")
                stream.truncate(max(size for size, _ in blocks))
            sample = read_tensor(
                path, shape=(32,), dtype=torch.uint8, allocation=args.allocation
            )
            verify([sample])
            del sample
            for index, (size, views) in enumerate(blocks):
                backing = read_tensor(
                    path,
                    shape=(size,),
                    dtype=torch.uint8,
                    allocation=args.allocation,
                )
                for offset, length in views:
                    held.append(backing.narrow(0, offset, length))
                    offsets.append(offset)
                del backing
                now = time.monotonic()
                if now >= next_report or index == len(blocks) - 1:
                    point = {
                        "seconds": now - start,
                        "allocated": index + 1,
                        **storage_stats(),
                        **memory(),
                    }
                    report["progress"].append(point)
                    save()
                    print(json.dumps(point), flush=True)
                    next_report = now + 10
                    if point["MemAvailable"] < 2 * 2**30:
                        raise MemoryError(
                            "stopped with less than 2 GiB system headroom; capacity qualification failed"
                        )
            report["allocation_and_read_seconds"] = time.monotonic() - start
            report["resident"] = memory()
            verify(held, offsets)
            report["gpu_endpoints_and_graphs_correct"] = True
            report["success"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["held_storage"] = storage_stats()
        release = time.monotonic()
        held.clear()
        gc.collect()
        torch.cuda.synchronize()
        report["release_seconds"] = time.monotonic() - release
        report["after_storage"] = storage_stats()
        report["after"] = memory()
        save()
    print(
        json.dumps({key: value for key, value in report.items() if key != "progress"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
