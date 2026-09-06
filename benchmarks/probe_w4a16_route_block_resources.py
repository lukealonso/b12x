"""Offline register/shared-memory budget of the W4A16 fused MoE route blocks.

Compiles the fused W4A16 MoE megakernel for the served Kimi-K3 QSRT prefill
geometry at several route block sizes on a host with no CUDA device, then reads
the per-kernel resource usage (registers, spill traffic, shared memory) out of
the emitted cubin. The CuTe DSL lowers and runs ptxas without a device, so the
whole probe is a CPU proof: it answers "does this route block fit an SM120
streaming multiprocessor" before any GPU time is spent.

Two independent limits decide that:

* the per-thread architectural cap of 255 registers, and
* the register file of one SM (``_DEVICE_MAX_REG_BYTES`` = 255 KiB = 65,280
  registers), which a single 256-thread CTA at 255 registers already fills.

The fused kernel keeps the whole CTA output tile in fp32 registers across the
K loop: ``cta_m_blocks`` sets of 32 accumulators per thread (16 rows x 128
columns x 4 warp K-slices / 256 threads). The probe prints that accounting next
to the measured numbers so the two can be compared directly.

Usage (production image, no GPU):

    docker run --rm --network none -e CUDA_VISIBLE_DEVICES= \
      -v <worktree>:/opt/infernal-invocation/b12x:ro \
      -w /opt/infernal-invocation/b12x --entrypoint /opt/venv/bin/python \
      <image> benchmarks/probe_w4a16_route_block_resources.py \
      --blocks 16,32,48,64 --output /tmp/route-block-resources.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# The served Kimi-K3 QSRT prefill launch: 2-bpw SQG-XOR-Cheb-T12 experts, E4M3
# K/32 group scales, FC1 K=3584 -> N=2*width, FC2 K=width -> N=3584, one pinned
# 128x128 CTA tile with 256 threads on 188 SMs.
K3_HIDDEN_SIZE = 3584
K3_NUM_EXPERTS = 896
K3_TOP_K = 16
K3_SMS = 188
K3_TILE = (128, 128, 128, 128)
K3_CAPACITY = 4608

# Per-thread fp32 accumulators of one 16-row m-block at the pinned tile:
# 16 rows x 128 columns x 4 warp K-slices / 256 threads.
ACCUMULATORS_PER_M_BLOCK = 32
MAX_REGS_PER_THREAD = 255


def _fused_kwargs(*, block: int, width: int) -> dict[str, Any]:
    from b12x.moe._shared.kernels.w4a16.kernel import _DEFAULT_MAX_SHARED_MEM

    return dict(
        max_shared_mem=_DEFAULT_MAX_SHARED_MEM,
        size_m=K3_CAPACITY,
        hidden_size=K3_HIDDEN_SIZE,
        intermediate_size=width,
        num_experts=K3_NUM_EXPERTS,
        top_k=K3_TOP_K,
        activation="situ",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=block,
        max_m_blocks=2000,
        element_dtype="fp16",
        sms=K3_SMS,
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        w13_layout="trellis3_t256_proj",
        trellis_bits=2,
        force_tile_config=K3_TILE,
        intermediate_rotation=True,
        full_rotation=True,
        coupled_hadamard=True,
        rotation_input_dtype="bf16",
    )


def _run(command: list[str]) -> tuple[int, str]:
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    return proc.returncode, (proc.stdout + proc.stderr)


def _cuobjdump(path: Path, flag: str) -> str:
    binary = os.environ.get("CUOBJDUMP", "/usr/local/cuda/bin/cuobjdump")
    code, text = _run([binary, flag, str(path)])
    if code != 0:
        return f"<{binary} {flag} failed rc={code}: {text.strip()}>"
    return text


def ptxas_command(ptx: Path, arch: str, out: Path) -> list[str]:
    """The ptxas invocation this probe uses; recorded in the report."""
    binary = os.environ.get("PTXAS", "/usr/local/cuda/bin/ptxas")
    return [
        binary,
        f"-arch={arch}",
        "-O3",
        "-v",
        "--warn-on-spills",
        str(ptx),
        "-o",
        str(out),
    ]


_PTXAS_FUNCTION_RE = re.compile(r"Function properties for (?P<name>\S+)")
_PTXAS_STACK_RE = re.compile(
    r"(?P<stack>\d+) bytes stack frame, (?P<spill_st>\d+) bytes spill stores, "
    r"(?P<spill_ld>\d+) bytes spill loads"
)
_PTXAS_USED_RE = re.compile(
    r"Used (?P<regs>\d+) registers(?:, used (?P<barriers>\d+) barriers)?"
    r"(?:, (?P<smem>\d+) bytes smem)?"
)


def parse_ptxas_resource_usage(text: str) -> list[dict[str, Any]]:
    """Group ptxas --resource-usage lines into one record per kernel."""
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        name = _PTXAS_FUNCTION_RE.search(line)
        if name is not None:
            current = {"function": name.group("name")}
            records.append(current)
            continue
        if current is None:
            continue
        stack = _PTXAS_STACK_RE.search(line)
        if stack is not None:
            current["stack_frame_bytes"] = int(stack.group("stack"))
            current["spill_store_bytes"] = int(stack.group("spill_st"))
            current["spill_load_bytes"] = int(stack.group("spill_ld"))
            continue
        used = _PTXAS_USED_RE.search(line)
        if used is not None:
            current["registers_per_thread"] = int(used.group("regs"))
            if used.group("smem"):
                current["smem_bytes"] = int(used.group("smem"))
    return records


_CUOBJDUMP_FUNCTION_RE = re.compile(
    r"Function\s+(?P<name>\S+)\s*:(?P<body>.*?)(?=\n\s*Function\s|\Z)", re.DOTALL
)
_CUOBJDUMP_FIELD_RE = re.compile(r"(REG|STACK|SHARED|LOCAL|TEXTURE|SURFACE|SAMPLER)\s*:\s*(\d+)")


def parse_cuobjdump_resource_usage(text: str) -> list[dict[str, Any]]:
    records = []
    for match in _CUOBJDUMP_FUNCTION_RE.finditer(text):
        fields = {
            key: int(value)
            for key, value in _CUOBJDUMP_FIELD_RE.findall(match.group("body"))
        }
        records.append({"function": match.group("name"), **fields})
    return records


def route_block_budget(block: int) -> dict[str, Any]:
    """The three per-CTA limits a route block has to satisfy, from the code.

    Computed for every block whether or not the kernel can be built for it, so
    an unbuildable block still reports why. ``layout_bytes_by_stages`` recomputes
    the kernel's own shared-memory layout (``W4A16GemmKernel.shared_words``) for
    hypothetical pipeline depths: the route metadata (one int4 unit per routed
    row), the B stages aliased with the epilogue reduction buffer, the A and
    scale stages, and the 4 KiB modal trellis table plus the 16-byte copy
    barrier the fused launch adds. At the served four stages it reproduces the
    kernel's own number exactly, which the probe asserts for every block it
    builds.
    """
    from b12x.moe._shared.kernels.w4a16.kernel import (
        _DEFAULT_MAX_SHARED_MEM,
        _DEVICE_MAX_REG_BYTES,
        _covering_count,
        _shared_memory_footprint,
        _w4a16_accumulator_regs_per_thread,
        _w4a16_b_unit_bytes,
    )

    cta_m_blocks = _covering_count(block, 16)
    accumulators = _w4a16_accumulator_regs_per_thread(
        cta_m_blocks=cta_m_blocks, tile_n=K3_TILE[1]
    )
    two_bpw = _w4a16_b_unit_bytes(weight_layout="trellis3_t256", trellis_bits=2)
    planner_bytes = _shared_memory_footprint(
        cta_m_blocks=cta_m_blocks,
        tile_n=K3_TILE[1],
        tile_k=K3_TILE[0],
        scale_format="e4m3_k32",
        weight_layout="trellis3_t256",
        b_unit_bytes=two_bpw,
    )
    # sh_block_route_indices + sh_rd_block_route_indices + sh_block_topk_weights
    # = block/4 + block/4 + block/2 int4 units.
    metadata_int4 = block
    reduction_int4 = (2 * (K3_TILE[1] // 16) + 1) * 16 * cta_m_blocks
    per_stage_int4 = cta_m_blocks * 16 * (16 * (K3_TILE[0] // 16) // 8) + 32
    return {
        "cta_m_blocks": cta_m_blocks,
        "accumulator_regs_per_thread": accumulators,
        "accumulator_regs_per_cta": accumulators * 256,
        "sm_register_file_share": round(
            accumulators * 256 * 4 / _DEVICE_MAX_REG_BYTES, 4
        ),
        "regs_left_for_kernel_body": MAX_REGS_PER_THREAD - accumulators,
        "planner_shared_bytes": planner_bytes,
        "layout_bytes_by_stages": {
            str(stages): (
                metadata_int4
                + max(reduction_int4, stages * 256)
                + stages * per_stage_int4
            )
            * 16
            + 4096
            + 16
            for stages in (2, 3, 4)
        },
        "max_shared_mem": _DEFAULT_MAX_SHARED_MEM,
    }


def probe_block(block: int, width: int, out_dir: Path) -> dict[str, Any]:
    """Plan, compile and measure one route block size."""
    from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    record: dict[str, Any] = {
        "block_m": block,
        "intermediate_size": width,
        "budget": route_block_budget(block),
    }

    captured: list[Any] = []
    original_compile = w4a16_kernel.b12x_compile
    original_stream = w4a16_kernel.current_cuda_stream
    original_resources = w4a16_kernel._query_w4a16_kernel_resources

    def _capturing_compile(func, *args, **kwargs):
        compiled = original_compile(func, *args, **kwargs)
        # The compiled entry point is the kernel object's traced call, so the
        # planned layout travels with the artifact.
        owner = getattr(func, "__self__", None)
        if owner is None:
            owner = getattr(func, "__wrapped__", None)
            owner = getattr(owner, "__self__", None)
        if owner is None and hasattr(func, "cta_threads"):
            owner = func
        captured.append((owner, compiled))
        return compiled

    def _null_stream():
        # The launch argument is only a handle in the lowered IR; compilation
        # never dereferences it, so the null stream stands in for the serving
        # stream on a host with no driver.
        from cuda.bindings import driver as cuda_driver

        return cuda_driver.CUstream(0)

    def _no_resources(_compiled):
        # Reading numRegs through cudaFuncGetAttributes needs a loaded module
        # on a device; this probe reads the same numbers out of the cubin.
        return None

    w4a16_kernel.b12x_compile = _capturing_compile
    w4a16_kernel.current_cuda_stream = _null_stream
    w4a16_kernel._query_w4a16_kernel_resources = _no_resources
    try:
        result = w4a16_kernel.compile_w4a16_fused_moe(**_fused_kwargs(block=block, width=width))
    except BaseException as exc:  # noqa: BLE001 - the rejection is the result
        import traceback

        record["planner"] = {
            "accepted": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc().splitlines()[-25:],
        }
        return record
    finally:
        w4a16_kernel.b12x_compile = original_compile
        w4a16_kernel.current_cuda_stream = original_stream
        w4a16_kernel._query_w4a16_kernel_resources = original_resources

    record["planner"] = {
        "accepted": True,
        "fc1_tile": [result.fc1_tile_k, result.fc1_tile_n],
        "fc2_tile": [result.fc2_tile_k, result.fc2_tile_n],
        "blocks_per_sm": result.blocks_per_sm,
    }
    tag = f"block{block}-w{width}"
    kernels = []
    for owner, _compiled in captured:
        if owner is None:
            continue
        entry: dict[str, Any] = {"owner": type(owner).__name__}
        for attribute in (
            "cta_threads",
            "cta_m_blocks",
            "moe_block_size",
            "tile_n",
            "tile_k",
            "blocks_per_sm",
            "b_unit_bytes",
        ):
            if hasattr(owner, attribute):
                entry[attribute] = int(getattr(owner, attribute))
        if hasattr(owner, "shared_words"):
            entry["planned_shared_bytes"] = int(owner.shared_words) * 4
            entry["layout_formula_matches"] = (
                int(owner.shared_words) * 4 + 16
                == record["budget"]["layout_bytes_by_stages"]["4"]
            )
        for part_name in ("fc1", "fc2"):
            part = getattr(owner, part_name, None)
            if part is None:
                continue
            part_entry = {}
            for attribute in ("cta_m_blocks", "cta_threads", "tile_n", "tile_k", "b_unit_bytes"):
                if hasattr(part, attribute):
                    part_entry[attribute] = int(getattr(part, attribute))
            if hasattr(part, "shared_words"):
                part_entry["planned_shared_bytes"] = int(part.shared_words) * 4
            entry[part_name] = part_entry
        kernels.append(entry)
    record["kernels"] = kernels

    # The DSL wrote PTX and cubin for every kernel it lowered; move them into a
    # per-block directory so a later block cannot overwrite them.
    block_dir = out_dir / tag
    block_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for path in sorted(out_dir.glob("*")):
        if path.is_dir():
            continue
        path.rename(block_dir / path.name)
    for ptx in sorted(block_dir.glob("*.ptx")):
        arch_match = re.search(r"\.(sm_\w+)\.ptx$", ptx.name)
        arch = arch_match.group(1) if arch_match else "sm_120a"
        cubin_out = ptx.with_suffix(".probe.cubin")
        command = ptxas_command(ptx, arch, cubin_out)
        code, text = _run(command)
        artifact: dict[str, Any] = {
            "ptx": ptx.name,
            "arch": arch,
            "ptxas_command": " ".join(command),
            "ptxas_returncode": code,
            "ptxas_resource_usage": parse_ptxas_resource_usage(text),
            "ptxas_output": text.strip().splitlines()[:40],
        }
        cubin = ptx.with_name(ptx.name.replace(".ptx", ".cubin"))
        if cubin.exists():
            artifact["cubin"] = cubin.name
            artifact["cubin_resource_usage"] = parse_cuobjdump_resource_usage(
                _cuobjdump(cubin, "--dump-resource-usage")
            )
        artifacts.append(artifact)
    record["artifacts"] = artifacts
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", default="16,32,48,64")
    parser.add_argument(
        "--width",
        type=int,
        default=384,
        help="rank-owned intermediate width (384 or 256 for the TP9 rotation)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="keep the emitted PTX/cubin here instead of a temporary directory",
    )
    parser.add_argument(
        "--arch",
        default="sm_120a",
        help="target architecture passed to the DSL and to ptxas",
    )
    args = parser.parse_args()

    os.environ.setdefault("B12X_SQG_XOR_CHEB_T12_SMEM", "1")
    # A device-less probe must not consult or pollute a shared compile cache.
    os.environ.setdefault("B12X_COMPILE_DISK_CACHE", "0")

    blocks = [int(value) for value in args.blocks.split(",") if value.strip()]
    out_dir = args.artifact_dir
    tmp: tempfile.TemporaryDirectory | None = None
    if out_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="w4a16-route-block-")
        out_dir = Path(tmp.name)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # The CuTe DSL writes PTX/cubin for every lowered kernel when asked; both
    # variables are read when cutlass is imported, so they must be set before
    # the first b12x import below.
    os.environ["CUTE_DSL_KEEP"] = os.environ.get("CUTE_DSL_KEEP", "ptx,cubin")
    os.environ["CUTE_DSL_DUMP_DIR"] = str(out_dir)
    os.environ.setdefault("CUTE_DSL_ARCH", args.arch)

    from b12x.moe._shared.kernels.w4a16.kernel import (
        _DEVICE_MAX_REG_BYTES,
        _DEFAULT_MAX_SHARED_MEM,
    )

    report: dict[str, Any] = {
        "geometry": {
            "hidden_size": K3_HIDDEN_SIZE,
            "intermediate_size": args.width,
            "num_experts": K3_NUM_EXPERTS,
            "top_k": K3_TOP_K,
            "capacity": K3_CAPACITY,
            "tile": list(K3_TILE),
            "sms": K3_SMS,
            "max_shared_mem": _DEFAULT_MAX_SHARED_MEM,
            "sm_register_file_bytes": _DEVICE_MAX_REG_BYTES,
            "max_regs_per_thread": MAX_REGS_PER_THREAD,
            "accumulators_per_m_block_per_thread": ACCUMULATORS_PER_M_BLOCK,
        },
        "records": [],
    }
    for block in blocks:
        print(f"[probe] route block {block}", flush=True)
        record = probe_block(block, args.width, out_dir)
        report["records"].append(record)
        planner = record["planner"]
        budget = record["budget"]
        print(
            "  budget: m_blocks={mb} acc/thread={acc} ({share:.1%} of the SM "
            "register file) body_regs<={body} layout@4stages={l4} "
            "layout@3stages={l3} limit={limit}".format(
                mb=budget["cta_m_blocks"],
                acc=budget["accumulator_regs_per_thread"],
                share=budget["sm_register_file_share"],
                body=budget["regs_left_for_kernel_body"],
                l4=budget["layout_bytes_by_stages"]["4"],
                l3=budget["layout_bytes_by_stages"]["3"],
                limit=budget["max_shared_mem"],
            ),
            flush=True,
        )
        for kernel in record.get("kernels", []):
            print(
                "  planned: {owner} threads={threads} m_blocks={mb} "
                "smem={smem} blocks_per_sm={bps}".format(
                    owner=kernel.get("owner"),
                    threads=kernel.get("cta_threads"),
                    mb=kernel.get("cta_m_blocks"),
                    smem=kernel.get("planned_shared_bytes"),
                    bps=kernel.get("blocks_per_sm"),
                ),
                flush=True,
            )
        if not planner["accepted"]:
            print(f"  planner rejected: {planner['error']}", flush=True)
            continue
        for artifact in record.get("artifacts", []):
            for usage in artifact["ptxas_resource_usage"]:
                print(
                    "  {name}: regs={regs} stack={stack} spill_st={st} "
                    "spill_ld={ld} smem={smem} ({arch})".format(
                        name=usage["function"][:56],
                        regs=usage.get("registers_per_thread", "?"),
                        stack=usage.get("stack_frame_bytes", "?"),
                        st=usage.get("spill_store_bytes", "?"),
                        ld=usage.get("spill_load_bytes", "?"),
                        smem=usage.get("smem_bytes", "?"),
                        arch=artifact["arch"],
                    ),
                    flush=True,
                )

    text = json.dumps(report, indent=1)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
