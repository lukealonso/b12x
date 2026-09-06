#!/usr/bin/env python3
"""Qualify one exact K6/MCG checkpoint projection through normal v39 policy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
from typing import Any


GPU_QUERY_FIELDS = (
    "timestamp",
    "index",
    "uuid",
    "name",
    "pci.bus_id",
    "pstate",
    "power.draw",
    "power.limit",
    "clocks.current.graphics",
    "clocks.current.memory",
    "clocks.current.sm",
    "clocks_throttle_reasons.active",
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
    "compute_mode",
    "driver_version",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_capture(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "command_shell": shlex.join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def gpu_snapshot(gpu: int) -> dict[str, Any]:
    return run_capture(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_rows(value: str) -> tuple[int, ...]:
    try:
        rows = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "rows must be comma-separated integers"
        ) from exc
    if not rows or len(set(rows)) != len(rows) or min(rows) <= 0:
        raise argparse.ArgumentTypeError("rows must be positive and unique")
    return rows


def image_id(image: str) -> str:
    result = run_capture(["docker", "image", "inspect", "--format", "{{.Id}}", image])
    if result["returncode"]:
        raise RuntimeError(f"cannot inspect image {image}: {result['stderr'].strip()}")
    return result["stdout"].strip()


def validate_result(
    path: Path,
    *,
    expected_grid_x: int,
    expected_scratch_elements: int | None,
    expected_rows: tuple[int, ...],
    experimental_grid_x: int | None,
) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("qualification_pass"):
        raise RuntimeError(f"checkpoint qualification failed: {path}")
    plan = result.get("b12x_plan", {})
    observed_grid = plan.get("launch", {}).get("grid_x")
    if observed_grid != expected_grid_x:
        raise RuntimeError(
            f"planner grid mismatch: expected={expected_grid_x}, observed={observed_grid}"
        )
    observed_scratch = plan.get("fused_scratch_elements")
    if (
        expected_scratch_elements is not None
        and observed_scratch != expected_scratch_elements
    ):
        raise RuntimeError(
            "planner scratch mismatch: "
            f"expected={expected_scratch_elements}, observed={observed_scratch}"
        )
    override = plan.get("planner_override")
    if experimental_grid_x is None and override is not None:
        raise RuntimeError(f"normal-planner run unexpectedly used override: {override}")
    if experimental_grid_x is not None:
        if not override or override.get("requested_grid_x") != experimental_grid_x:
            raise RuntimeError(f"experimental planner override mismatch: {override}")
    observed_rows = tuple(row.get("rows") for row in result.get("rows", []))
    if observed_rows != expected_rows:
        raise RuntimeError(
            f"row coverage mismatch: expected={expected_rows}, observed={observed_rows}"
        )
    for row in result["rows"]:
        dot = row["captures"]["fused_b12x"]["dot"]
        if (
            not row.get("pass")
            or not dot.get("pass")
            or dot.get("k6_mcg_small_m_count") != 1
            or dot.get("separate_rotation_like_count") != 0
        ):
            raise RuntimeError(f"M={row['rows']} graph identity or row gate failed")
    return result


def result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "qualification_pass": result["qualification_pass"],
        "shape_k_n": [
            result["b12x_plan"]["launch"]["size_k"],
            result["b12x_plan"]["launch"]["size_n"],
        ],
        "grid_x": result["b12x_plan"]["launch"]["grid_x"],
        "scratch_elements": result["b12x_plan"]["fused_scratch_elements"],
        "planner_override": result["b12x_plan"]["planner_override"],
        "rows": {
            str(row["rows"]): {
                "fused_median_us": row["warm_graph_timing"]["summary"]["fused_b12x"][
                    "median_ms"
                ]
                * 1000.0,
                "exllamav3_median_us": row["warm_graph_timing"]["summary"][
                    "served_exllamav3"
                ]["median_ms"]
                * 1000.0,
                "exllamav3_over_fused_latency": row["warm_graph_timing"]["ratios"][
                    "exllamav3_over_fused_latency"
                ],
                "correctness": row["correctness"],
                "active_gpu_snapshot": row["warm_graph_timing"]["active_gpu_snapshot"],
            }
            for row in result["rows"]
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-tree", type=Path, required=True)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--image", required=True)
    result.add_argument("--image-id", required=True)
    result.add_argument("--source-revision", required=True)
    result.add_argument("--integration-tree", required=True)
    result.add_argument("--tensor-prefix", required=True)
    result.add_argument("--params-dtype", choices=("fp16", "bf16"), required=True)
    result.add_argument("--expected-grid-x", type=int, required=True)
    result.add_argument("--expected-scratch-elements", type=int)
    result.add_argument("--experimental-grid-x", type=int)
    result.add_argument("--rows", type=parse_rows, default=(1, 4, 8, 12, 16))
    result.add_argument("--gpu", type=int, default=6)
    result.add_argument("--warmups", type=int, default=6000)
    result.add_argument("--iterations", type=int, default=600)
    result.add_argument("--cold-replays", type=int, default=12)
    result.add_argument(
        "--verify-shard-sha",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return result


def main() -> int:
    args = parser().parse_args()
    args.source_tree = args.source_tree.resolve()
    args.model_dir = args.model_dir.resolve()
    args.output_root = args.output_root.resolve()
    if not args.source_tree.is_dir() or not args.model_dir.is_dir():
        raise FileNotFoundError("source tree and model directory must exist")
    if args.output_root.exists():
        raise FileExistsError(
            f"output root already exists; choose a fresh path: {args.output_root}"
        )
    if min(args.expected_grid_x, args.warmups, args.iterations, args.cold_replays) <= 0:
        raise ValueError("grid and timing counts must be positive")
    if args.iterations % 6 or args.cold_replays % 6:
        raise ValueError("iterations and cold replays must be multiples of six")
    resolved_image_id = image_id(args.image)
    if resolved_image_id != args.image_id:
        raise RuntimeError(
            f"image identity mismatch: declared={args.image_id}, resolved={resolved_image_id}"
        )
    args.output_root.mkdir(parents=True)

    adapter = args.source_tree / (
        "validation/trellis_decode/glm52_o4096x6144_optimization/"
        "benchmark_v39_checkpoint.py"
    )
    benchmark = args.source_tree / "benchmarks/benchmark_trellis_k6_mcg_checkpoint.py"
    git_diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=args.source_tree,
        capture_output=True,
        check=True,
    ).stdout
    safe_prefix = re.sub(r"[^a-zA-Z0-9]+", "-", args.tensor_prefix).strip("-")[-48:]
    container_name = f"b12x-k6-qual-{safe_prefix}-{os.getpid()}"
    command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--ipc=none",
        "--name",
        container_name,
        "--gpus",
        f"device={args.gpu}",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--entrypoint",
        "/opt/venv/bin/python",
        "-e",
        "PYTHONPATH=/workspace",
        "-e",
        "CUTE_DSL_ARCH=sm_120a",
        "-e",
        "TORCH_CUDA_ARCH_LIST=12.0a",
        "-e",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "-e",
        "SPARKINFER_COMPILE_CACHE_DIR=/output/sparkinfer-cache",
        "-e",
        "CUTE_DSL_CACHE_DIR=/output/cute-dsl-cache",
        "-e",
        "CUDA_CACHE_PATH=/output/cuda-cache",
        "-e",
        "TORCH_EXTENSIONS_DIR=/output/torch-extensions",
        "-e",
        "XDG_CACHE_HOME=/output/xdg-cache",
        "-v",
        f"{args.source_tree}:/workspace:ro",
        "-v",
        f"{args.model_dir}:/model:ro",
        "-v",
        f"{args.output_root}:/output",
        args.image,
        "/workspace/validation/trellis_decode/"
        "glm52_o4096x6144_optimization/benchmark_v39_checkpoint.py",
        "--model-dir",
        "/model",
        "--tensor-prefix",
        args.tensor_prefix,
        "--rows",
        ",".join(str(row) for row in args.rows),
        "--params-dtype",
        args.params_dtype,
        "--graph-dump-dir",
        "/output/graphs",
        "--output",
        "/output/result.json",
        "--compile-warmups",
        "3",
        "--replay-checks",
        "8",
        "--cold-replays",
        str(args.cold_replays),
        "--warmups",
        str(args.warmups),
        "--iterations",
        str(args.iterations),
        "--source-revision",
        args.source_revision,
        "--integration-tree",
        args.integration_tree,
        "--image",
        args.image,
        "--image-id",
        args.image_id,
        "--verify-shard-sha" if args.verify_shard_sha else "--no-verify-shard-sha",
    ]
    if args.experimental_grid_x is not None:
        command.extend(["--experimental-grid-x", str(args.experimental_grid_x)])

    receipt: dict[str, Any] = {
        "schema": "b12x.k6_mcg_checkpoint_policy_qualification.v1",
        "started_at": utc_now(),
        "completed_at": None,
        "argv": sys.argv,
        "host": platform.node(),
        "source_tree": str(args.source_tree),
        "model_dir": str(args.model_dir),
        "output_root": str(args.output_root),
        "tensor_prefix": args.tensor_prefix,
        "params_dtype": args.params_dtype,
        "rows": list(args.rows),
        "expected_grid_x": args.expected_grid_x,
        "expected_scratch_elements": args.expected_scratch_elements,
        "experimental_grid_x": args.experimental_grid_x,
        "image": args.image,
        "image_id": args.image_id,
        "source_revision": args.source_revision,
        "integration_tree": args.integration_tree,
        "physical_gpu_index": args.gpu,
        "timing": {
            "warmups": args.warmups,
            "iterations": args.iterations,
            "cold_replays": args.cold_replays,
            "route_order": "all six permutations cycled equally",
            "ratio_direction": "exllamav3_over_fused_latency > 1 favors fused",
        },
        "source_identity": {
            "adapter_sha256": sha256_file(adapter),
            "benchmark_sha256": sha256_file(benchmark),
            "git_diff_sha256": sha256_bytes(git_diff),
            "git_head": run_capture(["git", "rev-parse", "HEAD"], cwd=args.source_tree),
            "git_status": run_capture(
                ["git", "status", "--short"], cwd=args.source_tree
            ),
        },
        "docker_image_inspect": run_capture(["docker", "image", "inspect", args.image]),
        "host_gpu_before": gpu_snapshot(args.gpu),
        "command": command,
        "command_shell": shlex.join(command),
        "fresh_cache_directories": [
            str(args.output_root / name)
            for name in (
                "sparkinfer-cache",
                "cute-dsl-cache",
                "cuda-cache",
                "torch-extensions",
                "xdg-cache",
            )
        ],
    }
    receipt_path = args.output_root / "qualification_receipt.json"
    write_json(receipt_path, receipt)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (args.output_root / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (args.output_root / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    receipt.update(
        {
            "completed_at": utc_now(),
            "returncode": completed.returncode,
            "host_gpu_after": gpu_snapshot(args.gpu),
            "stdout_sha256": sha256_file(args.output_root / "stdout.log"),
            "stderr_sha256": sha256_file(args.output_root / "stderr.log"),
        }
    )
    write_json(receipt_path, receipt)
    if completed.returncode:
        raise RuntimeError(
            f"benchmark container failed with exit {completed.returncode}; "
            f"see {args.output_root}"
        )
    result = validate_result(
        args.output_root / "result.json",
        expected_grid_x=args.expected_grid_x,
        expected_scratch_elements=args.expected_scratch_elements,
        expected_rows=args.rows,
        experimental_grid_x=args.experimental_grid_x,
    )
    receipt["summary"] = result_summary(result)
    write_json(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), "summary": receipt["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
