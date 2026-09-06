#!/usr/bin/env python3
"""Run the required GLM o_proj CTA-grid sweep in isolated v39 containers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Any


DEFAULT_GRIDS = (48, 64, 80, 96, 112, 120, 128, 144, 160, 176, 188)
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


def resolve_image_id(image: str) -> tuple[str, dict[str, Any]]:
    inspection = run_capture(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image]
    )
    resolved = inspection["stdout"].strip()
    if inspection["returncode"] != 0 or not resolved:
        raise RuntimeError(
            f"cannot resolve local image identity for {image}: "
            f"{inspection['stderr'].strip()}"
        )
    if "\n" in resolved:
        raise RuntimeError(f"image reference resolves to multiple IDs: {image}")
    return resolved, inspection


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


def parse_grids(value: str) -> tuple[int, ...]:
    try:
        grids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "grids must be comma-separated integers"
        ) from exc
    if not grids or len(set(grids)) != len(grids) or min(grids) <= 0:
        raise argparse.ArgumentTypeError("grids must be positive and unique")
    return grids


def validate_result(path: Path, expected_grid: int) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("qualification_pass"):
        raise RuntimeError(f"grid {expected_grid} did not qualify: {path}")
    plan = result.get("b12x_plan", {})
    observed_grid = plan.get("launch", {}).get("grid_x")
    if observed_grid != expected_grid:
        raise RuntimeError(
            f"grid {expected_grid} bound unexpected launch grid {observed_grid}"
        )
    expected_rows = [1, 4, 8, 12, 16]
    observed_rows = [row.get("rows") for row in result.get("rows", [])]
    if observed_rows != expected_rows:
        raise RuntimeError(
            f"grid {expected_grid} row coverage mismatch: {observed_rows}"
        )
    for row in result["rows"]:
        dot = row["captures"]["fused_b12x"]["dot"]
        if (
            not dot.get("pass")
            or dot.get("k6_mcg_small_m_count") != 1
            or dot.get("separate_rotation_like_count") != 0
        ):
            raise RuntimeError(
                f"grid {expected_grid} M={row['rows']} graph identity failed"
            )
    return result


def summary_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "grid_x": result["b12x_plan"]["launch"]["grid_x"],
        "scratch_elements": result["b12x_plan"]["fused_scratch_elements"],
        "qualification_pass": result["qualification_pass"],
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
    result.add_argument("--gpu", type=int, default=6)
    result.add_argument("--grids", type=parse_grids, default=DEFAULT_GRIDS)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--warmups", type=int, default=60)
    result.add_argument("--iterations", type=int, default=240)
    result.add_argument("--cold-replays", type=int, default=12)
    return result


def main() -> int:
    args = parser().parse_args()
    args.source_tree = args.source_tree.resolve()
    args.model_dir = args.model_dir.resolve()
    args.output_root = args.output_root.resolve()
    if not args.source_tree.is_dir() or not args.model_dir.is_dir():
        raise FileNotFoundError("source tree and model directory must exist")
    if args.output_root.exists() and not args.resume:
        raise FileExistsError(
            f"output root already exists; choose a fresh path: {args.output_root}"
        )
    resolved_image_id, image_id_inspect = resolve_image_id(args.image)
    if resolved_image_id != args.image_id:
        raise RuntimeError(
            "image identity mismatch: "
            f"declared={args.image_id}, resolved={resolved_image_id}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

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
    manifest: dict[str, Any] = {
        "schema": "b12x.glm52_o4096x6144_grid_sweep.v1",
        "started_at": utc_now(),
        "completed_at": None,
        "argv": sys.argv,
        "host": platform.node(),
        "source_tree": str(args.source_tree),
        "model_dir": str(args.model_dir),
        "output_root": str(args.output_root),
        "image": args.image,
        "image_id": resolved_image_id,
        "image_id_inspect": image_id_inspect,
        "source_revision": args.source_revision,
        "integration_tree": args.integration_tree,
        "physical_gpu_index": args.gpu,
        "grids": list(args.grids),
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
        "runs": [],
        "summary": [],
    }
    manifest_path = args.output_root / "sweep_manifest.json"
    write_json(manifest_path, manifest)

    for grid in args.grids:
        grid_dir = args.output_root / f"grid_{grid:03d}"
        result_path = grid_dir / "result.json"
        if args.resume and result_path.is_file():
            result = validate_result(result_path, grid)
            manifest["runs"].append(
                {"grid_x": grid, "resumed": True, "result": str(result_path)}
            )
            manifest["summary"].append(summary_row(result))
            write_json(manifest_path, manifest)
            continue
        if grid_dir.exists():
            raise FileExistsError(
                f"partial grid directory exists; inspect it before rerun: {grid_dir}"
            )
        grid_dir.mkdir(parents=True)
        container_name = f"b12x-glm-o-g{grid}-{os.getpid()}"
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
            f"{grid_dir}:/output",
            args.image,
            "/workspace/validation/trellis_decode/"
            "glm52_o4096x6144_optimization/benchmark_v39_checkpoint.py",
            "--model-dir",
            "/model",
            "--tensor-prefix",
            "model.layers.3.self_attn.o_proj",
            "--rows",
            "1,4,8,12,16",
            "--params-dtype",
            "fp16",
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
            "--experimental-grid-x",
            str(grid),
            "--source-revision",
            args.source_revision,
            "--integration-tree",
            args.integration_tree,
            "--image",
            args.image,
            "--image-id",
            resolved_image_id,
        ]
        run_record: dict[str, Any] = {
            "grid_x": grid,
            "started_at": utc_now(),
            "host_gpu_before": gpu_snapshot(args.gpu),
            "command": command,
            "command_shell": shlex.join(command),
            "fresh_cache_directories": [
                str(grid_dir / "sparkinfer-cache"),
                str(grid_dir / "cute-dsl-cache"),
                str(grid_dir / "cuda-cache"),
                str(grid_dir / "torch-extensions"),
                str(grid_dir / "xdg-cache"),
            ],
        }
        write_json(grid_dir / "run.json", run_record)
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        (grid_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (grid_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        run_record.update(
            {
                "completed_at": utc_now(),
                "returncode": completed.returncode,
                "host_gpu_after": gpu_snapshot(args.gpu),
                "stdout_sha256": sha256_file(grid_dir / "stdout.log"),
                "stderr_sha256": sha256_file(grid_dir / "stderr.log"),
            }
        )
        write_json(grid_dir / "run.json", run_record)
        manifest["runs"].append(run_record)
        write_json(manifest_path, manifest)
        if completed.returncode != 0:
            raise RuntimeError(
                f"grid {grid} container failed with exit {completed.returncode}; "
                f"see {grid_dir}"
            )
        result = validate_result(result_path, grid)
        manifest["summary"].append(summary_row(result))
        write_json(manifest_path, manifest)

    manifest["host_gpu_after"] = gpu_snapshot(args.gpu)
    manifest["completed_at"] = utc_now()
    write_json(manifest_path, manifest)
    print(json.dumps({"output": str(manifest_path), "summary": manifest["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
