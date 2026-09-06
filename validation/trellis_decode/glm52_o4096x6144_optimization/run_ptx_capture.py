#!/usr/bin/env python3
"""Capture retained PTX for every required grid in isolated containers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


GRIDS = (48, 64, 80, 96, 112, 120, 128, 144, 160, 176, 188)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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
    return result


def main() -> int:
    args = parser().parse_args()
    args.source_tree = args.source_tree.resolve()
    args.model_dir = args.model_dir.resolve()
    args.output_root = args.output_root.resolve()
    if args.output_root.exists():
        raise FileExistsError(f"output root must be fresh: {args.output_root}")
    args.output_root.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": "b12x.glm52_o4096x6144_retained_ptx_matrix.v1",
        "started_at": utc_now(),
        "completed_at": None,
        "argv": sys.argv,
        "source_tree": str(args.source_tree),
        "model_dir": str(args.model_dir),
        "output_root": str(args.output_root),
        "image": args.image,
        "image_id": args.image_id,
        "source_revision": args.source_revision,
        "integration_tree": args.integration_tree,
        "physical_gpu_index": args.gpu,
        "grids": list(GRIDS),
        "runs": [],
        "pass": False,
    }
    manifest_path = args.output_root / "capture_manifest.json"
    write_json(manifest_path, manifest)
    for grid in GRIDS:
        grid_dir = args.output_root / f"grid_{grid:03d}"
        grid_dir.mkdir()
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            f"b12x-glm-o-ptx-g{grid}-{os.getpid()}",
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
            "CUTE_DSL_KEEP=ptx",
            "-e",
            "CUTE_DSL_DUMP_DIR=/output/cute-dsl-dump",
            "-e",
            "CUDA_CACHE_PATH=/output/cuda-cache",
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
            "glm52_o4096x6144_optimization/compile_v39_grid_artifact.py",
            "--model-dir",
            "/model",
            "--output",
            "/output/result.json",
            "--grid",
            str(grid),
            "--source-revision",
            args.source_revision,
            "--integration-tree",
            args.integration_tree,
            "--image",
            args.image,
            "--image-id",
            args.image_id,
        ]
        record: dict[str, Any] = {
            "grid_x": grid,
            "started_at": utc_now(),
            "command": command,
            "command_shell": shlex.join(command),
        }
        write_json(grid_dir / "run.json", record)
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        (grid_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (grid_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        record.update({"completed_at": utc_now(), "returncode": completed.returncode})
        write_json(grid_dir / "run.json", record)
        manifest["runs"].append(record)
        write_json(manifest_path, manifest)
        if completed.returncode:
            raise RuntimeError(f"grid {grid} PTX capture failed: {grid_dir}")
        result = json.loads((grid_dir / "result.json").read_text(encoding="utf-8"))
        if not result.get("pass") or result.get("grid_x") != grid:
            raise RuntimeError(f"grid {grid} PTX result did not qualify")
    manifest["completed_at"] = utc_now()
    manifest["pass"] = True
    write_json(manifest_path, manifest)
    print(json.dumps({"output": str(manifest_path), "pass": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
