#!/usr/bin/env python3
"""Compile one exact GLM TP4 o_proj grid while retaining CUTLASS PTX."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import torch

from validation.trellis_decode.glm52_o4096x6144_optimization import (
    benchmark_v39_checkpoint as adapter,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    return [
        {
            "path": str(path.resolve()),
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--grid", type=int, required=True)
    result.add_argument("--source-revision", required=True)
    result.add_argument("--integration-tree", required=True)
    result.add_argument("--image", required=True)
    result.add_argument("--image-id", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    report: dict[str, Any] = {
        "schema": "b12x.glm52_o4096x6144_retained_ptx_compile.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "grid_x": args.grid,
        "source_revision": args.source_revision,
        "integration_tree": args.integration_tree,
        "image": args.image,
        "image_id": args.image_id,
        "pass": False,
    }
    exit_code = 0
    try:
        if args.grid <= 0:
            raise ValueError("grid must be positive")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        if torch.cuda.get_device_capability(device) != (12, 0):
            raise RuntimeError("retained artifact compile requires sm_120")
        cache_dir = Path(os.environ["SPARKINFER_COMPILE_CACHE_DIR"])
        dump_dir = Path(os.environ["CUTE_DSL_DUMP_DIR"])
        keep_tokens = {
            token.strip().lower()
            for token in os.environ.get("CUTE_DSL_KEEP", "").split(",")
            if token.strip()
        }
        if "ptx" not in keep_tokens:
            raise RuntimeError("CUTE_DSL_KEEP must include ptx")
        if cache_dir.exists() and any(cache_dir.rglob("*")):
            raise RuntimeError(f"compile cache is not fresh: {cache_dir}")
        if dump_dir.exists() and any(dump_dir.rglob("*")):
            raise RuntimeError(f"CUTLASS dump directory is not fresh: {dump_dir}")
        dump_dir.mkdir(parents=True, exist_ok=True)
        payload = adapter._load_glm_payload(
            args.model_dir,
            "model.layers.3.self_attn.o_proj",
            verify_shard_sha=True,
        )
        gpu_tensors = {
            "trellis": payload.trellis.to(device).contiguous(),
            "suh": payload.suh.to(device).contiguous(),
            "svh": payload.svh.to(device).contiguous(),
            "mcg": payload.mcg.to(device).contiguous(),
        }
        before_hashes = {
            name: adapter.benchmark._sha256_tensor(tensor)
            for name, tensor in gpu_tensors.items()
        }
        started_ns = time.perf_counter_ns()
        with adapter.benchmark._temporary_k6_mcg_grid_override(
            size_k=4096,
            size_n=6144,
            requested_grid_x=args.grid,
        ) as planner_override:
            weight = adapter.trellis_linear.prepare_weight(
                gpu_tensors["trellis"],
                gpu_tensors["suh"],
                gpu_tensors["svh"],
                mcg=gpu_tensors["mcg"],
                params_dtype=torch.float16,
            )
        torch.cuda.synchronize(device)
        compile_wall_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
        launch = weight.k6_mcg_small_m_launch
        if launch is None or int(launch.grid_x) != args.grid:
            raise RuntimeError(
                f"requested grid {args.grid} did not bind: {getattr(launch, 'grid_x', None)}"
            )
        after_hashes = {
            name: adapter.benchmark._sha256_tensor(tensor)
            for name, tensor in gpu_tensors.items()
        }
        if before_hashes != after_hashes:
            raise RuntimeError("checkpoint tensors changed during prepare")
        cache_files = file_inventory(cache_dir)
        dump_files = file_inventory(dump_dir)
        ptx_files = [
            item
            for item in dump_files
            if Path(item["relative_path"]).suffix.lower() == ".ptx"
        ]
        object_files = [
            item
            for item in cache_files
            if Path(item["relative_path"]).suffix.lower() == ".o"
        ]
        manifest_files = [
            item
            for item in cache_files
            if Path(item["relative_path"]).suffix.lower() == ".json"
        ]
        if len(object_files) != 1 or len(manifest_files) != 1:
            raise RuntimeError(
                "expected one compiled object and manifest, got "
                f"objects={len(object_files)} manifests={len(manifest_files)}"
            )
        if not ptx_files:
            raise RuntimeError("CUTLASS retained no PTX files")
        report.update(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "environment": adapter._environment_metadata(args, device),
                "checkpoint": payload.metadata,
                "planner_override": planner_override,
                "compile_wall_ms": compile_wall_ms,
                "launch": {
                    "grid_x": int(launch.grid_x),
                    "cta_threads": int(launch.cta_threads),
                    "resident_ctas": int(launch.resident_ctas),
                    "blocks_per_sm": int(launch.blocks_per_sm),
                    "shared_memory_bytes": int(launch.shared_memory_bytes),
                    "required_scratch_elements": int(launch.required_scratch_elements),
                    "required_workspace_elements": int(
                        launch.required_workspace_elements
                    ),
                },
                "source_immutability": {
                    "before": before_hashes,
                    "after": after_hashes,
                    "pass": True,
                },
                "compile_cache": cache_files,
                "cutlass_dump": dump_files,
                "ptx_files": ptx_files,
                "gpu_snapshot": adapter.benchmark._gpu_snapshot(),
                "pass": True,
            }
        )
    except Exception as exc:
        exit_code = 2
        report.update(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "fatal_error": {
                    "type": f"{type(exc).__module__}.{type(exc).__name__}",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "grid_x": args.grid,
                "pass": report["pass"],
                "fatal_error": report.get("fatal_error", {}).get("message"),
            }
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
