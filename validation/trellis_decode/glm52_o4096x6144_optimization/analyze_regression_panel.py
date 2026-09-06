#!/usr/bin/env python3
"""Analyze a balanced K6/MCG base-candidate-base regression panel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

from audit_grid_artifacts import (
    embedded_cuda_elf,
    sha256_bytes,
    validate_manifest,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_arm(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    receipt_path = directory / "qualification_receipt.json"
    result_path = directory / "result.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifests = list((directory / "sparkinfer-cache").glob("*/*.json"))
    target_manifests = []
    for path in manifests:
        raw = json.loads(path.read_text(encoding="utf-8"))
        compile_spec = json.loads(raw.get("compile_spec_json", "{}"))
        if compile_spec.get("kernel") == "gemm.trellis.k6_mcg_small_m":
            target_manifests.append(path)
    if len(target_manifests) != 1:
        raise ValueError(
            f"{directory}: expected one K6/MCG manifest, got "
            f"{len(target_manifests)} among {len(manifests)} manifests"
        )
    manifest_path = target_manifests[0]
    object_path = manifest_path.with_suffix(".o")
    if not object_path.is_file():
        raise ValueError(f"{directory}: K6/MCG object is missing: {object_path}")
    object_bytes = object_path.read_bytes()
    manifest = validate_manifest(manifest_path, object_bytes)
    cubin = embedded_cuda_elf(object_bytes)
    return {
        "directory": str(directory),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "receipt": receipt,
        "result": result,
        "artifact": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "cache_key": manifest["cache_key"],
            "package_fingerprint": manifest["package_fingerprint"],
            "compile_spec_hash": manifest["compile_spec_hash"],
            "semantic_key": manifest["semantic_key"],
            "launch_metadata": manifest["launch_metadata"],
            "object_path": str(object_path),
            "object_bytes": len(object_bytes),
            "object_sha256": sha256_bytes(object_bytes),
            "cubin_bytes": len(cubin),
            "cubin_sha256": sha256_bytes(cubin),
        },
    }


def parse_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    fields = [item.strip() for item in snapshot["stdout"].strip().split(",")]
    if len(fields) != 15:
        raise ValueError(f"unexpected active GPU snapshot: {snapshot}")
    return {
        "timestamp": fields[0],
        "name": fields[1],
        "uuid": fields[2],
        "pci_bus_id": fields[3],
        "compute_capability": fields[4],
        "memory_total_mib": int(fields[5]),
        "memory_used_mib": int(fields[6]),
        "power_w": float(fields[7]),
        "power_limit_w": float(fields[8]),
        "pstate": fields[9],
        "graphics_clock_mhz": int(fields[10]),
        "memory_clock_mhz": int(fields[11]),
        "sm_clock_mhz": int(fields[12]),
        "compute_mode": fields[13],
        "throttle_mask": fields[14],
    }


def timing_rows(arm: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in arm["result"]["rows"]:
        rows = int(row["rows"])
        result[rows] = {
            "fused_median_us": row["warm_graph_timing"]["summary"]["fused_b12x"][
                "median_ms"
            ]
            * 1000.0,
            "fused_samples_us": [
                sample["milliseconds"] * 1000.0
                for sample in row["warm_graph_timing"]["raw_samples"]["fused_b12x"]
            ],
            "exllamav3_median_us": row["warm_graph_timing"]["summary"][
                "served_exllamav3"
            ]["median_ms"]
            * 1000.0,
            "telemetry": parse_snapshot(
                row["warm_graph_timing"]["active_gpu_snapshot"]
            ),
        }
    return result


def validate_arm(
    label: str,
    arm: dict[str, Any],
    *,
    size_k: int,
    size_n: int,
    grid_x: int,
    rows: tuple[int, ...],
) -> list[str]:
    failures: list[str] = []
    result = arm["result"]
    launch = result.get("b12x_plan", {}).get("launch", {})
    if not result.get("qualification_pass"):
        failures.append(f"{label}: qualification failed")
    if (launch.get("size_k"), launch.get("size_n")) != (size_k, size_n):
        failures.append(f"{label}: shape mismatch")
    if launch.get("grid_x") != grid_x:
        failures.append(f"{label}: grid mismatch")
    if result.get("b12x_plan", {}).get("planner_override") is not None:
        failures.append(f"{label}: benchmark planner override was used")
    observed_rows = tuple(int(row["rows"]) for row in result.get("rows", []))
    if observed_rows != rows:
        failures.append(f"{label}: row coverage mismatch {observed_rows}")
    if not result.get("source_immutability", {}).get("pass"):
        failures.append(f"{label}: checkpoint source mutation")
    for row in result.get("rows", []):
        if not row.get("pass") or not row.get("correctness", {}).get("pass"):
            failures.append(f"{label}: M={row['rows']} row/correctness failure")
    return failures


def prior_exact_evidence(path: Path | None, size_k: int, size_n: int) -> dict | None:
    if path is None:
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    launch = result.get("b12x_plan", {}).get("launch", {})
    if not result.get("qualification_pass"):
        raise ValueError("prior exact-checkpoint evidence did not qualify")
    if (launch.get("size_k"), launch.get("size_n")) != (size_k, size_n):
        raise ValueError("prior exact-checkpoint shape mismatch")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_revision": result["environment"]["declared_identity"][
            "source_revision"
        ],
        "checkpoint": result["checkpoint"],
        "params_dtype": result["b12x_plan"]["params_dtype"],
        "launch": launch,
        "qualification_pass": True,
        "timing": {
            str(row["rows"]): {
                "fused_median_us": row["warm_graph_timing"]["summary"]["fused_b12x"][
                    "median_ms"
                ]
                * 1000.0,
                "exllamav3_median_us": row["warm_graph_timing"]["summary"][
                    "served_exllamav3"
                ]["median_ms"]
                * 1000.0,
            }
            for row in result["rows"]
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-before", type=Path, required=True)
    result.add_argument("--candidate", type=Path, required=True)
    result.add_argument("--base-after", type=Path, required=True)
    result.add_argument("--size-k", type=int, required=True)
    result.add_argument("--size-n", type=int, required=True)
    result.add_argument("--grid-x", type=int, required=True)
    result.add_argument("--rows", default="1,4,8,16")
    result.add_argument("--max-regression-percent", type=float, default=1.0)
    result.add_argument(
        "--max-sm-clock-spread-mhz",
        type=int,
        default=30,
        help="maximum allowed per-row SM-clock spread across panel arms",
    )
    result.add_argument("--prior-exact-result", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.max_sm_clock_spread_mhz < 0:
        raise ValueError("--max-sm-clock-spread-mhz must be nonnegative")
    rows = tuple(int(item) for item in args.rows.split(","))
    arms = {
        "base_before": read_arm(args.base_before),
        "candidate": read_arm(args.candidate),
        "base_after": read_arm(args.base_after),
    }
    failures: list[str] = []
    for label, arm in arms.items():
        failures.extend(
            validate_arm(
                label,
                arm,
                size_k=args.size_k,
                size_n=args.size_n,
                grid_x=args.grid_x,
                rows=rows,
            )
        )

    checkpoint_tensors = [
        arm["result"]["checkpoint"]["tensors"] for arm in arms.values()
    ]
    if not all(value == checkpoint_tensors[0] for value in checkpoint_tensors[1:]):
        failures.append("checkpoint tensor identities differ between arms")
    artifacts = [arm["artifact"] for arm in arms.values()]
    for field in ("compile_spec_hash", "semantic_key", "cubin_sha256"):
        if len({artifact[field] for artifact in artifacts}) != 1:
            failures.append(f"compiled artifact {field} differs between arms")

    timings = {label: timing_rows(arm) for label, arm in arms.items()}
    comparisons: dict[str, Any] = {}
    allowed_regression = args.max_regression_percent / 100.0
    for row in rows:
        outer = statistics.fmean(
            (
                timings["base_before"][row]["fused_median_us"],
                timings["base_after"][row]["fused_median_us"],
            )
        )
        candidate = timings["candidate"][row]["fused_median_us"]
        ratio = candidate / outer
        regression_percent = (ratio - 1.0) * 100.0
        outer_drift_percent = (
            abs(
                timings["base_after"][row]["fused_median_us"]
                / timings["base_before"][row]["fused_median_us"]
                - 1.0
            )
            * 100.0
        )
        comparisons[str(row)] = {
            "base_before_median_us": timings["base_before"][row]["fused_median_us"],
            "candidate_median_us": candidate,
            "base_after_median_us": timings["base_after"][row]["fused_median_us"],
            "outer_mean_us": outer,
            "candidate_over_outer_latency": ratio,
            "candidate_regression_percent": regression_percent,
            "outer_drift_percent": outer_drift_percent,
            "pass": ratio <= 1.0 + allowed_regression,
        }
        if ratio > 1.0 + allowed_regression:
            failures.append(
                f"M={row}: {regression_percent:.6f}% regression exceeds "
                f"{args.max_regression_percent:.6f}%"
            )
    for row in rows:
        sm_clocks = {
            label: int(per_row[row]["telemetry"]["sm_clock_mhz"])
            for label, per_row in timings.items()
        }
        sm_clock_spread = max(sm_clocks.values()) - min(sm_clocks.values())
        comparisons[str(row)]["sm_clock_mhz_by_arm"] = sm_clocks
        comparisons[str(row)]["sm_clock_spread_mhz"] = sm_clock_spread
        comparisons[str(row)]["sm_clock_spread_pass"] = (
            sm_clock_spread <= args.max_sm_clock_spread_mhz
        )
        if sm_clock_spread > args.max_sm_clock_spread_mhz:
            failures.append(
                f"M={row}: SM-clock spread {sm_clock_spread} MHz exceeds "
                f"{args.max_sm_clock_spread_mhz} MHz: {sm_clocks}"
            )

    for label, per_row in timings.items():
        for row, values in per_row.items():
            telemetry = values["telemetry"]
            if telemetry["pstate"] != "P1":
                failures.append(f"{label} M={row}: pstate={telemetry['pstate']}")
            if telemetry["throttle_mask"] not in {
                "0x0000000000000000",
                "0x0000000000000004",
            }:
                failures.append(
                    f"{label} M={row}: throttle={telemetry['throttle_mask']}"
                )
    uuids = {
        values["telemetry"]["uuid"]
        for per_row in timings.values()
        for values in per_row.values()
    }
    if len(uuids) != 1:
        failures.append(f"physical GPU UUID differs: {sorted(uuids)}")
    if not all(
        math.isfinite(item["candidate_over_outer_latency"])
        for item in comparisons.values()
    ):
        failures.append("non-finite timing ratio")

    compact_arms = {
        label: {
            key: value for key, value in arm.items() if key not in {"receipt", "result"}
        }
        for label, arm in arms.items()
    }
    receipt = {
        "schema": "b12x.k6_mcg_regression_panel.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "order": ["base_before", "candidate", "base_after"],
        "shape_k_n": [args.size_k, args.size_n],
        "grid_x": args.grid_x,
        "rows": list(rows),
        "max_regression_percent": args.max_regression_percent,
        "max_sm_clock_spread_mhz": args.max_sm_clock_spread_mhz,
        "arms": compact_arms,
        "checkpoint_tensors": checkpoint_tensors[0],
        "timings": timings,
        "comparisons": comparisons,
        "prior_exact_checkpoint_evidence": prior_exact_evidence(
            args.prior_exact_result,
            args.size_k,
            args.size_n,
        ),
        "physical_gpu_uuids": sorted(uuids),
        "validation_failures": failures,
        "valid": not failures,
    }
    write_json(args.output.resolve(), receipt)
    print(json.dumps({"output": str(args.output.resolve()), "valid": not failures}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
