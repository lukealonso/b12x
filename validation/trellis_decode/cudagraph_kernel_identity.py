#!/usr/bin/env python3
"""Correlate vLLM reachability records with CUDA graph DOT kernel nodes.

The serving diagnostic emits two bounded record streams during model load:

* ``B12X_TRELLIS_REACHABILITY`` inventories projection planning and the route
  selected during eager warmup and CUDA graph capture.
* ``B12X_CUDAGRAPH_KERNEL_DUMP`` binds a CUDA graph DOT file to its graph owner
  and batch descriptor.

This tool verifies the copied DOT artifacts, extracts exact CUDA kernel names
and launch geometry, and proves that each bound K6/MCG projection contributes
one cooperative ``K6McgSmallMKernel`` node to every applicable graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from b12x._lib.cuda_graph_dot import parse_cuda_graph_dot


_REPORT_SCHEMA = "b12x.trellis.cudagraph_kernel_identity.v2"
_REACHABILITY_MARKER = "B12X_TRELLIS_REACHABILITY "
_DUMP_MARKER = "B12X_CUDAGRAPH_KERNEL_DUMP "
_K6_MCG_SYMBOL_FRAGMENT = "K6McgSmallMKernel"
_ROTATION_SYMBOL_FRAGMENTS = ("hadamard", "h128", "rotate", "rotation")
_FUSED_RUNTIME_TO_PLAN_ROUTE = {
    "b12x_k6_mcg_small_m": "b12x_k6_mcg_small_m_planned",
    "b12x_k6_mcg_small_m_bf16": "b12x_k6_mcg_small_m_bf16_planned",
}

_DESCRIPTOR_M_RE = re.compile(r"(?:^|, )num_tokens=(?P<m>\d+)(?:,|\))")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marker_records(lines: Iterable[str], marker: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1]
        try:
            record = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"service log line {line_number}: invalid {marker.strip()} JSON"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"service log line {line_number}: {marker.strip()} is not an object"
            )
        records.append(record)
    return records


def _channel_role(channel_id: str) -> str | None:
    if ":target:" in channel_id:
        return "target_or_shared"
    if ":draft:" in channel_id:
        return "mtp_or_draft"
    return None


def _descriptor_m(descriptor: str) -> int | None:
    match = _DESCRIPTOR_M_RE.search(descriptor)
    return int(match.group("m")) if match is not None else None


def _string_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def _runtime_route_summary(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, int, bool]] = Counter()
    for record in records:
        if record.get("phase") != "runtime":
            continue
        counts[
            (
                str(record.get("selected_route")),
                str(record.get("model_role")),
                int(record.get("m", -1)),
                bool(record.get("cuda_graph_capture")),
            )
        ] += 1
    return [
        {
            "selected_route": key[0],
            "model_role": key[1],
            "m": key[2],
            "cuda_graph_capture": key[3],
            "count": count,
        }
        for key, count in sorted(counts.items())
    ]


def _projection_inventory(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "model_role",
        "projection",
        "shard",
        "k",
        "n",
        "e",
        "trellis_bits",
        "trellis_codebook",
        "weight_layout",
        "model_input_dtype",
        "prepared_compute_dtype",
        "kernel_input_dtype",
        "route_output_dtype",
        "b12x_supported",
        "bound_k6_mcg_small_m_launch",
        "selected_route",
        "reason",
        "launch",
    )
    inventory = [
        {field: record.get(field) for field in fields}
        for record in records
        if record.get("phase") == "plan"
    ]
    return sorted(
        inventory,
        key=lambda row: (
            str(row.get("model_role")),
            str(row.get("projection")),
            str(row.get("shard")),
        ),
    )


def analyze_receipt(
    receipt_dir: Path,
    *,
    service_log: Path | None = None,
    graphs_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed exact-kernel accounting report."""
    receipt_dir = receipt_dir.resolve()
    service_log = (service_log or receipt_dir / "service.log").resolve()
    graphs_dir = (graphs_dir or receipt_dir / "graphs").resolve()
    errors: list[str] = []

    if not service_log.is_file():
        raise FileNotFoundError(f"service log not found: {service_log}")
    if not graphs_dir.is_dir():
        raise FileNotFoundError(f"graph directory not found: {graphs_dir}")

    log_lines = service_log.read_text(encoding="utf-8").splitlines()
    reachability_records = _marker_records(log_lines, _REACHABILITY_MARKER)
    dump_records = _marker_records(log_lines, _DUMP_MARKER)
    if not reachability_records:
        errors.append("no reachability records found")
    if not dump_records:
        errors.append("no CUDA graph dump records found")

    dump_record_by_name: dict[str, dict[str, Any]] = {}
    for record in dump_records:
        name = Path(str(record.get("path", ""))).name
        if not name:
            errors.append("CUDA graph dump record has no path")
            continue
        if name in dump_record_by_name:
            errors.append(f"duplicate CUDA graph dump record: {name}")
            continue
        dump_record_by_name[name] = record

    dot_paths = sorted(graphs_dir.glob("*.dot"))
    dot_names = {path.name for path in dot_paths}
    record_names = set(dump_record_by_name)
    for name in sorted(record_names - dot_names):
        errors.append(f"recorded CUDA graph dump is missing: {name}")
    for name in sorted(dot_names - record_names):
        errors.append(f"unrecorded CUDA graph dump is present: {name}")

    expected_capture_rows: defaultdict[tuple[str, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    active_fused_runtime_routes: set[str] = set()
    for record in reachability_records:
        selected_route = str(record.get("selected_route"))
        if (
            record.get("phase") == "runtime"
            and selected_route in _FUSED_RUNTIME_TO_PLAN_ROUTE
            and record.get("cuda_graph_capture") is True
        ):
            active_fused_runtime_routes.add(selected_route)
            expected_capture_rows[
                (str(record.get("model_role")), int(record.get("m", -1)))
            ].append(record)

    # A serving diagnostic may prepare both FP16 and BF16 forms of the same
    # projection while only one dtype is active in captured decode. Count only
    # the plan family paired to an observed fused runtime route. Otherwise the
    # inactive fallback plan double-counts projections even though the DOT is
    # correct (one node per executed shard).
    active_plan_routes = {
        _FUSED_RUNTIME_TO_PLAN_ROUTE[route] for route in active_fused_runtime_routes
    }
    plan_bound_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[tuple[str, str]] = Counter()
    for record in reachability_records:
        if record.get("phase") != "plan":
            continue
        role = str(record.get("model_role"))
        if (
            record.get("bound_k6_mcg_small_m_launch") is True
            and record.get("selected_route") in active_plan_routes
        ):
            plan_bound_counts[role] += 1
        elif record.get("bound_k6_mcg_small_m_launch") is not True:
            fallback_reason_counts[(role, str(record.get("reason")))] += 1

    if expected_capture_rows and not active_plan_routes:
        errors.append("fused capture routes have no recognized plan family")

    graph_reports: list[dict[str, Any]] = []
    observed_roles: set[str] = set()
    for dot_path in dot_paths:
        record = dump_record_by_name.get(dot_path.name)
        if record is None:
            continue
        graph_errors: list[str] = []
        actual_sha256 = _sha256(dot_path)
        actual_bytes = dot_path.stat().st_size
        if actual_sha256 != record.get("sha256"):
            graph_errors.append("DOT SHA-256 does not match dump record")
        if actual_bytes != record.get("bytes"):
            graph_errors.append("DOT byte count does not match dump record")

        try:
            kernel_nodes = parse_cuda_graph_dot(dot_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        k6_nodes = [
            node for node in kernel_nodes if _K6_MCG_SYMBOL_FRAGMENT in node.symbol
        ]
        rotation_nodes = [
            node
            for node in kernel_nodes
            if any(
                fragment in node.symbol.lower()
                for fragment in _ROTATION_SYMBOL_FRAGMENTS
            )
        ]
        exllamav3_nodes = [
            node for node in kernel_nodes if "exl3_gemm_kernel" in node.symbol
        ]

        channel_id = str(record.get("channel_id"))
        role = _channel_role(channel_id)
        descriptor = str(record.get("batch_descriptor"))
        m = _descriptor_m(descriptor)
        if role is None:
            graph_errors.append(f"unrecognized graph owner: {channel_id}")
        else:
            observed_roles.add(role)
        if m is None:
            graph_errors.append(f"batch descriptor has no num_tokens: {descriptor}")

        expected_rows = (
            expected_capture_rows.get((role, m), [])
            if role is not None and m is not None
            else []
        )
        expected_grids: Counter[int] = Counter()
        for row in expected_rows:
            launch = row.get("launch")
            grid_x = (
                launch.get("launch_grid_x")
                if isinstance(launch, dict)
                else None
            )
            if (
                not isinstance(grid_x, int)
                or isinstance(grid_x, bool)
                or grid_x <= 0
            ):
                graph_errors.append(
                    "runtime record has invalid launch.launch_grid_x: "
                    f"projection={row.get('projection')!r}, value={grid_x!r}"
                )
                continue
            expected_grids[grid_x] += 1
        observed_grids = Counter(node.grid_x for node in k6_nodes)
        expected_runtime_count = len(expected_rows)
        expected_plan_count = plan_bound_counts.get(role, 0) if role else 0

        checks = {
            "dump_sha256_matches": actual_sha256 == record.get("sha256"),
            "dump_bytes_match": actual_bytes == record.get("bytes"),
            "exact_k6_mcg_symbol_present": bool(k6_nodes),
            "all_k6_mcg_nodes_cooperative": bool(k6_nodes)
            and all(node.cooperative for node in k6_nodes),
            "one_k6_node_per_bound_projection": len(k6_nodes) == expected_plan_count,
            "runtime_capture_route_count_matches": len(k6_nodes)
            == expected_runtime_count,
            "launch_grid_multiset_matches_runtime": observed_grids == expected_grids,
            "no_separate_rotation_kernel_symbol": not rotation_nodes,
        }
        if not all(checks.values()):
            graph_errors.append("one or more exact-kernel checks failed")

        graph_reports.append(
            {
                "path": str(dot_path),
                "sha256": actual_sha256,
                "bytes": actual_bytes,
                "channel_id": channel_id,
                "batch_descriptor": descriptor,
                "model_role": role,
                "m": m,
                "kernel_node_count": len(kernel_nodes),
                "k6_mcg_node_count": len(k6_nodes),
                "expected_bound_projection_count": expected_plan_count,
                "expected_runtime_capture_route_count": expected_runtime_count,
                "k6_mcg_grid_x_counts": _string_counter(observed_grids),
                "expected_grid_x_counts": _string_counter(expected_grids),
                "k6_mcg_block_x_counts": _string_counter(
                    Counter(node.block_x for node in k6_nodes)
                ),
                "k6_mcg_dynamic_smem_bytes_counts": _string_counter(
                    Counter(node.dynamic_smem_bytes for node in k6_nodes)
                ),
                "k6_mcg_noncooperative_nodes": [
                    asdict(node) for node in k6_nodes if not node.cooperative
                ],
                "exllamav3_gemm_node_count": len(exllamav3_nodes),
                "rotation_like_kernel_nodes": [asdict(node) for node in rotation_nodes],
                "checks": checks,
                "errors": graph_errors,
                "status": "pass" if not graph_errors else "fail",
            }
        )

    for role in sorted(plan_bound_counts):
        if role not in observed_roles:
            errors.append(f"no CUDA graph dump covers bound role {role}")
    if not graph_reports:
        errors.append("no CUDA graph DOT files were analyzed")

    failed_graphs = [
        graph["path"] for graph in graph_reports if graph["status"] != "pass"
    ]
    if failed_graphs:
        errors.append(f"{len(failed_graphs)} CUDA graph checks failed")

    channel_counts = Counter(str(record.get("channel_id")) for record in dump_records)
    report = {
        "schema": _REPORT_SCHEMA,
        "status": "pass" if not errors else "fail",
        "receipt_dir": str(receipt_dir),
        "inputs": {
            "service_log": str(service_log),
            "service_log_sha256": _sha256(service_log),
            "graphs_dir": str(graphs_dir),
        },
        "summary": {
            "reachability_record_count": len(reachability_records),
            "graph_dump_record_count": len(dump_records),
            "dot_file_count": len(dot_paths),
            "graph_checks_passed": len(graph_reports) - len(failed_graphs),
            "graph_checks_failed": len(failed_graphs),
            "bound_projection_counts": _string_counter(plan_bound_counts),
            "active_fused_runtime_routes": sorted(active_fused_runtime_routes),
            "active_fused_plan_routes": sorted(active_plan_routes),
            "channel_counts": _string_counter(channel_counts),
            "all_exact_k6_mcg_symbols_present": bool(graph_reports)
            and all(
                graph["checks"]["exact_k6_mcg_symbol_present"]
                for graph in graph_reports
            ),
            "all_k6_mcg_nodes_cooperative": bool(graph_reports)
            and all(
                graph["checks"]["all_k6_mcg_nodes_cooperative"]
                for graph in graph_reports
            ),
            "one_cooperative_launch_per_bound_projection": bool(graph_reports)
            and all(
                graph["checks"]["one_k6_node_per_bound_projection"]
                and graph["checks"]["runtime_capture_route_count_matches"]
                and graph["checks"]["launch_grid_multiset_matches_runtime"]
                for graph in graph_reports
            ),
            "no_separate_rotation_kernel_symbols": bool(graph_reports)
            and all(
                graph["checks"]["no_separate_rotation_kernel_symbol"]
                for graph in graph_reports
            ),
        },
        "fallback_reason_counts": [
            {"model_role": key[0], "reason": key[1], "count": count}
            for key, count in sorted(fallback_reason_counts.items())
        ],
        "runtime_route_counts": _runtime_route_summary(reachability_records),
        "projection_inventory": _projection_inventory(reachability_records),
        "graphs": graph_reports,
        "errors": errors,
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "receipt_dir",
        type=Path,
        help="receipt directory containing service.log and graphs/*.dot",
    )
    parser.add_argument(
        "--service-log",
        type=Path,
        default=None,
        help="override the service log path",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=None,
        help="override the CUDA graph DOT directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the JSON report to this path instead of stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = analyze_receipt(
        args.receipt_dir,
        service_log=args.service_log,
        graphs_dir=args.graphs_dir,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(
            f"{report['status']}: {len(report['graphs'])} graph(s); "
            f"report={args.output}",
            file=sys.stderr,
        )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
