#!/usr/bin/env python3
"""Qualify native K6/MCG small-row decode on exact EXL3 checkpoint tensors.

This benchmark deliberately exercises the three production-relevant GPU paths
through their real public/runtime boundaries:

* the b12x-owned cooperative K6/MCG launch bound by ``prepare_weight``;
* the generic b12x dense Trellis scheduler with the bound launch removed; and
* the exact served ``exllamav3_ext.exl3_gemm`` route, including its explicit
  BF16/FP16 boundary copies when the model activation contract is BF16.

Timing is gated on manifest binding, an independently expressed H128 + Torch
matmul oracle, CUDA-graph identity, poison/mutation checks, stable addresses,
and allocator-stable replay.  The JSON result retains every CUDA-event sample
and defines latency ratios explicitly: a ratio above one means the fused b12x
path is faster than the named comparator.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
import contextlib
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

import torch
from safetensors import safe_open

from b12x._lib.cuda_graph_dot import parse_cuda_graph_dot
from b12x._lib.runtime_control import (
    freeze_kernel_resolution,
    unfreeze_kernel_resolution,
)
from b12x.gemm import trellis_linear
from b12x.moe._shared.kernels.w4a16.host import packed_gemm_scratch_elements


_SCHEMA = "b12x.trellis.k6_mcg_checkpoint_benchmark.v3"
_DEFAULT_PREFIX = "model.language_model.layers.0.mlp.down_proj"
_TENSOR_SUFFIXES = ("trellis", "suh", "svh", "mcg")
_ROUTE_NAMES = ("fused_b12x", "generic_b12x", "served_exllamav3")
_K6_SYMBOL_FRAGMENT = "K6McgSmallMKernel"
_ROTATION_SYMBOL_FRAGMENTS = ("hadamard", "h128", "rotate", "rotation")
_ALLOCATOR_COUNTERS = (
    "allocation.all.allocated",
    "allocation.all.freed",
    "segment.all.allocated",
    "segment.all.freed",
    "num_alloc_retries",
    "num_ooms",
)


class QualificationError(RuntimeError):
    """Raised when a mandatory pre-timing qualification gate fails."""


@dataclass(frozen=True)
class CheckpointPayload:
    prefix: str
    shard_name: str
    trellis: torch.Tensor
    suh: torch.Tensor
    svh: torch.Tensor
    mcg: torch.Tensor
    metadata: dict[str, Any]


@dataclass
class RouteState:
    name: str
    run: Callable[[], None]
    output: torch.Tensor
    poison_tensors: dict[str, torch.Tensor]
    stable_tensors: dict[str, torch.Tensor]
    graph: torch.cuda.CUDAGraph | None = None
    dot_path: Path | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    host = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    digest.update(memoryview(host.numpy()))
    return digest.hexdigest()


def _parse_sha256sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+[*]?(.+)", line)
        if match is None:
            raise ValueError(f"{path}:{line_number}: malformed SHA256SUMS row")
        name = match.group(2)
        if name in entries:
            raise ValueError(f"{path}:{line_number}: duplicate checksum for {name}")
        entries[name] = match.group(1).lower()
    return entries


def _resolve_checkpoint_binding(
    model_dir: Path,
    prefix: str,
) -> tuple[Path, dict[str, Any]]:
    index_path = model_dir / "model.safetensors.index.json"
    sums_path = model_dir / "SHA256SUMS"
    if not index_path.is_file():
        raise FileNotFoundError(f"checkpoint index not found: {index_path}")
    if not sums_path.is_file():
        raise FileNotFoundError(f"checkpoint SHA256SUMS not found: {sums_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} has no object-valued weight_map")
    keys = {suffix: f"{prefix}.{suffix}" for suffix in _TENSOR_SUFFIXES}
    missing = [key for key in keys.values() if key not in weight_map]
    if missing:
        raise KeyError(f"checkpoint index is missing keys: {missing}")
    shards = {str(weight_map[key]) for key in keys.values()}
    if len(shards) != 1:
        raise ValueError(f"projection tensors span multiple shards: {sorted(shards)}")
    shard_name = next(iter(shards))
    shard_path = model_dir / shard_name
    if not shard_path.is_file():
        raise FileNotFoundError(f"checkpoint shard not found: {shard_path}")
    checksums = _parse_sha256sums(sums_path)
    expected_shard_sha = checksums.get(shard_name)
    if expected_shard_sha is None:
        raise KeyError(f"{sums_path} has no checksum for {shard_name}")
    return shard_path, {
        "model_dir": str(model_dir.resolve()),
        "prefix": prefix,
        "keys": keys,
        "index_path": str(index_path.resolve()),
        "index_sha256": _sha256_file(index_path),
        "sha256sums_path": str(sums_path.resolve()),
        "sha256sums_sha256": _sha256_file(sums_path),
        "shard_name": shard_name,
        "shard_path": str(shard_path.resolve()),
        "shard_bytes": shard_path.stat().st_size,
        "expected_shard_sha256": expected_shard_sha,
    }


def _load_checkpoint_payload(
    model_dir: Path,
    prefix: str,
    *,
    verify_shard_sha: bool,
) -> CheckpointPayload:
    shard_path, metadata = _resolve_checkpoint_binding(model_dir, prefix)
    actual_shard_sha = None
    if verify_shard_sha:
        actual_shard_sha = _sha256_file(shard_path)
        if actual_shard_sha != metadata["expected_shard_sha256"]:
            raise QualificationError(
                "checkpoint shard SHA-256 does not match SHA256SUMS: "
                f"expected={metadata['expected_shard_sha256']}, "
                f"actual={actual_shard_sha}"
            )
    metadata["actual_shard_sha256"] = actual_shard_sha
    metadata["shard_sha256_verified"] = bool(verify_shard_sha)

    keys = metadata["keys"]
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        tensors = {
            suffix: handle.get_tensor(keys[suffix]) for suffix in _TENSOR_SUFFIXES
        }
    trellis = tensors["trellis"]
    suh = tensors["suh"]
    svh = tensors["svh"]
    mcg = tensors["mcg"]
    if trellis.dtype != torch.int16 or trellis.ndim != 3:
        raise QualificationError(
            "exact K6 payload must be rank-3 int16, got "
            f"shape={tuple(trellis.shape)} dtype={trellis.dtype}"
        )
    if int(trellis.shape[-1]) != 16 * 6:
        raise QualificationError(
            f"exact payload is not six-bit Trellis storage: {tuple(trellis.shape)}"
        )
    size_k = int(trellis.shape[0]) * 16
    size_n = int(trellis.shape[1]) * 16
    if suh.dtype != torch.float16 or tuple(suh.shape) != (size_k,):
        raise QualificationError(
            f"suh contract mismatch: shape={tuple(suh.shape)} dtype={suh.dtype}"
        )
    if svh.dtype != torch.float16 or tuple(svh.shape) != (size_n,):
        raise QualificationError(
            f"svh contract mismatch: shape={tuple(svh.shape)} dtype={svh.dtype}"
        )
    if mcg.dtype not in (torch.int32, torch.uint32) or mcg.numel() != 1:
        raise QualificationError(
            f"mcg marker must be scalar int32/uint32, got {mcg.dtype} {tuple(mcg.shape)}"
        )
    tensor_metadata = {}
    for suffix, tensor in tensors.items():
        tensor_metadata[suffix] = {
            "key": keys[suffix],
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
            "bytes": tensor.numel() * tensor.element_size(),
            "sha256": _sha256_tensor(tensor),
        }
    metadata.update(
        {
            "size_k": size_k,
            "size_n": size_n,
            "trellis_bits": 6,
            "codebook": "mcg",
            "mcg_marker_unsigned": int(mcg.item()) & 0xFFFFFFFF,
            "tensors": tensor_metadata,
        }
    )
    return CheckpointPayload(
        prefix=prefix,
        shard_name=metadata["shard_name"],
        trellis=trellis,
        suh=suh,
        svh=svh,
        mcg=mcg,
        metadata=metadata,
    )


def _parse_rows(value: str) -> tuple[int, ...]:
    try:
        rows = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "rows must be comma-separated integers"
        ) from exc
    if not rows or len(set(rows)) != len(rows):
        raise argparse.ArgumentTypeError("rows must be a non-empty unique list")
    if any(row < 1 or row > 16 for row in rows):
        raise argparse.ArgumentTypeError("fused K6 rows must lie in [1, 16]")
    return rows


def _parse_params_dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    if normalized == "fp16":
        return torch.float16
    if normalized == "bf16":
        return torch.bfloat16
    raise argparse.ArgumentTypeError("params dtype must be fp16 or bf16")


def _dot_report(path: Path, route: str) -> dict[str, Any]:
    nodes = parse_cuda_graph_dot(path)
    k6_nodes = [node for node in nodes if _K6_SYMBOL_FRAGMENT in node.symbol]
    separate_rotations = [
        node
        for node in nodes
        if any(
            fragment in node.symbol.lower() for fragment in _ROTATION_SYMBOL_FRAGMENTS
        )
    ]
    errors: list[str] = []
    if not nodes:
        errors.append("CUDA graph DOT contains no kernel nodes")
    if route == "fused_b12x":
        if len(k6_nodes) != 1:
            errors.append(f"expected one K6McgSmallMKernel node, found {len(k6_nodes)}")
        elif not k6_nodes[0].cooperative:
            errors.append("K6McgSmallMKernel node is not cooperative")
        if separate_rotations:
            errors.append(
                "fused graph contains separate rotation-like symbols: "
                + ", ".join(node.symbol for node in separate_rotations)
            )
    elif k6_nodes:
        errors.append(f"{route} graph unexpectedly contains the fused b12x kernel")
    symbol_counts = Counter(node.symbol for node in nodes)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "kernel_count": len(nodes),
        "k6_mcg_small_m_count": len(k6_nodes),
        "separate_rotation_like_count": len(separate_rotations),
        "symbol_counts": dict(sorted(symbol_counts.items())),
        "nodes": [asdict(node) for node in nodes],
        "errors": errors,
        "pass": not errors,
    }


def _normalized_h128(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] % 128:
        raise ValueError(f"H128 width must be divisible by 128, got {values.shape[-1]}")
    blocks = values.float().reshape(*values.shape[:-1], -1, 128)
    stride = 1
    while stride < 128:
        stages = blocks.reshape(*blocks.shape[:-1], -1, 2, stride)
        lower = stages[..., 0, :]
        upper = stages[..., 1, :]
        blocks = torch.stack((lower + upper, lower - upper), dim=-2).flatten(-3, -1)
        stride *= 2
    return blocks.reshape_as(values) * (1.0 / math.sqrt(128.0))


def _independent_oracle(
    source: torch.Tensor,
    raw_weight: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
) -> torch.Tensor:
    element_dtype = source.dtype
    if element_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"oracle source must be fp16 or bf16, got {element_dtype}")
    rotated = _normalized_h128(source.float() * suh.float()).to(element_dtype)
    # Express the GEMM independently from both custom kernels.  FP32 operands
    # and disabled TF32 make the result a high-precision semantic oracle. The
    # explicit element rounding matches the fused MMA and rotation boundaries;
    # BF16 also rounds the reconstructed FP16 weight to its MMA operand dtype.
    gemm_weight = raw_weight.to(element_dtype)
    gemm = torch.matmul(rotated.float(), gemm_weight.float()).to(element_dtype)
    return (_normalized_h128(gemm.float()) * svh.float()).to(element_dtype)


def _tensor_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    topk: int,
) -> dict[str, Any]:
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {reference.shape}")
    if actual.dtype != reference.dtype:
        raise ValueError(f"dtype mismatch: {actual.dtype} != {reference.dtype}")
    actual_f = actual.float()
    reference_f = reference.float()
    delta = actual_f - reference_f
    actual_rows = actual_f.reshape(-1, actual_f.shape[-1])
    reference_rows = reference_f.reshape(-1, reference_f.shape[-1])
    cosine = torch.nn.functional.cosine_similarity(actual_rows, reference_rows, dim=-1)
    k = min(int(topk), int(actual_f.shape[-1]))
    actual_topk = torch.topk(actual_f, k=k, dim=-1).indices
    reference_topk = torch.topk(reference_f, k=k, dim=-1).indices
    order_exact_rows = torch.all(actual_topk == reference_topk, dim=-1)
    set_exact_rows = torch.all(
        torch.sort(actual_topk, dim=-1).values
        == torch.sort(reference_topk, dim=-1).values,
        dim=-1,
    )
    overlaps = []
    for actual_row, reference_row in zip(actual_topk, reference_topk, strict=True):
        overlap = torch.isin(actual_row, reference_row).sum()
        overlaps.append(float(overlap.item()) / k)
    reference_norm = torch.linalg.vector_norm(reference_f).clamp_min(1.0e-12)
    if k < int(reference_f.shape[-1]):
        boundary_values = torch.topk(reference_f, k=k + 1, dim=-1).values
        boundary_margins = boundary_values[..., k - 1] - boundary_values[..., k]
    else:
        boundary_margins = torch.full(
            reference_f.shape[:-1], float("inf"), device=reference_f.device
        )
    global_max_abs = float(delta.abs().max().item())
    row_max_abs = delta.abs().reshape(-1, delta.shape[-1]).max(dim=-1).values
    flat_boundary_margins = boundary_margins.reshape(-1)
    flat_set_exact_rows = set_exact_rows.reshape(-1)
    boundary_ambiguous_rows = flat_boundary_margins <= row_max_abs
    membership_mismatch_rows = ~flat_set_exact_rows
    unambiguous_membership_mismatch_rows = (
        membership_mismatch_rows & ~boundary_ambiguous_rows
    )
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "actual_finite": bool(torch.isfinite(actual).all().item()),
        "actual_nonzero": bool(torch.count_nonzero(actual).item()),
        "reference_finite": bool(torch.isfinite(reference).all().item()),
        "reference_nonzero": bool(torch.count_nonzero(reference).item()),
        "relative_l2": float(
            torch.linalg.vector_norm(delta).item() / reference_norm.item()
        ),
        "max_abs": global_max_abs,
        "mean_abs": float(delta.abs().mean().item()),
        "cosine_min": float(cosine.min().item()),
        "cosine_mean": float(cosine.mean().item()),
        "topk": k,
        "topk_order_exact_rows": int(order_exact_rows.sum().item()),
        "topk_set_exact_rows": int(set_exact_rows.sum().item()),
        "topk_total_rows": int(set_exact_rows.numel()),
        "topk_all_rows_order_exact": bool(order_exact_rows.all().item()),
        "topk_all_rows_set_exact": bool(set_exact_rows.all().item()),
        "topk_overlap_min": min(overlaps),
        "topk_overlap_mean": statistics.fmean(overlaps),
        "topk_boundary_margin_min": float(boundary_margins.min().item()),
        "topk_boundary_ambiguous_rows_at_max_abs": int(
            boundary_ambiguous_rows.sum().item()
        ),
        "topk_membership_mismatch_rows": int(membership_mismatch_rows.sum().item()),
        "topk_ambiguity_qualified_mismatch_rows": int(
            (membership_mismatch_rows & boundary_ambiguous_rows).sum().item()
        ),
        "topk_unambiguous_membership_mismatch_rows": int(
            unambiguous_membership_mismatch_rows.sum().item()
        ),
    }


def _metrics_pass(
    metrics: dict[str, Any],
    *,
    min_cosine: float,
    max_relative_l2: float,
    max_abs: float,
    require_exact_topk: bool,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for key in (
        "actual_finite",
        "actual_nonzero",
        "reference_finite",
        "reference_nonzero",
    ):
        if not metrics[key]:
            errors.append(f"{key}=false")
    if metrics["cosine_min"] < min_cosine:
        errors.append(f"cosine_min={metrics['cosine_min']:.9g} < {min_cosine:.9g}")
    if metrics["relative_l2"] > max_relative_l2:
        errors.append(
            f"relative_l2={metrics['relative_l2']:.9g} > {max_relative_l2:.9g}"
        )
    if metrics["max_abs"] > max_abs:
        errors.append(f"max_abs={metrics['max_abs']:.9g} > {max_abs:.9g}")
    if require_exact_topk and metrics["topk_unambiguous_membership_mismatch_rows"] > 0:
        errors.append(
            "top-k membership differs beyond the measured numerical boundary "
            f"for {metrics['topk_unambiguous_membership_mismatch_rows']} rows"
        )
    return not errors, errors


def _allocator_snapshot(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    result = {name: int(stats.get(name, 0)) for name in _ALLOCATOR_COUNTERS}
    result.update(
        {
            "memory_allocated": int(torch.cuda.memory_allocated(device)),
            "memory_reserved": int(torch.cuda.memory_reserved(device)),
            "active_bytes.all.current": int(stats.get("active_bytes.all.current", 0)),
            "allocated_bytes.all.current": int(
                stats.get("allocated_bytes.all.current", 0)
            ),
            "reserved_bytes.all.current": int(
                stats.get("reserved_bytes.all.current", 0)
            ),
        }
    )
    return result


def _snapshot_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in before}


def _tensor_contract(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": tensor.is_contiguous(),
        "data_ptr": int(tensor.data_ptr()),
        "numel": tensor.numel(),
        "bytes": tensor.numel() * tensor.element_size(),
    }


def _stable_contract(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {name: _tensor_contract(tensor) for name, tensor in sorted(tensors.items())}


def _assert_same_addresses(
    before: dict[str, dict[str, Any]],
    tensors: dict[str, torch.Tensor],
) -> None:
    after = _stable_contract(tensors)
    changed = {
        name: (before[name]["data_ptr"], after[name]["data_ptr"])
        for name in before
        if before[name]["data_ptr"] != after[name]["data_ptr"]
    }
    if changed:
        raise QualificationError(f"stable tensor addresses changed: {changed}")


def _run_command(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return {"command": list(command), "available": False, "error": str(exc)}
    return {
        "command": list(command),
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _gpu_snapshot() -> dict[str, Any]:
    fields = (
        "timestamp",
        "name",
        "uuid",
        "pci.bus_id",
        "compute_cap",
        "memory.total",
        "memory.used",
        "power.draw",
        "power.limit",
        "pstate",
        "clocks.current.graphics",
        "clocks.current.memory",
        "clocks.current.sm",
        "compute_mode",
        "clocks_throttle_reasons.active",
    )
    return _run_command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    )


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment_metadata(
    args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "captured_at": _utc_now(),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "packages": {
            name: _package_version(name)
            for name in (
                "b12x",
                "cuda-python",
                "nvidia-cutlass-dsl",
                "safetensors",
            )
        },
        "device": {
            "index": device.index,
            "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "")),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(properties.total_memory),
            "multi_processor_count": int(properties.multi_processor_count),
            "shared_memory_per_block_optin": int(
                getattr(properties, "shared_memory_per_block_optin", 0)
            ),
        },
        "declared_identity": {
            "source_revision": args.source_revision,
            "integration_tree": args.integration_tree,
            "image": args.image,
            "image_id": args.image_id,
        },
        "architecture_environment": {
            name: os.environ.get(name)
            for name in (
                "CUTE_DSL_ARCH",
                "TORCH_CUDA_ARCH_LIST",
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "XDG_CACHE_HOME",
                "B12X_CUTE_COMPILE_CACHE_DIR",
                "B12X_COMPILE_CACHE_DIR",
                "VLLM_EXL3_EXT_PATH",
            )
        },
        "toolchains": {
            "nvcc": _run_command(["nvcc", "--version"]),
            "ptxas": _run_command(["ptxas", "--version"]),
        },
    }


def _import_exllamav3_ext(path: Path | None):
    if path is not None:
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    extension = importlib.import_module("exllamav3_ext")
    for name in ("exl3_gemm", "reconstruct"):
        if not callable(getattr(extension, name, None)):
            raise QualificationError(f"exllamav3_ext does not export callable {name}")
    return extension


def _capture_route(
    route: RouteState,
    *,
    graph_dir: Path,
    rows: int,
    device: torch.device,
) -> dict[str, Any]:
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    graph.enable_debug_mode()
    # Graph construction may create framework-owned RNG/capture bookkeeping.
    # Place the measurement boundary after that setup so the delta describes
    # the captured route call, whose tensors and workspaces are all preplanned.
    torch.cuda.synchronize(device)
    before = _allocator_snapshot(device)
    started_ns = time.perf_counter_ns()
    with torch.cuda.graph(graph):
        route.run()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
    after_capture = _allocator_snapshot(device)
    dot_path = graph_dir / f"m{rows:02d}-{route.name}.dot"
    graph.debug_dump(str(dot_path))
    if not dot_path.is_file() or dot_path.stat().st_size == 0:
        raise QualificationError(f"CUDA graph debug dump was not created: {dot_path}")
    graph.instantiate()
    route.graph = graph
    route.dot_path = dot_path
    return {
        "elapsed_wall_ms": elapsed_ms,
        "allocator_before": before,
        "allocator_after_capture": after_capture,
        "allocator_delta": _snapshot_delta(before, after_capture),
        "allocator_stable": before == after_capture,
        "dot": _dot_report(dot_path, route.name),
    }


def _poison_route(route: RouteState) -> dict[str, int]:
    poisoned = {}
    for name, tensor in route.poison_tensors.items():
        if tensor.is_floating_point():
            tensor.fill_(float("nan"))
        else:
            tensor.fill_(0x5A)
        poisoned[name] = tensor.numel()
    return poisoned


def _poison_report(route: RouteState) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, tensor in route.poison_tensors.items():
        if tensor.is_floating_point():
            finite = torch.isfinite(tensor)
            report[name] = {
                "finite_elements": int(finite.sum().item()),
                "total_elements": tensor.numel(),
                "any_overwritten": bool(finite.any().item()),
                "all_overwritten": bool(finite.all().item()),
            }
        else:
            report[name] = {"total_elements": tensor.numel()}
    return report


def _replay_allocation_check(
    route: RouteState,
    *,
    replays: int,
    device: torch.device,
) -> dict[str, Any]:
    if route.graph is None:
        raise RuntimeError(f"route {route.name} has no captured graph")
    _poison_route(route)
    torch.cuda.synchronize(device)
    before = _allocator_snapshot(device)
    for _ in range(replays):
        route.graph.replay()
    torch.cuda.synchronize(device)
    after = _allocator_snapshot(device)
    return {
        "replays": replays,
        "allocator_before": before,
        "allocator_after": after,
        "allocator_delta": _snapshot_delta(before, after),
        "allocator_stable": before == after,
        "poison": _poison_report(route),
    }


def _time_eager_call(run: Callable[[], None], device: torch.device) -> dict[str, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter_ns()
    start.record()
    run()
    end.record()
    end.synchronize()
    return {
        "cuda_ms": float(start.elapsed_time(end)),
        "wall_ms": (time.perf_counter_ns() - wall_start) / 1.0e6,
    }


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _sample_statistics(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("timing sample list is empty")
    ordered = sorted(float(value) for value in values)
    median = statistics.median(ordered)
    deviations = sorted(abs(value - median) for value in ordered)
    return {
        "count": len(ordered),
        "min_ms": ordered[0],
        "p10_ms": _percentile(ordered, 0.10),
        "median_ms": median,
        "mean_ms": statistics.fmean(ordered),
        "p90_ms": _percentile(ordered, 0.90),
        "max_ms": ordered[-1],
        "stdev_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "mad_ms": statistics.median(deviations),
    }


def _balanced_orders() -> tuple[tuple[str, ...], ...]:
    return (
        ("fused_b12x", "generic_b12x", "served_exllamav3"),
        ("served_exllamav3", "generic_b12x", "fused_b12x"),
        ("generic_b12x", "fused_b12x", "served_exllamav3"),
        ("generic_b12x", "served_exllamav3", "fused_b12x"),
        ("fused_b12x", "served_exllamav3", "generic_b12x"),
        ("served_exllamav3", "fused_b12x", "generic_b12x"),
    )


def _time_graphs(
    routes: dict[str, RouteState],
    *,
    warmups: int,
    iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    orders = _balanced_orders()
    if iterations % len(orders):
        raise ValueError(
            f"iterations must be a multiple of {len(orders)} to keep route "
            "ordering balanced"
        )
    for iteration in range(warmups):
        order = orders[iteration % len(orders)]
        for name in order:
            graph = routes[name].graph
            if graph is None:
                raise RuntimeError(f"route {name} has no graph")
            graph.replay()
    torch.cuda.synchronize(device)
    active_gpu_snapshot = _gpu_snapshot()

    events: dict[
        str, list[tuple[int, int, tuple[str, ...], torch.cuda.Event, torch.cuda.Event]]
    ] = {name: [] for name in _ROUTE_NAMES}
    for iteration in range(iterations):
        order = orders[iteration % len(orders)]
        for position, name in enumerate(order):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph = routes[name].graph
            if graph is None:
                raise RuntimeError(f"route {name} has no graph")
            graph.replay()
            end.record()
            events[name].append((iteration, position, order, start, end))
    torch.cuda.synchronize(device)
    raw: dict[str, list[dict[str, Any]]] = {}
    summaries = {}
    for name in _ROUTE_NAMES:
        samples = []
        for iteration, position, order, start, end in events[name]:
            samples.append(
                {
                    "round": iteration,
                    "position": position,
                    "order": list(order),
                    "milliseconds": float(start.elapsed_time(end)),
                }
            )
        raw[name] = samples
        summaries[name] = _sample_statistics(
            [sample["milliseconds"] for sample in samples]
        )
    fused = float(summaries["fused_b12x"]["median_ms"])
    generic = float(summaries["generic_b12x"]["median_ms"])
    exllama = float(summaries["served_exllamav3"]["median_ms"])
    return {
        "method": "single-replay CUDA events, all six route orders cycled equally",
        "cache_policy": "warm graph replay; no synthetic L2 flush",
        "warmups": warmups,
        "iterations_per_route": iterations,
        "active_gpu_snapshot": active_gpu_snapshot,
        "raw_samples": raw,
        "summary": summaries,
        "ratios": {
            "generic_over_fused_latency": generic / fused,
            "exllamav3_over_fused_latency": exllama / fused,
            "fused_speedup_percent_vs_generic": (generic / fused - 1.0) * 100.0,
            "fused_speedup_percent_vs_exllamav3": (exllama / fused - 1.0) * 100.0,
            "direction": "ratio > 1 means fused_b12x has lower median latency",
        },
    }


def _bound_fused_scratch_elements(fused_weight: Any) -> int:
    """Return the immutable scratch capacity selected during public planning."""

    small_m_launch = getattr(fused_weight, "k6_mcg_small_m_launch", None)
    required = int(getattr(small_m_launch, "required_scratch_elements", 0))
    if required <= 0:
        raise QualificationError(
            "fused weight has no positive bound K6/MCG scratch contract"
        )
    return required


def _make_routes(
    *,
    rows: int,
    source: torch.Tensor,
    trellis_native: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    fused_weight: Any,
    generic_weight: Any,
    exllamav3_ext: Any,
    device: torch.device,
) -> dict[str, RouteState]:
    size_k = int(source.shape[1])
    size_n = int(svh.numel())
    element_dtype = source.dtype
    if element_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"source must be fp16 or bf16, got {element_dtype}")
    fused_scratch_elements = _bound_fused_scratch_elements(fused_weight)
    sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    fused_output = torch.empty((rows, size_n), dtype=element_dtype, device=device)
    fused_rotated_compute = torch.empty(
        (rows, size_k), dtype=element_dtype, device=device
    )
    fused_c_tmp = torch.empty(
        fused_scratch_elements,
        dtype=torch.float32,
        device=device,
    )

    generic_output = torch.empty((rows, size_n), dtype=element_dtype, device=device)
    generic_gemm = torch.empty((rows, size_n), dtype=element_dtype, device=device)
    generic_input_f16 = torch.empty(
        (rows, size_k), dtype=torch.float16, device=device
    )
    generic_rotated_f16 = torch.empty_like(generic_input_f16)
    generic_rotated_compute = torch.empty(
        (rows, size_k), dtype=element_dtype, device=device
    )
    generic_gemm_output_f16 = torch.empty(
        (rows, size_n), dtype=torch.float16, device=device
    )
    generic_output_f16 = torch.empty_like(generic_gemm_output_f16)
    generic_c_tmp_elements = max(
        packed_gemm_scratch_elements(
            size_n=size_n,
            route_slots=block_size,
            moe_block_size=block_size,
            sms=sms,
        )
        for block_size in (8, 16, 32, 48, 64)
    )
    generic_c_tmp = torch.empty(
        generic_c_tmp_elements, dtype=torch.float32, device=device
    )

    exllama_input_f16 = torch.empty(
        (rows, size_k), dtype=torch.float16, device=device
    )
    exllama_output_f16 = torch.empty(
        (rows, size_n), dtype=torch.float16, device=device
    )
    exllama_output = torch.empty(
        (rows, size_n), dtype=element_dtype, device=device
    )
    exllama_x_had = torch.empty_like(exllama_input_f16)

    def run_fused() -> None:
        result = trellis_linear.run(
            source,
            fused_weight,
            output=fused_output,
            rotated_compute=fused_rotated_compute,
            c_tmp=fused_c_tmp,
        )
        if result.data_ptr() != fused_output.data_ptr():
            raise RuntimeError("fused b12x did not return caller-owned output")

    def run_generic() -> None:
        result = trellis_linear.run(
            source,
            generic_weight,
            output=generic_output,
            gemm_output=generic_gemm,
            input_f16=generic_input_f16 if element_dtype == torch.bfloat16 else None,
            rotated_f16=generic_rotated_f16,
            rotated_compute=(
                generic_rotated_compute
                if element_dtype == torch.bfloat16
                else None
            ),
            gemm_output_f16=(
                generic_gemm_output_f16
                if element_dtype == torch.bfloat16
                else None
            ),
            output_f16=(
                generic_output_f16 if element_dtype == torch.bfloat16 else None
            ),
            c_tmp=generic_c_tmp,
        )
        if result.data_ptr() != generic_output.data_ptr():
            raise RuntimeError("generic b12x did not return caller-owned output")

    def run_exllamav3() -> None:
        if element_dtype == torch.bfloat16:
            exllama_input_f16.copy_(source)
        exllamav3_ext.exl3_gemm(
            source if element_dtype == torch.float16 else exllama_input_f16,
            trellis_native,
            exllama_output
            if element_dtype == torch.float16
            else exllama_output_f16,
            suh,
            exllama_x_had,
            svh,
            -1,
            True,
            False,
            0,
        )
        if element_dtype == torch.bfloat16:
            exllama_output.copy_(exllama_output_f16)

    shared = {
        "source": source,
        "trellis": trellis_native,
        "suh": suh,
        "svh": svh,
    }
    return {
        "fused_b12x": RouteState(
            name="fused_b12x",
            run=run_fused,
            output=fused_output,
            poison_tensors={
                "output": fused_output,
                "rotated_compute": fused_rotated_compute,
                "c_tmp": fused_c_tmp,
            },
            stable_tensors={
                **shared,
                "output": fused_output,
                "rotated_compute": fused_rotated_compute,
                "c_tmp": fused_c_tmp,
                "workspace": fused_weight.workspace,
            },
        ),
        "generic_b12x": RouteState(
            name="generic_b12x",
            run=run_generic,
            output=generic_output,
            poison_tensors={
                "output": generic_output,
                "gemm_output": generic_gemm,
                "input_f16": generic_input_f16,
                "rotated_f16": generic_rotated_f16,
                "rotated_compute": generic_rotated_compute,
                "gemm_output_f16": generic_gemm_output_f16,
                "output_f16": generic_output_f16,
                "c_tmp": generic_c_tmp,
            },
            stable_tensors={
                **shared,
                "output": generic_output,
                "gemm_output": generic_gemm,
                "input_f16": generic_input_f16,
                "rotated_f16": generic_rotated_f16,
                "rotated_compute": generic_rotated_compute,
                "gemm_output_f16": generic_gemm_output_f16,
                "output_f16": generic_output_f16,
                "c_tmp": generic_c_tmp,
                "workspace": generic_weight.workspace,
            },
        ),
        "served_exllamav3": RouteState(
            name="served_exllamav3",
            run=run_exllamav3,
            output=exllama_output,
            poison_tensors={
                "output": exllama_output,
                "x_had": exllama_x_had,
                **(
                    {
                        "input_f16": exllama_input_f16,
                        "output_f16": exllama_output_f16,
                    }
                    if element_dtype == torch.bfloat16
                    else {}
                ),
            },
            stable_tensors={
                **shared,
                "output": exllama_output,
                "x_had": exllama_x_had,
                **(
                    {
                        "input_f16": exllama_input_f16,
                        "output_f16": exllama_output_f16,
                    }
                    if element_dtype == torch.bfloat16
                    else {}
                ),
            },
        ),
    }


def _correctness_scenarios(
    routes: dict[str, RouteState],
    *,
    source: torch.Tensor,
    scenarios: Sequence[torch.Tensor],
    raw_weight: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "scenarios": [],
        "pass": True,
        "errors": [],
        "comparator_warnings": [],
    }
    qualification_pairs = {"fused_vs_oracle"}
    if source.dtype == torch.float16:
        # All three FP16 paths share the same rotation-boundary semantics. In
        # native BF16 mode, the generic and served controls deliberately retain
        # their FP16 rotation boundaries; keep those comparisons visible as
        # warnings while gating the candidate against the independent BF16
        # oracle and deferring model-level behavior to serving qualification.
        qualification_pairs.update({"generic_vs_oracle", "fused_vs_generic"})
    report["qualification_pairs"] = sorted(qualification_pairs)
    for scenario_index, scenario in enumerate(scenarios):
        source.copy_(scenario)
        source_before = source.clone()
        outputs = {}
        route_reports = {}
        for name in _ROUTE_NAMES:
            route = routes[name]
            _poison_route(route)
            if route.graph is None:
                raise RuntimeError(f"route {name} has no graph")
            route.graph.replay()
            torch.cuda.synchronize(device)
            outputs[name] = route.output.clone()
            poison = _poison_report(route)
            output_poison = poison["output"]
            route_reports[name] = {
                "poison": poison,
                "output_fully_overwritten": output_poison["all_overwritten"],
            }
            if not output_poison["all_overwritten"]:
                report["errors"].append(
                    f"scenario {scenario_index} {name}: output poison survived replay"
                )
        if not torch.equal(source, source_before):
            report["errors"].append(
                f"scenario {scenario_index}: a route mutated the source activation"
            )
        oracle = _independent_oracle(source, raw_weight, suh, svh)
        comparisons: dict[str, Any] = {}
        reference_pairs = {
            "fused_vs_oracle": (outputs["fused_b12x"], oracle),
            "generic_vs_oracle": (outputs["generic_b12x"], oracle),
            "exllamav3_vs_oracle": (outputs["served_exllamav3"], oracle),
            "fused_vs_generic": (outputs["fused_b12x"], outputs["generic_b12x"]),
            "fused_vs_exllamav3": (
                outputs["fused_b12x"],
                outputs["served_exllamav3"],
            ),
        }
        for label, (actual, reference) in reference_pairs.items():
            metrics = _tensor_metrics(actual, reference, topk=args.topk)
            passed, errors = _metrics_pass(
                metrics,
                min_cosine=args.min_cosine,
                max_relative_l2=args.max_relative_l2,
                max_abs=args.max_abs,
                require_exact_topk=args.require_exact_topk,
            )
            comparisons[label] = {**metrics, "pass": passed, "errors": errors}
            comparisons[label]["qualification_gate"] = label in qualification_pairs
            destination = (
                report["errors"]
                if label in qualification_pairs
                else report["comparator_warnings"]
            )
            destination.extend(
                f"scenario {scenario_index} {label}: {error}" for error in errors
            )
        report["scenarios"].append(
            {
                "index": scenario_index,
                "source_sha256": _sha256_tensor(source),
                "source_norm": float(torch.linalg.vector_norm(source.float()).item()),
                "routes": route_reports,
                "comparisons": comparisons,
            }
        )
    report["pass"] = not report["errors"]
    return report


def _benchmark_rows(
    rows: int,
    *,
    payload_gpu: dict[str, torch.Tensor],
    fused_weight: Any,
    generic_weight: Any,
    raw_weight: torch.Tensor,
    exllamav3_ext: Any,
    graph_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    size_k = int(payload_gpu["suh"].numel())
    generators = [
        torch.Generator(device="cpu").manual_seed(args.seed + rows * 1009 + offset)
        for offset in (0, 1)
    ]
    scenarios = [
        (torch.randn((rows, size_k), generator=generator) * args.input_std)
        .to(args.params_dtype)
        .to(device)
        for generator in generators
    ]
    source = torch.empty_like(scenarios[0])
    source.copy_(scenarios[0])
    routes = _make_routes(
        rows=rows,
        source=source,
        trellis_native=payload_gpu["trellis"],
        suh=payload_gpu["suh"],
        svh=payload_gpu["svh"],
        fused_weight=fused_weight,
        generic_weight=generic_weight,
        exllamav3_ext=exllamav3_ext,
        device=device,
    )
    stable_before = {
        name: _stable_contract(route.stable_tensors) for name, route in routes.items()
    }
    eager_first = {
        name: _time_eager_call(routes[name].run, device) for name in _ROUTE_NAMES
    }
    for _ in range(args.compile_warmups - 1):
        for name in _ROUTE_NAMES:
            routes[name].run()
    torch.cuda.synchronize(device)

    captures = {}
    freeze_kernel_resolution(
        f"checkpoint K6/MCG graph qualification M={rows} after eager warmup"
    )
    try:
        for name in _ROUTE_NAMES:
            captures[name] = _capture_route(
                routes[name], graph_dir=graph_dir, rows=rows, device=device
            )
    finally:
        unfreeze_kernel_resolution()

    capture_errors = []
    for name, capture in captures.items():
        capture_errors.extend(f"{name}: {error}" for error in capture["dot"]["errors"])
    if capture_errors:
        raise QualificationError("; ".join(capture_errors))
    capture_allocator_signatures = {
        name: {
            key: capture["allocator_delta"][key]
            for key in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "allocated_bytes.all.current",
                "active_bytes.all.current",
            )
        }
        for name, capture in captures.items()
    }
    route_independent_capture_overhead = (
        len(
            {
                tuple(signature.items())
                for signature in capture_allocator_signatures.values()
            }
        )
        == 1
    )
    if not route_independent_capture_overhead:
        raise QualificationError(
            "capture-time Torch allocator deltas differ by route: "
            f"{capture_allocator_signatures}"
        )

    # Preserve the first balanced post-capture samples before the deliberate
    # replay stress and correctness passes warm every graph further.
    cold_timing = _time_graphs(
        routes,
        warmups=0,
        iterations=args.cold_replays,
        device=device,
    )
    allocation_replay = {
        name: _replay_allocation_check(
            routes[name], replays=args.replay_checks, device=device
        )
        for name in _ROUTE_NAMES
    }
    replay_errors = [
        f"{name}: Torch allocator changed during graph replay"
        for name, check in allocation_replay.items()
        if not check["allocator_stable"]
    ]
    if replay_errors:
        raise QualificationError("; ".join(replay_errors))

    correctness = _correctness_scenarios(
        routes,
        source=source,
        scenarios=scenarios,
        raw_weight=raw_weight,
        suh=payload_gpu["suh"],
        svh=payload_gpu["svh"],
        args=args,
        device=device,
    )
    if not correctness["pass"]:
        raise QualificationError("; ".join(correctness["errors"]))
    for name, route in routes.items():
        _assert_same_addresses(stable_before[name], route.stable_tensors)

    source.copy_(scenarios[0])
    timing = _time_graphs(
        routes,
        warmups=args.warmups,
        iterations=args.iterations,
        device=device,
    )
    for name, route in routes.items():
        _assert_same_addresses(stable_before[name], route.stable_tensors)

    return {
        "rows": rows,
        "source_dtype": str(source.dtype),
        "output_dtype": str(routes["fused_b12x"].output.dtype),
        "accumulator_contract": (
            "FP32 internal accumulation; "
            f"{str(source.dtype).removeprefix('torch.').upper()} inter-rotation boundary"
        ),
        "served_exllamav3_boundary": (
            "direct FP16 exl3_gemm"
            if source.dtype == torch.float16
            else "BF16 source -> FP16 copy -> exl3_gemm -> BF16 output copy"
        ),
        "activation_scale_math": False,
        "stable_tensors": stable_before,
        "eager_first_call": eager_first,
        "captures": captures,
        "capture_allocation_contract": {
            "caller_owned_route_storage": True,
            "route_independent_framework_overhead": route_independent_capture_overhead,
            "signatures": capture_allocator_signatures,
            "interpretation": (
                "torch.cuda.graph capture bookkeeping is retained separately; "
                "the route call receives every output and scratch tensor"
            ),
        },
        "replay_allocation": allocation_replay,
        "correctness": correctness,
        "cold_graph_timing": cold_timing,
        "warm_graph_timing": timing,
        "pass": True,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.chmod(0o644)
    os.replace(temporary, path)


@contextlib.contextmanager
def _temporary_k6_mcg_grid_override(
    *,
    size_k: int,
    size_n: int,
    requested_grid_x: int | None,
) -> Iterator[dict[str, Any] | None]:
    """Apply one benchmark-only planner override and always restore it."""
    if requested_grid_x is None:
        yield None
        return
    requested_grid_x = int(requested_grid_x)
    if requested_grid_x <= 0:
        raise ValueError("--experimental-grid-x must be positive")

    from b12x.gemm.trellis_linear import _k6_mcg_cute

    table = _k6_mcg_cute._MEASURED_GRID_CTA
    shape = (int(size_k), int(size_n))
    had_previous = shape in table
    previous = table.get(shape)
    table[shape] = requested_grid_x
    try:
        yield {
            "experimental": True,
            "method": (
                "temporary benchmark-only mutation of b12x planner shape table "
                "during public prepare"
            ),
            "shape": list(shape),
            "previous_requested_grid_x": previous,
            "requested_grid_x": requested_grid_x,
        }
    finally:
        if had_previous:
            table[shape] = previous
        else:
            table.pop(shape, None)


def _run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if args.compile_warmups < 1:
        raise ValueError("--compile-warmups must be at least one")
    if min(args.cold_replays, args.warmups, args.iterations, args.replay_checks) < 1:
        raise ValueError("replay and timing counts must all be positive")
    order_count = len(_balanced_orders())
    if args.iterations % order_count:
        raise ValueError(
            f"--iterations must be a multiple of {order_count} to keep route "
            "ordering balanced"
        )
    if args.cold_replays % order_count:
        raise ValueError(
            f"--cold-replays must be a multiple of {order_count} to keep "
            "route ordering balanced"
        )
    if not torch.cuda.is_available():
        raise QualificationError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        raise QualificationError(
            f"this benchmark requires sm_120, got sm_{capability[0]}{capability[1]}"
        )
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report["environment"] = _environment_metadata(args, device)
    report["gpu_snapshots"] = {"start": _gpu_snapshot()}

    payload = _load_checkpoint_payload(
        args.model_dir.resolve(),
        args.tensor_prefix,
        verify_shard_sha=args.verify_shard_sha,
    )
    report["checkpoint"] = payload.metadata
    exllamav3_ext = _import_exllamav3_ext(args.exllamav3_ext_path)
    extension_path = Path(exllamav3_ext.__file__).resolve()
    report["exllamav3"] = {
        "extension_path": str(extension_path),
        "extension_sha256": _sha256_file(extension_path),
        "call": "exl3_gemm(x, trellis, output, suh, x_had, svh, -1, True, False, 0)",
    }

    payload_gpu = {
        "trellis": payload.trellis.to(device=device, non_blocking=False).contiguous(),
        "suh": payload.suh.to(device=device, non_blocking=False).contiguous(),
        "svh": payload.svh.to(device=device, non_blocking=False).contiguous(),
        "mcg": payload.mcg.to(device=device, non_blocking=False).contiguous(),
    }
    source_hashes_before = {
        name: _sha256_tensor(tensor) for name, tensor in payload_gpu.items()
    }
    with _temporary_k6_mcg_grid_override(
        size_k=int(payload_gpu["suh"].numel()),
        size_n=int(payload_gpu["svh"].numel()),
        requested_grid_x=args.experimental_grid_x,
    ) as planner_override:
        prepare_start_ns = time.perf_counter_ns()
        fused_weight = trellis_linear.prepare_weight(
            payload_gpu["trellis"],
            payload_gpu["suh"],
            payload_gpu["svh"],
            mcg=payload_gpu["mcg"],
            params_dtype=args.params_dtype,
        )
    torch.cuda.synchronize(device)
    prepare_wall_ms = (time.perf_counter_ns() - prepare_start_ns) / 1.0e6
    launch = fused_weight.k6_mcg_small_m_launch
    if launch is None:
        raise QualificationError("public prepare_weight did not bind K6McgSmallMKernel")
    generic_weight = replace(
        fused_weight,
        workspace=torch.zeros_like(fused_weight.workspace),
        k6_mcg_small_m_launch=None,
    )
    report["b12x_plan"] = {
        "public_prepare_run_boundary": True,
        "prepare_wall_ms": prepare_wall_ms,
        "weight_layout": fused_weight.weight_layout,
        "trellis_bits": fused_weight.trellis_bits,
        "trellis_codebook": fused_weight.trellis_codebook,
        "params_dtype": str(fused_weight.params_dtype),
        "in_features": fused_weight.in_features,
        "out_features": fused_weight.out_features,
        "bound_launch_type": f"{type(launch).__module__}.{type(launch).__name__}",
        "bound_launch_params_dtype": str(launch.params_dtype),
        "launch": {
            name: int(getattr(launch, name))
            for name in (
                "device_index",
                "size_k",
                "size_n",
                "grid_x",
                "cta_threads",
                "resident_ctas",
                "blocks_per_sm",
                "shared_memory_bytes",
            )
        },
        "launch_grid_by_rows": {
            str(rows): int(launch.launch_grid_x(rows)) for rows in args.rows
        },
        "fused_scratch_elements": int(launch.required_scratch_elements),
        "generic_forcing": "dataclasses.replace(weight, k6_mcg_small_m_launch=None)",
        "planner_override": planner_override,
    }
    raw_weight = torch.empty(
        (fused_weight.in_features, fused_weight.out_features),
        dtype=torch.float16,
        device=device,
    )
    reconstruct_start_ns = time.perf_counter_ns()
    exllamav3_ext.reconstruct(
        raw_weight,
        payload_gpu["trellis"],
        6,
        True,
        False,
    )
    torch.cuda.synchronize(device)
    report["oracle"] = {
        "reconstruction": "exllamav3_ext.reconstruct exact native payload",
        "rotation": "independent mathematical normalized H128 in Torch FP32",
        "gemm": (
            "Torch FP32 matmul with TF32 disabled; operands and rotation "
            f"boundaries rounded to {str(args.params_dtype).removeprefix('torch.')}"
        ),
        "reconstruct_wall_ms": (time.perf_counter_ns() - reconstruct_start_ns) / 1.0e6,
        "raw_weight": _tensor_contract(raw_weight),
        "raw_weight_finite": bool(torch.isfinite(raw_weight).all().item()),
        "raw_weight_nonzero": bool(torch.count_nonzero(raw_weight).item()),
        "logit_agreement": {
            "applicable": False,
            "reason": "this is an internal projection; model-level logits are a serving gate",
        },
    }
    if (
        not report["oracle"]["raw_weight_finite"]
        or not report["oracle"]["raw_weight_nonzero"]
    ):
        raise QualificationError(
            "reconstructed checkpoint weight is non-finite or all zero"
        )

    args.graph_dump_dir.mkdir(parents=True, exist_ok=True)
    report["rows"] = []
    for rows in args.rows:
        row_report = _benchmark_rows(
            rows,
            payload_gpu=payload_gpu,
            fused_weight=fused_weight,
            generic_weight=generic_weight,
            raw_weight=raw_weight,
            exllamav3_ext=exllamav3_ext,
            graph_dir=args.graph_dump_dir,
            args=args,
            device=device,
        )
        report["rows"].append(row_report)
        _write_json(args.output, report)

    source_hashes_after = {
        name: _sha256_tensor(tensor) for name, tensor in payload_gpu.items()
    }
    report["source_immutability"] = {
        "before": source_hashes_before,
        "after": source_hashes_after,
        "pass": source_hashes_before == source_hashes_after,
    }
    if source_hashes_before != source_hashes_after:
        raise QualificationError("checkpoint source tensors were mutated")
    report["gpu_snapshots"]["end"] = _gpu_snapshot()
    report["completed_at"] = _utc_now()
    report["qualification_pass"] = True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tensor-prefix", default=_DEFAULT_PREFIX)
    parser.add_argument("--rows", type=_parse_rows, default=(1, 4, 8, 16))
    parser.add_argument(
        "--params-dtype",
        type=_parse_params_dtype,
        default=torch.float16,
        metavar="{fp16,bf16}",
        help="activation, fused MMA, and output dtype (default: fp16)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--exllamav3-ext-path", type=Path, default=Path("/opt/exllamav3")
    )
    parser.add_argument("--graph-dump-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0x4B364D43)
    parser.add_argument("--input-std", type=float, default=0.05)
    parser.add_argument("--compile-warmups", type=int, default=3)
    parser.add_argument("--replay-checks", type=int, default=8)
    parser.add_argument("--cold-replays", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-relative-l2", type=float, default=0.003)
    parser.add_argument("--max-abs", type=float, default=1.0)
    parser.add_argument(
        "--require-exact-topk",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--verify-shard-sha",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--integration-tree", default="unknown")
    parser.add_argument("--image", default="unknown")
    parser.add_argument("--image-id", default="unknown")
    parser.add_argument(
        "--experimental-grid-x",
        type=int,
        help=(
            "benchmark-only b12x planner-table override applied before public "
            "prepare_weight; never a serving integration policy"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report: dict[str, Any] = {
        "schema": _SCHEMA,
        "started_at": _utc_now(),
        "qualification_pass": False,
        "rows": [],
    }
    exit_code = 0
    try:
        _run(args, report)
    except Exception as exc:
        exit_code = 2
        report["fatal_error"] = {
            "type": f"{type(exc).__module__}.{type(exc).__name__}",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        report["completed_at"] = _utc_now()
        with contextlib.suppress(Exception):
            report.setdefault("gpu_snapshots", {})["failure"] = _gpu_snapshot()
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "schema": _SCHEMA,
                "output": str(args.output.resolve()),
                "qualification_pass": report["qualification_pass"],
                "fatal_error": report.get("fatal_error", {}).get("message"),
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
