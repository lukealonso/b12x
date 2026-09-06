#!/usr/bin/env python3
"""Run the PR #243 checkpoint benchmark against GLM v39's TP4 rank-zero slice.

The reusable benchmark lives in ``benchmarks/benchmark_trellis_k6_mcg_checkpoint.py``.
This adapter changes only package naming (v39 installs b12x as ``sparkinfer``)
and checkpoint loading (the GLM safetensors store global projections while the
served rank receives one tensor-parallel slice).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import safetensors
from safetensors import safe_open
import sparkinfer
from sparkinfer import _lib as sparkinfer_lib
from sparkinfer._lib import cuda_graph_dot, runtime_control
from sparkinfer import gemm as sparkinfer_gemm
from sparkinfer.gemm import trellis_linear
from sparkinfer.gemm.trellis_linear import _k6_mcg_cute
from sparkinfer import moe as sparkinfer_moe
from sparkinfer.moe import _shared as sparkinfer_moe_shared
from sparkinfer.moe._shared import kernels as sparkinfer_moe_kernels
from sparkinfer.moe._shared.kernels import w4a16 as sparkinfer_w4a16
from sparkinfer.moe._shared.kernels.w4a16 import host as sparkinfer_w4a16_host
import torch


# The upstream benchmark imports b12x. v39 carries the same code under the
# historical sparkinfer package name. Alias the exact already-imported module
# objects so the planner override and public runtime operate on one module set.
_MODULE_ALIASES = {
    "b12x": sparkinfer,
    "b12x._lib": sparkinfer_lib,
    "b12x._lib.cuda_graph_dot": cuda_graph_dot,
    "b12x._lib.runtime_control": runtime_control,
    "b12x.gemm": sparkinfer_gemm,
    "b12x.gemm.trellis_linear": trellis_linear,
    "b12x.gemm.trellis_linear._k6_mcg_cute": _k6_mcg_cute,
    "b12x.moe": sparkinfer_moe,
    "b12x.moe._shared": sparkinfer_moe_shared,
    "b12x.moe._shared.kernels": sparkinfer_moe_kernels,
    "b12x.moe._shared.kernels.w4a16": sparkinfer_w4a16,
    "b12x.moe._shared.kernels.w4a16.host": sparkinfer_w4a16_host,
}
sys.modules.update(_MODULE_ALIASES)

from benchmarks import benchmark_trellis_k6_mcg_checkpoint as benchmark  # noqa: E402


_GLM_TP_SIZE = 4
_ORIGINAL_LOAD_CHECKPOINT_PAYLOAD = benchmark._load_checkpoint_payload
_GLM_GLOBAL_CONTRACTS = {
    "model.layers.3.self_attn.o_proj": {
        "trellis_shape": (1024, 384, 96),
        "slice_axis": "k",
        "local_shape": (4096, 6144),
    },
    "model.layers.3.self_attn.q_b_proj": {
        "trellis_shape": (128, 1024, 96),
        "slice_axis": "n",
        "local_shape": (2048, 4096),
    },
}


def _verify_source_manifest(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "SOURCE_R7_MANIFEST.json"
    checksum_path = model_dir / "SOURCE_R7_MANIFEST.sha256"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(
            "GLM checkpoint source manifest or its checksum is missing: "
            f"{manifest_path}, {checksum_path}"
        )
    fields = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != manifest_path.name:
        raise benchmark.QualificationError(
            f"malformed source-manifest checksum: {checksum_path}"
        )
    expected = fields[0].lower()
    actual = benchmark._sha256_file(manifest_path)
    if expected != actual:
        raise benchmark.QualificationError(
            "SOURCE_R7_MANIFEST.json hash mismatch: "
            f"expected={expected}, actual={actual}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "path": str(manifest_path.resolve()),
        "sha256_path": str(checksum_path.resolve()),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "verified": True,
        "schema": manifest.get("schema"),
        "recipe_version": manifest.get("recipe_version"),
        "payload_hash_verification_declared_by_checkpoint": manifest.get(
            "payload_hash_verification"
        ),
    }


def _slice_rank_zero(
    prefix: str,
    tensors: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _GLM_GLOBAL_CONTRACTS.get(prefix)
    if contract is None:
        raise benchmark.QualificationError(
            f"no reviewed GLM TP4 contract for tensor prefix {prefix!r}"
        )
    trellis = tensors["trellis"]
    suh = tensors["suh"]
    svh = tensors["svh"]
    if tuple(trellis.shape) != contract["trellis_shape"]:
        raise benchmark.QualificationError(
            f"global Trellis shape mismatch for {prefix}: "
            f"expected={contract['trellis_shape']}, actual={tuple(trellis.shape)}"
        )
    full_hashes = {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": benchmark._sha256_tensor(tensor),
        }
        for name, tensor in tensors.items()
    }
    if contract["slice_axis"] == "k":
        local_trellis = trellis[: trellis.shape[0] // _GLM_TP_SIZE, :, :].contiguous()
        local_suh = suh[: suh.numel() // _GLM_TP_SIZE].contiguous()
        local_svh = svh.contiguous()
    elif contract["slice_axis"] == "n":
        local_trellis = trellis[:, : trellis.shape[1] // _GLM_TP_SIZE, :].contiguous()
        local_suh = suh.contiguous()
        local_svh = svh[: svh.numel() // _GLM_TP_SIZE].contiguous()
    else:
        raise AssertionError(f"unknown TP slice axis: {contract['slice_axis']}")
    local = {
        "trellis": local_trellis,
        "suh": local_suh,
        "svh": local_svh,
        "mcg": tensors["mcg"].contiguous(),
    }
    local_shape = (int(local_suh.numel()), int(local_svh.numel()))
    if local_shape != contract["local_shape"]:
        raise benchmark.QualificationError(
            f"TP4 local projection mismatch: expected={contract['local_shape']}, "
            f"actual={local_shape}"
        )
    return local, {
        "tp_size": _GLM_TP_SIZE,
        "tp_rank": 0,
        "slice_axis": contract["slice_axis"],
        "global_tensors": full_hashes,
        "local_shape_k_n": list(local_shape),
    }


def _load_glm_payload(
    model_dir: Path,
    prefix: str,
    *,
    verify_shard_sha: bool,
) -> benchmark.CheckpointPayload:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"checkpoint index not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} has no object-valued weight_map")
    keys = {suffix: f"{prefix}.{suffix}" for suffix in benchmark._TENSOR_SUFFIXES}
    missing = [key for key in keys.values() if key not in weight_map]
    if missing:
        raise KeyError(f"checkpoint index is missing keys: {missing}")
    shard_names = {str(weight_map[key]) for key in keys.values()}
    if len(shard_names) != 1:
        raise ValueError(f"projection tensors span shards: {sorted(shard_names)}")
    shard_name = next(iter(shard_names))
    shard_path = model_dir / shard_name
    if not shard_path.is_file():
        raise FileNotFoundError(f"checkpoint shard not found: {shard_path}")

    # This checkpoint does not ship SHA256SUMS for model shards. Bind every run
    # to the exact shard bytes and say explicitly that this is not comparison
    # against an independently supplied expected digest.
    actual_shard_sha = benchmark._sha256_file(shard_path)
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        full_tensors = {
            suffix: handle.get_tensor(keys[suffix])
            for suffix in benchmark._TENSOR_SUFFIXES
        }
    tensors, tp_contract = _slice_rank_zero(prefix, full_tensors)
    trellis = tensors["trellis"]
    suh = tensors["suh"]
    svh = tensors["svh"]
    mcg = tensors["mcg"]
    if trellis.dtype != torch.int16 or trellis.ndim != 3:
        raise benchmark.QualificationError(
            f"local K6 payload must be rank-3 int16, got {trellis.dtype} "
            f"{tuple(trellis.shape)}"
        )
    if int(trellis.shape[-1]) != 96:
        raise benchmark.QualificationError(
            f"local payload is not six-bit Trellis storage: {tuple(trellis.shape)}"
        )
    expected_k = 4096 if prefix.endswith("o_proj") else 2048
    expected_n = 6144 if prefix.endswith("o_proj") else 4096
    if suh.dtype != torch.float16 or tuple(suh.shape) != (expected_k,):
        raise benchmark.QualificationError(
            f"local suh contract mismatch: {suh.dtype} {tuple(suh.shape)}"
        )
    if svh.dtype != torch.float16 or tuple(svh.shape) != (expected_n,):
        raise benchmark.QualificationError(
            f"local svh contract mismatch: {svh.dtype} {tuple(svh.shape)}"
        )
    if mcg.dtype not in (torch.int32, torch.uint32) or mcg.numel() != 1:
        raise benchmark.QualificationError(
            f"mcg marker must be scalar int32/uint32, got {mcg.dtype} {tuple(mcg.shape)}"
        )
    tensor_metadata = {
        suffix: {
            "key": keys[suffix],
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
            "bytes": tensor.numel() * tensor.element_size(),
            "sha256": benchmark._sha256_tensor(tensor),
        }
        for suffix, tensor in tensors.items()
    }
    metadata = {
        "model_dir": str(model_dir.resolve()),
        "prefix": prefix,
        "keys": keys,
        "index_path": str(index_path.resolve()),
        "index_sha256": benchmark._sha256_file(index_path),
        "shard_name": shard_name,
        "shard_path": str(shard_path.resolve()),
        "shard_bytes": shard_path.stat().st_size,
        "expected_shard_sha256": None,
        "actual_shard_sha256": actual_shard_sha,
        "shard_sha256_verified": False,
        "shard_identity_mode": (
            "exact file SHA-256 recorded; checkpoint supplies no external shard digest"
        ),
        "verify_shard_sha_requested": verify_shard_sha,
        "source_manifest": _verify_source_manifest(model_dir),
        "tensor_parallel_contract": tp_contract,
        "size_k": int(suh.numel()),
        "size_n": int(svh.numel()),
        "trellis_bits": 6,
        "codebook": "mcg",
        "mcg_marker_unsigned": int(mcg.item()) & 0xFFFFFFFF,
        "tensors": tensor_metadata,
    }
    return benchmark.CheckpointPayload(
        prefix=prefix,
        shard_name=shard_name,
        trellis=trellis,
        suh=suh,
        svh=svh,
        mcg=mcg,
        metadata=metadata,
    )


def _load_v39_payload(
    model_dir: Path,
    prefix: str,
    *,
    verify_shard_sha: bool,
) -> benchmark.CheckpointPayload:
    """Use reviewed GLM TP slicing, otherwise preserve the upstream loader."""
    if prefix in _GLM_GLOBAL_CONTRACTS:
        return _load_glm_payload(
            model_dir,
            prefix,
            verify_shard_sha=verify_shard_sha,
        )
    return _ORIGINAL_LOAD_CHECKPOINT_PAYLOAD(
        model_dir,
        prefix,
        verify_shard_sha=verify_shard_sha,
    )


_ORIGINAL_ENVIRONMENT_METADATA = benchmark._environment_metadata


def _environment_metadata(args, device):
    metadata = _ORIGINAL_ENVIRONMENT_METADATA(args, device)
    metadata["packages"]["sparkinfer"] = benchmark._package_version("sparkinfer")
    metadata["packages"]["safetensors"] = safetensors.__version__
    metadata["architecture_environment"].update(
        {
            name: os.environ.get(name)
            for name in (
                "SPARKINFER_COMPILE_CACHE_DIR",
                "SPARKINFER_CUTE_COMPILE_CACHE_DIR",
                "CUTE_DSL_CACHE_DIR",
                "CUTE_DSL_COMPILE_CACHE_DIR",
                "CUDA_CACHE_PATH",
                "TORCH_EXTENSIONS_DIR",
            )
        }
    )
    adapter_path = Path(__file__).resolve()
    benchmark_path = Path(benchmark.__file__).resolve()
    metadata["v39_adapter"] = {
        "path": str(adapter_path),
        "sha256": benchmark._sha256_file(adapter_path),
        "upstream_benchmark_path": str(benchmark_path),
        "upstream_benchmark_sha256": benchmark._sha256_file(benchmark_path),
        "package_aliases": sorted(_MODULE_ALIASES),
    }
    return metadata


benchmark._load_checkpoint_payload = _load_v39_payload
benchmark._environment_metadata = _environment_metadata


if __name__ == "__main__":
    raise SystemExit(benchmark.main())
