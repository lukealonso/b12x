"""Transfer exact NVFP4 benchmark operands between GPU hosts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.benchmark_moe import ModelSpec

_WEIGHT_FIELDS = (
    "w13_weight", "w13_blockscale_swizzled", "w2_weight", "w2_blockscale_swizzled",
    "g1_alphas", "g2_alphas", "w13_input_scale_quant", "w2_input_scale_quant",
)


def tensor_digest(tensor: torch.Tensor) -> dict:
    value = tensor.contiguous().reshape(-1).view(torch.uint8).cpu()
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(value.numpy()).hexdigest(),
    }


def save_snapshot(path: Path, *, weights, inputs, metadata: dict) -> None:
    tensors = {name: getattr(weights, name).cpu() for name in _WEIGHT_FIELDS}
    for m, operands in inputs.items():
        for name, value in zip(("x", "ids", "route_weights"), operands, strict=True):
            tensors[f"m{m}.{name}"] = value.cpu()
    torch.save({
        "schema_version": 1,
        "metadata": metadata,
        "spec": asdict(weights.spec),
        "layer": weights.layer_idx,
        "source_format": weights.source_format,
        "w13_layout": weights.w13_layout,
        "counts": sorted(inputs),
        "tensors": tensors,
        "tensor_digests": {name: tensor_digest(value) for name, value in tensors.items()},
    }, path)


def load_snapshot(path: Path, *, device, expected_metadata: dict):
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    if snapshot["schema_version"] != 1:
        raise ValueError("unsupported MoE checkpoint snapshot schema")
    if snapshot["metadata"] != expected_metadata:
        raise ValueError("snapshot checkpoint metadata does not match the benchmark")
    tensors = snapshot["tensors"]
    if {name: tensor_digest(value) for name, value in tensors.items()} != snapshot["tensor_digests"]:
        raise ValueError("snapshot tensor identity verification failed")
    tensors = {name: value.to(device) for name, value in tensors.items()}
    weights = SimpleNamespace(
        **{name: tensors[name] for name in _WEIGHT_FIELDS},
        spec=ModelSpec(**snapshot["spec"]), layer_idx=snapshot["layer"],
        source_format=snapshot["source_format"], w13_layout=snapshot["w13_layout"],
    )
    inputs = {
        m: tuple(tensors[f"m{m}.{name}"] for name in ("x", "ids", "route_weights"))
        for m in snapshot["counts"]
    }
    return weights, inputs, snapshot["tensor_digests"]
