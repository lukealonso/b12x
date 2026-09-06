#!/usr/bin/env python3
"""Generate a deterministic K6/MCG checkpoint fixture for shape regression tests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from safetensors.torch import save_file
import torch


DEFAULT_PREFIX = "model.language_model.layers.0.mlp.down_proj"
MCG_MARKER_SIGNED = -877_912_083


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(value.numpy())).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--size-k", type=int, required=True)
    result.add_argument("--size-n", type=int, required=True)
    result.add_argument("--prefix", default=DEFAULT_PREFIX)
    result.add_argument("--seed", type=int, default=0x51384B36)
    return result


def main() -> int:
    args = parser().parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"fixture output already exists: {output_dir}")
    if args.size_k % 16 or args.size_n % 16:
        raise ValueError("K and N must be divisible by 16")
    output_dir.mkdir(parents=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    trellis = torch.randint(
        -(1 << 15),
        1 << 15,
        (args.size_k // 16, args.size_n // 16, 96),
        dtype=torch.int32,
        generator=generator,
    ).to(torch.int16)
    suh_magnitude = 0.005 + 0.015 * torch.rand(
        args.size_k, generator=generator, dtype=torch.float32
    )
    svh_magnitude = 0.5 + torch.rand(
        args.size_n, generator=generator, dtype=torch.float32
    )
    suh_sign = torch.randint(
        0, 2, (args.size_k,), generator=generator, dtype=torch.int8
    ).to(torch.float32)
    svh_sign = torch.randint(
        0, 2, (args.size_n,), generator=generator, dtype=torch.int8
    ).to(torch.float32)
    suh = (suh_magnitude * (suh_sign * 2.0 - 1.0)).to(torch.float16)
    svh = (svh_magnitude * (svh_sign * 2.0 - 1.0)).to(torch.float16)
    mcg = torch.tensor(MCG_MARKER_SIGNED, dtype=torch.int32)
    tensors = {"trellis": trellis, "suh": suh, "svh": svh, "mcg": mcg}
    keyed = {f"{args.prefix}.{name}": tensor for name, tensor in tensors.items()}

    shard_name = "model-00001-of-00001.safetensors"
    shard_path = output_dir / shard_name
    save_file(
        keyed,
        shard_path,
        metadata={
            "format": "pt",
            "purpose": "deterministic shape-only K6/MCG regression fixture",
        },
    )
    weight_map = {name: shard_name for name in keyed}
    index_path = output_dir / "model.safetensors.index.json"
    write_json(
        index_path,
        {
            "metadata": {"total_size": shard_path.stat().st_size},
            "weight_map": weight_map,
        },
    )
    shard_sha = sha256_file(shard_path)
    sums_path = output_dir / "SHA256SUMS"
    sums_path.write_text(f"{shard_sha}  {shard_name}\n", encoding="utf-8")
    receipt = {
        "schema": "b12x.k6_mcg_shape_fixture.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "shape-level performance and compiled-object regression only; "
            "not checkpoint numerical qualification"
        ),
        "prefix": args.prefix,
        "size_k": args.size_k,
        "size_n": args.size_n,
        "trellis_bits": 6,
        "codebook": "mcg",
        "mcg_marker_signed": MCG_MARKER_SIGNED,
        "mcg_marker_unsigned": MCG_MARKER_SIGNED & 0xFFFFFFFF,
        "seed": args.seed,
        "shard": {
            "name": shard_name,
            "bytes": shard_path.stat().st_size,
            "sha256": shard_sha,
        },
        "index_sha256": sha256_file(index_path),
        "sha256sums_sha256": sha256_file(sums_path),
        "tensors": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": sha256_tensor(tensor),
            }
            for name, tensor in tensors.items()
        },
    }
    write_json(output_dir / "fixture_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
