#!/usr/bin/env python3
"""Lossless EXL3-to-b12x_trellis exporter for trellis-coded MoE checkpoints.

Converts checkpoints whose routed experts carry exllamav3-family trellis
tensors into the `b12x_trellis` version-2 in-checkpoint standard consumed
by `b12x.moe.fused_moe.config.TrellisConfig` and the projection-mixed
Trellis preparation path. Because the standard's payload IS the
topology-neutral whole-matrix tensor layout, conversion is metadata-only:
the exporter derives the four model-level metadata tensors and the
`quantization_config` block from the payload and the producer's
per-tensor scale vectors, and never rewrites, decodes, or re-encodes a
trellis symbol.

Scale granularities are elected from the tensors themselves, per family:
layer-shared vectors produce `vectors: "per_layer"`, per-expert vectors
produce `vectors: "per_expert"`; `gains` is `"none"`. Sources whose gate
and up hidden-axis input vectors differ for the same expert are not
representable (the standard carries one input-scale family) and fail
closed, as do rank-sliced payloads (normalize to whole-matrix first) and
non-registry MCG multipliers.

Every emitted value is verified: rate bytes are validated against the
trellis tensors' own bit widths, the configuration block round-trips
through the in-tree `TrellisConfig` parser with the rate tensor matching
its declared model-level shape, and the written metadata tensors are
re-read and compared element-for-element against the vectors collected
from the source.

Usage:
    python scripts/export_exl3_to_b12x_trellis.py \
        --model-dir /path/to/checkpoint --output /path/to/out \
        --report out/conversion-report.json
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pathlib
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from b12x.moe._shared.trellis_codebooks import (  # noqa: E402
    MCG,
    MCG_MULTIPLIER,
)
from b12x.moe.fused_moe.config import TrellisConfig  # noqa: E402

RATE_TENSOR = "b12x_trellis.rate"
INPUT_SCALES_TENSOR = "b12x_trellis.input_scales"
INTERMEDIATE_SCALES_TENSOR = "b12x_trellis.intermediate_scales"
OUTPUT_SCALES_TENSOR = "b12x_trellis.output_scales"

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_WHOLE = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|"
    r"down_proj)\.(?P<suffix>trellis|suh|svh|mcg)$"
)
_SLICED = re.compile(r"^.+\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.rank\d+\.")
_SHARED = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<name>[A-Za-z0-9_]+)\."
    r"(?P<vec>gate_up_suh|down_svh)$"
)
_LAYER = re.compile(r"\.layers\.(\d+)\.")


def _rate_byte(bits: int) -> int:
    """Whole-matrix rate byte: low nibble == high nibble == bits."""

    return (int(bits) << 4) | int(bits)


def _fail(message: str) -> SystemExit:
    return SystemExit(f"export_exl3_to_b12x_trellis: error: {message}")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Source:
    """Index of expert and shared-vector tensors across the shards.

    Tensors for layers outside ``moe_layers`` — for example a rank-sliced
    MTP draft block carried in the model's own format — are ignored and
    counted, not errors: the standard covers the routed-expert layers.
    """

    def __init__(self, root: pathlib.Path, moe_layers: set[int]) -> None:
        self.root = root
        self.moe_layers = moe_layers
        self.whole: dict[int, dict] = {}
        self.shared: dict[int, dict[str, tuple[str, str]]] = {}
        self.ignored_out_of_scope = 0
        shards = sorted(root.glob("*.safetensors"))
        if not shards:
            raise _fail(f"no safetensors shards under {root}")
        for shard in shards:
            with safe_open(str(shard), framework="pt") as handle:
                for name in handle.keys():
                    self._index(shard.name, name)

    def _index(self, filename: str, name: str) -> None:
        sliced = bool(_SLICED.match(name))
        whole = _WHOLE.match(name)
        shared = _SHARED.match(name)
        if not (sliced or whole or shared):
            return
        layer_match = _LAYER.search(name)
        if layer_match is None:
            raise _fail(
                f"{name!r} matches an expert tensor pattern but carries "
                "no '.layers.<N>.' index"
            )
        layer = int(layer_match.group(1))
        if layer not in self.moe_layers:
            self.ignored_out_of_scope += 1
            return
        if sliced:
            raise _fail(
                f"{name!r} is a rank-sliced payload tensor; the "
                "b12x_trellis standard carries topology-neutral "
                "whole-matrix payloads — normalize the checkpoint to "
                "whole-matrix tensors first"
            )
        if shared:
            vec = shared.group("vec")
            slot = self.shared.setdefault(layer, {})
            previous = slot.setdefault(vec, (filename, name))
            if previous != (filename, name):
                raise _fail(
                    f"layer {layer} {vec} resolves to two tensors: "
                    f"{previous[1]!r} in {previous[0]!r} and {name!r} in "
                    f"{filename!r}"
                )
            return
        key = (int(whole.group("expert")), whole.group("proj"))
        slot = self.whole.setdefault(layer, {}).setdefault(key, {})
        suffix = whole.group("suffix")
        if suffix in slot:
            raise _fail(f"duplicate tensor for {name!r}")
        slot[suffix] = (filename, name)

    def layers(self) -> list[int]:
        return sorted(self.whole)


@functools.lru_cache(maxsize=None)
def _tensor_cached(root: str, filename: str, name: str) -> torch.Tensor:
    with safe_open(str(pathlib.Path(root) / filename), framework="pt") as h:
        return h.get_tensor(name)


def _tensor(source: _Source, entry: tuple[str, str]) -> torch.Tensor:
    return _tensor_cached(str(source.root), entry[0], entry[1])


def _shape(source: _Source, entry: tuple[str, str]) -> tuple[int, ...]:
    with safe_open(str(source.root / entry[0]), framework="pt") as handle:
        return tuple(handle.get_slice(entry[1]).get_shape())


def _check_mcg(value: torch.Tensor, name: str) -> None:
    seed = int(value.reshape(()).item()) & 0xFFFFFFFF
    if seed != MCG_MULTIPLIER:
        raise _fail(
            f"{name} declares MCG multiplier {seed:#010x}; b12x_trellis "
            f"mcg checkpoints require {MCG_MULTIPLIER:#010x}"
        )


def _sidecar_bit_map(source: _Source, layer: int) -> dict | None:
    """Merge every matching sidecar bit map, rejecting conflicts."""

    merged: dict | None = None
    for filename in sorted(
        {
            entry[0]
            for slots in source.whole.get(layer, {}).values()
            for entry in slots.values()
        }
    ):
        sidecar = (source.root / filename).with_suffix(".json")
        if not sidecar.exists():
            continue
        data = json.loads(sidecar.read_text())
        bit_map = data.get("bit_map")
        if not isinstance(bit_map, dict):
            continue
        if merged is None:
            merged = {}
        for key, declared in bit_map.items():
            previous = merged.setdefault(key, declared)
            if int(previous) != int(declared):
                raise _fail(
                    f"conflicting sidecar bit maps for {key!r}: "
                    f"{previous} and {declared}"
                )
    return merged


def _to_fp16_exact(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Narrow to fp16 only when every value is exactly representable."""

    if tensor.dtype == torch.float16:
        return tensor
    narrowed = tensor.to(torch.float16)
    if not torch.equal(narrowed.to(tensor.dtype), tensor):
        raise _fail(
            f"{name} is {tensor.dtype} with values not exactly "
            "representable in fp16; refusing a lossy scale conversion"
        )
    return narrowed


def export(args: argparse.Namespace) -> dict:
    model_dir = pathlib.Path(args.model_dir)
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((model_dir / "config.json").read_text())
    hidden = int(config["hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    experts = int(config["n_routed_experts"])
    first_dense = int(config.get("first_k_dense_replace", 0))
    num_layers = int(config["num_hidden_layers"])

    block = int(args.block_size)
    if block < 32 or block & (block - 1):
        raise _fail(f"--block-size must be a power of two of at least 32, got {block}")
    if intermediate % block:
        raise _fail(
            f"--block-size {block} does not divide the "
            f"{intermediate}-channel expert intermediate axis; no aligned "
            "rank extents would cover the payload"
        )

    expected = list(range(first_dense, num_layers))
    if not expected:
        raise _fail(
            f"config declares no MoE layers (first_k_dense_replace="
            f"{first_dense}, num_hidden_layers={num_layers})"
        )
    source = _Source(model_dir, set(expected))
    layers = source.layers()
    if layers != expected:
        raise _fail(
            f"payload covers layers {layers[:3]}..{layers[-1:]} but the "
            f"config declares MoE layers {first_dense}..{num_layers - 1}; "
            "refusing to build metadata from a partial checkpoint"
        )
    L = len(layers)
    print(
        f"model: L={L} (layers {layers[0]}..{layers[-1]}), E={experts}, "
        f"H={hidden}, I={intermediate}"
    )

    rate = torch.zeros((L, experts, 3), dtype=torch.uint8)
    input_rows = torch.zeros((L, experts, hidden), dtype=torch.float16)
    output_rows = torch.zeros((L, experts, hidden), dtype=torch.float16)
    inter_rows = torch.zeros((L, experts, 3, intermediate), dtype=torch.float16)
    checks = {"rate_vs_trellis_widths": 0, "sidecar_bit_map_layers": 0}

    for row, layer in enumerate(layers):
        entries = source.whole[layer]
        shared = source.shared.get(layer, {})
        bit_map = _sidecar_bit_map(source, layer)
        shared_in = (
            _tensor(source, shared["gate_up_suh"]) if "gate_up_suh" in shared else None
        )
        shared_out = (
            _tensor(source, shared["down_svh"]) if "down_svh" in shared else None
        )
        for expert in range(experts):
            for pi, proj in enumerate(_PROJECTIONS):
                slot = entries.get((expert, proj))
                if slot is None or "trellis" not in slot:
                    raise _fail(
                        f"layer {layer} expert {expert} {proj} has no trellis tensor"
                    )
                fc1 = proj != "down_proj"
                shape = _shape(source, slot["trellis"])
                if len(shape) != 3 or shape[2] % 16:
                    raise _fail(f"{slot['trellis'][1]} must be [in/16, out/16, 16K]")
                h_tiles = shape[0] if fc1 else shape[1]
                i_tiles = shape[1] if fc1 else shape[0]
                if (h_tiles * 16, i_tiles * 16) != (hidden, intermediate):
                    raise _fail(
                        f"{slot['trellis'][1]} geometry {shape} does not "
                        "match the model configuration"
                    )
                bits = shape[2] // 16
                if bits not in (3, 4, 5):
                    raise _fail(
                        f"{slot['trellis'][1]} carries K{bits}; the "
                        "b12x_trellis rate bytes cover K3/K4/K5"
                    )
                if bit_map is not None:
                    declared = bit_map.get(slot["trellis"][1].rsplit(".", 1)[0])
                    if declared is not None and int(declared) != bits:
                        raise _fail(
                            f"sidecar declares {slot['trellis'][1]} at "
                            f"K{declared}; the tensor carries K{bits}"
                        )
                checks["rate_vs_trellis_widths"] += 1
                rate[row, expert, pi] = _rate_byte(bits)
                if "mcg" in slot:
                    _check_mcg(_tensor(source, slot["mcg"]), slot["trellis"][1])
                ivec = slot.get("svh" if fc1 else "suh")
                if ivec is None:
                    raise _fail(
                        f"layer {layer} expert {expert} {proj} has no "
                        "intermediate-axis scale vector"
                    )
                vector = _tensor(source, ivec)
                if tuple(vector.shape) != (intermediate,):
                    raise _fail(
                        f"{ivec[1]} shape {tuple(vector.shape)} != [{intermediate}]"
                    )
                inter_rows[row, expert, pi] = _to_fp16_exact(vector, ivec[1])
                hvec_entry = slot.get("suh" if fc1 else "svh")
                if hvec_entry is not None:
                    hvec = _to_fp16_exact(_tensor(source, hvec_entry), hvec_entry[1])
                elif fc1 and shared_in is not None:
                    hvec = _to_fp16_exact(shared_in, "gate_up_suh")
                elif not fc1 and shared_out is not None:
                    hvec = _to_fp16_exact(shared_out, "down_svh")
                else:
                    raise _fail(
                        f"layer {layer} expert {expert} {proj} has no "
                        "hidden-axis scale vector and the layer shares "
                        "none"
                    )
                if tuple(hvec.shape) != (hidden,):
                    raise _fail(
                        f"hidden vector for layer {layer} expert {expert} "
                        f"{proj} has shape {tuple(hvec.shape)}"
                    )
                if fc1:
                    if pi == 0:
                        input_rows[row, expert] = hvec
                    elif not torch.equal(input_rows[row, expert], hvec):
                        raise _fail(
                            f"layer {layer} expert {expert}: gate and up "
                            "hidden-axis input vectors differ; the "
                            "b12x_trellis input_scales family carries one "
                            "vector for both and cannot represent this "
                            "checkpoint"
                        )
                else:
                    output_rows[row, expert] = hvec
        if bit_map is not None:
            checks["sidecar_bit_map_layers"] += 1
        _tensor_cached.cache_clear()
        print(f"  layer {layer} -> index {row}: collected")

    def _elect_hidden(rows: torch.Tensor) -> tuple[str, torch.Tensor]:
        if bool(torch.all(rows == rows[:, :1, :].expand_as(rows))):
            return "per_layer", rows[:, 0, :].contiguous()
        return "per_expert", rows.contiguous()

    def _elect_intermediate(rows: torch.Tensor) -> tuple[str, torch.Tensor]:
        if bool(torch.all(rows == rows[:, :1, :, :].expand_as(rows))):
            return "per_layer", rows[:, 0, :, :].contiguous()
        return "per_expert", rows.contiguous()

    in_gran, input_scales = _elect_hidden(input_rows)
    out_gran, output_scales = _elect_hidden(output_rows)
    inter_gran, inter_scales = _elect_intermediate(inter_rows)

    trellis_block = {
        "version": 2,
        "codebook": MCG,
        "rate": {"granularity": "per_expert_projection"},
        "scale": {
            "input_scales": {"vectors": in_gran, "gains": "none"},
            "intermediate_scales": {"vectors": inter_gran, "gains": "none"},
            "output_scales": {"vectors": out_gran, "gains": "none"},
        },
        "transform": {
            "projection": {"kind": "scaled_hadamard", "block_size": block},
            "expert": {"kind": "none"},
        },
    }
    # The in-tree parser is the contract: the emitted block must
    # round-trip through it, and the rate tensor must match its declared
    # model-level shape.
    parsed = TrellisConfig.from_dict(trellis_block)
    declared_shape = parsed.rate.tensor_shape(
        num_layers=L, num_experts=experts, intermediate_size=intermediate
    )
    if tuple(rate.shape) != declared_shape:
        raise _fail(
            f"rate tensor shape {tuple(rate.shape)} does not match the "
            f"declared granularity's {declared_shape}"
        )

    tensors = {
        RATE_TENSOR: rate,
        INPUT_SCALES_TENSOR: input_scales,
        INTERMEDIATE_SCALES_TENSOR: inter_scales,
        OUTPUT_SCALES_TENSOR: output_scales,
    }
    out_file = output / "b12x-trellis.safetensors"
    save_file(
        tensors,
        str(out_file),
        metadata={
            "format": "pt",
            "quant_method": "b12x_trellis",
            "b12x_trellis_version": "2",
            "moe_layer_indices": f"{layers[0]}..{layers[-1]}",
            "projection_order": "gate,up,down",
        },
    )
    (output / "quantization_config.b12x_trellis.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "b12x_trellis",
                    "b12x_trellis": trellis_block,
                }
            },
            indent=2,
        )
        + "\n"
    )
    patched = dict(config)
    merged = dict(patched.get("quantization_config") or {})
    merged["quant_method"] = "b12x_trellis"
    merged["b12x_trellis"] = trellis_block
    patched["quantization_config"] = merged
    (output / "config.patched.json").write_text(
        json.dumps(patched, indent=2, sort_keys=True) + "\n"
    )

    # Re-read the written file and compare every tensor to the collected
    # values, so the recorded result covers the bytes on disk.
    verified = 0
    with safe_open(str(out_file), framework="pt") as handle:
        for name, expected_tensor in tensors.items():
            written = handle.get_tensor(name)
            if not torch.equal(written, expected_tensor):
                raise _fail(f"round-trip mismatch on {name}")
            verified += 1

    report = {
        "tool": "export_exl3_to_b12x_trellis",
        "standard": "b12x_trellis version 2",
        "model_dir": str(model_dir),
        "geometry": {
            "L": L,
            "E": experts,
            "H": hidden,
            "I": intermediate,
            "moe_layers": f"{layers[0]}..{layers[-1]}",
        },
        "elections": {
            "input_scales": in_gran,
            "intermediate_scales": inter_gran,
            "output_scales": out_gran,
        },
        "rate_byte_histogram": {
            f"0x{byte:02x}": int((rate == byte).sum()) for byte in (0x33, 0x44, 0x55)
        },
        "gate_up_divergent_assignments": int((rate[:, :, 0] != rate[:, :, 1]).sum()),
        "checks": checks,
        "ignored_out_of_scope_tensors": source.ignored_out_of_scope,
        "verification": {
            "config_block": "parsed by b12x.moe.fused_moe.config."
            "TrellisConfig with the rate tensor matching its declared "
            "shape",
            "tensors_reread_element_identical": verified,
            "result": "element-identical",
        },
        "output": {
            "file": out_file.name,
            "sha256": _sha256_file(out_file),
            "bytes": out_file.stat().st_size,
        },
    }
    print(
        f"verification: config block parsed by TrellisConfig; {verified} "
        "written tensors re-read element-identical"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="scaled-Hadamard projection transform block width in "
        "channels: a power of two, at least 32, dividing the expert "
        "intermediate axis (default 128)",
    )
    parser.add_argument("--report", help="write the JSON report here")
    args = parser.parse_args(argv)
    report = export(args)
    if args.report:
        path = pathlib.Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
