#!/usr/bin/env python3
"""Benchmark real Kimi-K3 QSRT checkpoint profiles on one tensor-parallel rank.

The benchmark compares the complete fused W4A16 MoE path for:

* uniform K2 with coupled activation-boundary Hadamard transforms; and
* 3.08-bpw K3/K4 coding with ordinary per-matrix Hadamard transforms.

The default run loads both profiles from their canonical tensor-parallel-
independent atom slabs. ``--profiles`` can select either profile alone so two
clean source revisions can be measured independently with the same input,
routes, weights, layer, TP rank, warmup, and CUDA-graph timing protocol.
Checkpoint loading and rank-local preparation are outside the timed region.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safetensors import safe_open
import torch

from benchmarks.common import (
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
    resolve_l2_flush_bytes,
)
from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
from b12x.moe._shared.kernels.w4a16.btx import prepare_btx_moe_weights
from b12x.moe._shared.kernels.w4a16.btx_compat import (
    lift_qsrt_atoms_v2_extent,
)


_RESULT_KIND = "b12x_qsrt_real_checkpoint_profile_benchmark"
_RESULT_SCHEMA_VERSION = 2

_DEFAULT_K2_ROOT = Path("/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-v1")
_DEFAULT_H308_ROOT = Path("/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-3p08-v2")

_COMPLETION = "qsrt-completion.json"
_COMPLETION_KIND = "kquant_kimi_k3_qsrt_completion"
_STORAGE_SCHEMAS = frozenset(
    {
        "kquant_kimi_k3_qsrt_atoms_v2",
        "qsrt_kimi_k3_qsrt_atoms_v2",
    }
)
_ENCODING = "qsrt_sqg_e4m3"
_CODEBOOK = "sqg_xor_cheb_t12"
_VERSION = 2

_FORMAT_TENSOR = "_qsrt_format_section"
_SHARED_SCALE_TENSOR = "_qsrt_shared_scale_section"
_ATOM_TENSOR = "qsrt_atoms"
_TENSOR_INVENTORY = {
    _FORMAT_TENSOR,
    _SHARED_SCALE_TENSOR,
    _ATOM_TENSOR,
}

_EXPERTS = 896
_HIDDEN_SIZE = 3584
_GLOBAL_INTERMEDIATE_SIZE = 3072
_ATOM_CHANNELS = 32
_ATOM_SLOTS = 96
_FORMAT_SECTION_BYTES = 4096
_SHARED_SCALE_SECTION_BYTES = 24576
_SHARED_SCALE_BYTES = 3 * _HIDDEN_SIZE * torch.float16.itemsize
_ROTATION_DRAW_OFFSET = _EXPERTS
_TOPK = 16
_ACTIVATION = "situ"
_SUPPORTED_TP_SIZES = (8, 12)
_DEFAULT_TP_SIZE = 12


@dataclass(frozen=True)
class ProfileContract:
    label: str
    profile: str
    profile_id: int
    trellis_bits: int
    atom_slot_stride_bytes: int
    atom_bundle_metadata: tuple[tuple[str, int], ...]
    format_code: int
    tile_config: tuple[int, int, int, int]
    coupled_hadamard: bool
    supported_tp_sizes: tuple[int, ...]


_K2 = ProfileContract(
    label="uniform_k2_coupled",
    profile="k2_coupled_h512_h128",
    profile_id=3,
    trellis_bits=2,
    atom_slot_stride_bytes=77_242_368,
    atom_bundle_metadata=(("p22_atom_bundle_bytes", 86_208),),
    format_code=0x44,
    tile_config=(128, 128, 128, 128),
    coupled_hadamard=True,
    supported_tp_sizes=(8, 12),
)
_H308 = ProfileContract(
    label="legacy_3p08_k34",
    profile="k3x22_k4x2",
    profile_id=2,
    trellis_bits=3,
    atom_slot_stride_bytes=119_005_184,
    atom_bundle_metadata=(
        ("p33_atom_bundle_bytes", 129_216),
        ("p43_atom_bundle_bytes", 150_720),
    ),
    format_code=0x33,
    tile_config=(64, 256, 64, 256),
    coupled_hadamard=False,
    supported_tp_sizes=(12,),
)
_CONTRACTS = (_K2, _H308)
_CONTRACT_BY_LABEL = {contract.label: contract for contract in _CONTRACTS}


@dataclass(frozen=True)
class LayerSource:
    root: Path
    path: Path
    layer: int
    contract: ProfileContract
    completion_sha256: str
    candidate_pool_content_sha256: str
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    down_svh: torch.Tensor
    rotation_draws: torch.Tensor | None


@dataclass
class CapturedCase:
    graph: torch.cuda.CUDAGraph
    eager_output: torch.Tensor
    captured_output: torch.Tensor
    keepalive: list[object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "b12x/moe/_shared/execution.py",
        root / "b12x/moe/_shared/kernels/w4a16/host.py",
        root / "b12x/moe/_shared/kernels/w4a16/kernel.py",
        root / "b12x/moe/_shared/kernels/w4a16/prepare.py",
        root / "b12x/moe/fused_moe/_impl.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args),
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _parse_positive_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of integers"
        ) from error
    if (
        not result
        or any(item <= 0 for item in result)
        or len(set(result)) != len(result)
    ):
        raise argparse.ArgumentTypeError("batch sizes must be unique positive integers")
    return result


def _validate_profile_tp_size(
    contracts: Sequence[ProfileContract], tp_size: int
) -> None:
    incompatible = [
        contract
        for contract in contracts
        if tp_size not in contract.supported_tp_sizes
    ]
    if incompatible:
        details = ", ".join(
            f"{contract.label} supports {contract.supported_tp_sizes}"
            for contract in incompatible
        )
        raise ValueError(f"tensor-parallel size {tp_size} is incompatible: {details}")


def _balanced_atom_partition(shard_count: int, shard_index: int) -> tuple[int, int]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid atom-shard identity")
    base, remainder = divmod(_ATOM_SLOTS, shard_count)
    rows = base + int(shard_index < remainder)
    first = shard_index * base + min(shard_index, remainder)
    return first, rows


def _metadata_int(metadata: dict[str, str], name: str) -> int:
    try:
        return int(metadata[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"QSRT metadata {name!r} is missing or invalid") from error


def _validate_completion(
    root: Path, contract: ProfileContract, layer: int
) -> tuple[Path, dict[str, object]]:
    root = root.expanduser().resolve()
    completion_path = root / _COMPLETION
    try:
        completion = json.loads(completion_path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(completion_path) from None
    except json.JSONDecodeError as error:
        raise ValueError(
            f"malformed QSRT completion file: {completion_path}"
        ) from error
    if not isinstance(completion, dict):
        raise ValueError("QSRT completion document must be a JSON object")
    expected = {
        "kind": _COMPLETION_KIND,
        "schema_version": _VERSION,
        "profile": contract.profile,
        "complete": True,
        "layer_count": 92,
    }
    for name, value in expected.items():
        if completion.get(name) != value:
            raise ValueError(
                f"QSRT completion {name!r} mismatch: "
                f"{completion.get(name)!r} != {value!r}"
            )
    if completion.get("storage_schema") not in _STORAGE_SCHEMAS:
        raise ValueError(
            "QSRT completion 'storage_schema' must be one of "
            f"{sorted(_STORAGE_SCHEMAS)!r}, got "
            f"{completion.get('storage_schema')!r}"
        )
    candidate_hash = completion.get("candidate_pool_content_sha256")
    if (
        not isinstance(candidate_hash, str)
        or len(candidate_hash) != 64
        or any(character not in "0123456789abcdef" for character in candidate_hash)
    ):
        raise ValueError("QSRT completion has an invalid candidate-pool hash")
    layers = completion.get("layers")
    entry = layers.get(str(layer)) if isinstance(layers, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"QSRT completion omits layer {layer}")
    filename = entry.get("file")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or filename != f"qsrt-layer-{layer:05d}.safetensors"
    ):
        raise ValueError(f"QSRT completion has an invalid layer-{layer} filename")
    path = root / filename
    expected_bytes = entry.get("bytes")
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError(
            f"QSRT layer-{layer} file size does not match its completion record"
        )
    return path, {
        "root": str(root),
        "completion": str(completion_path),
        "completion_sha256": _sha256(completion_path),
        "candidate_pool_content_sha256": candidate_hash,
        "layer_file": str(path),
        "layer_file_bytes": path.stat().st_size,
    }


def _read_layer_source(
    root: Path, contract: ProfileContract, layer: int
) -> tuple[LayerSource, dict[str, object]]:
    path, provenance = _validate_completion(root, contract, layer)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if set(handle.keys()) != _TENSOR_INVENTORY:
            raise ValueError("QSRT layer tensor inventory is noncanonical")
        expected_metadata: dict[str, str] = {
            "format": "pt",
            "version": str(_VERSION),
            "encoding": _ENCODING,
            "codebook": _CODEBOOK,
            "profile": contract.profile,
            "profile_id": str(contract.profile_id),
            "layer": str(layer),
            "experts": str(_EXPERTS),
            "intermediate_channels": str(_GLOBAL_INTERMEDIATE_SIZE),
            "latent_channels": str(_HIDDEN_SIZE),
            "atom_channels": str(_ATOM_CHANNELS),
            "atom_slots": str(_ATOM_SLOTS),
            "atom_slot_stride_bytes": str(contract.atom_slot_stride_bytes),
            "alignment_bytes": "4096",
        }
        expected_metadata.update(
            {name: str(value) for name, value in contract.atom_bundle_metadata}
        )
        if metadata.get("schema") not in _STORAGE_SCHEMAS:
            raise ValueError(
                "QSRT layer metadata 'schema' must be one of "
                f"{sorted(_STORAGE_SCHEMAS)!r}, got {metadata.get('schema')!r}"
            )
        if contract.coupled_hadamard:
            expected_metadata.update(
                {
                    "residual_hadamard_block_size": "512",
                    "preactivation_hadamard_block_size": "128",
                    "postactivation_hadamard_block_size": "128",
                    "intermediate_rotation_draws": "format_section[896:1792]",
                }
            )
        for name, value in expected_metadata.items():
            if metadata.get(name) != value:
                raise ValueError(
                    f"QSRT layer metadata {name!r} mismatch: "
                    f"{metadata.get(name)!r} != {value!r}"
                )
        if _metadata_int(metadata, "atom_slot_stride_bytes") != (
            contract.atom_slot_stride_bytes
        ):
            raise AssertionError("validated atom stride changed unexpectedly")

        format_section = handle.get_tensor(_FORMAT_TENSOR)
        padding_begin = _EXPERTS
        rotation_draws = None
        if contract.coupled_hadamard:
            padding_begin = _ROTATION_DRAW_OFFSET + _EXPERTS
            rotation_draws = format_section[
                _ROTATION_DRAW_OFFSET : _ROTATION_DRAW_OFFSET + _EXPERTS
            ].clone()
            if bool(torch.any(rotation_draws > 7)):
                raise ValueError("uniform-K2 layer contains an invalid rotation draw")
        if (
            format_section.dtype != torch.uint8
            or tuple(format_section.shape) != (_FORMAT_SECTION_BYTES,)
            or bool(torch.any(format_section[:_EXPERTS] != contract.format_code))
            or bool(torch.any(format_section[padding_begin:] != 0))
        ):
            raise ValueError("QSRT format section is malformed")

        shared = handle.get_tensor(_SHARED_SCALE_TENSOR)
        if (
            shared.dtype != torch.uint8
            or tuple(shared.shape) != (_SHARED_SCALE_SECTION_BYTES,)
            or bool(torch.any(shared[_SHARED_SCALE_BYTES:] != 0))
        ):
            raise ValueError("QSRT shared-scale section is malformed")
        scales = (
            shared[:_SHARED_SCALE_BYTES]
            .clone()
            .view(torch.float16)
            .reshape(3, _HIDDEN_SIZE)
            .contiguous()
        )
        if not bool(torch.all(torch.isfinite(scales))):
            raise ValueError("QSRT shared scales contain non-finite values")
        atom_shape = handle.get_slice(_ATOM_TENSOR).get_shape()
        expected_shape = [_ATOM_SLOTS, contract.atom_slot_stride_bytes]
        if atom_shape != expected_shape:
            raise ValueError(f"QSRT atom shape {atom_shape} != {expected_shape}")

    return (
        LayerSource(
            root=root.expanduser().resolve(),
            path=path,
            layer=layer,
            contract=contract,
            completion_sha256=str(provenance["completion_sha256"]),
            candidate_pool_content_sha256=str(
                provenance["candidate_pool_content_sha256"]
            ),
            gate_suh=scales[0].contiguous(),
            up_suh=scales[1].contiguous(),
            down_svh=scales[2].contiguous(),
            rotation_draws=rotation_draws,
        ),
        provenance,
    )


def _read_atom_extent(
    source: LayerSource, *, tp_size: int, tp_rank: int
) -> tuple[int, torch.Tensor]:
    first, rows = _balanced_atom_partition(tp_size, tp_rank)
    if _ATOM_SLOTS % tp_size or first % rows:
        raise AssertionError("tensor-parallel ranks must own equal atom extents")
    with safe_open(source.path, framework="pt", device="cpu") as handle:
        atoms = handle.get_slice(_ATOM_TENSOR)[first : first + rows].contiguous()
    if atoms.dtype != torch.uint8 or tuple(atoms.shape) != (
        rows,
        source.contract.atom_slot_stride_bytes,
    ):
        raise ValueError("loaded QSRT atom extent has the wrong tensor contract")
    return first, atoms


def _prepare_profile(
    source: LayerSource,
    *,
    tp_size: int,
    tp_rank: int,
    device: torch.device,
):
    started = time.perf_counter()
    first_atom_slot, atoms = _read_atom_extent(
        source, tp_size=tp_size, tp_rank=tp_rank
    )
    atom_slots = int(atoms.shape[0])
    read_seconds = time.perf_counter() - started
    extent_bytes = atoms.numel() * atoms.element_size()
    print(
        f"loaded {source.contract.label}: layer={source.layer} "
        f"atoms={first_atom_slot}:{first_atom_slot + atom_slots} "
        f"bytes={extent_bytes / (1 << 20):.1f} MiB "
        f"read={read_seconds:.2f}s"
    )

    started = time.perf_counter()
    btx_layer = lift_qsrt_atoms_v2_extent(
        atoms,
        profile=source.contract.profile,
        first_atom_slot=first_atom_slot,
        layer_index=source.layer,
        hidden_size=_HIDDEN_SIZE,
        global_intermediate_size=96 * _ATOM_CHANNELS,
        num_experts=_EXPERTS,
        gate_suh=source.gate_suh,
        up_suh=source.up_suh,
        down_svh=source.down_svh,
        rotation_draws=(
            source.rotation_draws
            if source.contract.coupled_hadamard
            else None
        ),
    )
    prepared = prepare_btx_moe_weights(
        btx_layer,
        activation=_ACTIVATION,
        device=device,
        tile_config=source.contract.tile_config,
    )
    torch.cuda.synchronize(device)
    prepare_seconds = time.perf_counter() - started
    if bool(prepared.coupled_hadamard) != source.contract.coupled_hadamard:
        raise RuntimeError("prepared transform policy disagrees with the checkpoint")
    del atoms
    gc.collect()
    print(
        f"prepared {source.contract.label}: "
        f"tiles={source.contract.tile_config} prepare={prepare_seconds:.2f}s"
    )
    return prepared, {
        "first_atom_slot": first_atom_slot,
        "atom_slots": atom_slots,
        "extent_bytes": extent_bytes,
        "read_seconds": read_seconds,
        "prepare_seconds": prepare_seconds,
        "tile_config": list(source.contract.tile_config),
        "coupled_hadamard": source.contract.coupled_hadamard,
    }


def _make_inputs(
    tokens: int, *, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = (
        torch.randn(
            (tokens, _HIDDEN_SIZE),
            generator=generator,
            dtype=torch.float32,
        )
        * 1.0e-3
    ).to(device=device, dtype=torch.bfloat16)
    logits = torch.randn((tokens, _EXPERTS), generator=generator, dtype=torch.float32)
    topk_logits, topk_ids = torch.topk(logits, _TOPK, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    return (
        source.contiguous(),
        topk_ids.to(device=device, dtype=torch.int32).contiguous(),
        topk_weights.to(device=device, dtype=torch.float32).contiguous(),
    )


def _capture_case(
    prepared,
    *,
    source: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> CapturedCase:
    tokens = int(source.shape[0])
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=tokens,
        topk=_TOPK,
        dtype=torch.float16,
        device=source.device,
        full_rotation=True,
        block_size_m=8,
    )

    def run() -> torch.Tensor:
        return run_w4a16_moe(
            source,
            prepared,
            topk_weights,
            topk_ids,
            activation=_ACTIVATION,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_counts=buffers.expert_counts,
            route_block_size_m=8,
            intermediate_rotation_scales=prepared.intermediate_rotations,
            full_rotation=True,
            suh_gate_table=prepared.gate_suh,
            suh_up_table=prepared.up_suh,
            svh_table=prepared.down_svh,
            rotation_a_gate=buffers.rotation_a_gate,
            rotation_a_up=buffers.rotation_a_up,
        )

    # Complete lazy compilation and cache population before capturing or timing.
    run()
    eager = run().clone()
    torch.cuda.synchronize(source.device)
    if not bool(torch.all(torch.isfinite(eager))):
        raise RuntimeError("QSRT eager execution produced a non-finite output")
    if not bool(torch.any(eager != 0)):
        raise RuntimeError("QSRT eager execution produced an all-zero output")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize(source.device)
    if not torch.equal(captured, eager):
        raise RuntimeError("QSRT CUDA-graph replay differs from eager execution")
    return CapturedCase(
        graph=graph,
        eager_output=eager,
        captured_output=captured,
        keepalive=[
            buffers,
            source,
            topk_ids,
            topk_weights,
            run,
        ],
    )


def _interleaved_times_us(
    graphs: dict[str, torch.cuda.CUDAGraph],
    *,
    replays: int,
    flush_l2: Callable[[], None] | None,
) -> dict[str, list[float]]:
    samples = {name: [] for name in graphs}
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    names = list(graphs)
    for replay in range(replays):
        order = names if replay % 2 == 0 else list(reversed(names))
        for name in order:
            if flush_l2 is not None:
                flush_l2()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            events.append((name, start, end))
    torch.cuda.synchronize()
    for name, start, end in events:
        samples[name].append(start.elapsed_time(end) * 1000.0)
    return samples


def _summary(samples: Sequence[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90_us": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _paired_ratio_bootstrap(
    samples: Sequence[float],
    baseline: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    if len(samples) != len(baseline) or not samples:
        raise ValueError("paired samples must have the same nonzero length")
    sample_tensor = torch.tensor(samples, dtype=torch.float64)
    baseline_tensor = torch.tensor(baseline, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        len(samples),
        (replicates, len(samples)),
        dtype=torch.int64,
        generator=generator,
    )
    ratios = torch.quantile(sample_tensor[indices], 0.5, dim=1) / torch.quantile(
        baseline_tensor[indices], 0.5, dim=1
    )
    bounds = torch.quantile(ratios, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "point_estimate": statistics.median(samples) / statistics.median(baseline),
        "bootstrap_ci95_low": float(bounds[0]),
        "bootstrap_ci95_high": float(bounds[1]),
        "bootstrap_replicates": replicates,
    }


def _output_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_f32 = left.float()
    right_f32 = right.float()
    difference = left_f32 - right_f32
    denominator = torch.linalg.vector_norm(right_f32).clamp_min(1.0e-20)
    cosine = torch.nn.functional.cosine_similarity(
        left_f32.reshape(1, -1), right_f32.reshape(1, -1)
    )
    return {
        "relative_l2_to_legacy": float(
            torch.linalg.vector_norm(difference) / denominator
        ),
        "cosine_similarity": float(cosine[0]),
    }


def _write_new_result(path: Path, payload: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("x") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k2-checkpoint", type=Path, default=_DEFAULT_K2_ROOT)
    parser.add_argument("--legacy-checkpoint", type=Path, default=_DEFAULT_H308_ROOT)
    parser.add_argument(
        "--profiles",
        choices=tuple(_CONTRACT_BY_LABEL),
        nargs="+",
        default=[contract.label for contract in _CONTRACTS],
        help="checkpoint profiles to measure; select one for implementation A/B runs",
    )
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument(
        "--tp-size",
        type=int,
        choices=_SUPPORTED_TP_SIZES,
        default=_DEFAULT_TP_SIZE,
    )
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--tokens",
        type=_parse_positive_list,
        default=_parse_positive_list("1,2,4,8"),
        help="comma-separated routed-token counts",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--cold-replays", type=int, default=40)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile-kernels",
        action="store_true",
        help="print one CUDA-profiler table per profile at the first token count",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if len(set(args.profiles)) != len(args.profiles):
        raise SystemExit("--profiles entries must be unique")
    contracts = tuple(_CONTRACT_BY_LABEL[label] for label in args.profiles)
    try:
        _validate_profile_tp_size(contracts, args.tp_size)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.output is not None and args.output.expanduser().exists():
        raise SystemExit(f"refusing to overwrite benchmark result: {args.output}")
    if not 1 <= args.layer <= 92:
        raise SystemExit("--layer must lie in 1..92")
    if not 0 <= args.tp_rank < args.tp_size:
        raise SystemExit(f"--tp-rank must lie in 0..{args.tp_size - 1}")
    if (
        args.warmup < 1
        or args.replays < 1
        or args.cold_replays < 0
        or args.bootstrap_replicates < 1
        or args.l2_flush_bytes < 0
    ):
        raise SystemExit(
            "warmup, replays, and bootstrap replicates must be positive; "
            "cold replays and L2 flush bytes must be non-negative"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    if major != 12 or minor not in (0, 1):
        raise SystemExit(f"SM120/SM121 is required, got SM{major}{minor}")

    torch.manual_seed(args.seed)
    profile_roots = {
        _K2.label: args.k2_checkpoint,
        _H308.label: args.legacy_checkpoint,
    }
    layer_sources: dict[str, LayerSource] = {}
    artifact_provenance: dict[str, object] = {}
    for contract in contracts:
        source, provenance = _read_layer_source(
            profile_roots[contract.label], contract, args.layer
        )
        layer_sources[contract.label] = source
        artifact_provenance[contract.label] = provenance

    prepared_profiles: dict[str, object] = {}
    preparation: dict[str, object] = {}
    for contract in contracts:
        prepared, details = _prepare_profile(
            layer_sources[contract.label],
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            device=device,
        )
        prepared_profiles[contract.label] = prepared
        preparation[contract.label] = details

    properties = torch.cuda.get_device_properties(device)
    result: dict[str, object] = {
        "kind": _RESULT_KIND,
        "schema_version": _RESULT_SCHEMA_VERSION,
        "complete": False,
        "provenance": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "commit": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "source_sha256": _source_sha256(),
            "worktree": str(Path(__file__).resolve().parents[1]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "environment": {
                "B12X_SQG_XOR_CHEB_T12_DIRECT_SMEM": os.environ.get(
                    "B12X_SQG_XOR_CHEB_T12_DIRECT_SMEM"
                ),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "artifacts": artifact_provenance,
        },
        "contract": {
            "model": "Kimi-K3",
            "layer": args.layer,
            "tp_size": args.tp_size,
            "tp_rank": args.tp_rank,
            "hidden_size": _HIDDEN_SIZE,
            "global_intermediate_size": _GLOBAL_INTERMEDIATE_SIZE,
            "local_intermediate_size": _GLOBAL_INTERMEDIATE_SIZE // args.tp_size,
            "num_experts": _EXPERTS,
            "top_k": _TOPK,
            "activation": _ACTIVATION,
            "activation_dtype": "bfloat16",
            "weight_path": "W4A16",
            "execution": "complete fused route/FC1/SiTU/FC2/reduction",
            "timing": "interleaved CUDA graph replay",
            "correctness": "finite nonzero eager output and exact eager/graph replay",
            "shared_inputs_and_routes": True,
            "profile_contracts": {
                contract.label: {
                    "profile": contract.profile,
                    "trellis_bits": contract.trellis_bits,
                    "tile_config": list(contract.tile_config),
                    "coupled_hadamard": contract.coupled_hadamard,
                    "supported_tp_sizes": list(contract.supported_tp_sizes),
                }
                for contract in contracts
            },
        },
        "hardware": {
            "device": str(device),
            "name": properties.name,
            "sm": f"{major}{minor}",
            "total_memory_bytes": properties.total_memory,
            "gpu_mode_before": nvidia_smi_gpu_mode_snapshot(),
        },
        "parameters": {
            "tokens": list(args.tokens),
            "profiles": [contract.label for contract in contracts],
            "warmup": args.warmup,
            "hot_replays": args.replays,
            "cold_replays": args.cold_replays,
            "bootstrap_replicates": args.bootstrap_replicates,
            "l2_flush_bytes_hint": args.l2_flush_bytes,
            "seed": args.seed,
        },
        "preparation": preparation,
        "cases": [],
    }

    cold_flush = make_l2_flush_fn(args.cold_replays > 0, bytes_hint=args.l2_flush_bytes)
    l2_flush_bytes = (
        resolve_l2_flush_bytes(args.l2_flush_bytes) if args.cold_replays else 0
    )
    print(
        f"device={properties.name} layer={args.layer} tp={args.tp_size} "
        f"rank={args.tp_rank} H={_HIDDEN_SIZE} "
        f"I_local={_GLOBAL_INTERMEDIATE_SIZE // args.tp_size} "
        f"E={_EXPERTS} topk={_TOPK}"
    )
    compare_profiles = len(contracts) == 2
    if compare_profiles:
        print(
            "tokens  cache  uniform-K2 coupled   legacy 3.08 K3/K4   "
            "K2/legacy   legacy/K2"
        )
    else:
        print(f"tokens  cache  {contracts[0].label}")

    for case_index, tokens in enumerate(args.tokens):
        source, topk_ids, topk_weights = _make_inputs(
            tokens, seed=args.seed + 1009 * tokens, device=device
        )
        captured: dict[str, CapturedCase] = {}
        for contract in contracts:
            captured[contract.label] = _capture_case(
                prepared_profiles[contract.label],
                source=source,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )
        graphs = {name: value.graph for name, value in captured.items()}
        for warmup in range(args.warmup):
            order = list(graphs) if warmup % 2 == 0 else list(reversed(graphs))
            for name in order:
                graphs[name].replay()
        torch.cuda.synchronize(device)

        hot = _interleaved_times_us(graphs, replays=args.replays, flush_l2=None)
        cold = (
            _interleaved_times_us(
                graphs, replays=args.cold_replays, flush_l2=cold_flush
            )
            if args.cold_replays
            else None
        )
        hot_ratio = (
            _paired_ratio_bootstrap(
                hot[_K2.label],
                hot[_H308.label],
                replicates=args.bootstrap_replicates,
                seed=args.seed + case_index,
            )
            if compare_profiles
            else None
        )
        cold_ratio = (
            _paired_ratio_bootstrap(
                cold[_K2.label],
                cold[_H308.label],
                replicates=args.bootstrap_replicates,
                seed=args.seed + 100 + case_index,
            )
            if compare_profiles and cold is not None
            else None
        )
        if compare_profiles:
            k2_median = statistics.median(hot[_K2.label])
            h308_median = statistics.median(hot[_H308.label])
            print(
                f"{tokens:>6}  hot   {k2_median:>13.3f} us   "
                f"{h308_median:>13.3f} us   "
                f"{k2_median / h308_median:>8.4f}x   "
                f"{h308_median / k2_median:>8.4f}x"
            )
            if cold is not None:
                k2_cold_median = statistics.median(cold[_K2.label])
                h308_cold_median = statistics.median(cold[_H308.label])
                print(
                    f"{tokens:>6}  cold  {k2_cold_median:>13.3f} us   "
                    f"{h308_cold_median:>13.3f} us   "
                    f"{k2_cold_median / h308_cold_median:>8.4f}x   "
                    f"{h308_cold_median / k2_cold_median:>8.4f}x"
                )
        else:
            label = contracts[0].label
            print(
                f"{tokens:>6}  hot   {statistics.median(hot[label]):>13.3f} us"
            )
            if cold is not None:
                print(
                    f"{tokens:>6}  cold  "
                    f"{statistics.median(cold[label]):>13.3f} us"
                )

        case: dict[str, object] = {
            "tokens": tokens,
            "routes": {
                "unique_experts": int(torch.unique(topk_ids).numel()),
                "ids_shape": list(topk_ids.shape),
                "weights_shape": list(topk_weights.shape),
            },
            "correctness": {
                contract.label: {
                    "finite": True,
                    "nonzero": int(
                        torch.count_nonzero(captured[contract.label].eager_output)
                    ),
                    "eager_graph_exact": True,
                    "output_sha256": _tensor_sha256(
                        captured[contract.label].eager_output
                    ),
                }
                for contract in contracts
            },
            "output_comparison": (
                _output_comparison(
                    captured[_K2.label].eager_output,
                    captured[_H308.label].eager_output,
                )
                if compare_profiles
                else None
            ),
            "hot": {
                "samples_us": hot,
                "summary": {name: _summary(values) for name, values in hot.items()},
                "uniform_k2_to_legacy_time_ratio": hot_ratio,
                "legacy_over_uniform_k2_speedup": (
                    1.0 / float(hot_ratio["point_estimate"])
                    if hot_ratio is not None
                    else None
                ),
            },
            "cold": None,
        }
        if cold is not None:
            case["cold"] = {
                "samples_us": cold,
                "summary": {name: _summary(values) for name, values in cold.items()},
                "uniform_k2_to_legacy_time_ratio": cold_ratio,
                "legacy_over_uniform_k2_speedup": (
                    1.0 / float(cold_ratio["point_estimate"])
                    if cold_ratio is not None
                    else None
                ),
            }
        result["cases"].append(case)

        if args.profile_kernels and case_index == 0:
            for contract in contracts:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as profile:
                    graphs[contract.label].replay()
                    torch.cuda.synchronize(device)
                print(f"\n{contract.label} CUDA kernels")
                print(
                    profile.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=30
                    )
                )

        del captured, graphs, source, topk_ids, topk_weights
        gc.collect()

    result["hardware"]["gpu_mode_after"] = nvidia_smi_gpu_mode_snapshot()
    result["complete"] = True
    if args.output is not None:
        result["parameters"]["l2_flush_bytes"] = l2_flush_bytes
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        _write_new_result(args.output, encoded)
        print(f"wrote {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
