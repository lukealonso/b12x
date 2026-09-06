"""EXL3-to-b12x_trellis exporter: elections, losslessness, fail-closed.

The exporter under test is ``scripts/export_exl3_to_b12x_trellis.py``.
Sources are synthesized as whole-matrix exllamav3-family checkpoints with
per-expert or layer-shared hidden-axis vectors; the emitted configuration
block is asserted through the in-tree ``TrellisConfig`` parser. All
tests are host-side.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from b12x.moe._shared.trellis_codebooks import MCG_MULTIPLIER
from b12x.moe.fused_moe.config import RateGranularity, TrellisConfig

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "export_exl3_to_b12x_trellis",
    _REPO / "scripts" / "export_exl3_to_b12x_trellis.py",
)
exporter = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("export_exl3_to_b12x_trellis", exporter)
_SPEC.loader.exec_module(exporter)

_HIDDEN = 64
_INTERMEDIATE = 512
_EXPERTS = 3
_MOE_LAYERS = (1, 2)
_TRIPLES = ((3, 4, 5), (3, 3, 3), (4, 4, 3))


def _mcg() -> torch.Tensor:
    return torch.tensor(MCG_MULTIPLIER, dtype=torch.int64).to(torch.int32)


def _write_source(root: pathlib.Path, *, layout: str = "per_expert") -> dict:
    """Write a synthetic source; return its tensors keyed for assertions."""

    generator = torch.Generator().manual_seed(20260824)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": _HIDDEN,
                "moe_intermediate_size": _INTERMEDIATE,
                "n_routed_experts": _EXPERTS,
                "first_k_dense_replace": _MOE_LAYERS[0],
                "num_hidden_layers": _MOE_LAYERS[-1] + 1,
            }
        )
    )

    def _values(width: int) -> torch.Tensor:
        raw = torch.rand((width,), generator=generator, dtype=torch.float32)
        return (0.5 + raw).to(torch.float16)

    recorded: dict = {}
    for layer in _MOE_LAYERS:
        prefix = f"model.layers.{layer}.mlp.experts"
        tensors: dict[str, torch.Tensor] = {}
        if layout == "shared":
            tensors[f"{prefix}.shared_vectors.gate_up_suh"] = _values(_HIDDEN)
            tensors[f"{prefix}.shared_vectors.down_svh"] = _values(_HIDDEN)
        for expert, triple in enumerate(_TRIPLES):
            gate_up_suh = _values(_HIDDEN)
            for pi, (proj, bits) in enumerate(
                zip(("gate_proj", "up_proj", "down_proj"), triple, strict=True)
            ):
                fc1 = proj != "down_proj"
                base = f"{prefix}.{expert}.{proj}"
                shape = (
                    (_HIDDEN // 16, _INTERMEDIATE // 16, 16 * bits)
                    if fc1
                    else (_INTERMEDIATE // 16, _HIDDEN // 16, 16 * bits)
                )
                tensors[f"{base}.trellis"] = torch.randint(
                    -(1 << 15),
                    1 << 15,
                    shape,
                    dtype=torch.int16,
                    generator=generator,
                )
                tensors[f"{base}.mcg"] = _mcg()
                if fc1:
                    ivec = _values(_INTERMEDIATE)
                    tensors[f"{base}.svh"] = ivec
                    if layout == "per_expert":
                        tensors[f"{base}.suh"] = gate_up_suh.clone()
                else:
                    ivec = _values(_INTERMEDIATE)
                    tensors[f"{base}.suh"] = ivec
                    if layout == "per_expert":
                        tensors[f"{base}.svh"] = _values(_HIDDEN)
                recorded[(layer, expert, pi)] = ivec
        save_file(tensors, str(root / f"experts-layer-{layer:03d}.safetensors"))
        for name, tensor in tensors.items():
            recorded[name] = tensor
    return recorded


def _run(source: pathlib.Path, output: pathlib.Path, *extra: str) -> dict:
    report_path = output / "report.json"
    code = exporter.main(
        [
            "--model-dir",
            str(source),
            "--output",
            str(output),
            "--report",
            str(report_path),
            *extra,
        ]
    )
    assert code == 0
    return json.loads(report_path.read_text())


def _emitted_block(output: pathlib.Path) -> dict:
    data = json.loads((output / "quantization_config.b12x_trellis.json").read_text())
    return data["quantization_config"]["b12x_trellis"]


@pytest.mark.parametrize("layout", ("per_expert", "shared"))
def test_export_elects_granularities_and_verifies(tmp_path, layout) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    recorded = _write_source(source, layout=layout)
    report = _run(source, output)

    assert report["verification"]["result"] == "element-identical"
    assert report["verification"]["tensors_reread_element_identical"] == 4
    assert report["elections"]["intermediate_scales"] == "per_expert"
    expected_hidden = "per_layer" if layout == "shared" else "per_expert"
    assert report["elections"]["input_scales"] == expected_hidden
    assert report["elections"]["output_scales"] == expected_hidden
    assert report["gate_up_divergent_assignments"] == 2  # expert 0, both layers

    # The emitted block parses through the in-tree contract and declares
    # the model-level rate shape the tensor actually has.
    parsed = TrellisConfig.from_dict(_emitted_block(output))
    assert parsed.rate.granularity is RateGranularity.PER_EXPERT_PROJECTION
    with safe_open(str(output / "b12x-trellis.safetensors"), framework="pt") as handle:
        rate = handle.get_tensor("b12x_trellis.rate")
        inter = handle.get_tensor("b12x_trellis.intermediate_scales")
    assert tuple(rate.shape) == parsed.rate.tensor_shape(
        num_layers=len(_MOE_LAYERS),
        num_experts=_EXPERTS,
        intermediate_size=_INTERMEDIATE,
    )
    for row, layer in enumerate(_MOE_LAYERS):
        for expert, triple in enumerate(_TRIPLES):
            for pi, bits in enumerate(triple):
                assert int(rate[row, expert, pi]) == (bits << 4) | bits
                assert torch.equal(
                    inter[row, expert, pi], recorded[(layer, expert, pi)]
                )


def test_export_rejects_divergent_gate_up_input_vectors(tmp_path) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout="per_expert")
    shard = source / "experts-layer-001.safetensors"
    with safe_open(str(shard), framework="pt") as handle:
        # Clone: get_tensor returns file-backed views and the shard is
        # rewritten in place below.
        tensors = {name: handle.get_tensor(name).clone() for name in handle.keys()}
    name = "model.layers.1.mlp.experts.0.up_proj.suh"
    tensors[name] = tensors[name] + 1.0
    save_file(tensors, str(shard))
    with pytest.raises(SystemExit, match="gate and up"):
        _run(source, output)


def test_export_rejects_rank_sliced_payloads(tmp_path) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout="per_expert")
    save_file(
        {
            "model.layers.1.mlp.experts.0.gate_proj.rank0.trellis": (
                torch.zeros((4, 8, 48), dtype=torch.int16)
            )
        },
        str(source / "sliced.safetensors"),
    )
    with pytest.raises(SystemExit, match="rank-sliced"):
        _run(source, output)


def test_export_rejects_foreign_mcg_multiplier(tmp_path) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout="per_expert")
    shard = source / "experts-layer-001.safetensors"
    with safe_open(str(shard), framework="pt") as handle:
        # Clone: get_tensor returns file-backed views and the shard is
        # rewritten in place below.
        tensors = {name: handle.get_tensor(name).clone() for name in handle.keys()}
    tensors["model.layers.1.mlp.experts.0.gate_proj.mcg"] = torch.tensor(
        1234567, dtype=torch.int32
    )
    save_file(tensors, str(shard))
    with pytest.raises(SystemExit, match="MCG multiplier"):
        _run(source, output)


def test_export_cross_checks_sidecar_bit_maps(tmp_path) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout="per_expert")
    sidecar = {
        "bit_map": {"model.layers.1.mlp.experts.0.gate_proj": 5}
    }  # tensor carries K3
    (source / "experts-layer-001.json").write_text(json.dumps(sidecar))
    with pytest.raises(SystemExit, match="sidecar"):
        _run(source, output)

    sidecar["bit_map"]["model.layers.1.mlp.experts.0.gate_proj"] = 3
    (source / "experts-layer-001.json").write_text(json.dumps(sidecar))
    report = _run(source, output)
    assert report["checks"]["sidecar_bit_map_layers"] == 1


def test_export_rejects_duplicate_shared_vectors(tmp_path) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout="shared")
    save_file(
        {
            "model.layers.1.mlp.experts.other_shared.gate_up_suh": (
                torch.ones((_HIDDEN,), dtype=torch.float16)
            )
        },
        str(source / "zz-extra-shared.safetensors"),
    )
    with pytest.raises(SystemExit, match="resolves to two tensors"):
        _run(source, output)


def test_export_rejects_conflicting_multi_shard_sidecars(tmp_path) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout="per_expert")
    # Move one expert's tensors for layer 1 into a second shard with its
    # own sidecar whose bit map conflicts with the first shard's.
    shard = source / "experts-layer-001.safetensors"
    with safe_open(str(shard), framework="pt") as handle:
        # Clone: get_tensor returns file-backed views and the shard is
        # rewritten in place below.
        tensors = {name: handle.get_tensor(name).clone() for name in handle.keys()}
    # Clone: get_tensor returns file-backed views, and the shard is
    # rewritten below while `moved` is still needed.
    moved = {
        name: tensors.pop(name).clone()
        for name in list(tensors)
        if ".experts.2." in name
    }
    save_file(tensors, str(shard))
    save_file(moved, str(source / "experts-layer-001-extra.safetensors"))
    key = "model.layers.1.mlp.experts.0.gate_proj"
    (source / "experts-layer-001.json").write_text(json.dumps({"bit_map": {key: 3}}))
    (source / "experts-layer-001-extra.json").write_text(
        json.dumps({"bit_map": {key: 5}})
    )
    with pytest.raises(SystemExit, match="conflicting sidecar bit maps"):
        _run(source, output)

    # Consistent maps across both shards merge and pass.
    (source / "experts-layer-001-extra.json").write_text(
        json.dumps({"bit_map": {key: 3}})
    )
    report = _run(source, output)
    assert report["checks"]["sidecar_bit_map_layers"] == 1


def test_export_rejects_lossy_scale_dtypes(tmp_path) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout="per_expert")
    shard = source / "experts-layer-001.safetensors"
    with safe_open(str(shard), framework="pt") as handle:
        # Clone: get_tensor returns file-backed views and the shard is
        # rewritten in place below.
        tensors = {name: handle.get_tensor(name).clone() for name in handle.keys()}
    name = "model.layers.1.mlp.experts.0.gate_proj.svh"
    lossy = torch.full((_INTERMEDIATE,), 1.0 / 3.0, dtype=torch.float32)
    tensors[name] = lossy
    save_file(tensors, str(shard))
    with pytest.raises(SystemExit, match="not exactly representable"):
        _run(source, output)

    # Exactly representable wider dtypes convert without loss.
    tensors[name] = torch.full((_INTERMEDIATE,), 1.5, dtype=torch.float32)
    save_file(tensors, str(shard))
    report = _run(source, output)
    assert report["verification"]["result"] == "element-identical"


def test_export_rejects_partial_checkpoints(tmp_path) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout="per_expert")
    (source / "experts-layer-002.safetensors").unlink()
    with pytest.raises(SystemExit, match="partial checkpoint"):
        _run(source, output)


@pytest.mark.parametrize(
    ("block_size", "message"),
    (
        ("48", "power of two"),
        ("16", "power of two"),
        ("0", "power of two"),
        ("-128", "power of two"),
        ("2048", "does not divide"),
    ),
)
def test_export_validates_block_size(tmp_path, block_size, message) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout="per_expert")
    with pytest.raises(SystemExit, match=message):
        _run(source, output, "--block-size", block_size)


def test_export_ignores_out_of_scope_layers(tmp_path) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout="per_expert")
    save_file(
        {
            "model.layers.9.mlp.experts.0.gate_proj.rank0.trellis": (
                torch.zeros((4, 8, 48), dtype=torch.int16)
            )
        },
        str(source / "draft-layer.safetensors"),
    )
    report = _run(source, output)
    assert report["ignored_out_of_scope_tensors"] == 1
