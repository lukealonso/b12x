from __future__ import annotations

import json

import pytest
import torch

from benchmarks.benchmark_qsrt_checkpoint_profiles import (
    _ATOM_SLOTS,
    _COMPLETION_KIND,
    _H308,
    _K2,
    _STORAGE_SCHEMAS,
    _VERSION,
    _balanced_atom_partition,
    _paired_ratio_bootstrap,
    _parse_positive_list,
    _tensor_sha256,
    _validate_completion,
    _validate_profile_tp_size,
    _write_new_result,
)


def test_profile_contracts_match_production_transform_policies() -> None:
    assert _K2.profile == "k2_coupled_h512_h128"
    assert _K2.trellis_bits == 2
    assert _K2.coupled_hadamard is True
    assert _K2.tile_config == (128, 128, 128, 128)
    assert _K2.supported_tp_sizes == (8, 12)

    assert _H308.profile == "k3x22_k4x2"
    assert _H308.trellis_bits == 3
    assert _H308.coupled_hadamard is False
    assert _H308.tile_config == (64, 256, 64, 256)
    assert _H308.supported_tp_sizes == (12,)


def test_profile_contract_rejects_unsupported_tp_size_before_loading() -> None:
    _validate_profile_tp_size((_K2,), 8)
    _validate_profile_tp_size((_K2, _H308), 12)

    with pytest.raises(ValueError, match="legacy_3p08_k34 supports \\(12,\\)"):
        _validate_profile_tp_size((_H308,), 8)


def test_tp12_atom_partition_is_complete_and_pair_aligned() -> None:
    extents = [_balanced_atom_partition(12, rank) for rank in range(12)]

    assert extents == [(8 * rank, 8) for rank in range(12)]
    assert sum(rows for _first, rows in extents) == _ATOM_SLOTS


def test_tp8_atom_partition_is_complete_and_uniform() -> None:
    extents = [_balanced_atom_partition(8, rank) for rank in range(8)]

    assert extents == [(12 * rank, 12) for rank in range(8)]
    assert sum(rows for _first, rows in extents) == _ATOM_SLOTS


def test_parse_positive_list_rejects_duplicates_and_nonpositive_values() -> None:
    assert _parse_positive_list("1,2,4,8") == (1, 2, 4, 8)
    with pytest.raises(Exception, match="unique positive"):
        _parse_positive_list("1,1")
    with pytest.raises(Exception, match="unique positive"):
        _parse_positive_list("0,1")


def test_paired_ratio_bootstrap_closes_exact_scale() -> None:
    baseline = [10.0, 11.0, 12.0, 13.0, 14.0]

    identity = _paired_ratio_bootstrap(baseline, baseline, replicates=200, seed=7)
    half = _paired_ratio_bootstrap(
        [0.5 * value for value in baseline],
        baseline,
        replicates=200,
        seed=7,
    )

    assert identity["point_estimate"] == pytest.approx(1.0)
    assert identity["bootstrap_ci95_low"] == pytest.approx(1.0)
    assert identity["bootstrap_ci95_high"] == pytest.approx(1.0)
    assert half["point_estimate"] == pytest.approx(0.5)
    assert half["bootstrap_ci95_low"] == pytest.approx(0.5)
    assert half["bootstrap_ci95_high"] == pytest.approx(0.5)


def test_tensor_digest_tracks_exact_output_bytes() -> None:
    original = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    equal = original.clone()
    changed = original.clone()
    changed[0, 1] = -1.0

    assert _tensor_sha256(original) == _tensor_sha256(equal)
    assert _tensor_sha256(original) != _tensor_sha256(changed)


def test_completion_selects_the_sealed_layer_file(tmp_path) -> None:
    layer = 24
    layer_name = f"qsrt-layer-{layer:05d}.safetensors"
    layer_path = tmp_path / layer_name
    layer_path.write_bytes(b"sealed-layer")
    storage_schema = "kquant_kimi_k3_qsrt_atoms_v2"
    assert storage_schema in _STORAGE_SCHEMAS
    completion = {
        "kind": _COMPLETION_KIND,
        "schema_version": _VERSION,
        "storage_schema": storage_schema,
        "profile": _K2.profile,
        "complete": True,
        "layer_count": 92,
        "candidate_pool_content_sha256": "a" * 64,
        "layers": {
            str(layer): {
                "file": layer_name,
                "bytes": layer_path.stat().st_size,
            }
        },
    }
    (tmp_path / "qsrt-completion.json").write_text(json.dumps(completion))

    selected, provenance = _validate_completion(tmp_path, _K2, layer)

    assert selected == layer_path
    assert provenance["candidate_pool_content_sha256"] == "a" * 64
    assert len(provenance["completion_sha256"]) == 64


def test_benchmark_result_is_atomic_and_never_overwritten(tmp_path) -> None:
    path = tmp_path / "result.json"

    _write_new_result(path, '{"complete": true}\n')

    assert path.read_text() == '{"complete": true}\n'
    with pytest.raises(FileExistsError):
        _write_new_result(path, '{"complete": false}\n')
    assert path.read_text() == '{"complete": true}\n'
