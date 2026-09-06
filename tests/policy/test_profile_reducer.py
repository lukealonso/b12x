from __future__ import annotations

from b12x.policy.generation import (
    DecisionRecord,
    build_axis_tree,
    decision_node_to_dict,
    synthesize_integer_axis_coverage,
)
from b12x.policy.serialization import profile_from_dict
from b12x.policy.types import RangeDecisionNode


def _record(family: str, rows: int, backend: str) -> DecisionRecord:
    return DecisionRecord.create(
        query={"family": family, "rows": rows},
        config={"backend": backend},
    )


def test_reducer_coalesces_only_consecutive_equal_decisions() -> None:
    tree = build_axis_tree(
        (
            _record("a", 1, "micro"),
            _record("a", 2, "micro"),
            _record("a", 4, "micro"),
            _record("a", 5, "dynamic"),
            _record("b", 1, "dynamic"),
        ),
        field_order=("family", "rows"),
        range_fields=frozenset({"rows"}),
        evidence="profile-evidence.json",
    )

    assert tree.lookup({"family": "a", "rows": 1}).config["backend"] == "micro"
    assert tree.lookup({"family": "a", "rows": 2}).config["backend"] == "micro"
    assert tree.lookup({"family": "a", "rows": 3}) is None
    assert tree.lookup({"family": "a", "rows": 4}).config["backend"] == "micro"
    assert tree.lookup({"family": "a", "rows": 5}).config["backend"] == "dynamic"
    assert tree.lookup({"family": "c", "rows": 1}) is None


def test_serialized_generated_tree_round_trips_through_runtime_parser() -> None:
    tree = build_axis_tree(
        (_record("a", 4, "micro"), _record("a", 5, "micro")),
        field_order=("family", "rows"),
        range_fields=frozenset({"rows"}),
        evidence="profile-evidence.json",
    )
    profile = profile_from_dict(
        {
            "profile_id": "nvidia.synthetic.tree",
            "targets": [
                {
                    "vendor": "nvidia",
                    "compute_capability": [12, 1],
                    "sm_count": 48,
                    "product_name": "Synthetic GPU",
                }
            ],
            "components": [
                {
                    "component_id": "test.decode",
                    "query_schema_version": 1,
                    "config_schema_version": 1,
                    "coverage": {
                        "corpus_sha256": "abc123",
                        "query_points": 2,
                    },
                    "planner": decision_node_to_dict(tree),
                }
            ],
        }
    )

    component = profile.component("test.decode")
    assert component is not None
    assert component.coverage["query_points"] == 2
    assert component.lookup({"family": "a", "rows": 5}).config["backend"] == "micro"


def test_reducer_rle_compresses_unmarked_consecutive_integer_branches() -> None:
    tree = build_axis_tree(
        (_record("a", 4, "micro"), _record("a", 5, "micro")),
        field_order=("family", "rows"),
    )

    rows = tree.branches[0][1]
    assert isinstance(rows, RangeDecisionNode)
    assert rows.branches[0][0].minimum == 4
    assert rows.branches[0][0].maximum == 5
    assert rows.lookup({"family": "a", "rows": 6}) is None


def test_serializer_groups_exact_values_with_the_same_subtree() -> None:
    tree = build_axis_tree(
        (_record("a", 4, "micro"), _record("b", 4, "micro")),
        field_order=("family", "rows"),
    )

    encoded = decision_node_to_dict(tree)
    branches = encoded["branches"]
    assert isinstance(branches, list)
    assert len(branches) == 1
    assert branches[0]["values"] == ["a", "b"]
    assert branches[0]["node"]["kind"] == "exact"
    assert branches[0]["node"]["field"] == "rows"
    profile = profile_from_dict(
        {
            "profile_id": "nvidia.synthetic.grouped-exact",
            "targets": [
                {
                    "vendor": "nvidia",
                    "compute_capability": [12, 1],
                    "sm_count": 48,
                    "product_name": "Synthetic GPU",
                }
            ],
            "components": [
                {
                    "component_id": "test.decode",
                    "query_schema_version": 1,
                    "config_schema_version": 1,
                    "planner": encoded,
                }
            ],
        }
    )
    component = profile.component("test.decode")
    assert component is not None
    assert component.lookup({"family": "a", "rows": 4}) is not None
    assert component.lookup({"family": "b", "rows": 4}) is not None
    assert component.lookup({"family": "c", "rows": 4}) is None


def test_integer_axis_coverage_uses_nearest_valid_anchor() -> None:
    records = synthesize_integer_axis_coverage(
        (_record("a", 1, "micro"), _record("a", 4, "dynamic")),
        field="rows",
        minimum=1,
        maximum=4,
        config_is_valid=lambda query, config: (
            not (config["backend"] == "micro" and query["rows"] > 2)
        ),
    )
    tree = build_axis_tree(
        records,
        field_order=("family", "rows"),
        range_fields=frozenset({"rows"}),
        evidence="profile-evidence.json",
    )

    assert tree.lookup({"family": "a", "rows": 2}).config["backend"] == "micro"
    assert tree.lookup({"family": "a", "rows": 3}).config["backend"] == "dynamic"


def test_nearest_range_bounds_cover_gaps_and_coalesce_equal_configs() -> None:
    records = (
        DecisionRecord.create(query={"tokens": 1}, config={"backend": "a"}),
        DecisionRecord.create(query={"tokens": 4}, config={"backend": "a"}),
        DecisionRecord.create(query={"tokens": 16}, config={"backend": "b"}),
    )

    tree = build_axis_tree(
        records,
        field_order=("tokens",),
        range_fields=frozenset({"tokens"}),
        nearest_range_bounds={"tokens": (1, 32)},
    )

    assert tree.lookup({"tokens": 1}).config["backend"] == "a"
    assert tree.lookup({"tokens": 9}).config["backend"] == "a"
    assert tree.lookup({"tokens": 10}).config["backend"] == "a"
    assert tree.lookup({"tokens": 11}).config["backend"] == "b"
    assert tree.lookup({"tokens": 32}).config["backend"] == "b"
    assert tree.lookup({"tokens": 33}) is None
