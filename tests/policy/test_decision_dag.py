from __future__ import annotations

from copy import deepcopy
from itertools import product

import pytest

from b12x.policy import ExactDecisionNode, FrozenMapping, MatchRange, ProfileLeaf, RangeDecisionNode
from b12x.policy.decision_dag import encode_planner_dag
from b12x.policy.serialization import _planner_node


def _leaf(name: str) -> ProfileLeaf:
    return ProfileLeaf.create(name=name, config={"backend": name}, evidence="qualified")


def _hit(node, query):
    leaf = node.lookup(query)
    return None if leaf is None else (leaf.name, leaf.config, leaf.evidence)


def test_diagram_sharing_preserves_guards_holes_types_and_provenance():
    def branch():
        return ExactDecisionNode("enabled", ((True, RangeDecisionNode("rows", (
            (MatchRange(1, 3), _leaf("a")), (MatchRange(8, 10), _leaf("b")),
        ))),))
    root = ExactDecisionNode("family", (("left", branch()), ("right", branch())))
    encoded = encode_planner_dag(root)
    decoded = _planner_node(encoded, name="planner")
    for family, enabled, rows in product(
        ("left", "right", "missing", None), (True, False, 1, None), range(12),
    ):
        query = {"family": family, "enabled": enabled, "rows": rows}
        assert _hit(decoded, query) == _hit(root, query)
        for field in query:
            incomplete = {key: value for key, value in query.items() if key != field}
            assert _hit(decoded, incomplete) == _hit(root, incomplete)
    assert decoded.query_fields == root.query_fields
    assert len(tuple(decoded.iter_leaves())) == 2
    assert encoded["nodes"][encoded["root"]]["field"] == "enabled"


def test_diagram_defaults_keep_their_scope():
    default = _leaf("fallback")
    root = ExactDecisionNode("family", (
        ("left", ExactDecisionNode("mode", (("decode", _leaf("a")),), default)),
        ("right", ExactDecisionNode("mode", (("decode", _leaf("b")),))),
    ), default)
    decoded = _planner_node(encode_planner_dag(root), name="planner")
    for family, mode in product(("left", "right", "unknown", None), ("decode", "other", None)):
        query = {"family": family, "mode": mode}
        assert _hit(decoded, query) == _hit(root, query)


def test_diagram_config_interning_preserves_scalar_types():
    root = ExactDecisionNode("mode", (
        (True, ProfileLeaf("bool", FrozenMapping({"value": True}))),
        (1, ProfileLeaf("int", FrozenMapping({"value": 1}))),
    ))
    encoded = encode_planner_dag(root)
    assert len(encoded["configs"]) == 2
    decoded = _planner_node(encoded, name="planner")
    assert type(decoded.lookup({"mode": True}).config["value"]) is bool
    assert type(decoded.lookup({"mode": 1}).config["value"]) is int


@pytest.mark.parametrize("corruption", ("cycle", "negative", "boolean", "config", "unused", "version"))
def test_diagram_rejects_invalid_references(corruption):
    encoded = encode_planner_dag(ExactDecisionNode("mode", (("decode", _leaf("a")),)))
    encoded = deepcopy(encoded)
    if corruption == "cycle":
        encoded["nodes"][-1]["branches"][0]["node"] = encoded["root"]
    elif corruption == "negative":
        encoded["root"] = -1
    elif corruption == "boolean":
        encoded["root"] = True
    elif corruption == "config":
        encoded["nodes"][0]["config"] = len(encoded["configs"])
    elif corruption == "unused":
        encoded["configs"].append({"backend": "unused"})
    else:
        encoded["schema_version"] = 2
    with pytest.raises(ValueError):
        _planner_node(encoded, name="planner")


def test_diagram_validation_visits_shared_nodes_once():
    node = _leaf("a")
    for index in range(30):
        node = ExactDecisionNode(f"axis{index}", ((0, node), (1, node)))
    decoded = _planner_node(encode_planner_dag(node), name="planner")
    assert len(decoded.query_fields) == 30
    assert tuple(decoded.iter_leaves()) == (_leaf("a"),)


def test_diagram_rejects_excessive_depth():
    node = _leaf("a")
    for index in range(65):
        node = ExactDecisionNode(f"axis{index}", ((0, node),))
    with pytest.raises(ValueError, match="maximum decision-tree depth"):
        _planner_node(encode_planner_dag(node), name="planner")
