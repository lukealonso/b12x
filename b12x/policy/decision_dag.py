"""Lossless encoding of component policies as shared decision diagrams."""

from __future__ import annotations

import json

from .types import DecisionNode, ExactDecisionNode, ProfileLeaf, _decision_nodes


def _common_guards(root: DecisionNode) -> dict[str, object]:
    guards: dict[str, object] = {}
    invalid: set[str] = set()
    occurrences: dict[str, int] = {}
    required: dict[int, frozenset[str]] = {}

    def required_fields(node: DecisionNode) -> frozenset[str]:
        cached = required.get(id(node))
        if cached is not None:
            return cached
        if isinstance(node, ProfileLeaf):
            fields = frozenset()
        else:
            children = [child for _, child in node.branches]
            if node.default is not None:
                children.append(node.default)
            fields = frozenset({node.field}) | frozenset.intersection(
                *(required_fields(child) for child in children)
            )
        required[id(node)] = fields
        return fields

    for node in _decision_nodes(root):
        if isinstance(node, ProfileLeaf):
            continue
        field = node.field
        occurrences[field] = occurrences.get(field, 0) + 1
        if (
            not isinstance(node, ExactDecisionNode)
            or len(node.branches) != 1
            or node.default is not None
        ):
            invalid.add(field)
            continue
        value = node.branches[0][0]
        if field in guards and (
            type(value) is not type(guards[field]) or value != guards[field]
        ):
            invalid.add(field)
        guards[field] = value
    every_path = required_fields(root)
    return {
        field: value for field, value in guards.items()
        if field not in invalid and field in every_path and occurrences[field] > 1
    }


def encode_planner_dag(
    root: DecisionNode,
    *,
    hoist_guards: bool = True,
) -> dict[str, object]:
    """Intern equal configs and subtrees without extending query coverage.

    Child references precede their parents. Common equality guards are moved
    only when every accepted path contains the identical test and no occurrence
    has a default. Missing and uncovered query values retain their behavior.
    """
    if isinstance(root, ProfileLeaf):
        result: dict[str, object] = {
            "kind": "leaf", "name": root.name, "config": root.config.to_dict(),
        }
        if root.evidence is not None:
            result["evidence"] = root.evidence
        return result

    guards = _common_guards(root) if hoist_guards else {}
    nodes: list[dict[str, object]] = []
    configs: list[dict[str, object]] = []
    node_ids: dict[str, int] = {}
    config_ids: dict[str, int] = {}
    visited: dict[int, int] = {}

    def intern(value: dict[str, object], table: list, index: dict[str, int]) -> int:
        key = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        cached = index.get(key)
        if cached is None:
            cached = len(table)
            table.append(value)
            index[key] = cached
        return cached

    def visit(node: DecisionNode) -> int:
        cached = visited.get(id(node))
        if cached is not None:
            return cached
        if isinstance(node, ProfileLeaf):
            value: dict[str, object] = {
                "kind": "leaf", "name": node.name,
                "config": intern(node.config.to_dict(), configs, config_ids),
            }
            if node.evidence is not None:
                value["evidence"] = node.evidence
        elif node.field in guards:
            result = visit(node.branches[0][1])
            visited[id(node)] = result
            return result
        else:
            if isinstance(node, ExactDecisionNode):
                grouped: dict[int, list[object]] = {}
                for scalar, child in node.branches:
                    grouped.setdefault(visit(child), []).append(scalar)
                branches = [
                    {
                        "value" if len(values) == 1 else "values": (
                            values[0] if len(values) == 1 else values
                        ),
                        "node": child_id,
                    }
                    for child_id, values in grouped.items()
                ]
                kind = "exact"
            else:
                branches = []
                for bounds, child in node.branches:
                    child_id = visit(child)
                    if (
                        branches and branches[-1]["node"] == child_id
                        and branches[-1]["maximum"] + 1 == bounds.minimum
                    ):
                        branches[-1]["maximum"] = bounds.maximum
                    else:
                        branches.append({
                            "minimum": bounds.minimum, "maximum": bounds.maximum,
                            "node": child_id,
                        })
                kind = "range"
            value = {"kind": kind, "field": node.field, "branches": branches}
            if node.default is not None:
                value["default"] = visit(node.default)
        result = intern(value, nodes, node_ids)
        visited[id(node)] = result
        return result

    root_id = visit(root)
    for field, scalar in reversed(tuple(guards.items())):
        root_id = intern({
            "kind": "exact", "field": field,
            "branches": [{"value": scalar, "node": root_id}],
        }, nodes, node_ids)
    return {
        "kind": "dag", "schema_version": 1, "root": root_id,
        "configs": configs, "nodes": nodes,
    }


__all__ = ["encode_planner_dag"]
