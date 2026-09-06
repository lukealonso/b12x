"""Generic reduction from discrete sweep winners to an axis decision tree."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from b12x.policy.types import (
    DecisionNode,
    ExactDecisionNode,
    FrozenMapping,
    MatchRange,
    ProfileLeaf,
    RangeDecisionNode,
)


@dataclass(frozen=True, kw_only=True)
class DecisionRecord:
    """Winning config for one fully specified, route-robust query point."""

    query: FrozenMapping
    config: FrozenMapping

    @classmethod
    def create(
        cls,
        *,
        query: Mapping[str, object],
        config: Mapping[str, object],
    ) -> "DecisionRecord":
        return cls(query=FrozenMapping(query), config=FrozenMapping(config))

    def __post_init__(self) -> None:
        if not self.query or not self.config:
            raise ValueError("decision records require query and config fields")


def _leaf(config: FrozenMapping, *, evidence: str | None) -> ProfileLeaf:
    encoded = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    return ProfileLeaf(
        name=f"config-{digest}",
        config=config,
        evidence=evidence,
    )


def _build_node(
    records: Sequence[DecisionRecord],
    *,
    fields: tuple[str, ...],
    range_fields: frozenset[str],
    nearest_range_bounds: Mapping[str, MatchRange],
    evidence: str | None,
) -> DecisionNode:
    if not fields:
        configs = {record.config for record in records}
        if len(configs) != 1:
            raise ValueError(
                "field order does not distinguish records with different configs"
            )
        return _leaf(next(iter(configs)), evidence=evidence)

    field, *remaining = fields
    grouped: dict[object, list[DecisionRecord]] = {}
    for record in records:
        grouped.setdefault(record.query[field], []).append(record)
    children = {
        value: _build_node(
            group,
            fields=tuple(remaining),
            range_fields=range_fields,
            nearest_range_bounds=nearest_range_bounds,
            evidence=evidence,
        )
        for value, group in grouped.items()
    }
    values = tuple(children)
    integer_axis = all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    )
    if field in range_fields and not integer_axis:
        raise TypeError(f"range field {field!r} must contain only integers")
    if integer_axis:
        nearest_bounds = nearest_range_bounds.get(field)
        if nearest_bounds is not None:
            ordered = sorted(int(value) for value in values)
            if (
                ordered[0] < nearest_bounds.minimum
                or ordered[-1] > nearest_bounds.maximum
            ):
                raise ValueError(
                    f"nearest range field {field!r} has anchors outside "
                    f"[{nearest_bounds.minimum}, {nearest_bounds.maximum}]"
                )
            ranges: list[tuple[MatchRange, DecisionNode]] = []
            minimum = nearest_bounds.minimum
            for index, value in enumerate(ordered):
                maximum = (
                    nearest_bounds.maximum
                    if index + 1 == len(ordered)
                    else (value + ordered[index + 1]) // 2
                )
                child = children[value]
                if ranges and ranges[-1][1] == child:
                    previous, _ = ranges[-1]
                    ranges[-1] = (
                        MatchRange(previous.minimum, maximum),
                        child,
                    )
                else:
                    ranges.append((MatchRange(minimum, maximum), child))
                minimum = maximum + 1
            return RangeDecisionNode(field=field, branches=tuple(ranges))
        ranges: list[tuple[MatchRange, DecisionNode]] = []
        for value in sorted(values):
            child = children[value]
            if ranges:
                previous, previous_child = ranges[-1]
                if value == previous.maximum + 1 and child == previous_child:
                    ranges[-1] = (
                        MatchRange(previous.minimum, value),
                        previous_child,
                    )
                    continue
            ranges.append((MatchRange(value, value), child))
        if field in range_fields or len(ranges) < len(values):
            return RangeDecisionNode(field=field, branches=tuple(ranges))

    branches = tuple(
        (value, children[value])
        for value in sorted(
            children, key=lambda item: (type(item).__name__, repr(item))
        )
    )
    return ExactDecisionNode(field=field, branches=branches)


def build_axis_tree(
    records: Sequence[DecisionRecord],
    *,
    field_order: tuple[str, ...],
    range_fields: frozenset[str] = frozenset(),
    nearest_range_bounds: Mapping[str, tuple[int, int] | MatchRange] | None = None,
    evidence: str | None = None,
) -> DecisionNode:
    """Build a deterministic tree without extrapolating across unswept gaps."""

    records = tuple(records)
    if not records:
        raise ValueError("cannot build a decision tree without records")
    if not field_order or len(field_order) != len(set(field_order)):
        raise ValueError("field_order must be non-empty and unique")
    if not range_fields <= frozenset(field_order):
        raise ValueError("range_fields must be present in field_order")
    normalized_nearest_bounds = {
        field: bounds if isinstance(bounds, MatchRange) else MatchRange(*bounds)
        for field, bounds in (nearest_range_bounds or {}).items()
    }
    if not frozenset(normalized_nearest_bounds) <= range_fields:
        raise ValueError("nearest range fields must also be range_fields")
    expected = frozenset(field_order)
    seen_queries: set[FrozenMapping] = set()
    for record in records:
        actual = frozenset(record.query)
        if actual != expected:
            raise ValueError(
                f"decision query fields differ from field_order; "
                f"missing={sorted(expected - actual)}, "
                f"unknown={sorted(actual - expected)}"
            )
        if record.query in seen_queries:
            raise ValueError("decision records cannot repeat a query")
        seen_queries.add(record.query)
    return _build_node(
        records,
        fields=field_order,
        range_fields=range_fields,
        nearest_range_bounds=normalized_nearest_bounds,
        evidence=evidence,
    )


def synthesize_integer_axis_coverage(
    records: Sequence[DecisionRecord],
    *,
    field: str,
    minimum: int,
    maximum: int,
    config_is_valid: Callable[
        [Mapping[str, object], Mapping[str, object]], bool
    ]
    | None = None,
) -> tuple[DecisionRecord, ...]:
    """Fill a bounded integer axis from its nearest measured anchors."""

    records = tuple(records)
    if not records:
        raise ValueError("cannot synthesize coverage without decision records")
    if not field:
        raise ValueError("coverage field must be non-empty")
    if minimum > maximum:
        raise ValueError("coverage minimum cannot exceed maximum")

    grouped: dict[FrozenMapping, list[DecisionRecord]] = {}
    seen_queries: set[FrozenMapping] = set()
    for record in records:
        if record.query in seen_queries:
            raise ValueError("decision records cannot repeat a query")
        seen_queries.add(record.query)
        value = record.query.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"coverage field {field!r} must contain integers")
        if not minimum <= value <= maximum:
            raise ValueError(
                f"coverage field {field!r} value {value} is outside "
                f"[{minimum}, {maximum}]"
            )
        group = FrozenMapping(
            {key: item for key, item in record.query.items() if key != field}
        )
        grouped.setdefault(group, []).append(record)

    synthesized: list[DecisionRecord] = []
    for group in sorted(grouped, key=repr):
        anchors = sorted(grouped[group], key=lambda item: int(item.query[field]))
        for value in range(minimum, maximum + 1):
            ranked = sorted(
                anchors,
                key=lambda item: (
                    abs(int(item.query[field]) - value),
                    int(item.query[field]),
                ),
            )
            selected: DecisionRecord | None = None
            for anchor in ranked:
                query = anchor.query.to_dict()
                query[field] = value
                if config_is_valid is None or config_is_valid(
                    query, anchor.config
                ):
                    selected = DecisionRecord.create(
                        query=query,
                        config=anchor.config,
                    )
                    break
                if int(anchor.query[field]) == value:
                    raise ValueError(
                        f"measured config is invalid at {field}={value}"
                    )
            if selected is None:
                raise ValueError(
                    f"no measured config can cover {field}={value} for {group}"
                )
            synthesized.append(selected)
    return tuple(synthesized)


def decision_node_to_dict(node: DecisionNode) -> dict[str, object]:
    if isinstance(node, ProfileLeaf):
        result: dict[str, object] = {
            "kind": "leaf",
            "name": node.name,
            "config": node.config.to_dict(),
        }
        if node.evidence is not None:
            result["evidence"] = node.evidence
        return result
    if isinstance(node, ExactDecisionNode):
        grouped: dict[DecisionNode, list[object]] = {}
        for value, child in node.branches:
            grouped.setdefault(child, []).append(value)
        result = {
            "kind": "exact",
            "field": node.field,
            "branches": [
                {
                    ("value" if len(values) == 1 else "values"): (
                        values[0] if len(values) == 1 else values
                    ),
                    "node": decision_node_to_dict(child),
                }
                for child, values in grouped.items()
            ],
        }
        if node.default is not None:
            result["default"] = decision_node_to_dict(node.default)
        return result
    result = {
        "kind": "range",
        "field": node.field,
        "branches": [
            {
                "minimum": bounds.minimum,
                "maximum": bounds.maximum,
                "node": decision_node_to_dict(child),
            }
            for bounds, child in node.branches
        ],
    }
    if node.default is not None:
        result["default"] = decision_node_to_dict(node.default)
    return result


__all__ = [
    "DecisionRecord",
    "build_axis_tree",
    "decision_node_to_dict",
    "synthesize_integer_axis_coverage",
]
