"""Strict parser for generated GPU-profile payloads."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from .types import (
    ComponentProfile,
    DecisionNode,
    DeviceIdentity,
    ExactDecisionNode,
    FrozenMapping,
    GpuProfile,
    MatchRange,
    ProfileLeaf,
    ProfileRule,
    RangeDecisionNode,
)


def _object(
    value: object,
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    keys = frozenset(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ValueError(f"{name} is missing fields {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields {sorted(unknown)}")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an array")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _device(value: object, *, index: int) -> DeviceIdentity:
    name = f"profile.targets[{index}]"
    data = _object(
        value,
        name=name,
        required=frozenset(
            {
                "compute_capability",
                "product_name",
                "sm_count",
                "vendor",
            }
        ),
    )
    capability = _sequence(
        data["compute_capability"],
        name=f"{name}.compute_capability",
    )
    if len(capability) != 2:
        raise ValueError(f"{name}.compute_capability must have two elements")
    return DeviceIdentity(
        vendor=str(data["vendor"]),
        compute_capability=(int(capability[0]), int(capability[1])),
        sm_count=_positive_int(data["sm_count"], name=f"{name}.sm_count"),
        product_name=str(data["product_name"]),
    )


def _rule(value: object, *, component: str, index: int) -> ProfileRule:
    name = f"profile.components[{component!r}].rules[{index}]"
    data = _object(
        value,
        name=name,
        required=frozenset({"config", "exact", "name", "ranges"}),
        optional=frozenset({"evidence", "priority"}),
    )
    exact = data["exact"]
    ranges = data["ranges"]
    config = data["config"]
    if not isinstance(exact, Mapping):
        raise TypeError(f"{name}.exact must be an object")
    if not isinstance(ranges, Mapping):
        raise TypeError(f"{name}.ranges must be an object")
    if not isinstance(config, Mapping):
        raise TypeError(f"{name}.config must be an object")
    parsed_ranges: dict[str, tuple[int, int]] = {}
    for field, bounds in ranges.items():
        values = _sequence(bounds, name=f"{name}.ranges.{field}")
        if len(values) != 2:
            raise ValueError(f"{name}.ranges.{field} must have two elements")
        parsed_ranges[str(field)] = (int(values[0]), int(values[1]))
    evidence = data.get("evidence")
    return ProfileRule.create(
        name=str(data["name"]),
        exact=exact,
        ranges=parsed_ranges,
        config=config,
        priority=int(data.get("priority", 0)),
        evidence=None if evidence is None else str(evidence),
    )


def _planner_node(
    value: object, *, name: str, depth: int = 0,
    resolve_child: Callable[[object, str], DecisionNode] | None = None,
) -> DecisionNode:
    if depth > 64:
        raise ValueError(f"{name} exceeds the maximum decision-tree depth")
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    kind = value.get("kind")
    if kind == "dag" and resolve_child is None and depth == 0:
        return _planner_dag(value, name=name)
    if kind == "leaf":
        data = _object(
            value,
            name=name,
            required=frozenset({"config", "kind", "name"}),
            optional=frozenset({"evidence"}),
        )
        config = data["config"]
        if not isinstance(config, Mapping):
            raise TypeError(f"{name}.config must be an object")
        evidence = data.get("evidence")
        return ProfileLeaf(
            name=str(data["name"]),
            config=config if isinstance(config, FrozenMapping) else FrozenMapping(config),
            evidence=None if evidence is None else str(evidence),
        )
    if kind not in {"exact", "range"}:
        raise ValueError(f"{name}.kind must be 'leaf', 'exact', or 'range'")
    data = _object(
        value,
        name=name,
        required=frozenset({"branches", "field", "kind"}),
        optional=frozenset({"default"}),
    )
    field = str(data["field"])
    branches = _sequence(data["branches"], name=f"{name}.branches")

    def child(value: object, child_name: str) -> DecisionNode:
        if resolve_child is not None:
            return resolve_child(value, child_name)
        return _planner_node(value, name=child_name, depth=depth + 1)

    default_value = data.get("default")
    default = (
        None
        if default_value is None
        else child(default_value, f"{name}.default")
    )
    if kind == "exact":
        parsed_exact: list[tuple[object, DecisionNode]] = []
        for index, branch in enumerate(branches):
            branch_name = f"{name}.branches[{index}]"
            branch_data = _object(
                branch,
                name=branch_name,
                required=frozenset({"node"}),
                optional=frozenset({"value", "values"}),
            )
            have_value = "value" in branch_data
            have_values = "values" in branch_data
            if have_value == have_values:
                raise ValueError(
                    f"{branch_name} requires exactly one of value or values"
                )
            raw_values = (
                (branch_data["value"],)
                if have_value
                else tuple(
                    _sequence(branch_data["values"], name=f"{branch_name}.values")
                )
            )
            if not raw_values:
                raise ValueError(f"{branch_name}.values must not be empty")
            if any(
                not isinstance(item, (str, int, float, bool, type(None)))
                for item in raw_values
            ):
                raise TypeError(f"{branch_name} exact values must be scalars")
            node = child(branch_data["node"], f"{branch_name}.node")
            parsed_exact.extend((item, node) for item in raw_values)
        return ExactDecisionNode(
            field=field,
            branches=tuple(parsed_exact),
            default=default,
        )

    parsed_ranges: list[tuple[MatchRange, DecisionNode]] = []
    for index, branch in enumerate(branches):
        branch_name = f"{name}.branches[{index}]"
        branch_data = _object(
            branch,
            name=branch_name,
            required=frozenset({"maximum", "minimum", "node"}),
        )
        parsed_ranges.append(
            (
                MatchRange(
                    minimum=int(branch_data["minimum"]),
                    maximum=int(branch_data["maximum"]),
                ),
                child(branch_data["node"], f"{branch_name}.node"),
            )
        )
    return RangeDecisionNode(
        field=field,
        branches=tuple(parsed_ranges),
        default=default,
    )


def _planner_dag(value: object, *, name: str) -> DecisionNode:
    data = _object(
        value, name=name,
        required=frozenset({"kind", "schema_version", "root", "configs", "nodes"}),
    )
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError(f"{name} has an unsupported decision-DAG schema version")
    raw_configs = _sequence(data["configs"], name=f"{name}.configs")
    configs = []
    for index, config in enumerate(raw_configs):
        if not isinstance(config, Mapping) or not config:
            raise TypeError(f"{name}.configs[{index}] must be a non-empty object")
        configs.append(FrozenMapping(config))
    raw_nodes = _sequence(data["nodes"], name=f"{name}.nodes")
    nodes: list[DecisionNode] = []
    heights: list[int] = []
    children: list[list[int]] = []
    used_configs: set[int] = set()

    def reference(value: object, size: int, field: str) -> int:
        if type(value) is not int or not 0 <= value < size:
            raise ValueError(f"{field} must reference an earlier node or an existing config")
        return value

    for index, raw_node in enumerate(raw_nodes):
        node_name = f"{name}.nodes[{index}]"
        if not isinstance(raw_node, Mapping):
            raise TypeError(f"{node_name} must be an object")
        payload = dict(raw_node)
        references: list[int] = []

        def resolve(value: object, field: str) -> DecisionNode:
            child_id = reference(value, len(nodes), field)
            references.append(child_id)
            return nodes[child_id]

        if payload.get("kind") == "leaf":
            config_id = reference(payload.get("config"), len(configs), f"{node_name}.config")
            used_configs.add(config_id)
            payload["config"] = configs[config_id]
        node = _planner_node(payload, name=node_name, resolve_child=resolve)
        height = max((heights[child_id] + 1 for child_id in references), default=0)
        if height > 64:
            raise ValueError(f"{name} exceeds the maximum decision-tree depth")
        nodes.append(node)
        heights.append(height)
        children.append(references)
    root_id = reference(data["root"], len(nodes), f"{name}.root")
    pending = [root_id]
    reachable: set[int] = set()
    while pending:
        node_id = pending.pop()
        if node_id not in reachable:
            reachable.add(node_id)
            pending.extend(children[node_id])
    if len(reachable) != len(nodes) or used_configs != set(range(len(configs))):
        raise ValueError(f"{name} contains unreachable nodes or configs")
    return nodes[root_id]


def _component(value: object, *, index: int) -> ComponentProfile:
    name = f"profile.components[{index}]"
    data = _object(
        value,
        name=name,
        required=frozenset(
            {
                "component_id",
                "config_schema_version",
                "query_schema_version",
            }
        ),
        optional=frozenset({"coverage", "planner", "rules"}),
    )
    component_id = str(data["component_id"])
    have_rules = "rules" in data
    have_planner = "planner" in data
    if have_rules == have_planner:
        raise ValueError(f"{name} requires exactly one of rules or planner")
    rules = _sequence(data["rules"], name=f"{name}.rules") if have_rules else ()
    coverage = data.get("coverage", {})
    if not isinstance(coverage, Mapping):
        raise TypeError(f"{name}.coverage must be an object")
    return ComponentProfile(
        component_id=component_id,
        query_schema_version=_positive_int(
            data["query_schema_version"],
            name=f"{name}.query_schema_version",
        ),
        config_schema_version=_positive_int(
            data["config_schema_version"],
            name=f"{name}.config_schema_version",
        ),
        rules=tuple(
            _rule(rule, component=component_id, index=rule_index)
            for rule_index, rule in enumerate(rules)
        ),
        planner=(
            _planner_node(data["planner"], name=f"{name}.planner")
            if have_planner
            else None
        ),
        coverage=FrozenMapping(coverage),
    )


def profile_from_dict(value: object) -> GpuProfile:
    """Parse one strict, JSON-compatible GPU profile payload."""

    data = _object(
        value,
        name="profile",
        required=frozenset({"components", "profile_id", "targets"}),
        optional=frozenset({"metadata"}),
    )
    targets = _sequence(data["targets"], name="profile.targets")
    components = _sequence(data["components"], name="profile.components")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("profile.metadata must be an object")
    return GpuProfile(
        profile_id=str(data["profile_id"]),
        targets=tuple(
            _device(target, index=index) for index, target in enumerate(targets)
        ),
        components=tuple(
            _component(component, index=index)
            for index, component in enumerate(components)
        ),
        metadata=FrozenMapping(metadata),
    )


__all__ = ["profile_from_dict"]
