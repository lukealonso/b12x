"""Immutable data contracts for GPU component policies."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

_COMPONENT_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SCALAR_TYPES = (str, int, float, bool, type(None))
_MISSING = object()
ConfigT = TypeVar("ConfigT")


def _normalized_name(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _freeze_profile_value(value: object, *, field: str) -> object:
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_profile_value(item, field=f"{field}[]") for item in value)
    if isinstance(value, _SCALAR_TYPES):
        return value
    raise TypeError(
        f"profile field {field!r} must contain JSON-compatible values, "
        f"got {type(value).__name__}"
    )


def _thaw_profile_value(value: object) -> object:
    if isinstance(value, FrozenMapping):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_thaw_profile_value(item) for item in value]
    return value


@dataclass(frozen=True, init=False)
class FrozenMapping(Mapping[str, object]):
    """A recursively immutable, hashable mapping for generated profile data."""

    _items: tuple[tuple[str, object], ...]

    def __init__(self, values: Mapping[str, object] | None = None) -> None:
        items: list[tuple[str, object]] = []
        for key, value in (values or {}).items():
            if not isinstance(key, str) or not key:
                raise ValueError("profile mapping keys must be non-empty strings")
            items.append((key, _freeze_profile_value(value, field=key)))
        object.__setattr__(self, "_items", tuple(sorted(items)))

    def __getitem__(self, key: str) -> object:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> dict[str, object]:
        return {key: _thaw_profile_value(value) for key, value in self._items}


@dataclass(frozen=True)
class DeviceIdentity:
    """Portable hardware identity used by embedded profile matching."""

    vendor: str
    compute_capability: tuple[int, int]
    sm_count: int
    product_name: str

    def __post_init__(self) -> None:
        vendor = _normalized_name(self.vendor)
        product_name = _normalized_name(self.product_name)
        if not vendor or not product_name:
            raise ValueError("vendor and product_name must be non-empty")
        capability = tuple(self.compute_capability)
        if len(capability) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in capability
        ):
            raise ValueError("compute_capability must contain two nonnegative integers")
        if not isinstance(self.sm_count, int) or isinstance(self.sm_count, bool):
            raise TypeError("sm_count must be an integer")
        if self.sm_count <= 0:
            raise ValueError("sm_count must be positive")
        object.__setattr__(self, "vendor", vendor)
        object.__setattr__(self, "compute_capability", capability)
        object.__setattr__(self, "product_name", product_name)


@dataclass(frozen=True)
class MatchRange:
    """Inclusive integer range for one component query axis."""

    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.minimum, int)
            or isinstance(self.minimum, bool)
            or not isinstance(self.maximum, int)
            or isinstance(self.maximum, bool)
        ):
            raise TypeError("profile range bounds must be integers")
        if self.minimum > self.maximum:
            raise ValueError("profile range minimum cannot exceed maximum")

    def contains(self, value: object) -> bool:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and self.minimum <= value <= self.maximum
        )

    def overlaps(self, other: "MatchRange") -> bool:
        return self.minimum <= other.maximum and other.minimum <= self.maximum


@dataclass(frozen=True)
class ProfileRule:
    """One preplanned component config and the query region it covers."""

    name: str
    exact: FrozenMapping
    ranges: tuple[tuple[str, MatchRange], ...]
    config: FrozenMapping
    priority: int = 0
    evidence: str | None = None

    @classmethod
    def create(
        cls,
        *,
        name: str,
        exact: Mapping[str, object] | None,
        ranges: Mapping[str, MatchRange | tuple[int, int]] | None,
        config: Mapping[str, object],
        priority: int = 0,
        evidence: str | None = None,
    ) -> "ProfileRule":
        frozen_exact = FrozenMapping(exact)
        frozen_ranges: list[tuple[str, MatchRange]] = []
        for field, bounds in (ranges or {}).items():
            if not isinstance(field, str) or not field:
                raise ValueError("profile range fields must be non-empty strings")
            if field in frozen_exact:
                raise ValueError(f"profile field {field!r} cannot be exact and ranged")
            if not isinstance(bounds, MatchRange):
                bounds = MatchRange(*bounds)
            frozen_ranges.append((field, bounds))
        return cls(
            name=name,
            exact=frozen_exact,
            ranges=tuple(sorted(frozen_ranges)),
            config=FrozenMapping(config),
            priority=priority,
            evidence=evidence,
        )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile rule names must be non-empty")
        if not self.exact and not self.ranges:
            raise ValueError("profile rules must constrain at least one field")
        for field, value in self.exact.items():
            if not isinstance(value, _SCALAR_TYPES):
                raise TypeError(f"exact profile field {field!r} must be a scalar")
        range_fields = [field for field, _ in self.ranges]
        if len(range_fields) != len(set(range_fields)):
            raise ValueError("profile rule contains duplicate range fields")
        if any(field in self.exact for field in range_fields):
            raise ValueError("profile fields cannot be exact and ranged")
        if not self.config:
            raise ValueError("profile rule configs must be non-empty")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise TypeError("profile rule priority must be an integer")
        if self.evidence is not None and not self.evidence:
            raise ValueError("profile rule evidence must be non-empty when set")

    @property
    def query_fields(self) -> frozenset[str]:
        return frozenset((*self.exact, *(field for field, _ in self.ranges)))

    def matches(self, query: Mapping[str, object]) -> bool:
        for field, expected in self.exact.items():
            if query.get(field, _MISSING) != expected:
                return False
        for field, bounds in self.ranges:
            if not bounds.contains(query.get(field, _MISSING)):
                return False
        return True

    def overlaps(self, other: "ProfileRule") -> bool:
        fields = self.query_fields | other.query_fields
        left_ranges = dict(self.ranges)
        right_ranges = dict(other.ranges)
        for field in fields:
            left_exact = self.exact.get(field, _MISSING)
            right_exact = other.exact.get(field, _MISSING)
            left_range = left_ranges.get(field)
            right_range = right_ranges.get(field)
            if left_exact is not _MISSING and right_exact is not _MISSING:
                if left_exact != right_exact:
                    return False
            elif left_exact is not _MISSING and right_range is not None:
                if not right_range.contains(left_exact):
                    return False
            elif right_exact is not _MISSING and left_range is not None:
                if not left_range.contains(right_exact):
                    return False
            elif left_range is not None and right_range is not None:
                if not left_range.overlaps(right_range):
                    return False
        return True


@dataclass(frozen=True)
class ProfileLeaf:
    """Terminal preplanned config in a generated component decision tree."""

    name: str
    config: FrozenMapping
    evidence: str | None = None

    @classmethod
    def create(
        cls,
        *,
        name: str,
        config: Mapping[str, object],
        evidence: str | None = None,
    ) -> "ProfileLeaf":
        return cls(
            name=name,
            config=FrozenMapping(config),
            evidence=evidence,
        )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile leaf names must be non-empty")
        if not self.config:
            raise ValueError("profile leaf configs must be non-empty")
        if self.evidence is not None and not self.evidence:
            raise ValueError("profile leaf evidence must be non-empty when set")

    @property
    def query_fields(self) -> frozenset[str]:
        return frozenset()

    def lookup(self, query: Mapping[str, object]) -> "ProfileLeaf":
        del query
        return self

    def iter_leaves(self) -> Iterator["ProfileLeaf"]:
        yield self


@dataclass(frozen=True)
class ExactDecisionNode:
    """Dispatch on one exact scalar query field."""

    field: str
    branches: tuple[tuple[object, "DecisionNode"], ...]
    default: "DecisionNode | None" = None

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("decision fields must be non-empty")
        if not self.branches:
            raise ValueError("exact decision nodes must contain branches")
        keys: list[tuple[type[object], object]] = []
        for value, node in self.branches:
            if not isinstance(value, _SCALAR_TYPES):
                raise TypeError("exact decision values must be scalars")
            if not isinstance(
                node, (ProfileLeaf, ExactDecisionNode, RangeDecisionNode)
            ):
                raise TypeError("exact decision branches must contain decision nodes")
            keys.append((type(value), value))
        if len(keys) != len(set(keys)):
            raise ValueError("exact decision nodes cannot repeat branch values")
        if self.default is not None and not isinstance(
            self.default,
            (ProfileLeaf, ExactDecisionNode, RangeDecisionNode),
        ):
            raise TypeError("exact decision defaults must be decision nodes")

    @property
    def query_fields(self) -> frozenset[str]:
        fields = {self.field}
        for _, node in self.branches:
            fields.update(node.query_fields)
        if self.default is not None:
            fields.update(self.default.query_fields)
        return frozenset(fields)

    def lookup(self, query: Mapping[str, object]) -> ProfileLeaf | None:
        value = query.get(self.field, _MISSING)
        for expected, node in self.branches:
            if type(value) is type(expected) and value == expected:
                return node.lookup(query)
        return None if self.default is None else self.default.lookup(query)

    def iter_leaves(self) -> Iterator[ProfileLeaf]:
        for _, node in self.branches:
            yield from node.iter_leaves()
        if self.default is not None:
            yield from self.default.iter_leaves()


@dataclass(frozen=True)
class RangeDecisionNode:
    """Dispatch on disjoint inclusive intervals of one integer field."""

    field: str
    branches: tuple[tuple[MatchRange, "DecisionNode"], ...]
    default: "DecisionNode | None" = None

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("decision fields must be non-empty")
        if not self.branches:
            raise ValueError("range decision nodes must contain branches")
        for index, (bounds, node) in enumerate(self.branches):
            if not isinstance(bounds, MatchRange):
                raise TypeError("range decision bounds must be MatchRange values")
            if not isinstance(
                node, (ProfileLeaf, ExactDecisionNode, RangeDecisionNode)
            ):
                raise TypeError("range decision branches must contain decision nodes")
            for other, _ in self.branches[index + 1 :]:
                if bounds.overlaps(other):
                    raise ValueError("range decision branches cannot overlap")
        if self.default is not None and not isinstance(
            self.default,
            (ProfileLeaf, ExactDecisionNode, RangeDecisionNode),
        ):
            raise TypeError("range decision defaults must be decision nodes")

    @property
    def query_fields(self) -> frozenset[str]:
        fields = {self.field}
        for _, node in self.branches:
            fields.update(node.query_fields)
        if self.default is not None:
            fields.update(self.default.query_fields)
        return frozenset(fields)

    def lookup(self, query: Mapping[str, object]) -> ProfileLeaf | None:
        value = query.get(self.field, _MISSING)
        for bounds, node in self.branches:
            if bounds.contains(value):
                return node.lookup(query)
        return None if self.default is None else self.default.lookup(query)

    def iter_leaves(self) -> Iterator[ProfileLeaf]:
        for _, node in self.branches:
            yield from node.iter_leaves()
        if self.default is not None:
            yield from self.default.iter_leaves()


DecisionNode = ProfileLeaf | ExactDecisionNode | RangeDecisionNode


@dataclass(frozen=True)
class ComponentProfile:
    """One generated component planner on one GPU profile."""

    component_id: str
    query_schema_version: int
    config_schema_version: int
    rules: tuple[ProfileRule, ...] = ()
    planner: DecisionNode | None = None
    coverage: FrozenMapping = FrozenMapping()

    def __post_init__(self) -> None:
        if not _COMPONENT_ID_RE.fullmatch(self.component_id):
            raise ValueError(f"invalid component ID {self.component_id!r}")
        if self.query_schema_version <= 0 or self.config_schema_version <= 0:
            raise ValueError("component schema versions must be positive")
        if bool(self.rules) == bool(self.planner):
            raise ValueError(
                "component profiles require exactly one of rules or planner"
            )
        if not isinstance(self.coverage, FrozenMapping):
            object.__setattr__(self, "coverage", FrozenMapping(self.coverage))
        if self.planner is not None:
            if not isinstance(
                self.planner,
                (ProfileLeaf, ExactDecisionNode, RangeDecisionNode),
            ):
                raise TypeError("component planner must be a decision node")
            return
        names = [rule.name for rule in self.rules]
        if len(names) != len(set(names)):
            raise ValueError(
                f"component {self.component_id!r} has duplicate rule names"
            )
        for index, left in enumerate(self.rules):
            for right in self.rules[index + 1 :]:
                if left.priority == right.priority and left.overlaps(right):
                    raise ValueError(
                        f"component {self.component_id!r} rules {left.name!r} "
                        f"and {right.name!r} overlap at priority {left.priority}"
                    )

    @property
    def query_fields(self) -> frozenset[str]:
        if self.planner is not None:
            return self.planner.query_fields
        return frozenset(field for rule in self.rules for field in rule.query_fields)

    def lookup(
        self,
        query: Mapping[str, object],
    ) -> ProfileRule | ProfileLeaf | None:
        if self.planner is not None:
            return self.planner.lookup(query)
        matches = [rule for rule in self.rules if rule.matches(query)]
        if not matches:
            return None
        return max(matches, key=lambda rule: rule.priority)

    @property
    def config_entries(self) -> tuple[ProfileRule | ProfileLeaf, ...]:
        if self.planner is not None:
            return tuple(self.planner.iter_leaves())
        return self.rules


@dataclass(frozen=True)
class GpuProfile:
    """Sparse set of generated component configs for equivalent GPUs."""

    profile_id: str
    targets: tuple[DeviceIdentity, ...]
    components: tuple[ComponentProfile, ...]
    metadata: FrozenMapping = FrozenMapping()

    def __post_init__(self) -> None:
        if not _PROFILE_ID_RE.fullmatch(self.profile_id):
            raise ValueError(f"invalid GPU profile ID {self.profile_id!r}")
        if not self.targets:
            raise ValueError("GPU profiles must contain at least one target")
        if len(self.targets) != len(set(self.targets)):
            raise ValueError(f"GPU profile {self.profile_id!r} has duplicate targets")
        component_ids = [component.component_id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError(
                f"GPU profile {self.profile_id!r} has duplicate components"
            )

    def component(self, component_id: str) -> ComponentProfile | None:
        return next(
            (
                component
                for component in self.components
                if component.component_id == component_id
            ),
            None,
        )


class PolicyMode(str, Enum):
    AUTO = "auto"
    HEURISTIC_ONLY = "heuristic-only"
    PREPLANNED_ONLY = "preplanned-only"


class PolicySource(str, Enum):
    OVERRIDE = "override"
    PREPLANNED = "preplanned"
    HEURISTIC = "heuristic"


@dataclass(frozen=True)
class ProfileHit:
    profile_id: str
    component_id: str
    rule_name: str
    config: FrozenMapping
    evidence: str | None


@dataclass(frozen=True)
class PolicyResolution(Generic[ConfigT]):
    """Typed component config and its complete selection provenance."""

    config: ConfigT
    source: PolicySource
    component_id: str
    device: DeviceIdentity | None
    profile_id: str | None = None
    rule_name: str | None = None
    evidence: str | None = None

    @property
    def is_preplanned(self) -> bool:
        return self.source is PolicySource.PREPLANNED


__all__ = [
    "ComponentProfile",
    "DecisionNode",
    "DeviceIdentity",
    "ExactDecisionNode",
    "FrozenMapping",
    "GpuProfile",
    "MatchRange",
    "PolicyMode",
    "PolicyResolution",
    "PolicySource",
    "ProfileHit",
    "ProfileLeaf",
    "ProfileRule",
    "RangeDecisionNode",
]
