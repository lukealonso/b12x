"""Typed ownership and independent coordinates of GPU tuning problems."""

from __future__ import annotations

import hashlib
import json
import math
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from .types import DeviceIdentity, FrozenMapping


class FieldRole(str, Enum):
    FAMILY = "family"
    AXIS = "axis"
    CONSTRAINT = "constraint"
    DERIVED = "derived"
    ENVIRONMENT = "environment"


class BindingTime(str, Enum):
    MODEL = "model"
    PLAN = "plan"
    RUNTIME = "runtime"
    ENVIRONMENT = "environment"


def stable_identity(value: object) -> str:
    if isinstance(value, FrozenMapping):
        value = value.to_dict()
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True, kw_only=True)
class ProblemField:
    name: str
    role: FieldRole
    binding: BindingTime
    minimum: int | None = None
    alignment: int = 1
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or type(self.alignment) is not int or self.alignment <= 0:
            raise ValueError("problem fields require a name and positive integer alignment")
        if self.minimum is not None and type(self.minimum) is not int:
            raise TypeError("axis minimum must be an integer")
        if self.role is not FieldRole.AXIS and (self.minimum is not None or self.alignment != 1):
            raise ValueError("only independent axes have numeric domains")
        if self.role is FieldRole.DERIVED and not self.dependencies:
            raise ValueError("derived inputs must declare their dependencies")
        if self.dependencies and self.role is not FieldRole.DERIVED:
            raise ValueError("only derived inputs may declare dependencies")

    def validate(self, value: object) -> None:
        if self.role is FieldRole.AXIS:
            if type(value) is not int:
                raise TypeError(f"axis {self.name!r} must be an integer")
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f"axis {self.name!r} is below its minimum")
            if value % self.alignment:
                raise ValueError(f"axis {self.name!r} violates its alignment")


@dataclass(frozen=True, kw_only=True)
class DecisionParameter:
    name: str
    values: tuple[object, ...] | None = None
    ordered: bool = False
    when: FrozenMapping = FrozenMapping()

    def __post_init__(self) -> None:
        if not self.name or self.values == ():
            raise ValueError("decision parameters require a name and nonempty domain")
        if self.values is None:
            return
        identities = tuple(stable_identity(value) for value in self.values)
        if len(set(identities)) != len(identities):
            raise ValueError(f"decision parameter {self.name!r} has duplicate values")
        numeric = tuple(value for value in self.values if value is not None)
        if self.ordered and any(type(value) is not int for value in numeric):
            raise TypeError("ordered decision parameters require integer values or null")
        if self.ordered and tuple(sorted(numeric)) != numeric:
            raise ValueError("ordered decision values must be increasing")


@dataclass(frozen=True, kw_only=True)
class AxisInterval:
    name: str
    minimum: int
    maximum: int
    alignment: int = 1

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.minimum, self.maximum, self.alignment)):
            raise TypeError("axis bounds and alignment must be integers")
        if not self.name or self.minimum > self.maximum or self.alignment <= 0:
            raise ValueError("axis interval must be named, ordered, and positively aligned")
        if self.minimum % self.alignment or self.maximum % self.alignment:
            raise ValueError("axis endpoints must satisfy alignment")

    @property
    def count(self) -> int:
        return (self.maximum - self.minimum) // self.alignment + 1

    def contains(self, value: object) -> bool:
        return (type(value) is int and self.minimum <= value <= self.maximum
                and value % self.alignment == 0)

    def midpoint(self) -> int:
        return self.minimum + ((self.count - 1) // 2) * self.alignment


@dataclass(frozen=True, kw_only=True)
class SearchDomain:
    """A bounded slice with explicit coordinates and fixed independent inputs."""

    fixed: FrozenMapping
    axes: tuple[AxisInterval, ...]

    def __post_init__(self) -> None:
        names = tuple(axis.name for axis in self.axes)
        if len(names) != len(set(names)) or set(names) & set(self.fixed):
            raise ValueError("domain coordinates must be unique and disjoint from fixed inputs")

    @property
    def size(self) -> int:
        return math.prod(axis.count for axis in self.axes)

    def contains(self, query: Mapping[str, object]) -> bool:
        return (all(name in query and type(query[name]) is type(value) and query[name] == value
                    for name, value in self.fixed.items())
                and all(axis.contains(query.get(axis.name)) for axis in self.axes))

    def to_dict(self) -> dict[str, object]:
        return {"fixed": self.fixed.to_dict(), "axes": [
            {"name": axis.name, "minimum": axis.minimum,
             "maximum": axis.maximum, "alignment": axis.alignment}
            for axis in self.axes
        ]}

    def queries(self):
        """Enumerate the declared integer lattice without materializing it."""
        for values in itertools.product(*(range(axis.minimum, axis.maximum + 1, axis.alignment)
                                          for axis in self.axes)):
            yield {**self.fixed, **dict(zip((axis.name for axis in self.axes), values, strict=True))}


@dataclass(frozen=True, kw_only=True)
class TuningProblem:
    """Component-owned field accounting, lowering, and search coordinates.

    Family membership organizes search; it is not an execution or measurement
    identity. Numeric features and dimensions never define compilation keys.
    """

    component_id: str
    query_type: type
    config_type: type
    policy: Any
    inputs: tuple[ProblemField, ...]
    decisions: tuple[DecisionParameter, ...]
    derived_config_fields: tuple[str, ...] = ()
    sampled_inputs: tuple[ProblemField, ...] = ()
    derive_inputs: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None
    materialize_decision: Callable[[object, DeviceIdentity | None, FrozenMapping], object] | None = None
    materialize_collection: Callable | None = None
    scenario_reduction: str = "geometric_mean"
    selection_contract: str = "minimum_qualified_latency"
    contract_version: int = 1

    def __post_init__(self) -> None:
        if type(self.contract_version) is not int or self.contract_version <= 0:
            raise ValueError("tuning problem contract version must be a positive integer")
        if not is_dataclass(self.query_type) or not is_dataclass(self.config_type):
            raise TypeError("tuning problems require typed dataclass queries and configs")
        names = tuple(field.name for field in self.inputs)
        expected = {field.name for field in fields(self.query_type)}
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError(f"{self.component_id} input accounting differs from its query: "
                             f"missing={sorted(expected - set(names))}, "
                             f"unknown={sorted(set(names) - expected)}")
        sampled_names = {field.name for field in self.sampled_inputs}
        if len(sampled_names) != len(self.sampled_inputs) or sampled_names & expected:
            raise ValueError("sampled inputs must be unique and separate from query inputs")
        decision_names = tuple(parameter.name for parameter in self.decisions)
        if len(decision_names) != len(set(decision_names)):
            raise ValueError("independent decision parameters must be unique")
        for parameter in self.decisions:
            if parameter.name in parameter.when or set(parameter.when) - set(decision_names):
                raise ValueError("conditional decisions must depend on other declared decisions")
        pending_decisions = {parameter.name: set(parameter.when) for parameter in self.decisions}
        available_decisions = set()
        while pending_decisions:
            ready = {name for name, dependencies in pending_decisions.items()
                     if dependencies <= available_decisions}
            if not ready:
                raise ValueError("conditional decisions have cyclic dependencies")
            available_decisions |= ready
            pending_decisions = {name: dependencies for name, dependencies in pending_decisions.items()
                                 if name not in ready}
        by_name = {parameter.name: parameter for parameter in self.decisions}
        for parameter in self.decisions:
            for name, value in parameter.when.items():
                values = by_name[name].values
                if values is not None and not any(type(value) is type(item) and value == item for item in values):
                    raise ValueError("conditional decision tests a value outside its dependency domain")
        config_names = {field.name for field in fields(self.config_type)}
        if set(self.derived_config_fields) & set(decision_names):
            raise ValueError("config fields cannot be both independent and derived")
        if config_names != (set(decision_names) & config_names) | set(self.derived_config_fields):
            raise ValueError(f"{self.component_id} does not account for every resolved config field")
        derived = {field.name: field.dependencies for field in self.inputs
                   if field.role is FieldRole.DERIVED}
        available = expected - derived.keys()
        pending = dict(derived)
        while pending:
            ready = {name for name, dependencies in pending.items() if set(dependencies) <= available}
            if not ready:
                raise ValueError("derived inputs have cyclic or missing dependencies")
            available |= ready
            pending = {name: dependencies for name, dependencies in pending.items() if name not in ready}
        if derived and self.derive_inputs is None:
            raise ValueError("derived inputs require a deterministic derivation")
        if self.component_id != self.policy.component_id:
            raise ValueError("tuning problem and runtime policy component IDs differ")

    @property
    def axes(self) -> tuple[ProblemField, ...]:
        return tuple(field for field in (*self.inputs, *self.sampled_inputs)
                     if field.role is FieldRole.AXIS)

    def canonical_inputs(self, query: object) -> FrozenMapping:
        if not isinstance(query, self.query_type):
            raise TypeError(f"query must be {self.query_type.__name__}")
        result = {}
        for field in self.inputs:
            value = getattr(query, field.name)
            field.validate(value)
            if field.role is not FieldRole.ENVIRONMENT:
                result[field.name] = value
        if self.derive_inputs is not None:
            for name, value in self.derive_inputs(result).items():
                if name not in result or type(value) is not type(result[name]) or value != result[name]:
                    raise ValueError(f"inconsistent derived input {name!r}")
        return FrozenMapping(result)

    def family_key(self, query: object) -> FrozenMapping:
        values = self.canonical_inputs(query)
        return FrozenMapping({field.name: values[field.name] for field in self.inputs
                              if field.role in (FieldRole.FAMILY, FieldRole.CONSTRAINT)})

    def query_from_inputs(self, values: Mapping[str, object]) -> object:
        """Construct a typed query from offline inputs without device discovery."""
        inputs = dict(values)
        for field in self.sampled_inputs:
            inputs.pop(field.name, None)
        if self.derive_inputs is not None:
            derived = self.derive_inputs(inputs)
            for name, value in derived.items():
                if name in inputs and (type(inputs[name]) is not type(value) or inputs[name] != value):
                    raise ValueError(f"inconsistent derived input {name!r}")
                inputs[name] = value
        for field in self.inputs:
            if field.role is FieldRole.ENVIRONMENT:
                inputs[field.name] = None
        query = self.query_type(**inputs)
        self.canonical_inputs(query)
        return query

    def lower(self, query: object, device: DeviceIdentity | None,
              decision: Mapping[str, object]) -> object:
        self.canonical_inputs(query)
        self.validate_decision(decision)
        if self.materialize_collection is not None:
            raise ValueError("this policy requires a collection of sampled-input decisions")
        materialize = self.materialize_decision or self.policy.decode_profile
        config = materialize(query, device, FrozenMapping(decision))
        self.policy.validate_config(query, config, device)
        return config

    def validate_decision(self, decision: Mapping[str, object]) -> None:
        active = {parameter.name: parameter for parameter in self.decisions
                  if all(name in decision and type(decision[name]) is type(value)
                         and decision[name] == value for name, value in parameter.when.items())}
        if set(decision) != set(active):
            raise ValueError("decision fields differ from the active kernel-parameter contract")
        for name, parameter in active.items():
            if parameter.values is not None and not any(
                type(decision[name]) is type(value) and decision[name] == value
                for value in parameter.values
            ):
                raise ValueError(f"decision {name!r} is outside its declared domain")

    def lower_collection(self, query: object, device: DeviceIdentity | None,
                         decisions: Sequence[tuple[Mapping, Mapping]]) -> object:
        self.canonical_inputs(query)
        if self.materialize_collection is None:
            raise ValueError("this policy does not define sampled-input collection lowering")
        expected = {field.name for field in self.sampled_inputs}
        seen = set()
        for sample, decision in decisions:
            if set(sample) != expected:
                raise ValueError("sampled inputs differ from the collection contract")
            for field in self.sampled_inputs:
                field.validate(sample[field.name])
            identity = stable_identity(dict(sample))
            if identity in seen:
                raise ValueError("sampled-input decisions must be unique")
            seen.add(identity)
            self.validate_decision(decision)
        config = self.materialize_collection(query, device, decisions)
        self.policy.validate_config(query, config, device)
        return config

    def describe(self) -> dict[str, object]:
        def field_dict(field: ProblemField) -> dict[str, object]:
            return {"name": field.name, "role": field.role.value,
                    "binding": field.binding.value, "minimum": field.minimum,
                    "alignment": field.alignment, "dependencies": list(field.dependencies)}
        return {"component_id": self.component_id, "contract_version": self.contract_version,
                "query_type": f"{self.query_type.__module__}.{self.query_type.__name__}",
                "config_type": f"{self.config_type.__module__}.{self.config_type.__name__}",
                "inputs": [field_dict(field) for field in self.inputs],
                "sampled_inputs": [field_dict(field) for field in self.sampled_inputs],
                "decisions": [{"name": parameter.name,
                               "values": None if parameter.values is None else list(parameter.values),
                               "domain": "provider" if parameter.values is None else "declared",
                               "ordered": parameter.ordered, "when": parameter.when.to_dict()}
                              for parameter in self.decisions],
                "derived_config_fields": list(self.derived_config_fields),
                "scenario_reduction": self.scenario_reduction,
                "selection_contract": self.selection_contract}


def define_problem(*, policy: Any, query_type: type, config_type: type,
                   axes: tuple[str, ...], family: tuple[str, ...],
                   decisions: Mapping[str, tuple[object, ...] | None],
                   constraints: tuple[str, ...] = (), environment: tuple[str, ...] = (),
                   model_fields: tuple[str, ...] = (), ordered: tuple[str, ...] = (),
                   derived_config_fields: tuple[str, ...] = (),
                   derived_inputs: Mapping[str, tuple[str, ...]] | None = None,
                   derive_inputs: Callable | None = None,
                   materialize_decision: Callable | None = None,
                   materialize_collection: Callable | None = None,
                   decision_conditions: Mapping[str, Mapping[str, object]] | None = None,
                   sampled_inputs: tuple[ProblemField, ...] = (),
                   axis_domains: Mapping[str, tuple[int, int]] | None = None,
                   selection_contract: str = "minimum_qualified_latency") -> TuningProblem:
    declarations = []
    model = frozenset(model_fields)
    for role, names in ((FieldRole.AXIS, axes), (FieldRole.FAMILY, family),
                        (FieldRole.CONSTRAINT, constraints), (FieldRole.ENVIRONMENT, environment),
                        (FieldRole.DERIVED, tuple(derived_inputs or {}))):
        for name in names:
            declarations.append(ProblemField(
                name=name, role=role,
                binding=(BindingTime.ENVIRONMENT if role is FieldRole.ENVIRONMENT
                         else BindingTime.MODEL if name in model else BindingTime.PLAN),
                dependencies=tuple((derived_inputs or {}).get(name, ())),
                minimum=(axis_domains or {}).get(name, (None, 1))[0],
                alignment=(axis_domains or {}).get(name, (None, 1))[1],
            ))
    if set(axis_domains or {}) - set(axes):
        raise ValueError("numeric domains must name independent axes")
    unknown_model = model - {field.name for field in declarations}
    if unknown_model:
        raise ValueError(f"unknown model-bound inputs: {sorted(unknown_model)}")
    return TuningProblem(component_id=policy.component_id, policy=policy,
                         query_type=query_type, config_type=config_type,
                         inputs=tuple(declarations),
                         decisions=tuple(DecisionParameter(name=name, values=None if values is None else tuple(values),
                                                           ordered=name in ordered,
                                                           when=FrozenMapping((decision_conditions or {}).get(name, {})))
                                         for name, values in decisions.items()),
                         derived_config_fields=derived_config_fields,
                         derive_inputs=derive_inputs, sampled_inputs=sampled_inputs,
                         materialize_decision=materialize_decision,
                         materialize_collection=materialize_collection,
                         selection_contract=selection_contract)
