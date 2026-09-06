"""Typed policy resolution with profile and heuristic precedence."""

from __future__ import annotations

import logging
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, TypeVar, cast

from .device import detect_device
from .registry import EMBEDDED_REGISTRY, ProfileRegistry
from .types import (
    ComponentProfile,
    DeviceIdentity,
    FrozenMapping,
    PolicyMode,
    PolicyResolution,
    PolicySource,
)

QueryT = TypeVar("QueryT")
ConfigT = TypeVar("ConfigT")
NO_POLICY_OVERRIDE = object()
_COMPONENT_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_QUERY_SCALAR_TYPES = (str, int, float, bool, type(None))
_RESOLUTION_CACHE_CAPACITY = 4096
_HEURISTIC_WARNING_KEYS: set[tuple[str, DeviceIdentity | None, str, FrozenMapping]] = (
    set()
)
_HEURISTIC_WARNING_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


def _device_label(device: DeviceIdentity | None) -> str:
    if device is None:
        return "an unknown device"
    major, minor = device.compute_capability
    return (
        f"{device.product_name} (compute capability {major}.{minor}, "
        f"{device.sm_count} SMs)"
    )


def _warn_heuristic_fallback(
    *,
    component_id: str,
    device: DeviceIdentity | None,
    reason: str,
    query: FrozenMapping,
) -> None:
    key = (component_id, device, reason, query)
    with _HEURISTIC_WARNING_LOCK:
        if key in _HEURISTIC_WARNING_KEYS:
            return
        _HEURISTIC_WARNING_KEYS.add(key)
    logger.warning(
        "b12x policy fallback: %s is using a heuristic on %s because %s; query=%s",
        component_id,
        _device_label(device),
        reason,
        query.to_dict(),
    )


class PreplannedPolicyNotFoundError(LookupError):
    """Raised when preplanned-only mode has no valid matching config."""


class InvalidPreplannedPolicyError(ValueError):
    """Raised when embedded profile data violates a component contract."""


@dataclass(frozen=True)
class ComponentPolicy(Generic[QueryT, ConfigT]):
    """Typed adapter between a component and declarative profile data."""

    component_id: str
    query_schema_version: int
    config_schema_version: int
    query_fields: frozenset[str]
    config_fields: frozenset[str]
    encode_query: Callable[[QueryT], Mapping[str, object]]
    decode_profile: Callable[[FrozenMapping], ConfigT]
    heuristic: Callable[[QueryT, DeviceIdentity | None], ConfigT]
    validate_config: Callable[
        [QueryT, ConfigT, DeviceIdentity | None],
        None,
    ]

    def __post_init__(self) -> None:
        if not _COMPONENT_ID_RE.fullmatch(self.component_id):
            raise ValueError(f"invalid component ID {self.component_id!r}")
        if self.query_schema_version <= 0 or self.config_schema_version <= 0:
            raise ValueError("component schema versions must be positive")
        if not self.query_fields or not self.config_fields:
            raise ValueError("component query and config fields must be non-empty")
        if any(not field for field in (*self.query_fields, *self.config_fields)):
            raise ValueError("component fields must be non-empty strings")


def validate_component_profile_contract(
    policy: ComponentPolicy[object, object],
    profile: ComponentProfile,
) -> None:
    """Validate serialized schema ownership against one runtime policy."""

    if profile.query_schema_version != policy.query_schema_version:
        raise InvalidPreplannedPolicyError(
            f"{policy.component_id} query schema mismatch: component uses "
            f"{policy.query_schema_version}, profile uses "
            f"{profile.query_schema_version}"
        )
    if profile.config_schema_version != policy.config_schema_version:
        raise InvalidPreplannedPolicyError(
            f"{policy.component_id} config schema mismatch: component uses "
            f"{policy.config_schema_version}, profile uses "
            f"{profile.config_schema_version}"
        )
    unknown_query = profile.query_fields - policy.query_fields
    if unknown_query:
        raise InvalidPreplannedPolicyError(
            f"component profile has unknown query fields {sorted(unknown_query)}"
        )
    for entry in profile.config_entries:
        unknown_config = frozenset(entry.config) - policy.config_fields
        if unknown_config:
            raise InvalidPreplannedPolicyError(
                f"profile entry {entry.name!r} has unknown config fields "
                f"{sorted(unknown_config)}"
            )


class PolicyContext:
    """Immutable per-device policy selection with explicit precedence.

    Explicit config overrides win first. In AUTO mode, a matching embedded
    entry is authoritative; malformed or invalid embedded data never silently
    falls back to a heuristic. Heuristics are used only when no entry matches,
    no profile owns the device, or the caller explicitly requests
    ``HEURISTIC_ONLY``.
    """

    def __init__(
        self,
        *,
        device: DeviceIdentity | None,
        mode: PolicyMode | str = PolicyMode.AUTO,
        registry: ProfileRegistry = EMBEDDED_REGISTRY,
        device_ordinal: int | None = None,
        overrides: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(mode, PolicyMode):
            mode = PolicyMode(mode)
        self.device = device
        self.device_ordinal = device_ordinal
        self.mode = mode
        self._registry = registry
        self._profile = registry.find(device)
        normalized_overrides = dict(overrides or {})
        for component_id in normalized_overrides:
            if not _COMPONENT_ID_RE.fullmatch(component_id):
                raise ValueError(f"invalid component ID {component_id!r}")
        self._overrides = MappingProxyType(normalized_overrides)
        self._resolution_cache: OrderedDict[
            tuple[ComponentPolicy[object, object], FrozenMapping],
            PolicyResolution[object],
        ] = OrderedDict()
        self._validated_profile_contracts: set[ComponentPolicy[object, object]] = set()
        self._cache_lock = threading.RLock()

    @classmethod
    def for_device(
        cls,
        device: object | None = None,
        *,
        mode: PolicyMode | str = PolicyMode.AUTO,
    ) -> "PolicyContext":
        if not isinstance(mode, PolicyMode):
            mode = PolicyMode(mode)
        detected = detect_device(device)
        key = (
            detected.ordinal,
            detected.identity,
            mode,
            EMBEDDED_REGISTRY.revision,
        )
        context = _AUTO_CONTEXT_CACHE.get(key)
        if context is None:
            context = cls(
                device=detected.identity,
                device_ordinal=detected.ordinal,
                mode=mode,
            )
            _AUTO_CONTEXT_CACHE[key] = context
        return context

    @classmethod
    def for_identity(
        cls,
        device: DeviceIdentity | None,
        *,
        mode: PolicyMode | str = PolicyMode.AUTO,
        registry: ProfileRegistry = EMBEDDED_REGISTRY,
    ) -> "PolicyContext":
        return cls(device=device, mode=mode, registry=registry)

    @property
    def profile_id(self) -> str | None:
        return None if self._profile is None else self._profile.profile_id

    def with_override(self, component_id: str, config: object) -> "PolicyContext":
        """Return a context with one component config taking precedence."""

        overrides = dict(self._overrides)
        overrides[component_id] = config
        return type(self)(
            device=self.device,
            device_ordinal=self.device_ordinal,
            mode=self.mode,
            registry=self._registry,
            overrides=overrides,
        )

    def require_device(self, device: object) -> None:
        detected = detect_device(device)
        if detected.identity != self.device:
            raise ValueError(
                "policy context device mismatch: context is for "
                f"{self.device!r}, requested {detected.identity!r}"
            )
        if self.device_ordinal is not None and detected.ordinal != self.device_ordinal:
            raise ValueError(
                "policy context CUDA ordinal mismatch: context is for "
                f"cuda:{self.device_ordinal}, requested cuda:{detected.ordinal}"
            )

    def _cached_resolution(
        self,
        key: tuple[ComponentPolicy[object, object], FrozenMapping],
    ) -> PolicyResolution[object] | None:
        with self._cache_lock:
            resolution = self._resolution_cache.get(key)
            if resolution is not None:
                self._resolution_cache.move_to_end(key)
            return resolution

    def _cache_resolution(
        self,
        key: tuple[ComponentPolicy[object, object], FrozenMapping],
        resolution: PolicyResolution[object],
    ) -> PolicyResolution[object]:
        with self._cache_lock:
            existing = self._resolution_cache.get(key)
            if existing is not None:
                self._resolution_cache.move_to_end(key)
                return existing
            self._resolution_cache[key] = resolution
            if len(self._resolution_cache) > _RESOLUTION_CACHE_CAPACITY:
                self._resolution_cache.popitem(last=False)
            return resolution

    def _validate_profile_contract_once(
        self,
        component: ComponentPolicy[object, object],
        profile: ComponentProfile,
    ) -> None:
        with self._cache_lock:
            if component in self._validated_profile_contracts:
                return
            validate_component_profile_contract(component, profile)
            self._validated_profile_contracts.add(component)

    def resolve(
        self,
        component: ComponentPolicy[QueryT, ConfigT],
        query: QueryT,
        *,
        override: ConfigT | object = NO_POLICY_OVERRIDE,
    ) -> PolicyResolution[ConfigT]:
        fields = dict(component.encode_query(query))
        actual_fields = frozenset(fields)
        if actual_fields != component.query_fields:
            raise ValueError(
                f"{component.component_id} query schema mismatch; "
                f"missing={sorted(component.query_fields - actual_fields)}, "
                f"unknown={sorted(actual_fields - component.query_fields)}"
            )
        for field, value in fields.items():
            if not isinstance(value, _QUERY_SCALAR_TYPES):
                raise TypeError(f"component query field {field!r} must be a scalar")

        if override is not NO_POLICY_OVERRIDE:
            config = cast(ConfigT, override)
            component.validate_config(query, config, self.device)
            return PolicyResolution(
                config=config,
                source=PolicySource.OVERRIDE,
                component_id=component.component_id,
                device=self.device,
            )

        cache_key = (
            cast(ComponentPolicy[object, object], component),
            FrozenMapping(fields),
        )
        cached = self._cached_resolution(cache_key)
        if cached is not None:
            return cast(PolicyResolution[ConfigT], cached)

        configured_override = self._overrides.get(
            component.component_id,
            NO_POLICY_OVERRIDE,
        )
        if configured_override is not NO_POLICY_OVERRIDE:
            config = cast(ConfigT, configured_override)
            component.validate_config(query, config, self.device)
            resolution = PolicyResolution(
                config=config,
                source=PolicySource.OVERRIDE,
                component_id=component.component_id,
                device=self.device,
            )
            return cast(
                PolicyResolution[ConfigT],
                self._cache_resolution(cache_key, resolution),
            )

        fallback_reason = "no embedded profile matches the device"
        if self.mode is not PolicyMode.HEURISTIC_ONLY and self._profile is not None:
            component_profile = self._profile.component(component.component_id)
            if component_profile is not None:
                fallback_reason = (
                    f"profile {self._profile.profile_id!r} does not cover the query"
                )
                try:
                    self._validate_profile_contract_once(
                        cast(ComponentPolicy[object, object], component),
                        component_profile,
                    )
                    hit = self._registry.lookup(
                        profile=self._profile,
                        component_id=component.component_id,
                        query=fields,
                    )
                    if hit is not None:
                        config = component.decode_profile(hit.config)
                        component.validate_config(query, config, self.device)
                        resolution = PolicyResolution(
                            config=config,
                            source=PolicySource.PREPLANNED,
                            component_id=component.component_id,
                            device=self.device,
                            profile_id=hit.profile_id,
                            rule_name=hit.rule_name,
                            evidence=hit.evidence,
                        )
                        return cast(
                            PolicyResolution[ConfigT],
                            self._cache_resolution(cache_key, resolution),
                        )
                except InvalidPreplannedPolicyError:
                    raise
                except (TypeError, ValueError) as exc:
                    raise InvalidPreplannedPolicyError(
                        f"invalid preplanned {component.component_id} config"
                    ) from exc
            else:
                fallback_reason = (
                    f"profile {self._profile.profile_id!r} has no component entry"
                )

        if self.mode is PolicyMode.PREPLANNED_ONLY:
            raise PreplannedPolicyNotFoundError(
                f"no preplanned {component.component_id!r} config matches "
                f"device={self.device!r}, query={fields!r}"
            )

        config = component.heuristic(query, self.device)
        component.validate_config(query, config, self.device)
        if self.mode is PolicyMode.AUTO:
            _warn_heuristic_fallback(
                component_id=component.component_id,
                device=self.device,
                reason=fallback_reason,
                query=cache_key[1],
            )
        resolution = PolicyResolution(
            config=config,
            source=PolicySource.HEURISTIC,
            component_id=component.component_id,
            device=self.device,
        )
        return cast(
            PolicyResolution[ConfigT],
            self._cache_resolution(cache_key, resolution),
        )


_AUTO_CONTEXT_CACHE: dict[
    tuple[int | None, DeviceIdentity | None, PolicyMode, int],
    PolicyContext,
] = {}


def get_auto_policy(device: object | None = None) -> PolicyContext:
    mode = os.environ.get("B12X_POLICY_MODE", PolicyMode.AUTO.value)
    return PolicyContext.for_device(device, mode=mode)


__all__ = [
    "ComponentPolicy",
    "InvalidPreplannedPolicyError",
    "NO_POLICY_OVERRIDE",
    "PolicyContext",
    "PreplannedPolicyNotFoundError",
    "get_auto_policy",
    "validate_component_profile_contract",
]
