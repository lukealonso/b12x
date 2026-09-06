from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import pytest

import b12x.policy.context as policy_context_impl
from b12x.policy import (
    EMBEDDED_REGISTRY,
    ComponentPolicy,
    ComponentProfile,
    DeviceIdentity,
    ExactDecisionNode,
    FrozenMapping,
    GpuProfile,
    InvalidPreplannedPolicyError,
    MatchRange,
    PolicyContext,
    PolicyMode,
    PolicySource,
    PreplannedPolicyNotFoundError,
    ProfileLeaf,
    ProfileRegistry,
    ProfileRule,
    RangeDecisionNode,
    get_auto_policy,
    list_profiled_components,
)

_DEVICE = DeviceIdentity(
    vendor="nvidia",
    compute_capability=(12, 1),
    sm_count=12,
    product_name="Synthetic GPU",
)


@dataclass(frozen=True)
class _Query:
    family: str
    rows: int


@dataclass(frozen=True)
class _Config:
    backend: str
    workers: int


def _component(
    *,
    heuristic_backend: str = "heuristic",
) -> ComponentPolicy[_Query, _Config]:
    def decode(payload: FrozenMapping) -> _Config:
        return _Config(
            backend=str(payload["backend"]),
            workers=int(payload["workers"]),
        )

    def heuristic(query: _Query, _device: DeviceIdentity | None) -> _Config:
        return _Config(backend=heuristic_backend, workers=query.rows)

    def validate(
        _query: _Query,
        config: _Config,
        _device: DeviceIdentity | None,
    ) -> None:
        if config.workers <= 0:
            raise ValueError("workers must be positive")

    return ComponentPolicy(
        component_id="test.decode",
        query_schema_version=1,
        config_schema_version=1,
        query_fields=frozenset({"family", "rows"}),
        config_fields=frozenset({"backend", "workers"}),
        encode_query=lambda query: {
            "family": query.family,
            "rows": query.rows,
        },
        decode_profile=decode,
        heuristic=heuristic,
        validate_config=validate,
    )


def _registry(
    *,
    config: dict[str, object] | None = None,
    component_id: str = "test.decode",
) -> ProfileRegistry:
    registry = ProfileRegistry()
    registry.register(
        GpuProfile(
            profile_id="nvidia.synthetic.12sm",
            targets=(_DEVICE,),
            components=(
                ComponentProfile(
                    component_id=component_id,
                    query_schema_version=1,
                    config_schema_version=1,
                    rules=(
                        ProfileRule.create(
                            name="family-a-small",
                            exact={"family": "a"},
                            ranges={"rows": (4, 7)},
                            config=config or {"backend": "planned", "workers": 4},
                            evidence="synthetic.json",
                        ),
                    ),
                ),
            ),
            metadata=FrozenMapping(),
        )
    )
    registry.freeze()
    return registry


def test_auto_prefers_matching_preplanned_config() -> None:
    context = PolicyContext.for_identity(_DEVICE, registry=_registry())

    result = context.resolve(_component(), _Query(family="a", rows=5))

    assert result.source is PolicySource.PREPLANNED
    assert result.config == _Config(backend="planned", workers=4)
    assert result.profile_id == "nvidia.synthetic.12sm"
    assert result.rule_name == "family-a-small"
    assert result.evidence == "synthetic.json"


def test_auto_uses_heuristic_for_uncovered_query_or_device() -> None:
    registry = _registry()
    uncovered = PolicyContext.for_identity(_DEVICE, registry=registry)
    unknown_device = PolicyContext.for_identity(
        DeviceIdentity(
            vendor="nvidia",
            compute_capability=(12, 1),
            sm_count=13,
            product_name="Synthetic GPU",
        ),
        registry=registry,
    )

    assert (
        uncovered.resolve(_component(), _Query(family="a", rows=8)).source
        is PolicySource.HEURISTIC
    )
    assert (
        unknown_device.resolve(_component(), _Query(family="a", rows=5)).source
        is PolicySource.HEURISTIC
    )


def test_auto_warns_once_per_missing_profile_query(caplog) -> None:
    component = replace(_component(), component_id="test.warning")
    context = PolicyContext.for_identity(
        _DEVICE,
        registry=_registry(component_id=component.component_id),
    )

    with caplog.at_level(logging.WARNING, logger="b12x.policy.context"):
        context.resolve(component, _Query(family="a", rows=8))
        context.resolve(component, _Query(family="b", rows=6))
        context.resolve(component, _Query(family="a", rows=8))

    messages = [
        record.getMessage()
        for record in caplog.records
        if "test.warning is using a heuristic" in record.getMessage()
    ]
    assert len(messages) == 2
    assert all("does not cover the query" in message for message in messages)
    assert any("'family': 'a'" in message for message in messages)
    assert any("'family': 'b'" in message for message in messages)


def test_heuristic_only_does_not_warn(caplog) -> None:
    component = replace(_component(), component_id="test.explicit_heuristic")
    context = PolicyContext.for_identity(
        _DEVICE,
        mode=PolicyMode.HEURISTIC_ONLY,
        registry=_registry(),
    )

    with caplog.at_level(logging.WARNING, logger="b12x.policy.context"):
        context.resolve(component, _Query(family="a", rows=5))

    assert not any(
        "test.explicit_heuristic is using a heuristic" in record.getMessage()
        for record in caplog.records
    )


def test_explicit_override_precedes_profile() -> None:
    context = PolicyContext.for_identity(_DEVICE, registry=_registry())
    override = _Config(backend="override", workers=2)

    result = context.resolve(
        _component(),
        _Query(family="a", rows=5),
        override=override,
    )

    assert result.source is PolicySource.OVERRIDE
    assert result.config is override


def test_context_override_precedes_profile_and_is_immutable() -> None:
    original = PolicyContext.for_identity(_DEVICE, registry=_registry())
    override = _Config(backend="override", workers=3)
    configured = original.with_override("test.decode", override)

    original_result = original.resolve(_component(), _Query(family="a", rows=5))
    configured_result = configured.resolve(
        _component(),
        _Query(family="a", rows=5),
    )

    assert original_result.source is PolicySource.PREPLANNED
    assert configured_result.source is PolicySource.OVERRIDE
    assert configured_result.config is override


def test_repeated_preplanned_resolution_is_cached() -> None:
    base = _component()
    decode_calls = 0
    validate_calls = 0

    def decode(payload: FrozenMapping) -> _Config:
        nonlocal decode_calls
        decode_calls += 1
        return base.decode_profile(payload)

    def validate(
        query: _Query,
        config: _Config,
        device: DeviceIdentity | None,
    ) -> None:
        nonlocal validate_calls
        validate_calls += 1
        base.validate_config(query, config, device)

    component = replace(
        base,
        decode_profile=decode,
        validate_config=validate,
    )
    context = PolicyContext.for_identity(_DEVICE, registry=_registry())

    first = context.resolve(component, _Query(family="a", rows=5))
    second = context.resolve(component, _Query(family="a", rows=5))

    assert second is first
    assert decode_calls == 1
    assert validate_calls == 1


def test_repeated_heuristic_resolution_is_cached() -> None:
    base = replace(_component(), component_id="test.cached_heuristic")
    heuristic_calls = 0

    def heuristic(query: _Query, device: DeviceIdentity | None) -> _Config:
        nonlocal heuristic_calls
        heuristic_calls += 1
        return base.heuristic(query, device)

    component = replace(base, heuristic=heuristic)
    context = PolicyContext.for_identity(_DEVICE, registry=_registry())

    first = context.resolve(component, _Query(family="b", rows=5))
    second = context.resolve(component, _Query(family="b", rows=5))

    assert second is first
    assert heuristic_calls == 1


def test_call_override_bypasses_resolution_cache() -> None:
    context = PolicyContext.for_identity(_DEVICE, registry=_registry())
    query = _Query(family="a", rows=5)
    first_config = _Config(backend="first", workers=1)
    second_config = _Config(backend="second", workers=2)

    first = context.resolve(_component(), query, override=first_config)
    second = context.resolve(_component(), query, override=second_config)

    assert first.config is first_config
    assert second.config is second_config


def test_profile_contract_is_validated_once_per_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = policy_context_impl.validate_component_profile_contract

    def validate_contract(component, profile) -> None:
        nonlocal calls
        calls += 1
        original(component, profile)

    monkeypatch.setattr(
        policy_context_impl,
        "validate_component_profile_contract",
        validate_contract,
    )
    context = PolicyContext.for_identity(_DEVICE, registry=_registry())
    component = _component()

    context.resolve(component, _Query(family="a", rows=4))
    context.resolve(component, _Query(family="a", rows=7))

    assert calls == 1


def test_call_override_precedes_context_override() -> None:
    context_config = _Config(backend="context", workers=3)
    call_config = _Config(backend="call", workers=2)
    context = PolicyContext.for_identity(
        _DEVICE,
        registry=_registry(),
    ).with_override("test.decode", context_config)

    result = context.resolve(
        _component(),
        _Query(family="a", rows=5),
        override=call_config,
    )

    assert result.source is PolicySource.OVERRIDE
    assert result.config is call_config


def test_heuristic_only_and_preplanned_only_modes() -> None:
    registry = _registry()
    heuristic = PolicyContext.for_identity(
        _DEVICE,
        mode=PolicyMode.HEURISTIC_ONLY,
        registry=registry,
    )
    strict = PolicyContext.for_identity(
        _DEVICE,
        mode=PolicyMode.PREPLANNED_ONLY,
        registry=registry,
    )

    result = heuristic.resolve(_component(), _Query(family="a", rows=5))
    assert result.source is PolicySource.HEURISTIC
    with pytest.raises(PreplannedPolicyNotFoundError):
        strict.resolve(_component(), _Query(family="a", rows=8))


def test_preplanned_only_miss_reports_the_query() -> None:
    context = PolicyContext.for_identity(
        _DEVICE,
        mode=PolicyMode.PREPLANNED_ONLY,
        registry=_registry(),
    )

    with pytest.raises(
        PreplannedPolicyNotFoundError,
        match=r"query=\{'family': 'a', 'rows': 8\}",
    ):
        context.resolve(_component(), _Query(family="a", rows=8))


def test_invalid_matching_profile_fails_closed() -> None:
    context = PolicyContext.for_identity(
        _DEVICE,
        registry=_registry(
            config={"backend": "planned", "workers": 0},
        ),
    )

    with pytest.raises(
        InvalidPreplannedPolicyError,
        match="invalid preplanned test.decode config",
    ):
        context.resolve(_component(), _Query(family="a", rows=5))


def test_equal_priority_overlapping_rules_are_rejected() -> None:
    left = ProfileRule.create(
        name="left",
        exact={"family": "a"},
        ranges={"rows": (1, 5)},
        config={"backend": "left", "workers": 1},
    )
    right = ProfileRule.create(
        name="right",
        exact={"family": "a"},
        ranges={"rows": (5, 8)},
        config={"backend": "right", "workers": 1},
    )

    with pytest.raises(ValueError, match="overlap"):
        ComponentProfile(
            component_id="test.decode",
            query_schema_version=1,
            config_schema_version=1,
            rules=(left, right),
        )


def test_embedded_gb10_profile_is_component_scoped() -> None:
    profile = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm")

    assert [component.component_id for component in profile.components] == [
        str(item.component_id) for item in list_profiled_components()
    ]
    assert profile.targets[0] == DeviceIdentity(
        vendor="nvidia",
        compute_capability=(12, 1),
        sm_count=48,
        product_name="NVIDIA GB10",
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto", PolicyMode.AUTO),
        ("heuristic-only", PolicyMode.HEURISTIC_ONLY),
        ("preplanned-only", PolicyMode.PREPLANNED_ONLY),
    ],
)
def test_get_auto_policy_honors_environment_mode(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: PolicyMode,
) -> None:
    monkeypatch.setenv("B12X_POLICY_MODE", value)

    assert get_auto_policy("cpu").mode is expected


def test_get_auto_policy_rejects_invalid_environment_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_POLICY_MODE", "invalid")

    with pytest.raises(ValueError, match="invalid"):
        get_auto_policy("cpu")


def test_explicit_policy_context_ignores_environment_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_POLICY_MODE", "heuristic-only")

    assert PolicyContext.for_device("cpu").mode is PolicyMode.AUTO


def test_generated_axis_tree_resolves_in_tree_depth() -> None:
    planner = ExactDecisionNode(
        field="family",
        branches=(
            (
                "a",
                RangeDecisionNode(
                    field="rows",
                    branches=(
                        (
                            MatchRange(4, 7),
                            ProfileLeaf.create(
                                name="family-a-m4-m7",
                                config={"backend": "planned", "workers": 4},
                                evidence="synthetic-tree.json",
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    registry = ProfileRegistry()
    registry.register(
        GpuProfile(
            profile_id="nvidia.synthetic.tree",
            targets=(_DEVICE,),
            components=(
                ComponentProfile(
                    component_id="test.decode",
                    query_schema_version=1,
                    config_schema_version=1,
                    planner=planner,
                    coverage=FrozenMapping(
                        {"corpus": "synthetic-v1", "query_points": 4}
                    ),
                ),
            ),
        )
    )
    registry.freeze()
    context = PolicyContext.for_identity(_DEVICE, registry=registry)

    hit = context.resolve(_component(), _Query(family="a", rows=5))
    miss = context.resolve(_component(), _Query(family="a", rows=8))

    assert hit.source is PolicySource.PREPLANNED
    assert hit.rule_name == "family-a-m4-m7"
    assert hit.evidence == "synthetic-tree.json"
    assert hit.config == _Config(backend="planned", workers=4)
    assert miss.source is PolicySource.HEURISTIC
