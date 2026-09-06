"""Registry for embedded and test GPU component profiles."""

from __future__ import annotations

from collections.abc import Mapping

from .types import DeviceIdentity, GpuProfile, ProfileHit


class ProfileRegistry:
    """Validated GPU-profile collection with exact hardware ownership."""

    def __init__(self) -> None:
        self._profiles_by_id: dict[str, GpuProfile] = {}
        self._profiles_by_target: dict[DeviceIdentity, GpuProfile] = {}
        self._frozen = False
        self._revision = 0

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def revision(self) -> int:
        return self._revision

    def register(self, profile: GpuProfile) -> None:
        if self._frozen:
            raise RuntimeError("cannot mutate a frozen GPU profile registry")
        if not isinstance(profile, GpuProfile):
            raise TypeError("profile must be a GpuProfile")
        if profile.profile_id in self._profiles_by_id:
            raise ValueError(f"duplicate GPU profile ID {profile.profile_id!r}")
        for target in profile.targets:
            owner = self._profiles_by_target.get(target)
            if owner is not None:
                raise ValueError(
                    f"GPU target {target!r} is already owned by profile "
                    f"{owner.profile_id!r}"
                )
        self._profiles_by_id[profile.profile_id] = profile
        for target in profile.targets:
            self._profiles_by_target[target] = profile
        self._revision += 1

    def freeze(self) -> None:
        self._frozen = True

    def list_profiles(self) -> tuple[GpuProfile, ...]:
        return tuple(
            self._profiles_by_id[profile_id]
            for profile_id in sorted(self._profiles_by_id)
        )

    def get(self, profile_id: str) -> GpuProfile:
        return self._profiles_by_id[profile_id]

    def find(self, device: DeviceIdentity | None) -> GpuProfile | None:
        if device is None:
            return None
        return self._profiles_by_target.get(device)

    def lookup(
        self,
        *,
        profile: GpuProfile,
        component_id: str,
        query: Mapping[str, object],
    ) -> ProfileHit | None:
        component = profile.component(component_id)
        if component is None:
            return None
        rule = component.lookup(query)
        if rule is None:
            return None
        return ProfileHit(
            profile_id=profile.profile_id,
            component_id=component_id,
            rule_name=rule.name,
            config=rule.config,
            evidence=rule.evidence,
        )


EMBEDDED_REGISTRY = ProfileRegistry()


__all__ = ["EMBEDDED_REGISTRY", "ProfileRegistry"]
