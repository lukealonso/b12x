"""Cache identity for W4A16 route-pack launch preparation."""

from __future__ import annotations

from typing import Any


def route_pack_prewarm_key(
    device_type: str,
    device_index: int,
    route_ids_dtype: Any,
    packed_route_slots: int,
    route_blocks: int,
    block_size: int,
    num_experts: int,
    mapped: bool,
) -> tuple[object, ...]:
    """Identify the fixed routing arena, device, and expert-map contract."""

    values = {
        "packed_route_slots": packed_route_slots,
        "route_blocks": route_blocks,
        "block_size": block_size,
        "num_experts": num_experts,
    }
    normalized = {name: int(value) for name, value in values.items()}
    invalid = {name: value for name, value in normalized.items() if value < 1}
    if invalid:
        raise ValueError(
            f"route-pack prewarm dimensions must be positive: {invalid}"
        )
    return (
        str(device_type),
        int(device_index),
        str(route_ids_dtype),
        normalized["packed_route_slots"],
        normalized["route_blocks"],
        normalized["block_size"],
        normalized["num_experts"],
        bool(mapped),
    )


__all__ = ["route_pack_prewarm_key"]
