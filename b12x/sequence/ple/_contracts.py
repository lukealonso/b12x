"""Capacity plans and runtime bindings for PLE kernels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from b12x._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor
from b12x._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    dtype_nbytes,
    materialize_scratch_view,
)
from b12x.policy import PolicyContext, get_auto_policy

from ._policy import PLE_POLICY, PleQuery


def _canonical_device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


def _require_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_conv_state_tensor(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"conv_state must have shape {shape}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != dtype:
        raise TypeError(f"conv_state must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"conv_state must be on {device}, got {tensor.device}")

    _, channels, state_capacity = shape
    expected_inner_strides = (state_capacity, 1)
    if tuple(tensor.stride()[1:]) != expected_inner_strides:
        raise ValueError(
            "conv_state must be dense within each state slot with inner strides "
            f"{expected_inner_strides}, got {tuple(tensor.stride()[1:])}"
        )
    minimum_slot_stride = channels * state_capacity
    if tensor.stride(0) < minimum_slot_stride:
        raise ValueError(
            "conv_state state-slot stride must be at least "
            f"{minimum_slot_stride}, got {tensor.stride(0)}"
        )


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    start = int(tensor.untyped_storage().data_ptr()) + int(
        tensor.storage_offset()
    ) * int(tensor.element_size())
    if tensor.numel() == 0:
        return start, start
    span_elements = 1 + sum(
        (int(size) - 1) * int(stride)
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return start, start + span_elements * int(tensor.element_size())


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_start, left_end = _byte_interval(left)
    right_start, right_end = _byte_interval(right)
    return left_start < right_end and right_start < left_end


def _require_mutation_alias_contract(
    *,
    mutable: tuple[tuple[str, torch.Tensor], ...],
    read_only: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    for index, (left_name, left) in enumerate(mutable):
        for right_name, right in mutable[index + 1 :]:
            if _overlaps(left, right):
                raise ValueError(
                    f"mutable buffers {left_name} and {right_name} must not overlap"
                )
        for right_name, right in read_only:
            if _overlaps(left, right):
                raise ValueError(
                    f"mutable buffer {left_name} must not overlap read-only "
                    f"tensor {right_name}"
                )


@dataclass(frozen=True, kw_only=True)
class LayerCaps:
    """Capacity contract for the stateful PLE residual contribution."""

    device: torch.device | str
    mode: str
    max_tokens: int
    max_seqs: int
    max_state_slots: int
    max_speculative_tokens: int
    streams: int
    hidden_size: int
    kernel_size: int
    dilation: int
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _canonical_device(self.device))
        mode = str(self.mode)
        if mode not in ("decode", "prefill", "mixed"):
            raise ValueError(
                f"PLE mode must be decode, prefill, or mixed, got {mode!r}"
            )
        object.__setattr__(self, "mode", mode)
        for name in (
            "max_tokens",
            "max_seqs",
            "max_state_slots",
            "streams",
            "hidden_size",
            "dilation",
        ):
            object.__setattr__(self, name, _require_positive(name, getattr(self, name)))
        kernel_size = int(self.kernel_size)
        if kernel_size < 2:
            raise ValueError(f"kernel_size must be at least 2, got {kernel_size}")
        object.__setattr__(self, "kernel_size", kernel_size)
        max_speculative_tokens = int(self.max_speculative_tokens)
        if max_speculative_tokens < 0:
            raise ValueError(
                "max_speculative_tokens must be nonnegative, got "
                f"{max_speculative_tokens}"
            )
        object.__setattr__(self, "max_speculative_tokens", max_speculative_tokens)
        if self.dtype != torch.bfloat16:
            raise TypeError(
                f"PLE layer currently requires torch.bfloat16, got {self.dtype}"
            )

    @property
    def channels(self) -> int:
        return self.streams * self.hidden_size

    @property
    def state_length(self) -> int:
        return self.dilation * (self.kernel_size - 1)

    @property
    def state_capacity(self) -> int:
        return self.state_length + self.max_speculative_tokens


@dataclass(frozen=True, kw_only=True)
class LayerBinding:
    """Caller-owned PLE residual inputs, convolution state, and scratch views.

    ``conv_state`` and ``out`` are mutable. Projection tensors and norm weights
    are read-only; normalized inputs and gathered state are scratch views.
    """

    plan: LayerPlan
    scratch: torch.Tensor
    residual: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    k_norm_weight: torch.Tensor
    q_norm_weight: torch.Tensor
    u_norm_weight: torch.Tensor
    conv_weight: torch.Tensor
    query_start_loc: torch.Tensor
    state_slot_ids: torch.Tensor
    state_is_fresh: torch.Tensor
    num_accepted_tokens: torch.Tensor
    num_seqs: torch.Tensor
    num_tokens: torch.Tensor
    conv_state: torch.Tensor
    out: torch.Tensor
    request_is_prefill: torch.Tensor | None
    normalized_u: torch.Tensor
    gathered_state: torch.Tensor
    request_ids: torch.Tensor
    error_code: torch.Tensor


@dataclass(frozen=True)
class _LayerScratchLayout:
    nbytes: int
    normalized_u_offset_bytes: int
    gathered_state_offset_bytes: int
    request_ids_offset_bytes: int
    error_code_offset_bytes: int


@dataclass(frozen=True, kw_only=True)
class LayerPlan:
    """Fixed PLE residual-state geometry and scratch-buffer contract."""

    caps: LayerCaps
    layout: _LayerScratchLayout
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    policy_resolution: object | None = None

    @property
    def state_length(self) -> int:
        return self.caps.state_length

    @property
    def state_capacity(self) -> int:
        return self.caps.state_capacity

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(self, **kwargs) -> LayerBinding:
        return bind_layer(self, **kwargs)


def plan_layer(
    caps: LayerCaps,
    *,
    policy: PolicyContext | None = None,
) -> LayerPlan:
    """Plan fixed-capacity PLE math and state-gather scratch."""
    if not isinstance(caps, LayerCaps):
        raise TypeError("caps must be LayerCaps")
    policy = policy or get_auto_policy(caps.device)
    if not isinstance(policy, PolicyContext):
        raise TypeError("policy must be a PolicyContext")
    policy.require_device(caps.device)
    resolution = policy.resolve(
        PLE_POLICY,
        PleQuery(
            mode=caps.mode,
            dtype=str(caps.dtype).removeprefix("torch."),
            max_tokens=caps.max_tokens,
            max_seqs=caps.max_seqs,
            max_speculative_tokens=caps.max_speculative_tokens,
            streams=caps.streams,
            hidden_size=caps.hidden_size,
            kernel_size=caps.kernel_size,
            dilation=caps.dilation,
        ),
    )
    normalized_u_offset_bytes = align_up(0, SCRATCH_ALIGN_BYTES)
    cursor = normalized_u_offset_bytes
    cursor += caps.max_tokens * caps.channels * dtype_nbytes(caps.dtype)
    gathered_state_offset_bytes = align_up(cursor, SCRATCH_ALIGN_BYTES)
    cursor = gathered_state_offset_bytes
    cursor += (
        caps.max_seqs * caps.channels * caps.state_length * dtype_nbytes(caps.dtype)
    )
    request_ids_offset_bytes = align_up(cursor, SCRATCH_ALIGN_BYTES)
    cursor = request_ids_offset_bytes + caps.max_tokens * dtype_nbytes(torch.int32)
    error_code_offset_bytes = align_up(cursor, SCRATCH_ALIGN_BYTES)
    cursor = error_code_offset_bytes + dtype_nbytes(torch.int32)
    layout = _LayerScratchLayout(
        nbytes=cursor,
        normalized_u_offset_bytes=normalized_u_offset_bytes,
        gathered_state_offset_bytes=gathered_state_offset_bytes,
        request_ids_offset_bytes=request_ids_offset_bytes,
        error_code_offset_bytes=error_code_offset_bytes,
    )
    spec = scratch_buffer_spec("ple_layer", nbytes=cursor, device=caps.device)
    return LayerPlan(
        caps=caps,
        layout=layout,
        _scratch_specs=(spec,),
        policy_resolution=resolution,
    )


def bind_layer(
    plan: LayerPlan,
    *,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    residual: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    u_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_slot_ids: torch.Tensor,
    state_is_fresh: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    conv_state: torch.Tensor,
    out: torch.Tensor,
    request_is_prefill: torch.Tensor | None = None,
) -> LayerBinding:
    """Bind PLE tensors without allocating or reading device metadata."""
    caps = plan.caps
    scratch_storage = scratch_tensor(scratch, plan.scratch_specs(), owner="PLE layer")
    normalized_u, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.layout.normalized_u_offset_bytes,
        shape=(caps.max_tokens, caps.channels),
        dtype=caps.dtype,
    )
    gathered_state, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.layout.gathered_state_offset_bytes,
        shape=(caps.max_seqs, caps.channels, caps.state_length),
        dtype=caps.dtype,
    )
    request_ids, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.layout.request_ids_offset_bytes,
        shape=(caps.max_tokens,),
        dtype=torch.int32,
    )
    error_code, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.layout.error_code_offset_bytes,
        shape=(1,),
        dtype=torch.int32,
    )
    common = {
        "dtype": caps.dtype,
        "device": caps.device,
    }
    _require_tensor(
        "residual",
        residual,
        shape=(caps.max_tokens, caps.streams, caps.hidden_size),
        **common,
    )
    _require_tensor(
        "key", key, shape=(caps.max_tokens, caps.streams, caps.hidden_size), **common
    )
    _require_tensor("value", value, shape=(caps.max_tokens, caps.hidden_size), **common)
    for name, tensor in (
        ("k_norm_weight", k_norm_weight),
        ("q_norm_weight", q_norm_weight),
        ("u_norm_weight", u_norm_weight),
    ):
        _require_tensor(
            name,
            tensor,
            shape=(caps.channels,),
            **common,
        )
    _require_tensor(
        "conv_weight",
        conv_weight,
        shape=(caps.channels, caps.kernel_size),
        **common,
    )
    _require_tensor(
        "query_start_loc",
        query_start_loc,
        shape=(caps.max_seqs + 1,),
        dtype=torch.int32,
        device=caps.device,
    )
    _require_tensor(
        "state_slot_ids",
        state_slot_ids,
        shape=(caps.max_seqs,),
        dtype=torch.int64,
        device=caps.device,
    )
    _require_tensor(
        "state_is_fresh",
        state_is_fresh,
        shape=(caps.max_seqs,),
        dtype=torch.bool,
        device=caps.device,
    )
    _require_tensor(
        "num_accepted_tokens",
        num_accepted_tokens,
        shape=(caps.max_seqs,),
        dtype=torch.int32,
        device=caps.device,
    )
    for name, tensor in (("num_seqs", num_seqs), ("num_tokens", num_tokens)):
        _require_tensor(
            name,
            tensor,
            shape=(1,),
            dtype=torch.int32,
            device=caps.device,
        )
    _require_conv_state_tensor(
        conv_state,
        shape=(caps.max_state_slots, caps.channels, caps.state_capacity),
        **common,
    )
    _require_tensor(
        "out",
        out,
        shape=(caps.max_tokens, caps.streams, caps.hidden_size),
        **common,
    )
    if caps.mode == "mixed":
        if request_is_prefill is None:
            raise ValueError("request_is_prefill is required for a mixed PLE LayerPlan")
        _require_tensor(
            "request_is_prefill",
            request_is_prefill,
            shape=(caps.max_seqs,),
            dtype=torch.bool,
            device=caps.device,
        )
    elif request_is_prefill is not None:
        raise ValueError("request_is_prefill is only valid for a mixed PLE LayerPlan")
    read_only = (
        ("residual", residual),
        ("key", key),
        ("value", value),
        ("k_norm_weight", k_norm_weight),
        ("q_norm_weight", q_norm_weight),
        ("u_norm_weight", u_norm_weight),
        ("conv_weight", conv_weight),
        ("query_start_loc", query_start_loc),
        ("state_slot_ids", state_slot_ids),
        ("state_is_fresh", state_is_fresh),
        ("num_accepted_tokens", num_accepted_tokens),
        ("num_seqs", num_seqs),
        ("num_tokens", num_tokens),
    )
    if request_is_prefill is not None:
        read_only += (("request_is_prefill", request_is_prefill),)
    _require_mutation_alias_contract(
        mutable=(
            ("scratch", scratch_storage),
            ("conv_state", conv_state),
            ("out", out),
        ),
        read_only=read_only,
    )
    return LayerBinding(
        plan=plan,
        scratch=scratch_storage,
        residual=residual,
        key=key,
        value=value,
        k_norm_weight=k_norm_weight,
        q_norm_weight=q_norm_weight,
        u_norm_weight=u_norm_weight,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        conv_state=conv_state,
        out=out,
        request_is_prefill=request_is_prefill,
        normalized_u=normalized_u,
        gathered_state=gathered_state,
        request_ids=request_ids,
        error_code=error_code,
    )


def run_decode(binding: LayerBinding, *, eps: float) -> torch.Tensor:
    """Run decode and mutate only live nonnegative state slots."""
    if binding.plan.caps.mode != "decode":
        raise ValueError("run_decode requires a decode LayerPlan")
    if binding.plan.caps.device.type != "cuda":
        raise ValueError("PLE GPU run requires CUDA; use the explicit reference oracle")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0:
        raise ValueError(f"eps must be finite and positive, got {eps_value}")
    from ._kernels import run_layer_kernels

    run_layer_kernels(binding, eps=eps_value, decode=True)
    return binding.out


def run_prefill(binding: LayerBinding, *, eps: float) -> torch.Tensor:
    """Run packed prefill and persist each live request's newest state."""
    if binding.plan.caps.mode != "prefill":
        raise ValueError("run_prefill requires a prefill LayerPlan")
    if binding.plan.caps.device.type != "cuda":
        raise ValueError("PLE GPU run requires CUDA; use the explicit reference oracle")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0:
        raise ValueError(f"eps must be finite and positive, got {eps_value}")
    from ._kernels import run_layer_kernels

    run_layer_kernels(binding, eps=eps_value, decode=False)
    return binding.out


def run_mixed(
    binding: LayerBinding, *, eps: float, token_count: int | None = None
) -> torch.Tensor:
    """Run packed prefill and decode requests without reordering rows.

    Args:
        binding: Fixed-capacity inputs, output, and recurrent state.
        eps: Positive normalization epsilon.
        token_count: Host launch bound, defaulting to the planned capacity.
            Device metadata may select fewer tokens during graph replay.

    Returns:
        The output prefix of length token_count. Rows beyond it are untouched.
    """
    if binding.plan.caps.mode != "mixed":
        raise ValueError("run_mixed requires a mixed LayerPlan")
    if binding.plan.caps.device.type != "cuda":
        raise ValueError("PLE GPU run requires CUDA; use the explicit reference oracle")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0:
        raise ValueError(f"eps must be finite and positive, got {eps_value}")
    if token_count is None:
        token_count = binding.plan.caps.max_tokens
    if not 0 <= token_count <= binding.plan.caps.max_tokens:
        raise ValueError("token_count must fit the planned token capacity")
    from ._kernels import run_layer_mixed_kernels

    run_layer_mixed_kernels(binding, eps=eps_value, token_count=token_count)
    return binding.out[:token_count]


__all__ = [
    "LayerCaps",
    "LayerPlan",
    "LayerBinding",
    "plan_layer",
    "bind_layer",
    "run_decode",
    "run_prefill",
    "run_mixed",
]
