"""Planned contract for chunked KDA prefill: caps, plan, bind, run."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from b12x._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor
from b12x._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    dtype_nbytes,
    materialize_scratch_view,
)
from b12x.policy import PolicyContext, get_auto_policy

from .._shared.kda_math import KDA_HEAD_DIM
from .._shared.tensors import (
    canonical_device,
    overlaps,
    positive,
    require_paged_recurrent_state,
    require_row_contiguous,
    require_tensor,
)
from ._policy import (
    CHUNK_TOKENS,
    KDA_PREFILL_POLICY,
    V_SPLIT_CHOICES,
    KdaPrefillQuery,
    WorkspaceRecord,
    tiles_capacity,
)

MetadataValidation = Literal["transactional", "trusted"]


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Static geometry and planned capacity of a KDA prefill plan."""

    device: torch.device | str
    max_tokens: int
    max_seqs: int
    max_state_slots: int
    heads: int
    head_dim: int = KDA_HEAD_DIM
    model_dtype: torch.dtype = torch.bfloat16
    state_dtype: torch.dtype = torch.float32
    qk_l2norm: bool = True
    checkpoint_export: bool = False
    null_state_index: int | None = None
    metadata_validation: MetadataValidation = "transactional"
    chunk_tokens: int = 16

    def __post_init__(self) -> None:
        device = canonical_device(self.device)
        if device.type != "cuda":
            raise ValueError(f"KDA prefill requires a CUDA device, got {device}")
        object.__setattr__(self, "device", device)
        for name in ("max_tokens", "max_seqs", "max_state_slots", "heads"):
            object.__setattr__(self, name, positive(name, getattr(self, name)))
        if self.max_seqs > 4096:
            raise ValueError("max_seqs must be at most 4096")
        if int(self.head_dim) != KDA_HEAD_DIM:
            raise ValueError(f"head_dim must be {KDA_HEAD_DIM}, got {self.head_dim}")
        object.__setattr__(self, "head_dim", KDA_HEAD_DIM)
        if self.model_dtype != torch.bfloat16:
            raise ValueError("model_dtype must be torch.bfloat16")
        if self.state_dtype != torch.float32:
            raise ValueError("state_dtype must be torch.float32")
        if self.chunk_tokens != CHUNK_TOKENS:
            raise ValueError(f"chunk_tokens must be {CHUNK_TOKENS}")
        if self.metadata_validation not in ("transactional", "trusted"):
            raise ValueError("metadata_validation must be 'transactional' or 'trusted'")
        object.__setattr__(self, "qk_l2norm", bool(self.qk_l2norm))
        object.__setattr__(self, "checkpoint_export", bool(self.checkpoint_export))
        if self.null_state_index is not None:
            null = int(self.null_state_index)
            if null < 0 or null >= self.max_state_slots:
                raise ValueError("null_state_index must be a valid slot index")
            object.__setattr__(self, "null_state_index", null)

    @property
    def tiles_capacity(self) -> int:
        """Upper bound on packed chunk tiles: one partial tile per sequence."""
        return tiles_capacity(self.max_tokens, self.max_seqs)


@dataclass(frozen=True)
class Plan:
    """Fixed launch policy and caller-allocated scratch layout for one Caps.

    Chunk tiles are ordered in bands (local tile index major, sequence rank
    minor, longest sequences first) and processed in windows of
    ``window_tiles`` consecutive positions, so every live sequence advances
    in every window. The prepare kernel writes each window into one of two
    workspace ring slots and the recurrence kernel of a window consumes that
    slot while the next window is being prepared. A sequence that continues
    across a window boundary keeps its running state in its final state slot,
    which must therefore not be null. ``max_windows`` is the launch count
    that covers the full capacity.
    """

    caps: Caps
    v_split: int
    k_split: int
    stages: int
    window_tiles: int
    max_windows: int
    duplicate_table_size: int
    offsets: Mapping[str, int]
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    policy_resolution: object | None = None

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    @property
    def recurrence_rows(self) -> int:
        """Grid rows of one recurrence launch: sequences a window can intersect."""
        return min(self.caps.max_seqs, self.window_tiles)

    def launched_windows(self, max_live_tokens: int | None, max_live_seqs: int | None) -> int:
        """Windows to launch for a run bounded by the given live counts."""
        if max_live_tokens is None and max_live_seqs is None:
            return self.max_windows
        tokens = self.caps.max_tokens if max_live_tokens is None else int(max_live_tokens)
        seqs = self.caps.max_seqs if max_live_seqs is None else int(max_live_seqs)
        if tokens < 0 or tokens > self.caps.max_tokens:
            raise ValueError(f"max_live_tokens={tokens} exceeds capacity {self.caps.max_tokens}")
        if seqs < 0 or seqs > self.caps.max_seqs:
            raise ValueError(f"max_live_seqs={seqs} exceeds capacity {self.caps.max_seqs}")
        tiles = tiles_capacity(tokens, seqs)
        return max(1, min(self.max_windows, -(-tiles // self.window_tiles)))

    def output_shape(self, tokens: int | None = None) -> tuple[int, int, int]:
        live_tokens = self.caps.max_tokens if tokens is None else int(tokens)
        if live_tokens < 0 or live_tokens > self.caps.max_tokens:
            raise ValueError(f"tokens={live_tokens} exceeds capacity {self.caps.max_tokens}")
        return (live_tokens, self.caps.heads, KDA_HEAD_DIM)

    def bind(self, **kwargs) -> "Binding":
        return bind(self, **kwargs)


@dataclass(frozen=True)
class Binding:
    """Caller-owned tensors and scratch views for one prefill invocation."""

    plan: Plan
    scratch: torch.Tensor
    error_code: torch.Tensor
    duplicate_slots: torch.Tensor
    band_base: torch.Tensor
    sorted_seq: torch.Tensor
    rank_of: torch.Tensor
    pos_seq: torch.Tensor
    pos_local: torch.Tensor
    window_table: torch.Tensor
    ready_flags: torch.Tensor
    ws: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    raw_g: torch.Tensor
    raw_beta: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    recurrent_state: torch.Tensor
    cu_seqlens: torch.Tensor
    initial_state_indices: torch.Tensor
    final_state_indices: torch.Tensor
    checkpoint_state_indices: torch.Tensor
    checkpoint_offsets: torch.Tensor
    num_seqs: torch.Tensor
    num_tokens: torch.Tensor
    output: torch.Tensor
    token_capacity: int
    seq_capacity: int


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value) - 1).bit_length()


def _query(caps: Caps) -> KdaPrefillQuery:
    return KdaPrefillQuery(
        heads=caps.heads,
        head_dim=caps.head_dim,
        model_dtype=str(caps.model_dtype).removeprefix("torch."),
        state_dtype=str(caps.state_dtype).removeprefix("torch."),
        qk_l2norm=caps.qk_l2norm,
        checkpoint_export=caps.checkpoint_export,
        max_tokens=caps.max_tokens,
        max_seqs=caps.max_seqs,
    )


def _materialize_plan(
    caps: Caps,
    *,
    v_split: int,
    k_split: int,
    stages: int,
    window_tiles: int,
    policy_resolution: object | None,
) -> Plan:
    if v_split not in V_SPLIT_CHOICES:
        raise ValueError(f"v_split must be one of {V_SPLIT_CHOICES}, got {v_split}")
    tiles = caps.tiles_capacity
    heads = caps.heads
    window_tiles = max(1, min(int(window_tiles), tiles))
    max_windows = -(-tiles // window_tiles)
    ring_records = 2 * window_tiles * heads
    duplicate_table_size = _next_power_of_two(4 * caps.max_seqs)
    regions = (
        ("error_code", 1, torch.int32),
        ("duplicate_slots", duplicate_table_size, torch.int32),
        ("band_base", tiles + 2, torch.int32),
        ("sorted_seq", caps.max_seqs, torch.int32),
        ("rank_of", caps.max_seqs, torch.int32),
        ("pos_seq", tiles, torch.int32),
        ("pos_local", tiles, torch.int32),
        ("window_table", 2 * max_windows, torch.int32),
        ("ready_flags", ring_records, torch.int32),
        ("ws", ring_records * WorkspaceRecord.BYTES, torch.uint8),
    )
    offsets: dict[str, int] = {}
    cursor = 0
    for name, elements, dtype in regions:
        cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
        offsets[name] = cursor
        cursor += elements * dtype_nbytes(dtype)
    spec = scratch_buffer_spec("kda_prefill", nbytes=cursor, device=caps.device)
    return Plan(
        caps=caps,
        v_split=int(v_split),
        k_split=int(k_split),
        stages=int(stages),
        window_tiles=window_tiles,
        max_windows=max_windows,
        duplicate_table_size=duplicate_table_size,
        offsets=offsets,
        _scratch_specs=(spec,),
        policy_resolution=policy_resolution,
    )


def plan(caps: Caps, *, policy: PolicyContext | None = None) -> Plan:
    """Resolve the policy once and lay out the scratch for ``caps``."""
    if not isinstance(caps, Caps):
        raise TypeError("caps must be kda_prefill.Caps")
    policy = policy or get_auto_policy(caps.device)
    if not isinstance(policy, PolicyContext):
        raise TypeError("policy must be a PolicyContext")
    policy.require_device(caps.device)
    resolution = policy.resolve(KDA_PREFILL_POLICY, _query(caps))
    return _materialize_plan(
        caps,
        v_split=int(resolution.config.v_split),
        k_split=int(resolution.config.k_split),
        stages=int(resolution.config.stages),
        window_tiles=int(resolution.config.window_tiles),
        policy_resolution=resolution,
    )


def _record_view(
    storage: torch.Tensor, plan: Plan, name: str, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    view, _ = materialize_scratch_view(
        storage, offset_bytes=plan.offsets[name], shape=shape, dtype=dtype
    )
    return view


def bind(
    plan: Plan,
    *,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state_indices: torch.Tensor,
    final_state_indices: torch.Tensor,
    checkpoint_state_indices: torch.Tensor,
    checkpoint_offsets: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    output: torch.Tensor,
) -> Binding:
    """Bind live tensors to a plan without allocating or launching work.

    Live capacities come from the bound tensors: ``q.shape[0]`` tokens and
    ``cu_seqlens.numel() - 1`` sequences, each at most the planned capacity.
    """
    if not isinstance(plan, Plan):
        raise TypeError("plan must be kda_prefill.Plan")
    caps = plan.caps
    device = caps.device
    heads = caps.heads
    tiles = caps.tiles_capacity
    token_capacity = positive("q token capacity", q.shape[0]) if q.dim() == 3 else 0
    if token_capacity > caps.max_tokens:
        raise ValueError(f"token capacity {token_capacity} exceeds planned {caps.max_tokens}")
    seq_capacity = int(cu_seqlens.numel()) - 1
    if seq_capacity < 1 or seq_capacity > caps.max_seqs:
        raise ValueError(
            f"cu_seqlens must hold 2..{caps.max_seqs + 1} entries, got {cu_seqlens.numel()}"
        )
    row_shape = (token_capacity, heads, KDA_HEAD_DIM)
    for name, tensor in (("q", q), ("k", k), ("v", v), ("raw_g", raw_g)):
        require_row_contiguous(name, tensor, shape=row_shape, device=device, dtypes=(torch.bfloat16,))
    require_tensor(
        "raw_beta", raw_beta, shape=(token_capacity, heads), device=device,
        dtypes=(torch.bfloat16,), contiguous=False,
    )
    require_tensor("A_log", A_log, shape=(heads,), device=device, dtypes=(torch.bfloat16, torch.float32))
    require_tensor(
        "dt_bias", dt_bias, shape=(heads, KDA_HEAD_DIM), device=device,
        dtypes=(torch.bfloat16, torch.float32),
    )
    require_paged_recurrent_state(
        recurrent_state,
        shape=(caps.max_state_slots, heads, KDA_HEAD_DIM, KDA_HEAD_DIM),
        device=device,
        dtype=caps.state_dtype,
    )
    require_tensor("cu_seqlens", cu_seqlens, shape=(seq_capacity + 1,), device=device, dtypes=(torch.int32,))
    index_dtypes = (torch.int32, torch.int64)
    for name, tensor in (
        ("initial_state_indices", initial_state_indices),
        ("checkpoint_state_indices", checkpoint_state_indices),
    ):
        require_tensor(name, tensor, shape=(seq_capacity,), device=device, dtypes=index_dtypes)
    require_tensor(
        "final_state_indices",
        final_state_indices,
        shape=(seq_capacity,),
        device=device,
        dtypes=index_dtypes,
        contiguous=False,
    )
    if final_state_indices.stride(0) <= 0:
        raise ValueError("final_state_indices must have a positive stride")
    if not (initial_state_indices.dtype == final_state_indices.dtype == checkpoint_state_indices.dtype):
        raise TypeError("state index tensors must share one dtype")
    require_tensor(
        "checkpoint_offsets", checkpoint_offsets, shape=(seq_capacity,), device=device, dtypes=(torch.int32,)
    )
    for name, tensor in (("num_seqs", num_seqs), ("num_tokens", num_tokens)):
        require_tensor(name, tensor, shape=(1,), device=device, dtypes=(torch.int32,))
    require_row_contiguous("output", output, shape=row_shape, device=device, dtypes=(torch.bfloat16,))

    storage = scratch_tensor(scratch, plan.scratch_specs(), owner="KDA prefill")
    ring = 2 * plan.window_tiles
    views = {
        "error_code": _record_view(storage, plan, "error_code", (1,), torch.int32),
        "duplicate_slots": _record_view(storage, plan, "duplicate_slots", (plan.duplicate_table_size,), torch.int32),
        "band_base": _record_view(storage, plan, "band_base", (tiles + 2,), torch.int32),
        "sorted_seq": _record_view(storage, plan, "sorted_seq", (caps.max_seqs,), torch.int32),
        "rank_of": _record_view(storage, plan, "rank_of", (caps.max_seqs,), torch.int32),
        "pos_seq": _record_view(storage, plan, "pos_seq", (tiles,), torch.int32),
        "pos_local": _record_view(storage, plan, "pos_local", (tiles,), torch.int32),
        "window_table": _record_view(storage, plan, "window_table", (plan.max_windows, 2), torch.int32),
        "ready_flags": _record_view(storage, plan, "ready_flags", (ring, heads), torch.int32),
        "ws": _record_view(storage, plan, "ws", (ring, heads, WorkspaceRecord.BYTES), torch.uint8),
    }
    mutable = {"scratch": storage, "recurrent_state": recurrent_state, "output": output}
    read_only = {
        "q": q, "k": k, "v": v, "raw_g": raw_g, "raw_beta": raw_beta, "A_log": A_log,
        "dt_bias": dt_bias, "cu_seqlens": cu_seqlens,
        "initial_state_indices": initial_state_indices,
        "final_state_indices": final_state_indices,
        "checkpoint_state_indices": checkpoint_state_indices,
        "checkpoint_offsets": checkpoint_offsets, "num_seqs": num_seqs, "num_tokens": num_tokens,
    }
    names = list(mutable)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            if overlaps(mutable[names[left]], mutable[names[right]]):
                raise ValueError(f"{names[left]} and {names[right]} must not overlap")
    for name, tensor in mutable.items():
        for other, candidate in read_only.items():
            if overlaps(tensor, candidate):
                raise ValueError(f"{name} must not overlap read-only tensor {other}")
    return Binding(
        plan=plan,
        scratch=storage,
        **views,
        q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta, A_log=A_log, dt_bias=dt_bias,
        recurrent_state=recurrent_state, cu_seqlens=cu_seqlens,
        initial_state_indices=initial_state_indices,
        final_state_indices=final_state_indices,
        checkpoint_state_indices=checkpoint_state_indices,
        checkpoint_offsets=checkpoint_offsets, num_seqs=num_seqs, num_tokens=num_tokens,
        output=output, token_capacity=token_capacity, seq_capacity=seq_capacity,
    )


def _check_run_scalars(lower_bound: float, scale: float | None, eps: float) -> tuple[float, float, float]:
    lower_bound_value = float(lower_bound)
    if not math.isfinite(lower_bound_value) or not -5.0 <= lower_bound_value < 0.0:
        raise ValueError(f"lower_bound must be in [-5, 0), got {lower_bound_value}")
    scale_value = KDA_HEAD_DIM**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale_value}")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps_value}")
    return lower_bound_value, scale_value, eps_value


def run(
    binding: Binding,
    *,
    lower_bound: float,
    scale: float | None = None,
    eps: float = 1e-6,
    max_live_tokens: int | None = None,
    max_live_seqs: int | None = None,
) -> torch.Tensor:
    """Run the prologue, prepare, and recurrence kernels; capture safe.

    ``max_live_tokens`` and ``max_live_seqs`` are optional host-side upper
    bounds on the device counts; they only limit how many pipeline windows are
    launched. Under transactional validation a run whose live tiles exceed the
    launched windows fails closed like any other malformed metadata; under
    trusted validation the bounds are part of the caller's contract.

    A sequence whose tiles span more than one pipeline window keeps its
    running state in its final state slot between windows, so such a
    sequence must have a non-null final slot (transactional validation flags
    a null one as an invalid slot).
    """
    if not isinstance(binding, Binding):
        raise TypeError("binding must be kda_prefill.Binding")
    lower_bound_value, scale_value, eps_value = _check_run_scalars(lower_bound, scale, eps)
    windows = binding.plan.launched_windows(max_live_tokens, max_live_seqs)
    from ._cute_kernels import run_prefill

    run_prefill(
        binding, lower_bound=lower_bound_value, scale=scale_value, eps=eps_value, windows=windows
    )
    return binding.output


def prewarm(binding: Binding) -> None:
    """Compile every kernel specialization of ``binding`` without launching."""
    if not isinstance(binding, Binding):
        raise TypeError("binding must be kda_prefill.Binding")
    from ._cute_kernels import prewarm_binding

    prewarm_binding(binding)


__all__ = [
    "Binding",
    "Caps",
    "Plan",
    "bind",
    "plan",
    "prewarm",
    "run",
]
