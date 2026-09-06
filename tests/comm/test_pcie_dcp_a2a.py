from __future__ import annotations

import inspect
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie import kimi_topk16
from b12x.comm.pcie.pcie_dcp_a2a import (
    PCIeDCPA2A,
    PCIeDCPA2APool,
    _SINGLE_CHANNEL_ID,
    _staging_layout,
    lse_reduce_scatter_reference,
)


class _FakeExt:
    def __init__(self) -> None:
        self.disposed = []

    def init_dcp_a2a(self, *args) -> int:
        return 1234

    def dispose(self, pointer: int) -> None:
        self.disposed.append(pointer)


class _FakeRuntime(PCIeDCPA2A):
    def __init__(self) -> None:
        self.run_calls = []
        super().__init__(
            rank=0,
            world_size=2,
            device=torch.device("cpu"),
            signal_ptrs=(100, 200),
            staging0_ptrs=(300, 400),
            staging1_ptrs=(500, 600),
            max_batch_size=4,
            total_heads=32,
            head_dim=64,
            output_capacity_elems=4 * 32 * 64,
            lse_offset=4 * 32 * 64 * 2,
            lse_capacity=4 * 32,
            ext_module=_FakeExt(),
        )

    def _launch_lse_reduce_scatter(
        self,
        partial_output,
        partial_lse,
        out,
        *,
        slot,
        natural_log,
        threads,
        blocks,
        device_slot_selection,
    ):
        self.run_calls.append(
            (
                slot,
                natural_log,
                threads,
                blocks,
                device_slot_selection,
                tuple(partial_output.shape),
            )
        )
        heads_per_rank = partial_output.shape[1] // 2
        out.copy_(partial_output[:, :heads_per_rank])

    def _launch_all_gather_heads(
        self,
        local_input,
        out,
        *,
        slot,
        threads,
        blocks,
        device_slot_selection,
    ):
        self.run_calls.append(
            (
                slot,
                "all_gather_heads",
                threads,
                blocks,
                device_slot_selection,
                tuple(local_input.shape),
            )
        )
        out.copy_(torch.cat((local_input, local_input), dim=1))

    def _launch_all_gather_pair(
        self,
        local_first,
        local_second,
        out_first,
        out_second,
        *,
        slot,
        threads,
        device_slot_selection,
    ):
        self.run_calls.append(
            (
                slot,
                "all_gather_pair",
                threads,
                device_slot_selection,
                tuple(local_first.shape),
                tuple(local_second.shape),
            )
        )
        out_first.copy_(torch.cat((local_first, local_first), dim=1))
        out_second.copy_(torch.cat((local_second, local_second), dim=1))


class _FakeKimiRuntime(PCIeDCPA2A):
    def _launch_all_gather_pair_kimi_topk(
        self,
        local_down,
        local_router,
        correction_bias,
        out_down,
        topk_weights,
        topk_ids,
        *,
        slot,
        device_slot_selection,
    ):
        del local_router, correction_bias, slot, device_slot_selection
        out_down.copy_(torch.cat((local_down,) * self.world_size, dim=1))
        topk_weights.fill_(1.0 / 16.0)
        topk_ids.copy_(torch.arange(16, dtype=torch.int32).view(1, 16))

    def _launch_kimi_topk16(
        self,
        router_logits,
        correction_bias,
        output_weights,
        output_ids,
        *,
        threads,
    ):
        del router_logits, correction_bias, threads
        output_weights.fill_(1.0 / 16.0)
        output_ids.copy_(
            torch.arange(16, dtype=torch.int32)
            .view(1, 16)
            .expand(output_ids.shape[0], -1)
        )


def _make_runtime() -> PCIeDCPA2A:
    return _FakeRuntime()


def _make_kimi_runtime(
    world_size: int,
    ext: _FakeExt | None = None,
    *,
    max_batch_size: int = 8,
) -> PCIeDCPA2A:
    query_head_dim = 7168 // world_size + 3584 // world_size
    return _FakeKimiRuntime(
        rank=0,
        world_size=world_size,
        device=torch.device("cpu"),
        signal_ptrs=tuple(range(100, 100 + world_size)),
        staging0_ptrs=tuple(range(200, 200 + world_size)),
        staging1_ptrs=tuple(range(300, 300 + world_size)),
        max_batch_size=max_batch_size,
        total_heads=world_size,
        head_dim=query_head_dim,
        output_capacity_elems=max_batch_size * world_size * query_head_dim,
        lse_offset=max_batch_size * world_size * query_head_dim * 2,
        lse_capacity=max_batch_size * world_size,
        query_head_dim=query_head_dim,
        ext_module=ext or _FakeExt(),
    )


def test_staging_layout_has_aligned_disjoint_slots():
    layout = _staging_layout(
        signal_bytes=12345,
        world_size=2,
        max_batch_size=4,
        total_heads=32,
        head_dim=512,
    )

    assert layout.staging0_offset % 256 == 0
    assert layout.staging1_offset == layout.staging0_offset + layout.slot_bytes
    assert layout.lse_offset % 256 == 0
    assert layout.slab_bytes == layout.staging1_offset + layout.slot_bytes
    assert layout.output_capacity_elems >= 4 * 32 * 512
    assert layout.lse_capacity >= 4 * 32

    wider_query_layout = _staging_layout(
        signal_bytes=12345,
        world_size=2,
        max_batch_size=4,
        total_heads=32,
        head_dim=512,
        query_head_dim=576,
    )
    assert wider_query_layout.output_capacity_elems >= 4 * 32 * 576
    assert wider_query_layout.lse_offset > layout.lse_offset


def test_graph_epoch_uses_only_barrier_record_padding() -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    record_words = kernels._SELF_COUNTER_WORDS
    assert record_words + 1 == kernels._GRAPH_EPOCH_INDEX
    assert record_words + 2 == kernels._GRAPH_ARRIVED_INDEX
    assert record_words + 32 > kernels._GRAPH_ARRIVED_INDEX


def test_a2a_graph_epoch_tail_uses_serialized_stream_fast_path() -> None:
    from b12x.comm.pcie import _cute_intrinsics, _dcp_a2a_cute as kernels

    source = inspect.getsource(kernels._a2a_graph_epoch_arrive)
    assert "setp.eq.u32 single_block, $2, 1;" in source
    assert "@single_block bra a2a_epoch_advance;" in source
    assert source.count("atom.global.add.u32") == 1
    assert "atom.global.add.u32 prior, [$1], 1;" in source
    assert "st.global.u32 [$1], 0;" in source
    assert "ld.global.u32 generation, [$0];" in source
    assert "st.global.u32 [$0], generation;" in source
    assert "fence.sc.gpu" not in source
    assert "generation" not in inspect.signature(
        kernels._a2a_graph_epoch_arrive
    ).parameters

    assert "_a2a_graph_epoch_arrive(" in inspect.getsource(
        kernels._LseReduceScatterLaunch.kernel
    )
    assert "_a2a_graph_epoch_arrive(" in inspect.getsource(
        kernels._AllGatherHeadsLaunch.kernel
    )

    # OneShot and TwoShot retain the stronger shared primitive.
    shared_source = inspect.getsource(_cute_intrinsics.graph_epoch_arrive)
    assert "fence.sc.gpu" in shared_source
    assert shared_source.count("atom.global.add.u32") == 2


def test_a2a_epoch_change_bumps_both_compile_specs() -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    identities = (
        (
            kernels._get_compiled_lse_reduce_scatter,
            "comm.pcie.dcp_a2a.lse_reduce_scatter",
            36,
        ),
        (
            kernels._get_compiled_all_gather_heads,
            "comm.pcie.dcp_a2a.all_gather_heads",
            13,
        ),
    )
    for launcher, identity, version in identities:
        suffix = inspect.getsource(launcher).split(f'"{identity}",', maxsplit=1)[1]
        assert suffix.lstrip().startswith(f"{version},")


def test_block_pair_barrier_selects_once_and_keeps_scaled_offsets_int64() -> None:
    from b12x.comm.pcie import _dcp_cute_common as common

    source = inspect.getsource(common.block_pair_barrier)
    prefix, body = source.split(
        "if Int32(tidx) < Int32(world_size):", maxsplit=1
    )
    assert "for peer in cutlass.range_constexpr(1, world_size):" not in prefix
    assert "for peer in cutlass.range_constexpr(1, world_size):" in body
    assert "peer_signal_address = Int64(signals[peer].toint())" in body
    assert body.count("_membar_sys()") == 2
    assert body.count("_store_relaxed_sys_u32(") == 1
    assert body.count("_load_relaxed_sys_u32(") == 2
    assert "Int64(bidx) * Int64(_MAX_RANKS)" in body
    assert "Int64(tidx) * Int64(_FLAG_STRIDE)" in body
    assert body.index("_membar_sys()") < body.index("cute.arch.load(self_ptr")
    # The second fence is the push transport's acquire, after the wait only.
    acquire = body.split("if cutlass.const_expr(acquire):", maxsplit=1)[1]
    assert acquire.lstrip().startswith("_membar_sys()")
    assert body.index("while observed != value:") < body.index(
        "if cutlass.const_expr(acquire):"
    )


def test_graph_slot_delta_encoding_keeps_scaled_offsets_64_bit() -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    # The compact graph ABI still represents offsets well beyond Int32; the
    # device kernel widens these units before multiplying by slot parity.
    assert kernels._slot_delta_256b(1 << 38) == 1 << 30
    assert kernels._slot_delta_256b(-(1 << 38)) == -(1 << 30)
    with pytest.raises(ValueError, match="nonzero 256B multiple"):
        kernels._slot_delta_256b(257)
    with pytest.raises(ValueError, match="512 GiB"):
        kernels._slot_delta_256b(1 << 39)


def test_a2a_epoch_load_uses_gpu_scoped_ordering() -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    lse_source = inspect.getsource(kernels._LseReduceScatterLaunch.kernel)
    gather_source = inspect.getsource(kernels._AllGatherHeadsLaunch.kernel)
    assert "generation = ld_relaxed_gpu_u32(" in lse_source
    assert "generation = ld_relaxed_gpu_u32(" in gather_source
    assert "generation = ld_global_u32(" not in lse_source
    assert "generation = ld_global_u32(" not in gather_source


def test_lse_log_base_is_a_runtime_kernel_argument() -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    key = kernels._lse_launcher_key(8, 0, "bf16", 256, True)
    assert key == (8, 0, "bf16", 256, True, False)
    assert "natural_log" in inspect.signature(
        kernels._LseReduceScatterLaunch.__call__
    ).parameters
    assert "natural_log" in inspect.signature(
        kernels._LseReduceScatterLaunch.kernel
    ).parameters
    source = inspect.getsource(kernels._LseReduceScatterLaunch.kernel)
    assert source.count("if natural_log != Int32(0):") == 1
    branch = source.split("if natural_log != Int32(0):", maxsplit=1)[1]
    assert "cute.math.exp(delta, fastmath=False)" in branch
    assert "cute.math.exp2(delta, approx=True)" in branch


def test_lse_uses_one_runtime_selected_lse_load_per_lane() -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    source = inspect.getsource(kernels._LseReduceScatterLaunch.kernel)
    read_phase = source.split("block_pair_barrier(", maxsplit=1)[1]
    lse_phase = read_phase.split("weights = cute.make_rmem_tensor", maxsplit=1)[0]

    assert "lane_lse_base = Int64(local_lse.toint())" in lse_phase
    assert (
        "for source_index in cutlass.range_constexpr(1, self._world_size):"
        in lse_phase
    )
    assert "if lane == Int32(source_index):" in lse_phase
    assert "if lane < Int32(self._world_size):" in lse_phase
    assert lse_phase.count("lane_lse = ld_generic_f32(") == 1
    assert "source_row * Int64(4)" in lse_phase
    assert "cute.arch.load(local_lse + source_row" not in lse_phase
    assert "cute.arch.load(source_lse + source_row" not in lse_phase

    generic_load = inspect.getsource(kernels.ld_generic_f32)
    assert '"ld.f32 $0, [$1];"' in generic_load
    assert '"=f,l"' in generic_load


def test_lse_payload_pack_loop_matches_native_generic_load_order() -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    source = inspect.getsource(kernels._LseReduceScatterLaunch.kernel)
    read_phase = source.split("block_pair_barrier(", maxsplit=1)[1]
    payload = read_phase.split("staging_base = source_row", maxsplit=1)[1]

    assert "for pack in cutlass.range(" in payload
    assert "unroll=1," in payload
    assert "source_addresses = cute.make_rmem_tensor(" in payload
    assert "payload_row_addresses = cute.make_rmem_tensor(" in payload
    assert "words = _ld_generic_v4_u32(" in payload
    assert "source_addresses[source_index]" in payload
    assert payload.index("normalized_weight =") < payload.index(
        "words = _ld_generic_v4_u32("
    )
    assert "if normalized_weight != Float32(0.0):" in payload
    assert payload.index("words = _ld_generic_v4_u32(") < payload.index(
        "lo, hi = self._unpack_pair(words[pair])"
    )
    assert "at_least_once=True" not in payload
    generic_load = inspect.getsource(kernels._ld_generic_v4_u32)
    assert generic_load.count("ld.v4.b32") == 1
    assert '"=r,=r,=r,=r,l"' in generic_load
    assert "ld.global" not in generic_load
    assert "bar.warp.sync" not in generic_load
    assert payload.count("self._pack_pair(") == 4


def test_graph_lse_forms_remote_slot_addresses_after_the_barrier() -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    source = inspect.getsource(kernels._LseReduceScatterLaunch.kernel)
    write_phase, read_phase = source.split("block_pair_barrier(", maxsplit=1)

    assert "local_staging = staging[self._rank] + slot_offset" in write_phase
    assert "staging = (\n                staging0 + slot_offset" not in write_phase
    assert "Int64(staging[source].toint())" in read_phase
    assert "+ slot_offset\n                    + lse_offset" in read_phase


def test_lse_payload_address_add_is_late_and_opaque() -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    kernel_source = inspect.getsource(kernels._LseReduceScatterLaunch.kernel)
    write_phase, read_phase = kernel_source.split("block_pair_barrier(", maxsplit=1)
    assert "_add_u64_opaque(" not in write_phase
    assert read_phase.index("inv_weight_sum =") < read_phase.index(
        "source_address = _add_u64_opaque("
    )
    assert "slot_offset + staging_base * Int64(16)" in read_phase

    add_source = inspect.getsource(kernels._add_u64_opaque)
    assert '"add.u64 $0, $1, $2;"' in add_source
    assert '"=l,l,l"' in add_source
    assert "has_side_effects=True" in add_source


def test_lse_pair_conversions_match_native_scalar_bf16_contract() -> None:
    from b12x.comm.pcie import _cute_intrinsics, _dcp_a2a_cute as kernels

    unpack_source = inspect.getsource(kernels._LseReduceScatterLaunch._unpack_pair)
    pack_source = inspect.getsource(kernels._LseReduceScatterLaunch._pack_pair)
    assert "unpack_f16x2(value)" in unpack_source
    assert "unpack_bf16x2(value)" in unpack_source
    assert "pack_f32x2_to_f16x2(lo, hi)" in pack_source
    assert "pack_f32x2_to_bf16x2(lo, hi)" in pack_source
    assert "scaled" not in unpack_source

    bf16_unpack = inspect.getsource(_cute_intrinsics.unpack_bf16x2)
    bf16_pack = inspect.getsource(_cute_intrinsics.pack_f32x2_to_bf16x2)
    assert "mov.b32 {lo, hi}" in bf16_unpack
    assert bf16_unpack.count("cvt.f32.bf16") == 2
    assert bf16_pack.count("cvt.rn.bf16.f32") == 2
    assert "SATFINITE" not in bf16_pack


def test_lse_launch_preserves_native_launch_bounds_contract() -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    source = inspect.getsource(kernels._LseReduceScatterLaunch.__call__)
    assert "block=(self._threads, 1, 1)," in source
    assert "max_number_threads=(512, 1, 1)," in source
    assert "min_blocks_per_mp=1," in source


def test_all_gather_resolves_one_peer_address_before_each_copy() -> None:
    """The peer selector must not clone the memory transaction per rank."""

    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    kernel_source = inspect.getsource(kernels._AllGatherHeadsLaunch.kernel)
    read_phase = kernel_source.split("block_pair_barrier(", maxsplit=1)[1]
    assert "for source in cutlass.range_constexpr(self._world_size):" in read_phase
    assert "if source_rank == Int32(source):" in read_phase
    assert "if cutlass.const_expr(source == self._rank):" in read_phase
    assert "source_words = local_input" in read_phase
    assert "source_words = self._staging_words(staging[source])" in read_phase
    assert "source_address = Int64(" in read_phase
    assert "_copy_16b_addr(" in read_phase
    assert "source_words = cute.make_ptr(" not in read_phase


def test_graph_lse_prepare_warms_shared_runtime_log_launcher(
    monkeypatch,
) -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    runtime = _make_runtime()
    calls = []
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._is_current_stream_capturing",
        lambda device: False,
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        kernels,
        "_get_compiled_lse_reduce_scatter",
        lambda *args: calls.append(args),
    )

    runtime.prepare_graph_lse_reduce_scatter(dtype=torch.bfloat16, threads=256)

    assert calls == [(2, 0, "bf16", 256, True, False)]


@pytest.mark.parametrize("natural_log", [False, True])
def test_graph_capture_checks_the_shared_runtime_log_launcher(
    monkeypatch,
    natural_log: bool,
) -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    runtime = _make_runtime()
    lookups = []
    partial_output = torch.zeros(1, 32, 64, dtype=torch.bfloat16)
    partial_lse = torch.zeros(1, 32, dtype=torch.float32)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._is_current_stream_capturing",
        lambda device: True,
    )
    monkeypatch.setattr(
        kernels,
        "is_lse_reduce_scatter_prepared",
        lambda *args: lookups.append(args) or False,
    )

    with pytest.raises(RuntimeError, match="cold PCIe DCP LSE CUDA graph"):
        runtime.lse_reduce_scatter(
            partial_output,
            partial_lse,
            is_lse_base_on_e=natural_log,
        )

    assert lookups == [(2, 0, "bf16", 256, True, False)]


def test_reference_selects_destination_heads_and_combines_lse_weights():
    outputs = torch.tensor(
        [
            [[[[1.0], [2.0], [10.0], [20.0]]]],
            [[[[3.0], [4.0], [30.0], [40.0]]]],
        ],
        dtype=torch.float32,
    ).reshape(2, 1, 4, 1)
    lses = torch.tensor(
        [
            [[0.0, 0.0, 0.0, -torch.inf]],
            [[0.0, -torch.inf, 0.0, -torch.inf]],
        ]
    )

    rank0 = lse_reduce_scatter_reference(outputs, lses, 0)
    rank1 = lse_reduce_scatter_reference(outputs, lses, 1)

    torch.testing.assert_close(rank0, torch.tensor([[[2.0], [2.0]]]))
    torch.testing.assert_close(rank1, torch.tensor([[[20.0], [0.0]]]))


def test_reference_ignores_nan_output_from_empty_shard():
    outputs = torch.tensor(
        [
            [[[[torch.nan], [2.0]]]],
            [[[[4.0], [6.0]]]],
        ],
        dtype=torch.float32,
    ).reshape(2, 1, 2, 1)
    lses = torch.tensor([[[-torch.inf, 0.0]], [[0.0, 0.0]]])

    actual = lse_reduce_scatter_reference(outputs, lses, 0)

    torch.testing.assert_close(actual, torch.tensor([[[4.0]]]))


def test_runtime_validates_and_dispatches_to_cute_plan():
    runtime = _make_runtime()
    partial_output = torch.arange(2 * 32 * 64, dtype=torch.bfloat16).reshape(2, 32, 64)
    partial_lse = torch.zeros(2, 32, dtype=torch.float32)

    out = runtime.lse_reduce_scatter(
        partial_output,
        partial_lse,
        is_lse_base_on_e=False,
        threads=256,
        block_limit=32,
    )

    assert out.shape == (2, 16, 64)
    assert torch.equal(out, partial_output[:, :16])
    assert runtime.run_calls == [(0, False, 256, 4, False, (2, 32, 64))]

    local_input = partial_output[:, :16].contiguous()
    gathered = runtime.all_gather_heads(
        local_input,
        threads=64,
        block_limit=16,
    )
    assert gathered.shape == partial_output.shape
    assert torch.equal(gathered, torch.cat((local_input, local_input), dim=1))
    assert runtime.run_calls[-1] == (
        1,
        "all_gather_heads",
        64,
        16,
        False,
        (2, 16, 64),
    )

    fp8_input = torch.arange(16 * 64, dtype=torch.float32).reshape(1, 16, 64)
    fp8_input = fp8_input.to(torch.float8_e4m3fn)
    fp8_gathered = runtime.all_gather_heads(fp8_input)
    assert fp8_gathered.dtype == torch.float8_e4m3fn
    expected_fp8 = torch.cat((fp8_input, fp8_input), dim=1)
    assert torch.equal(fp8_gathered.view(torch.uint8), expected_fp8.view(torch.uint8))

    local_first = torch.arange(2 * 16, dtype=torch.bfloat16).reshape(2, 16)
    local_second = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8)
    paired_first, paired_second = runtime.all_gather_pair(
        local_first,
        local_second,
        threads=256,
    )
    assert torch.equal(paired_first, torch.cat((local_first, local_first), dim=1))
    assert torch.equal(paired_second, torch.cat((local_second, local_second), dim=1))
    assert runtime.run_calls[-1] == (
        1,
        "all_gather_pair",
        256,
        False,
        (2, 16),
        (2, 8),
    )
    runtime.close()
    assert runtime._closed


def test_first_capture_freezes_the_next_eager_slot_as_graph_base(monkeypatch) -> None:
    runtime = _make_runtime()
    partial_output = torch.zeros(1, 32, 64, dtype=torch.bfloat16)
    partial_lse = torch.zeros(1, 32, dtype=torch.float32)
    capturing = [False]
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._is_current_stream_capturing",
        lambda device: capturing[0],
    )
    monkeypatch.setattr(
        "b12x.comm.pcie._dcp_a2a_cute.is_all_gather_heads_prepared",
        lambda *args: True,
    )

    runtime.lse_reduce_scatter(partial_output, partial_lse)
    assert runtime._next_slot == 1
    capturing[0] = True
    runtime.all_gather_heads(partial_output[:, :16].contiguous())
    assert runtime._device_slot_selection
    assert runtime._graph_base_slot == 1
    assert runtime._next_slot == 1
    assert runtime.run_calls[-1][0] == 1

    capturing[0] = False
    runtime.lse_reduce_scatter(partial_output, partial_lse)
    assert runtime._graph_base_slot == 1
    assert runtime._next_slot == 1
    assert runtime.run_calls[-1][0] == 1


def test_runtime_accepts_head_major_input_and_output():
    runtime = _make_runtime()
    input_storage = torch.arange(
        32 * 4 * 64, dtype=torch.bfloat16
    ).reshape(32, 4, 64)
    partial_output = input_storage.transpose(0, 1)[:2]
    partial_lse = torch.zeros(2, 32, dtype=torch.float32)
    output_storage = torch.empty(16, 2, 64, dtype=torch.bfloat16)
    out = output_storage.transpose(0, 1)

    actual = runtime.lse_reduce_scatter(partial_output, partial_lse, out=out)

    assert actual is out
    assert actual.stride() == (64, 2 * 64, 1)
    torch.testing.assert_close(actual, partial_output[:, :16])


def test_runtime_accepts_token_major_head_tail_capacity():
    runtime = _make_runtime()
    storage = torch.arange(2 * 40 * 64, dtype=torch.bfloat16).reshape(2, 40, 64)
    partial_output = storage[:, :32]
    partial_lse = torch.zeros(2, 32, dtype=torch.float32)
    actual = runtime.lse_reduce_scatter(partial_output, partial_lse)
    torch.testing.assert_close(actual, partial_output[:, :16])


@pytest.mark.parametrize("world_size", (2, 4, 8, 16))
def test_kimi_pair_topk_dispatches_compact_outputs(world_size: int) -> None:
    ext = _FakeExt()
    runtime = _make_kimi_runtime(world_size, ext)
    local_down_width = 3584 // world_size
    local_router_width = 896 // world_size
    local_down = torch.arange(
        local_down_width, dtype=torch.bfloat16
    ).view(1, local_down_width)
    local_router = torch.arange(
        local_router_width, dtype=torch.float32
    ).view(1, local_router_width)
    correction_bias = torch.zeros(896, dtype=torch.float32)

    down, weights, ids = runtime.all_gather_pair_kimi_topk(
        local_down,
        local_router,
        correction_bias,
    )

    assert down.shape == (1, 3584)
    assert weights.shape == (1, 16)
    assert ids.shape == (1, 16)
    torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / 16.0))
    assert torch.equal(ids, torch.arange(16, dtype=torch.int32).view(1, 16))
    runtime.close()


@pytest.mark.parametrize("world_size", (2, 4, 8, 16))
@pytest.mark.parametrize("rows", (1, 8))
def test_kimi_topk16_dispatches_compact_outputs(
    world_size: int, rows: int
) -> None:
    runtime = _make_kimi_runtime(world_size)
    router_logits = torch.arange(
        rows * 896, dtype=torch.float32
    ).view(rows, 896)
    correction_bias = torch.zeros(896, dtype=torch.float32)

    weights, ids = runtime.kimi_topk16(router_logits, correction_bias)

    assert weights.shape == (rows, 16)
    assert ids.shape == (rows, 16)
    torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / 16.0))
    assert torch.equal(
        ids,
        torch.arange(16, dtype=torch.int32).view(1, 16).expand(rows, -1),
    )
    runtime.close()


@pytest.mark.parametrize("rows", (0, 9))
def test_kimi_topk16_rejects_out_of_range_rows(rows: int) -> None:
    runtime = _make_kimi_runtime(2)
    router_logits = torch.zeros((rows, 896), dtype=torch.float32)
    correction_bias = torch.zeros(896, dtype=torch.float32)

    try:
        with pytest.raises(ValueError, match="must be between 1"):
            runtime.kimi_topk16(router_logits, correction_bias)
    finally:
        runtime.close()


def test_kimi_topk16_capture_requires_caller_owned_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_kimi_runtime(2)
    router_logits = torch.zeros((1, 896), dtype=torch.float32)
    correction_bias = torch.zeros(896, dtype=torch.float32)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._is_current_stream_capturing",
        lambda device: True,
    )

    try:
        with pytest.raises(RuntimeError, match="caller-owned output_weights"):
            runtime.kimi_topk16(router_logits, correction_bias)
    finally:
        runtime.close()


def test_stateless_kimi_topk16_rejects_cpu_tensors() -> None:
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        kimi_topk16(
            torch.zeros((1, 896), dtype=torch.float32),
            torch.zeros(896, dtype=torch.float32),
        )


def test_stateless_kimi_topk16_requires_no_communication_state() -> None:
    parameters = inspect.signature(kimi_topk16).parameters
    communication_state = {
        "rank",
        "world_size",
        "process_group",
        "runtime",
        "pool",
        "dcp_pool",
        "channel",
        "channel_id",
    }
    assert communication_state.isdisjoint(parameters)


@pytest.mark.parametrize("world_size", (2, 4, 8, 16))
def test_kimi_pair_topk_graph_prewarm_uses_runtime_world_size(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
) -> None:
    runtime = _make_kimi_runtime(world_size)
    compiled: list[tuple[int, int, int, bool, bool]] = []
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._is_current_stream_capturing",
        lambda device: False,
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setitem(
        sys.modules,
        "b12x.comm.pcie._dcp_a2a_cute",
        SimpleNamespace(
            _get_compiled_all_gather_pair=lambda *args: compiled.append(args)
        ),
    )

    runtime.prepare_graph_all_gather_pair_kimi_topk()

    assert compiled == [(world_size, 0, 512, True, True)]
    runtime.close()


def test_runtime_rejects_shape_dtype_and_capacity_mismatches():
    runtime = _make_runtime()
    good_output = torch.zeros(1, 32, 64, dtype=torch.bfloat16)
    good_lse = torch.zeros(1, 32, dtype=torch.float32)

    with pytest.raises(ValueError, match="float32"):
        runtime.lse_reduce_scatter(good_output, good_lse.bfloat16())
    with pytest.raises(ValueError, match="configured heads/head_dim"):
        runtime.lse_reduce_scatter(good_output[:, :, :32], good_lse)
    with pytest.raises(ValueError, match="exceeds configured capacity"):
        runtime.lse_reduce_scatter(
            torch.zeros(5, 32, 64, dtype=torch.bfloat16),
            torch.zeros(5, 32, dtype=torch.float32),
        )
    unsupported_view = torch.zeros(1, 32, 128, dtype=torch.bfloat16)[:, :, ::2]
    with pytest.raises(ValueError, match="packed token-major or head-major"):
        runtime.lse_reduce_scatter(unsupported_view, good_lse)

    with pytest.raises(ValueError, match="configured local heads/head_dim"):
        runtime.all_gather_heads(good_output[:, :8])
    with pytest.raises(ValueError, match="exceeds configured capacity"):
        runtime.all_gather_heads(torch.zeros(5, 16, 64, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="paired row bytes"):
        runtime.all_gather_pair(
            torch.zeros(1, 8, dtype=torch.bfloat16),
            torch.zeros(1, 8, dtype=torch.float32),
        )


def test_constructor_rejects_invalid_query_and_staging_capacity():
    common = dict(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        signal_ptrs=(100, 200),
        staging0_ptrs=(300, 400),
        staging1_ptrs=(500, 600),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        lse_offset=4 * 32 * 64 * 2,
        lse_capacity=4 * 32,
        ext_module=_FakeExt(),
    )
    with pytest.raises(ValueError, match="query_head_dim"):
        PCIeDCPA2A(
            **common,
            query_head_dim=0,
            output_capacity_elems=4 * 32 * 64,
        )
    with pytest.raises(ValueError, match="output capacity"):
        PCIeDCPA2A(
            **common,
            query_head_dim=32,
            output_capacity_elems=4 * 32 * 32,
        )


def test_cuda_direct_runtime_requires_collective_factory():
    with pytest.raises(ValueError, match="exchange_group is required"):
        PCIeDCPA2A(
            rank=0,
            world_size=2,
            device=torch.device("cuda:0"),
            signal_ptrs=(100, 200),
            staging0_ptrs=(300, 400),
            staging1_ptrs=(500, 600),
            max_batch_size=4,
            total_heads=32,
            head_dim=64,
            output_capacity_elems=4 * 32 * 64,
            lse_offset=4 * 32 * 64 * 2,
            lse_capacity=4 * 32,
            ext_module=_FakeExt(),
        )


def test_pool_uses_distinct_channels_for_target_and_draft_captures(monkeypatch):
    created = []
    current_stream = [7]
    capturing = [False]

    def make_channel(stream_key):
        runtime = _make_runtime()
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=make_channel,
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: (
            current_stream[0] if stream is None else int(stream)
        ),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._is_current_stream_capturing",
        lambda device: capturing[0],
    )

    with pool.capture(7) as target_channel:
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is target_channel
        capturing[0] = False

    with pool.capture(8) as draft_channel:
        capturing[0] = True
        current_stream[0] = 80
        assert pool.for_stream() is draft_channel
        capturing[0] = False

    assert target_channel is not draft_channel
    assert pool._channels == {}
    assert target_channel in pool._all_channels
    assert draft_channel in pool._all_channels
    assert [entry[0] for entry in created] == [7, 8]

    capturing[0] = True
    current_stream[0] = 70
    with pytest.raises(RuntimeError, match="requires an active pool.capture"):
        pool.for_stream()


def test_pool_collectively_prepares_logical_channels_in_canonical_order(
    monkeypatch,
):
    created = []
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: created.append(stream_key) or object(),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._broadcast_gather_object",
        lambda local_state, group: [local_state, local_state],
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_status, group: [local_status, local_status],
    )

    pool.prepare_channels(("target", "draft"))
    pool.prepare_channels(("draft", "target"))

    assert list(pool._logical_channels) == ["draft", "target"]
    assert created == [None, None]


def test_pool_rejects_logical_channel_set_mismatch_before_allocation(monkeypatch):
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )

    def gather(local_state, group):
        if not local_state or isinstance(local_state[0], str):
            return [local_state, ()]
        _requested, existing = local_state
        return [(("other",), existing), local_state]

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._broadcast_gather_object", gather
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", gather
    )

    with pytest.raises(RuntimeError, match="differs across ranks"):
        pool.prepare_channels(("target",))


def test_pool_invalid_logical_id_is_rejected_collectively(monkeypatch):
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_status, group: [local_status, ()],
    )

    with pytest.raises(RuntimeError, match="must not be empty"):
        pool.prepare_channels(("",))


def test_pool_eager_channel_requires_id_and_rejects_duplicate_stream_owner(
    monkeypatch,
):
    eager = _make_runtime()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: eager,
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels["eager:dcp"] = eager
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: int(stream),
    )

    with pytest.raises(RuntimeError, match="explicit semantic channel_id"):
        pool.for_stream(7)
    assert pool.for_stream(7, channel_id="eager:dcp") is eager
    with pytest.raises(RuntimeError, match="stream-affine"):
        pool.for_stream(8, channel_id="eager:dcp")


def test_distributed_single_channel_pool_uses_prepared_default() -> None:
    eager = _make_runtime()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        single_channel=True,
        channel_factory=lambda stream_key: eager,
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels[_SINGLE_CHANNEL_ID] = eager

    assert pool.for_stream() is eager
    assert pool._channels == {0: eager}


def test_pool_capture_requires_stable_semantic_id(monkeypatch):
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()

    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_status, group: [local_status, ()],
    )

    with (
        pytest.raises(RuntimeError, match="stable semantic channel_id"),
        pool.capture(7),
    ):
        pass


def test_pool_capture_allows_opposite_order_from_agreed_catalog(monkeypatch):
    target = _make_runtime()
    draft = _make_runtime()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels.update({"graph:draft": draft, "graph:target": target})
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: int(stream),
    )

    def gather(local_state, group):
        if local_state == ():
            return [(), ()]
        requested, catalog = local_state
        assert requested == "graph:target"
        return [local_state, ("graph:draft", catalog)]

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", gather
    )

    with pool.capture(7, channel_id="graph:target") as channel:
        assert channel is target

    assert pool._captured_channel_ids == {"graph:target"}


def test_pool_capture_rejects_divergent_catalog_before_allocation(monkeypatch):
    target = _make_runtime()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels["graph:target"] = target
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )

    def gather(local_state, group):
        if local_state == ():
            return [(), ()]
        return [local_state, ("graph:target", ())]

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", gather
    )

    with (
        pytest.raises(RuntimeError, match="prepared channel catalog differs"),
        pool.capture(7, channel_id="graph:target"),
    ):
        pass


def test_pool_capture_rejects_differing_unprepared_ids(monkeypatch):
    target = _make_runtime()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels["graph:target"] = target
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )

    def gather(local_state, group):
        if local_state == ():
            return [(), ()]
        _, catalog = local_state
        return [local_state, ("graph:unknown", catalog)]

    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object", gather
    )

    with (
        pytest.raises(RuntimeError, match="unprepared logical channels"),
        pool.capture(7, channel_id="graph:target"),
    ):
        pass


def test_pool_capture_preserves_same_id_convenience_allocation(monkeypatch):
    created = []
    channel = _make_runtime()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: created.append(stream_key) or channel,
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: int(stream),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_state, group: [local_state, local_state],
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._broadcast_gather_object",
        lambda local_state, group: [local_state, local_state],
    )

    with pool.capture(7, channel_id="graph:target") as captured:
        assert captured is channel

    assert created == [None]
    assert pool._logical_channels == {"graph:target": channel}


def test_pool_capture_routes_eager_warmup_to_graph_channel(monkeypatch):
    eager = _make_runtime()
    target = _make_runtime()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels.update({"eager:dcp": eager, "graph:target": target})
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: 7 if stream is None else int(stream),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_state, group: [local_state, local_state],
    )

    with pool.capture(7, channel_id="graph:target") as captured:
        assert captured is target
        assert pool.for_stream(7, channel_id="eager:dcp") is target
        with pytest.raises(RuntimeError, match="stream-affine"):
            pool.for_stream(8, channel_id="eager:dcp")

    assert pool.for_stream(7, channel_id="eager:dcp") is eager


def test_pool_isolates_reused_capture_stream_keys(monkeypatch):
    created = []
    current_stream = [7]
    capturing = [False]

    def make_channel(stream_key):
        runtime = _make_runtime()
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=make_channel,
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: (
            current_stream[0] if stream is None else int(stream)
        ),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._is_current_stream_capturing",
        lambda device: capturing[0],
    )

    with pool.capture(7) as target_channel:
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is target_channel
        capturing[0] = False

    with pool.capture(7) as draft_channel:
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is draft_channel
        capturing[0] = False

    assert target_channel is not draft_channel
    assert pool._channels == {}
    assert target_channel in pool._all_channels
    assert draft_channel in pool._all_channels
    assert [entry[0] for entry in created] == [7, 7]

    pool.close()
    assert target_channel._closed
    assert draft_channel._closed


def test_pool_restores_eager_mapping_after_capture(monkeypatch):
    current_stream = [7]
    capturing = [False]
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: (
            current_stream[0] if stream is None else int(stream)
        ),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._is_current_stream_capturing",
        lambda device: capturing[0],
    )

    eager_channel = pool.for_stream()
    with pool.capture(7) as graph_channel:
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is graph_channel
        capturing[0] = False

    assert graph_channel is not eager_channel
    assert pool._channels == {7: eager_channel}
    assert graph_channel in pool._all_channels


def test_pool_restores_nested_capture_mappings(monkeypatch):
    current_stream = [7]
    capturing = [False]
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: (
            current_stream[0] if stream is None else int(stream)
        ),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._is_current_stream_capturing",
        lambda device: capturing[0],
    )

    eager_channel = pool.for_stream()
    with pool.capture(7) as outer_channel:
        assert pool._channels == {7: outer_channel}

        current_stream[0] = 8
        with pool.capture() as inner_channel:
            capturing[0] = True
            current_stream[0] = 80
            assert pool.for_stream() is inner_channel
            assert pool._channels == {
                7: outer_channel,
                8: inner_channel,
                80: inner_channel,
            }
            capturing[0] = False

        assert pool._channels == {7: outer_channel}
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is outer_channel
        capturing[0] = False

    assert eager_channel is not outer_channel
    assert outer_channel is not inner_channel
    assert pool._channels == {7: eager_channel}
    assert pool._capture_channel_stack == []


def test_pool_rolls_back_throwaway_capture_channels(monkeypatch):
    created = []

    def make_channel(stream_key):
        runtime = _make_runtime()
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=make_channel,
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: 3 if stream is None else int(stream),
    )

    eager_channel = pool.for_stream()
    checkpoint = pool.checkpoint_channels()
    with pool.capture(7) as profile_channel:
        pass

    pool.rollback_channels(checkpoint)

    assert pool._all_channels == [eager_channel]
    assert pool._channels == {3: eager_channel}
    assert profile_channel._closed
    assert not eager_channel._closed


def test_pool_coordinates_ipc_teardown_across_ranks(monkeypatch):
    events = []

    class FakeChannel:
        def _close_ipc_imports(self):
            events.append("close-imports")

        def _free_ipc_exports(self):
            events.append("free-exports")

    group = object()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        exchange_group=group,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    retained = FakeChannel()
    transient = FakeChannel()
    pool._all_channels = [retained]
    pool._channels = {3: retained}
    checkpoint = pool.checkpoint_channels()
    pool._all_channels.append(transient)
    pool._channels[7] = transient
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a.dist.barrier",
        lambda *, group: events.append("barrier"),
    )

    pool.rollback_channels(checkpoint)

    assert events == [
        "barrier",
        "close-imports",
        "barrier",
        "free-exports",
        "barrier",
    ]
    assert pool._all_channels == [retained]
    assert pool._channels == {3: retained}


def test_pool_rejects_channel_rollback_during_capture():
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    checkpoint = pool.checkpoint_channels()

    with pool.capture(7), pytest.raises(RuntimeError, match="during capture"):
        pool.rollback_channels(checkpoint)


def test_a2a_transport_env_selects_push_and_rejects_unknown(monkeypatch) -> None:
    from b12x.comm.pcie import pcie_dcp_a2a as module

    monkeypatch.delenv("B12X_PCIE_DCP_A2A_TRANSPORT", raising=False)
    assert module.a2a_transport() == "pull"
    assert _make_runtime().transport == "pull"
    assert not _make_runtime().push_transport

    monkeypatch.setenv("B12X_PCIE_DCP_A2A_TRANSPORT", " Push ")
    assert module.a2a_transport() == "push"
    runtime = _make_runtime()
    assert runtime.transport == "push"
    assert runtime.push_transport

    monkeypatch.setenv("B12X_PCIE_DCP_A2A_TRANSPORT", "copy_engine")
    with pytest.raises(ValueError, match="B12X_PCIE_DCP_A2A_TRANSPORT"):
        module.a2a_transport()
    with pytest.raises(ValueError, match="B12X_PCIE_DCP_A2A_TRANSPORT"):
        _make_runtime()


def test_push_transport_reaches_graph_prepare_and_capture_checks(monkeypatch) -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    monkeypatch.setenv("B12X_PCIE_DCP_A2A_TRANSPORT", "push")
    runtime = _make_runtime()
    calls = []
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._is_current_stream_capturing",
        lambda device: False,
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        kernels,
        "_get_compiled_lse_reduce_scatter",
        lambda *args: calls.append(args),
    )
    monkeypatch.setattr(
        kernels,
        "_get_compiled_all_gather_heads",
        lambda *args: calls.append(args),
    )
    runtime.prepare_graph_lse_reduce_scatter(dtype=torch.bfloat16, threads=256)
    runtime.prepare_graph_all_gather_heads(threads=256)
    assert calls == [(2, 0, "bf16", 256, True, True), (2, 0, 256, True, True)]

    lookups = []
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_a2a._is_current_stream_capturing",
        lambda device: True,
    )
    monkeypatch.setattr(
        kernels,
        "is_lse_reduce_scatter_prepared",
        lambda *args: lookups.append(args) or False,
    )
    monkeypatch.setattr(
        kernels,
        "is_all_gather_heads_prepared",
        lambda *args: lookups.append(args) or False,
    )
    partial_output = torch.zeros(1, 32, 64, dtype=torch.bfloat16)
    partial_lse = torch.zeros(1, 32, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="cold PCIe DCP LSE"):
        runtime.lse_reduce_scatter(partial_output, partial_lse)
    with pytest.raises(RuntimeError, match="cold PCIe DCP gather"):
        runtime.all_gather_heads(partial_output[:, :16].contiguous())
    assert lookups == [(2, 0, "bf16", 256, True, True), (2, 0, 256, True, True)]


def test_kernel_wrappers_forward_the_push_flag_to_both_launcher_variants(
    monkeypatch,
) -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    compiled = []
    launched = []

    def fake_compile(*args):
        compiled.append(args)
        return lambda *launch_args: launched.append(launch_args)

    monkeypatch.setattr(kernels, "_get_compiled_lse_reduce_scatter", fake_compile)
    monkeypatch.setattr(kernels, "_get_compiled_all_gather_heads", fake_compile)
    kernels.lse_reduce_scatter(
        world_size=2,
        rank=0,
        dtype_name="bf16",
        threads=256,
        local_output_ptr=16,
        local_lse_ptr=16,
        output_ptr=16,
        staging_ptrs=(16, 32),
        signal_ptrs=(16, 32),
        lse_offset=256,
        batch=1,
        total_heads=32,
        head_dim=64,
        input_stride_batch=256,
        input_stride_head=8,
        output_stride_batch=128,
        output_stride_head=8,
        natural_log=False,
        device_slot_selection=False,
        slot_delta_bytes=256,
        blocks=1,
        push=True,
    )
    kernels.all_gather_heads(
        world_size=2,
        rank=0,
        threads=256,
        local_input_ptr=16,
        output_ptr=16,
        staging_ptrs=(16, 32),
        signal_ptrs=(16, 32),
        batch=1,
        local_heads=16,
        head_dim=64,
        element_size=2,
        device_slot_selection=True,
        slot_delta_bytes=256,
        blocks=1,
        push=True,
    )
    # The eager launch also warms the graph (device slot selection) variant
    # of the same transport.
    assert compiled == [
        (2, 0, "bf16", 256, False, True),
        (2, 0, "bf16", 256, True, True),
        (2, 0, 256, True, True),
    ]
    assert len(launched) == 2


def test_launcher_keys_and_compile_specs_carry_the_transport() -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    assert kernels._lse_launcher_key(2, 0, "bf16", 256, True) == (
        2, 0, "bf16", 256, True, False,
    )
    assert kernels._lse_launcher_key(2, 0, "bf16", 256, True, True)[-1] is True
    assert kernels._gather_launcher_key(2, 0, 256, True) == (2, 0, 256, True, False)
    assert kernels._gather_launcher_key(2, 0, 256, True, True)[-1] is True
    for launcher in (
        kernels._get_compiled_lse_reduce_scatter,
        kernels._get_compiled_all_gather_heads,
    ):
        labels = inspect.getsource(launcher).split("labels=(", maxsplit=1)[1]
        assert '"push",' in labels.split(")", maxsplit=1)[0]


def test_push_lse_kernel_writes_peers_before_the_barrier_and_reads_locally() -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    source = inspect.getsource(kernels._LseReduceScatterLaunch.kernel)
    write_phase, read_phase = source.split("block_pair_barrier(", maxsplit=1)
    push_write = write_phase.split("if cutlass.const_expr(self._push):", 1)[1]
    push_write = push_write.split("else:", 1)[0]
    assert "push_row = Int64(self._rank) * Int64(rows) + Int64(row)" in push_write
    assert "for destination_index in cutlass.range_constexpr(" in push_write
    assert "Int64(staging[destination].toint()) + slot_offset" in push_write
    assert "push_row * Int64(packs_per_head) * Int64(16)" in push_write
    assert "_copy_16b_addr(" in push_write
    assert "st_global_f32(" in push_write
    assert "lse_offset + push_row * Int64(4)" in push_write
    assert "local_stage_words" not in push_write
    assert "local_stage_lse" not in push_write

    barrier_args = read_phase.split(")", maxsplit=1)[0]
    assert "acquire=self._push," in barrier_args
    push_reads = read_phase.split("if cutlass.const_expr(self._push):")[1:]
    assert len(push_reads) == 2
    lse_base, payload = push_reads
    lse_base = lse_base.split("if lane == Int32(source_index):", 1)[0]
    assert "Int64(staging[self._rank].toint())" in lse_base
    assert "- source_row" in lse_base
    assert "Int64(source) * Int64(rows)" in lse_base
    assert "staging[source]" not in lse_base
    payload = payload.split("else:", 1)[0]
    assert "staging[self._rank].toint()," in payload
    assert "(Int64(source) * Int64(rows) + Int64(row))" in payload
    assert "* Int64(packs_per_head)" in payload
    assert "staging[source]" not in payload


def test_push_gather_kernel_writes_output_rows_to_peers_and_copies_out_locally() -> None:
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    source = inspect.getsource(kernels._AllGatherHeadsLaunch.kernel)
    write_phase, read_phase = source.split("block_pair_barrier(", maxsplit=1)
    push_write = write_phase.split("if cutlass.const_expr(self._push):", 1)[1]
    push_write = push_write.split("else:", 1)[0]
    assert "push_base = Int64(row) * Int64(packs_per_head)" in push_write
    assert "Int64(staging[destination].toint())" in push_write
    assert "pack_offset = (push_base + Int64(pack)) * Int64(16)" in push_write
    # One local load per pack, one posted store per peer.
    assert push_write.count("ld_global_v4_u32(") == 1
    assert push_write.count("st_global_v4_u32(") == 1
    assert push_write.index("ld_global_v4_u32(") < push_write.index(
        "for destination_index in cutlass.range_constexpr("
    )
    assert "local_stage" not in push_write

    barrier_args = read_phase.split(")", maxsplit=1)[0]
    assert "acquire=self._push," in barrier_args
    push_read = read_phase.split("if cutlass.const_expr(self._push):", 1)[1]
    push_read = push_read.split("else:", 1)[0]
    assert "if source_rank != Int32(self._rank):" in push_read
    assert "(local_stage + output_base * Int64(4)).toint()" in push_read
    assert "staging[source]" not in push_read


def _emulate_push_staging(world_size: int, batch: int, heads_per_rank: int, head_dim: int):
    """Model the push transport's staging addressing on the host.

    Returns, for every rank, the per-source partial tensors and LSE rows that
    its reduce phase reads back from its own staging (or from its local input
    for its own rank) and the gathered query tensor its copy-out phase
    produces, both built with the kernels' index formulas: a writer stores
    row ``row`` of destination ``d`` at compact row ``self * rows + row`` of
    ``d``'s staging (LSE at the same compact row), and a gather writer stores
    output row ``row`` at output position ``row`` of every peer's staging.
    """

    total_heads = world_size * heads_per_rank
    rows = batch * heads_per_rank
    generator = torch.Generator().manual_seed(7)
    partials = [
        torch.randn(batch, total_heads, head_dim, generator=generator).to(torch.bfloat16)
        for _ in range(world_size)
    ]
    lses = [torch.randn(batch, total_heads, generator=generator) for _ in range(world_size)]
    queries = [
        torch.randn(batch, heads_per_rank, head_dim, generator=generator).to(torch.bfloat16)
        for _ in range(world_size)
    ]

    staging = [torch.zeros(world_size * rows, head_dim, dtype=torch.bfloat16) for _ in range(world_size)]
    staging_lse = [torch.full((world_size * rows,), float("nan")) for _ in range(world_size)]
    gather_staging = [torch.zeros(batch * total_heads, head_dim, dtype=torch.bfloat16) for _ in range(world_size)]
    for rank in range(world_size):
        for row in range(rows):
            batch_index, local_head = divmod(row, heads_per_rank)
            push_row = rank * rows + row
            for destination in range(world_size):
                if destination == rank:
                    continue
                source_head = destination * heads_per_rank + local_head
                staging[destination][push_row] = partials[rank][batch_index, source_head]
                staging_lse[destination][push_row] = lses[rank][batch_index, source_head]
        for row in range(batch * total_heads):
            batch_index, global_head = divmod(row, total_heads)
            if global_head // heads_per_rank != rank:
                continue
            local_head = global_head - rank * heads_per_rank
            for destination in range(world_size):
                if destination != rank:
                    gather_staging[destination][row] = queries[rank][batch_index, local_head]

    reads = []
    gathers = []
    for rank in range(world_size):
        per_source = torch.empty(world_size, batch, heads_per_rank, head_dim, dtype=torch.bfloat16)
        per_source_lse = torch.empty(world_size, batch, heads_per_rank)
        for row in range(rows):
            batch_index, local_head = divmod(row, heads_per_rank)
            global_head = rank * heads_per_rank + local_head
            source_row = batch_index * total_heads + global_head
            for source in range(world_size):
                if source == rank:
                    per_source[source, batch_index, local_head] = partials[rank][batch_index, global_head]
                    per_source_lse[source, batch_index, local_head] = lses[rank][batch_index, global_head]
                else:
                    compact = source * rows + row
                    # The kernel keeps one shared ``base + source_row`` load per
                    # lane; the push base pre-subtracts source_row.
                    lse_index = (compact - source_row) + source_row
                    per_source[source, batch_index, local_head] = staging[rank][compact]
                    per_source_lse[source, batch_index, local_head] = staging_lse[rank][lse_index]
        gathered = torch.empty(batch * total_heads, head_dim, dtype=torch.bfloat16)
        for row in range(batch * total_heads):
            batch_index, global_head = divmod(row, total_heads)
            source_rank, local_head = divmod(global_head, heads_per_rank)
            if source_rank == rank:
                gathered[row] = queries[rank][batch_index, local_head]
            else:
                gathered[row] = gather_staging[rank][row]
        reads.append((per_source, per_source_lse))
        gathers.append(gathered.view(batch, total_heads, head_dim))
    return partials, lses, queries, reads, gathers


@pytest.mark.parametrize("world_size,batch", [(2, 1), (4, 3), (9, 4)])
def test_push_staging_layout_reassembles_every_peer_row(world_size: int, batch: int) -> None:
    heads_per_rank = 2
    head_dim = 16
    partials, lses, queries, reads, gathers = _emulate_push_staging(
        world_size, batch, heads_per_rank, head_dim
    )
    for rank in range(world_size):
        per_source, per_source_lse = reads[rank]
        heads = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)
        for source in range(world_size):
            assert torch.equal(per_source[source], partials[source][:, heads])
            assert torch.equal(per_source_lse[source], lses[source][:, heads])
        assert torch.equal(gathers[rank], torch.cat(queries, dim=1))
