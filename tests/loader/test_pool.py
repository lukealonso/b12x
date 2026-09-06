"""Final shared weights retain normal Tensor ownership and graph behavior."""

import gc
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from b12x.loader._pool import HostWeightWriter, owns_tensor, shared_pool
from b12x.loader import storage_stats


@pytest.mark.parametrize("allocation", ["registered", "pinned", "pinned_wc", "managed"])
def test_checkpoint_copy_survives_file_close_and_graph_replay(tmp_path, allocation):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    path = tmp_path / "weights.safetensors"
    expected = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    save_file({"weight": expected}, path)
    writer = HostWeightWriter()
    with shared_pool(allocation=allocation):
        parameter = torch.nn.Parameter(
            torch.empty((128, 64), device="cuda"), requires_grad=False
        )
        parameter.zero_()
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            assert writer(parameter[64:], checkpoint.get_tensor("weight"))
    path.unlink()
    alias = parameter[64:]
    del parameter
    gc.collect()
    assert owns_tensor(alias)
    assert writer.host_bytes == expected.nbytes
    output = torch.empty_like(alias)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        output.copy_(alias)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output.copy_(alias)
    for _ in range(3):
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_host_writer_declines_dtype_layout_and_shape_changes():
    """Only byte-preserving copies may bypass Torch's conversion semantics."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    writer = HostWeightWriter()
    with shared_pool():
        target = torch.empty((2, 3), device="cuda", dtype=torch.float32)
        for source in (
            torch.ones((2, 3), dtype=torch.float64),
            torch.ones((3, 2)),
            torch.ones((3, 2)).T,
            torch.ones((2, 3))._neg_view(),
        ):
            assert not writer(target, source)
    assert writer.host_copies == 0


def test_host_write_waits_for_current_stream_before_overwriting():
    """A prior asynchronous reader must finish before a CPU weight write."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    writer = HostWeightWriter()
    source = torch.full((4096,), 3.0)
    with shared_pool():
        target = torch.empty_like(source, device="cuda")
        assert writer(target, source)
    result = torch.empty_like(target)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        result.copy_(target)
        assert writer(target, source + 1)
    stream.synchronize()
    torch.testing.assert_close(result.cpu(), source, rtol=0, atol=0)
    torch.testing.assert_close(target.cpu(), source + 1, rtol=0, atol=0)


def test_registered_final_weights_are_locked_until_last_alias_release():
    """CUDA host registration alone does not prevent swap on GB10 ATS."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    def locked_bytes():
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmLck:"):
                return int(line.split()[1]) * 1024
        raise AssertionError("VmLck is unavailable")

    gc.collect()
    torch.cuda.empty_cache()
    before = locked_bytes()
    with shared_pool(allocation="registered"):
        weight = torch.empty(4 << 20, device="cuda", dtype=torch.uint8)
    alias = weight[1024:]
    del weight
    assert locked_bytes() >= before + 4 * 2**20
    del alias
    gc.collect()
    torch.cuda.empty_cache()
    assert locked_bytes() == before


def test_pool_releases_transients_and_restores_allocator_after_exception():
    """Loading must not keep freed transform buffers or change inference allocation."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    previous = torch.cuda.memory._snapshot()["allocator_settings"][
        "expandable_segments"
    ]
    gc.collect()
    torch.cuda.empty_cache()
    before = storage_stats()
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    try:
        with pytest.raises(RuntimeError, match="abort loading"), shared_pool():
            survivor = torch.empty(1024, device="cuda")
            transient = torch.empty(4 << 20, device="cuda")
            assert owns_tensor(transient)
            del transient
            raise RuntimeError("abort loading")
        assert torch.cuda.memory._snapshot()["allocator_settings"][
            "expandable_segments"
        ]
        assert owns_tensor(survivor)
        assert storage_stats()["live_bytes"] - before["live_bytes"] < 4 << 20
        survivor.fill_(7)
        assert survivor.sum().item() == 7 * 1024
        del survivor
        gc.collect()
        torch.cuda.empty_cache()
        assert storage_stats() == before
    finally:
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{previous}")
