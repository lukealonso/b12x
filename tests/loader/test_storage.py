"""Direct-read bytes and ownership, including ranges beyond 32-bit offsets."""

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from b12x.loader import capabilities, read_tensor, storage_stats


@pytest.fixture(
    params=["system", "pinned", "pinned_wc", "registered", "managed", "file"]
)
def allocation(request):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    caps = capabilities()
    kind = request.param
    if kind in ("system", "file") and not (
        caps["pageable_memory_access"] and caps["host_page_tables"]
    ):
        pytest.skip("requires host page tables")
    if kind == "managed" and not caps["concurrent_managed_access"]:
        pytest.skip("requires concurrent managed access")
    if kind == "registered" and not caps["host_register_supported"]:
        pytest.skip("requires host registration")
    return kind


@pytest.mark.parametrize("offset", [4098, 2**32 + 4098])
def test_reads_exact_range_and_keeps_storage_through_aliases(
    tmp_path, allocation, offset
):
    """A closed reader and deleted parent must not invalidate retained views."""
    path = tmp_path / "weights"
    expected = torch.arange(2048, dtype=torch.int16).reshape(32, 64)
    with path.open("wb") as stream:
        stream.seek(offset)
        stream.write(memoryview(expected.numpy()))
    before = storage_stats()
    tensor = read_tensor(
        path,
        offset=offset,
        shape=expected.shape,
        dtype=expected.dtype,
        allocation=allocation,
    )
    assert tensor.is_cuda
    torch.testing.assert_close(tensor.cpu(), expected, rtol=0, atol=0)
    alias = tensor[3:7]
    pointer = tensor.data_ptr()
    del tensor
    gc.collect()
    assert alias.data_ptr() == pointer + 3 * 64 * 2
    assert storage_stats()["live_bytes"] == before["live_bytes"] + expected.nbytes
    torch.testing.assert_close(alias.cpu(), expected[3:7], rtol=0, atol=0)
    del alias
    gc.collect()
    assert storage_stats() == before


def test_release_waits_for_other_stream_readers(tmp_path, allocation):
    path = tmp_path / "weights"
    path.write_bytes(bytes(range(256)) * 256)
    before = storage_stats()
    tensor = read_tensor(path, shape=(65536,), dtype=torch.uint8, allocation=allocation)
    result = torch.empty_like(tensor)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        result.copy_(tensor)
    del tensor
    gc.collect()
    assert storage_stats() == before
    stream.synchronize()
    torch.testing.assert_close(result.cpu(), torch.arange(65536).remainder(256).byte())


def test_rejects_truncated_input_without_publishing_storage(tmp_path, allocation):
    path = tmp_path / "short"
    path.write_bytes(b"abcd")
    before = storage_stats()
    with pytest.raises(RuntimeError, match="file's size"):
        read_tensor(
            path, offset=2, shape=(4,), dtype=torch.uint8, allocation=allocation
        )
    assert storage_stats() == before


def test_concurrent_reads_keep_independent_owned_storage(tmp_path, allocation):
    path = tmp_path / "weights"
    path.write_bytes(bytes(range(256)) * 16)
    before = storage_stats()

    def read(index):
        return read_tensor(
            path,
            offset=index * 256,
            shape=(256,),
            dtype=torch.uint8,
            allocation=allocation,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        tensors = list(pool.map(read, range(16)))
    assert storage_stats()["live_allocations"] == before["live_allocations"] + 16
    for tensor in tensors:
        torch.testing.assert_close(tensor.cpu(), torch.arange(256).byte())
    del tensor, tensors
    gc.collect()
    assert storage_stats() == before


def test_failed_tensor_import_releases_unconsumed_capsule(
    tmp_path, allocation, monkeypatch
):
    path = tmp_path / "weights"
    path.write_bytes(b"abcd")
    before = storage_stats()

    def reject(capsule):
        raise RuntimeError("consumer rejected tensor")

    monkeypatch.setattr(torch.utils.dlpack, "from_dlpack", reject)
    with pytest.raises(RuntimeError, match="consumer rejected tensor"):
        read_tensor(path, shape=(4,), dtype=torch.uint8, allocation=allocation)
    gc.collect()
    assert storage_stats() == before


def test_empty_and_scalar_tensors(tmp_path, allocation):
    path = tmp_path / "scalar"
    path.write_bytes(b"\x7b\x00\x00\x00")
    scalar = read_tensor(path, shape=(), dtype=torch.int32, allocation=allocation)
    assert scalar.item() == 123
    empty = read_tensor(
        path, offset=4, shape=(0, 128), dtype=torch.bfloat16, allocation=allocation
    )
    assert empty.shape == (0, 128)
    assert empty.nbytes == 0


@pytest.mark.parametrize(
    "shape,offset,error",
    [
        ((-1,), 0, ValueError),
        ((1,), -1, ValueError),
        ((2**63,), 0, OverflowError),
        ((2,), 2**63 - 1, OverflowError),
    ],
)
def test_invalid_ranges_fail_before_opening_file(shape, offset, error):
    with pytest.raises(error):
        read_tensor(
            "does-not-exist",
            shape=shape,
            dtype=torch.uint8,
            offset=offset,
            allocation="system",
        )
