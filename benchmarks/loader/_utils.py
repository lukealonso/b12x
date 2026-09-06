"""Storage adapters used only by the allocation qualification benchmarks."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
import statistics

import torch

from b12x.loader import read_tensor


def source_identity():
    from b12x.loader._native import load

    repo = Path(__file__).resolve().parents[2]
    paths = [
        path
        for directory in (repo / "b12x/loader", repo / "benchmarks/loader")
        for path in directory.iterdir()
        if path.suffix in (".py", ".c")
    ]
    native = Path(load().__file__)
    return {
        "sources_sha256": {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)
        },
        "native_helper": str(native),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
    }


class WeightFiles:
    def __init__(self, directory, allocation):
        self.directory = Path(directory)
        self.allocation = allocation
        self.count = 0
        self.storage_bytes = 0

    def load(self, value):
        if isinstance(value, torch.Tensor):
            self.count += 1
            extent = 1 + sum(
                (d - 1) * s for d, s in zip(value.shape, value.stride(), strict=True)
            )
            physical = value.as_strided((extent,), (1,))
            if self.allocation == "cuda":
                result = physical.clone()
            else:
                path = self.directory / str(self.count)
                with path.open("wb") as stream:
                    stream.write(memoryview(physical.cpu().view(torch.uint8).numpy()))
                result = read_tensor(
                    path, shape=(extent,), dtype=value.dtype, allocation=self.allocation
                )
            assert result.data_ptr() % 16 == 0
            result = result.as_strided(value.shape, value.stride())
            torch.testing.assert_close(
                result.view(torch.uint8), value.view(torch.uint8), rtol=0, atol=0
            )
            self.storage_bytes += extent * value.element_size()
            return result
        if dataclasses.is_dataclass(value):
            return dataclasses.replace(
                value,
                **{
                    field.name: self.load(getattr(value, field.name))
                    for field in dataclasses.fields(value)
                },
            )
        return value


def paired_times(run, *, cold_l2_cache=True):
    from flashinfer.testing import bench_gpu_time_with_cupti

    samples = [[], []]
    for order in ((0, 1), (1, 0), (0, 1), (1, 0)):
        for index in order:
            samples[index].append(
                list(
                    bench_gpu_time_with_cupti(
                        lambda: run(index),
                        use_cuda_graph=True,
                        cold_l2_cache=cold_l2_cache,
                        dry_run_time_ms=100,
                        repeat_time_ms=150,
                    )
                )
            )
    medians = [
        statistics.median(v for sample in arm for v in sample) for arm in samples
    ]
    return {
        "cuda_ms": medians[0],
        "shared_ms": medians[1],
        "shared_over_cuda_latency": medians[1] / medians[0],
        "raw_ms": samples,
    }
