"""Diagnostic global-timer stamps around unchanged production CUDA graphs.

Timestamp kernels perturb launch scheduling. Their intervals retain graph-node
transition costs and are not interchangeable with CUDA event measurements.
"""

from collections.abc import Mapping
from contextlib import ExitStack

import torch
import triton
import triton.language as tl

from .timing import CapturedGraphRaceTimer


@triton.jit(do_not_specialize=["offset"], do_not_specialize_on_alignment=["offset"])
def _timestamp(values, offset):
    now = tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", constraints="=l", args=[],
        dtype=tl.uint64, is_pure=False, pack=1,
    )
    tl.store(values + offset, now)


class GlobaltimerGraphRaceTimer(CapturedGraphRaceTimer):
    """Bracket each cold graph replay with dependent global-timer kernels."""

    def __init__(self, graphs, *, count, device, flush=None, before_each=None):
        from cuda.bindings import runtime

        if type(count) is not int and not isinstance(count, Mapping):
            raise ValueError("timing requires an integer or per-candidate sample counts")
        counts = {name: count for name in graphs} if type(count) is int else dict(count)
        if not graphs or set(counts) != set(graphs) or any(type(value) is not int or value <= 0 for value in counts.values()):
            raise ValueError("timing requires graphs and positive sample counts")
        self._runtime, self._device, self._production = runtime, device, dict(graphs)
        self._owners = ExitStack()
        self._executable = None
        self._offsets = {name: [] for name in graphs}
        try:
            with torch.cuda.device(device):
                self._timestamps = torch.empty(2 * sum(counts.values()), device=device, dtype=torch.uint64)
                _timestamp[(1,)](self._timestamps, 0, num_warps=1)
                preparations = {}
                names = tuple(graphs)
                for name in names:
                    if before_each is not None or flush is not None:
                        preparation = torch.cuda.CUDAGraph(keep_graph=True)
                        self._owners.callback(preparation.reset)
                        with torch.cuda.graph(preparation):
                            if before_each is not None:
                                before_each(name)
                            if flush is not None:
                                flush()
                        preparations[name] = preparation
                parent = self._checked(runtime.cudaGraphCreate(0))
                self._owners.callback(self._checked_call, runtime.cudaGraphDestroy, parent)
                previous = []
                offset = 0
                for index in range(max(counts.values())):
                    rotation = index % len(names)
                    order = names[rotation:] + names[:rotation]
                    if (index // len(names)) % 2:
                        order = order[::-1]
                    for name in order:
                        if index >= counts[name]:
                            continue
                        if name in preparations:
                            previous = [self._child(parent, previous, preparations[name])]
                        self._offsets[name].append(offset)
                        for production in (graphs[name], None):
                            stamp = torch.cuda.CUDAGraph(keep_graph=True)
                            self._owners.callback(stamp.reset)
                            with torch.cuda.graph(stamp):
                                _timestamp[(1,)](self._timestamps, offset, num_warps=1)
                            previous = [self._child(parent, previous, stamp)]
                            offset += 1
                            if production is not None:
                                previous = [self._child(parent, previous, production)]
                self._executable = self._checked(runtime.cudaGraphInstantiate(parent, 0))
                self._owners.callback(self._checked_call, runtime.cudaGraphExecDestroy, self._executable)
        except BaseException:
            self._owners.close()
            raise

    def _child(self, parent, previous, graph):
        return self._checked(self._runtime.cudaGraphAddChildGraphNode(
            parent, previous, len(previous), self._runtime.cudaGraph_t(graph.raw_cuda_graph()),
        ))

    def samples_us(self):
        if self._executable is None:
            raise RuntimeError("CUDA graph timer is closed")
        torch.cuda.synchronize(self._device)
        stamps = self._timestamps.cpu().tolist()
        return {name: tuple((stamps[index + 1] - stamps[index]) / 1000. for index in offsets)
                for name, offsets in self._offsets.items()}


def warm_globaltimer(device):
    """Resolve the timestamp kernel before freezing a measurement session."""
    with torch.cuda.device(device):
        values = torch.empty(1, dtype=torch.uint64, device=device)
        _timestamp[(1,)](values, 0, num_warps=1)
        torch.cuda.synchronize(device)


def globaltimer_race_samples_us(graphs, *, count, device, flush=None, before_each=None):
    with GlobaltimerGraphRaceTimer(graphs, count=count, device=device, flush=flush, before_each=before_each) as timer:
        timer.replay()
        return timer.samples_us()
