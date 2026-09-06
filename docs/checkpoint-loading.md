# Checkpoint loader

Status: C99 O_DIRECT transport and initial vLLM integration, September 6, 2026.
The `--load-format b12x` adapter allocates checkpoint destination weights through
a CPU-addressable CUDA pool. It parses safetensors headers, routes metadata-only
tensor views through model sharding, and reads selected file ranges into owned
destinations. Payload reads require `O_DIRECT`; there is no buffered retry or
file-mapping input path. The native helper uses CPython/DLPack and PyTorch's
pluggable allocator C interface without the PyTorch C++ ABI.

The loader owns allocation policy. vLLM's `allocate_weights(factory, ...)` marks
weight creation; the active loader supplies its allocator through `weight_transfer`.
The default allocator remains CUDA during construction, loading and preparation.
Runtime outputs, workspaces and mutable state never enter the shared weight pool.
Preparation may reuse weight storage in place, but shared weights are read-only
during inference. New preparation outputs use ordinary CUDA storage. A final
allocation audit rejects shared non-persistent buffers before serving starts.

Initial serving checks cover Qwen 3.8 Flash Next with TP=1 and GLM 5.3 Flash
with TP=2 and MTP off. Install the matching vLLM weight-transfer hooks:

```sh
VLLM_PLUGINS=b12x_loader vllm serve MODEL --load-format b12x
```

The adapter uses vLLM's standard checkpoint-shard progress format and honors
`use_tqdm_on_load` and rank-zero output. The loader always uses write-combined
pinned storage for weights; no `allocation` option is needed or accepted.
Write combining changes the CPU cache policy, not the ownership or direct-I/O
contract.

The adapter records destination views and submits packed native descriptors
`(fd, offset, row_bytes, destination, operation, rows, source_stride, destination_stride)`
in batches. Strides are measured in bytes; contiguous ranges have one row. Operations
cover direct file reads, in-place BF16 expansion, and copies of owned CPU control
metadata. For the latter, the offset field contains the source address. C validates
ownership and non-overlap, orders jobs by file offset, splits large ranges, and
distributes reads across a persistent pthread pool. `io_threads` defaults to 8
and accepts 1–16. Each worker owns 8 MiB of locked alignment scratch; header and
metadata reads use a separate 8 MiB reader. No Python callback runs per read.
The same descriptor contract can support another native I/O backend.

Python retains tensor owners and file descriptors until the batch completes.
Explicit fences precede online quantization, composed weight transforms, PLE
scale validation, and final weight preparation. A failed batch drains all workers
before reporting failure. Arbitrary consumers of queued parameter values require
an explicit completion fence; this is an initial-load integration contract.
Numerical loading callbacks use `materialize_weight` to read owned inputs before
operating on them, including GLM's paired selector weights and scales.

Weight destinations use `cudaHostAllocMapped | cudaHostAllocWriteCombined` and
are explicitly `mlock`ed. Failure to lock final storage fails the allocation.
Alternative mappings remain available to the allocation-qualification tools,
outside the serving configuration. The initial adapter
requires GPU host page tables and PyTorch's native CUDA allocator, and does
not support vLLM sleep mode. It preserves index/prefix filtering, including
MTP. Byte-preserving contiguous routes read into the CPU alias after synchronizing
the loading stream. BF16-to-FP32 reads occupy the first half of the final FP32
allocation, then C99 expands backwards in place, preserving all BF16 bits.
Other contiguous casts use one reusable 8 MiB input allocation. Arbitrary
arithmetic on source descriptors and unsupported layouts fail explicitly.

Aligned file/address ranges read straight into the destination. Large misaligned
ranges also read into the destination, then realign in place with `memmove`.
Each aligned read window fits within the remaining destination bytes. Only
small edges need a worker's fixed, locked 8 MiB alignment buffer and a CPU copy.
For TP slices with rows smaller than 4 KiB, C reads adjacent rows together into
that same fixed scratch buffer and scatters selected bytes into the destination.
Larger rows use the direct range path. In-place moves, edge copies and strided
copies are counted separately. This path bypasses the
page cache but does not promise zero copying for every safetensors layout.
Scalar and explicitly declared control metadata are coalesced into owned CPU
spans, with a 64 MiB aggregate span limit per session, including intervening bytes.

Logs distinguish physical reads, bytes read into destinations, in-place alignment,
edge and strided copies, in-place conversions, other casts, and final parameter ownership. Existing b12x
weight-preparation policies retain or reuse source storage; the loader does not
make a second packed-weight copy. Existing quantization callbacks can still
allocate full tensors, so these counters do not establish an aggregate transform
memory bound. A shared target/draft transform budget remains design work.
Final allocations use ordinary Torch tensors and its stream bookkeeping;
there is no Tensor subclass or global Torch monkeypatch.

`b12x.loader.read_tensor` is the earlier allocation-qualification primitive.
Its buffered raw-file reader is separate from the adapter's O_DIRECT transport.

Add bounded checkpoint loading to b12x. The component owns checkpoint manifests,
I/O scheduling, buffer capacity, CUDA completion, and generic weight transforms.
Serving integrations supply model mappings, source/destination slices,
dependencies, required numerical semantics, and capacity limits. This follows
the existing rule that b12x owns planning and policy.

## GB10 destination storage

Choose the allocation type before choosing the transport. The first candidate
for unchanged checkpoint bytes is final, GPU-mapped host storage. Ordinary
PyTorch CUDA allocations are not interchangeable with CPU-addressable buffers:
[NVIDIA's GB10 porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)
states that CPU and I/O devices cannot coherently access `cudaMalloc` memory.
This restricts that allocation type, not every possible final destination.

The prototype supports anonymous system mappings, `cudaHostAlloc` (cached and
write-combined), anonymous mappings registered with `cudaHostRegister`, managed
memory, and private file mappings. System/file mappings require pageable-memory
access through GPU host page tables. The GB10 reports these capabilities and
support for registration and concurrent managed access.

DLPack wraps the GPU address without copying. Its storage deleter retains the
allocation through tensor aliases and synchronizes the consuming device before
final release. Keep model tensors alive while replaying graphs referencing
them; graph capture does not acquire an ownership reference to external inputs.
Reads complete before publishing a tensor. File mappings are lazy and require
an immutable backing file for the entire tensor lifetime.

| Weight operation | Candidate GB10 path |
| --- | --- |
| Checkpoint bytes already match the runtime layout | Read selected ranges into final mapped storage; inference kernels read that storage. |
| Expand BF16 to FP32 | Read into the final allocation and expand backwards in place. |
| Repack | Reuse source storage through b12x preparation when the layout permits it; bound any workspace. |
| Quantize or other cast | Use the destination allocation when safely possible, otherwise bounded mapped input tiles. |
| A consumer cannot use mapped storage efficiently or correctly | Select ordinary CUDA destinations and bounded host-to-device transfers during planning. |

Aligned direct reads into mapped final storage need no loader payload ring or
separate host-to-device copy. Misaligned ranges use in-place realignment or the
bounded edge buffer described above. Header and payload reads both require
O_DIRECT.

Mapped final weights are persistent model storage, counted once in the total
capacity check. Mapped conversion input is temporary storage and counts against
the aggregate staging cap until its last GPU reader completes. CPU writers and
GPU readers never overlap on the same range. Use supported CUDA ordering and
completion mechanisms to publish reads and retire input tiles.

First qualify allocation and execution with exact byte comparisons, the actual
b12x TMA and ordinary-load kernels, quantized MoE/dense layouts, scale tensors,
embeddings, and graph capture/replay. Measure repeated decode reads against
ordinary CUDA allocations; a faster load cannot justify slower inference.
Also qualify large allocations, pin/registration failures, alignment, and
pointer arithmetic beyond 4 GiB. General mapped-memory support alone does not
prove that every current b12x kernel supports these buffers.

This qualification precedes implementing a large I/O scheduler. The initial
library still needs a bounded copy route, but mapped final storage and direct
GPU consumption of mapped conversion input are primary GB10 candidates.
Allocation selection is a plan decision, never an oversized-tensor fallback.
CUDA documents the general integrated-GPU mapped-memory approach in its
[programming guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#data-transfer-between-host-and-device).

## Package boundary

Proposed layout:

```text
b12x/loader/
    __init__.py         lazy public exports
    api.py              manifest, plan, and session interface
    _contract.py        references, ranges, destinations, capabilities
    _manifest.py        indexed safetensors discovery and validation
    _planner.py         dependency scheduling, coalescing, budget admission
    _session.py         ownership, submission, retirement, error handling
    _native.py          build/cache and C ABI bindings
    _storage.c         implemented C99 allocation, positional reads, DLPack ownership
    _transport.c       planned batched positional I/O and CUDA transfers
    _transforms.py      bounded quantization/packing using existing b12x ops
b12x/integration/vllm/
    loader.py           public vLLM loader registration and adapters
```

The loader imports without vLLM. Importing `b12x` or `b12x.loader` must not build
native code, initialize CUDA, or import the kernel compiler. An explicit
session/preparation operation loads required dependencies before serving.

This is host-side startup infrastructure, with a session lifecycle rather than
an inference kernel's graph-replay lifecycle. Do not put disk I/O or blocking
waits in CUDA graph capture. GPU transforms use existing b12x component plans
and policy; any new planned GPU op must satisfy catalog/profile registration
and frozen-resolution requirements. Keep normal namespace/registry imports
lightweight and extend their existing tests.

## Ownership and memory

`WeightRef` keeps immutable source metadata, not a transient tensor. A planned
operation names its source ranges and final owned destinations. Its temporary
storage is reserved before scheduling. Arbitrary model callbacks never receive
a reusable staging view.

Host slots remain live through reads and transfers; GPU workspace remains live
through its last transform consumer. Requests hold strong references to storage
owners and file handles until completion. Retirement uses explicit CUDA events,
including all participating streams. Error/cancellation handling drains users
before unregistering or freeing memory and never publishes partial weights.

One session cap includes pinned host buffers, GPU staging, retained source
materializations, transform workspace, and alignment overhead. Target and draft
share that cap. On GB10, charge host and CUDA staging together because they
consume the same physical memory pool. Final model storage, metadata, CUDA
overhead, and external memory use are separately included in capacity checks.
Do not describe the staging cap as a bound on total system memory or page cache.

Start qualification with 256 MiB aggregate staging and 8 MiB read chunks; tune
only from measured loading results. Large tensors stream into final storage or
through bounded transform tiles. Tensor size must not trigger a full CPU tensor
fallback or silently enlarge the cap. Reserve progress workspace before
prefetching to prevent capacity deadlock.

## Native helper and integration

Use packaged C99 source and a cached host C-compiler build, following the native
helper approach in `comm.roce`. The allocation helper uses the CPython C API to
create owned DLPack capsules directly, without ctypes callbacks, pybind11, or
the PyTorch C++ ABI. Keep future batches inside native code to avoid Python
calls per small tensor. Use positional I/O and CUDA completion APIs, adding
liburing only if measurements justify it. Future asynchronous requests must
retain storage owners until native completion.

Include source, ABI version, CPU architecture, compiler identity, flags, and
relevant dependency identity in the build-cache key. Build atomically and
validate the ABI on load. Build/probe failures must be clear, and preparation
must finish before inference warmup/frozen kernel resolution. Validate both
aarch64 and x86_64 builds and include native source in package data.

The first transport supports local reads into declared CPU-addressable ranges,
including mapped final storage and bounded mapped conversion input. Start
qualification with batched positional reads; select a small native reader pool
or io_uring from measurements. Keep the C ABI independent of that choice.
CUDA copies and bounded transforms serve consumers that require them. Direct
I/O and alternate transports preserve the same ownership/capacity contract.

Register `--load-format b12x` through vLLM's public `register_model_loader`
interface using a dedicated general-plugin registration function. Keep plugin
registration cheap and idempotent in spawned workers; initialize transport only
on an actual load. The existing FP6 plugin remains independently selectable.
Model-specific Qwen/GLM descriptions may require vLLM changes; the plugin does
not eliminate that integration work. Integrations describe numerical recipes
and slices, while b12x selects chunking, batching, workspace, and execution order.

Reuse existing MXFP8/NVFP4 quantizers and packing implementations where their
output, rounding, scale domain, and layout match. Add bounded output interfaces
where necessary rather than duplicating quantization math in the integration.
The NVFP4 head needs a first pass over its existing global-scale domain before
chunk quantization. Shared target/draft source reads must preserve their
different final precision requirements.

## Qualification

Add behavior tests under `tests/loader/` and startup/transport benchmarks under
`benchmarks/loader/`. The existing `benchmarks/checkpoint_loader.py` is a small
indexed safetensors helper, not the streaming implementation.

Test ownership under repeated slot reuse, delayed and multi-stream consumers,
partial reads, cancellation, allocation failures, aliases, and paired scales
in either order. Include tensors larger than the cap, 64-bit offsets above
4 GiB, packed dtypes, TP slices, and checkpoint index overlays. Check actual
host/device high-water marks and memory release at session close.

Compare loaded bytes and transformed weights with trusted safetensors and the
existing quantizers before measuring speed. Initially qualify Qwen TP1 on GB10
with MTP off/on and cold/warm cache. Extend to GLM TP2 and RTX after the GB10
allocation and loading contracts pass. Record command, revision, physical GPU, memory
budget, correctness state, physical/selected bytes, raw timings, and comparison
direction. Do not switch the serving default until correctness, bounded memory,
and startup performance pass. The full model integration and acceptance plan
is in the companion vLLM checkout's `docs/design/streaming_weight_loading.md`.
