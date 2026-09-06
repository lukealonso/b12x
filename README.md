# b12x

`b12x` is an SM120/SM121 CuTe DSL and Triton kernel library for local LLM inference.
It specifically targets DGX Spark, RTX Spark and the Blackwell-based RTX
cards (RTX 6000 Pro, RTX 5090).

It is *not* intended to be used in production/datacenter environments, both due to
architecture mismatches and the fast-moving pace of the library. For mission-critical
use cases please use FlashInfer, CUTLASS or TRTLLM.

## Install

```bash
pip install b12x
```

You need Python 3.10+, `torch >= 2.12`, and an SM120/SM121 GPU. The CuTe DSL
compiler and its CUDA 13 libraries come in as wheel dependencies
(`nvidia-cutlass-dsl == 4.6.2`), so there is no build step — kernels are
JIT-compiled on first use and cached.

## What's in here

Every kernel is one op at `b12x.<group>.<op>`; `list_ops()` enumerates the
complete set. The op owns its `plan`/`bind`/`run` facade in `api.py`; the
kernel guts sit in `_impl.py`/`_kernel*.py`; cross-op lowering lives in
`<group>/_shared/` and the universal compile/scratch spine in `b12x/_lib/`.

**`gemm`** — `gemm.blockscaled` is the common dense interface for raw
NVFP4/MXFP4/MXFP8/block-FP8 operands and packed MXFP8/tensor-FP8 weights; it
owns `mm`, `pack_weight`, and serving `prewarm`. The legacy
`gemm.mxfp8_linear` and `gemm.tensor_fp8_linear` imports are compatibility
aliases. `gemm.block_fp8_linear` retains a separate planned interface because
it owns caller-provided scratch and inline requantization. The fused MLA query
projection (`gemm.mla_query_projection`) and grouped WO projection
(`gemm.wo_projection`) are used around MLA attention.

**`attention`** — `attention.paged` (paged-KV decode/extend, FP8 KV, MSA block
sparse, CUDA-graph-replayable), `attention.sparse_mla` and
`attention.compressed_sparse_mla` (top-k / compressed-page MLA — distinct
contracts, kept separate on purpose), `attention.dsa_indexer` (the DSA/MSA quantize →
score → select pipeline), `attention.qsa` (group-selected exact sparse GQA
decode over caller-populated, read-only main BF16 K/V), and `attention.varlen`
(contiguous batched/varlen).

**`moe`** — `moe.fused_moe`, fused FP4 TP MoE across a micro-kernel decode
path, a unified dynamic path (persistent grid, `nvfp4`/`w4a8_mx`/`w4a8_nvfp4`),
and W4A16 (BF16 activations, inline FP4 weight dequant — no activation-scale
math), with SiLU/ReLU2/SwiGLU-OAI activations; plus `moe.ep_moe` (expert
parallel).

**the rest** — `norm.mhc` (fused RMSNorm + hyper-connection residual),
`norm.hyperconnection` (learned multi-stream residual primitives),
`sequence.{ple_hash,ple_embedding,ple}` (prime-hashed embedding IDs, fused
quantized lookup, and short-convolution state), `sequence.gdn_decode` (packed
recurrent decode),
`sequence.mtp_feedback` (MTP token/multi-stream feedback fusion),
`quantization.{nvfp4,mxfp8}` (row quantizers), `comm.roce` (RoCEnante: one-shot
RDMA all-reduce/all-gather for multi-node DGX Spark TP, see `docs/rocenante.md`),
and `comm.pcie` (IPC-backed PCIe
collectives). The Qwen3.8-Flash-Next QSA, HyperConnection, PLE, GDN decode,
and MTP feedback Triton implementations are correctness references and are
not throughput-qualified production kernels.
`b12x` owns planning, scratch layout, and policy, so serving stacks only supply
metadata and capacity limits.

## Using it

Every stateful kernel lives at `b12x.<group>.<op>` and shares the **same
shape** — `plan` the work, size scratch from the plan, `bind` your tensors as
views, `run`. The module path carries the context, so the verbs and role
classes (`Caps`/`Plan`/`Binding`) are uniform across families:

```python
# norm — fused RMSNorm + hyper-connection residual mixing
from b12x.norm import mhc

plan    = mhc.plan(mhc.Caps(...))
spec    = plan.scratch_specs()[0]
scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
binding = mhc.bind(plan, scratch=scratch, ...)
residual, post, comb, y = mhc.run_post_pre(..., binding=binding)
```

```python
# moe — fused tensor-parallel routed-expert FFN (weights prepped once per model)
from b12x.moe import fused_moe

wplan   = fused_moe.plan_weights(quant_modes="nvfp4",
                                 source_format="modelopt_nvfp4", ...)
experts = fused_moe.prepare_weights(plan=wplan, ...)
plan    = fused_moe.plan(fused_moe.Caps(...))
spec    = plan.scratch_specs()[0]
scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
binding = fused_moe.bind(plan, scratch=scratch, a=x, experts=experts,
                         topk_weights=tw, topk_ids=ti)
out     = fused_moe.run(binding=binding)
```

```python
# attention — sparse MLA from compressed KV pages (DeepSeek V4)
from b12x.attention import compressed_sparse_mla

plan    = compressed_sparse_mla.plan(compressed_sparse_mla.Caps(...))
spec    = plan.scratch_specs()[0]
scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
binding = compressed_sparse_mla.bind(plan, scratch=scratch, q=q,
                              swa_indices=idx, swa_lengths=lens, ...)
out = compressed_sparse_mla.run(swa_k_cache=swa, binding=binding, sm_scale=scale, ...)
```

`plan` is host-side and may allocate; `bind` only narrows/views (never
allocates), which is what makes captured graphs safe; `run*` executes and is
CUDA-graph-capture safe. One-shot ops (`gemm.blockscaled.mm`,
`quantization.mxfp8.quantize_rows`) are plain functions; `comm.pcie`
collectives are stateful classes. `b12x.list_ops()` enumerates the full
set; every op exports `is_supported()`. Underneath, kernels register as torch
custom ops in the private `b12x::` namespace (torch.compile / CUDA-graph
integration) — prefer the Python API.

## PCIe DMA wire modes

`PCIeDmaAllReduce` can compress eligible BF16 all-reduces. Configure it with
`B12X_PCIE_DMA_FP8`, or pass the same value as the `fp8=` constructor
argument. Integrations such as vLLM can forward their own launch setting to
that constructor.

| Mode | Reduce-scatter | All-gather | When to use it |
|---|---|---|---|
| `0` | BF16 ring | BF16 ring | Unquantized baseline |
| `ag` | BF16 ring | block E4M3 ring | Limit E4M3 quantization to the final broadcast |
| `ring` | block E4M3 ring, requantized per hop | block E4M3 ring | Compress both phases with the neighbor ring |
| `a2a` | block E4M3 scatter with FP32 accumulation | block E4M3 broadcast | Quantize each input once and overlap direct peer transfers |
| `i8` | BF16 ring | block INT8 ring | Limit INT8 quantization to the final broadcast |
| `i8_ring` | block INT8 ring, requantized per hop | block INT8 ring | Compress both phases with the INT8 codec |
| `i8_a2a` | block INT8 scatter with FP32 accumulation | block INT8 broadcast | Use the quantize-once all-to-all topology with INT8 |
| `mx` | BF16 ring | MXFP8 ring | Limit MXFP8 quantization to the final broadcast |
| `mx_ring` | MXFP8 ring, requantized per hop | MXFP8 ring | Compress both phases with standard E4M3/E8M0 MXFP8 |
| `mx_a2a` | MXFP8 scatter with FP32 accumulation | MXFP8 broadcast | Use the quantize-once all-to-all topology with MXFP8 |

Every compressed mode uses 132 bytes per 128 values instead of 256 bytes for
BF16, a 48.4% wire-byte reduction. E4M3 and INT8 store one FP32 scale per 128
values; MXFP8 stores four E8M0 scales, one per 32 values. These modes are most
useful for large prefill collectives on PCIe-only multi-GPU systems where peer
transport is the bottleneck; they do not change the KV-cache format and usually
do not affect small decode collectives. Choose a codec by model quality gates,
then benchmark the ring and all-to-all variants on the target PCIe topology.

Compressed transport requires BF16 input and a per-rank shard divisible by
128 elements; other shapes use the BF16 path:

```bash
B12X_PCIE_DMA_FP8=i8_ring python -m your_server
```

### Graph replay of eager ring all-reduces

An eager `PCIeDmaAllReduce.all_reduce` call issues roughly 250 CUDA API calls
from Python (per-piece copies, flag kernels, adds, cross-stream events), so a
Python-driven serving loop that reduces prefill-size tensors is bound by that
issue time rather than by the wire. `B12X_PCIE_DMA_GRAPH_REPLAY=1` captures
each eligible shape once into a CUDA graph over static input/output buffers
(on the shape's second call, so one-off sizes stay eager) and replays it
afterward with a copy in and a copy out. The graph records the eager kernels
in their eager order over the same flag counters, so a replay computes the
same bits as the eager call (`tests/comm/test_pcie_dma_gpu.py` asserts
`torch.equal` against the eager result across replays, shape reuse and an
enclosing capture); calls made while an enclosing CUDA graph is being
captured keep recording the eager sequence into that graph. A caller-supplied
`out` must match the input shape and dtype, live on the ring's device and be
contiguous on both paths.

| Variable | Default | Meaning |
|---|---|---|
| `B12X_PCIE_DMA_GRAPH_REPLAY` | `0` | `1` enables replay for eager calls |
| `B12X_PCIE_DMA_GRAPH_REPLAY_MIN_BYTES` | `8388608` | Smallest tensor size that is captured |
| `B12X_PCIE_DMA_GRAPH_REPLAY_MAX_ENTRIES` | `4` | Cached shapes; least recently used is dropped first. The static buffers for all entries (`2 x max_bytes` each) are reserved when the ring is constructed, so serving never allocates for a capture |

Compilation happens lazily per shape/config and is cached. For serving, warm
up the shapes you need, then freeze:

```python
import b12x

# ... run warmup traffic covering every shape you will serve ...
b12x.freeze_kernel_resolution("serving")
```

After the freeze, any request that would trigger a new kernel compile raises
`KernelResolutionFrozenError` instead of stalling a live request (or worse,
compiling inside CUDA graph capture).

Set `B12X_PRINT_COMPILE_PROGRESS=1` to log each compiler invocation with its
cache-key parameters and duration — useful for figuring out what warmup
actually covered. `B12X_TIMING=1` enables per-kernel timing logs.

## Where to look next

- `tests/` is the executable spec — per-group API and numerical-reference
  tests showing exact tensor layouts and `plan`/`bind`/`run` call sequences.
  (`tests/_legacy/` holds the pre-namespace flat-API suite, being migrated.)
- `benchmarks/` has tuned invocations per kernel family (and `probe_*` scripts
  from tile-sweep experiments).
- `docs/` has design notes: the MoE execution model, the eager-plan-bind
  architecture, and an SM120 MLA postmortem.

Failing that, ask your friendly neighborhood AI agent — it does fine here.
