# Qwen3.8 BF16 down-projection qualification

This directory is the raw, reproducible qualification receipt for b12x source
commit `f37324070604147dc9499dac956f96d6ea264290` (tree
`13aa1bff1d743945492090b366109673ed0a02de`). The result passed all mandatory
correctness, CUDA graph, allocation, and timing gates.

## Comparison identity

- PR base revision: `9ae32e297c7d8d5258e0953c111a933adcb687f5`
- frozen b12x source revision:
  `f37324070604147dc9499dac956f96d6ea264290`
- clean detached source worktree on the benchmark host:
  `/tmp/b12x-pr243-exact-qualify`
- integration revision: `0f6d68bcb4fc2b0c13fa6f7dd74a4cce617a6eeb`
- integration overlay SHA-256:
  `49a80169aac1ca29a5272fe764253001875da34859bdf91f7d011a43a7aa5c6b`
- exact image:
  `qwen38-27b-exl3:b12x-pr243-reviewfix-exact`
- image ID:
  `sha256:6de72d088cdb1f70f60ff8ce9dc896dc97c020307b4dab548d1fe0348480f28c`
- result SHA-256:
  `2a37e0af9e81850fd6ac391ec3362dd5c40273060f9c4489cdf3f94b248b6201`

The timed comparators are the existing served `exllamav3` route and the generic
b12x dense scheduler from the same exact image. The PR base revision is
recorded for review identity but is not presented as a timed arm: it does not
contain the fused route. Running all three routes in one process keeps the
checkpoint, Torch/CUDA stack, clocks, and source tensors identical.

## Target and hardware

- checkpoint:
  `/home/jon/models/Qwen3.8-27B-EXL3-K5K6-hydrated`
- tensor:
  `model.language_model.layers.0.mlp.down_proj`
- shape: `K=17408`, `N=5120`, BF16 activations, rows `1,4,8,16`
- checkpoint shard SHA-256:
  `792b46b54a79871030a5a029a483b8938ff509912169a1c7627913d614365f0d`
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB
- physical UUID: `GPU-ca537fb1-a522-7429-e8c6-4795af22ab78`
- compute mode: Default
- architecture: `sm_120a`
- measured state: P1, 16,365 MHz memory clock, throttle mask `0x0`
- PyTorch/CUDA: `2.12.0+cu132` / CUDA 13.2
- CUTLASS DSL and all DSL libraries: `4.6.2`

## Invocation

The production service was stopped before this isolated run so the benchmark
owned the physical GPU. The exact invocation was:

```bash
docker run --name b12x-pr243-f373-down-qual \
  --gpus all --ipc=none \
  -e CUTE_DSL_ARCH=sm_120a \
  -e XDG_CACHE_HOME=/tmp/receipt/cache \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/tmp/receipt/cache/b12x/cute \
  -e B12X_COMPILE_CACHE_DIR=/tmp/receipt/cache/b12x/compile \
  -v /home/jon/models/Qwen3.8-27B-EXL3-K5K6-hydrated:/model:ro \
  -v /tmp/b12x-pr243-exact-qualify:/src:ro \
  --entrypoint /opt/venv/bin/python \
  qwen38-27b-exl3:b12x-pr243-reviewfix-exact \
  /src/benchmarks/benchmark_trellis_k6_mcg_checkpoint.py \
  --model-dir /model \
  --tensor-prefix model.language_model.layers.0.mlp.down_proj \
  --rows 1,4,8,16 --params-dtype bf16 \
  --graph-dump-dir /tmp/receipt/graphs \
  --output /tmp/receipt/result.json \
  --verify-shard-sha \
  --compile-warmups 3 --replay-checks 8 --cold-replays 12 \
  --warmups 60 --iterations 240 --topk 8 \
  --source-revision f37324070604147dc9499dac956f96d6ea264290 \
  --integration-tree 49a80169aac1ca29a5272fe764253001875da34859bdf91f7d011a43a7aa5c6b \
  --image qwen38-27b-exl3:b12x-pr243-reviewfix-exact \
  --image-id sha256:6de72d088cdb1f70f60ff8ce9dc896dc97c020307b4dab548d1fe0348480f28c
```

## Correctness and serving gates

All four row counts passed. Across the adversarial scenarios, the fused route
had minimum cosine `0.99999958`, maximum relative L2 `0.00039491`, and exact
top-8 set and order. Outputs were BF16, finite, nonzero, and fully overwritten.
Checkpoint tensors remained immutable.

Each fused CUDA graph contained exactly one cooperative
`K6McgSmallMKernel`, no separate rotation kernel, and the expected launch grid.
Caller-owned output, `rotated_compute`, and FP32 scratch addresses stayed
stable. Eight replays per row produced zero allocator, segment, retry, and OOM
deltas.

## Timings

The benchmark cycles all six route orders equally. Each warm median contains
240 single-replay CUDA-event samples after 60 warmups; each cold median contains
12 samples with no timing warmups. A ratio greater than one means the fused
b12x route has lower latency than the served `exllamav3` comparator.

| Rows | Warm fused ms | Warm served ms | Warm served/fused | Cold fused ms | Cold served ms | Cold served/fused |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.032416 | 0.042624 | 1.314906 | 0.032416 | 0.041360 | 1.275913 |
| 4 | 0.032528 | 0.042560 | 1.308411 | 0.032416 | 0.042288 | 1.304541 |
| 8 | 0.033824 | 0.042656 | 1.261116 | 0.034448 | 0.042336 | 1.228983 |
| 16 | 0.034432 | 0.042656 | 1.238848 | 0.034432 | 0.042656 | 1.238848 |

`result.json` contains every raw cold and warm sample, balanced order and
position, correctness scenario, allocator snapshot, GPU snapshot, package and
toolchain version, declared identity, and ratio direction. `graphs/` contains
the exact CUDA graph DOT artifacts referenced and hashed by the result.
