# Qwen3.8 BF16 down-projection qualification

This directory is the raw, reproducible qualification receipt for b12x source
commit <code>81595330adba568f6361b396088f91e18b8116f0</code> (tree
<code>4e238da8d46f5091430741053a76b57881efb4a6</code>). The result passed all
mandatory correctness, CUDA graph, allocation, and timing gates.

## Comparison identity

- PR base revision:
  <code>9ae32e297c7d8d5258e0953c111a933adcb687f5</code>
- frozen b12x source revision:
  <code>81595330adba568f6361b396088f91e18b8116f0</code>
- clean detached source worktree on the benchmark host:
  <code>/tmp/b12x-pr243-exact-qualify</code>
- Qwen build-context revision:
  <code>67c338876acc904418d60c6b469d9a62ce62e225</code>
- Qwen executable integration revision:
  <code>0f6d68bcb4fc2b0c13fa6f7dd74a4cce617a6eeb</code>
- integration overlay SHA-256:
  <code>49a80169aac1ca29a5272fe764253001875da34859bdf91f7d011a43a7aa5c6b</code>
- exact image:
  <code>qwen38-27b-exl3:b12x-pr243-hotpath-exact</code>
- image ID:
  <code>sha256:cb7666cb44c214a7ffa3fcbbaf53ce8179017dd18e6b749f9bac2c0c633f5ef3</code>
- result SHA-256:
  <code>74372b9860ebf1854e17130aa923037063c45e3386ed38158288f90f78b4f1ff</code>

The timed comparators are the existing served ExLlamaV3 route and the generic
b12x dense scheduler from the same exact image. The PR base revision is
recorded for review identity but is not presented as a timed arm because it
does not contain the fused route. Running all three routes in one process keeps
the checkpoint, Torch/CUDA stack, clocks, and source tensors identical.

## Target and hardware

- checkpoint:
  <code>/home/jon/models/Qwen3.8-27B-EXL3-K5K6-hydrated</code>
- tensor:
  <code>model.language_model.layers.0.mlp.down_proj</code>
- shape: <code>K=17408</code>, <code>N=5120</code>, BF16 activations, rows
  <code>1,4,8,16</code>
- checkpoint shard SHA-256:
  <code>792b46b54a79871030a5a029a483b8938ff509912169a1c7627913d614365f0d</code>
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB
- physical UUID:
  <code>GPU-ca537fb1-a522-7429-e8c6-4795af22ab78</code>
- compute mode: Default
- architecture: <code>sm_120a</code>
- measured state: P1, 16,365 MHz memory clock, throttle mask <code>0x0</code>
- PyTorch/CUDA: <code>2.12.0+cu132</code> / CUDA 13.2
- CUTLASS DSL and all DSL libraries: <code>4.6.2</code>

## Invocation

The production service was stopped before this isolated run so the benchmark
owned the physical GPU. The exact invocation was:

~~~bash
docker run --name b12x-pr243-815-down-qual \
  --gpus all --ipc=none \
  -e CUTE_DSL_ARCH=sm_120a \
  -e XDG_CACHE_HOME=/tmp/receipt/cache \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/tmp/receipt/cache/b12x/cute \
  -e B12X_COMPILE_CACHE_DIR=/tmp/receipt/cache/b12x/compile \
  -v /home/jon/models/Qwen3.8-27B-EXL3-K5K6-hydrated:/model:ro \
  -v /tmp/b12x-pr243-exact-qualify:/src:ro \
  --entrypoint /opt/venv/bin/python \
  qwen38-27b-exl3:b12x-pr243-hotpath-exact \
  /src/benchmarks/benchmark_trellis_k6_mcg_checkpoint.py \
  --model-dir /model \
  --tensor-prefix model.language_model.layers.0.mlp.down_proj \
  --rows 1,4,8,16 --params-dtype bf16 \
  --graph-dump-dir /tmp/receipt/graphs \
  --output /tmp/receipt/result.json \
  --verify-shard-sha \
  --compile-warmups 3 --replay-checks 8 --cold-replays 12 \
  --warmups 60 --iterations 240 --topk 8 \
  --source-revision 81595330adba568f6361b396088f91e18b8116f0 \
  --integration-tree 49a80169aac1ca29a5272fe764253001875da34859bdf91f7d011a43a7aa5c6b \
  --image qwen38-27b-exl3:b12x-pr243-hotpath-exact \
  --image-id sha256:cb7666cb44c214a7ffa3fcbbaf53ce8179017dd18e6b749f9bac2c0c633f5ef3
~~~

## Correctness and serving gates

All four row counts passed. Across the adversarial scenarios, the fused route
had minimum cosine <code>0.99999958</code>, maximum relative L2
<code>0.00039491</code>, and exact top-8 set and order. Outputs were BF16,
finite, nonzero, and fully overwritten. Checkpoint tensors remained immutable.

Each fused CUDA graph contained exactly one cooperative
<code>K6McgSmallMKernel</code>, no separate rotation kernel, and the expected
launch grid. Caller-owned output, <code>rotated_compute</code>, and FP32 scratch
addresses stayed stable. Eight replays per row produced zero allocator,
segment, retry, and OOM deltas.

## Timings

The benchmark cycles all six route orders equally. Each warm median contains
240 single-replay CUDA-event samples after 60 warmups; each cold median contains
12 samples. A ratio greater than one means the fused b12x route has lower
latency than the served ExLlamaV3 comparator.

| Rows | Warm fused ms | Warm served ms | Warm served/fused | Cold fused ms | Cold served ms | Cold served/fused |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.032416 | 0.042592 | 1.313919 | 0.032416 | 0.041360 | 1.275913 |
| 4 | 0.032608 | 0.042560 | 1.305201 | 0.032416 | 0.041216 | 1.271471 |
| 8 | 0.033440 | 0.042656 | 1.275598 | 0.034432 | 0.042336 | 1.229554 |
| 16 | 0.034432 | 0.042656 | 1.238848 | 0.034432 | 0.042656 | 1.238848 |

<code>result.json</code> contains every raw cold and warm sample, balanced
order and position, correctness scenario, allocator snapshot, GPU snapshot,
package and toolchain version, declared identity, and ratio direction.
<code>graphs/</code> contains the exact CUDA graph DOT artifacts referenced and
hashed by the result.
