# KDA prefill evidence

Measurements that back the `sequence.kda_prefill` performance claims. Each JSON
file records the device, the commit of the checkout that produced it, and raw
samples; the tables below summarize them.

## 20260903-rtx-pro-6000-flashkda-triton-baselines.json

Baseline for the vLLM KDA prefill backends that b12x must beat, measured on an
idle NVIDIA RTX PRO 6000 Blackwell Max-Q (188 SMs, P8 before the run, P1 after)
from the vLLM checkout `/home/luke/projects/vllm-hh-rebase` at commit
`83cb22a0e3` with its venv, `CUDA_VISIBLE_DEVICES=1`, script
`baseline_kda_prefill.py` (the later `benchmarks/benchmark_kda_prefill_vllm_baselines.py`).
Inputs: random bf16 projections, `lower_bound=-5`, fp32 initial states, packed
`cu_seqlens`. Timing: captured CUDA graphs of the kernel launches only, cold
(L2 flushed before every replay) and warm; each sample averages enough
independently bracketed replays to span 128 µs (2.048 µs event quantum).
`gather_scatter_us` is the eager cost of the dense initial-state gather and the
final-state scatter that the vLLM layer runs around FlashKDA. Lower is better.

| case | chain tiles | FlashKDA cold µs | warm µs | µs/tile | gather+scatter µs | Triton cold µs | warm µs |
|---|---|---|---|---|---|---|---|
| h16-t1024-n1 | 64 | 68.0 | 63.0 | 1.063 | 23.0 | 109.2 | 102.0 |
| h16-t4096-n1 | 256 | 219.1 | 214.6 | 0.856 | 23.9 | 392.7 | 396.9 |
| h16-t4096-n4 | 64 | 98.4 | 104.6 | 1.538 | 22.7 | 328.2 | 329.8 |
| h16-t8192-n1 | 512 | 440.3 | 451.2 | 0.860 | 23.4 | 895.0 | 900.2 |
| h16-t512x8 | 32 | 100.3 | 104.0 | 3.135 | 24.8 | 337.4 | 341.6 |
| h16-t32768-n1 | 2048 | 1828.2 | 1841.7 | 0.893 | 23.0 | 3811.9 | 3865.2 |
| h64-t1024-n1 | 64 | 100.7 | 106.0 | 1.573 | 23.1 | 237.6 | 239.2 |
| h64-t4096-n1 | 256 | 469.0 | 487.0 | 1.832 | 23.7 | 1407.5 | 1415.3 |
| h64-t4096-n4 | 64 | 484.8 | 495.2 | 7.575 | 25.9 | 1449.0 | 1453.2 |
| h64-t32768-n1 | 2048 | 3928.1 | 4196.8 | 1.918 | 23.5 | 12574.7 | 12595.8 |

`h16` is the GLM-5.3 Flash TP4 geometry, `h64` is TP1. Both Triton captures
succeeded. FlashKDA's per-tile cost doubles from 16 to 64 heads at the same
sequence, so its prepare kernel and workspace traffic are a substantial share
of the total, not only the sequential recurrence.

## 20260903-rtx-pro-6000-flashkda-triton-baselines-h32.json

The same baseline at 32 heads, the GLM-5.3 Flash TP2 geometry, measured on the
same idle device from the same vLLM checkout at commit `83cb22a0e3` with
`CUDA_VISIBLE_DEVICES=2`. Method, inputs, and columns are those of the file
above.

| case | chain tiles | FlashKDA cold µs | warm µs | µs/tile | gather+scatter µs | Triton cold µs | warm µs |
|---|---|---|---|---|---|---|---|
| h32-t1024-n1 | 64 | 79.1 | 69.2 | 1.237 | 23.2 | 129.4 | 117.4 |
| h32-t2048-n1 | 128 | 139.3 | 136.8 | 1.088 | 22.7 | 241.7 | 243.8 |
| h32-t4096-n1 | 256 | 284.7 | 294.5 | 1.112 | 22.7 | 621.5 | 629.3 |
| h32-t4096-n4 | 64 | 223.2 | 233.1 | 3.488 | 24.2 | 632.8 | 640.6 |
| h32-t512x8 | 32 | 237.6 | 245.3 | 7.424 | 24.3 | 650.7 | 657.5 |
| h32-t8192-n1 | 512 | 594.0 | 604.2 | 1.160 | 23.1 | 1433.4 | 1440.4 |
| h32-t32768-n1 | 2048 | 2414.9 | 2427.5 | 1.179 | 22.5 | 6158.7 | 6218.3 |

FlashKDA's per-tile cost at 32 heads is roughly 1.1 µs for a single prompt but
3.5 µs for four 1024-token requests and 7.4 µs for eight 512-token requests, so
its batched-prefill cost is set by per-sequence work rather than by tile count.
