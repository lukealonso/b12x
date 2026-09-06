# RoCE one-shot all-reduce A/B, 2026-09-02 (stinger -> maxwell:8000, llm-decode-bench 0.4.32)

Only difference between arms: `docker-compose-roce-allreduce.override.yml` (overlay image roce2,
VLLM_ENABLE_ROCE_ALLREDUCE=1). Same model, profile, dual-HCA NCCL config and 8192 budget.
A = RoCE one-shot for all-reduces <= 1 MB (decode), NCCL above. B = NCCL only.
Decode cells: 30 s sustained, max_tokens 512, exact token targeting. Coding peak: 5 sequential cc1
runs of the built-in coding prompt, 2000 max tokens.

cell            A-roce tok/s    B-nccl tok/s    delta  A-roce itl  B-nccl itl
c1@0k                   37.9            42.9   -11.5%        59ms        64ms
c1@16k                  40.8            38.6     5.6%        60ms        65ms
c1@32k                  41.8            40.3     3.7%        59ms        66ms
c2@0k                   62.9            60.0     4.8%        82ms        88ms
c2@16k                  62.5            58.0     7.8%        80ms        88ms
c2@32k                  60.0            57.8     3.8%        83ms        89ms
c4@0k                   98.2            99.8    -1.6%        99ms       105ms
c4@16k                 103.2            86.3    19.7%       100ms       112ms
c4@32k                 102.7            89.3    15.0%        98ms       113ms
c8@0k                  146.1           128.6    13.6%       137ms       157ms
c8@16k                 130.3           118.7     9.7%       151ms       163ms
c8@32k                 116.7           118.0    -1.1%       156ms       165ms
c16@0k                 213.9           174.4    22.6%       195ms       242ms
c16@16k                192.0           166.7    15.2%       210ms       251ms
c16@32k                190.4           159.6    19.3%       208ms       244ms
coding_peak A-roce: runs_requested=5, runs_ok=5, max_tokens=2000
coding_peak B-nccl: runs_requested=5, runs_ok=5, max_tokens=2000
prefill         A-roce tok/s    B-nccl tok/s    delta
8192                    2613            2612     0.0%
32768                   2781            2761     0.7%
131072                  2717            2699     0.7%

coding peak A-roce: {"mean_generation_tok_s": 77.42278010746709, "median_generation_tok_s": 78.733491124409, "max_generation_tok_s": 80.22239605570647, "min_generation_tok_s": 73.23098680600266, "cjk_runs": 0}
coding peak B-nccl: {"mean_generation_tok_s": 68.98821569989114, "median_generation_tok_s": 69.81319326384002, "max_generation_tok_s": 72.18023155423319, "min_generation_tok_s": 62.86256438913216, "cjk_runs": 0}

Reading: per-step latency (ITL p50) is lower with RoCE in every cell (5-19%); throughput gains grow
with concurrency (c16: +15-23%). c=1 padding-text throughput is dominated by speculative-acceptance
noise (+-10% run to run); the coding-peak run, a real c=1 workload, is +12.7% median.
The cluster was left on the RoCE stack afterwards.

## Arm C: RoCE + --async-scheduling (test launcher copy serve-async.sh)

No consistent gain: ITL unchanged in every cell; throughput deltas vs A scattered (c1@0k +26% on the
noisiest cell, c2@16k/32k -9/-12%, c8/c16 within +-5%), coding peak 75.9 vs 78.7 tok/s. Not adopted.
Comparison in C-vs-A.txt.

## c=1 decode profile on the RoCE stack (profiles/roce-c1-20260902 on maxwell, 8 iterations, CUDA graphs on)

GPU busy 97% of wall time (no CPU stall left); step ~52 ms. Per step: MoE expert kernel ~31 ms (52%),
b12x dense GEMMs ~14 ms (24%, ~700 launches at ~20 us: latency-bound GEMVs), cutlass sm80 WMMA GEMMs
(torch fallback) ~5 ms (9%), RoCE all-reduce ~3.2 ms (5%, 104/step at 31 us incl. wait), NCCL all-gather
~0.5 ms (3/step), rest ~5 ms. Geometry: 288 routed experts, top-8, moe_intermediate 2048, hidden 4096,
42 sparse layers; a 6-token verify batch touches ~44 experts/layer = ~150 MB/rank/layer (NVFP4, TP4),
so 745 us/layer = ~200 GB/s achieved vs ~230 GB/s practical: the MoE kernel is near the memory roofline.
Communication is done as a lever (fused RMSNorm <= ~1.5 ms). Next non-roofline targets: dense GEMV launch
latency (fusion across projections), the sm80 WMMA fallbacks, then acceptance length (bytes per accepted token).

## GEMM fallback attribution (capture trace + code)

- DFlash speculator `propose` (vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:346) runs a bf16 cuBLAS GEMM
  of ~1.4 ms with grid (8, 303, 1) (vocab-shard sized), ~2 per step: ~5% of the c=1 step. Candidate for a b12x
  vocab-projection kernel.
- MoE router gate: 42 launches/step of a cuBLAS bf16 sm80 WMMA kernel (grid (8,3,8) = 288 experts, split-K 8),
  ~65 us each = ~2.7 ms/step. Cause: GateLinear enables its cuteDSL ll_bf16 path only for exact capability
  (12, 0); GB10 is (12, 1). Fix: vllm-router-sm121.patch (overlay image roce3). The warmup module already runs
  the ll_bf16 kernel on this GPU.
- Remaining ~3 bf16 GEMMs per full-attention layer (33/step, ~65 us) sit in the sparse-attention indexer path
  (weights_proj fp32 F.linear, index_kpool_compress_gate F.linear) and are unattributed by trace; ~2 ms/step.

## Arm D: roce3 = roce2 + router SM121 fast path

No measurable change (c=1 ITL 60-61 ms vs 59-60; coding peak 74.9 vs A 78.7 / C 75.9; other cells within
noise; sanity output at temperature 0 correct). Comparison in D-vs-A.txt. Investigating with a direct layer
probe (router_probe.py) with the cluster stopped.
Probe result (router_probe.py, GB10, cluster stopped): with the SM121 patch the ll_bf16 tier engages at M<=16 and
is numerically correct (max abs err 1.4e-6), but takes 15.3 us vs 9.2 us for plain cuBLAS F.linear at M=6
(K=4096, N=288). The patch is dropped; roce2 stays deployed. cuBLAS at 9 us also means the 65 us in-graph
sm80 kernel is not the router's intrinsic cost.

## Corrected c=1 step accounting (replay trace, 52 ms/step)

- MoE expert kernel: ~31 ms (near memory roofline: ~6.1 GB/rank/step of NVFP4 experts at ~200 GB/s).
- b12x dense GEMVs: ~14 ms; the KDA/MLA/dense-MLP/shared-expert projections are ~2.3 GB/rank/step of MXFP8,
  i.e. mostly bandwidth-bound too (~11 ms at 200 GB/s), not launch-bound as first assumed.
- Torch/cuBLAS GEMMs (sm80 WMMA): ~2-3 ms scattered: router 42 x 14 us (0.6), DFlash vocab 3 x 186 us (0.6),
  MLA absorption + indexer in 11 full-attention layers ~60 us/layer (0.7), remainder small.
- RoCE all-reduce 3.2 ms, NCCL all-gather 0.5 ms, other ~2 ms.
Weight bytes per step (~8.4 GB/rank) put the step at ~80% of the memory roofline. Levers left: bytes per step
(dense projections MXFP8 -> NVFP4 ~ -1.1 GB, ~10%), acceptance length (bytes per accepted token), comm
(RoCE all-gather + fused RMSNorm, ~1.5-2 ms), and ~1-2 ms of torch GEMM fallbacks.
The router SM121 patch (roce3) is dropped: the cuteDSL router kernel is slower than cuBLAS on GB10.

## NCCL fully out of decode (in progress)

Added `all_gather` to b12x.comm.roce (same transport, strided-copy kernel writing the concatenated layout),
all-gather shard cap 16 MB and all-reduce cap raised to 2 MB (covers the 192-token max capture batch).
Shim: communicator `all_gather` routes eligible dim-0/last-dim gathers to it. Overlay image roce4.

## Arm E: roce4 (RoCE all-reduce cap 2 MB + RoCE all-gather) vs A (roce2)

Standalone 4-node: all-gather [6,38720] 96 us (graph) vs 331 us NCCL+copy; all-reduce 1.5 MB 277 vs 877 us.
Serving: within noise of A (ITL 58-60 vs 59-60 ms at c=1, coding peak 76.3 vs 78.7/75.9/74.9 across A/C/D,
other cells +-8%): the expected ~0.3 ms/step is below the benchmark's resolution. No regressions.
Comparison in E-vs-A.txt. Acceptance criterion is the roce4 profile: NCCL kernels per decode step must be zero.
roce4 profile: 1 NCCL kernel/step remained (top-k ids all-gather, int64 [batch, k]) because the gather
rejected integer dtypes. Fixed in b12x (any dtype; unaligned shapes via a padded contiguous gather +
torch reshape); overlay roce5. Acceptance profile repeated on roce5.

## Result: NCCL fully out of decode (roce5, deployed)

roce5 profile (8 iterations, c=1, CUDA graphs): NCCL kernels 0/step; RoCE all-reduce 102/step; RoCE
all-gather 3/step (logits gather, top-k values, top-k ids). NCCL remains only for prefill all-reduces
above 2 MB. b12x branch roce-oneshot-allreduce @ 86bd0e9; overlay build context
~/spark_vllm/build-contexts/roce-oneshot-20260902 (vllm-roce-allreduce.patch: 181 lines).
Cluster left running on roce5 via docker-compose-roce-allreduce.override.yml.

roce6 (deployed): backend renamed RoCEnante (label B12X_ROCENANTE, runtime algorithm 'rocenante'); first-use log lines on rank 0 only.
Enable with VLLM_ENABLE_ROCE_ALLREDUCE=1; VLLM_ROCE_ALLREDUCE_MAX_SIZE (2MB) and VLLM_ROCE_ALLGATHER_MAX_SIZE (16MB) bound routing.
