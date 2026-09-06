# Dense GEMM activation precision

Status: implemented on SM120/SM121. `gemm.blockscaled` accepts BF16 activations with
NVFP4 or MXFP8 weights and selects an activation precision for dense linear
projections. MoE routing and expert GEMMs have separate implementations.

`mode="a16"` uses the BF16 warp-MMA specialization of
`b12x/_lib/dense_gemm.py::DenseGemmKernel`. The specialization retains the dense
engine's TMA producer, shared-memory pipeline, tile scheduler, accumulator
epilogue, compiler, and launch wrapper. It loads compressed weights and their
scales into shared memory, converts weight pairs directly into BF16 MMA
registers, and accumulates in FP32. Split-K writes FP32 partials and reduces
them into BF16 output. Activations remain BF16 throughout this route.

The implementation follows the inline weight conversion and narrow-M warp-MMA
patterns in `b12x/moe/_shared/kernels/w4a16/kernel.py` and the native FP8
conversion helpers in `b12x/_lib/intrinsics.py`. The MoE engine and its prepared
scale layout are references, not dependencies of the dense launch path. Triton
is used only for supporting activation quantization and packing in
`b12x/gemm/blockscaled/_quantize.py`.

## Shared weight contract

| Recipe | Stored values | Block scales | Reconstructed weight |
| --- | --- | --- | --- |
| NVFP4 | `uint8[N,K/2]`, low nibble first | E4M3, one per 16 K values | `E2M1 * block_scale * global_scale` |
| MXFP8 | `float8_e4m3fn[N,K]` | UE8M0, one per 32 K values | `E4M3 * block_scale` |

Both activation precision routes accept the same F8_128x4-swizzled weight
scale storage. `w4a16`/`w8a16` accept its flat physical storage or native six
dimensional MMA view. `pack_weight` also supports the established compact
MXFP8 scale input, which it swizzles during weight preparation.

NVFP4 `pack_weight` borrows the packed values and scale tensors without
rewriting them. `global_scale_kind="reciprocal"` interprets the supplied weight
global scale as a quantizer multiplier and divides by it in the epilogue.
Neither mode creates a second weight-scale tensor. Global scales must be
finite and positive; reconstructed weights must fit BF16.

A16 requires contiguous, 16-byte-aligned CUDA tensors, N divisible by 8,
stored K divisible by 32, and input K divisible by 8. Its native packed BF16
conversions require PTX 9.2 (CUDA 13.3). Quantized activation execution requires
stored K divisible by 128. MXFP8's established functional path remains available
for its other supported layouts and devices.

## One-shot calls and graph capture

```python
import torch
from b12x.gemm import blockscaled

# All tensors are already on the same SM120/SM121 device.
weight = blockscaled.pack_weight(
    packed_w, swizzled_weight_scales, recipe="nvfp4",
    global_scale=weight_global_scale, global_scale_kind="multiplier",
)
scratch = torch.empty(
    blockscaled.workspace_size(weight, max_tokens),
    dtype=torch.uint8, device=packed_w.device,
)
blockscaled.prewarm(
    weight, token_counts, workspace=scratch,
    activation_global_scale=activation_quantizer_multiplier,
)
blockscaled.mm(
    x, weight, out=y, workspace=scratch, mode="auto",
    activation_global_scale=activation_quantizer_multiplier,
)

# Standalone W4A16 consumes only weight scales.
blockscaled.w4a16(
    x, packed_w, swizzled_weight_scales, weight_global_scale, out=y,
)
```

Prewarm every precision/configuration route needed by the captured token
counts. The workspace is caller-owned and reusable across sequential calls;
concurrent calls require disjoint output and workspace buffers. No public
plan/bind/run API is introduced. A CUDA graph retains its captured precision
route. Kernel compilation and policy resolution use static geometry and device
identity; live M drives launch grids and masks.

## Measured dispatch

The registered offline component is `gemm.blockscaled_precision`:

```bash
.venv/bin/python scripts/generate_gpu_profile.py \
  --components gemm.blockscaled_precision --warmup 3 \
  --work-dir /tmp/blockscaled-profile-work \
  --output /tmp/blockscaled-profile.json
```

The corpus covers `(N,K)` equal to `(4096,5376)`, `(16384,1024)`,
`(17408,5120)`, `(5120,17408)`, and `(248320,2560)`, for both recipes at M=1 through 16 and
24, 32, 64, 128, 256, 512, 1024, and 2048. Each case races the quantized path
against sixteen A16 configurations. Small-M MXFP8 also includes the existing
fused activation-quantization GEMM as a baseline.

Timings include per-block activation scale computation, quantization, packing,
and GEMM. NVFP4's activation global scale is supplied as an input; computing
that scalar is outside the timed graph. Weight preparation is outside all
timings. Separate GEMM-only benchmark records are diagnostic and do not decide
precision promotion.

Candidates must pass independent numerical oracles and poisoned-buffer graph
replay before timing. A16 must have median latency no greater than every
quantized baseline in both an initial race and a separate confirmation pass.
The promotion threshold is 0%; equal latency prefers A16 because it preserves
BF16 activations. Bootstrapped 95% ratio intervals can be reconstructed from
the retained samples; they do not impose a statistically significant speedup
requirement. Paired replay order is balanced, L2 is flushed by default, and
raw samples, clock snapshots,
allocation checks, and source/toolchain identity are retained in checkpoints.
The Max-Q diagnostic clock contract allows only throttle masks 0x0/0x4, P1,
stable memory clocks, and a maximum 30 MHz SM-clock difference.

Runtime queries contain only recipe, input features, and output features.
An autotuned config stores exact measured M routes; it does not interpolate
an M threshold. M values omitted from that config retain quantized activation
execution. Measurements establish a projection's end-to-end crossover; they
do not isolate tensor-core instruction latency as its cause.

For an uncovered device or model geometry on SM120/SM121, the registered
component heuristic promotes both NVFP4 and MXFP8 to A16 at M=1 through 8.
It requires K divisible by 32 and N divisible by 8, and chooses a 128-column,
K=64 tile with four K slices. Larger M retains quantized activations. This is
a heuristic prediction, not a measured guarantee for an uncovered geometry
or device.

Autotuned entries take precedence over the heuristic, including an entry that
selects quantized activation execution for every M. `B12X_POLICY_MODE` supports
the existing `heuristic-only` and `preplanned-only` qualification modes. Explicit
`mode="a16"` or `mode="quantized"` fixes activation precision. Forced A16 uses
the profile's tile configuration when one exists, otherwise the default A16
tile. An explicit `_config` overrides that tile selection. Already-quantized
activation inputs retain their supplied precision. A16 promotion requires an
eligible BF16 input layout; MXFP8's functional API retains its established
handling of noncontiguous input.

## Qualified Max-Q coverage

Status: qualified. The embedded profile
`nvidia.rtx.pro.6000.blackwell.max-q` matches the NVIDIA RTX PRO 6000 Blackwell
Max-Q Workstation Edition with 188 SMs. Its precision table uses the 0%
promotion threshold. All 192 GPU cases passed timing qualification and all
3,264 case/candidate combinations passed correctness. The table contains 61
NVFP4 and 44 MXFP8 promotion routes:

| Weight recipe | N | K | M values selecting A16 |
| --- | ---: | ---: | --- |
| NVFP4 | 4096 | 5376 | 1–16 |
| NVFP4 | 16384 | 1024 | 1–16 |
| NVFP4 | 17408 | 5120 | 1, 3–14 |
| NVFP4 | 5120 | 17408 | 1–16 |
| MXFP8 | 4096 | 5376 | 2–16 |
| MXFP8 | 16384 | 1024 | 8–14 |
| MXFP8 | 17408 | 5120 | 1–5, 9–16 |
| MXFP8 | 5120 | 17408 | 5, 9–16 |

All measured M values from 24 through 2048 retain quantized activations.
Representative independent-confirmation medians are:

| Recipe | N | K | M | A16, µs | Quantized, µs | A16 / quantized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NVFP4 | 4096 | 5376 | 1 | 16.288 | 19.136 | 0.8512 |
| NVFP4 | 4096 | 5376 | 8 | 14.336 | 18.336 | 0.7818 |
| NVFP4 | 5120 | 17408 | 8 | 35.520 | 40.864 | 0.8692 |
| MXFP8 | 5120 | 17408 | 16 | 55.296 | 60.096 | 0.9201 |

The ratio is A16 latency divided by quantized latency; lower is faster. These
are cold-L2 graph timings that include activation quantization, with supplied
NVFP4 activation global scales and a 325 W power limit. The toolchain was
PyTorch 2.12.0+cu132, CUTLASS DSL 4.6.0, Triton 3.7.0, and PTXAS 13.3.73.
The physical GPU UUID was `GPU-f0121aa7-a898-82be-f537-a099d50ef7d8`.
The source was held fixed at revision
`214b37b36dfbe0af6c47bccc8860be8719d8506e` in
`/home/luke/projects/b12x-research/rs-2` throughout qualification.

```bash
CUDA_VISIBLE_DEVICES=GPU-f0121aa7-a898-82be-f537-a099d50ef7d8 \
.venv/bin/python scripts/generate_gpu_profile.py \
  --components gemm.blockscaled_precision --warmup 25 \
  --work-dir /tmp/b12x-maxq-gpu8-parity-work \
  --output /tmp/b12x-maxq-gpu8-parity.json
```

Local evidence is `/tmp/b12x-maxq-gpu8-parity.json`, its source manifest
`/tmp/b12x-maxq-gpu8-parity-source.json`, the selected-route audit
`/tmp/b12x-maxq-gpu8-parity-audit.json`, and raw checkpoints in
`/tmp/b12x-maxq-gpu8-parity-work/`. The artifact SHA256 is
`16fab276312457dad4ca50677d9e8616a5e5d25215895773ba824d977c14a79b`;
the source-manifest SHA256 is
`51121ed2e9eee453574134ed66b310af2f35664d294bd0098930d42a56a5643b`.

Eleven checkpoint resumes were required for SM-clock drift above 30 MHz.
Rejected samples are preserved in `/tmp/b12x-maxq-gpu8-parity-rejected/`.
One NVFP4 M=1 checkpoint required requeueing because its initial race passed
but its confirmation clock gate failed; the accepted replacement passed both
races. Every selected A16 route has valid clock and allocation checks in both
passes and median latency no greater than every quantized baseline. The audit
binds all 192 decisions to their raw samples, selected configurations, physical
GPU identity, and source hashes.

With the embedded profile loaded, `tests/gemm/test_blockscaled_a16.py`,
`tests/policy/test_blockscaled_precision.py`, and
`tests/policy/test_discrete_sweep_generator.py` passed 97 checks; four GB10-only
checks were skipped. The vLLM integration at revision `1310d69c3d` passed all
six `test_b12x_dense_precision_gpu_graph_replay` cases on the same physical GPU,
using `/home/luke/projects/vllm-hh-rebase/.venv/bin/python` and
`PYTHONPATH=/home/luke/projects/b12x-research/rs-2`. This checks `auto`, `a16`,
and `quantized` for both recipes, shared weight storage, numerical output,
frozen kernel resolution, stable output addresses, and allocation-free replay.
That integration check used PyTorch 2.13.0 and CUTLASS DSL 4.6.2; the performance
measurements use the separate toolchain recorded above.

Qualification covers the dense one-shot API. These measurements do not qualify
other RTX PRO 6000 product variants or SM121; those devices retain their own
profile coverage and heuristic behavior. The precision generator supports
SM120 and SM121.

## SM121 qualification

Status: correctness qualified on NVIDIA GB10, 48 SMs,
`GPU-87533355-db2d-9b70-eeab-5a9159ee4bc1`. The BF16 specialization selects its
architecture identity from the device and uses the same inline packed
conversions as SM120. With `CUTE_DSL_ARCH=sm_121a`, PyTorch 2.12.0+cu130,
CUTLASS DSL 4.6.2, and PTXAS 13.3.73, 58 targeted checks passed: exhaustive
FP4/FP8 value and scale conversion, both weight formats, split-K variants,
scale-tile tails, and graph replay under frozen kernel resolution.

Status: performance qualified for the four weight geometries and 24 row counts
listed above, for both recipes. All 3,264 candidate checks passed correctness;
all 192 cases passed timing qualification. The embedded profile contains 32
NVFP4 and 62 MXFP8 A16 promotion routes under the 0% threshold:

| Weight recipe | N | K | M values selecting A16 |
| --- | ---: | ---: | --- |
| NVFP4 | 4096 | 5376 | 1–12, 14–16 |
| NVFP4 | 16384 | 1024 | 1–14, 16 |
| NVFP4 | 17408 | 5120 | 1, 2 |
| NVFP4 | 5120 | 17408 | None |
| MXFP8 | 4096 | 5376 | 1–16 |
| MXFP8 | 16384 | 1024 | 1–14, 16, 24, 32 |
| MXFP8 | 17408 | 5120 | 1–3, 7–16 |
| MXFP8 | 5120 | 17408 | 1–16 |

Every promoted route passed the latency comparison in both independent timing
passes. All measured M values from 64 through 2048 retain quantized activations.
The uncovered-geometry heuristic promotes to A16 at M=1 through 8 on GB10.
Explicit `mode="a16"` is supported.

For N=4096, K=5376, representative independent-confirmation cold-L2 graph
medians in microseconds are:

| Recipe | M | Selected A16 | Fastest quantized baseline | A16 / quantized |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 | 1 | 59.456 | 61.408 | 0.9682 |
| NVFP4 | 8 | 59.424 | 61.440 | 0.9672 |
| MXFP8 | 1 | 104.448 | 106.496 | 0.9808 |
| MXFP8 | 8 | 107.808 | 108.544 | 0.9932 |

The ratio is A16 latency divided by quantized latency; lower is faster. Timings
include activation quantization under the API contract described above.
The device-specific timing gate requires P0, zero throttle mask, and at most
30 MHz SM-clock change between snapshots. NVML does not report the GB10 memory
clock; evidence records that limitation. SM120 Max-Q measurements retain their
separate timing gate.

The 25-warmup, 25-trial run completed without checkpoint retries in 11m06s.
The command was:

```bash
CUTE_DSL_ARCH=sm_121a \
CUDA_VISIBLE_DEVICES=GPU-87533355-db2d-9b70-eeab-5a9159ee4bc1 \
/home/luke/projects/vllm/.venv/bin/python scripts/generate_gpu_profile.py \
  --components gemm.blockscaled_precision --warmup 25 \
  --work-dir /tmp/b12x-sm121-parity-work \
  --output /tmp/b12x-sm121-parity.json
```

Evidence on `chroniton.local` is `/tmp/b12x-sm121-parity.json`, the source
manifest `/tmp/b12x-sm121-parity-source.json`, the selected-route audit
`/tmp/b12x-sm121-parity-audit.json`, and raw checkpoints under
`/tmp/b12x-sm121-parity-work/`. The artifact SHA256 is
`87f8ead04a89c27ee12993c7f80ca51462ea1fef073600c0b23ec936b1f3af49`;
the source-manifest SHA256 is
`ae7f817a3b0ba2d5b72c505b82daeba676eada8d6536ae0aebf0322af45ff3cc`.
The isolated source directory is `/home/luke/projects/b12x-precision-parity`,
based on revision `6698bee5f4793ac0139884439e1b4a0c621a39ba` with the
manifest-bound parity selection changes. The manifest records the source and
toolchain independently of installed package-version metadata.

With the embedded profile loaded, the A16 and precision-policy suites passed
96 checks; three SM120-specific checks were skipped. This includes BF16-reference
output on promoted decode routes and exact quantized output on retained
M=2048 routes under PREPLANNED_ONLY resolution, with poisoned-buffer graph
replay and frozen kernel resolution for both recipes.


## GB10 vocabulary projection extension

The September 5, 2026 head measurements add exact `(N,K)=(248320,2560)`
coverage on GB10. The target verifier uses M=4 with MTP=3; the compacted draft
head uses M=1. The profile also covers MXFP8 M=1 for non-speculative decode.

| Recipe | M | A16 tile `(N,K,split)` | A16, µs | Quantized wrapper, µs |
| --- | ---: | --- | ---: | ---: |
| NVFP4 | 1 | `(128,128,4)` | 1605.5 | 1741.3 |
| MXFP8 | 1 | `(64,128,8)` | 2729.2 | 2882.5 |
| MXFP8 | 4 | `(64,128,1)` | 2870.9 | 2936.8 |

These are full projection measurements using the checkpoint LM-head weights,
BF16 random hidden states, CUPTI, CUDA graph replay, and cold L2. The NVFP4
baseline includes its dynamic activation global-maximum reduction. Weight
preparation is excluded. Four blocks alternate candidate order, with 40 samples
per arm per block. The table retains complete balanced blocks passing P0,
zero throttling, and at most 30 MHz SM-clock drift; rejected blocks remain in
the raw artifact. At least two valid blocks support each promoted route.
NVFP4 M=4 measured 1706.1 versus 1699.3 µs, so it remains unpromoted in AUTO.

The GPU was `GPU-fceb76a4-e080-225e-f0e6-64ca5eaafd1f`, with 48 SMs,
PyTorch 2.13.0, CUTLASS DSL 4.6.2, and PTXAS 13.3.73. Starting source was
`3b668de5`, in `/home/luke/projects/b12x-lm-head-a16`, with the PLE fixes.
Raw timings, clock snapshots, checkpoint identity, source manifests, and the
real-wrapper benchmark are under
`/home/luke/projects/vllm-upstream-main/.profiles/qwen38-gb10-decode-20260904/mtp-optimizations/a16/`.
The embedded profile records the raw artifact's SHA256 and qualified medians.
The initial 16-configuration shape sweep is reproducible with:

```bash
CUTE_DSL_ARCH=sm_121a .venv/bin/python benchmarks/benchmark_dense_gemm.py \
  --dtype fp4-a16 --n 248320 --k 2560 --shape-name qwen38-lm-head \
  --batch-sizes 1 4 --warmup 25 --iters 25 --tune-a16 \
  --evidence /tmp/qwen38-head-nvfp4.jsonl
```

Repeat with `--dtype fp8-a16` and a fresh evidence filename for MXFP8.
The actual-checkpoint comparison reduced relative logit error against the
original BF16 weight by roughly 30% when preserving BF16 activations. This
numerical diagnostic uses synthetic hidden states, not a language-model
accuracy benchmark.

The integration also qualifies the opaque custom-op boundary through AOT
compilation. Its schema uses `activation_mode` to avoid PyTorch's reserved
functionalization argument name, and passes UE8M0 scales as a zero-copy byte
view across that boundary. Kernel-side interpretation and storage remain
unchanged. Tests cover both functionalization versions, functional and
caller-allocated outputs, workspace mutation, and both weight recipes.
