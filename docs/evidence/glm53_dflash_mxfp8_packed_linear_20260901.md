# GLM-5.3 DFlash2 packed MXFP8 linear qualification

Status: **qualified** for the configuration recorded below.

## Operation under test

The packed MXFP8 linear path quantizes an immediately consumed BF16 activation
without initializing scale-padding bytes that the selected GEMM cannot read.
The candidate changes only the quantizer selected by
`_packed_mxfp8_op`; the public allocating quantizer keeps its initialized-padding
contract.

The production workload is GLM-5.3-Flash speculative decoding with the DFlash2
draft model proposing seven tokens per verifier step. The target model uses
NVFP4 weights, tensor parallelism 4, decode-context parallelism 1, B12X target
attention, B12X target MoE, B12X target and draft linear operations, and B12X
PCIe all-reduce.

## Source and artifact identity

Both arms used vLLM commit
`30c869eef96df5a8ffbd05cc58b7d2cde846259a` and the same model snapshots:

- target: `local-inference-lab/GLM-5.3-Flash-NVFP4` revision
  `378ca54585c46542bad1f3cb3ed0d73ae51cdb62`;
- draft source: `incoai/GLM-5.3-Flash-DFlash2` revision
  `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` converted to MXFP8 with output
  weights SHA-256
  `c033e03d47c7d5608596c8fc4e9336a1fe086eb781c08fe031be2bdea1614e58`.

The comparison arms were clean source worktrees before image construction:

| Arm | B12X revision | Worktree | Image ID |
|---|---|---|---|
| Baseline | `49d82706a793e4207fd1eb2ad7921729b394c8b1` | `/root/vllm/worktrees/b12x-luke-selection-r13-20260901` | `sha256:b59a97f6a797d454e2bdd5a2a22345d92502d215c2a09651cd1c27fa1da81cf1` |
| Candidate | `86a4283add23b3fd139ed05f7dd9a43a8b799432` | `/root/vllm/worktrees/b12x-glm53-dflash-fill-20260901` | `sha256:a94c2f23d1b214600d93cc3ad41428a99c9df76eda0b0d756ecf27c2aa3d78b9` |

Candidate commit `86a4283add23b3fd139ed05f7dd9a43a8b799432` and pull-request
commit `b6a70dbef612ef3584be0656946344b58f1811b9` have stable patch ID
`fb9222f0416774158f942f6ecd0590309a8f8ddb`. The measured source change is
therefore identical to the implementation in this pull request.

## Hardware and operating mode

The server used physical GPUs 4 through 7, all NVIDIA RTX PRO 6000 Blackwell
Workstation Edition devices, at the stock 14,001 MHz maximum memory clock and a
600 W power limit. No memory-clock offset was active.

| GPU | PCI address | UUID |
|---:|---|---|
| 4 | `0000:43:00.0` | `GPU-8800cf0c-1ba5-7136-d796-2a91f9e9586e` |
| 5 | `0000:44:00.0` | `GPU-4a0aa20b-8e36-2e05-4efb-8befbf1181d4` |
| 6 | `0000:63:00.0` | `GPU-1a0323f7-8113-a1e1-c68b-f23fecf77171` |
| 7 | `0000:64:00.0` | `GPU-0027fc86-3322-ce2a-856c-f49eb61eb63e` |

## Serving and benchmark commands

Each image was launched separately with the following command. `IMAGE` was the
baseline or candidate image identified above.

```bash
GLM53_MODEL_DIR=/root/.cache/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4 \
GLM53_MODEL_SUBDIR=snapshots/378ca54585c46542bad1f3cb3ed0d73ae51cdb62 \
DFLASH_MODEL_DIR=/root/vllm/glm53f/models/GLM-5.3-Flash-DFlash2-MXFP8-dc77ff1 \
GPU_DEVICES=4,5,6,7 PORT=5051 \
/root/vllm/glm53f/diagnostics/launch_glm53_r13_selection_ab.sh \
  "$IMAGE" dflash2 glm53-mode-eval
```

The launcher fixed `MAX_NUM_BATCHED_TOKENS=4096`, seven DFlash2 draft tokens,
probabilistic draft sampling, standard rejection sampling, B12X M8 split
route/compute, a compute grid limit of 224 CTAs, B12X PCIe all-reduce, and 32
NCCL channels. It unset `NCCL_GRAPH_FILE` rather than setting an empty path.

One 4,096-token request with seed 20260902 warmed each server. The five measured
requests used this command:

```bash
python3 /root/vllm/glm53f/diagnostics/benchmark_dflash_acceptance.py \
  --base-url http://127.0.0.1:5051 \
  --model GLM-5.3-Flash \
  --max-tokens 4096 \
  --temperature 1 \
  --top-p 1 \
  --seeds 20260828,20260829,20260830,20260831,20260901
```

The client waited for an idle server around every request. It measured decode
time from the first generated content or reasoning token through stream
completion and read rank-zero vLLM speculative-decoding counters before and
after the request. Verifier throughput is
`verifier_steps / decode_seconds`, where verifier steps are draft steps plus
non-speculative target steps. This is the real serving path and isolates
execution throughput from stochastic accepted-token count better than output
tokens per second.

## Raw results

Every request generated 4,096 output tokens. Values are not rounded in the
median calculation.

| Arm | Seed | Verifier steps | Decode seconds | Verifier steps/s |
|---|---:|---:|---:|---:|
| Baseline | 20260828 | 992 | 12.860295342980 | 77.1366421644 |
| Baseline | 20260829 | 919 | 11.963052813022 | 76.8198564667 |
| Baseline | 20260830 | 1000 | 13.045868148969 | 76.6526220088 |
| Baseline | 20260831 | 899 | 11.763830488024 | 76.4206863500 |
| Baseline | 20260901 | 925 | 12.114199936972 | 76.3566727322 |
| Candidate | 20260828 | 940 | 11.749577172974 | 80.0028789259 |
| Candidate | 20260829 | 867 | 10.801281055960 | 80.2682566548 |
| Candidate | 20260830 | 928 | 11.620485752010 | 79.8589680160 |
| Candidate | 20260831 | 981 | 12.192421427990 | 80.4598172557 |
| Candidate | 20260901 | 929 | 11.591788363000 | 80.1429400631 |

The baseline median was 76.6526220088 verifier steps/s. The candidate median
was 80.1429400631 verifier steps/s. The reported ratio is
`candidate_median / baseline_median - 1`, so the candidate improved verifier
throughput by **4.55%**. This result is scoped to the configuration and hardware
recorded above.

The three independently measured B12X execution changes are not additive. A
server containing this change, parallel M8 route packing, and PCIe/MHC
dependent-launch support measured 81.2987080029 verifier steps/s, or 6.06%
above the same baseline.

## Correctness gate

The candidate source passed all 15 tests in `tests/gemm/test_mxfp8_linear.py` on
an RTX PRO 6000 Blackwell GPU. The suite covers quantized-reference parity,
unaligned output widths, K-dimension padding, CUDA graph capture, the
no-scale-padding-initialization contract, and prequantized MXFP8 graph replay.
The exact command was:

```bash
docker run --rm --gpus '"device=1"' --ipc=host --shm-size=8g \
  --entrypoint bash \
  -e XDG_CACHE_HOME=/tmp/cache-fill \
  -v /root/vllm/worktrees/b12x-mxfp8-packed-linear-direct-quant-20260901:/src:ro \
  -w /src local/glm53-r13-acceptance-eval:full-seven-flashinfer-b12x-da8f2cb3 \
  -lc 'unset NCCL_GRAPH_FILE; PYTHONPATH=/src /opt/venv/bin/python -m pytest -q tests/gemm/test_mxfp8_linear.py'
```
