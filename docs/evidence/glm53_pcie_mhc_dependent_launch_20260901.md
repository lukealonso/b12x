# GLM-5.3 PCIe and MHC dependent-launch qualification

Status: **qualified** for the configuration recorded below.

## Operation under test

The PCIe one-shot all-reduce producer can release a programmatic dependent
launch after it publishes peer-visible data. The MHC partial consumer can use
programmatic dependent launch and wait for its producer. The behavior is
enabled by `B12X_PCIE_ONESHOT_PDL=1` and `B12X_MHC_PDL=1`; both gates retain a
disabled compatibility path.

The production workload is GLM-5.3-Flash speculative decoding with the DFlash2
draft model proposing seven tokens per verifier step. The target uses NVFP4
weights, tensor parallelism 4, decode-context parallelism 1, B12X attention,
B12X MoE, B12X linear operations, and B12X PCIe all-reduce.

## Source and artifact identity

Both arms used vLLM commit
`30c869eef96df5a8ffbd05cc58b7d2cde846259a` and identical model snapshots:

- target: `local-inference-lab/GLM-5.3-Flash-NVFP4` revision
  `378ca54585c46542bad1f3cb3ed0d73ae51cdb62`;
- draft source: `incoai/GLM-5.3-Flash-DFlash2` revision
  `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, converted to MXFP8 with output
  weights SHA-256
  `c033e03d47c7d5608596c8fc4e9336a1fe086eb781c08fe031be2bdea1614e58`.

The comparison arms were clean source worktrees before image construction:

| Arm | B12X revision | Worktree | Image ID |
|---|---|---|---|
| Baseline | `49d82706a793e4207fd1eb2ad7921729b394c8b1` | `/root/vllm/worktrees/b12x-luke-selection-r13-20260901` | `sha256:b59a97f6a797d454e2bdd5a2a22345d92502d215c2a09651cd1c27fa1da81cf1` |
| Candidate | `e884df0107a25c0a4918cd00b295f62716a75938` | `/root/vllm/worktrees/b12x-glm53-dflash-pdl-20260901` | `sha256:be5c52da54b8c32a3696f9b7bf08d03b4cc78c600301bd531e8c78d6d424a292` |

Candidate commit `e884df0107a25c0a4918cd00b295f62716a75938` and pull-request
commit `af375a3c6579eb31950935ba03cb6a294b3efbfa` have stable patch ID
`9728e06d91cb5f6f5430079fa1071b783ff84527`. The measured source change is
identical to the implementation in this pull request.

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
baseline or candidate image identified above. The baseline did not contain the
dependent-launch change; the candidate launcher set both feature gates to one.

```bash
GLM53_MODEL_DIR=/root/.cache/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4 \
GLM53_MODEL_SUBDIR=snapshots/378ca54585c46542bad1f3cb3ed0d73ae51cdb62 \
DFLASH_MODEL_DIR=/root/vllm/glm53f/models/GLM-5.3-Flash-DFlash2-MXFP8-dc77ff1 \
GPU_DEVICES=4,5,6,7 PORT=5051 \
B12X_PCIE_ONESHOT_PDL=1 B12X_MHC_PDL=1 \
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
completion and read rank-zero vLLM speculative-decoding counters around the
request. Verifier throughput is `verifier_steps / decode_seconds`, where
verifier steps are draft steps plus non-speculative target steps. It exercises
the production PCIe all-reduce and MHC launch path and reduces sensitivity to
stochastic DFlash2 acceptance relative to output tokens per second.

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
| Candidate | 20260828 | 857 | 10.696098148997 | 80.1226753964 |
| Candidate | 20260829 | 1007 | 12.576726559960 | 80.0685293744 |
| Candidate | 20260830 | 874 | 10.914943808980 | 80.0737058564 |
| Candidate | 20260831 | 1065 | 13.271201774010 | 80.2489494271 |
| Candidate | 20260901 | 911 | 11.418855249010 | 79.7803264981 |

The baseline median was 76.6526220088 verifier steps/s. The candidate median
was 80.0737058564 verifier steps/s. The reported ratio is
`candidate_median / baseline_median - 1`, so the candidate improved verifier
throughput by **4.46%**. This result is scoped to the configuration and hardware
recorded above.

The three independently measured B12X execution changes are not additive. A
server containing this change, packed MXFP8 fill elision, and parallel M8 route
packing measured 81.2987080029 verifier steps/s, or 6.06% above the same
baseline.

## Correctness gate and limitation

The production candidate completed the five long requests above without a
server error. The focused MHC suite was also run once with both dependent-launch
gates disabled and once with both enabled. Both modes produced the same two
passes and the same four strict reference-tolerance failures, including the
same maximum errors and indices. The dependent-launch change therefore did not
alter those results, but this evidence does not claim that the complete MHC
suite is green.

The enabled command was:

```bash
docker run --rm --gpus '"device=3"' --ipc=host --shm-size=8g \
  --entrypoint bash \
  -e XDG_CACHE_HOME=/tmp/cache-pdl \
  -e B12X_MHC_PDL=1 -e B12X_PCIE_ONESHOT_PDL=1 \
  -v /root/vllm/worktrees/b12x-pcie-mhc-pdl-20260901:/src:ro \
  -w /src local/glm53-r13-acceptance-eval:full-seven-flashinfer-b12x-da8f2cb3 \
  -lc 'unset NCCL_GRAPH_FILE; PYTHONPATH=/src /opt/venv/bin/python -m pytest -q tests/norm/test_mhc.py'
```

The disabled run used the same command with both gate values set to zero. The
shared failures were maximum absolute errors of `3.6239624e-05` and
`2.4294853e-04` in MHC post values and two BF16 one-ULP differences of
`0.0078125` in normalized activations. These failures predate and are invariant
under the dependent-launch toggle; their tolerances were not weakened.
