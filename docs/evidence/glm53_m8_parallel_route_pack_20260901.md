# GLM-5.3 M8 parallel route-pack qualification

Status: **qualified** for the configuration recorded below.

## Operation under test

The GLM-5.3 M8 route-pack kernel prepares eight routed-expert assignments for
each of eight live tokens. The candidate launches one CTA for each
`(token, route)` pair instead of assigning one CTA to a token and serializing
its eight route-specific activation quantizations.

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
| Candidate | `945faa62bf8748d3c39c04f89d4c1560f248d0d5` | `/root/vllm/worktrees/b12x-glm53-dflash-route-20260901` | `sha256:c1b7e3d61d60e89c9e6672189712c2d466d19977e2c0268122d8528a5ca6e8ab` |

Candidate commit `945faa62bf8748d3c39c04f89d4c1560f248d0d5` and pull-request
commit `0b518ccb847d60a96c10741b5cd2465243fdf9f4` have stable patch ID
`5d632d1a918e24ac8971776eb798f99970f026e5`. The measured source change is
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
probabilistic draft sampling, standard rejection sampling, the B12X M8 split
route/compute path, a compute grid limit of 224 CTAs, B12X PCIe all-reduce, and
32 NCCL channels. It unset `NCCL_GRAPH_FILE` rather than setting an empty path.

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
the production MoE launch path and reduces sensitivity to stochastic DFlash2
acceptance relative to output tokens per second.

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
| Candidate | 20260828 | 1140 | 14.073257310956 | 81.0047009595 |
| Candidate | 20260829 | 923 | 11.352523586010 | 81.3035086875 |
| Candidate | 20260830 | 1168 | 14.414625770000 | 81.0288118912 |
| Candidate | 20260831 | 1010 | 12.411948753990 | 81.3732009388 |
| Candidate | 20260901 | 935 | 11.528770381000 | 81.1014504670 |

The baseline median was 76.6526220088 verifier steps/s. The candidate median
was 81.1014504670 verifier steps/s. The reported ratio is
`candidate_median / baseline_median - 1`, so the candidate improved verifier
throughput by **5.80%**. This result is scoped to the configuration and hardware
recorded above.

The three independently measured B12X execution changes are not additive. A
server containing this change, packed MXFP8 fill elision, and PCIe/MHC
dependent-launch support measured 81.2987080029 verifier steps/s, or 6.06%
above the same baseline.

## Correctness gate

The exact GLM-5.3 M8 split route/compute CUDA graph test passed on an RTX PRO
6000 Blackwell GPU. The test checks the NVFP4 oracle before and after live-input
mutation, three graph replays, caller-owned output storage, zero replay
allocations, frozen kernel resolution, and reuse of the compiled dynamic
kernel. The exact command was:

```bash
docker run --rm --gpus '"device=2"' --ipc=host --shm-size=12g \
  --entrypoint bash \
  -e XDG_CACHE_HOME=/tmp/cache-route \
  -v /root/vllm/worktrees/b12x-glm53-m8-route-pack-20260901:/src:ro \
  -w /src local/glm53-r13-acceptance-eval:full-seven-flashinfer-b12x-da8f2cb3 \
  -lc 'unset NCCL_GRAPH_FILE; PYTHONPATH=/src /opt/venv/bin/python -m pytest -q tests/moe/test_cute_migration_moe_standard_corpus.py::test_standard_moe_glm53_m8_split_route_compute_live_graph_oracle'
```
