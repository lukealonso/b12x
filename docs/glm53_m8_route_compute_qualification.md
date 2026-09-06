# GLM-5.3 M8 Routed-Expert Qualification

## Status and contract

Status: **implemented and qualified** for the GLM-5.3 Flash target-model
routed-expert shape.

The specialization applies only to ModelOpt NVFP4 SiLU experts with 288
experts, hidden size 4096, intermediate size 512, eight input rows, top-k 8,
grouped external routing, and a materialized or persistent work source. It
separates fixed-geometry route preparation from expert computation. Other
shapes retain the existing dynamic fused path.

Host policy is immutable after `plan_tp_moe_execution` returns. Scratch
planning compiles the route-preparation and compute artifacts for both int32
and int64 route indices and retains strong references to all four artifacts.
The execution path therefore cannot perform first-use compilation during CUDA
graph capture. Live input-row counts and cluster limits remain launch scalars
and are not compiled-kernel cache keys.

## Source identity

The isolated performance comparison used the same vLLM package and changed
only the B12X source range shown below:

| Role | B12X commit | Git tree | Behavior |
| --- | --- | --- | --- |
| Reference | `b85d9e88fcdc1ae8c0dfef2ab907e357f7b53331` | `42f0cc48b605af343e6427829e9c208459e25037` | Packed MXFP8 fill elision, without M8 route/compute specialization |
| Specialized | `f5274e4c369b8252612c5c66118686a3a8e5f234` | `134dfa06eb2b6d3994aeddab4666bbb7bf3e2a92` | Reference behavior plus the M8 route/compute specialization and its safety checks |

Both benchmark arms used vLLM ref
`build/jovian-judgement-community-r13-performance-20260901` at commit
`75b0286720f5d8d9fdd8bc3f1c3849cb3751ec8f`, source tree
`fb69743d8e6b05528d06a94a746326183699ddb9`, and installed package tree
`a1ec669685da5cf01e0c8d5a307ad30aa47761fb`. The durable benchmark artifacts
are `r13-component-mxfp8-fill-nospec-c1-c8-ctx0.json` for the reference and
`r13-component-m8-nospec-c1-c8-ctx0.json` for the specialization. The source
and package identities are recorded by
`glm53-jovian-judgement-community-20260901-r13.source.lock`.

The pull-request representation of the specialized source ends at
`83a10936753e2f9aeec2bdb416dc026f7e2caba5`. Commit
`13bbc002f0d4cc1e4ce8b929ff61e2341bdcd880` adds immutable plan metadata and
plan-time artifact compilation without changing the kernel implementation.
The implementation commit retains `MadeBy561 <madeby561@gmail.com>` as author.

## Hardware and serving configuration

Both performance arms used physical GPUs 4, 5, 6, and 7 on one host:

- four NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs;
- GPU UUIDs `GPU-8800cf0c-1ba5-7136-d796-2a91f9e9586e`,
  `GPU-4a0aa20b-8e36-2e05-4efb-8befbf1181d4`,
  `GPU-1a0323f7-8113-a1e1-c68b-f23fecf77171`, and
  `GPU-0027fc86-3322-ce2a-856c-f49eb61eb63e` at PCI buses `43:00.0`,
  `44:00.0`, `63:00.0`, and `64:00.0`, respectively;
- PCIe Gen5 x16 links;
- default compute mode, persistence mode enabled, and a 600 W power limit;
- tensor parallelism 4 and decode-context parallelism 1;
- a +6000 MHz memory-clock offset in both isolated performance arms;
- full decode CUDA graphs;
- B12X attention, linear, NVFP4 W4A4 MoE, and PCIe all-reduce;
- independent 512-token target and recurrent cache pages;
- 4096 maximum batched target tokens and 32 NCCL channels;
- GLM-5.3 Flash NVFP4 checkpoint revision
  `378ca54585c46542bad1f3cb3ed0d73ae51cdb62`.

Each server completed CUDA-graph capture and a five-second concurrency-one
warmup before measurement. No other process used GPUs 4-7.

## Isolated performance evidence

The production server was measured with `llm-decode-bench` 0.4.29:

```bash
python /root/llm_decode_bench.py \
  --host 127.0.0.1 \
  --port 5051 \
  --model GLM-5.3-Flash \
  --concurrency 1,8 \
  --contexts 0 \
  --max-tokens 8192 \
  --duration 30 \
  --decode-warmup-seconds 5 \
  --temperature 0 \
  --skip-prefill \
  --display-mode plain \
  --no-hw-monitor \
  --no-resume \
  --output RESULT.json
```

The table contains every measured 30-second sample. Warmup throughput is not
included. Change is `(specialized / reference - 1) * 100`.

| Concurrency | Reference output tok/s | Specialized output tok/s | Change |
| --- | ---: | ---: | ---: |
| 1 | 167.543290554 | 168.022133113 | +0.286% |
| 8 | 752.782287994 | 790.077085974 | +4.954% |

The specialization is throughput-neutral at concurrency one and improves the
eight-request production path by 4.95% under the declared configuration.
Both artifacts completed with zero request errors. The throughput workload is
not a semantic output oracle; implementation correctness is established by the
exact-shape tests described below. Those tests passed for the specialized arm,
while the reference arm retained the previously qualified unspecialized
dynamic path.

## Plan-time compilation correctness

The plan-time compilation commit was validated on one NVIDIA RTX PRO 6000
Blackwell Workstation Edition GPU with these repository test targets:

```bash
pytest -q tests/moe/test_fused_moe_planning.py
pytest -q \
  tests/moe/test_cute_migration_moe_standard_corpus.py::test_standard_moe_glm53_m8_split_route_compute_live_graph_oracle
pytest -q \
  tests/moe/test_tp_moe_scratch_bindings.py \
  tests/moe/test_w4a8_dynamic_kernel.py
```

The results were 54 passed, 1 passed, and 56 passed. The exact-shape oracle
poisons caller-owned scratch, freezes kernel resolution before eager execution
and graph capture, captures and replays the CUDA graph, verifies stable tensor
addresses and allocation-free replay, and checks the output against the
pure-Torch NVFP4 reference. Changing
`B12X_DYNAMIC_SPLIT_COMPUTE_MAC` from 224 to 112 after planning does not change
the planned scalar or any compiled-artifact identity.

## End-to-end compatibility after plan-time compilation

An integration comparison isolated commit `13bbc002f0d4cc1e4ce8b929ff61e2341bdcd880`
on stock-clock GPUs 4-7. Both arms used the same vLLM tree, target and DFlash2
MXFP8 checkpoints, seven probabilistic draft tokens, and the complete B12X
execution stack. The reference B12X integration commit was
`035a74c2`; the planned-artifact integration commit was `d393b6a2` with package
tree `d3f504f98ca7f645c322304a4cb3674ffeab6569`.

Five 4096-token concurrency-one requests used seeds 20260828 through 20260901.
Draft sampling does not honor per-request generators, so emitted-token rate and
accepted length vary stochastically. Verifier steps per second is the stable
execution metric.

| Seed | Reference steps/s | Planned-artifact steps/s |
| ---: | ---: | ---: |
| 20260828 | 81.298708003 | 80.947930733 |
| 20260829 | 81.306328871 | 81.456265853 |
| 20260830 | 81.286600180 | 81.172607529 |
| 20260831 | 81.332062822 | 82.130121545 |
| 20260901 | 81.205706571 | 81.379033115 |
| Median | 81.298708003 | 81.379033115 |

The median change is +0.099%, which is within run-to-run variation. A separate
30-second sustained-decode run measured 85.144 and 278.445 verifier steps/s at
concurrency one and eight. A 30-second cold 32k prefill run measured 14,415
prompt tok/s; the matching integration reference measured 14,416 prompt tok/s.
The plan-time compilation contract therefore introduces no measured decode or
prefill regression.
