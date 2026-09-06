# GPU component profiles

GPU profiles replace component-local tuning guesses with generated decisions for
recognized devices. They are consulted while plans are built; binding and replay
remain allocation-free and do not perform profile lookup.

## Integrator sequence

Omitting a policy synthesizes the cached AUTO policy for the plan's device:

```python
from b12x.attention import paged

caps = paged.Caps(...)
plan = paged.plan(caps)
binding = paged.bind(plan, ...)
paged.compile(binding=binding)
paged.run(binding=binding)
```

An integrator that constructs several component plans can resolve the device once
and pass one immutable context through all of them:

```python
from b12x.moe import fused_moe
from b12x.policy import get_auto_policy
from b12x.sequence import gdn_decode

policy = get_auto_policy("cuda")
gdn_plan = gdn_decode.plan(gdn_caps, policy=policy)
moe_plan = fused_moe.plan_execution(
    experts=experts,
    capacity=fused_moe.ExecutionCapacity(max_tokens=8, top_k=10),
    policy=policy,
)
```

The same contract is used by every planned op: attention, GEMM scratch plans,
MoE, normalization, quantization, and sequence components. A component owns its
typed query, profile decoder, heuristic, validation, planning, and execution.
The generic policy layer owns device matching, precedence, provenance,
serialization, and registry lookup. Components with only one production
implementation still resolve a typed backend config so a future implementation
can be introduced without changing the integration sequence.

`b12x.policy.list_planning_components()` is the authoritative inventory for
every planned op. All built-in planned ops are `profiled` and own lazy
runtime-policy and generator registrations. Package loading rejects an embedded
profile that omits a registered component. The component schema is validated
before a matching config is returned; invalid matching data fails closed.

One-shot `gemm.blockscaled` participates through the separate catalog inventory
`ONESHOT_COMPONENTS`. Its `gemm.blockscaled_precision` policy resolves static
weight geometry during prewarm; live M selects a precision route from
that cached config. Graph replay retains the captured route. This does not
introduce a public GEMM planning API. See
[dense GEMM activation precision](dense-gemm-precision.md) for shared weight
storage, workspace, and qualification contracts.

## Precedence and overrides

Resolution order is:

1. A call-specific config override.
2. A component override stored in the `PolicyContext`.
3. A matching entry in the library-embedded device profile.
4. The component heuristic when the device, component, or query is not covered.

A matching embedded entry is authoritative: invalid embedded data fails closed
instead of falling back to a heuristic. Explicit operational modes remain
available for qualification and emergency rollback:

In AUTO mode, a device, component, or query miss logs a warning once for each
distinct component, device, reason, and encoded query before using the
heuristic. Replanning the same missing query is quiet, while a different
uncovered capacity is still reported. Explicit `HEURISTIC_ONLY` mode is
intentional and does not warn.

```python
from b12x.policy import PolicyContext, PolicyMode

heuristic = PolicyContext.for_device(
    "cuda",
    mode=PolicyMode.HEURISTIC_ONLY,
)
preplanned = PolicyContext.for_device(
    "cuda",
    mode=PolicyMode.PREPLANNED_ONLY,
)
```

`B12X_POLICY_MODE=auto|heuristic-only|preplanned-only` selects the default
context used when an integration omits an explicit policy. Explicit contexts and
component config overrides still take precedence.

## Inspecting model selections

The model-policy inspector expands a reviewed model preset into its relevant
component queries and prints the selected kernels, configs, rules, and
provenance. It performs policy lookup only and does not allocate model weights:

```bash
./scripts/inspect_model_policy.py --list-models
./scripts/inspect_model_policy.py qwen3.8-flash-next-180b \
    --tp 1 --device gb10
./scripts/inspect_model_policy.py deepseek-v4-flash \
    --tp 2 --device gb10
./scripts/inspect_model_policy.py minimax-m3 \
    --tp 4 --device gb10
./scripts/inspect_model_policy.py glm-5.2 \
    --tp 8 --device gb10
./scripts/inspect_model_policy.py glm-5.3 \
    --tp 8 --device nvidia.rtx.pro.6000.blackwell
./scripts/inspect_model_policy.py glm-5.3-flash \
    --tp 4 --device gb10
./scripts/inspect_model_policy.py qwen3.8-27b \
    --tp 2 --device nvidia.gb10.48sm --json
```

The installed command is `b12x-inspect-model-policy`. Device selection accepts
`auto`, an embedded profile ID, or an unambiguous product-name fragment. Each
row reports `preplanned` or `heuristic`, so uncovered shapes are visible rather
than silently presented as tuned decisions.

Preset contracts are derived from the production presets in
`benchmarks/benchmark_moe.py` and the attention benchmark suite. GLM-5.2
includes its DSA indexer, sparse MLA, and ModelOpt W4A8/NVFP4 MoE paths.
GLM-5.3 Flash composes KDA, pooled DSA indexing, no-RoPE sparse MLA, mHC, and
ModelOpt NVFP4/A16 MoE. The GLM attention presets are qualified through TP8; TP16
would leave four local attention heads and is rejected until that kernel shape
passes its oracle. The independent MoE corpus still covers TP1 through TP16.

The catalog also includes every model profile exposed by
`benchmark_moe.py`: Qwen3.5-397B, Nemotron Super, Nano3.5, both DSV4F weight
recipes, MiniMax-M2.7/M3, Laguna S-2.1, DeepSeek V4 Flash, and GLM-5.1. The
paged-attention, QSA, GDN, dense/sparse/compressed MLA, DSA/MSA indexer, and
paged-indexer benchmark presets and default suites contribute their component
contracts. Shape-only and historical benchmark spellings remain accepted
aliases, while `--list-models` prints one canonical name for each deduplicated
model.

## Generation boundary

One top-level command discovers every registered component, prints the complete
work estimate, runs every provider, reduces measured races into decision trees,
validates the serialized profile, and optionally embeds the compact runtime
payload:

```bash
./scripts/generate_gpu_profile.py --dry-run
./scripts/generate_gpu_profile.py --overwrite --embed
```

Identical GPUs can measure one profile concurrently. CUDA ordinals are relative
to `CUDA_VISIBLE_DEVICES`; `all` selects every visible GPU:

```bash
./scripts/generate_gpu_profile.py --devices 0-11 --dry-run
./scripts/generate_gpu_profile.py --devices 0-11 --overwrite --embed
# Equivalent when the tuning node exposes only the intended GPUs:
./scripts/generate_gpu_profile.py --devices all --overwrite --embed
```

The parent uses spawned worker processes, pins one process to each selected GPU,
and dynamically schedules checkpoint-disjoint measurement partitions. Discrete
sweeps keep each allocation group together; MoE keeps every screen, coarse race,
and route distribution for one physical geometry together. Fixed-backend
qualifications stay whole. A single parent process performs the final reduction
and writes the artifact after every worker succeeds, so concurrent workers never
write competing profile files.

Every selected GPU must report the same product name, compute capability, and SM
count. Completed cases use the same shared checkpoint directory as a single-GPU
run. After an interruption, rerun the command with the same `--work-dir`; the
number or ordinals of identical GPUs may change without invalidating completed
measurements.

No `--components` argument is needed for a full device profile; the default is
all registered components. `--components` exists only for targeted development
and resume diagnostics. A subset run automatically merges into an existing
output profile, retaining every unselected component; `--merge-from` selects an
explicit base when the output does not already exist. Every completed provider
must report at least one real
production-path GPU measurement. Components with one legal implementation run
a correctness-gated qualification sweep rather than inventing alternatives or
serializing an unmeasured heuristic.

Completed MoE geometries resume entirely from checkpoint metadata. Candidate
enumeration and eligibility run on the host; a CUDA worker and expert weights
are created lazily only when a race checkpoint is missing.

Checkpoint compatibility is based on checkpoint schema, device identity,
measurement case and candidate IDs, and timing settings. `source_revision` is
provenance rather than a measurement input, so committing an identical source
tree or extending a corpus does not discard unrelated measurements. A cached
sampling protocol may satisfy a weaker requested protocol, but not the reverse.
Discrete sweep checkpoints also carry a component-owned candidate-contract
version. A fully compatible allocation group skips session setup and candidate
enumeration even after unrelated source changes or a commit. Providers must
bump that version when candidate enumeration or eligibility changes; case IDs
independently invalidate corpus changes. Legacy checkpoints receive one
candidate-ID comparison before being upgraded to this contract.
Fixed-backend qualifications similarly persist the ordered probe case IDs and
the qualified config, so changing either invalidates only that component's
checkpoint.

The built-in measured corpus covers common model geometries and TP sizes 1
through 16,
common top-k and decode token counts, multiple route distributions, GDN serving
shapes, GQA context/page/KV-dtype combinations, and dense and sparse MLA shapes.
Unaligned low-width MoE shards are padded to their recipe's physical minimum
instead of being discarded.

MoE model entries declare compatible checkpoint-format families rather than a
single benchmark recipe. Generation crosses each geometry with every recipe in
that family that supports the model activation, so ModelOpt NVFP4, W4A16, and
W4A8 variants share geometry coverage without pretending unsupported
activation/recipe pairs are runnable.

Attention serving capacities are dense from one through sixteen sequences and
then use 32, 64, 128, and 256 as larger anchors. Components with a prefill path
also capture 1,024, 2,048, 4,096, and 8,192 query-token capacities. GDN
state-index columns are a physical tensor and loop capacity, independent of
whether an integrator calls the corresponding multi-token transaction
speculative verification.

MoE measures every token count from 1 through 8 and additional anchors through
128. Fixed-precision reduction fills the bounded 1--128 serving domain from the nearest valid
measured anchor. It never extends micro beyond eight tokens or Triton route
packing beyond 256 routed rows, and it does not extrapolate outside the recorded
domain. Profile coverage reports measured and synthesized runtime query counts
separately.

The `modelopt-nvfp4-auto` recipe races A4 and A16 over shared native NVFP4
storage for SiLU. It selects precision at execution capacity, covers only
exact measured capacities, and falls back to native A16 direct decode at
capacities 1–8 on SM120 and SM121 when supported, or A4 otherwise. It independently
qualifies every capacity without coarse precision pruning. A16 wins confirmed
median parity as well as speedups; each candidate uses its own precision oracle.
See [MoE storage and precision planning](moe-execution-model.md#sharing-nvfp4-storage-across-activation-precisions).

`benchmarks/benchmark_moe_precision_policy.py` runs this registered provider for
one reviewed geometry and a bounded set of capacities. Its default input is
the generator's synthetic corpus; `--input-snapshot` instead races the exact
checkpoint operands exported by `benchmark_nvfp4_decode_precisions.py`.
The manifest distinguishes those inputs and preserves tensor identities.

Corpus definitions live in generator code and are not repeated or referenced in
JSON. The full local artifact retains the device, settings, aggregate winners,
and source revision. The checkpoint tree retains per-candidate correctness and
timing results needed to audit a run. Package data under
`b12x/policy/_profiles/data/` is gzip-compressed and contains only the validated
runtime planner: no corpus pointers, repeated evidence, coverage, or metadata.
