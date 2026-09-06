# b12x Guidance

- Follow the repository's existing CuTe DSL patterns for core compute kernels,
  including GEMM, attention, and MoE. Use Triton only for minor supporting
  metadata or packing kernels; do not use it for core compute kernels or their
  prototypes.
- Benchmark the real target path before making performance claims. Capture the
  command, commit, worktree, GPU mode, correctness state, raw timings, and
  ratio direction.
- Treat CUDA graph capture/replay, warmup behavior, stable allocation, and
  fixed or preplanned workspace capacity as serving requirements, not optional
  benchmark details.
- Never include live request quantities in a kernel compile or cache key. Token,
  row, batch, sequence, expert, page, block, and occupancy counts must be runtime
  scalar launch arguments that drive dynamic grids and masks. Compile and cache
  keys may include only static model geometry, planned capacity, device identity,
  and toolchain identity. Precompile capacity specializations before graph capture,
  and test multiple live counts under frozen kernel resolution to prove that they
  reuse the same compiled callable.
- Correctness gates come before performance claims. Validate oracles,
  cosine/top-k equality, nonzero tensors, quantization semantics, and boundary
  behavior before interpreting timings.
- When tile, knob, or local sweeps plateau, profile or inspect architecture
  evidence and pivot to structural ideas. Debug promising crashes rather than
  discarding high-signal variants immediately.
- `b12x` should own planner and policy decisions. Integrations should supply
  metadata and capacity limits rather than duplicating b12x policy.
- W4A16 means BF16 activations with inline FP4/NVFP4 weight dequantization. Do
  not reintroduce activation-scale math into W4A16 kernel math.
- Compressed MLA and GLM MLA/NSA are distinct contracts. Verify tensor layout,
  head dimensions, local/global roles, and TP axes before combining assumptions.
- Do not claim fused, direct, native, or production progress through packed
  adapters, CPU/Torch/reference fallbacks, or fake serving routes.
- Verify the current hardware and skill docs before choosing architecture env
  values. Do not reuse stale arch strings or benchmark leaders without current
  evidence.

## GPU component policy system

The policy layer selects plan-time backends and launch configuration. It does
not participate in binding or replay. See `docs/gpu-profiles.md` for the
integrator-facing sequence and `b12x/policy/` for the contracts.

- `b12x/policy/catalog.py` is the authoritative inventory of planned ops. Every
  built-in op with `api_style="planned"` must have exactly one `PROFILED`
  registration containing its component ID, runtime policy, and offline
  generator. Do not introduce an unregistered component-local device heuristic.
- Each component owns a typed `Query`, typed `Config`, and `ComponentPolicy` in
  its `_policy.py`. Queries describe immutable model geometry, dtype/layout,
  recipe, and planned capacity. They must not contain live request values.
  Configs contain the selected backend and any real launch or planner knobs.
- A component policy owns `encode_query`, `decode_profile`, `heuristic`, and
  `validate_config`. Both preplanned and heuristic configs must pass the same
  validation before a plan may use them. Bump `query_schema_version` when query
  fields or semantics change; bump `config_schema_version` when serialized
  config fields or semantics change.
- Public planning entry points accept `policy: PolicyContext | None`. When it is
  omitted, synthesize the cached AUTO context with `get_auto_policy()` for the
  plan's device, verify the context/device match, resolve exactly once during
  planning, use the typed config, and retain `PolicyResolution` on the plan for
  provenance. Bind and run paths must not perform policy lookup.
- Resolution precedence is call override, context override, matching embedded
  profile entry, then the component heuristic. Unknown devices, missing
  components, and uncovered queries use the heuristic in AUTO mode. A matching
  but malformed or invalid embedded entry fails closed; never disguise bad
  profile data as a heuristic miss. Preserve `HEURISTIC_ONLY` and
  `PREPLANNED_ONLY` qualification modes.
- Device profiles match exact normalized `vendor`, `product_name`, compute
  capability, and SM count. A component entry carries independent query/config
  schema versions and exactly one planner tree or rule set. Planner nodes are
  unconditional `leaf`, scalar `exact`, or disjoint inclusive `range` nodes,
  each with an optional default. The tree determines dispatch coverage;
  `coverage`, `evidence`, and `source_revision` are audit metadata only.
- The embedded registry is immutable and must contain exactly the component set
  registered in the catalog. Provider IDs and schema versions must match their
  runtime policies. Keep package-embedded profiles compact and validated.
  Generator checkpoints, full evidence artifacts, probes, and service A/B logs
  are local working data unless a reviewed change explicitly requires them; do
  not leave large untracked validation directories in the repository.
- The top-level generator must discover every registered provider, show the
  complete work estimate before execution, support component subsets and resume,
  and run every registered component by default from
  `scripts/generate_gpu_profile.py`. Multi-candidate components race real
  production plans; single-implementation components must qualify that path on
  the GPU before emitting a profile. A completed profile may not contain a
  zero-measurement or precomputed provider.
- Resumable discrete sweeps own a positive `candidate_contract_version`. Bump
  it whenever candidate enumeration or eligibility changes; corpus changes are
  independently invalidated by case IDs. Fixed-backend probes must expose
  stable, ordered case IDs, and their checkpoints must bind those IDs to the
  qualified serialized config.
- Tests must keep planned-op metadata, catalog registrations, generators, and
  every embedded profile in lockstep. Cover a recognized GPU resolving to
  `PREPLANNED`, an unknown synthetic GPU resolving to `HEURISTIC`, invalid
  matching data failing closed, override precedence, and representative public
  plan construction for every component.

## 64-bit addressing for pool-scaled offsets

Any arithmetic that scales a page/block/row id into a byte or element offset
(`pid * page_stride`, `block_id * row_stride`, ...) must be done in `Int64`.
Serving pools on 100GB+ unified-memory parts exceed `2^31 / stride` pages, and
allocators hand out high recycled ids — an `Int32` product works in every
benchmark and dies in production.

Corollary for testing: every repro or test for a kernel that indexes a paged
pool MUST include a big-pid case with live pages parked past the
`2^31 / stride` line (allocate a large mostly-uninitialized pool and point the
page table at its tail). Small sequential test ids can never catch a 32-bit
offset overflow: the direct-K indexer score (fixed in 778f66d) passed an
exact-reference adversarial sweep and compute-sanitizer clean, then crashed
vLLM with an illegal access on the second pass of every cached prompt — the
first pass gets low pool ids, the second lands on high recycled ones.

## CUTLASS DSL migration methodology

- Keep compiler migrations in a dedicated worktree with a matching isolated
  virtual environment. Keep the main worktree pinned to its known-good version
  until the migration is accepted.
- Preserve migration evidence infrastructure as durable project tooling. Keep
  corpus launchers, exact-object/resource auditors, SASS register-set tools,
  graph-replay benchmarks, source/contract manifests, and integrity checks.
  Remove rejected experiments and temporary overlays, not the machinery needed
  to reproduce the decision. Keep GPU tests in `tests/`, general benchmarks in
  `benchmarks/`, and migration acceptance, diagnostics, evidence, and offline
  integrity checks under `validation/cutlass_migration/`.
- Freeze the source before final evidence collection and compile that identical
  source under both compiler versions with fresh caches. Keep the pristine
  pre-migration comparison separate so source fixes cannot hide a compiler
  regression.
- Keep raw artifact identity separate from cross-toolchain comparison identity.
  Pair arms only through a reviewed normalized identity that removes explicitly
  enumerated operational/toolchain-only fields; never weaken raw cache,
  manifest, object, cubin, PTX, or semantic hashes. Load exact cache objects
  through verified temporary copies because the CUTLASS external-binary loader
  may patch an ELF while loading it; verify the manifest-bound source object
  again after timing.
- Migration acceptance is GPU-only and must exercise the real production
  launch path on the explicitly selected physical GPUs. Static validation may
  prove coverage or artifact integrity, but CPU-only tests and reference-only
  routes are not migration acceptance evidence.
- Run one closed-corpus resource census that covers every production kernel,
  compile specialization, and CUDA entry point. Include decode, boundary, and
  prefill shapes. Compare allocated GPRs and exact R, UR, P, and UP use, plus
  register reconfiguration, stack/frame and local loads/stores, launch/static
  SMEM, occupancy, instruction count, and code size. Flag every positive
  register-use delta, not only spills.
- A flagged resource delta does not automatically require a full benchmark
  matrix. Prioritize new or increased local memory, occupancy or SMEM changes,
  large register jumps, and known performance cliffs. Keep smaller positive
  deltas visible in the accounting table with a concise disposition. Use
  common-PTXAS/PTX/SASS analysis when the cause is ambiguous or the delta
  correlates with performance, not as a blanket rerun for every row.
- Keep runtime qualification proportional. Exercise changed and cliff-prone
  production paths plus representative decode, boundary, and prefill sizes.
  Confirm a real cliff and its final fix on both target GPUs; do not run the
  full GPU-by-toolchain-by-specialization cross-product by default.
- Require correctness and serving invariants before timing: the relevant GPU
  oracle, finite/nonzero output, quantization semantics, graph replay, stable
  addresses, fixed workspace capacity, and no replay allocation. Add poison,
  mutation, or immutability checks where the path can otherwise pass falsely.
- Benchmark exact cached objects with balanced ordering and adequate timing
  resolution. Record the command, source/artifact hashes, CUTLASS/PTXAS map,
  physical GPU UUID and mode, correctness state, raw warm/cold samples, and
  ratio direction. Formal release evidence requires its declared exact
  throttle mask. A targeted diagnostic on Max-Q hardware may explicitly allow
  only the observed `0x0`/`0x4` software-power-cap transition when both arms are
  interleaved, both snapshots remain P1, identity and memory clocks are stable,
  the SM-clock delta stays within the declared limit, and the transition is
  recorded. Reject every other throttle reason.
- Prefer structural fixes that shorten true live ranges or express the new
  typed-SMEM/pipeline contract while preserving math, quantization,
  synchronization, launch geometry, and planner policy. Treat a performance
  cliff as unresolved until fixed or explicitly documented with evidence.
