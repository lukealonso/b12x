# GPU tuning problems

The catalog binds each GPU policy to a `TuningProblem` in its component's
`_policy.py`. The contract accounts for every typed query and configuration
field. Each generator exposes its production measurement program as
candidate-race tasks, fixed-backend probes, or a composition of both. The DSA
indexer composes production-path qualification with a merge-kernel race.
Inspect the executable inventory with:

```sh
python scripts/inspect_tuning_problems.py
```

## Coverage and ownership

**Implemented:** all 21 planned public APIs and the prepared blockscaled
precision selector have registered runtime contracts. Their 22 components expose
23 measurement tasks: 16 candidate races and seven fixed-backend probes. Six
additional API providers qualify BF16 GEMV, native MXFP8 BMM, MLA query projection,
tensor-FP8 linear, MXFP8 row quantization, and native Trellis linear. Their typed contracts live in
component `_tuning.py` modules. The generator preserves their measurements under
`evidence.api_qualifications`; they do not add runtime profile entries. The
MXFP8 linear compatibility API declares its shared `mm` and `pack_weight`
callables as aliases of the blockscaled API. Tests verify that identity.

All 16 candidate-race tasks provide factories for complete typed shape queries.
Blockscaled precision retains sampled live-row inputs and its independent
confirmation constraints; it cannot emit an unmeasured precision selector.
Registration establishes ownership and executable measurement tasks, rather
than complete coverage of internal kernels, recipes, or compilation specializations.

**Implemented:** providers use the shared preparation, reset, and balanced
candidate timing lifecycle. Specialization tooling inventories source entry
points, compile specs, memoized functions, persistent host state, and possible
cache accesses. It can also verify supplied cached objects, observation stores,
and CUDA launch traces. Complete ownership, alias resolution, and instantiated
GPU coverage remain unqualified. Several decision domains delegate enumeration
to their component provider. Independent region qualification for all components
remains required. Spatial emission requires a provider-owned legality validator for
unmeasured queries. NVFP4 quantization, fused MoE, GQA, varlen attention, KDA
prefill, BF16 vocabulary projection, and block-FP8 linear provide validators and
production fixture factories for their supported shape contracts. HyperConnection
and MTP feedback support arbitrary positive token capacities at their benchmark
geometry: BF16, hidden width 2560, four streams, and HyperConnection low rank 320.
mHC supports positive BF16 token capacities for hidden-width/scratch-split pairs
4096/64 and 7168/112; its candidate set permits TF32 projection from 384 tokens.
Dense MLA supports full-window graph contracts with one row per decode sequence
or one extend sequence, including mixed BF16-query/FP8-cache quantization scales;
its validator checks the production shared-memory limit and split candidates.
GDN supports Qwen's 3:1 value/key head ratio and KDA's 1:1 ratio, with full and
partial live scenarios for token capacities within the sequence/column capacity.
Its fixtures preserve the requested gate, QK normalization, and state dtype.
Vocabulary projection fixtures require one token; block-FP8 fixtures require
BF16 output and complete 128-row weight blocks. Unsupported combinations fail
preflight instead of substituting a nearby fixture.

**Qualified on the declared compressed-MLA corpus:** the public production
plans use 512-wide BF16 queries and outputs with DSV4 cache records. RTX PRO
6000 Max-Q passed 768 candidate measurements across 288 queries; GB10 passed
748 across the same 288 queries. Both runs had zero correctness or replay
allocation failures. Minimum output cosine was 0.9999889 on RTX and 0.9999893
on GB10. Their measured entries are embedded in the two device assets.
The runtime policy and generator share the production single-pass chunk limit;
dual-cache single-pass fixtures require a 128-token SWA width. High-pool-offset
and partial-index graph tests pass on both GPUs. The three-case RTX memcheck
run completed with zero errors. Production dispatch preserves its selected QK
compute mode, partial index loads are bounded, and prefill uses planned LSE
storage. Public-plan tests construct all 288 queries in `PREPLANNED_ONLY` mode.

QSA factories preserve the typed planning capacities while measuring full and
partial prefill plus decode. The public sparse-GQA path is tested for both BF16
and FP8 KV storage and all three warp decisions, including page offsets beyond
2 GiB. DSA query schema 2 defines `max_k_rows` as the planned K capacity for
both contiguous and paged sources. A paged capacity equals page-table width
times page size. Merge-race fixtures execute public plans with full and partial
live counts in the same fixed-capacity pool; mismatched metadata fails before
measurement. The model-policy inspector uses the same K-capacity and 512-wide
compressed-MLA output contracts.

**Qualified for the measured API cases:** the five qualification providers
passed 87 cases on RTX PRO 6000 Max-Q and GB10, with independent references,
poisoned replay, stable output addresses, and zero replay allocation growth.
MXFP8 quantization checks both linear and native-MMA byte orders bitwise.
MLA query projection records its actual implementation: CuTe for MXFP8 weights
and the existing Triton path for BF16 weights. This evidence does not imply
that every API shares one compiled callable across changing live counts.

**Qualified for the Trellis corpus on RTX PRO 6000 Max-Q and GB10:** 72 queries per device
cover 40 native-weight cases and 32 compact P24/P33 cases. Native cases use
the ten supported codebook/rate pairs; compact cases use MCG and SQG E4M3
with K-axis or N-axis rate records. Both FP16 and BF16 and two row capacities
are covered. All 656 launch candidates passed independent decoded-weight and Hadamard
references, frozen-resolution capture, three poisoned replays, and zero replay
allocation growth. Minimum cosine was 0.99999988 on both devices. Every query's
production heuristic was included in its candidate race. The public default
CuTe rotations were exercised throughout. Both devices passed 51 rotation and
public-API tests; 19 RTX memcheck cases passed with zero errors.

**Implemented:** dense Trellis uses the shared CuTe H128 butterfly, with FP16
rounding before the input transform and before the output scale multiplication.
The launcher accepts live rows as a runtime scalar, uses caller-owned output,
and caches only device, width, and scale-presence specializations. The public
API retains explicit callback overrides. The qualification provider exercises
the built-in rotations and prepares weights and workspaces once per query;
it requires no external rotation extension. Query schema 2 distinguishes native
storage and four compact pair layouts. Compact pairs require a 256-channel rate
axis; N-axis pairs require a complete 256-column CTA tile.

**Implemented:** BF16 GEMV, native MXFP8 BMM, and MXFP8 MLA query projection
accept live row counts through runtime arguments. Their compiled callable
identities contain model geometry and fixed capacity, without live row counts.
The 31 GEMV tests and 92 BMM/query-projection tests pass on RTX PRO 6000 Max-Q
and GB10, including multiple live counts under frozen kernel resolution and
CUDA graph replay. BMM and query projection use 16-row tiles and a runtime
row-grid dimension. Their four-case RTX memcheck passed with zero errors.
GEMV config schema 2 records the selected row grouping. Its production
compiler and offline contract share the device/model-geometry rule; live row
counts do not change that configuration. The nine-query GB10 A/B/B/A run
measured a geometric-mean candidate/baseline latency ratio of 0.877 and a
worst ratio of 1.0 using 260 cold replay samples per query and arm. This is
evidence for that corpus and protocol, not performance coverage of every
supported GEMV geometry. Correctness and compilation reuse alone do not
establish performance parity.

The blockscaled precision shape factory qualifies K=384, N=256, and three
sampled rows for both NVFP4 and MXFP8 on RTX PRO 6000 Max-Q and GB10. Each
query races all 17 production candidates with independent references, zero
replay allocation growth, and independent confirmation for selected A16
routes. This evidence covers the declared geometry; it does not qualify
unmeasured precision decisions.

Public APIs outside direct registration are PCIe/RoCE collectives.
Distributed collectives require topology and
transport identity as well as device identity; a single-device profile cannot
represent their qualification. PCIe crossover autotuning uses a separate
distributed implementation in `b12x/comm/pcie/pcie_dma.py`.

The intended ownership boundary is component-specific semantics, legal
decisions, production preparation, and correctness oracles, with shared search,
measurement scheduling, evidence, selection, qualification, and profile
serialization. A component that uses an existing backend must declare that
dependency and qualify its own input and launch contract. A single backend
still requires production-path qualification.

An input has two independent classifications. Its role is family membership,
search coordinate, caller constraint, derived quantity, or environment. Its
binding time is model preparation, planning, launch, or device discovery.
Numeric type alone does not determine either classification. A planned token
capacity can be a policy coordinate; a device scalar holding live tokens
cannot be a compilation or policy-cache key.

For a fixed semantic family and caller constraints, let `x` denote the
independent shape coordinates, `p` the independent kernel decisions, and `s`
a measurement scenario. Component-owned lowering maps `(x,p,device)` to a
validated launch configuration. Derived workspace sizes, routing products,
and capacity tables are outputs of that lowering, rather than independent
coordinates or knobs. Execution equivalence and measurement identity are
separate contracts: equivalent launches do not imply identical input tensors,
oracles, timing cohorts, or physical hardware.

For ordinary latency selection, the objective is

\[
 p^*(x)=\arg\min_{p\in A(x)}
 \exp\left(\frac{1}{|S|}\sum_{s\in S}\log T(x,p,s)\right),
\]

where `A(x)` contains candidates that satisfy legality, correctness, and all
component-specific selection constraints. Automatic activation precision adds
independent confirmation and parity constraints to this admissible set.
`generation/selection.py` performs scenario reduction for ordinary sweeps,
MoE, and blockscaled precision. Correct, timed candidates remain visible when
a precision constraint makes them ineligible. A correctness failure and an
ineligible precision decision have different qualification consequences.

Conditional decisions declare when each knob applies. Blockscaled precision
declares tile and split-K knobs only for A16. Its sampled live-row decisions
lower together into a validated, exact row-count selector. Unmeasured rows
retain the quantized path; sampled live rows never become policy-query fields.

MoE exposes hidden width, intermediate width, planned tokens, expert count,
and top-k as independent coordinates. Fixing expert count and top-k produces
the three-dimensional width × width × capacity slice. Routed rows are derived
from tokens × top-k. Source format, quantization semantics, and activation
identify semantic families. Allocation groups organize fixture reuse without
changing the mathematical family or its runtime identity.

## Offline search

**Implemented, research-only:** `generation/search.py` supplies exhaustive,
farthest-point, adaptive boundary, and Bayesian samplers. They share query,
measurement, budget, and accounting contracts. The adapters in
`generation/engine.py` preserve complete production candidate races and all
scenarios at each requested query. The MoE adapter runs an independent oracle
at each selected query and uses the component's aggregate precision-confirmation
rule. Fixed-backend tasks qualify their production paths without manufacturing
a kernel-choice search problem.

The adaptive sampler refines disagreement in neighboring decisions or
candidate eligibility while retaining mandatory exploration. Agreement at
sampled corners cannot establish that the interior has one winner. Tile and
wave quantization can introduce reversals, disconnected winner regions, and
legality boundaries.

Axis-aligned brackets keep the other coordinates fixed and propose an
unmeasured point near the logarithmic midpoint of a decision change. Ordered
knob audits report constant, increasing, decreasing, and reversing segments.
Inapplicable knobs break a segment. These observations test progression
hypotheses without imposing monotonicity on unsampled points.

The Bayesian sampler fits local Matérn-3/2 models to relative log latency,
using at most 32 nearby observations per prediction. Its acquisition score
prioritizes uncertainty about competing decisions. These posterior intervals
are sampling heuristics; they are not certified regret bounds. Neither a GP
nor a spatial-neighbor model participates in runtime policy lookup.

**Implemented, research-only:** `generation/regions.py` fits bounded decision
trees by aggregate log-latency regret. A region may share one near-optimal
decision even when individual sample winners differ. Candidate eligibility
must hold at every training point assigned to a leaf. Prefix sums bound the
cost of evaluating split proposals. Family guards and axis bounds restrict
coverage; aligned domains use exact branches to preserve unsupported gaps.
The fitter reports training regret and enforces construction budgets. Training
fit does not establish legality or regret at unmeasured queries.

`benchmarks/benchmark_tuning_search.py` compares all four samplers through the
same production adapters. Each strategy runs in a separate process with a
fresh compile cache and observation store, and repetitions reverse strategy
order. It retains raw measurements and includes initialization, sampling,
region fitting, and holdout measurement in generation time. Its deterministic
within-family corpus holdouts are research evidence; component-specific
geometry and decision-boundary holdouts remain necessary for profile emission.

**Implemented:** `generation/qualification.py` evaluates independent holdouts.
Both aggregate and per-partition geometric-mean regret must be at most 0.5%,
and observed worst regret at most 2%. Training-query overlap, missing
partitions, relabeled initial measurements, failed production candidates,
and invalid policy selections fail qualification. This is an empirical
qualification statement about its declared holdouts, not a uniform theorem
over unmeasured points.

Among qualified strategies, selection uses total generation time. A simpler
deterministic strategy is preferred when its cost is within 5% of the fastest.
Budget exhaustion is recorded separately from coverage and qualification;
it cannot emit a completed profile.

## Checked profile generation

**Implemented:** `generation/program.py` connects sampling, region fitting,
production eligibility, and independent holdouts to
`scripts/generate_gpu_profile.py --search-plan coverage.json`. The JSON object
contains `schema_version: 1` and a `components` map keyed by registered
component ID. Each component plan declares:

- `domains`: fixed family/constraint values and ordered integer axis intervals,
  each with `name`, `minimum`, `maximum`, and `alignment`.
- `training`: the pool of complete tuning queries eligible for sampling.
- `holdouts`: independent queries reserved before fitting.
- `strategy`: `exhaustive`, `space_filling`, `adaptive`, or `bayesian`.
- `query_budget` and optional `seconds`: sampling limits.
- `legality_query_budget`: the maximum covered lattice size to validate on CPU.

The CLI estimates the training, holdout, and legality work before execution.
Each selected query races all production candidates across its complete
scenario set. Every covered lattice point must pass lowering, production
eligibility, and runtime-serialization equivalence checks. Alignment gaps remain
uncovered. This exhaustive CPU check is bounded by the declared budget; it
does not enumerate or time the full GPU search lattice.

Holdout categories are computed from coordinates and the frozen fitted tree.
Geometry and capacity holdouts use axis projections absent from training.
Interior holdouts lie strictly within every varying axis. Decision-boundary
holdouts are adjacent to a change in the fitted decision. Applicable categories
must pass both globally and within each family. Single-decision families have
no fitted boundary category. Missing categories or failed regret gates retain
a report in the checkpoint directory and prevent profile emission. The fitter
never consumes the holdout timings.

Fixed-path probes remain part of composite measurement programs. Blockscaled
precision uses its production sweep: a sampled live-row A16 decision requires
independent per-query precision confirmation, so it cannot be extended to
unmeasured rows by the spatial emitter.

`--measurement-cohort` identifies an explicitly fresh measurement run. Search
and holdout checkpoints have separate namespaces and share one observation
database. Resume retains the measured cohort and does not relabel observations
as independent confirmation. Changing a search strategy, budget, or region
does not invalidate identical source/protocol/query/candidate observations.
Training and holdout roles remain separate, and changing the explicit cohort
requires fresh measurements.

**Qualified on a bounded NVFP4 region:** rows 128–512 and columns 2048–8192,
both aligned to 128, use four corner training queries and 12 independent
holdouts. With 260 replay samples per candidate, GB10 achieved 0.122%
geometric-mean regret and 1.233% worst regret; RTX PRO 6000 Max-Q achieved
0.051% and 0.294%. All 196 lattice points passed CPU lowering. Each emitted
profile selected `PREPLANNED` through the public API at all 12 holdouts;
poisoned graph replays matched packed values and scales exactly with stable
addresses and zero allocation growth. The 25-sample RTX run failed at 1.87%
geometric-mean regret and 24.51% worst regret and emitted no profile. The
default remains 25 samples; these results qualify only the declared region
and 260-sample protocol.

**Bounded GQA region evidence:** BF16 queries and KV, 24 query heads, four KV
heads, head dimension 128, page size 128, batch capacity 1–4, and cache
capacity 1024–4096 were tested with full-window, window-511, and zero partial
buffer families. RTX passed 36 holdouts with zero observed regret. GB10 failed
the 2% worst-regret gate with corner training (3.31%), a denser capacity grid
(2.39%), and 54 adaptive training queries (4.49%). All production candidates
passed correctness. The adaptive training pool used even page counts while
holdouts used odd counts; it could not discover rounding-dependent changes
absent from its pool. No failed GB10 fit emitted a profile. A sampler's
coverage of its finite pool is distinct from coverage of schedule residues
and boundaries in the underlying integer shape space.

A 264-query exhaustive training grid spanning both page-count parities also
failed on GB10: 0.413% aggregate regret and 12.58% worst regret across 36
holdouts. All 300 production cases passed correctness. Three independent
repeats at the worst query, batch four and nine pages, measured the selected
decision's regret as 7.40%, 0%, and 0%. Adjacent eight- and ten-page cases
were repeated in alternating order. These observations establish measurement
instability; they do not isolate a kernel discontinuity or qualify that fit.
`benchmarks/benchmark_tuning_recheck.py` retains complete repeated races, raw
observations, clock snapshots, and separate cohorts without modifying the
failed qualification report.

A 260-sample run on the separate graviton GB10 passed the same full-page grid:
264 training queries, 36 independent holdouts, and 300 legality checks produced
12 leaves with 0.00743% geometric-mean regret and 0.1563% worst regret. Every
required global and per-family partition passed the unchanged 0.5%/2% gates.
The artifact qualifies that domain on the measured physical device; it does
not invalidate the chroniton failures or isolate sampling count as their cause.
A confirmation using the identical frozen source and 260-sample protocol on
chroniton failed: 0.1245% aggregate regret and 4.581% worst regret. The failing
holdout had batch capacity two and cache capacity 1152 in the full-window
family. All candidates passed correctness and all required partitions were
present. Its 13-leaf fit remains research-only and emitted no profile.

## Runtime representation and measurement evidence

**Implemented:** runtime profiles use a shared decision DAG. Exact branches
with four or more alternatives have immutable lookup indices; integer range
branches use binary search. Smaller branches retain direct comparisons.
Profile decoding receives the typed query and device before the common
configuration validator runs. Overrides retain their precedence, and matching
invalid profile data fails closed.

**Implemented:** GQA has a pure decision materializer for CTA budget and split
selection. Its workspace and chunk schedule depend on the supplied query and
device. Query schema 3 includes caller workspace constraints in profile and
cache identity. Config schema 3 stores independent CTA-budget and split
decisions. Integer interval arithmetic and quotient-change boundaries build
the exact compressed schedule without a per-page traversal during resolution.
All 12,960 original covered queries resolve to identical full configurations.
An alternating-process CPU benchmark measured 30.20 µs uncached resolution
versus 77.63 µs for the stored-schedule implementation; cached resolution was
7.36 versus 7.27 µs. This measures planning overhead, not GPU kernel latency.

Two logical device assets occupy 82,957 compressed bytes: GB10 and RTX PRO
6000 Blackwell. Exact Workstation, Server, and Max-Q target aliases select
the RTX asset; its qualification metadata identifies Max-Q as the measured
representative. The three stored-schedule assets used 179,326 bytes. This size
comparison includes alias consolidation and constraint coverage; it is not
solely a codec comparison.

GQA terminal guards cover an unspecified workspace limit and any explicit
limit large enough to preserve the identical measured launch. An explicitly
measured limit has an exact guard and precedes broader coverage. The emitted
tree defines this coverage using ordinary exact/range nodes. An audit checked
77,760 combinations of unspecified, exact-required, and doubled workspace
limits without a configuration change. Smaller binding limits remain separate
profile coordinates. Public graph preparation retains KV layout and policy
provenance; RTX and GB10 checks exercise profile reuse, windowing, zero partial
capacity, live-length changes, and page offsets beyond 2 GiB with stable
allocation under frozen kernel resolution.

**Qualified on the declared KDA corpus:** the KDA provider races value split,
key split, pipeline stages, and window size through public production plans.
Each of GB10 and RTX PRO 6000 Max-Q passed 9,216 candidate measurements across
96 scenarios. Checks cover independent FP32 token and state oracles, untouched
state slots, poisoned graph replay, and zero replay allocation growth. Profiles
cover the measured queries exactly. The Max-Q hardware is the physical RTX
representative; these results do not constitute measurements of other SKUs.

Candidate races and fixed-backend probes share one observation recorder and
transactional SQLite store. Observations are compressed once.
Their identities include source contents and revision, toolchain, physical
GPU UUID, input and scenario data, ordered candidate cohort, timing settings,
and oracle contract. Case checkpoints hold content-addressed references.
Independent confirmation has a separate cohort. MoE stage names do not enter
the measurement identity, while the requested oracle and ordered candidate
cohort do. Fresh and reused work are reported separately. See
[GPU component profiles](gpu-profiles.md) for the generator and integration
interfaces.

**Implemented:** `generation/replay.py` owns candidate preparation and balanced
interleaving; `generation/timing.py` owns the shared sampler, bounded repetition
count, and grouped-median estimator. Grouped providers
retain unrounded samples and their grouping in `metrics.timing`. State resets
and cold-L2 flushes stay outside each timed replay. Precision selection retains
its independent paired confirmation and correctness requirements. CUDA events
are the default clock; explicit global-timer sampling has a separate observation
identity. A stored median
alone is insufficient evidence for diagnosing timing variation or certifying
small regret differences.

Execution-equivalent GQA decisions retain all independent CTA-budget and
split-choice aliases while sharing one measured execution. Derived schedules
do not become candidate identities for shape-space learning. Blockscaled
precision similarly uses precision, tile, and split-K decisions independently
of the sampled live-row count.

**Research-only:** `CapturedGraphTimer` composes existing production graphs as
CUDA child nodes with device-side timing events. The composition preserves
reset/flush ordering and graph ownership, but does not by itself resolve
short-kernel estimator variance. `balanced_race_samples_us` interleaves
candidates while balancing their positions.
`benchmarks/benchmark_nvfp4_race_timing.py` compares sequential and interleaved
races, reverses method and sample-count ordering, and retains every duration,
clock snapshot, oracle check, and allocation check. Neither experimental
protocol replaces the default provider timer without qualification.

MoE NVFP4 timing fixtures use deterministic signed FP4 weights with fixed scale
magnitudes. The independent oracle reads the declared FC1 weight order and
accounts for the production normalization registry. Reference outputs must
agree before and after in-place weight normalization. The M64 swapped-FC1
regression covers intermediate widths 64, 96, and 128; planned token counts
1, 7, 8, 9, 64, and 128; and balanced/hot routes. All 36 cases passed on RTX
PRO 6000 Max-Q and GB10 with independent-oracle cosine above 0.99998 and zero
replay allocation growth.
