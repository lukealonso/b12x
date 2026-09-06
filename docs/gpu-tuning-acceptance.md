# GPU tuning acceptance

The tuning system is accepted when production specialization families are
accounted for, supported launch paths obey the serving contract, and published
policy regions have reproducible qualification evidence. These are separate
claims with separate checks. A source inventory alone does not establish runtime
coverage; passing a finite corpus does not establish performance over an
unbounded shape space.

## Scope

The component catalog owns policy and generator registration. Each registered
API must define semantic families, planned shape coordinates, caller constraints,
kernel decisions, and launch-time inputs. Internal compute and supporting kernels
must be associated with those contracts or an explicit separate owner.
Distributed collectives retain their topology-dependent qualification boundary.

The implementation boundary includes the typed contracts, compact runtime
decision DAG, offline observation store, provider preparation and timing,
candidate search, independent qualification, and specialization census. Additional
search algorithms and kernel optimizations require separate objectives once the
acceptance conditions below are met.

## Acceptance gates

| Gate | Required evidence |
| --- | --- |
| Inventory | Every GPU entry point, compile factory, and cache family has an owner and declared specialization dimensions. Indirect launch sites have explicit dispositions. Source, observation, and executed-launch inventories identify their coverage and omissions. |
| Serving invariants | Compile and cache identities exclude live request quantities. Each cache family has a production-path test that warms planned capacity, freezes resolution, varies live counts, and verifies correctness, graph replay, stable storage, and zero replay allocation. Relevant boundary and high-pool-offset cases pass. |
| Measurement | Providers prepare eligible candidates before racing them. State resets and cache preparation have consistent placement outside the timed interval. Balanced ordering, raw samples, clock identity, source identity, and physical device identity survive worker transport and resume. Precision selectors retain their independent correctness and confirmation requirements. |
| Qualification | Published regions pass exhaustive declared-lattice legality checks and independent production-candidate holdouts. The ordinary spatial gates remain 0.5% geometric-mean regret and 2% worst regret, including required family and boundary partitions. Held-out measurements do not influence the fitted policy. |
| Regression | The frozen source passes catalog/profile consistency, public-plan integration, affected GPU correctness, frozen-resolution, and relevant sanitizer checks. Changed hot paths have production graph timing against the preserved baseline, with repeated measurements for apparent regressions. Failures and missing checks have explicit dispositions. |
| Delivery | Reviewed commits contain the implementation, compact qualified assets, tests, and present-state documentation. Raw evidence remains outside the package and is bound by source and artifact hashes. Embedded profiles describe the source and protocol actually qualified. |

The census enumerates specialization families and their dimensions. The runtime
qualification manifest enumerates the concrete device, geometry, recipe, capacity,
and live-count cases exercised. Neither report may present one kind of coverage
as the other.

## Qualification decisions

GQA's bounded decode domain requires the unchanged 300-query qualification on
the target GB10 devices before accepting a profile for that domain. The explicit
global-timer protocol is research-only until its generation path and independent
holdouts pass. Its timestamp kernels perturb scheduling, so its durations remain
distinct from CUDA-event durations in stored evidence.

Block-FP8 region qualification must sample the observed capacity transitions.
Queries used to diagnose a failed fit become training data only in a separately
identified search contract with fresh holdouts. Previously inspected holdouts
cannot certify that refit as independent evidence.

An unqualified region is excluded from profile emission. AUTO policy resolution
continues to use its existing heuristic for uncovered queries. Such a disposition
permits an explicitly limited profile; it does not close the region's
qualification work or justify claiming comprehensive spatial coverage.

## Execution and review order

1. Finish the frozen-source GPU regressions, sanitizer run, and GQA timing
   qualification. Record each result against its exact source archive and command.
2. Close the specialization ownership matrix and remaining live-count cache
   families. Add qualification cases for uncovered families and rerun checks
   affected by their fixes.
3. Freeze the provider measurement protocol. Resolve GQA qualification and
   refine block-FP8 transition sampling with independent holdouts. Broaden other
   providers through declared bounded contracts rather than implicit coverage.
4. Measure changed runtime paths against their preserved baselines on RTX PRO
   6000 Max-Q and GB10. Resolve correctness failures before interpreting timings;
   retain pre-existing failures as explicit validation limitations.
5. Freeze the accepted source, qualify the affected profile assets, and produce
   one acceptance report with gate status and evidence references. Any required
   source change invalidates the affected evidence and requires its checks again.
6. Review the changes as contracts/runtime representation, serving/cache fixes,
   generator/census/qualification, and measured assets/documentation. Integrate
   only within the user's authorized scope.

Required implementation work is complete only when every acceptance gate passes.
Any excluded region or unsupported contract remains visible in the acceptance
report, with the exact missing qualification condition.
