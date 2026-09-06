from contextlib import contextmanager
from dataclasses import replace
import json
import sqlite3

import pytest

from b12x.policy.catalog import list_profiled_components
from b12x.policy.generation.contracts import GenerationContext, GenerationSettings
from b12x.policy.generation.engine import measurement_program
from b12x.policy.generation.program import ProfileSearchPlan, QualifiedSearchGenerator
from b12x.policy.generation.progress import NullProgressReporter
from b12x.policy.generation.runner import generate_profile_artifact
from b12x.policy.generation.store import CheckpointStore
from b12x.policy.generation.sweep import SweepCandidate, SweepMeasurement
from b12x.policy.serialization import profile_from_dict
from b12x.policy.types import DeviceIdentity, FrozenMapping


def _query(rows, columns):
    return {"dtype": "bfloat16", "rows": rows, "columns": columns}


def _plan():
    return dict(domains=[{"fixed": {"dtype": "bfloat16"}, "axes": [
        {"name": name, "minimum": 128, "maximum": 512, "alignment": 128}
        for name in ("rows", "columns")]}],
        training=[_query(rows, columns) for rows in (128, 512) for columns in (128, 512)],
        holdouts=[_query(256, 256), _query(384, 384)], strategy="exhaustive",
        query_budget=4, legality_query_budget=16)


def _generator(*, plan=None, calls=None, fail_holdouts=False, boundary=False):
    registration = next(item for item in list_profiled_components() if item.component_id == "quantization.nvfp4")
    generator = registration.create_generator()
    calls = [] if calls is None else calls

    @contextmanager
    def factory(group, cases, context):
        candidates = tuple(SweepCandidate.create({"backend": "cutedsl", "liveness_strategy": name})
                           for name in ("retain", "packed"))

        class Session:
            def candidates(self, case):
                return candidates

            def measure(self, case, candidates):
                calls.append((context.measurement_cohort, case.query))
                winner = "packed" if boundary and case.query["rows"] > 256 else "retain"
                if fail_holdouts and ":holdout:" in context.measurement_cohort:
                    winner = "packed"
                return tuple(SweepMeasurement(candidate=candidate, correct=True,
                             latency_us=1. if candidate.config["liveness_strategy"] == winner else 2.)
                             for candidate in candidates)

        yield Session()

    generator._benchmark_factory = factory
    generator.measurement_program = measurement_program(generator, generator.problem)
    return QualifiedSearchGenerator(generator, ProfileSearchPlan.from_dict(plan or _plan()))


def _context(tmp_path):
    return GenerationContext(device=DeviceIdentity(vendor="nvidia", product_name="Synthetic GPU",
        compute_capability=(12, 0), sm_count=188), device_ordinal=0, work_dir=tmp_path,
        source_revision="test-source", settings=GenerationSettings(),
        provenance=FrozenMapping({"source_sha256": "test-hash", "physical_device": "test-GPU",
                                  "toolchain": {"compiler": "test"}}))


@pytest.mark.parametrize("boundary", (False, True))
def test_normal_artifact_runner_emits_only_qualified_regions_and_resumes(tmp_path, boundary):
    calls = []
    generator = _generator(calls=calls, boundary=boundary)
    context = _context(tmp_path)
    artifacts = [generate_profile_artifact(profile_id="test", generators=(generator,), context=context,
                                          progress=NullProgressReporter()) for _ in range(2)]
    assert len(calls) == 6
    assert artifacts[0]["profile"] == artifacts[1]["profile"]
    evidence = artifacts[1]["evidence"]["components"][generator.component_id]
    assert evidence["qualification"]["status"] == "qualified"
    assert evidence["qualification"]["cases"] == 2
    assert evidence["gpu_measurement_cases"] == 6
    assert evidence["sampling"]["reused_queries"] == 4
    assert {"geometry", "capacity", "interior"} <= set(evidence["required_partitions"])
    assert ("decision_boundary" in evidence["required_partitions"]) is boundary
    planner = profile_from_dict(artifacts[0]["profile"]).components[0].planner
    for rows in (128, 256, 384, 512):
        for columns in (128, 256, 384, 512):
            leaf = planner.lookup(_query(rows, columns))
            assert leaf.config["liveness_strategy"] == ("packed" if boundary and rows > 256 else "retain")
    assert planner.lookup(_query(129, 256)) is None
    assert planner.lookup(_query(640, 256)) is None
    databases = list(tmp_path.rglob("observations.sqlite3"))
    assert databases == [tmp_path / "observations.sqlite3"]
    with sqlite3.connect(databases[0]) as connection:
        assert connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 6


def test_failed_holdouts_leave_reviewable_evidence_and_emit_no_profile(tmp_path):
    generator = _generator(fail_holdouts=True)
    with pytest.raises(RuntimeError, match="failed independent policy qualification"):
        generate_profile_artifact(profile_id="test", generators=(generator,), context=_context(tmp_path),
                                  progress=NullProgressReporter())
    report, = (tmp_path / "checkpoints" / generator.component_id).glob("search-report-*.json")
    evidence = json.loads(report.read_text())
    assert evidence["qualification"]["status"] == "research-only"
    assert evidence["qualification"]["observed_worst_regret"] == 1.
    assert evidence["holdouts"][0]["candidate_ids"]


def test_missing_holdout_categories_cannot_be_relabeled_into_qualification(tmp_path):
    plan = _plan()
    plan["holdouts"] = [_query(128, 256), _query(512, 384)]
    generator = _generator(plan=plan)
    with pytest.raises(RuntimeError, match="failed independent policy qualification"):
        generator.generate(_context(tmp_path), progress=NullProgressReporter(),
                           checkpoints=CheckpointStore(tmp_path / "checkpoints"))
    report, = (tmp_path / "checkpoints" / generator.component_id).glob("search-report-*.json")
    assert {"capacity", "interior"} <= set(json.loads(report.read_text())["qualification"]["missing_partitions"])


def test_legality_checks_interior_queries_before_holdout_measurement(tmp_path):
    calls = []
    generator = _generator(calls=calls)
    policy = generator.problem.policy
    def validate(query, config, device):
        policy.validate_config(query, config, device)
        if query.rows == 384:
            raise ValueError("interior launch is illegal")
    generator.problem = replace(generator.problem, policy=replace(policy, validate_config=validate))
    with pytest.raises(ValueError, match="interior launch is illegal"):
        generator.generate(_context(tmp_path), progress=NullProgressReporter(),
                           checkpoints=CheckpointStore(tmp_path / "checkpoints"))
    assert len(calls) == 4
    assert all(":search:" in cohort for cohort, _ in calls)


@pytest.mark.parametrize("invalid", ("overlap", "budget", "alignment"))
def test_invalid_coverage_fails_before_production_preparation(invalid):
    plan = _plan()
    if invalid == "overlap":
        plan["holdouts"].append(plan["training"][0])
    elif invalid == "budget":
        plan["legality_query_budget"] = 15
    else:
        plan["domains"][0]["axes"][0]["alignment"] = 64
        plan["legality_query_budget"] = 100
    with pytest.raises(ValueError):
        _generator(plan=plan)


def test_moe_fixture_factory_retains_complete_routes_and_exact_geometry():
    from b12x.policy.generation.providers.moe import MoeDecodeGenerator
    query = dict(quant_mode="nvfp4", source_format="modelopt_nvfp4", activation="silu",
                 num_experts=256, hidden_size=2048, intermediate_size=96,
                 top_k=8, num_tokens=9, routed_rows=72)
    cases = tuple(MoeDecodeGenerator.cases_for_tuning_queries((query,)))
    assert {case.route_pattern for case in cases} == {"balanced", "hot", "zipf", "disjoint"}
    assert all(case.query() == query for case in cases)
    assert len({case.geometry.key for case in cases}) == 1
    assert cases == tuple(MoeDecodeGenerator.cases_for_tuning_queries((query,)))
    with pytest.raises(ValueError, match="inconsistent derived"):
        tuple(MoeDecodeGenerator.cases_for_tuning_queries(({**query, "routed_rows": 71},)))


def test_search_plan_is_loaded_by_normal_cli_registry():
    from b12x.tools.generate_gpu_profile import _load_search_registry, _parser
    args = _parser().parse_args(["--search-plan", "coverage.json", "--measurement-cohort", "confirmation"])
    assert str(args.search_plan) == "coverage.json"
    assert args.measurement_cohort == "confirmation"
    registry = _load_search_registry({"quantization.nvfp4": _plan()})
    assert isinstance(registry.get("quantization.nvfp4"), QualifiedSearchGenerator)
    from b12x.policy.catalog import list_generation_components
    assert registry.component_ids() == tuple(item.component_id for item in list_generation_components())


def test_search_strategy_changes_reuse_identical_races_but_separate_holdout_roles(tmp_path):
    calls = []
    context = _context(tmp_path)
    def generate(plan):
        return generate_profile_artifact(profile_id="test", generators=(_generator(plan=plan, calls=calls),),
                                         context=context, progress=NullProgressReporter())
    first = generate(_plan())
    alternate = _plan()
    alternate["strategy"] = "space_filling"
    second = generate(alternate)
    assert len(calls) == 6
    first_evidence = first["evidence"]["components"]["quantization.nvfp4"]
    second_evidence = second["evidence"]["components"]["quantization.nvfp4"]
    assert first_evidence["contract_id"] != second_evidence["contract_id"]
    assert second_evidence["sampling"]["reused_queries"] == 4
    switched = _plan()
    switched["training"] = [_query(128, 128), _query(256, 256), _query(384, 384), _query(512, 512)]
    switched["holdouts"] = [_query(128, 512), _query(512, 128)]
    with pytest.raises(RuntimeError, match="failed independent policy qualification"):
        generate(switched)
    assert len(calls) == 10
    context = replace(context, measurement_cohort="independent-confirmation")
    generate(_plan())
    assert len(calls) == 16
