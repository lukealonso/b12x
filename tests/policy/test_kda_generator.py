import pytest

from b12x.policy.generation.providers.kda import KdaPrefillGenerator, kda_cases, kda_candidates
from b12x.sequence.kda_prefill._policy import KDA_PREFILL_POLICY, KdaPrefillConfig, KdaPrefillQuery, recurrence_shared_bytes


def test_kda_candidate_legality_matches_observed_shared_memory_limit():
    case = kda_cases()[0]
    query = KdaPrefillQuery(**case.query.to_dict())
    invalid = KdaPrefillConfig(v_split=128, k_split=2, stages=4, window_tiles=3)
    assert recurrence_shared_bytes(invalid) == 109760
    with pytest.raises(ValueError, match='109760 shared-memory bytes'):
        KDA_PREFILL_POLICY.validate_config(query, invalid, None)
    candidates = kda_candidates(query, None)
    assert len(candidates) == 32
    assert len({item.candidate_id for item in candidates}) == len(candidates)
    for candidate in candidates:
        KDA_PREFILL_POLICY.validate_config(query, KdaPrefillConfig.from_profile(candidate.config), None)
    KDA_PREFILL_POLICY.validate_config(query, KdaPrefillConfig(window_tiles=1_000_000), None)
    with pytest.raises(TypeError, match='integer'):
        KDA_PREFILL_POLICY.validate_config(query, KdaPrefillConfig(k_split=True), None)


def test_kda_corpus_covers_semantic_families_and_multiple_live_counts():
    generator = KdaPrefillGenerator()
    cases = kda_cases()
    assert len(cases) == 96
    assert generator._candidate_contract_version == 2
    assert {(c.query['qk_l2norm'], c.query['checkpoint_export']) for c in cases} == {
        (False, False), (False, True), (True, False), (True, True),
    }
    groups = {}
    for case in cases:
        groups.setdefault(case.group_id, []).append(case)
    for group in groups.values():
        assert len(group) == 2
        assert group[0].query == group[1].query
        assert group[0].metadata['live_tokens'] != group[1].metadata['live_tokens']
        assert 'live_tokens' not in group[0].query


def test_embedded_kda_winners_build_public_plans_without_cuda_discovery(monkeypatch):
    from b12x.policy import DetectedDevice, EMBEDDED_REGISTRY, PolicyContext, PolicyMode, PolicySource
    from b12x.sequence import kda_prefill

    query = KdaPrefillQuery(**kda_cases()[0].query.to_dict())
    for profile in EMBEDDED_REGISTRY.list_profiles():
        identity = profile.targets[0]
        monkeypatch.setattr('b12x.policy.context.detect_device', lambda device, identity=identity: DetectedDevice(ordinal=0, identity=identity))
        policy = PolicyContext.for_identity(identity, mode=PolicyMode.PREPLANNED_ONLY)
        plan = kda_prefill.plan(kda_prefill.Caps(device='cuda:0', heads=query.heads,
            max_tokens=query.max_tokens, max_seqs=query.max_seqs, max_state_slots=4,
            qk_l2norm=query.qk_l2norm, checkpoint_export=query.checkpoint_export), policy=policy)
        assert plan.policy_resolution.source is PolicySource.PREPLANNED
        assert plan.scratch_specs()[0].shape[0] > 0
