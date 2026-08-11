"""Tests for scripts/evaluation_protocol.py: seed-pool classification and
disjointness, the scope-exclusive seed guard (and the older FINAL_BLIND-only
guard kept for backward compatibility), record-pairing input validation,
the seed-block paired randomization test (primary) against hand-verified
small-N cases and Monte Carlo determinism, the seed-block bootstrap
(primary), exact McNemar (secondary, explicitly labeled and documented as
clustered), and the full paired_evaluation_summary report shape - including
that no promotion/GO/KILL boolean appears anywhere in it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "evaluation_protocol.py"

_spec = importlib.util.spec_from_file_location("evaluation_protocol", MODULE_PATH)
ep = importlib.util.module_from_spec(_spec)
sys.modules["evaluation_protocol"] = ep
_spec.loader.exec_module(ep)


# ── seed pools ──────────────────────────────────────────────────────────


def test_dev_promotion_final_blind_pools_are_pairwise_disjoint():
    assert ep.DEV_SEEDS.isdisjoint(ep.PROMOTION_SEEDS)
    assert ep.DEV_SEEDS.isdisjoint(ep.FINAL_BLIND_SEEDS)
    assert ep.PROMOTION_SEEDS.isdisjoint(ep.FINAL_BLIND_SEEDS)


@pytest.mark.parametrize(
    "seed",
    [
        42, 0, 20000, 23, 101, 501, 502, 503,
        10000, 10009, 10015, 20015,
        30000, 30009, 31000, 31004, 32000, 32009,
        40000, 40015, 41000, 41015,
    ],
)
def test_previously_used_seeds_classified_as_dev(seed):
    assert ep.classify_seed(seed) == ep.SEED_CLASS_DEV
    assert seed in ep.DEV_SEEDS


def test_promotion_and_final_blind_seeds_classified_correctly():
    lo, hi = ep.PROMOTION_SEED_RANGE
    assert ep.classify_seed(lo) == ep.SEED_CLASS_PROMOTION
    assert ep.classify_seed(hi) == ep.SEED_CLASS_PROMOTION

    lo, hi = ep.FINAL_BLIND_SEED_RANGE
    assert ep.classify_seed(lo) == ep.SEED_CLASS_FINAL_BLIND
    assert ep.classify_seed(hi) == ep.SEED_CLASS_FINAL_BLIND


def test_unclassified_seed_outside_all_pools():
    unused = 999999
    assert unused not in ep.DEV_SEEDS
    assert unused not in ep.PROMOTION_SEEDS
    assert unused not in ep.FINAL_BLIND_SEEDS
    assert ep.classify_seed(unused) == ep.SEED_CLASS_UNCLASSIFIED


# ── require_seed_scope (new standard, scope-exclusive) ─────────────────


def test_require_seed_scope_dev_accepts_dev_seeds():
    ep.require_seed_scope([42, 10000, 30000], ep.SEED_CLASS_DEV, context="dev_run")


def test_require_seed_scope_dev_rejects_promotion_seed():
    promotion_lo, _ = ep.PROMOTION_SEED_RANGE
    with pytest.raises(RuntimeError, match="'dev'"):
        ep.require_seed_scope([42, promotion_lo], ep.SEED_CLASS_DEV, context="dev_run")


def test_require_seed_scope_dev_rejects_final_blind_seed():
    final_blind_lo, _ = ep.FINAL_BLIND_SEED_RANGE
    with pytest.raises(RuntimeError, match="'dev'"):
        ep.require_seed_scope([42, final_blind_lo], ep.SEED_CLASS_DEV, context="dev_run")


def test_require_seed_scope_promotion_accepts_only_promotion_pool():
    promotion_lo, promotion_hi = ep.PROMOTION_SEED_RANGE
    ep.require_seed_scope(list(range(promotion_lo, promotion_hi + 1)), ep.SEED_CLASS_PROMOTION, context="promotion_run")


def test_require_seed_scope_promotion_rejects_dev_seed():
    with pytest.raises(RuntimeError, match="'promotion'"):
        ep.require_seed_scope([42], ep.SEED_CLASS_PROMOTION, context="promotion_run")


def test_require_seed_scope_promotion_rejects_final_blind_seed():
    final_blind_lo, _ = ep.FINAL_BLIND_SEED_RANGE
    with pytest.raises(RuntimeError, match="'promotion'"):
        ep.require_seed_scope([final_blind_lo], ep.SEED_CLASS_PROMOTION, context="promotion_run")


def test_require_seed_scope_final_blind_rejects_dev_and_promotion():
    promotion_lo, _ = ep.PROMOTION_SEED_RANGE
    with pytest.raises(RuntimeError, match="'final_blind'"):
        ep.require_seed_scope([42], ep.SEED_CLASS_FINAL_BLIND, context="final_read")
    with pytest.raises(RuntimeError, match="'final_blind'"):
        ep.require_seed_scope([promotion_lo], ep.SEED_CLASS_FINAL_BLIND, context="final_read")


def test_require_seed_scope_rejects_unclassified_seed_in_every_scope():
    unused = 999999
    for scope in (ep.SEED_CLASS_DEV, ep.SEED_CLASS_PROMOTION, ep.SEED_CLASS_FINAL_BLIND):
        with pytest.raises(RuntimeError):
            ep.require_seed_scope([unused], scope, context="test")


def test_require_seed_scope_mixed_batch_fails_entirely_not_partially():
    """A DEV-scoped batch that accidentally includes one PROMOTION seed
    among otherwise-fine DEV seeds must be refused entirely, not silently
    filtered down to the safe subset."""
    promotion_lo, _ = ep.PROMOTION_SEED_RANGE
    with pytest.raises(RuntimeError):
        ep.require_seed_scope([42, 10000, 30000, promotion_lo], ep.SEED_CLASS_DEV, context="dev_run")


def test_require_seed_scope_unknown_scope_string_raises_value_error():
    with pytest.raises(ValueError):
        ep.require_seed_scope([42], "not_a_real_scope", context="test")


# ── require_non_final_blind (kept for backward compatibility) ─────────


def test_require_non_final_blind_passes_for_dev_and_promotion_seeds():
    ep.require_non_final_blind([42, 10000], context="test")
    ep.require_non_final_blind(list(range(*ep.PROMOTION_SEED_RANGE)), context="test")


def test_require_non_final_blind_raises_for_final_blind_seed():
    lo, _ = ep.FINAL_BLIND_SEED_RANGE
    with pytest.raises(RuntimeError, match="FINAL_BLIND"):
        ep.require_non_final_blind([42, lo], context="fake_eval_script")


# ── Wilson interval (descriptive only) ─────────────────────────────────


def test_wilson_interval_matches_known_value():
    lower, upper = ep.wilson_95_interval(50, 100)
    assert lower == pytest.approx(0.4038, abs=1e-3)
    assert upper == pytest.approx(0.5962, abs=1e-3)


# ── pair_records: input validation ─────────────────────────────────────


def _arm(seed_seat_pairs, *, win: bool, net_worth: float = 1000.0):
    return [{"seed": seed, "seat": seat, "win": win, "net_worth": net_worth} for seed, seat in seed_seat_pairs]


def test_pair_records_basic_success():
    baseline = _arm([(1, 0), (1, 1), (1, 2), (1, 3)], win=False, net_worth=500.0)
    candidate = _arm([(1, 0), (1, 1), (1, 2), (1, 3)], win=True, net_worth=1500.0)
    paired = ep.pair_records(baseline, candidate, expected_seats=4)
    assert len(paired) == 4
    assert all(r["baseline_win"] is False and r["candidate_win"] is True for r in paired)
    assert all(r["baseline_net_worth"] == 500.0 and r["candidate_net_worth"] == 1500.0 for r in paired)


def test_pair_records_duplicate_seed_seat_in_baseline_fails():
    baseline = _arm([(1, 0), (1, 0)], win=True)
    candidate = _arm([(1, 0)], win=True)
    with pytest.raises(RuntimeError, match="duplicate"):
        ep.pair_records(baseline, candidate, expected_seats=None)


def test_pair_records_duplicate_seed_seat_in_candidate_fails():
    baseline = _arm([(1, 0)], win=True)
    candidate = _arm([(1, 0), (1, 0)], win=True)
    with pytest.raises(RuntimeError, match="duplicate"):
        ep.pair_records(baseline, candidate, expected_seats=None)


def test_pair_records_missing_pairing_fails_when_candidate_shorter():
    baseline = _arm([(1, 0), (1, 1)], win=True)
    candidate = _arm([(1, 0)], win=True)
    with pytest.raises(RuntimeError, match="incomplete baseline/candidate pairing"):
        ep.pair_records(baseline, candidate, expected_seats=None)


def test_pair_records_missing_pairing_fails_when_baseline_shorter():
    baseline = _arm([(1, 0)], win=True)
    candidate = _arm([(1, 0), (1, 1)], win=True)
    with pytest.raises(RuntimeError, match="incomplete baseline/candidate pairing"):
        ep.pair_records(baseline, candidate, expected_seats=None)


def test_pair_records_expected_seats_mismatch_fails():
    baseline = _arm([(1, 0), (1, 1), (1, 2)], win=True)  # only 3 seats for seed 1
    candidate = _arm([(1, 0), (1, 1), (1, 2)], win=True)
    with pytest.raises(RuntimeError, match="expected exactly 4 seats"):
        ep.pair_records(baseline, candidate, expected_seats=4)


def test_pair_records_expected_seats_none_skips_seat_count_check():
    baseline = _arm([(1, 0), (1, 1), (1, 2)], win=True)
    candidate = _arm([(1, 0), (1, 1), (1, 2)], win=True)
    paired = ep.pair_records(baseline, candidate, expected_seats=None)
    assert len(paired) == 3


# ── seed-block paired randomization test (PRIMARY) ─────────────────────


def test_randomization_test_exact_n2_hand_verified():
    """Hand-verified: 2 seed blocks, both diff=+1.0 -> null sign patterns
    give means {+1, 0, 0, -1}; |stat|>=1 in exactly 2 of 4 -> p=0.5."""
    paired = [
        {"seed": 1, "baseline_win": False, "candidate_win": True},
        {"seed": 2, "baseline_win": False, "candidate_win": True},
    ]
    result = ep.seed_block_paired_randomization_test(paired)
    assert result["n_seed_blocks"] == 2
    assert result["method"] == "exact_enumeration"
    assert result["observed_mean_win_rate_diff"] == pytest.approx(1.0)
    assert result["p_value"] == pytest.approx(0.5)
    assert result["randomization_seed"] is None


def test_randomization_test_exact_n3_hand_verified():
    """Hand-verified: 3 seed blocks, all diff=+1.0 -> the 8 sign-pattern
    means are {+1 x1, +1/3 x3, -1/3 x3, -1 x1}; |stat|>=1 in exactly 2 of 8
    -> p=0.25."""
    paired = [{"seed": seed, "baseline_win": False, "candidate_win": True} for seed in (1, 2, 3)]
    result = ep.seed_block_paired_randomization_test(paired)
    assert result["n_seed_blocks"] == 3
    assert result["method"] == "exact_enumeration"
    assert result["p_value"] == pytest.approx(0.25)


def test_randomization_test_empty_returns_none_stats():
    result = ep.seed_block_paired_randomization_test([])
    assert result == {
        "n_seed_blocks": 0, "observed_mean_win_rate_diff": None,
        "method": None, "n_null_samples": 0,
        "randomization_seed": None, "p_value": None,
    }


def _many_blocks(n: int):
    return [
        {"seed": seed, "baseline_win": seed % 2 != 0, "candidate_win": seed % 2 == 0}
        for seed in range(1, n + 1)
    ]


def test_randomization_test_uses_monte_carlo_above_exact_threshold():
    paired = _many_blocks(25)
    result = ep.seed_block_paired_randomization_test(paired, n_resamples=500, randomization_seed=3)
    assert result["n_seed_blocks"] == 25
    assert result["method"] == "monte_carlo"
    assert result["randomization_seed"] == 3
    assert result["n_null_samples"] == 500


def test_randomization_test_monte_carlo_deterministic_given_same_seed():
    paired = _many_blocks(25)
    result_a = ep.seed_block_paired_randomization_test(paired, n_resamples=500, randomization_seed=3)
    result_b = ep.seed_block_paired_randomization_test(paired, n_resamples=500, randomization_seed=3)
    assert result_a == result_b


def test_randomization_test_monte_carlo_different_seed_can_differ():
    paired = _many_blocks(25)
    result_a = ep.seed_block_paired_randomization_test(paired, n_resamples=500, randomization_seed=3)
    result_b = ep.seed_block_paired_randomization_test(paired, n_resamples=500, randomization_seed=4)
    assert result_a != result_b


def test_randomization_test_exact_threshold_is_20_blocks():
    exactly_20 = ep.seed_block_paired_randomization_test(_many_blocks(20))
    assert exactly_20["method"] == "exact_enumeration"
    exactly_21 = ep.seed_block_paired_randomization_test(_many_blocks(21), n_resamples=100, randomization_seed=0)
    assert exactly_21["method"] == "monte_carlo"


# ── seed-block paired bootstrap (PRIMARY) ───────────────────────────────


def _fake_bootstrap_records():
    """Heterogeneous across seeds on purpose (win pattern flips by seed
    parity, net-worth diff scales with seed) so seed-block resampling
    actually produces a non-degenerate distribution."""
    records = []
    for seed in (1, 2, 3, 4):
        candidate_favored = seed % 2 == 0
        for seat in range(4):
            records.append(
                {
                    "seed": seed,
                    "seat": seat,
                    "baseline_win": (seat == 0) and not candidate_favored,
                    "candidate_win": (seat == 1) and candidate_favored,
                    "baseline_net_worth": 1000.0,
                    "candidate_net_worth": 1000.0 + seed * 100.0,
                }
            )
    return records


def test_bootstrap_is_deterministic_given_same_seed():
    records = _fake_bootstrap_records()
    result_a = ep.paired_seed_block_bootstrap(records, n_resamples=200, bootstrap_seed=7)
    result_b = ep.paired_seed_block_bootstrap(records, n_resamples=200, bootstrap_seed=7)
    assert result_a == result_b


def test_bootstrap_different_seed_can_differ():
    records = _fake_bootstrap_records()
    result_a = ep.paired_seed_block_bootstrap(records, n_resamples=200, bootstrap_seed=7)
    result_b = ep.paired_seed_block_bootstrap(records, n_resamples=200, bootstrap_seed=8)
    assert result_a != result_b


def test_bootstrap_point_estimates_match_direct_computation():
    records = _fake_bootstrap_records()
    result = ep.paired_seed_block_bootstrap(records, n_resamples=50, bootstrap_seed=0)
    assert result["win_rate_diff"]["point"] == pytest.approx(0.0)
    assert result["net_worth_diff"]["point"] == pytest.approx(250.0)
    assert result["n_seed_blocks"] == 4
    assert result["n_records"] == 16


def test_bootstrap_resamples_at_seed_block_not_record_level():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "rng.integers(0, n_blocks, size=n_blocks)" in source


def test_bootstrap_empty_records_returns_none_stats():
    result = ep.paired_seed_block_bootstrap([], n_resamples=10, bootstrap_seed=0)
    assert result["win_rate_diff"] == {"point": None, "ci_95": None}
    assert result["net_worth_diff"] == {"point": None, "ci_95": None}
    assert result["n_seed_blocks"] == 0


# ── McNemar (SECONDARY, clustered) ──────────────────────────────────────


def test_mcnemar_no_discordant_pairs_is_p_one():
    outcomes = [(True, True), (False, False), (True, True)]
    result = ep.mcnemar_exact(outcomes)
    assert result == {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0}


def test_mcnemar_perfectly_symmetric_discordant_is_p_one():
    outcomes = [(False, True)] * 5 + [(True, False)] * 5
    result = ep.mcnemar_exact(outcomes)
    assert result["b"] == 5 and result["c"] == 5 and result["n_discordant"] == 10
    assert result["p_value"] == pytest.approx(1.0)


def test_mcnemar_all_discordant_one_direction_n2():
    outcomes = [(True, False), (True, False)]
    result = ep.mcnemar_exact(outcomes)
    assert result["b"] == 0 and result["c"] == 2
    assert result["p_value"] == pytest.approx(0.5)


def test_mcnemar_all_discordant_one_direction_n4():
    outcomes = [(True, False)] * 4
    result = ep.mcnemar_exact(outcomes)
    assert result["b"] == 0 and result["c"] == 4
    assert result["p_value"] == pytest.approx(0.125)


def test_mcnemar_matches_known_published_binomial_test_value():
    """k=1, n=10 two-sided exact binomial test: p = 22/1024 = 0.021484375
    (matches R's binom.test(1, 10))."""
    outcomes = [(False, True)] * 1 + [(True, False)] * 9
    result = ep.mcnemar_exact(outcomes)
    assert result["b"] == 1 and result["c"] == 9
    assert result["p_value"] == pytest.approx(22 / 1024, abs=1e-9)


def test_mcnemar_docstring_documents_clustering_and_secondary_status():
    source = MODULE_PATH.read_text(encoding="utf-8")
    doc_start = source.index("def mcnemar_exact")
    doc_block = source[doc_start:doc_start + 1600]
    assert "SECONDARY" in doc_block
    assert "cluster" in doc_block.lower()


# ── fallback contamination flag ────────────────────────────────────────


def test_fallback_contamination_flag():
    clean = ep.fallback_contamination(0, 0)
    assert clean == {"baseline_fallbacks": 0, "candidate_fallbacks": 0, "total_fallbacks": 0, "contaminated": False}

    dirty = ep.fallback_contamination(3, 0)
    assert dirty["contaminated"] is True
    assert dirty["total_fallbacks"] == 3


# ── paired_evaluation_summary: full report shape ───────────────────────


def _full_fixture():
    seeds = (1, 2, 3, 4)
    baseline_records = []
    candidate_records = []
    for seed in seeds:
        candidate_favored = seed % 2 == 0
        for seat in range(4):
            baseline_records.append(
                {"seed": seed, "seat": seat, "win": (seat == 0) and not candidate_favored, "net_worth": 1000.0}
            )
            candidate_records.append(
                {"seed": seed, "seat": seat, "win": (seat == 1) and candidate_favored, "net_worth": 1000.0 + seed * 100.0}
            )
    return baseline_records, candidate_records


def _walk_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _walk_keys(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _walk_keys(value)


def test_paired_evaluation_summary_structure():
    baseline_records, candidate_records = _full_fixture()
    summary = ep.paired_evaluation_summary(
        baseline_records=baseline_records, candidate_records=candidate_records,
        baseline_fallbacks=2, candidate_fallbacks=0,
        n_resamples=50, bootstrap_seed=0, n_randomization_samples=50, randomization_seed=0,
    )
    assert set(summary.keys()) == {"descriptive", "primary", "secondary", "fallback_contamination"}
    assert set(summary["descriptive"].keys()) == {"baseline_wilson_95", "candidate_wilson_95"}
    assert set(summary["primary"].keys()) == {"seed_block_paired_randomization", "seed_block_bootstrap"}
    assert set(summary["secondary"].keys()) == {"seat_level_mcnemar_secondary"}
    assert summary["fallback_contamination"]["contaminated"] is True


def test_paired_evaluation_summary_computes_no_promotion_verdict_anywhere():
    baseline_records, candidate_records = _full_fixture()
    summary = ep.paired_evaluation_summary(
        baseline_records=baseline_records, candidate_records=candidate_records,
        baseline_fallbacks=0, candidate_fallbacks=0,
        n_resamples=50, bootstrap_seed=0, n_randomization_samples=50, randomization_seed=0,
    )
    all_keys = set(_walk_keys(summary))
    forbidden = {"promote", "promotion_recommended", "go_kill", "recommended", "verdict", "kill_recommended"}
    assert forbidden.isdisjoint(all_keys)


def test_paired_evaluation_summary_deterministic():
    baseline_records, candidate_records = _full_fixture()
    kwargs = dict(
        baseline_records=baseline_records, candidate_records=candidate_records,
        baseline_fallbacks=0, candidate_fallbacks=0,
        n_resamples=50, bootstrap_seed=1, n_randomization_samples=50, randomization_seed=1,
    )
    result_a = ep.paired_evaluation_summary(**kwargs)
    result_b = ep.paired_evaluation_summary(**kwargs)
    assert result_a == result_b


def test_paired_evaluation_summary_raises_on_missing_pairing():
    baseline_records, candidate_records = _full_fixture()
    candidate_records = candidate_records[:-1]  # drop one seat's candidate record
    with pytest.raises(RuntimeError, match="incomplete baseline/candidate pairing"):
        ep.paired_evaluation_summary(
            baseline_records=baseline_records, candidate_records=candidate_records,
            baseline_fallbacks=0, candidate_fallbacks=0,
        )


def test_paired_evaluation_summary_raises_on_duplicate_seed_seat():
    baseline_records, candidate_records = _full_fixture()
    baseline_records.append(dict(baseline_records[0]))  # duplicate an existing (seed, seat)
    with pytest.raises(RuntimeError, match="duplicate"):
        ep.paired_evaluation_summary(
            baseline_records=baseline_records, candidate_records=candidate_records,
            baseline_fallbacks=0, candidate_fallbacks=0,
        )
