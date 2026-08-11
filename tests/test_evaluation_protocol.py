"""Tests for scripts/evaluation_protocol.py: seed-pool classification and
disjointness, the FINAL_BLIND guard, exact McNemar against hand-verified
and known published values, seed-block bootstrap determinism, and the
paired-evaluation summary's refusal to compute a promotion boolean.
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


def test_require_non_final_blind_passes_for_dev_and_promotion_seeds():
    ep.require_non_final_blind([42, 10000], context="test")
    ep.require_non_final_blind(list(range(*ep.PROMOTION_SEED_RANGE)), context="test")


def test_require_non_final_blind_raises_for_final_blind_seed():
    lo, _ = ep.FINAL_BLIND_SEED_RANGE
    with pytest.raises(RuntimeError, match="FINAL_BLIND"):
        ep.require_non_final_blind([42, lo], context="fake_eval_script")


def test_require_non_final_blind_cannot_be_bypassed_by_mixed_batch():
    """A normal eval seed list that ACCIDENTALLY includes one FINAL_BLIND
    seed among otherwise-fine seeds must still be refused entirely, not
    silently filtered down to the safe subset."""
    lo, _ = ep.FINAL_BLIND_SEED_RANGE
    mixed = [42, 10000, 30000, lo]
    with pytest.raises(RuntimeError):
        ep.require_non_final_blind(mixed, context="normal_eval")


# ── Wilson interval (descriptive only) ─────────────────────────────────


def test_wilson_interval_matches_known_value():
    lower, upper = ep.wilson_95_interval(50, 100)
    assert lower == pytest.approx(0.4038, abs=1e-3)
    assert upper == pytest.approx(0.5962, abs=1e-3)


# ── McNemar exact test: known/hand-verified cases ──────────────────────


def test_mcnemar_no_discordant_pairs_is_p_one():
    outcomes = [(True, True), (False, False), (True, True)]
    result = ep.mcnemar_exact(outcomes)
    assert result == {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0}


def test_mcnemar_perfectly_symmetric_discordant_is_p_one():
    """b == c (5 and 5 out of 10 discordant pairs) is the least-surprising
    outcome under the null - p must be 1.0."""
    outcomes = [(False, True)] * 5 + [(True, False)] * 5
    result = ep.mcnemar_exact(outcomes)
    assert result["b"] == 5 and result["c"] == 5 and result["n_discordant"] == 10
    assert result["p_value"] == pytest.approx(1.0)


def test_mcnemar_all_discordant_one_direction_n2():
    """Hand-verified: n=2, k=0 -> pmf=[0.25,0.5,0.25], p = pmf[0]+pmf[2] = 0.5."""
    outcomes = [(True, False), (True, False)]
    result = ep.mcnemar_exact(outcomes)
    assert result["b"] == 0 and result["c"] == 2
    assert result["p_value"] == pytest.approx(0.5)


def test_mcnemar_all_discordant_one_direction_n4():
    """Hand-verified: n=4, k=0 -> pmf=[1,4,6,4,1]/16, p = (pmf[0]+pmf[4]) = 0.125."""
    outcomes = [(True, False)] * 4
    result = ep.mcnemar_exact(outcomes)
    assert result["b"] == 0 and result["c"] == 4
    assert result["p_value"] == pytest.approx(0.125)


def test_mcnemar_matches_known_published_binomial_test_value():
    """k=1, n=10 two-sided exact binomial test: a standard textbook/R
    binom.test(1, 10) result, p = 22/1024 = 0.021484375."""
    outcomes = [(False, True)] * 1 + [(True, False)] * 9
    result = ep.mcnemar_exact(outcomes)
    assert result["b"] == 1 and result["c"] == 9
    assert result["p_value"] == pytest.approx(22 / 1024, abs=1e-9)


# ── seed-block paired bootstrap ────────────────────────────────────────


def _fake_records():
    """Heterogeneous across seeds on purpose (win pattern flips by seed
    parity, net-worth diff scales with seed) so seed-block resampling
    actually produces a non-degenerate distribution - a fixture where every
    block behaves identically would make the bootstrap CI collapse to a
    single point regardless of bootstrap_seed, which would defeat the
    determinism-vs-different-seed tests below."""
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
    records = _fake_records()
    result_a = ep.paired_seed_block_bootstrap(records, n_resamples=200, bootstrap_seed=7)
    result_b = ep.paired_seed_block_bootstrap(records, n_resamples=200, bootstrap_seed=7)
    assert result_a == result_b


def test_bootstrap_different_seed_can_differ():
    records = _fake_records()
    result_a = ep.paired_seed_block_bootstrap(records, n_resamples=200, bootstrap_seed=7)
    result_b = ep.paired_seed_block_bootstrap(records, n_resamples=200, bootstrap_seed=8)
    assert result_a["win_rate_diff"]["ci_95"] != result_b["win_rate_diff"]["ci_95"] or (
        result_a["net_worth_diff"]["ci_95"] != result_b["net_worth_diff"]["ci_95"]
    )


def test_bootstrap_point_estimates_match_direct_computation():
    records = _fake_records()
    result = ep.paired_seed_block_bootstrap(records, n_resamples=50, bootstrap_seed=0)
    # 2/16 baseline wins (odd seeds), 2/16 candidate wins (even seeds) -> diff 0
    assert result["win_rate_diff"]["point"] == pytest.approx(0.0)
    # net worth diff is 100*seed per record, mean seed over (1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4) = 2.5
    assert result["net_worth_diff"]["point"] == pytest.approx(250.0)
    assert result["n_seed_blocks"] == 4
    assert result["n_records"] == 16


def test_bootstrap_resamples_at_seed_block_not_record_level():
    """Structural check: the resampling loop must draw block indices sized
    to n_blocks (seeds), not n_records - guards against accidentally
    flattening the seed-block design into a plain per-record bootstrap."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "rng.integers(0, n_blocks, size=n_blocks)" in source


def test_bootstrap_empty_records_returns_none_stats():
    result = ep.paired_seed_block_bootstrap([], n_resamples=10, bootstrap_seed=0)
    assert result["win_rate_diff"] == {"point": None, "ci_95": None}
    assert result["net_worth_diff"] == {"point": None, "ci_95": None}
    assert result["n_seed_blocks"] == 0


# ── fallback contamination flag ────────────────────────────────────────


def test_fallback_contamination_flag():
    clean = ep.fallback_contamination(0, 0)
    assert clean == {"baseline_fallbacks": 0, "candidate_fallbacks": 0, "total_fallbacks": 0, "contaminated": False}

    dirty = ep.fallback_contamination(3, 0)
    assert dirty["contaminated"] is True
    assert dirty["total_fallbacks"] == 3


# ── paired_evaluation_summary: no promotion boolean computed ──────────


def test_paired_evaluation_summary_computes_no_promotion_verdict():
    summary = ep.paired_evaluation_summary(
        baseline_wins=2, baseline_games=40,
        candidate_wins=2, candidate_games=40,
        paired_outcomes=[(True, False), (False, True)],
        bootstrap_records=_fake_records(),
        baseline_fallbacks=1, candidate_fallbacks=0,
        n_resamples=50, bootstrap_seed=0,
    )
    assert "mcnemar" in summary
    assert "bootstrap" in summary
    assert "fallback_contamination" in summary
    forbidden_keys = {"promote", "promotion_recommended", "go_kill", "recommended", "verdict"}
    assert forbidden_keys.isdisjoint(summary.keys())


def test_module_never_derives_a_verdict_from_wilson_overlap_alone():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "_recommended" not in source
    assert "non_overlapping" not in source.lower()
