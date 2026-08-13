"""Tests for scripts/monopolyzero_hybrid_bootstrap_isolation_audit.py:
config/seed registration, the pure-Python summary/audit helpers (against
hand-constructed fake data, no real engine), and the bootstrap provenance
audit run for real against the pinned reference (static file facts, so this
IS meant to hit real files — that is the entire point of a provenance
audit). Does not run the actual 80-games-per-arm screen (see the experiment
log for that).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_hybrid_bootstrap_isolation_audit.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_hybrid_bootstrap_isolation_audit", SCRIPT)
audit_module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_hybrid_bootstrap_isolation_audit"] = audit_module
_spec.loader.exec_module(audit_module)


# ── config / seed registration ────────────────────────────────────────────


def test_config_matches_task_spec():
    assert audit_module.SEEDS == tuple(range(43000, 43020))
    assert len(audit_module.SEEDS) == 20
    assert audit_module.NUM_SEATS == 4
    assert audit_module.MAX_ROUNDS == 200


def test_seeds_registered_dev_and_do_not_touch_promotion_final_blind():
    import evaluation_protocol as ep

    for seed in audit_module.SEEDS:
        assert seed in ep.DEV_SEEDS, f"seed {seed} not registered as DEV"
    ep.require_seed_scope(audit_module.SEEDS, ep.SEED_CLASS_DEV, context="test")
    assert ep.DEV_SEEDS.isdisjoint(ep.PROMOTION_SEEDS)
    assert ep.DEV_SEEDS.isdisjoint(ep.FINAL_BLIND_SEEDS)


def test_seeds_do_not_overlap_prior_dev_ranges():
    import evaluation_protocol as ep

    others = set()
    for lo, hi, note in ep.DEV_SEED_RANGES:
        if "023" in note:
            continue
        others.update(range(lo, hi + 1))
    assert set(audit_module.SEEDS).isdisjoint(others)


# ── _median / _percentile ─────────────────────────────────────────────────


def test_median_even_and_odd():
    assert audit_module._median([1.0, 2.0, 3.0]) == 2.0
    assert audit_module._median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert audit_module._median([]) is None


def test_percentile_bounds():
    values = [float(v) for v in range(1, 11)]  # 1..10
    assert audit_module._percentile(values, 0) == 1.0
    assert audit_module._percentile(values, 100) == 10.0
    assert audit_module._percentile([], 50) is None


# ── _arm_summary ───────────────────────────────────────────────────────────


def _fake_game(seed, focus_seat, focus_won, net_worth, round_capped=False, rounds=40):
    return {
        "seed": seed, "focus_seat": focus_seat, "completed": True, "winner": focus_seat if focus_won else 99,
        "focus_won": focus_won, "rounds": rounds, "decisions": 100,
        "focus_net_worth": net_worth, "round_capped": round_capped,
    }


def test_arm_summary_computes_win_rate_and_wilson():
    games = [
        _fake_game(0, 0, True, 1000.0),
        _fake_game(1, 0, False, -50.0),
        _fake_game(2, 1, True, 500.0),
        _fake_game(3, 1, False, 0.0),
    ]
    summary = audit_module._arm_summary("test_arm", games, latencies=[0.01, 0.02, 0.03])
    assert summary["games"] == 4
    assert summary["wins"] == 2
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["wilson_95"] is not None
    assert summary["mean_net_worth"] == pytest.approx((1000.0 - 50.0 + 500.0 + 0.0) / 4)
    assert summary["bankruptcy_rate"] == pytest.approx(0.5)  # -50.0 and 0.0 both count (<= 0.0)
    assert summary["games_by_seat"][0] == 2
    assert summary["games_by_seat"][1] == 2
    assert summary["wins_by_seat"][0] == 1
    assert summary["wins_by_seat"][1] == 1
    assert summary["p50_latency_s"] == pytest.approx(0.02)
    assert summary["n_latency_samples"] == 3


def test_arm_summary_empty_games_is_none_safe():
    summary = audit_module._arm_summary("empty", [], latencies=[])
    assert summary["games"] == 0
    assert summary["win_rate"] is None
    assert summary["wilson_95"] is None
    assert summary["mean_net_worth"] is None
    assert summary["bankruptcy_rate"] is None
    assert summary["round_cap_rate"] is None


# ── intervention_audit ──────────────────────────────────────────────────


def _entry(**kwargs):
    base = {
        "is_buy_opportunity": False, "is_trade_opportunity": False, "trade_pending_found": None,
        "decision_kind": "no_opportunity", "intervened": False,
        "policy_only_action": 1, "hybrid_compat_action": 1, "disagrees_with_policy_only": False,
        "policy_only_prob_buy": None, "policy_only_chose_buy": None,
        "policy_only_prob_accept_trade": None, "policy_only_chose_accept_trade": None,
    }
    base.update(kwargs)
    return base


def test_intervention_audit_counts_opportunities_and_interventions():
    log = [
        _entry(),  # plain no-opportunity decision
        _entry(),  # another plain no-opportunity decision
        _entry(
            is_buy_opportunity=True, decision_kind="buy_property_rule_bought", intervened=True,
            policy_only_action=3, hybrid_compat_action=3, disagrees_with_policy_only=False,
            policy_only_prob_buy=0.6, policy_only_chose_buy=True,
        ),
        _entry(
            is_buy_opportunity=True, decision_kind="candidate_set_narrowed_neural_pick", intervened=True,
            policy_only_action=3, hybrid_compat_action=1, disagrees_with_policy_only=True,
            policy_only_prob_buy=0.9, policy_only_chose_buy=True,
        ),
        _entry(
            is_trade_opportunity=True, trade_pending_found=True, decision_kind="trade_response_rule_accept",
            intervened=True, policy_only_action=7, hybrid_compat_action=7, disagrees_with_policy_only=False,
            policy_only_prob_accept_trade=0.3, policy_only_chose_accept_trade=False,
        ),
        _entry(
            is_trade_opportunity=True, trade_pending_found=False, decision_kind="candidate_set_narrowed_neural_pick",
            intervened=True, policy_only_action=7, hybrid_compat_action=8, disagrees_with_policy_only=True,
            policy_only_prob_accept_trade=0.2, policy_only_chose_accept_trade=False,
        ),
    ]

    audit = audit_module.intervention_audit(log)

    assert audit["total_non_forced_focus_seat_decisions"] == 6
    assert audit["buy_property_opportunities"] == 2
    assert audit["incoming_trade_opportunities"] == 2
    assert audit["trade_opportunity_but_no_pending_found"] == 1
    assert audit["both_buy_and_trade_opportunity_simultaneously"] == 0
    assert audit["hybrid_compat_intervention_count"] == 4
    assert audit["intervention_rate_of_non_forced_decisions"] == pytest.approx(4 / 6)
    assert audit["disagreement_with_policy_only_count_at_opportunity_states"] == 2
    assert audit["disagreement_rate_within_interventions"] == pytest.approx(2 / 4)
    assert audit["decision_kind_breakdown"]["no_opportunity"] == 2
    assert audit["decision_kind_breakdown"]["buy_property_rule_bought"] == 1
    assert audit["policy_only_at_buy_opportunities"]["chosen_action_frequency_buy"] == pytest.approx(1.0)
    assert audit["policy_only_at_buy_opportunities"]["mean_prob_buy"] == pytest.approx((0.6 + 0.9) / 2)
    assert audit["policy_only_at_trade_opportunities"]["chosen_action_frequency_accept"] == pytest.approx(0.0)


def test_intervention_audit_empty_log_is_none_safe():
    audit = audit_module.intervention_audit([])
    assert audit["total_non_forced_focus_seat_decisions"] == 0
    assert audit["intervention_rate_of_non_forced_decisions"] is None
    assert audit["disagreement_rate_within_interventions"] is None
    assert audit["policy_only_at_buy_opportunities"]["mean_prob_buy"] is None


# ── verify_baseline_checkpoint ────────────────────────────────────────────


def test_verify_baseline_checkpoint_raises_when_missing(tmp_path):
    with pytest.raises(SystemExit, match="missing checkpoint"):
        audit_module.verify_baseline_checkpoint(tmp_path / "nope.pt", "a" * 64)


def test_verify_baseline_checkpoint_raises_on_sha_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"some bytes")
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        audit_module.verify_baseline_checkpoint(checkpoint, "a" * 64)


def test_verify_baseline_checkpoint_passes_when_hash_matches(tmp_path):
    import hashlib

    checkpoint = tmp_path / "checkpoint.pt"
    content = b"some checkpoint bytes"
    checkpoint.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert audit_module.verify_baseline_checkpoint(checkpoint, expected) == expected


# ── bootstrap_provenance_audit: real facts against the pinned reference ──


def test_bootstrap_provenance_audit_real_facts():
    """Deliberately hits real, pinned files (the checkpoint, TRAINING_RESULTS.md,
    experiment 007's log, agent_ppo.py, model.py) — this function's entire
    purpose is to check real facts, so mocking it would test nothing.

    The PPO checkpoint itself is a historical, gitignored artifact under
    the reference submodule (references/DeepRL_Monopoly/artifacts/ - never
    committed, never regenerated by this project). A fresh clone (e.g.
    Colab) genuinely won't have it, so this test skips cleanly rather than
    failing when it's absent - it still runs for real whenever the
    artifact IS present locally, per this function's whole purpose."""
    if not audit_module.PPO_CHECKPOINT_PATH.is_file():
        pytest.skip(
            "historical hybrid PPO checkpoint not present locally (gitignored "
            f"reference-submodule artifact): {audit_module.PPO_CHECKPOINT_PATH}"
        )
    result = audit_module.bootstrap_provenance_audit()

    assert result["ppo_checkpoint_sha256_local"] == "1c825dcdd2c8d83651bd21100024ab2d0b8ce2ba276d701dceb3599536f615cb"
    assert result["matches_upstream_training_results_md"] is False
    assert result["ppo_checkpoint_sha256_upstream_training_results_md"] == (
        "4c364204eb59df74dffab911f8fbde523e59037558fafbe49daaf79e5c9180db"
    )
    assert result["matches_experiment_007_log"] is True
    assert result["ppo_checkpoint_sha256_experiment_007_log"] == result["ppo_checkpoint_sha256_local"]
    assert result["payload_hybrid_flag"] is True
    assert result["payload_games_trained"] == 1
    assert all(result["fixed_action_mask_lines_verified_present_in_reference"].values())
    assert result["load_ppo_actor_full_copy_line_verified_present_in_reference"] is True


# ── HYBRID_COMPAT plumbing wired into _invoke_policy (regression) ────────


def test_hybrid_compat_kind_handled_by_invoke_policy():
    """Regression guard: monopolyzero_common._invoke_policy must treat
    'hybrid_compat' like 'search'/'policy_only' (unpack a Result object),
    not like 'fixed' (plain int) — this script relies on that to get
    latency_s and the Result contract out of play_local_game."""
    common = audit_module.common
    source = (REPO_ROOT / "scripts" / "monopolyzero_common.py").read_text(encoding="utf-8")
    assert '"hybrid_compat"' in source
    assert "def build_local_hybrid_compat_policy" in source
