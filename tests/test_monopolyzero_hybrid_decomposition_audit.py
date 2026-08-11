"""Tests for scripts/monopolyzero_hybrid_decomposition_audit.py: config/seed
reuse (no new DEV registration — reuses 023's 43000-43019), the pure-Python
reconciliation and recovery/synergy helpers against hand-constructed fake
data (no real engine), and that the 023 module is genuinely reused (not
redefined). Does not run the actual ~580-game decomposition (see the
experiment log for that).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_hybrid_decomposition_audit.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_hybrid_decomposition_audit", SCRIPT)
decomp_module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_hybrid_decomposition_audit"] = decomp_module
_spec.loader.exec_module(decomp_module)


# ── config / seed reuse ──────────────────────────────────────────────────


def test_reuses_023_seeds_exactly_no_new_registration():
    assert decomp_module.SEEDS == decomp_module.audit_v1.SEEDS
    assert decomp_module.SEEDS == tuple(range(43000, 43020))
    assert decomp_module.MAX_ROUNDS == decomp_module.audit_v1.MAX_ROUNDS == 200
    assert decomp_module.NUM_SEATS == decomp_module.audit_v1.NUM_SEATS == 4
    assert decomp_module.CHECKPOINT_PATH == decomp_module.audit_v1.CHECKPOINT_PATH
    assert decomp_module.BASELINE_CHECKPOINT_SHA256 == decomp_module.audit_v1.BASELINE_CHECKPOINT_SHA256


def test_seeds_registered_dev_and_do_not_touch_promotion_final_blind():
    import evaluation_protocol as ep

    for seed in decomp_module.SEEDS:
        assert seed in ep.DEV_SEEDS, f"seed {seed} not registered as DEV"
    ep.require_seed_scope(decomp_module.SEEDS, ep.SEED_CLASS_DEV, context="test")
    assert ep.DEV_SEEDS.isdisjoint(ep.PROMOTION_SEEDS)
    assert ep.DEV_SEEDS.isdisjoint(ep.FINAL_BLIND_SEEDS)


def test_prior_experiment_log_path_points_at_023():
    assert decomp_module.PRIOR_EXPERIMENT_LOG.name == "023-hybrid-bootstrap-isolation-audit.json"
    assert decomp_module.PRIOR_EXPERIMENT_LOG.is_file()


# ── _focus_policy_factory ────────────────────────────────────────────────


class _FakeModel:
    def predict(self, state, legal_actions, actor_id):
        legal = tuple(legal_actions)
        priors = {action: 0.1 for action in legal}
        priors[legal[-1]] = 0.9
        return priors, [0.5]


def test_focus_policy_factory_kinds():
    decomp_module.common.ensure_reference_on_path()
    model = _FakeModel()
    assert decomp_module._focus_policy_factory(model, decomp_module.ARM_POLICY_ONLY)().kind == "policy_only"
    assert decomp_module._focus_policy_factory(model, decomp_module.ARM_BUY_ONLY)().kind == "hybrid_compat"
    assert decomp_module._focus_policy_factory(model, decomp_module.ARM_TRADE_ONLY)().kind == "hybrid_compat"
    assert decomp_module._focus_policy_factory(model, decomp_module.ARM_BOTH)().kind == "hybrid_compat"


def test_focus_policy_factory_rejects_unknown_arm():
    with pytest.raises(ValueError):
        decomp_module._focus_policy_factory(_FakeModel(), "not_a_real_arm")


def test_buy_only_and_trade_only_gate_correctly():
    model = _FakeModel()

    # Monkeypatch BEFORE building the policies: build_local_hybrid_compat_policy
    # binds fixed_buy_decision/fixed_accept_trade_decision via `from ... import
    # ...` at build time (inside _focus_policy_factory's lambda call), so the
    # patch must land first or the real reference functions get captured.
    decomp_module.common.ensure_reference_on_path()
    import monopoly_game_engine.agent_ppo as agent_ppo_module
    from monopoly_game_engine.actions import ActionType

    agent_ppo_module.fixed_buy_decision = lambda env, pid: False
    agent_ppo_module.fixed_accept_trade_decision = lambda env, pid: False

    buy_only = decomp_module._focus_policy_factory(model, decomp_module.ARM_BUY_ONLY)()
    trade_only = decomp_module._focus_policy_factory(model, decomp_module.ARM_TRADE_ONLY)()
    both = decomp_module._focus_policy_factory(model, decomp_module.ARM_BOTH)()

    class _Env:
        def __init__(self, legal):
            self._legal = legal
            self.pending_trades = {}

        def get_allowed_actions(self, seat):
            return self._legal

        def _get_state(self, seat):
            return "state"

    class _Game:
        def __init__(self, env):
            self.env = env

    legal = (int(ActionType.END_TURN), int(ActionType.BUY_PROPERTY))
    game = _Game(_Env(legal))

    buy_only.choose(game, seat=0, decision_seed=1)
    assert buy_only.log[-1]["buy_rule_active"] is True
    assert buy_only.log[-1]["trade_rule_active"] is False

    trade_only.choose(game, seat=0, decision_seed=1)
    assert trade_only.log[-1]["buy_rule_active"] is False
    assert trade_only.log[-1]["trade_rule_active"] is True

    both.choose(game, seat=0, decision_seed=1)
    assert both.log[-1]["buy_rule_active"] is True
    assert both.log[-1]["trade_rule_active"] is True


# ── _run_rotation_arm / _run_self_play_uniform_arm: games_played ─────────
#
# Regression: an earlier version of the payload's config.games_run field
# summed len(per_game) (a RECORD count - 4 per physical game for the
# self-play-optimized arm) instead of the actual number of physical games
# executed, silently overstating context_2's BOTH arm as "80 games" when
# only 20 self-play games were actually played. Fixed by tracking
# games_played explicitly in each run's return dict.


def _fake_outcome(**overrides):
    import types

    base = dict(
        completed=True, winner=0, decisions=10, final_round=5,
        final_net_worth=(100.0, 200.0, 300.0, 400.0),
        illegal_actions=0, crashed=False, search_latencies_s=[0.001],
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_run_rotation_arm_reports_one_physical_game_per_seed_per_seat(monkeypatch):
    decomp_module.common.ensure_reference_on_path()
    monkeypatch.setattr(decomp_module.common, "play_local_game", lambda **kwargs: _fake_outcome())

    model = _FakeModel()
    run = decomp_module._run_rotation_arm(
        (100, 101), decomp_module._focus_policy_factory(model, decomp_module.ARM_POLICY_ONLY),
        lambda: decomp_module.common.build_local_policy_only(model),
    )
    assert run["games_played"] == 2 * decomp_module.NUM_SEATS  # 2 seeds x 4 seats = 8 physical games
    assert len(run["per_game"]) == 8
    assert len(run["records"]) == 8


def test_run_self_play_uniform_arm_reports_one_physical_game_per_seed(monkeypatch):
    decomp_module.common.ensure_reference_on_path()
    monkeypatch.setattr(decomp_module.common, "play_local_game", lambda **kwargs: _fake_outcome())

    model = _FakeModel()
    run = decomp_module._run_self_play_uniform_arm((100, 101), decomp_module.ARM_BOTH, model)
    assert run["games_played"] == 2  # 2 seeds = 2 physical self-play games
    assert len(run["per_game"]) == 8  # but 4 seat-records extracted per game
    assert len(run["records"]) == 8


# ── reconcile_arm_against_023 ─────────────────────────────────────────────


def _summary(**overrides):
    base = {
        "games": 80, "wins": 20, "win_rate": 0.25,
        "bankruptcy_rate": 0.4125, "mean_net_worth": 6386.575, "median_net_worth": 2017.0,
        "round_cap_rate": 0.75, "wins_by_seat": {"0": 4, "1": 6, "2": 4, "3": 6},
        "p50_latency_s": 0.0003, "p95_latency_s": 0.0008,
    }
    base.update(overrides)
    return base


def test_reconcile_arm_against_023_passes_on_exact_match():
    regenerated = _summary()
    logged = _summary()
    result = decomp_module.reconcile_arm_against_023(regenerated, logged, label="test")
    assert result["matches_023"] is True
    assert result["max_delta"] == 0.0
    assert result["wins_by_seat_match"] is True


def test_reconcile_arm_against_023_ignores_latency_fields():
    regenerated = _summary(p50_latency_s=0.9999, p95_latency_s=1.5)
    logged = _summary(p50_latency_s=0.0001, p95_latency_s=0.0002)
    result = decomp_module.reconcile_arm_against_023(regenerated, logged, label="test")
    assert result["matches_023"] is True


def test_reconcile_arm_against_023_raises_on_win_rate_mismatch():
    regenerated = _summary(win_rate=0.30, wins=24)
    logged = _summary()
    with pytest.raises(RuntimeError, match="Reconciliation FAILED"):
        decomp_module.reconcile_arm_against_023(regenerated, logged, label="test")


def test_reconcile_arm_against_023_matches_despite_int_vs_str_seat_keys():
    """Regression: a freshly computed _arm_summary has int wins_by_seat keys
    (0, 1, 2, 3), but 023's logged summary round-tripped through
    json.loads has string keys ("0", "1", ...) - JSON object keys are
    always strings. This must still reconcile as a match, not a spurious
    failure."""
    regenerated = _summary(wins_by_seat={0: 4, 1: 6, 2: 4, 3: 6})
    logged = _summary(wins_by_seat={"0": 4, "1": 6, "2": 4, "3": 6})
    result = decomp_module.reconcile_arm_against_023(regenerated, logged, label="test")
    assert result["matches_023"] is True
    assert result["wins_by_seat_match"] is True


def test_reconcile_arm_against_023_raises_on_wins_by_seat_mismatch():
    regenerated = _summary(wins_by_seat={"0": 5, "1": 5, "2": 5, "3": 5})
    logged = _summary()
    with pytest.raises(RuntimeError, match="Reconciliation FAILED"):
        decomp_module.reconcile_arm_against_023(regenerated, logged, label="test")


# ── recovery_and_synergy ──────────────────────────────────────────────────


def test_recovery_and_synergy_additive_case():
    baseline = {"win_rate": 0.25}
    buy_only = {"win_rate": 0.35}   # +0.10
    trade_only = {"win_rate": 0.30}  # +0.05
    both = {"win_rate": 0.40}        # +0.15 == 0.10 + 0.05 exactly additive

    result = decomp_module.recovery_and_synergy(
        baseline=baseline, buy_only=buy_only, trade_only=trade_only, both=both, metric="win_rate"
    )
    assert result["buy_only_effect"] == pytest.approx(0.10)
    assert result["trade_only_effect"] == pytest.approx(0.05)
    assert result["both_effect"] == pytest.approx(0.15)
    assert result["sum_of_individual_effects"] == pytest.approx(0.15)
    assert result["both_minus_sum_of_individual"] == pytest.approx(0.0)
    assert result["interaction_read"] == "approximately additive"
    assert result["buy_only_recovers_fraction_of_both"] == pytest.approx(0.10 / 0.15)
    assert result["trade_only_recovers_fraction_of_both"] == pytest.approx(0.05 / 0.15)


def test_recovery_and_synergy_super_additive_case():
    baseline = {"win_rate": 0.25}
    buy_only = {"win_rate": 0.30}    # +0.05
    trade_only = {"win_rate": 0.28}  # +0.03
    both = {"win_rate": 0.50}        # +0.25, way more than 0.08 sum

    result = decomp_module.recovery_and_synergy(
        baseline=baseline, buy_only=buy_only, trade_only=trade_only, both=both, metric="win_rate"
    )
    assert result["interaction_read"] == "super-additive (positive synergy)"


def test_recovery_and_synergy_sub_additive_case():
    baseline = {"win_rate": 0.25}
    buy_only = {"win_rate": 0.40}    # +0.15
    trade_only = {"win_rate": 0.35}  # +0.10
    both = {"win_rate": 0.30}        # +0.05, well below the 0.25 sum (clear of the threshold, no float-boundary risk)

    result = decomp_module.recovery_and_synergy(
        baseline=baseline, buy_only=buy_only, trade_only=trade_only, both=both, metric="win_rate"
    )
    assert result["interaction_read"] == "sub-additive (redundant/overlapping)"


def test_recovery_and_synergy_zero_both_effect_is_none_safe():
    baseline = {"win_rate": 0.25}
    buy_only = {"win_rate": 0.25}
    trade_only = {"win_rate": 0.25}
    both = {"win_rate": 0.25}
    result = decomp_module.recovery_and_synergy(
        baseline=baseline, buy_only=buy_only, trade_only=trade_only, both=both, metric="win_rate"
    )
    assert result["buy_only_recovers_fraction_of_both"] is None
    assert result["trade_only_recovers_fraction_of_both"] is None


def test_recovery_and_synergy_net_worth_metric_has_no_label_threshold():
    baseline = {"mean_net_worth": 6000.0}
    buy_only = {"mean_net_worth": 7000.0}
    trade_only = {"mean_net_worth": 6500.0}
    both = {"mean_net_worth": 8000.0}
    result = decomp_module.recovery_and_synergy(
        baseline=baseline, buy_only=buy_only, trade_only=trade_only, both=both, metric="mean_net_worth"
    )
    assert result["interaction_read"] is None
    assert result["both_effect"] == pytest.approx(2000.0)


# ── module reuse (not redefinition) ──────────────────────────────────────


def test_reuses_audit_v1_functions_not_redefined():
    assert decomp_module.audit_v1._arm_summary is decomp_module.audit_v1.__dict__["_arm_summary"]
    source = SCRIPT.read_text(encoding="utf-8")
    assert "def _arm_summary(" not in source, "should reuse audit_v1._arm_summary, not redefine it"
    assert "def intervention_audit(" not in source, "should reuse audit_v1.intervention_audit, not redefine it"
    assert "def verify_baseline_checkpoint(" not in source, "should reuse audit_v1.verify_baseline_checkpoint, not redefine it"


def test_does_not_import_asu_coupled_modules():
    source = SCRIPT.read_text(encoding="utf-8")
    import_lines = [line for line in source.splitlines() if line.strip().startswith(("import ", "from "))]
    forbidden = ("monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert hits == []
