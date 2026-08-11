"""Tests for scripts/monopolyzero_buy_trade_gradient_diagnostic.py:
- pure per-decision record building + aggregate math (synthetic fake
  SearchResult-shaped objects, no engine)
- the gradient-signal closed-form (predicted_prob - target_share) verified
  against a REAL backward() pass through this project's own
  monopolyzero_common._dense_target_cross_entropy - not just assumed
- config/helper reuse from monopolyzero_hybrid_bootstrap_isolation_audit
  (023), no redefinition
- source-level guards: no ASU, no fixed-rule import, no HYBRID_COMPAT
  usage, no weight update/backward/optimizer anywhere in the script
- one small REAL-engine smoke test (1 seed, tiny SearchConfig, short
  max_rounds) - not a benchmark, not a long diagnostic run
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_buy_trade_gradient_diagnostic.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_buy_trade_gradient_diagnostic", SCRIPT)
module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_buy_trade_gradient_diagnostic"] = module
_spec.loader.exec_module(module)


def _source_import_lines() -> list[str]:
    source = SCRIPT.read_text(encoding="utf-8")
    return [line for line in source.splitlines() if line.strip().startswith(("import ", "from "))]


# ── config/helper reuse from 023 - not redefined ───────────────────────────


def test_reuses_023_config_not_redefined():
    assert module.SEEDS == module.audit_v1.SEEDS == tuple(range(43000, 43020))
    assert module.MAX_ROUNDS == module.audit_v1.MAX_ROUNDS == 200
    assert module.NUM_SEATS == module.audit_v1.NUM_SEATS == 4
    assert module.CHECKPOINT_PATH == module.audit_v1.CHECKPOINT_PATH
    assert module.BASELINE_CHECKPOINT_SHA256 == module.audit_v1.BASELINE_CHECKPOINT_SHA256
    assert module.verify_baseline_checkpoint is module.audit_v1.verify_baseline_checkpoint
    assert module._median is module.audit_v1._median
    source = SCRIPT.read_text(encoding="utf-8")
    assert "SEEDS = tuple(range(" not in source, "should reuse audit_v1.SEEDS, not redefine it"
    assert "def _median(" not in source
    assert "def verify_baseline_checkpoint(" not in source


def test_consumes_no_new_dev_seed_range():
    """023's seeds are reused verbatim - this diagnostic must not touch any
    other registered DEV/PROMOTION/FINAL_BLIND range."""
    assert set(module.SEEDS).issubset(set(range(43000, 43020)))


# ── source-level guards ─────────────────────────────────────────────────


def test_does_not_import_adapters_arena_training_or_asu():
    import_lines = _source_import_lines()
    forbidden = ("monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert hits == [], f"found forbidden ASU-coupled module import(s): {hits}"
    assert not any("ASU_FROZEN_TEACHER" in line for line in import_lines)


def test_does_not_use_fixed_rule_or_hybrid_compat():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "fixed_buy_decision" not in source
    assert "fixed_accept_trade_decision" not in source
    assert "build_local_hybrid_compat_policy" not in source


def test_never_updates_weights():
    """Pure inference diagnostic - no backward pass, no optimizer, no
    weight change anywhere in the script itself (the gradient-signal number
    is the closed-form softmax-CE identity, computed without autograd)."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert ".step()" not in source
    assert "requires_grad_(True)" not in source
    assert "local_training_update" not in source


def test_uses_shared_monopolyzero_common_module():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import monopolyzero_common as common" in source


# ── _sign / _rate / _mean ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected", [(0.1, "positive"), (-0.1, "negative"), (0.0, "zero"), (None, None)]
)
def test_sign(value, expected):
    assert module._sign(value) == expected


def test_rate_handles_zero_total():
    assert module._rate(0, 0) is None
    assert module._rate(3, 6) == pytest.approx(0.5)


def test_mean_handles_empty_and_none():
    assert module._mean([]) is None
    assert module._mean([None, None]) is None
    assert module._mean([1.0, None, 3.0]) == pytest.approx(2.0)


# ── _prior_rank ─────────────────────────────────────────────────────────


def test_prior_rank_orders_by_descending_prior_then_action_id():
    legal = (3, 7, 8, 20)
    priors = {3: 0.5, 7: 0.5, 8: 0.1, 20: 0.9}
    assert module._prior_rank(20, legal, priors) == 1
    assert module._prior_rank(3, legal, priors) == 2  # tie with 7, lower action id wins
    assert module._prior_rank(7, legal, priors) == 3
    assert module._prior_rank(8, legal, priors) == 4


# ── _build_opportunity_record ───────────────────────────────────────────


def _fake_result(*, visits: dict, chosen_action: int, simulations: int = 32):
    return types.SimpleNamespace(visits=dict(visits), chosen_action=chosen_action, simulations=simulations)


BUY_ID, ACCEPT_ID, DECLINE_ID = 3, 7, 8


def test_build_opportunity_record_buy_with_visits():
    legal = (0, 1, 2, BUY_ID, 20)
    priors = {0: 0.1, 1: 0.05, 2: 0.05, BUY_ID: 0.2, 20: 0.6}
    result = _fake_result(visits={BUY_ID: 5, 20: 27}, chosen_action=20)
    record = module._build_opportunity_record(
        seat=2, legal=legal, raw_priors=priors, result=result,
        buy_id=BUY_ID, accept_id=ACCEPT_ID, decline_id=DECLINE_ID,
    )
    assert record["seat"] == 2
    assert record["is_buy_opportunity"] is True
    assert record["is_trade_opportunity"] is False
    assert record["root_chosen_action"] == 20
    assert record["policy_only_action"] == 20  # highest raw prior
    assert record["forced_single_legal"] is False
    buy = record["buy"]
    assert buy["raw_prior"] == pytest.approx(0.2)
    assert buy["visit_count"] == 5
    assert buy["target_share"] == pytest.approx(5 / 32)
    assert buy["gradient_signal"] == pytest.approx(0.2 - 5 / 32)
    assert buy["gradient_sign"] == "positive"
    assert buy["alternative_action"] == 20  # only other visited action
    assert buy["alternative_visit_count"] == 27
    assert buy["alternative_target_share"] == pytest.approx(27 / 32)
    assert "trade" not in record


def test_build_opportunity_record_buy_zero_visits():
    """The exact failure mode this diagnostic exists to detect: BUY_PROPERTY
    widened (present in visits with count 0) but never actually simulated -
    target_share=0, gradient_signal collapses to the full raw_prior with a
    positive sign (pure suppression direction, no explicit reward)."""
    legal = (0, BUY_ID, 20)
    priors = {0: 0.05, BUY_ID: 0.08, 20: 0.87}
    result = _fake_result(visits={BUY_ID: 0, 20: 30, 0: 2}, chosen_action=20)
    record = module._build_opportunity_record(
        seat=0, legal=legal, raw_priors=priors, result=result,
        buy_id=BUY_ID, accept_id=ACCEPT_ID, decline_id=DECLINE_ID,
    )
    buy = record["buy"]
    assert buy["visit_count"] == 0
    assert buy["target_share"] == 0.0
    assert buy["gradient_signal"] == pytest.approx(0.08)
    assert buy["gradient_sign"] == "positive"
    assert buy["alternative_action"] == 20


def test_build_opportunity_record_trade_both_legal():
    legal = (ACCEPT_ID, DECLINE_ID, 1)
    priors = {ACCEPT_ID: 0.3, DECLINE_ID: 0.6, 1: 0.1}
    result = _fake_result(visits={ACCEPT_ID: 10, DECLINE_ID: 20}, chosen_action=DECLINE_ID)
    record = module._build_opportunity_record(
        seat=1, legal=legal, raw_priors=priors, result=result,
        buy_id=BUY_ID, accept_id=ACCEPT_ID, decline_id=DECLINE_ID,
    )
    assert record["is_trade_opportunity"] is True
    trade = record["trade"]
    assert trade["decline_legal"] is True
    assert trade["accept_raw_prior"] == pytest.approx(0.3)
    assert trade["accept_target_share"] == pytest.approx(10 / 30)
    assert trade["accept_gradient_signal"] == pytest.approx(0.3 - 10 / 30)
    assert trade["decline_raw_prior"] == pytest.approx(0.6)
    assert trade["decline_target_share"] == pytest.approx(20 / 30)
    assert trade["decline_gradient_signal"] == pytest.approx(0.6 - 20 / 30)


def test_build_opportunity_record_trade_decline_not_legal():
    """Defensive edge case: ACCEPT_TRADE legal without DECLINE_TRADE legal -
    decline_* fields stay None rather than crashing."""
    legal = (ACCEPT_ID, 1)
    priors = {ACCEPT_ID: 0.4, 1: 0.6}
    result = _fake_result(visits={ACCEPT_ID: 4, 1: 28}, chosen_action=1)
    record = module._build_opportunity_record(
        seat=3, legal=legal, raw_priors=priors, result=result,
        buy_id=BUY_ID, accept_id=ACCEPT_ID, decline_id=DECLINE_ID,
    )
    trade = record["trade"]
    assert trade["decline_legal"] is False
    assert trade["decline_raw_prior"] is None
    assert trade["decline_visit_count"] is None
    assert trade["decline_target_share"] is None
    assert trade["decline_gradient_signal"] is None
    assert trade["decline_gradient_sign"] is None


def test_build_opportunity_record_forced_single_legal():
    legal = (BUY_ID,)
    priors = {BUY_ID: 1.0}
    result = _fake_result(visits={BUY_ID: 1}, chosen_action=BUY_ID, simulations=0)
    record = module._build_opportunity_record(
        seat=0, legal=legal, raw_priors=priors, result=result,
        buy_id=BUY_ID, accept_id=ACCEPT_ID, decline_id=DECLINE_ID,
    )
    assert record["forced_single_legal"] is True
    assert record["buy"]["alternative_action"] is None
    assert record["buy"]["alternative_visit_count"] is None


# ── buy_opportunity_aggregate / trade_response_aggregate ────────────────


def _buy_record(*, raw_prior, visit_count, total_visits, root_chosen, policy_only):
    share = visit_count / total_visits if total_visits else 0.0
    return {
        "is_buy_opportunity": True, "is_trade_opportunity": False,
        "root_chosen_action": root_chosen, "policy_only_action": policy_only,
        "buy": {
            "raw_prior": raw_prior, "prior_rank": 1, "visit_count": visit_count,
            "target_share": share, "gradient_signal": raw_prior - share,
            "gradient_sign": module._sign(raw_prior - share),
            "alternative_action": None, "alternative_visit_count": None,
            "alternative_target_share": None, "alternative_raw_prior": None,
        },
    }


def test_buy_opportunity_aggregate_rates_and_flips():
    records = [
        _buy_record(raw_prior=0.08, visit_count=0, total_visits=32, root_chosen=20, policy_only=20),  # starved, no flip
        _buy_record(raw_prior=0.5, visit_count=20, total_visits=32, root_chosen=BUY_ID, policy_only=20),  # flip toward
        _buy_record(raw_prior=0.9, visit_count=25, total_visits=32, root_chosen=1, policy_only=BUY_ID),  # flip away
    ]
    agg = module.buy_opportunity_aggregate(records, buy_action_id=BUY_ID)
    assert agg["opportunities"] == 3
    assert agg["visit_zero_rate"] == pytest.approx(1 / 3)
    assert agg["target_share_zero_rate"] == pytest.approx(1 / 3)
    assert agg["search_flipped_toward_buy"] == 1
    assert agg["search_flipped_away_from_buy"] == 1
    assert agg["positive_gradient_rate"] + agg["negative_gradient_rate"] + agg["zero_gradient_rate"] == pytest.approx(1.0)


def test_buy_opportunity_aggregate_empty_is_all_none():
    agg = module.buy_opportunity_aggregate([], buy_action_id=BUY_ID)
    assert agg["opportunities"] == 0
    assert agg["visit_zero_rate"] is None
    assert agg["mean_raw_prior"] is None
    assert agg["search_flipped_toward_buy"] == 0


def _trade_record(*, accept_prior, accept_visits, decline_prior, decline_visits, decline_legal, root_chosen, policy_only):
    total = accept_visits + (decline_visits if decline_legal else 0)
    accept_share = accept_visits / total if total else 0.0
    decline_share = (decline_visits / total if total else 0.0) if decline_legal else None
    return {
        "is_buy_opportunity": False, "is_trade_opportunity": True,
        "root_chosen_action": root_chosen, "policy_only_action": policy_only,
        "trade": {
            "accept_raw_prior": accept_prior, "accept_prior_rank": 1,
            "accept_visit_count": accept_visits, "accept_target_share": accept_share,
            "accept_gradient_signal": accept_prior - accept_share,
            "accept_gradient_sign": module._sign(accept_prior - accept_share),
            "decline_legal": decline_legal,
            "decline_raw_prior": decline_prior if decline_legal else None,
            "decline_prior_rank": 2 if decline_legal else None,
            "decline_visit_count": decline_visits if decline_legal else None,
            "decline_target_share": decline_share,
            "decline_gradient_signal": (decline_prior - decline_share) if decline_legal else None,
            "decline_gradient_sign": module._sign((decline_prior - decline_share) if decline_legal else None),
        },
    }


def test_trade_response_aggregate_accept_side():
    records = [
        _trade_record(accept_prior=0.1, accept_visits=0, decline_prior=0.9, decline_visits=30,
                      decline_legal=True, root_chosen=DECLINE_ID, policy_only=DECLINE_ID),
        _trade_record(accept_prior=0.6, accept_visits=25, decline_prior=0.4, decline_visits=5,
                      decline_legal=True, root_chosen=ACCEPT_ID, policy_only=DECLINE_ID),
    ]
    agg = module.trade_response_aggregate(records, side="accept", action_id=ACCEPT_ID)
    assert agg["opportunities"] == 2
    assert agg["visit_zero_rate"] == pytest.approx(0.5)
    assert agg["search_flipped_toward_accept"] == 1
    assert agg["search_flipped_away_from_accept"] == 0


def test_trade_response_aggregate_decline_side_filters_illegal_decline():
    records = [
        _trade_record(accept_prior=0.5, accept_visits=15, decline_prior=0.5, decline_visits=15,
                      decline_legal=True, root_chosen=ACCEPT_ID, policy_only=ACCEPT_ID),
        _trade_record(accept_prior=1.0, accept_visits=1, decline_prior=0.0, decline_visits=0,
                      decline_legal=False, root_chosen=ACCEPT_ID, policy_only=ACCEPT_ID),
    ]
    agg = module.trade_response_aggregate(records, side="decline", action_id=DECLINE_ID)
    assert agg["opportunities"] == 1  # the decline_legal=False record excluded


def test_trade_response_aggregate_rejects_bad_side():
    with pytest.raises(ValueError):
        module.trade_response_aggregate([], side="bogus", action_id=ACCEPT_ID)


# ── gradient-signal closed form vs a REAL backward() pass ──────────────


def test_gradient_signal_formula_matches_real_backward_pass():
    """predicted_prob - target_share must equal the exact gradient
    d(policy_loss)/d(logit) for that action - the standard softmax
    cross-entropy gradient identity - verified against a real backward()
    through this project's own monopolyzero_common._dense_target_cross_entropy,
    not just assumed. This is the diagnostic's stand-in for 'run train_step
    and see the gradient', per the task's explicit 'or an equivalent
    measurement via an actual backward pass' allowance."""
    module.common.ensure_reference_on_path()
    import torch

    from monopoly_bench.engine import ACTION_SPACE_SIZE
    from monopoly_bench.model import MonopolyZeroNet

    torch.manual_seed(0)
    model = MonopolyZeroNet()
    state = torch.zeros(1, 300)
    legal = [BUY_ID, ACCEPT_ID, DECLINE_ID, 20]
    mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[0, legal] = True

    logits, _ = model(state, mask)
    logits.retain_grad()
    predicted = torch.softmax(logits, dim=-1)[0]

    target_share_buy = 0.6
    target = torch.zeros(1, ACTION_SPACE_SIZE)
    target[0, BUY_ID] = target_share_buy
    target[0, 20] = 1.0 - target_share_buy

    loss = module.common._dense_target_cross_entropy(logits, target)
    loss.backward()

    expected_signal = float(predicted[BUY_ID]) - target_share_buy
    actual_grad = float(logits.grad[0, BUY_ID])
    assert actual_grad == pytest.approx(expected_signal, abs=1e-6)


# ── argparse-free entry point sanity ────────────────────────────────────


def test_main_is_callable_without_arguments_signature():
    import inspect

    signature = inspect.signature(module.main)
    assert list(signature.parameters) == ["argv"]


# ── tiny REAL-engine smoke run (not a benchmark, not a long diagnostic) ──


def test_real_engine_smoke_one_seed_tiny_search():
    """Genuinely plays 1 seed (1 physical self-play game, all 4 seats
    searched by the same diagnostic policy) against the real engine and
    checkpoint, with a small SearchConfig(simulations=4, max_depth=16) and
    max_rounds=5 - proves build_gradient_diagnostic_policy -> MaxNPUCT ->
    model.predict -> _build_opportunity_record -> the three aggregates are
    correctly wired end to end. Not a benchmark, not the 128-sim/200-round
    'real' invocation."""
    module.common.ensure_reference_on_path()
    import random

    import numpy as np
    import torch

    from monopoly_bench.config import SearchConfig
    from monopoly_bench.model import MonopolyZeroNet

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    if not module.CHECKPOINT_PATH.is_file():
        pytest.skip("baseline_pretraining.pt not present in this environment")

    model = MonopolyZeroNet.load_inference(module.CHECKPOINT_PATH)
    model.eval()
    search_config = SearchConfig(simulations=4, max_depth=16)

    result = module.run_diagnostic([43000], model, search_config, max_rounds=5)

    assert result["total_illegal"] == 0
    assert result["total_crashed"] == 0
    assert len(result["per_game"]) == 1
    assert result["per_game"][0]["seed"] == 43000
    assert result["per_game"][0]["search_calls"] >= 1

    buy_agg = module.buy_opportunity_aggregate(result["records"], buy_action_id=result["buy_id"])
    accept_agg = module.trade_response_aggregate(result["records"], side="accept", action_id=result["accept_id"])
    decline_agg = module.trade_response_aggregate(result["records"], side="decline", action_id=result["decline_id"])
    for aggregate in (buy_agg, accept_agg, decline_agg):
        assert aggregate["opportunities"] >= 0
