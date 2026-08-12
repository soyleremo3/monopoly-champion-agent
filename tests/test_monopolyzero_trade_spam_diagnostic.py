"""Tests for scripts/monopolyzero_trade_spam_diagnostic.py:
- source-level ASU-import guard
- SEEDS reuse from the already-registered DEV pool (no evaluation_protocol.py
  edit needed - verifies this script never touches that file's seed ranges)
- _records() shape matches evaluation_protocol.pair_records's expected input
- one small REAL-engine smoke: run_arm() with exclude_families for a tiny
  hand-built actor pair, proving the candidate's trade_offer_chosen count
  drops to exactly 0 in arm B while arm A is unconstrained, and that
  summarize()/paired_evaluation_summary() compose cleanly end to end
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_trade_spam_diagnostic.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import evaluation_protocol as ep  # noqa: E402
import monopolyzero_common as common  # noqa: E402

_spec = importlib.util.spec_from_file_location("monopolyzero_trade_spam_diagnostic", SCRIPT)
module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_trade_spam_diagnostic"] = module
_spec.loader.exec_module(module)


def test_source_never_imports_asu():
    import_lines = [
        line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("ASU_FROZEN_TEACHER", "monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert not hits, hits


def test_source_never_touches_evaluation_protocol_seed_ranges():
    """This task's frozen-scope rule forbids editing evaluation_protocol.py
    - a weak but cheap guard that this script's own source never assigns
    into DEV_SEED_RANGES/PROMOTION_SEED_RANGE/FINAL_BLIND_SEED_RANGE."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "DEV_SEED_RANGES" not in source
    assert "PROMOTION_SEED_RANGE" not in source
    assert "FINAL_BLIND_SEED_RANGE" not in source


def test_seeds_are_12_and_all_already_dev_classified():
    assert len(module.SEEDS) == 12
    assert len(set(module.SEEDS)) == 12
    for seed in module.SEEDS:
        assert ep.classify_seed(seed) == ep.SEED_CLASS_DEV
    assert module.SEEDS == list(range(46000, 46012))


def test_trade_offer_families_are_exactly_the_three_outgoing_families():
    assert module.TRADE_OFFER_FAMILIES == ("buy_trade", "sell_trade", "exch_trade")


def test_records_shape_matches_pair_records_expectations():
    fake_games = [
        {"seed": 1, "focus_seat": 0, "per_seat": {0: {"win": True, "net_worth": 100.0}}},
        {"seed": 1, "focus_seat": 2, "per_seat": {2: {"win": False, "net_worth": 50.0}}},
    ]
    records = module._records(fake_games)
    assert records == [
        {"seed": 1, "seat": 0, "win": True, "net_worth": 100.0},
        {"seed": 1, "seat": 2, "win": False, "net_worth": 50.0},
    ]
    # Round-trips through pair_records without raising (same shape contract).
    ep.pair_records(records, records, expected_seats=None)


# ── tiny REAL-engine smoke: run_arm end to end for both exclusion states ──


def test_real_engine_smoke_run_arm_zeroes_trade_offers_in_arm_b():
    import torch

    common.ensure_reference_on_path()
    from monopoly_game_engine.networks import ActorNetwork

    torch.manual_seed(0)
    candidate_actor = ActorNetwork(hidden_dim=8)
    torch.manual_seed(1)
    baseline_actor = ActorNetwork(hidden_dim=8)
    candidate_actor.eval()
    baseline_actor.eval()

    class _FakeAgent:
        def __init__(self, actor):
            self.actor = actor

    candidate_agent = _FakeAgent(candidate_actor)
    baseline_agent = _FakeAgent(baseline_actor)

    original_seeds = module.SEEDS
    try:
        module.SEEDS = [46000]  # 1 seed x 4 rotations = 4 tiny games per arm
        games_a = module.run_arm(
            candidate_agent=candidate_agent, baseline_agent=baseline_agent,
            exclude_families=(), device=torch.device("cpu"),
        )
        games_b = module.run_arm(
            candidate_agent=candidate_agent, baseline_agent=baseline_agent,
            exclude_families=module.TRADE_OFFER_FAMILIES, device=torch.device("cpu"),
        )
    finally:
        module.SEEDS = original_seeds

    assert len(games_a) == 4 and len(games_b) == 4
    for game in games_b:
        assert game["illegal_actions"] == 0 and game["crashed"] is False
        candidate_stats = game["per_seat"][game["focus_seat"]]
        assert candidate_stats["trade_offer_chosen"] == 0  # masked out entirely

    import monopolyzero_pure_ppo_strength_screen as screen

    summary_a = screen.summarize(games_a)
    summary_b = screen.summarize(games_b)
    assert summary_b["candidate_trade_offers_per_game"] == 0.0
    assert summary_a["integrity"] == {"illegal_actions": 0, "crashes": 0}

    paired = ep.paired_evaluation_summary(
        baseline_records=module._records(games_a), candidate_records=module._records(games_b),
        baseline_fallbacks=0, candidate_fallbacks=0,
    )
    assert set(paired.keys()) == {"descriptive", "primary", "secondary", "fallback_contamination"}
