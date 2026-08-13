"""Minimal tests for
scripts/monopolyzero_diagnostic_buy_property_intervention_a96.py
(isolated diagnostic, pre-registered - NOT a champion gate, NOT yet run):
- source-level ASU-import guard (zero ASU_FROZEN_TEACHER/ASUValueV1/
  ASURolloutV1 reference anywhere in this file, and the runtime
  loaded_asu_modules() check the real run will also perform)
- CHECKPOINTS dict hashes match the already-recorded A96/former-80 values
- DEV seed range 53000-53011: classified DEV, disjoint from PROMOTION/
  FINAL_BLIND, AND genuinely unconsumed by any logged experiment (the
  claim this diagnostic's pre-registration note makes)
- exact game count math (12 seeds x 4 rotations x 2 arms x 2 contexts = 192)
- fixed-lineup construction (structural-stress context, ascending
  non-focus-seat order, same as the prior A96-vs-80-vs-FPAgents diagnostic)
- assert_shadow_integrity(): both the passing case and the case it's
  designed to catch (a real bug would raise, not silently pass)
- build_buy_simple_policy()/play_one_clean_context_game(): tiny
  real-engine smoke proving BUY_SIMPLE only overrides BUY_PROPERTY
  opportunities (shadow-integrity holds) and is otherwise unchanged from
  PURE_A96's masked-argmax mechanics
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_diagnostic_buy_property_intervention_a96.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import evaluation_protocol as ep  # noqa: E402
import monopolyzero_common as common  # noqa: E402
import monopolyzero_pure_ppo_strength_screen as screen  # noqa: E402

_spec = importlib.util.spec_from_file_location("monopolyzero_diagnostic_buy_property_intervention_a96", SCRIPT)
module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_diagnostic_buy_property_intervention_a96"] = module
_spec.loader.exec_module(module)


def test_source_never_imports_asu():
    # Same convention as every other test file in this project - only scans
    # actual import lines, so the runner's own docstring explaining WHY
    # BUY_SAFETY (ASU-derived) was dropped can mention ASU_FROZEN_TEACHER/
    # safety_breakdown in prose without tripping this guard.
    import_lines = [
        line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("ASU_FROZEN_TEACHER", "monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert not hits, hits


def test_source_never_instantiates_or_calls_forbidden_asu_classes():
    # Stricter than the import-line check: these are unambiguous CODE call
    # patterns (constructor/call syntax with an open paren), not prose -
    # the pre-registration docstring never writes them this way, so this
    # cannot false-positive on the explanatory text above.
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden_call_patterns = ("ASUValueV1(", "ASURolloutV1(", "safety_breakdown(", ".decide(", ".semantic_priority(")
    hits = [pattern for pattern in forbidden_call_patterns if pattern in source]
    assert hits == [], f"forbidden ASU call pattern(s) found in source: {hits}"


def test_asu_modules_not_loaded_after_importing_this_module():
    assert common.loaded_asu_modules() == []


# ── checkpoint registry ──────────────────────────────────────────────────


def test_checkpoints_dict_matches_already_recorded_hashes():
    assert set(module.CHECKPOINTS) == {"A96", "former_80"}
    filename_a96, ck_a96, actor_a96 = module.CHECKPOINTS["A96"]
    assert filename_a96 == "candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt"
    assert ck_a96 == "78585ed4e2400d024633ee2878d8b88243f7f4a9f498d9019a73e44b3a830f51"
    assert actor_a96 == "2bd1e9bad3d6e0100e033507f72f15b5088d8f8fb2f04dba1ae9d759b5a34a40"

    filename_80, ck_80, actor_80 = module.CHECKPOINTS["former_80"]
    assert filename_80 == "candidate_ppo_80.pt"
    assert ck_80 == "e47c8d4559c1d74cfceffe114fa069a8e7a2845ba60af68be0b0ae4bd37e1dae"
    assert actor_80 == "7442f99e51619b7f6b53bc662e74d360e0a1f9cf2365b8dcb003d5a0cc3eda11"


def test_load_and_verify_raises_on_checkpoint_sha_mismatch(tmp_path):
    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    path = tmp_path / "checkpoint.pt"
    agent.save(str(path))

    with pytest.raises(SystemExit, match="checkpoint sha256 mismatch"):
        module.load_and_verify(path, "0" * 64, None)


# ── seed pool ─────────────────────────────────────────────────────────────


def test_dev_seeds_are_53000_53011_and_scope_correct():
    seeds = screen._seed_range(module.DEV_SEED_BASE, module.N_SEEDS)
    assert seeds == list(range(53000, 53012))
    assert len(seeds) == 12
    for seed in seeds:
        assert ep.classify_seed(seed) == ep.SEED_CLASS_DEV
    assert set(seeds).isdisjoint(ep.PROMOTION_SEEDS)
    assert set(seeds).isdisjoint(ep.FINAL_BLIND_SEEDS)
    ep.require_seed_scope(seeds, ep.SEED_CLASS_DEV, context="test")  # must not raise


def test_dev_seeds_53000_53011_not_consumed_by_any_logged_experiment():
    seeds = set(range(53000, 53012))
    logs_dir = common.REPO_ROOT / "logs" / "experiments"
    hits = []
    for path in logs_dir.glob("*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        for value in entry.get("seeds") or []:
            if isinstance(value, int) and value in seeds:
                hits.append((path.name, value))
    assert hits == [], f"53000-53011 already consumed by a real experiment: {hits}"


# ── exact game count ──────────────────────────────────────────────────────


def test_exact_game_count_is_192():
    seeds = screen._seed_range(module.DEV_SEED_BASE, module.N_SEEDS)
    games_per_arm_per_context = len(seeds) * screen.NUM_SEATS
    assert games_per_arm_per_context == 48
    total = games_per_arm_per_context * len(module.ARM_NAMES) * len(module.CONTEXTS)
    assert total == 192


# ── shadow-integrity assertion: both directions ──────────────────────────


def test_assert_shadow_integrity_passes_when_all_non_opportunity_decisions_agree():
    opportunity_log = [False, True, False]
    shadow_decisions = [
        {"turn_index": 0, "actual_action": 7, "shadow_action": 7, "agree": True},
        {"turn_index": 1, "actual_action": 3, "shadow_action": 9, "agree": False},  # BUY opportunity - allowed to differ
        {"turn_index": 2, "actual_action": 2, "shadow_action": 2, "agree": True},
    ]
    module.assert_shadow_integrity(opportunity_log, shadow_decisions)  # must not raise


def test_assert_shadow_integrity_raises_when_non_opportunity_decision_diverges():
    opportunity_log = [False]
    shadow_decisions = [{"turn_index": 0, "actual_action": 5, "shadow_action": 6, "agree": False}]
    with pytest.raises(RuntimeError, match="shadow-integrity check FAILED"):
        module.assert_shadow_integrity(opportunity_log, shadow_decisions)


def test_assert_shadow_integrity_raises_on_length_mismatch():
    with pytest.raises(RuntimeError, match="decision count mismatch"):
        module.assert_shadow_integrity([False, False], [{"turn_index": 0, "actual_action": 1, "shadow_action": 1, "agree": True}])


# ── fixed-lineup construction (structural-stress context) ───────────────


def test_stress_context_smoke_uses_fpagenta_b_c_in_ascending_non_focus_order():
    common.ensure_reference_on_path()
    import torch
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.actor.eval()
    device = torch.device("cpu")

    game = module.play_one_stress_context_game(
        game_id=1, seed=53000, arm="PURE_A96", focus_actor=agent.actor, focus_seat=1,
        device=device, max_rounds=50,
    )
    assert set(game["fixed_agent_fallbacks"]) == {"TheHoarder", "TheDealMaker", "TheGambler"}
    assert game["per_seat"][1]["is_candidate"] is True
    for seat in (0, 2, 3):
        assert game["per_seat"][seat]["is_candidate"] is False


# ── BUY_SIMPLE: tiny real-engine smoke, shadow integrity must hold ──────


def test_buy_simple_clean_context_smoke_shadow_integrity_holds():
    common.ensure_reference_on_path()
    import torch
    from monopoly_game_engine.agent_ppo import PPOAgent

    focus = PPOAgent(player_id=0, hybrid=False, device="cpu")
    focus.actor.eval()
    baseline = PPOAgent(player_id=0, hybrid=False, device="cpu")
    baseline.actor.eval()
    device = torch.device("cpu")

    # No exception here means assert_shadow_integrity (called internally)
    # found zero non-BUY-opportunity divergence from PURE_A96.
    game = module.play_one_clean_context_game(
        game_id=1, seed=53000, arm="BUY_SIMPLE", focus_actor=focus.actor,
        baseline_actor=baseline.actor, focus_seat=0, device=device, max_rounds=60,
    )
    assert game["illegal_actions"] == 0
    assert game["crashed"] is False
    assert game["fixed_agent_fallbacks"] is None  # clean context has no fixed agents


def test_summarize_arm_context_accepts_smoke_games():
    common.ensure_reference_on_path()
    import torch
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.actor.eval()
    device = torch.device("cpu")

    game = module.play_one_clean_context_game(
        game_id=1, seed=53001, arm="PURE_A96", focus_actor=agent.actor,
        baseline_actor=agent.actor, focus_seat=2, device=device, max_rounds=60,
    )
    summary = module.summarize_arm_context([game])
    assert summary["n_games"] == 1
    assert summary["integrity"] == {"illegal_actions": 0, "crashes": 0}
    assert "focus_inference_latency_s_mean" in summary
    assert "fixed_agent_fallback_counts" not in summary  # None fallbacks in clean context -> not added
