"""Minimal tests for scripts/monopolyzero_diagnostic_a96_vs_80_vs_fpagents.py
(isolated diagnostic, pre-registered - NOT a champion gate):
- source-level ASU-import guard
- CHECKPOINTS dict hashes match the already-recorded A96/80 values
- DEV seed range 44000-44007: classified DEV, disjoint from PROMOTION/
  FINAL_BLIND, AND genuinely unconsumed by any logged experiment (the
  claim this diagnostic's pre-registration note makes)
- exact game count math (32 games/checkpoint, 64 total)
- fixed_lineup_seats(): non-focus seats in ascending player-id order,
  matching the reference's own train.py::run_episode convention
- load_and_verify(): hash-mismatch STOP behavior (reused unmodified from
  the 034 challenger-gate runner)
- play_one_fixed_lineup_game(): tiny real-engine smoke proving the
  masked-argmax focus-seat policy is unchanged (build_masked_argmax_policy
  imported directly, not reimplemented) and the result is compatible with
  screen.summarize() unmodified
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_diagnostic_a96_vs_80_vs_fpagents.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import evaluation_protocol as ep  # noqa: E402
import monopolyzero_common as common  # noqa: E402
import monopolyzero_pure_ppo_strength_screen as screen  # noqa: E402

_spec = importlib.util.spec_from_file_location("monopolyzero_diagnostic_a96_vs_80_vs_fpagents", SCRIPT)
module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_diagnostic_a96_vs_80_vs_fpagents"] = module
_spec.loader.exec_module(module)


def test_source_never_imports_asu():
    import_lines = [
        line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("ASU_FROZEN_TEACHER", "monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert not hits, hits


# ── checkpoint registry ──────────────────────────────────────────────────


def test_checkpoints_dict_matches_already_recorded_a96_and_80_hashes():
    assert set(module.CHECKPOINTS) == {"A96", "80"}
    filename_a96, ck_a96, actor_a96 = module.CHECKPOINTS["A96"]
    assert filename_a96 == "candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt"
    assert ck_a96 == "78585ed4e2400d024633ee2878d8b88243f7f4a9f498d9019a73e44b3a830f51"
    assert actor_a96 == "2bd1e9bad3d6e0100e033507f72f15b5088d8f8fb2f04dba1ae9d759b5a34a40"

    filename_80, ck_80, actor_80 = module.CHECKPOINTS["80"]
    assert filename_80 == "candidate_ppo_80.pt"
    assert ck_80 == "e47c8d4559c1d74cfceffe114fa069a8e7a2845ba60af68be0b0ae4bd37e1dae"
    assert actor_80 == "7442f99e51619b7f6b53bc662e74d360e0a1f9cf2365b8dcb003d5a0cc3eda11"


def test_checkpoints_are_referenced_from_main_checkout_not_this_worktree():
    # This worktree's own artifacts/ dir is gitignored and empty - the
    # runner must point at the MAIN checkout's absolute path instead of
    # any REPO_ROOT-relative path that would resolve inside this worktree.
    assert "artifacts" in str(module.MAIN_CHECKOUT_ARTIFACT_DIR)
    assert str(module.MAIN_CHECKOUT_ARTIFACT_DIR) != str(common.REPO_ROOT / "artifacts" / "monopolyzero_pure_ppo_learnability_gate")
    assert module.MAIN_CHECKOUT_ARTIFACT_DIR.is_absolute()


# ── seed pool ─────────────────────────────────────────────────────────────


def test_dev_seeds_are_44000_44007_and_scope_correct():
    seeds = screen._seed_range(module.DEV_SEED_BASE, module.N_SEEDS)
    assert seeds == list(range(44000, 44008))
    assert len(seeds) == 8
    for seed in seeds:
        assert ep.classify_seed(seed) == ep.SEED_CLASS_DEV
    assert set(seeds).isdisjoint(ep.PROMOTION_SEEDS)
    assert set(seeds).isdisjoint(ep.FINAL_BLIND_SEEDS)
    ep.require_seed_scope(seeds, ep.SEED_CLASS_DEV, context="test")  # must not raise


def test_dev_seeds_44000_44007_not_consumed_by_any_logged_experiment():
    seeds = set(range(44000, 44008))
    logs_dir = common.REPO_ROOT / "logs" / "experiments"
    hits = []
    for path in logs_dir.glob("*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        logged_seeds = entry.get("seeds") or []
        for value in logged_seeds:
            if isinstance(value, int) and value in seeds:
                hits.append((path.name, value))
    assert hits == [], f"44000-44007 already consumed by a real experiment: {hits}"


# ── exact game count ──────────────────────────────────────────────────────


def test_exact_game_count_is_32_per_checkpoint_64_total():
    seeds = screen._seed_range(module.DEV_SEED_BASE, module.N_SEEDS)
    games_per_checkpoint = len(seeds) * screen.NUM_SEATS
    assert games_per_checkpoint == 32
    assert games_per_checkpoint * len(module.CHECKPOINTS) == 64


# ── fixed-lineup construction ───────────────────────────────────────────


def test_fixed_lineup_seats_is_ascending_non_focus_order():
    assert module.fixed_lineup_seats(0) == [1, 2, 3]
    assert module.fixed_lineup_seats(1) == [0, 2, 3]
    assert module.fixed_lineup_seats(2) == [0, 1, 3]
    assert module.fixed_lineup_seats(3) == [0, 1, 2]


# ── load_and_verify(): hash-gate STOP behavior (reused unmodified) ──────


def test_load_and_verify_raises_on_checkpoint_sha_mismatch(tmp_path):
    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    path = tmp_path / "checkpoint.pt"
    agent.save(str(path))

    with pytest.raises(SystemExit, match="checkpoint sha256 mismatch"):
        module.load_and_verify(path, "0" * 64, None)


# ── play_one_fixed_lineup_game(): tiny real-engine smoke ─────────────────


def test_play_one_fixed_lineup_game_smoke_uses_unmodified_masked_argmax_policy(monkeypatch):
    """Proves the focus seat's policy is exactly
    screen.build_masked_argmax_policy (not reimplemented) by monkeypatching
    it with a spy and asserting it was actually called with this game's
    focus actor/device."""
    common.ensure_reference_on_path()
    import torch
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.actor.eval()
    device = torch.device("cpu")

    calls = []
    original = screen.build_masked_argmax_policy

    def spy(actor, dev, counters, env_holder, exclude_families=()):
        calls.append((actor, dev))
        return original(actor, dev, counters, env_holder, exclude_families=exclude_families)

    monkeypatch.setattr(screen, "build_masked_argmax_policy", spy)

    game = module.play_one_fixed_lineup_game(
        game_id=1, seed=44000, focus_actor=agent.actor, focus_seat=0, device=device, max_rounds=50,
    )

    assert len(calls) == 1
    assert calls[0] == (agent.actor, device)
    assert game["illegal_actions"] == 0
    assert game["crashed"] is False
    assert set(game["fixed_agent_fallbacks"]) == {"TheHoarder", "TheDealMaker", "TheGambler"}
    assert game["per_seat"][0]["is_candidate"] is True
    for seat in (1, 2, 3):
        assert game["per_seat"][seat]["is_candidate"] is False


def test_summarize_checkpoint_accepts_smoke_game_and_adds_diagnostic_fields():
    common.ensure_reference_on_path()
    import torch
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.actor.eval()
    device = torch.device("cpu")

    game = module.play_one_fixed_lineup_game(
        game_id=1, seed=44001, focus_actor=agent.actor, focus_seat=1, device=device, max_rounds=50,
    )
    summary = module.summarize_checkpoint([game])

    assert summary["n_games"] == 1
    assert summary["integrity"] == {"illegal_actions": 0, "crashes": 0}
    assert "fixed_agent_fallback_total" in summary
    assert "focus_inference_latency_s_mean" in summary
    assert summary["focus_inference_latency_s_mean"] is None or summary["focus_inference_latency_s_mean"] >= 0.0
