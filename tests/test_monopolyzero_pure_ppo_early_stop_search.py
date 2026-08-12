"""Tests for scripts/monopolyzero_pure_ppo_early_stop_search.py:
- source-level ASU-import guard
- seed-derivation constants (BASE_SEED_ARG, CHECKPOINT_STAGES, train/selection/
  confirm seed ranges disjoint from each other and from every prior pool)
- resume_and_checkpoint(): tiny real-engine smoke (2 one-game segments, a
  fresh local checkpoint standing in for the real 32-game one) proving the
  per-segment episode-seed formula and games_trained bookkeeping are correct
- eval_candidate(): tiny real-engine smoke reusing 028's summarize() contract
- the checkpoint tie-break rule (win rate, then net worth, then earlier
  checkpoint) matches the task's pre-registered selection rule
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_pure_ppo_early_stop_search.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import evaluation_protocol as ep  # noqa: E402
import monopolyzero_common as common  # noqa: E402
import monopolyzero_pure_ppo_strength_screen as screen  # noqa: E402

_spec = importlib.util.spec_from_file_location("monopolyzero_pure_ppo_early_stop_search", SCRIPT)
module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_pure_ppo_early_stop_search"] = module
_spec.loader.exec_module(module)


def test_source_never_imports_asu():
    import_lines = [
        line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("ASU_FROZEN_TEACHER", "monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert not hits, hits


def test_seed_constants_derive_exact_episode_seed_range():
    assert module.BASE_SEED_ARG == 46968
    train_seeds = list(range(module.EPISODE_SEED_START, module.EPISODE_SEED_START + module.ADDITIONAL_GAMES))
    assert train_seeds[0] == 47000 and train_seeds[-1] == 47095 and len(train_seeds) == 96
    derived = [module.BASE_SEED_ARG + absolute_game - 1 for absolute_game in range(33, 129)]
    assert derived == train_seeds
    assert module.CHECKPOINT_STAGES == (48, 64, 80, 96, 112, 128)


def test_selection_and_confirm_seeds_are_dev_scoped_and_mutually_disjoint():
    train_seeds = set(range(module.EPISODE_SEED_START, module.EPISODE_SEED_START + module.ADDITIONAL_GAMES))
    selection_seeds = set(screen._seed_range(module.SELECTION_SEED_BASE, module.N_SELECTION_SEEDS))
    confirm_seeds = set(screen._seed_range(module.CONFIRM_SEED_BASE, module.N_CONFIRM_SEEDS))

    assert len(selection_seeds) == 8 and len(confirm_seeds) == 20
    assert selection_seeds.isdisjoint(confirm_seeds)
    assert selection_seeds.isdisjoint(train_seeds)
    assert confirm_seeds.isdisjoint(train_seeds)
    for pool in (selection_seeds, confirm_seeds, train_seeds):
        assert pool.isdisjoint(ep.FINAL_BLIND_SEEDS)
        assert pool.isdisjoint(ep.PROMOTION_SEEDS)
        for seed in pool:
            assert ep.classify_seed(seed) == ep.SEED_CLASS_DEV


def test_checkpoint_selection_tie_break_rule():
    def pick(selection: dict) -> int:
        return max(selection, key=lambda t: (
            selection[t]["candidate_win_rate"], selection[t]["candidate_mean_net_worth"], -t))

    higher_win_rate_wins = {
        48: {"candidate_win_rate": 0.30, "candidate_mean_net_worth": 1000.0},
        80: {"candidate_win_rate": 0.35, "candidate_mean_net_worth": 500.0},
    }
    assert pick(higher_win_rate_wins) == 80

    net_worth_tiebreak = {
        48: {"candidate_win_rate": 0.30, "candidate_mean_net_worth": 900.0},
        64: {"candidate_win_rate": 0.30, "candidate_mean_net_worth": 1100.0},
    }
    assert pick(net_worth_tiebreak) == 64

    earlier_checkpoint_tiebreak = {
        48: {"candidate_win_rate": 0.30, "candidate_mean_net_worth": 1000.0},
        64: {"candidate_win_rate": 0.30, "candidate_mean_net_worth": 1000.0},
    }
    assert pick(earlier_checkpoint_tiebreak) == 48


# ── tiny REAL-engine smokes ─────────────────────────────────────────────


def _tiny_32game_checkpoint(tmp_path):
    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.games_trained = 32
    path = tmp_path / "tiny_32game.pt"
    agent.save(str(path))
    return path, screen._full_actor_sha256(agent.actor)


def test_resume_and_checkpoint_smoke_two_tiny_segments(monkeypatch, tmp_path):
    common.ensure_reference_on_path()
    checkpoint_path, actor_hash = _tiny_32game_checkpoint(tmp_path)

    monkeypatch.setattr(screen, "DEFAULT_CANDIDATE_CHECKPOINT", checkpoint_path)
    monkeypatch.setattr(screen, "EXPECTED_CANDIDATE_ACTOR_SHA256", actor_hash)
    tiny_base = 99000
    monkeypatch.setattr(module, "CHECKPOINT_STAGES", (33, 34))
    monkeypatch.setattr(module, "ADDITIONAL_GAMES", 2)
    monkeypatch.setattr(module, "EPISODE_SEED_START", tiny_base + 32)
    monkeypatch.setattr(module, "BASE_SEED_ARG", tiny_base)
    monkeypatch.setattr(module, "CHECKPOINT_DIR", tmp_path)

    stages = module.resume_and_checkpoint()

    assert set(stages) == {33, 34}
    assert Path(stages[33]["path"]).is_file() and Path(stages[34]["path"]).is_file()
    assert stages[33]["path"] != stages[34]["path"]
    assert stages[34]["actor_sha256"] != actor_hash


def test_resume_and_checkpoint_raises_on_wrong_starting_games_trained(monkeypatch, tmp_path):
    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.games_trained = 5  # NOT 32
    path = tmp_path / "wrong_games_trained.pt"
    agent.save(str(path))

    monkeypatch.setattr(screen, "DEFAULT_CANDIDATE_CHECKPOINT", path)
    monkeypatch.setattr(screen, "EXPECTED_CANDIDATE_ACTOR_SHA256", screen._full_actor_sha256(agent.actor))

    import pytest

    with pytest.raises(SystemExit, match="games_trained==32"):
        module.resume_and_checkpoint()


def test_eval_candidate_smoke_returns_summarize_contract(monkeypatch, tmp_path):
    import torch

    common.ensure_reference_on_path()
    checkpoint_path, actor_hash = _tiny_32game_checkpoint(tmp_path)
    monkeypatch.setattr(screen, "DEFAULT_CANDIDATE_CHECKPOINT", checkpoint_path)
    monkeypatch.setattr(screen, "EXPECTED_CANDIDATE_ACTOR_SHA256", actor_hash)

    summary = module.eval_candidate(checkpoint_path, [46000], torch.device("cpu"))

    assert summary["n_games"] == 4  # 1 seed x 4 rotations
    assert summary["integrity"] == {"illegal_actions": 0, "crashes": 0}
    assert summary["verdict"] in ("GO", "KILL", "INCONCLUSIVE")
    assert "primary_seed_block_bootstrap_vs_25pct_null" in summary
