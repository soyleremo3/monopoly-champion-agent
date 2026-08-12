"""Tests for scripts/monopolyzero_pure_ppo_resume_128_strength_test.py:
- source-level ASU-import guard
- seed-derivation constants (BASE_SEED_ARG, train_seeds/eval_seeds ranges,
  the 128-game checkpoint path never equals the 32-game one)
- resume_training(): tiny real-engine smoke (hidden_dim=8, 2 games, a
  fresh local checkpoint stood in for the real 32-game one) proving the
  episode-seed formula/verification and games_trained bookkeeping are
  correct, without depending on the large real checkpoint artifact
- strength_test(): tiny real-engine smoke reusing 028's summarize() contract
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_pure_ppo_resume_128_strength_test.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import monopolyzero_common as common  # noqa: E402
import monopolyzero_pure_ppo_strength_screen as screen  # noqa: E402

_spec = importlib.util.spec_from_file_location("monopolyzero_pure_ppo_resume_128_strength_test", SCRIPT)
module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_pure_ppo_resume_128_strength_test"] = module
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
    # train()'s own formula: episode_seed = BASE_SEED_ARG + absolute_game - 1,
    # absolute_game running 33..128 for a resume starting at games_trained=32.
    derived = [module.BASE_SEED_ARG + absolute_game - 1 for absolute_game in range(33, 129)]
    assert derived == train_seeds


def test_eval_seeds_disjoint_from_train_seeds_and_prior_experiments():
    import evaluation_protocol as ep

    eval_seeds = screen._seed_range(module.EVAL_SEED_BASE, module.N_EVAL_SEEDS)
    train_seeds = range(module.EPISODE_SEED_START, module.EPISODE_SEED_START + module.ADDITIONAL_GAMES)
    assert set(eval_seeds).isdisjoint(set(train_seeds))
    assert set(eval_seeds).isdisjoint(ep.FINAL_BLIND_SEEDS)
    assert set(train_seeds).isdisjoint(ep.FINAL_BLIND_SEEDS)
    for seed in eval_seeds:
        assert ep.classify_seed(seed) == ep.SEED_CLASS_DEV


def test_output_checkpoint_never_equals_32game_checkpoint_path():
    assert module.OUTPUT_CHECKPOINT != screen.DEFAULT_CANDIDATE_CHECKPOINT
    assert module.OUTPUT_CHECKPOINT.name == "candidate_ppo_128.pt"


# ── tiny REAL-engine smoke: resume_training() end to end ───────────────


def _tiny_32game_checkpoint(tmp_path):
    """Stands in for the real 027 32-game checkpoint: a freshly
    initialized PPOAgent(hybrid=False) (default hidden_dim=256, required -
    screen.load_candidate_agent's PPOAgent instance uses the same default,
    and PPOAgent.load()'s own metadata check requires an exact match) with
    games_trained forced to 32, saved locally - decouples this test from
    the large real checkpoint artifact while keeping load() compatible."""
    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.games_trained = 32
    path = tmp_path / "tiny_32game.pt"
    agent.save(str(path))
    return path, screen._full_actor_sha256(agent.actor)


def test_resume_training_smoke_tiny_two_games(monkeypatch, tmp_path):
    common.ensure_reference_on_path()
    checkpoint_path, actor_hash = _tiny_32game_checkpoint(tmp_path)

    monkeypatch.setattr(screen, "DEFAULT_CANDIDATE_CHECKPOINT", checkpoint_path)
    monkeypatch.setattr(screen, "EXPECTED_CANDIDATE_ACTOR_SHA256", actor_hash)
    monkeypatch.setattr(module, "ADDITIONAL_GAMES", 2)
    tiny_base_seed = 99000
    monkeypatch.setattr(module, "BASE_SEED_ARG", tiny_base_seed)
    monkeypatch.setattr(module, "OUTPUT_CHECKPOINT", tmp_path / "tiny_128game_out.pt")

    train_seeds = [tiny_base_seed + 32, tiny_base_seed + 33]  # absolute_game 33, 34
    agent, actual_seeds = module.resume_training(train_seeds)

    assert actual_seeds == train_seeds
    assert agent.games_trained == 34
    assert module.OUTPUT_CHECKPOINT.is_file()


def test_resume_training_raises_on_wrong_starting_games_trained(monkeypatch, tmp_path):
    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.games_trained = 5  # NOT 32
    path = tmp_path / "wrong_games_trained.pt"
    agent.save(str(path))

    monkeypatch.setattr(screen, "DEFAULT_CANDIDATE_CHECKPOINT", path)
    monkeypatch.setattr(screen, "EXPECTED_CANDIDATE_ACTOR_SHA256", screen._full_actor_sha256(agent.actor))
    monkeypatch.setattr(module, "ADDITIONAL_GAMES", 1)

    import pytest

    with pytest.raises(SystemExit, match="games_trained==32"):
        module.resume_training([0])


# ── tiny REAL-engine smoke: strength_test() end to end ──────────────────


def test_strength_test_smoke_returns_summarize_contract(monkeypatch, tmp_path):
    import torch

    common.ensure_reference_on_path()
    checkpoint_path, actor_hash = _tiny_32game_checkpoint(tmp_path)
    monkeypatch.setattr(screen, "DEFAULT_CANDIDATE_CHECKPOINT", checkpoint_path)
    monkeypatch.setattr(screen, "EXPECTED_CANDIDATE_ACTOR_SHA256", actor_hash)

    candidate_agent = screen.load_candidate_agent(checkpoint_path)  # stands in for the 128-game agent
    summary = module.strength_test(candidate_agent, eval_seeds=[46000], device=torch.device("cpu"))

    assert summary["n_games"] == 4  # 1 seed x 4 rotations
    assert summary["integrity"] == {"illegal_actions": 0, "crashes": 0}
    assert summary["verdict"] in ("GO", "KILL", "INCONCLUSIVE")
    assert "primary_seed_block_bootstrap_vs_25pct_null" in summary
