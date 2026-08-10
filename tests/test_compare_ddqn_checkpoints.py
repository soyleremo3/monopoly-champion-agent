"""Tests for scripts/compare_ddqn_checkpoints.py, using tiny synthetic
checkpoint-shaped payloads (no real training needed)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "compare_ddqn_checkpoints.py"

_spec = importlib.util.spec_from_file_location("compare_ddqn_checkpoints", SCRIPT)
compare_ddqn_checkpoints = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_ddqn_checkpoints)


def _sample_checkpoint() -> dict:
    return {
        "format_version": 3,
        "ruleset": "ppo-plus-v2",
        "state_dim": 300,
        "action_dim": 2958,
        "player_id": 0,
        "hybrid": True,
        "hidden_dim": 1024,
        "step_count": 123,
        "games_trained": 20,
        "epsilon": 0.99,
        "training_config": {"gamma": 0.9999, "batch_size": 128},
        "online": {"w": torch.tensor([1.0, 2.0, 3.0])},
        "target": {"w": torch.tensor([1.0, 2.0, 3.0])},
        "optimizer": {"state": {}, "param_groups": [{"lr": 1e-5}]},
        "replay": {
            "capacity": 10_000,
            "states": torch.zeros(2, 4),
            "next_allowed": [[1, 2], [3]],
        },
    }


def test_identical_checkpoints_match(tmp_path):
    ckpt = _sample_checkpoint()
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    torch.save(ckpt, path_a)
    torch.save(ckpt, path_b)
    diffs = compare_ddqn_checkpoints.compare_checkpoints(path_a, path_b)
    assert diffs == []


def test_tensor_mismatch_is_detected(tmp_path):
    ckpt_a = _sample_checkpoint()
    ckpt_b = _sample_checkpoint()
    ckpt_b["online"]["w"] = torch.tensor([1.0, 2.0, 999.0])
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    torch.save(ckpt_a, path_a)
    torch.save(ckpt_b, path_b)
    diffs = compare_ddqn_checkpoints.compare_checkpoints(path_a, path_b)
    assert any("checkpoint.online.w" in d for d in diffs)


def test_scalar_mismatch_is_detected(tmp_path):
    ckpt_a = _sample_checkpoint()
    ckpt_b = _sample_checkpoint()
    ckpt_b["epsilon"] = 0.5
    ckpt_b["games_trained"] = 21
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    torch.save(ckpt_a, path_a)
    torch.save(ckpt_b, path_b)
    diffs = compare_ddqn_checkpoints.compare_checkpoints(path_a, path_b)
    assert any("checkpoint.epsilon" in d for d in diffs)
    assert any("checkpoint.games_trained" in d for d in diffs)


def test_replay_next_allowed_list_mismatch_is_detected(tmp_path):
    ckpt_a = _sample_checkpoint()
    ckpt_b = _sample_checkpoint()
    ckpt_b["replay"]["next_allowed"] = [[1, 2], [3, 4]]
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    torch.save(ckpt_a, path_a)
    torch.save(ckpt_b, path_b)
    diffs = compare_ddqn_checkpoints.compare_checkpoints(path_a, path_b)
    assert any("replay.next_allowed" in d for d in diffs)


def _sample_history(**overrides) -> dict:
    base = {
        "win_rates": [0.0, 0.0],
        "games": [1, 2],
        "rewards": [-11.0, -10.5],
        "avg_trades_initiated": [10.0, 12.0],
        "avg_trades_accepted": [1.0, 2.0],
        "avg_trades_declined": [3.0, 4.0],
        "avg_properties_acquired": [9.0, 10.0],
        "resumed_from_games": 0,
        "games_completed_this_run": 2,
        "games_completed": 2,
        "elapsed_seconds": 12.34,
        "seconds_per_game": 6.17,
        "peak_rss_gib": 0.3,
        "peak_cuda_gib": 0.0,
    }
    base.update(overrides)
    return base


def test_identical_deterministic_history_matches_despite_timing_diff(tmp_path):
    history_a = _sample_history(elapsed_seconds=100.0, peak_rss_gib=0.5)
    history_b = _sample_history(elapsed_seconds=50.0, peak_rss_gib=0.1)
    path_a = tmp_path / "a_history.json"
    path_b = tmp_path / "b_history.json"
    path_a.write_text(json.dumps(history_a), encoding="utf-8")
    path_b.write_text(json.dumps(history_b), encoding="utf-8")
    diffs, ignored_present = compare_ddqn_checkpoints.compare_history(path_a, path_b)
    assert diffs == []
    assert "elapsed_seconds" in ignored_present
    assert "peak_rss_gib" in ignored_present


def test_deterministic_history_mismatch_is_detected(tmp_path):
    history_a = _sample_history()
    history_b = _sample_history(rewards=[-11.0, -999.0])
    path_a = tmp_path / "a_history.json"
    path_b = tmp_path / "b_history.json"
    path_a.write_text(json.dumps(history_a), encoding="utf-8")
    path_b.write_text(json.dumps(history_b), encoding="utf-8")
    diffs, _ = compare_ddqn_checkpoints.compare_history(path_a, path_b)
    assert any("history.rewards" in d for d in diffs)


def test_cli_exit_code_match(tmp_path):
    ckpt = _sample_checkpoint()
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    torch.save(ckpt, path_a)
    torch.save(ckpt, path_b)
    assert compare_ddqn_checkpoints.main([str(path_a), str(path_b)]) == 0


def test_cli_exit_code_mismatch(tmp_path):
    ckpt_a = _sample_checkpoint()
    ckpt_b = _sample_checkpoint()
    ckpt_b["epsilon"] = 0.1
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    torch.save(ckpt_a, path_a)
    torch.save(ckpt_b, path_b)
    assert compare_ddqn_checkpoints.main([str(path_a), str(path_b)]) == 1
