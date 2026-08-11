"""Tests for scripts/colab_shard_runner.py: config/seed reuse, the pure
resume/games-per-seed helpers, the benchmark projection math and the
resumable shard loop (both driven by a monkeypatched play_local_game - no
real engine cost), and one tiny REAL-engine smoke run (2 seeds) to catch
integration bugs the mocked tests can't see. Does not run anything
resembling a real/large shard.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "colab_shard_runner.py"

_spec = importlib.util.spec_from_file_location("colab_shard_runner", SCRIPT)
runner_module = importlib.util.module_from_spec(_spec)
sys.modules["colab_shard_runner"] = runner_module
_spec.loader.exec_module(runner_module)


# ── config reuse ──────────────────────────────────────────────────────────


def test_reuses_024_config_not_redefined():
    assert runner_module.MAX_ROUNDS == runner_module.decomp.MAX_ROUNDS == 200
    assert runner_module.NUM_SEATS == runner_module.decomp.NUM_SEATS == 4
    assert runner_module.CHECKPOINT_PATH == runner_module.decomp.CHECKPOINT_PATH
    assert runner_module.BASELINE_CHECKPOINT_SHA256 == runner_module.decomp.BASELINE_CHECKPOINT_SHA256
    source = SCRIPT.read_text(encoding="utf-8")
    assert "def _arm_summary(" not in source, "should reuse audit_v1._arm_summary, not redefine it"
    assert "def verify_baseline_checkpoint(" not in source


def test_arms_match_decomp_constants():
    assert set(runner_module.ARMS) == {
        runner_module.decomp.ARM_POLICY_ONLY, runner_module.decomp.ARM_BUY_ONLY,
        runner_module.decomp.ARM_TRADE_ONLY, runner_module.decomp.ARM_BOTH,
    }


# ── is_self_play_optimized / games_per_seed ───────────────────────────────


@pytest.mark.parametrize(
    "context,arm,expected",
    [
        ("repaired", "both", True),
        ("crippled", "both", False),
        ("repaired", "buy_only", False),
        ("repaired", "trade_only", False),
        ("crippled", "policy_only", False),
    ],
)
def test_is_self_play_optimized(context, arm, expected):
    assert runner_module.is_self_play_optimized(context, arm) is expected


@pytest.mark.parametrize(
    "context,arm,expected",
    [("repaired", "both", 1), ("crippled", "both", 4), ("crippled", "buy_only", 4), ("repaired", "trade_only", 4)],
)
def test_games_per_seed(context, arm, expected):
    assert runner_module.games_per_seed(context, arm) == expected


# ── _read_completed_seeds ──────────────────────────────────────────────────


def test_read_completed_seeds_missing_file_returns_empty(tmp_path):
    assert runner_module._read_completed_seeds(tmp_path / "nope.jsonl", 4) == set()


def test_read_completed_seeds_counts_correctly(tmp_path):
    jsonl_path = tmp_path / "per_game.jsonl"
    lines = [
        {"seed": 1, "focus_seat": 0}, {"seed": 1, "focus_seat": 1},
        {"seed": 1, "focus_seat": 2}, {"seed": 1, "focus_seat": 3},  # seed 1: complete (4)
        {"seed": 2, "focus_seat": 0}, {"seed": 2, "focus_seat": 1},  # seed 2: incomplete (2 of 4)
    ]
    jsonl_path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    assert runner_module._read_completed_seeds(jsonl_path, required_games_per_seed=4) == {1}


def test_read_completed_seeds_self_play_mode_one_per_seed(tmp_path):
    jsonl_path = tmp_path / "per_game.jsonl"
    lines = [{"seed": 5, "focus_seat": 0}]
    jsonl_path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    assert runner_module._read_completed_seeds(jsonl_path, required_games_per_seed=1) == {5}


# ── run_benchmark (monkeypatched engine, no real games) ────────────────────


def _fake_outcome(**overrides):
    base = dict(
        completed=True, winner=0, decisions=10, final_round=5,
        final_net_worth=(100.0, 200.0, 300.0, 400.0),
        illegal_actions=0, crashed=False, search_latencies_s=[0.001],
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


class _FakeModel:
    def predict(self, state, legal_actions, actor_id):
        legal = tuple(legal_actions)
        priors = {action: 0.1 for action in legal}
        priors[legal[-1]] = 0.9
        return priors, [0.5]


def test_run_benchmark_reports_sec_per_game_and_projections(monkeypatch):
    runner_module.common.ensure_reference_on_path()
    monkeypatch.setattr(runner_module.common, "play_local_game", lambda **kwargs: _fake_outcome())

    result = runner_module.run_benchmark(
        n_games=8, seed_start=100, context="crippled", arm="policy_only", model=_FakeModel(),
    )
    assert result["games_completed"] == 8  # 2 seeds x 4 games (rotation mode)
    assert result["seeds_used"] == 2
    assert result["sec_per_game"] is not None
    assert result["projected_seconds"]["100"] == pytest.approx(result["sec_per_game"] * 100)
    assert result["projected_seconds"]["500"] == pytest.approx(result["sec_per_game"] * 500)
    assert result["projected_seconds"]["1000"] == pytest.approx(result["sec_per_game"] * 1000)


def test_run_benchmark_self_play_optimized_uses_fewer_seeds(monkeypatch):
    runner_module.common.ensure_reference_on_path()
    monkeypatch.setattr(runner_module.common, "play_local_game", lambda **kwargs: _fake_outcome())

    result = runner_module.run_benchmark(
        n_games=8, seed_start=100, context="repaired", arm="both", model=_FakeModel(),
    )
    assert result["games_completed"] == 8
    assert result["seeds_used"] == 2  # self-play yields 4 seat-records per seed, so 8 games needs only 2 seeds
    assert result["self_play_optimized"] is True


def test_run_benchmark_raises_on_illegal(monkeypatch):
    runner_module.common.ensure_reference_on_path()
    monkeypatch.setattr(runner_module.common, "play_local_game", lambda **kwargs: _fake_outcome(illegal_actions=1))
    with pytest.raises(RuntimeError, match="illegal/crash"):
        runner_module.run_benchmark(n_games=4, seed_start=100, context="crippled", arm="policy_only", model=_FakeModel())


# ── run_shard (monkeypatched engine): resume + fsync-per-seed + STOP ──────


def test_run_shard_fresh_then_resume(monkeypatch, tmp_path):
    runner_module.common.ensure_reference_on_path()
    monkeypatch.setattr(runner_module.common, "play_local_game", lambda **kwargs: _fake_outcome())

    output_dir = tmp_path / "shard_a"
    common_kwargs = dict(
        seed_start=44100, seed_count=2, context="crippled", arm="policy_only", output_dir=output_dir,
        model=_FakeModel(), git_head_sha="a" * 40, checkpoint_sha256="b" * 64, submodule_sha="c" * 40,
    )

    summary = runner_module.run_shard(resume=False, **common_kwargs)
    assert summary["status"] == "OK"
    assert summary["seeds_completed_this_run"] == 2
    assert (output_dir / "per_game.jsonl").is_file()
    assert (output_dir / "metadata.json").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "run_log.txt").is_file()

    lines = (output_dir / "per_game.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8  # 2 seeds x 4 rotation games

    # Resuming an already-fully-complete shard should skip everything.
    summary2 = runner_module.run_shard(resume=True, **common_kwargs)
    assert summary2["status"] == "OK"
    assert summary2["seeds_completed_this_run"] == 0
    assert summary2["seeds_already_complete_before_this_run"] == 2
    lines_after = (output_dir / "per_game.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_after) == 8  # unchanged - nothing re-played


def test_run_shard_without_resume_refuses_to_overwrite_nonempty_jsonl(monkeypatch, tmp_path):
    runner_module.common.ensure_reference_on_path()
    monkeypatch.setattr(runner_module.common, "play_local_game", lambda **kwargs: _fake_outcome())

    output_dir = tmp_path / "shard_b"
    kwargs = dict(
        seed_start=44200, seed_count=1, context="crippled", arm="policy_only", output_dir=output_dir,
        model=_FakeModel(), git_head_sha="a" * 40, checkpoint_sha256="b" * 64, submodule_sha="c" * 40,
    )
    runner_module.run_shard(resume=False, **kwargs)
    with pytest.raises(SystemExit, match="already exists"):
        runner_module.run_shard(resume=False, **kwargs)


def test_run_shard_resume_refuses_mismatched_config(monkeypatch, tmp_path):
    runner_module.common.ensure_reference_on_path()
    monkeypatch.setattr(runner_module.common, "play_local_game", lambda **kwargs: _fake_outcome())

    output_dir = tmp_path / "shard_c"
    runner_module.run_shard(
        seed_start=44300, seed_count=1, context="crippled", arm="policy_only", output_dir=output_dir,
        resume=False, model=_FakeModel(), git_head_sha="a" * 40, checkpoint_sha256="b" * 64, submodule_sha="c" * 40,
    )
    with pytest.raises(RuntimeError, match="config differs"):
        runner_module.run_shard(
            seed_start=44300, seed_count=1, context="crippled", arm="buy_only", output_dir=output_dir,  # different arm
            resume=True, model=_FakeModel(), git_head_sha="a" * 40, checkpoint_sha256="b" * 64, submodule_sha="c" * 40,
        )


def test_run_shard_stops_on_illegal_action(monkeypatch, tmp_path):
    runner_module.common.ensure_reference_on_path()
    calls = {"n": 0}

    def fake_play(**kwargs):
        calls["n"] += 1
        return _fake_outcome(illegal_actions=1) if calls["n"] > 4 else _fake_outcome()

    monkeypatch.setattr(runner_module.common, "play_local_game", fake_play)

    output_dir = tmp_path / "shard_d"
    summary = runner_module.run_shard(
        seed_start=44400, seed_count=3, context="crippled", arm="policy_only", output_dir=output_dir,
        resume=False, model=_FakeModel(), git_head_sha="a" * 40, checkpoint_sha256="b" * 64, submodule_sha="c" * 40,
    )
    assert summary["status"] == "FAILED"
    # seed 44400 completes clean (4 calls); seed 44401's first game trips illegal, but a
    # seed's 4 rotation games are written as one atomic batch (all-or-none) before the
    # stop check runs, so seed 44401's 4 games are still written, then the shard stops
    # before starting seed 44402.
    lines = (output_dir / "per_game.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8


# ── argument parser ─────────────────────────────────────────────────────


def test_arg_parser_requires_core_flags():
    parser = runner_module.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_arg_parser_accepts_benchmark_without_output_dir():
    parser = runner_module.build_arg_parser()
    args = parser.parse_args(
        ["--seed-start", "44000", "--seed-count", "1", "--arm", "both", "--context", "crippled", "--benchmark-games", "20"]
    )
    assert args.output_dir is None
    assert args.benchmark_games == 20


def test_arg_parser_seed_count_optional_at_parse_time():
    """--seed-count is only required in shard mode - main() enforces that,
    not argparse - so benchmark-only invocations don't need it."""
    parser = runner_module.build_arg_parser()
    args = parser.parse_args(
        ["--seed-start", "44000", "--arm", "both", "--context", "crippled", "--benchmark-games", "20"]
    )
    assert args.seed_count is None


def test_main_requires_seed_count_outside_benchmark_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["colab_shard_runner.py"])
    with pytest.raises(SystemExit, match="--seed-count is required"):
        runner_module.main(["--seed-start", "44000", "--arm", "both", "--context", "crippled"])


# ── tiny REAL-engine smoke run (not a "big run") ──────────────────────────


def test_real_engine_smoke_two_seeds(tmp_path):
    """Genuinely plays 2 seeds (8 games) against the real engine/checkpoint
    - catches integration bugs (wrong kind handling, wrong attribute names)
    the monkeypatched tests above structurally cannot see. Small and fast,
    not a benchmark/shard run."""
    runner_module.common.ensure_reference_on_path()
    import random

    import numpy as np
    import torch

    from monopoly_bench.model import MonopolyZeroNet

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    if not runner_module.CHECKPOINT_PATH.is_file():
        pytest.skip("baseline_pretraining.pt not present in this environment")

    model = MonopolyZeroNet.load_inference(runner_module.CHECKPOINT_PATH)
    model.eval()

    output_dir = tmp_path / "smoke_shard"
    summary = runner_module.run_shard(
        seed_start=44000, seed_count=2, context="crippled", arm="buy_only", output_dir=output_dir,
        resume=False, model=model, git_head_sha="a" * 40, checkpoint_sha256="b" * 64, submodule_sha="c" * 40,
    )
    assert summary["status"] == "OK"
    assert summary["games"] == 8
    lines = (output_dir / "per_game.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8
    for line in lines:
        record = json.loads(line)
        assert record["seed"] in (44000, 44001)
        assert 0 <= record["focus_seat"] <= 3
