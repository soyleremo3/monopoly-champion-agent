"""Focused tests for scripts/monopolyzero_horizon_diagnostic.py: the
checkpoint-integrity gates (two checkpoints), the pure aggregation/math
helpers (mean/median/final-rank/net-worth-snapshot/agreement/fallback
summaries) over fake game-report dicts, the state-encoding-ablation
isolation check (including its fail-loudly path when isolation is broken),
and config/import-graph checks. Does not run the actual 32-game diagnostic
(see the experiment log for that result instead).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_horizon_diagnostic.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_horizon_diagnostic", SCRIPT)
diag_module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_horizon_diagnostic"] = diag_module
_spec.loader.exec_module(diag_module)


def test_config_matches_task_spec():
    assert diag_module.BASELINE_CHECKPOINT_SHA256 == (
        "22ae2abea60478355f61ce0404a31de3b52fb97769d4f99f7e04f4c673629370"
    )
    assert diag_module.UPDATE500_CHECKPOINT_SHA256 == (
        "152c0a0f6136d1fc91e74973ac245b2f72774694c424d2a48854514ed2848383"
    )
    assert diag_module.SIMULATIONS == 4
    assert diag_module.MAX_DEPTH == 16
    assert diag_module.MAX_ROUNDS == 200
    assert diag_module.ROUND_SNAPSHOT == 50
    assert diag_module.GAMES_PER_CATEGORY == 16
    assert diag_module.ABLATION_TARGET_STATES == 200
    assert diag_module.ABLATION_MIN_ROUND == 1
    assert diag_module.ABLATION_MAX_ROUND == 50
    assert diag_module.ROUND_MAX_ROUNDS_STATE_INDEX == 278


def test_seeds_are_fresh_and_non_overlapping():
    self_play = set(diag_module.SELF_PLAY_SEEDS)
    vs_fixed = set(diag_module.VS_FIXED_SEEDS)
    assert len(self_play) == 16
    assert len(vs_fixed) == 16
    assert self_play.isdisjoint(vs_fixed)
    prior_pools = (
        set(range(10000, 10016)) | set(range(20000, 20016))
        | set(range(10000, 10010)) | set(range(31000, 31005))
        | set(range(32000, 32010))
    )
    assert self_play.isdisjoint(prior_pools)
    assert vs_fixed.isdisjoint(prior_pools)


def test_checkpoint_paths_point_at_expected_artifacts():
    assert diag_module.BASELINE_CHECKPOINT.name == "baseline_pretraining.pt"
    assert diag_module.UPDATE500_CHECKPOINT.name == "trained_updates_500.pt"


# ── checkpoint integrity gate (parameterized over both checkpoints) ────────


def test_verify_checkpoint_raises_when_missing(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        diag_module.verify_checkpoint(tmp_path / "nope.pt", "a" * 64)
    assert "missing checkpoint" in str(excinfo.value)


def test_verify_checkpoint_raises_on_sha_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"some bytes")
    with pytest.raises(SystemExit) as excinfo:
        diag_module.verify_checkpoint(checkpoint, "a" * 64)
    assert "SHA-256 mismatch" in str(excinfo.value)


def test_verify_checkpoint_passes_when_hash_matches(tmp_path):
    import hashlib

    checkpoint = tmp_path / "checkpoint.pt"
    content = b"some checkpoint bytes"
    checkpoint.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert diag_module.verify_checkpoint(checkpoint, expected) == expected


# ── pure math helpers ───────────────────────────────────────────────────────


def test_mean_and_median_basic():
    assert diag_module._mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert diag_module._mean([]) is None
    assert diag_module._median([3.0, 1.0, 2.0]) == 2.0
    assert diag_module._median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert diag_module._median([]) is None


def test_final_rank():
    final_net_worth = (1000.0, 4000.0, 2000.0, 500.0)
    assert diag_module._final_rank(final_net_worth, 1) == 1
    assert diag_module._final_rank(final_net_worth, 2) == 2
    assert diag_module._final_rank(final_net_worth, 0) == 3
    assert diag_module._final_rank(final_net_worth, 3) == 4


class _FakePlayer:
    def __init__(self, net_worth_value: float):
        self._nw = net_worth_value

    def net_worth(self) -> float:
        return self._nw


class _FakeEnvForSnapshot:
    def __init__(self, round_: int, net_worths: tuple):
        self.round = round_
        self.players = [_FakePlayer(nw) for nw in net_worths]


class _FakeGameForSnapshot:
    def __init__(self, round_: int, net_worths: tuple):
        self.env = _FakeEnvForSnapshot(round_, net_worths)


def test_net_worth_snapshot_leader_and_margin():
    game = _FakeGameForSnapshot(50, (1000.0, 4000.0, 2000.0, 500.0))
    snap = diag_module._net_worth_snapshot(game)
    assert snap["round"] == 50
    assert snap["leader_seat"] == 1
    assert snap["margin"] == pytest.approx(2000.0)
    assert snap["net_worth"] == [1000.0, 4000.0, 2000.0, 500.0]


def test_net_worth_snapshot_tie_uses_first_seat_by_stable_sort():
    game = _FakeGameForSnapshot(50, (3000.0, 3000.0, 1000.0, 0.0))
    snap = diag_module._net_worth_snapshot(game)
    assert snap["leader_seat"] == 0
    assert snap["margin"] == pytest.approx(0.0)


# ── _agreement_stats / _summarize_horizon / _summarize_fallbacks ──────────


def _fake_game_report(*, category, winner, leader_seat, margin, final_net_worth, completed=True, round50=True):
    return {
        "seed": 1,
        "category": category,
        "completed": completed,
        "winner": winner,
        "final_round": 200,
        "final_net_worth": final_net_worth,
        "round50_snapshot": (
            {"round": 50, "net_worth": list(final_net_worth), "leader_seat": leader_seat, "margin": margin}
            if round50 else None
        ),
        "fixed_fallbacks": {},
    }


def test_agreement_stats_basic():
    games = [
        _fake_game_report(category="self_play", winner=1, leader_seat=1, margin=100.0, final_net_worth=(0, 1, 0, 0)),
        _fake_game_report(category="self_play", winner=2, leader_seat=1, margin=200.0, final_net_worth=(0, 1, 2, 0)),
    ]
    stats = diag_module._agreement_stats(games)
    assert stats == {"games": 2, "agreements": 1, "agreement_rate": pytest.approx(0.5)}


def test_agreement_stats_empty():
    assert diag_module._agreement_stats([]) == {"games": 0, "agreements": 0, "agreement_rate": None}


def test_summarize_horizon_splits_self_play_and_vs_fixed_and_excludes_unfinished():
    games = [
        _fake_game_report(category="self_play", winner=0, leader_seat=0, margin=100.0, final_net_worth=(4, 3, 2, 1)),
        _fake_game_report(category="vs_fixed", winner=1, leader_seat=2, margin=300.0, final_net_worth=(1, 4, 3, 2)),
        _fake_game_report(category="vs_fixed", winner=None, leader_seat=0, margin=50.0, final_net_worth=(0, 0, 0, 0), round50=False),
    ]
    summary = diag_module._summarize_horizon(games)
    assert summary["overall"]["games"] == 2
    assert summary["self_play"]["games"] == 1
    assert summary["self_play"]["agreement_rate"] == pytest.approx(1.0)
    assert summary["vs_fixed"]["games"] == 1
    assert summary["vs_fixed"]["agreement_rate"] == pytest.approx(0.0)
    assert summary["games_finished_before_round_50"] == 1
    assert summary["round50_leader_final_rank_distribution"] == {"1": 1, "2": 1}


def test_summarize_horizon_margin_buckets_and_agreement_split():
    games = [
        _fake_game_report(category="self_play", winner=0, leader_seat=0, margin=100.0, final_net_worth=(4, 3, 2, 1)),
        _fake_game_report(category="self_play", winner=3, leader_seat=1, margin=3000.0, final_net_worth=(1, 2, 3, 4)),
    ]
    summary = diag_module._summarize_horizon(games)
    assert summary["margin_mean_when_agree"] == pytest.approx(100.0)
    assert summary["margin_mean_when_disagree"] == pytest.approx(3000.0)
    buckets = {b["margin_range"]: b for b in summary["agreement_rate_by_margin_bucket"]}
    assert buckets["[0.0,500.0)"]["games"] == 1
    assert buckets["[2000.0,inf)"]["games"] == 1


def test_summarize_fallbacks():
    games = [
        {"category": "self_play", "fixed_fallbacks": {}},
        {"category": "vs_fixed", "fixed_fallbacks": {"TheHoarder": 2, "TheDealMaker": 1}},
        {"category": "vs_fixed", "fixed_fallbacks": {"TheGambler": 3}},
    ]
    totals = diag_module._summarize_fallbacks(games)
    assert totals == {"self_play": 0, "vs_fixed": 6, "total": 6}


# ── state-encoding ablation: isolation check + fail-loudly path ───────────


class _FakeAblationEnv:
    """Mimics the subset of MonopolyEnv used by _run_state_encoding_ablation:
    get_allowed_actions, _get_state, and a plain mutable max_rounds
    attribute. copy.deepcopy works on it out of the box (plain attributes)."""

    def __init__(self, round_: int, max_rounds: int = 200, extra_index: int | None = None):
        self.round = round_
        self.max_rounds = max_rounds
        self._extra_index = extra_index  # simulates a broken isolation when set

    def get_allowed_actions(self, actor):
        return (0, 1)

    def _get_state(self, actor):
        import numpy as np

        state = np.zeros(300, dtype=np.float32)
        state[diag_module.ROUND_MAX_ROUNDS_STATE_INDEX] = min(self.round / max(self.max_rounds, 1), 1.0)
        if self._extra_index is not None:
            state[self._extra_index] = float(self.max_rounds)
        return state


class _FakeAblationModel:
    def predict(self, state, legal, actor):
        import numpy as np

        bias = float(state[diag_module.ROUND_MAX_ROUNDS_STATE_INDEX])
        priors = {legal[0]: 0.5 + bias * 0.4, legal[1]: 0.5 - bias * 0.4}
        value = np.array([bias, 1.0 - bias, 0.0, 0.0], dtype=np.float32)
        return priors, value


def test_state_encoding_ablation_isolates_single_index():
    diag_module.common.ensure_reference_on_path()
    snapshots = [
        {"seed": 1, "category": "self_play", "recording_seat": 0, "round": 10, "env_clone": _FakeAblationEnv(10, 200)},
        {"seed": 2, "category": "vs_fixed", "recording_seat": 1, "round": 25, "env_clone": _FakeAblationEnv(25, 200)},
    ]
    result = diag_module._run_state_encoding_ablation(_FakeAblationModel(), snapshots, "fake_checkpoint")
    assert result["states_used"] == 2
    assert result["state_vector_diff_indices_union"] == [diag_module.ROUND_MAX_ROUNDS_STATE_INDEX]
    assert result["action_disagreement_rate"] is not None
    assert result["policy_tv_distance_mean"] is not None
    assert result["value_mean_abs_delta_mean"] is not None
    assert result["states_with_no_index_diff"] == 0


def test_state_encoding_ablation_raises_when_isolation_is_broken():
    diag_module.common.ensure_reference_on_path()
    snapshots = [
        {"seed": 1, "category": "self_play", "recording_seat": 0, "round": 10, "env_clone": _FakeAblationEnv(10, 200, extra_index=5)},
    ]
    with pytest.raises(RuntimeError, match="isolation broken"):
        diag_module._run_state_encoding_ablation(_FakeAblationModel(), snapshots, "broken")


def test_state_encoding_ablation_empty_snapshots_returns_none_stats():
    diag_module.common.ensure_reference_on_path()
    result = diag_module._run_state_encoding_ablation(_FakeAblationModel(), [], "empty")
    assert result["states_used"] == 0
    assert result["action_disagreement_rate"] is None
    assert result["policy_tv_distance_mean"] is None
    assert result["value_mean_abs_delta_mean"] is None


# ── CLI / import-graph / structure checks ─────────────────────────────────


def test_cli_fails_fast_without_pinned_hash_seed():
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "PYTHONHASHSEED=0" in result.stderr


def test_does_not_import_adapters_arena_or_training():
    source = SCRIPT.read_text(encoding="utf-8")
    import_lines = [
        line for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert hits == [], f"found forbidden ASU-coupled module import(s): {hits}"
    assert not any("ASU_FROZEN_TEACHER" in line for line in import_lines)


def test_uses_shared_monopolyzero_common_module():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import monopolyzero_common as common" in source


def test_self_play_true_matches_013_training_data_recipe():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "self_play=True" in source


def test_no_training_update_call():
    """This is a pure inference diagnostic - it must never call the
    training-update step."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "local_training_update" not in source


def test_no_arbitrary_go_kill_verdict_computed():
    """Task instruction: measure and report only, no invented threshold -
    the script must not compute a kill/go boolean verdict itself."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "_recommended" not in source
    assert "kill_search" not in source.lower()


def test_ablation_reused_across_both_checkpoints_from_same_snapshots():
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("_run_state_encoding_ablation(") >= 2
    assert "ablation_snapshots" in source
