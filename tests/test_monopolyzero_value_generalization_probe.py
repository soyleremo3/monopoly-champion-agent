"""Tests for scripts/monopolyzero_value_generalization_probe.py: config,
quantile-spread sampling, the probabilistic net-worth-leader baseline and
its TRAIN+SELECTION-only temperature fit, the game-block (not state-level)
paired bootstrap against a hand-verified case, bucket provenance, and
structural checks that TEST is only ever touched once, after the learning
curve / temperature fit are done. Does not run the actual 96-game probe
(see the experiment log for that).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_value_generalization_probe.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_value_generalization_probe", SCRIPT)
gen_module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_value_generalization_probe"] = gen_module
_spec.loader.exec_module(gen_module)


def test_config_matches_task_spec():
    assert gen_module.GAMES_TOTAL == 96
    assert gen_module.SELF_PLAY_SEEDS == tuple(range(42100, 42196))
    assert gen_module.TRAIN_SEEDS == tuple(range(42100, 42164))
    assert gen_module.SELECTION_SEEDS == tuple(range(42164, 42180))
    assert gen_module.TEST_SEEDS == tuple(range(42180, 42196))
    assert len(gen_module.TRAIN_SEEDS) == 64
    assert len(gen_module.SELECTION_SEEDS) == 16
    assert len(gen_module.TEST_SEEDS) == 16
    assert set(gen_module.TRAIN_SEEDS).isdisjoint(gen_module.SELECTION_SEEDS)
    assert set(gen_module.TRAIN_SEEDS).isdisjoint(gen_module.TEST_SEEDS)
    assert set(gen_module.SELECTION_SEEDS).isdisjoint(gen_module.TEST_SEEDS)
    assert gen_module.SAMPLES_PER_CELL == 3
    assert gen_module.LEARNING_CURVE_TRAIN_GAME_COUNTS == (16, 32, 64)
    assert gen_module.ROUND_BUCKETS == gen_module.probe_v1.ROUND_BUCKETS


def test_seeds_registered_dev_and_do_not_touch_promotion_final_blind():
    import evaluation_protocol as ep

    for seed in gen_module.SELF_PLAY_SEEDS:
        assert seed in ep.DEV_SEEDS, f"seed {seed} not registered as DEV"
    ep.require_seed_scope(gen_module.SELF_PLAY_SEEDS, ep.SEED_CLASS_DEV, context="test")
    assert ep.DEV_SEEDS.isdisjoint(ep.PROMOTION_SEEDS)
    assert ep.DEV_SEEDS.isdisjoint(ep.FINAL_BLIND_SEEDS)


def test_does_not_reuse_020_seed_range():
    seeds_020 = set(gen_module.probe_v1.SELF_PLAY_SEEDS)
    seeds_021 = set(gen_module.SELF_PLAY_SEEDS)
    assert seeds_020.isdisjoint(seeds_021)


# ── quantile-spread sampling ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "count,n,expected",
    [
        (0, 3, []),
        (1, 3, [0]),
        (2, 3, [0, 1]),
        (3, 3, [0, 1, 2]),
        (5, 3, [0, 2, 4]),
        (7, 3, [0, 3, 6]),
        (10, 1, [5]),
        (10, 3, [0, 4, 9]),
    ],
)
def test_quantile_indices_known_cases(count, n, expected):
    assert gen_module.quantile_indices(count, n) == expected


def test_quantile_indices_spreads_across_full_range_not_clustered_at_start():
    result = gen_module.quantile_indices(100, 3)
    assert result[0] == 0
    assert result[-1] == 99
    assert result != [0, 1, 2]  # the old 020 "first N" behavior


def test_quantile_indices_never_exceeds_available_count():
    for count in range(0, 10):
        result = gen_module.quantile_indices(count, 3)
        assert len(result) <= max(count, 0)
        assert all(0 <= i < count for i in result)


# ── probabilistic net-worth-leader baseline ─────────────────────────────


def test_probabilistic_leader_probs_sums_to_one_and_favors_higher_net_worth():
    import numpy as np

    net_worth = [[4000.0, 1000.0, 1000.0, 1000.0]]
    probs = gen_module.probabilistic_leader_probs(net_worth, temperature=1000.0)
    assert probs.shape == (1, 4)
    assert np.sum(probs) == pytest.approx(1.0)
    assert probs[0][0] > probs[0][1]
    assert probs[0][1] == pytest.approx(probs[0][2])


def test_probabilistic_leader_probs_uniform_when_net_worth_tied():
    probs = gen_module.probabilistic_leader_probs([[500.0, 500.0, 500.0, 500.0]], temperature=1000.0)
    assert probs[0] == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_fit_probabilistic_leader_temperature_picks_grid_minimum():
    """Construct data where a low temperature (sharp) is clearly better
    than a high temperature (near-uniform): the true winner is always the
    net-worth leader by a wide margin, so a sharper distribution scores
    lower cross-entropy."""
    net_worth = [[10000.0, 100.0, 100.0, 100.0]] * 20
    true_classes = [0] * 20
    best_temp, grid_results = gen_module.fit_probabilistic_leader_temperature(
        net_worth, true_classes, grid=(50.0, 5000.0, 50000.0)
    )
    assert best_temp == 50.0  # sharpest available temperature wins when the leader always wins big
    assert len(grid_results) == 3
    ces = {row["temperature"]: row["cross_entropy"] for row in grid_results}
    assert ces[50.0] < ces[5000.0] < ces[50000.0]


# ── game-block paired bootstrap ─────────────────────────────────────────


def test_game_block_bootstrap_hand_verified_point_estimate():
    """2 games: in game 1, learned is perfect (CE~0) and baseline is
    wrong; in game 2, both are identical. The pooled point diff must equal
    the direct computation (not an approximation) - this only checks the
    'point' value, which does not involve resampling at all."""
    per_game = {
        1: [
            {"true_class": 0, "learned_probs": [0.97, 0.01, 0.01, 0.01], "baseline_probs": [0.25, 0.25, 0.25, 0.25]},
        ],
        2: [
            {"true_class": 1, "learned_probs": [0.25, 0.25, 0.25, 0.25], "baseline_probs": [0.25, 0.25, 0.25, 0.25]},
        ],
    }
    result = gen_module.game_block_bootstrap_metric_diff(per_game, n_resamples=100, bootstrap_seed=0)
    # direct pooled computation for comparison
    import numpy as np

    learned = np.array([[0.97, 0.01, 0.01, 0.01], [0.25, 0.25, 0.25, 0.25]])
    baseline = np.array([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])
    true_classes = np.array([0, 1])
    expected_ce_diff = gen_module.probe_v1.cross_entropy(learned, true_classes) - gen_module.probe_v1.cross_entropy(baseline, true_classes)
    assert result["cross_entropy_diff"]["point"] == pytest.approx(expected_ce_diff)
    assert result["n_games"] == 2


def test_game_block_bootstrap_is_deterministic():
    per_game = {
        seed: [
            {"true_class": seed % 4, "learned_probs": [0.4, 0.3, 0.2, 0.1], "baseline_probs": [0.25, 0.25, 0.25, 0.25]}
            for _ in range(3)
        ]
        for seed in range(1, 21)
    }
    result_a = gen_module.game_block_bootstrap_metric_diff(per_game, n_resamples=300, bootstrap_seed=5)
    result_b = gen_module.game_block_bootstrap_metric_diff(per_game, n_resamples=300, bootstrap_seed=5)
    assert result_a == result_b


def test_game_block_bootstrap_empty_returns_none_stats():
    result = gen_module.game_block_bootstrap_metric_diff({}, n_resamples=10, bootstrap_seed=0)
    assert result["n_games"] == 0
    assert result["cross_entropy_diff"] == {"point": None, "ci_95": None}


def test_game_block_bootstrap_resamples_games_not_states():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "rng.integers(0, n_games, size=n_games)" in source


# ── bucket provenance ────────────────────────────────────────────────────


def test_bucket_provenance_median_min_max():
    records = [
        {"bucket": "1-25", "round": 5, "seed": 1},
        {"bucket": "1-25", "round": 10, "seed": 1},
        {"bucket": "1-25", "round": 20, "seed": 2},
        {"bucket": "26-50", "round": 30, "seed": 1},
    ]
    result = gen_module._bucket_provenance(records)
    assert result["1-25"]["count"] == 3
    assert result["1-25"]["unique_games"] == 2
    assert result["1-25"]["min_round"] == 5
    assert result["1-25"]["max_round"] == 20
    assert result["1-25"]["median_round"] == 10
    assert result["26-50"]["count"] == 1
    assert result["51-100"]["count"] == 0
    assert result["51-100"]["median_round"] is None


# ── checkpoint integrity gate (delegates to probe_v1) ──────────────────


def test_verify_checkpoint_raises_when_missing(tmp_path):
    with pytest.raises(SystemExit, match="missing checkpoint"):
        gen_module.verify_checkpoint(tmp_path / "nope.pt", "a" * 64)


def test_verify_checkpoint_passes_when_hash_matches(tmp_path):
    import hashlib

    checkpoint = tmp_path / "checkpoint.pt"
    content = b"some checkpoint bytes"
    checkpoint.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert gen_module.verify_checkpoint(checkpoint, expected) == expected


# ── structural / CLI checks ──────────────────────────────────────────────


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
    import_lines = [line for line in source.splitlines() if line.strip().startswith(("import ", "from "))]
    forbidden = ("monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert hits == [], f"found forbidden ASU-coupled module import(s): {hits}"
    assert not any("ASU_FROZEN_TEACHER" in line for line in import_lines)


def test_reuses_v1_architecture_instead_of_duplicating():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import monopolyzero_value_learnability_probe as probe_v1" in source
    assert "probe_v1.train_value_probe(" in source
    assert "probe_v1.build_value_probe" not in source  # never redefines the architecture, only reuses train_value_probe


def test_no_puct_or_fixed_agents():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "MaxNPUCT" not in source
    assert "LocalFixedPolicy" not in source
    assert "FP_AGENT_CLASSES" not in source
    assert "build_local_policy_only" in source


def test_policy_network_never_trained():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "local_training_update(" not in source


def test_test_split_untouched_before_learning_curve_and_temperature_fit():
    """TEST arrays (test_x/test_y/test_nw) must not appear anywhere before
    the explicit '# TEST: touched exactly once' marker - guards against
    accidentally leaking TEST into the learning curve loop or the
    temperature fit, which both must only see TRAIN/SELECTION."""
    source = SCRIPT.read_text(encoding="utf-8")
    marker = "TEST: touched exactly once"
    assert marker in source
    marker_index = source.index(marker)
    before_marker = source[:marker_index]
    # test_x/test_y/test_nw are only ever *assigned* before the marker (via
    # _arrays(test_records)) - they must not be *read* before it.
    assignment_line = "test_x, test_y, test_leader_y, test_nw, test_buckets = _arrays(test_records)"
    assert assignment_line in before_marker
    after_assignment = before_marker[before_marker.index(assignment_line) + len(assignment_line):]
    for forbidden in ("test_x", "test_y", "test_nw", "test_leader_y", "test_buckets"):
        assert forbidden not in after_assignment, f"{forbidden} used before TEST is meant to be touched"


def test_temperature_fit_uses_train_and_selection_only():
    source = SCRIPT.read_text(encoding="utf-8")
    fit_call_index = source.index("fit_probabilistic_leader_temperature(pooled_nw, pooled_y)")
    nearby = source[max(0, fit_call_index - 300):fit_call_index]
    assert "concatenate([train_nw, selection_nw]" in nearby
    assert "test_nw" not in nearby


def test_learning_curve_evaluated_on_selection_not_test():
    source = SCRIPT.read_text(encoding="utf-8")
    loop_start = source.index("for game_count in LEARNING_CURVE_TRAIN_GAME_COUNTS")
    loop_end = source.index("# ── TEST:")
    loop_body = source[loop_start:loop_end]
    assert "test_x" not in loop_body
    assert "test_y" not in loop_body
    assert "selection_x" in loop_body
    assert "selection_y" in loop_body


def test_no_promotion_or_go_kill_boolean_in_source():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "_recommended" not in source
    for forbidden_key in ('"promote":', '"promotion_recommended":', '"go_kill":', '"verdict":', '"kill_recommended":'):
        assert forbidden_key not in source
