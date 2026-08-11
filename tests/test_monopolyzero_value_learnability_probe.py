"""Tests for scripts/monopolyzero_value_learnability_probe.py: checkpoint
integrity gate, round-bucket boundary logic, the pure metric helpers
(cross-entropy/Brier/accuracy/confusion/calibration/class-balance/baseline
probs) against hand-verified known values, the ValueProbe architecture and
its deterministic training loop on a tiny synthetic dataset, the nested
learning-curve seed-subset helper, and structural checks (no PUCT, no
fixed agents, DEV-scoped seeds, no promotion boolean, policy untouched).
Does not run the actual 64-game probe (see the experiment log for that).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_value_learnability_probe.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_value_learnability_probe", SCRIPT)
probe_module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_value_learnability_probe"] = probe_module
_spec.loader.exec_module(probe_module)


def test_config_matches_task_spec():
    assert probe_module.BASELINE_CHECKPOINT_SHA256 == (
        "22ae2abea60478355f61ce0404a31de3b52fb97769d4f99f7e04f4c673629370"
    )
    assert probe_module.MAX_ROUNDS == 200
    assert probe_module.GAMES_TOTAL == 64
    assert probe_module.SELF_PLAY_SEEDS == tuple(range(42000, 42064))
    assert probe_module.TRAIN_SEEDS == tuple(range(42000, 42048))
    assert probe_module.VALIDATION_SEEDS == tuple(range(42048, 42064))
    assert len(probe_module.TRAIN_SEEDS) == 48
    assert len(probe_module.VALIDATION_SEEDS) == 16
    assert set(probe_module.TRAIN_SEEDS).isdisjoint(probe_module.VALIDATION_SEEDS)
    assert [label for label, _, _ in probe_module.ROUND_BUCKETS] == [
        "1-25", "26-50", "51-100", "101-150", "151-terminal",
    ]
    assert probe_module.SAMPLES_PER_CELL == 3
    assert probe_module.VALUE_PROBE_INPUT_DIM == 300
    assert probe_module.VALUE_PROBE_HIDDEN_DIM == 256
    assert probe_module.VALUE_PROBE_OUTPUT_DIM == 4
    assert probe_module.LEARNING_CURVE_FRACTIONS == (0.25, 0.5, 1.0)


def test_train_seeds_are_registered_dev_in_evaluation_protocol():
    import evaluation_protocol as ep

    for seed in probe_module.SELF_PLAY_SEEDS:
        assert seed in ep.DEV_SEEDS, f"seed {seed} not registered as DEV"
    ep.require_seed_scope(probe_module.SELF_PLAY_SEEDS, ep.SEED_CLASS_DEV, context="test")


# ── checkpoint integrity gate ────────────────────────────────────────────


def test_verify_checkpoint_raises_when_missing(tmp_path):
    with pytest.raises(SystemExit, match="missing checkpoint"):
        probe_module.verify_checkpoint(tmp_path / "nope.pt", "a" * 64)


def test_verify_checkpoint_raises_on_sha_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"some bytes")
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        probe_module.verify_checkpoint(checkpoint, "a" * 64)


def test_verify_checkpoint_passes_when_hash_matches(tmp_path):
    import hashlib

    checkpoint = tmp_path / "checkpoint.pt"
    content = b"some checkpoint bytes"
    checkpoint.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert probe_module.verify_checkpoint(checkpoint, expected) == expected


# ── round bucket boundaries ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "round_,expected",
    [
        (0, None), (1, "1-25"), (25, "1-25"), (26, "26-50"), (50, "26-50"),
        (51, "51-100"), (100, "51-100"), (101, "101-150"), (150, "101-150"),
        (151, "151-terminal"), (200, "151-terminal"), (300, "151-terminal"),
    ],
)
def test_bucket_for_round(round_, expected):
    assert probe_module.bucket_for_round(round_) == expected


# ── pure metric helpers: hand-verified known values ─────────────────────


def test_cross_entropy_perfect_prediction_is_zero():
    probs = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    assert probe_module.cross_entropy(probs, [0, 1]) == pytest.approx(0.0, abs=1e-9)


def test_cross_entropy_uniform_matches_ln4():
    import math

    probs = [[0.25, 0.25, 0.25, 0.25]]
    assert probe_module.cross_entropy(probs, [0]) == pytest.approx(math.log(4))


def test_brier_score_hand_verified():
    """Uniform prediction [.25,.25,.25,.25] against true class 0:
    (.75^2 + .25^2*3) = 0.5625 + 0.1875 = 0.75."""
    probs = [[0.25, 0.25, 0.25, 0.25]]
    assert probe_module.brier_score(probs, [0]) == pytest.approx(0.75)


def test_top1_accuracy():
    probs = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7], [0.4, 0.3, 0.2, 0.1]]
    true_classes = [0, 3, 1]  # 2 correct (idx0, idx1), 1 wrong (idx2 predicts 0 not 1)
    assert probe_module.top1_accuracy(probs, true_classes) == pytest.approx(2 / 3)


def test_confusion_matrix_hand_verified():
    probs = [[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]
    true_classes = [0, 1, 1]  # pred: 0,1,0 ; true: 0,1,1
    matrix = probe_module.confusion_matrix(probs, true_classes)
    assert matrix[0][0] == 1  # true=0 pred=0
    assert matrix[1][1] == 1  # true=1 pred=1
    assert matrix[1][0] == 1  # true=1 pred=0
    assert sum(sum(row) for row in matrix) == 3


def test_expected_calibration_error_single_bin_hand_verified():
    """n_bins=1: confidences [0.8, 0.8], preds [0, 0], true [0, 1] ->
    accuracy 0.5, mean_confidence 0.8, ece = |0.8-0.5| = 0.3."""
    probs = [[0.8, 0.2 / 3, 0.2 / 3, 0.2 / 3], [0.8, 0.2 / 3, 0.2 / 3, 0.2 / 3]]
    result = probe_module.expected_calibration_error(probs, [0, 1], n_bins=1)
    assert result["ece"] == pytest.approx(0.3)
    assert result["bins"][0]["count"] == 2
    assert result["bins"][0]["mean_confidence"] == pytest.approx(0.8)
    assert result["bins"][0]["accuracy"] == pytest.approx(0.5)


def test_class_balance_counts():
    result = probe_module.class_balance([0, 0, 1, 2, 2, 2])
    assert result["0"] == {"count": 2, "fraction": pytest.approx(2 / 6)}
    assert result["1"] == {"count": 1, "fraction": pytest.approx(1 / 6)}
    assert result["2"] == {"count": 3, "fraction": pytest.approx(3 / 6)}
    assert result["3"] == {"count": 0, "fraction": 0.0}


def test_uniform_baseline_probs_shape_and_value():
    probs = probe_module.uniform_baseline_probs(5)
    assert probs.shape == (5, 4)
    assert (probs == 0.25).all()


def test_leader_baseline_probs_epsilon_smoothed():
    probs = probe_module.leader_baseline_probs([2, 0], eps=1e-6)
    assert probs.shape == (2, 4)
    assert probs[0][2] == pytest.approx(1.0 - 3e-6)
    assert probs[0][0] == pytest.approx(1e-6)
    assert probs[1][0] == pytest.approx(1.0 - 3e-6)
    # finite cross-entropy even when the leader is wrong on every sample
    ce = probe_module.cross_entropy(probs, [0, 1])
    assert ce < float("inf")
    assert ce > 0


def test_evaluate_predictor_overall_and_bucket_breakdown():
    probs = [[0.25] * 4] * 4
    true_classes = [0, 1, 2, 3]
    buckets = ["1-25", "1-25", "26-50", "151-terminal"]
    result = probe_module.evaluate_predictor(probs, true_classes, buckets)
    assert result["overall"]["top1_accuracy"] == pytest.approx(0.25)  # argmax always class 0 (first max)
    assert result["by_bucket"]["1-25"]["count"] == 2
    assert result["by_bucket"]["26-50"]["count"] == 1
    assert result["by_bucket"]["51-100"]["count"] == 0
    assert result["by_bucket"]["51-100"]["cross_entropy"] is None


# ── ValueProbe architecture + deterministic training ────────────────────


def test_build_value_probe_forward_shape():
    import torch

    model = probe_module.build_value_probe()
    batch = torch.zeros((5, 300), dtype=torch.float32)
    output = model(batch)
    assert tuple(output.shape) == (5, 4)


def _tiny_synthetic_dataset(n_per_class: int = 20, seed: int = 0):
    """A trivially learnable synthetic dataset: state[class] is large,
    everything else is noise - lets the determinism/sanity tests run fast
    without touching the real engine at all."""
    import numpy as np

    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for true_class in range(4):
        for _ in range(n_per_class):
            x = rng.normal(0.0, 0.05, size=300).astype(np.float32)
            x[true_class] += 5.0
            xs.append(x)
            ys.append(true_class)
    x = np.stack(xs)
    y = np.array(ys, dtype=np.int64)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def test_train_value_probe_learns_trivially_separable_data():
    train_x, train_y = _tiny_synthetic_dataset(seed=0)
    val_x, val_y = _tiny_synthetic_dataset(seed=1)
    model, stats = probe_module.train_value_probe(
        train_x, train_y, val_x, val_y, max_epochs=100, patience=10,
    )
    probs = probe_module.value_probe_predict_proba(model, val_x)
    accuracy = probe_module.top1_accuracy(probs, val_y)
    assert accuracy > 0.9
    assert stats["epochs_run"] >= 1
    assert stats["best_val_loss"] < 1.0


def test_train_value_probe_is_deterministic():
    train_x, train_y = _tiny_synthetic_dataset(seed=0)
    val_x, val_y = _tiny_synthetic_dataset(seed=1)
    model_a, stats_a = probe_module.train_value_probe(train_x, train_y, val_x, val_y, max_epochs=30, patience=5)
    model_b, stats_b = probe_module.train_value_probe(train_x, train_y, val_x, val_y, max_epochs=30, patience=5)
    assert stats_a == stats_b
    probs_a = probe_module.value_probe_predict_proba(model_a, val_x)
    probs_b = probe_module.value_probe_predict_proba(model_b, val_x)
    import numpy as np

    assert np.array_equal(probs_a, probs_b)


# ── learning-curve seed-subset helper: nested prefixes ──────────────────


def test_learning_curve_subset_seeds_are_nested_prefixes():
    train_seeds = probe_module.TRAIN_SEEDS
    subset_25 = probe_module._learning_curve_subset_seeds(train_seeds, 0.25)
    subset_50 = probe_module._learning_curve_subset_seeds(train_seeds, 0.5)
    subset_100 = probe_module._learning_curve_subset_seeds(train_seeds, 1.0)
    assert len(subset_25) == 12
    assert len(subset_50) == 24
    assert len(subset_100) == 48
    assert set(subset_25).issubset(set(subset_50))
    assert set(subset_50).issubset(set(subset_100))
    assert subset_100 == train_seeds


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


def test_uses_shared_modules():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import monopolyzero_common as common" in source
    assert "import evaluation_protocol as ep" in source


def test_no_puct_or_search_used():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "MaxNPUCT" not in source
    assert "build_local_search_policy" not in source
    assert "self_play=" not in source
    assert "build_local_policy_only" in source


def test_no_fixed_agents_used():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "LocalFixedPolicy" not in source
    assert "FP_AGENT_CLASSES" not in source


def test_policy_network_never_trained():
    """This probe must never CALL the shared training-update step, and
    must never construct an optimizer over the loaded checkpoint's own
    parameters - only the separate ValueProbe (a different local `model`
    inside train_value_probe's own scope, which legitimately calls
    `.train()`/`.eval()` on itself) gets gradient updates."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "local_training_update(" not in source
    assert "MonopolyZeroNet.load_inference(CHECKPOINT_PATH)" in source
    assert "model.eval()" in source
    load_line = source.index("MonopolyZeroNet.load_inference(CHECKPOINT_PATH)")
    nearby = source[load_line:load_line + 200]
    assert "model.eval()" in nearby  # loaded checkpoint goes straight to eval mode
    assert "torch.optim" not in nearby


def test_no_win_rate_or_wilson_evaluation():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "wilson_95_interval(" not in source
    assert "focus_won" not in source
    assert '"wins":' not in source


def test_uses_require_seed_scope_dev():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "ep.require_seed_scope(" in source
    assert "ep.SEED_CLASS_DEV" in source


def test_no_promotion_or_go_kill_boolean_in_source():
    """Checks for a computed verdict as a JSON-style dict key (colon
    right after the word) rather than a bare substring - this file
    legitimately has descriptive flag names like
    "no_arbitrary_go_kill_threshold" and "no_win_rate_evaluation" that
    contain these words without being a computed verdict."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "_recommended" not in source
    for forbidden_key in ('"promote":', '"promotion_recommended":', '"go_kill":', '"verdict":', '"kill_recommended":'):
        assert forbidden_key not in source
