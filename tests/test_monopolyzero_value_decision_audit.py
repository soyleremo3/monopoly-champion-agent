"""Tests for scripts/monopolyzero_value_decision_audit.py: the pure
segmentation/distribution helpers, margin-quartile labeling, the
value-probe-advantage-segment finder's threshold logic against hand-built
cases, and structural checks (reuses probe_v1/probe_v2 rather than
duplicating them, registers no new seeds, never touches TEST for new
model/hyperparameter selection, no PUCT/fixed agents). Does not run the
actual audit (see the experiment log for that - it also depends on 021's
log file already existing).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_value_decision_audit.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_value_decision_audit", SCRIPT)
audit_module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_value_decision_audit"] = audit_module
_spec.loader.exec_module(audit_module)


def test_config_matches_task_spec():
    assert audit_module.RECONCILIATION_TOLERANCE == 1e-6
    assert audit_module.VALUE_PROBE_WIN_MIN_SEGMENT_STATES == 20
    assert audit_module.VALUE_PROBE_WIN_MIN_ACCURACY_MARGIN == 0.05
    assert audit_module.OUTCOME_ADJACENT_AXES == frozenset({"margin_quartile", "current_player_rank"})
    assert audit_module.PRIOR_EXPERIMENT_LOG.name == "021-monopolyzero-value-generalization-probe.json"


# ── quantile summary + margin quartile labeling ─────────────────────────


def test_quantile_summary_known_values():
    result = audit_module._quantile_summary([1.0, 2.0, 3.0, 4.0])
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(4.0)
    assert result["median"] == pytest.approx(2.5)


def test_quantile_summary_empty_is_all_none():
    result = audit_module._quantile_summary([])
    assert result == {"min": None, "q25": None, "median": None, "q75": None, "max": None}


def test_margin_quartile_label_boundaries():
    assert audit_module.margin_quartile_label(0.0, q25=0.1, median=0.3, q75=0.6) == "Q1(low margin)"
    assert audit_module.margin_quartile_label(0.1, q25=0.1, median=0.3, q75=0.6) == "Q1(low margin)"
    assert audit_module.margin_quartile_label(0.2, q25=0.1, median=0.3, q75=0.6) == "Q2"
    assert audit_module.margin_quartile_label(0.45, q25=0.1, median=0.3, q75=0.6) == "Q3"
    assert audit_module.margin_quartile_label(0.9, q25=0.1, median=0.3, q75=0.6) == "Q4(high margin)"


# ── segment_report / distribution ────────────────────────────────────────


def _fake_audit_record(*, seed, true_class, leader_probs, learned_probs, **extra):
    return {"seed": seed, "true_class": true_class, "leader_probs": leader_probs, "learned_probs": learned_probs, **extra}


def test_segment_report_groups_and_scores_correctly():
    records = [
        _fake_audit_record(seed=1, true_class=0, leader_probs=[0.9, 0.03, 0.03, 0.04], learned_probs=[0.25] * 4, bucket="A"),
        _fake_audit_record(seed=2, true_class=1, leader_probs=[0.03, 0.9, 0.03, 0.04], learned_probs=[0.25] * 4, bucket="A"),
        _fake_audit_record(seed=3, true_class=2, leader_probs=[0.03, 0.03, 0.03, 0.91], learned_probs=[0.25] * 4, bucket="B"),
    ]
    result = audit_module.segment_report(records, lambda r: r["bucket"])
    assert result["A"]["count"] == 2
    assert result["A"]["unique_games"] == 2
    assert result["A"]["leader"]["top1_accuracy"] == pytest.approx(1.0)
    # argmax of a uniform [0.25]*4 always resolves to class 0 (first max) -
    # correct for record 1 (true_class=0), wrong for record 2 (true_class=1)
    assert result["A"]["learned_value_probe"]["top1_accuracy"] == pytest.approx(0.5)
    assert result["B"]["count"] == 1
    assert result["B"]["leader"]["top1_accuracy"] == pytest.approx(0.0)  # leader predicts class 3, true is class 2


def test_distribution_counts_and_fractions():
    records = [{"x": "a"}, {"x": "a"}, {"x": "b"}]
    result = audit_module.distribution(records, lambda r: r["x"])
    assert result["a"] == {"count": 2, "fraction": pytest.approx(2 / 3)}
    assert result["b"] == {"count": 1, "fraction": pytest.approx(1 / 3)}


def test_distribution_empty_records():
    assert audit_module.distribution([], lambda r: r["x"]) == {}


# ── value-probe-advantage-segment finder: threshold logic ──────────────


def _fake_segment_stats(count, leader_acc, learned_acc):
    return {
        "count": count,
        "unique_games": count,
        "leader": {"top1_accuracy": leader_acc, "cross_entropy": 1.0, "brier_score": 1.0},
        "learned_value_probe": {"top1_accuracy": learned_acc, "cross_entropy": 1.0, "brier_score": 1.0},
    }


def test_find_value_probe_advantage_segments_below_count_threshold_excluded():
    reports = {"round_bucket": {"1-25": _fake_segment_stats(count=10, leader_acc=0.5, learned_acc=0.9)}}
    wins = audit_module.find_value_probe_advantage_segments(reports)
    assert wins == []  # count=10 < min_count=20, excluded even though margin is huge


def test_find_value_probe_advantage_segments_below_margin_threshold_excluded():
    reports = {"round_bucket": {"1-25": _fake_segment_stats(count=100, leader_acc=0.60, learned_acc=0.62)}}
    wins = audit_module.find_value_probe_advantage_segments(reports)
    assert wins == []  # margin=0.02 < min_margin=0.05


def test_find_value_probe_advantage_segments_qualifying_segment_included():
    reports = {
        "decision_type": {"auction": _fake_segment_stats(count=50, leader_acc=0.40, learned_acc=0.55)},
        "round_bucket": {"1-25": _fake_segment_stats(count=100, leader_acc=0.60, learned_acc=0.61)},
    }
    wins = audit_module.find_value_probe_advantage_segments(reports)
    assert len(wins) == 1
    assert wins[0]["axis"] == "decision_type"
    assert wins[0]["segment"] == "auction"
    assert wins[0]["accuracy_margin"] == pytest.approx(0.15)
    assert wins[0]["outcome_adjacent_axis"] is False


def test_find_value_probe_advantage_segments_flags_outcome_adjacent_axis():
    reports = {"current_player_rank": {"1": _fake_segment_stats(count=50, leader_acc=0.3, learned_acc=0.5)}}
    wins = audit_module.find_value_probe_advantage_segments(reports)
    assert len(wins) == 1
    assert wins[0]["outcome_adjacent_axis"] is True


def test_find_value_probe_advantage_segments_custom_thresholds():
    reports = {"round_bucket": {"1-25": _fake_segment_stats(count=10, leader_acc=0.5, learned_acc=0.6)}}
    wins = audit_module.find_value_probe_advantage_segments(reports, min_count=5, min_accuracy_margin=0.05)
    assert len(wins) == 1


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


def test_reuses_v1_and_v2_instead_of_duplicating():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import monopolyzero_value_learnability_probe as probe_v1" in source
    assert "import monopolyzero_value_generalization_probe as probe_v2" in source
    assert "probe_v2.SELF_PLAY_SEEDS" in source
    assert "probe_v2.quantile_indices(" in source
    assert "probe_v1.train_value_probe(" in source


def test_no_puct_or_fixed_agents():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "MaxNPUCT" not in source
    assert "LocalFixedPolicy" not in source
    assert "FP_AGENT_CLASSES" not in source


def test_no_new_seed_range_registered():
    """This audit reuses 021's already-DEV-registered seeds - it must not
    add a new DEV_SEED_RANGES entry to evaluation_protocol.py."""
    protocol_source = (REPO_ROOT / "scripts" / "evaluation_protocol.py").read_text(encoding="utf-8")
    assert "022" not in protocol_source


def test_reconciliation_check_happens_before_segment_analysis():
    source = SCRIPT.read_text(encoding="utf-8")
    reconciliation_index = source.index("reconciliation[label] = ")
    segment_index = source.index("segment_reports = {")
    assert reconciliation_index < segment_index


def test_reconciliation_failure_raises():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Reconciliation failed" in source
    assert "raise RuntimeError" in source


def test_does_not_refit_temperature_against_test():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "fit_probabilistic_leader_temperature(pooled_nw, pooled_y)" in source
    pooled_nw_line = next(line for line in source.splitlines() if "pooled_nw = np.concatenate" in line)
    assert "train_nw, selection_nw" in pooled_nw_line
    assert "test_nw" not in pooled_nw_line


def test_no_promotion_or_hardcoded_go_kill_boolean_key():
    """This script DOES produce a real A/B decision (unlike prior
    experiments) - that's intentional and expected. It must not ALSO
    contain a separate arbitrary promotion-style boolean key."""
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden_key in ('"promote":', '"promotion_recommended":', '"go_kill":', '"kill_recommended":'):
        assert forbidden_key not in source
    assert '"final_decision"' in source
