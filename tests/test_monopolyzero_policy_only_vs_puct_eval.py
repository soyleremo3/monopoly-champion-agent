"""Focused tests for scripts/monopolyzero_policy_only_vs_puct_eval.py: the
checkpoint-integrity gate, pure math helpers (Wilson interval, percentile,
median, non-overlap test, disagreement rate), and config/import-graph
checks. Does not run the actual 80-game evaluation (see the experiment log
for that result instead).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_policy_only_vs_puct_eval.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_policy_only_vs_puct_eval", SCRIPT)
eval_module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_policy_only_vs_puct_eval"] = eval_module
_spec.loader.exec_module(eval_module)


def test_config_matches_task_spec():
    assert eval_module.EXPECTED_CHECKPOINT_SHA256 == (
        "152c0a0f6136d1fc91e74973ac245b2f72774694c424d2a48854514ed2848383"
    )
    assert eval_module.HELD_OUT_SEEDS == tuple(range(32000, 32010))
    assert len(eval_module.HELD_OUT_SEEDS) == 10
    assert eval_module.SIMULATIONS == 4
    assert eval_module.MAX_DEPTH == 16
    assert eval_module.MAX_ROUNDS == 200


def test_verify_checkpoint_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_module, "CHECKPOINT_PATH", tmp_path / "nope.pt")
    with pytest.raises(SystemExit) as excinfo:
        eval_module.verify_checkpoint()
    assert "missing checkpoint" in str(excinfo.value)


def test_verify_checkpoint_raises_on_sha_mismatch(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"some bytes")
    monkeypatch.setattr(eval_module, "CHECKPOINT_PATH", checkpoint)
    with pytest.raises(SystemExit) as excinfo:
        eval_module.verify_checkpoint()
    assert "SHA-256 mismatch" in str(excinfo.value)


def test_verify_checkpoint_passes_when_hash_matches(tmp_path, monkeypatch):
    import hashlib

    checkpoint = tmp_path / "checkpoint.pt"
    content = b"some checkpoint bytes"
    checkpoint.write_bytes(content)
    monkeypatch.setattr(eval_module, "CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(eval_module, "EXPECTED_CHECKPOINT_SHA256", hashlib.sha256(content).hexdigest())
    assert eval_module.verify_checkpoint() == hashlib.sha256(content).hexdigest()


def test_verify_checkpoint_points_at_the_500_update_artifact():
    assert eval_module.CHECKPOINT_PATH.name == "trained_updates_500.pt"


# ── pure math helpers ─────────────────────────────────────────────────────


def test_median_odd_count():
    assert eval_module._median([3.0, 1.0, 2.0]) == 2.0


def test_median_even_count():
    assert eval_module._median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_median_empty_is_none():
    assert eval_module._median([]) is None


def test_wilson_interval_matches_known_value():
    lower, upper = eval_module.wilson_95_interval(50, 100)
    assert lower == pytest.approx(0.4038, abs=1e-3)
    assert upper == pytest.approx(0.5962, abs=1e-3)


def test_non_overlapping_improvement_true_when_clearly_separated():
    worse = {"wilson_95": [0.0, 0.1]}
    better = {"wilson_95": [0.2, 0.4]}
    assert eval_module.non_overlapping_improvement(worse, better) is True


def test_non_overlapping_improvement_false_when_overlapping():
    worse = {"wilson_95": [0.0, 0.3]}
    better = {"wilson_95": [0.1, 0.5]}
    assert eval_module.non_overlapping_improvement(worse, better) is False


def test_non_overlapping_improvement_false_when_missing():
    assert eval_module.non_overlapping_improvement({"wilson_95": None}, {"wilson_95": [0.1, 0.2]}) is False


# ── disagreement_rate: pure function over a fake shadow_decisions list ────


def test_disagreement_rate_all_agree_is_zero():
    decisions = [{"agree": True} for _ in range(10)]
    assert eval_module.disagreement_rate(decisions) == 0.0


def test_disagreement_rate_all_disagree_is_one():
    decisions = [{"agree": False} for _ in range(10)]
    assert eval_module.disagreement_rate(decisions) == 1.0


def test_disagreement_rate_partial():
    decisions = [{"agree": True} for _ in range(8)] + [{"agree": False} for _ in range(2)]
    assert eval_module.disagreement_rate(decisions) == pytest.approx(0.2)


def test_disagreement_rate_empty_is_none():
    assert eval_module.disagreement_rate([]) is None


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


def test_self_play_is_false_for_evaluation():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "self_play=False" in source


def test_no_training_update_call():
    """This is a pure inference ablation - it must never call the
    training-update step, and must never build a replay-recording
    call (record_seats is always empty)."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "local_training_update" not in source
    assert "record_seats=set()" in source


def test_both_policies_share_one_checkpoint_path():
    """Guards against accidentally evaluating POLICY_ONLY and PUCT_4 on
    different checkpoints - both must be built from the same `model`
    object loaded once from CHECKPOINT_PATH."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("MonopolyZeroNet.load_inference(CHECKPOINT_PATH)") == 1
    assert "build_local_policy_only(model)" in source
    assert "build_local_search_policy(model, search_config, self_play=False)" in source


def test_shadow_query_uses_policy_only_against_puct_4_games():
    source = SCRIPT.read_text(encoding="utf-8")
    puct_call_start = source.index('"puct_4"')
    puct_call_block = source[puct_call_start:puct_call_start + 400]
    assert "shadow_builder=lambda: common.build_local_policy_only(model)" in puct_call_block


def test_raw_shadow_decisions_list_stripped_before_printing():
    """The per-decision shadow-query list (tens of thousands of small
    records across 40 games) must not end up in the printed JSON payload -
    only the aggregate rate/count. Otherwise raw stdout balloons to
    megabytes instead of staying a small, diffable summary."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'k != "shadow_decisions"' in source
    payload_start = source.index("payload = {")
    payload_block = source[payload_start:source.index("\n\n\n", payload_start)]
    assert "policy_only_result_public" in payload_block
    assert "puct_4_result_public" in payload_block
