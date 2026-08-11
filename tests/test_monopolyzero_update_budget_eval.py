"""Focused tests for scripts/monopolyzero_update_budget_eval.py: pure math
helpers (Wilson interval, percentile, median, non-overlap test) and
config/import-graph checks. Does not run the actual 160-game evaluation
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
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_update_budget_eval.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_update_budget_eval", SCRIPT)
eval_module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_update_budget_eval"] = eval_module
_spec.loader.exec_module(eval_module)


def test_held_out_seeds_are_ten_consecutive_seeds():
    assert eval_module.HELD_OUT_SEEDS == tuple(range(30000, 30010))
    assert len(eval_module.HELD_OUT_SEEDS) == 10


def test_config_matches_task_spec():
    assert eval_module.SIMULATIONS == 4
    assert eval_module.MAX_DEPTH == 16
    assert eval_module.MAX_ROUNDS == 200


def test_four_checkpoints_configured():
    assert set(eval_module.CHECKPOINTS.keys()) == {
        "budget_0_baseline", "budget_100", "budget_500", "budget_1000",
    }


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
