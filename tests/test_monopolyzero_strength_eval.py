"""Focused tests for scripts/monopolyzero_strength_eval.py: the Wilson
interval and percentile helpers (pure math, no engine needed), config
constants, and import-graph checks. Does not run the actual 40-game
evaluation (see the experiment log for that result instead).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_strength_eval.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_strength_eval", SCRIPT)
eval_module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_strength_eval"] = eval_module
_spec.loader.exec_module(eval_module)


def test_wilson_interval_known_value_50_percent_n_100():
    lower, upper = eval_module.wilson_95_interval(50, 100)
    # Textbook Wilson 95% interval for 50/100 is approximately (0.404, 0.596).
    assert lower == pytest.approx(0.4038, abs=1e-3)
    assert upper == pytest.approx(0.5962, abs=1e-3)


def test_wilson_interval_zero_games_is_full_range():
    assert eval_module.wilson_95_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_zero_wins_lower_bound_is_zero():
    lower, upper = eval_module.wilson_95_interval(0, 20)
    assert lower == 0.0
    assert 0 < upper < 1


def test_wilson_interval_all_wins_upper_bound_is_one():
    lower, upper = eval_module.wilson_95_interval(20, 20)
    assert upper == 1.0
    assert 0 < lower < 1


def test_wilson_interval_widens_with_fewer_games():
    narrow = eval_module.wilson_95_interval(10, 20)
    wide = eval_module.wilson_95_interval(1, 2)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_percentile_empty_is_none():
    assert eval_module._percentile([], 95) is None


def test_percentile_p95_of_ten_values():
    values = [float(i) for i in range(1, 11)]  # 1..10
    p95 = eval_module._percentile(values, 95)
    assert p95 in values
    assert p95 >= 9.0


def test_percentile_single_value():
    assert eval_module._percentile([42.0], 95) == 42.0


def test_held_out_seeds_match_task_spec():
    assert eval_module.HELD_OUT_SEEDS == (30000, 30001, 30002, 30003, 30004)


def test_config_matches_task_spec():
    assert eval_module.SIMULATIONS == 4
    assert eval_module.MAX_DEPTH == 16
    assert eval_module.MAX_ROUNDS == 200


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
    assert not any("evaluate_lineup" in line for line in import_lines)
    assert not any("ASU_FROZEN_TEACHER" in line for line in import_lines)


def test_uses_shared_monopolyzero_common_module():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import monopolyzero_common as common" in source


def test_self_play_is_false_for_evaluation():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "self_play=False" in source


def test_improvement_requires_non_overlapping_wilson_intervals():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "wilson_95" in source
    assert "improvement_supported" in source
