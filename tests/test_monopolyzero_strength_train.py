"""Focused tests for scripts/monopolyzero_strength_train.py: the seat-
balanced game plan (no engine/model needed) and import-graph/config-source
checks. Does not run the actual 32-game training (see the experiment log
for that result instead).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_strength_train.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_strength_train", SCRIPT)
train_module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_strength_train"] = train_module
_spec.loader.exec_module(train_module)


def test_game_plan_has_requested_total():
    plan = train_module._build_game_plan(32)
    assert len(plan) == 32


def test_game_plan_is_half_self_play_half_vs_fixed():
    plan = train_module._build_game_plan(32)
    categories = Counter(job["category"] for job in plan)
    assert categories == {"self_play": 16, "vs_fixed": 16}


def test_game_plan_vs_fixed_seats_are_evenly_balanced():
    plan = train_module._build_game_plan(32)
    focus_seats = [job["focus_seat"] for job in plan if job["category"] == "vs_fixed"]
    assert Counter(focus_seats) == {0: 4, 1: 4, 2: 4, 3: 4}


def test_game_plan_seeds_are_unique():
    plan = train_module._build_game_plan(32)
    seeds = [job["seed"] for job in plan]
    assert len(seeds) == len(set(seeds))


def test_game_plan_is_deterministic():
    assert train_module._build_game_plan(32) == train_module._build_game_plan(32)


def test_cli_fails_fast_without_pinned_hash_seed():
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "PYTHONHASHSEED=0" in result.stderr


def test_does_not_import_adapters_arena_training_or_trainer():
    source = SCRIPT.read_text(encoding="utf-8")
    import_lines = [
        line for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert hits == [], f"found forbidden ASU-coupled module import(s): {hits}"
    assert not any("Trainer" in line for line in import_lines)
    assert "Trainer(" not in source


def test_uses_shared_monopolyzero_common_module():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import monopolyzero_common as common" in source


def test_optimizer_hyperparameters_sourced_from_training_config():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "training_defaults.learning_rate" in source
    assert "training_defaults.weight_decay" in source
    assert "training_defaults.gradient_clip" in source
    assert "TrainingConfig()" in source


def test_stops_before_checkpoint_save_on_non_finite_loss():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "loss_all_finite and not asu_modules_loaded" in source


def test_fixed_config_values_match_task_spec():
    assert train_module.GAMES_TOTAL == 32
    assert train_module.GLOBAL_SEED == 42
    assert train_module.SIMULATIONS == 4
    assert train_module.MAX_ROUNDS == 50
