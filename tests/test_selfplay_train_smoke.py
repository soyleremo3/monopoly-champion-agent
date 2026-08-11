"""Focused test for scripts/selfplay_train_smoke.py as a thin CLI wrapper
over scripts/monopolyzero_common.py (see tests/test_monopolyzero_common.py
for the shared module's own guard/math/behavior tests).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "selfplay_train_smoke.py"


def test_cli_fails_fast_without_pinned_hash_seed():
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "PYTHONHASHSEED=0" in result.stderr


def test_no_asu_import_in_script_source():
    """Only real `import`/`from` statement lines, not prose mentioning ASU."""
    source = SCRIPT.read_text(encoding="utf-8")
    import_lines = [
        line for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    asu_imports = [line for line in import_lines if "asu" in line.lower()]
    assert asu_imports == [], f"found ASU-related import(s): {asu_imports}"


def test_does_not_import_adapters_arena_training_or_trainer():
    """adapters.py/training.py import ASU_FROZEN_TEACHER at module level;
    arena.py imports adapters.py transitively. This script must go through
    scripts/monopolyzero_common.py instead — see that module's docstring
    and docs/DECISIONS.md's 2026-08-11 (later) correction entry."""
    source = SCRIPT.read_text(encoding="utf-8")
    import_lines = [
        line for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert hits == [], f"found forbidden ASU-coupled module import(s): {hits}"
    assert not any("Trainer" in line for line in import_lines)
    assert not any("population_jobs" in line for line in import_lines)
    assert "Trainer(" not in source
    assert "population_jobs(" not in source


def test_uses_shared_monopolyzero_common_module():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import monopolyzero_common as common" in source


def test_opponent_pool_is_limited_to_self_and_fixed_a_b_c():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "FP_AGENT_CLASSES[:3]" in source


def test_global_seed_is_used_before_model_construction():
    source = SCRIPT.read_text(encoding="utf-8")
    seed_block = source.split("model = MonopolyZeroNet()")[0]
    assert "random.seed(GLOBAL_SEED)" in seed_block
    assert "np.random.seed(GLOBAL_SEED)" in seed_block
    assert "torch.manual_seed(GLOBAL_SEED)" in seed_block


def test_fixed_adapter_fallbacks_are_read_directly_not_assumed_zero():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "fixed_a.fallback_count" in source
    assert "fixed_b.fallback_count" in source
    assert "fixed_c.fallback_count" in source
