"""Focused test for the PYTHONHASHSEED guard in
scripts/selfplay_train_smoke.py, and that it imports no ASU code.

Loads the script as a module without triggering its heavy imports (deferred
to main(), after the guard), so this runs fast and needs no torch/numpy.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "selfplay_train_smoke.py"

_spec = importlib.util.spec_from_file_location("selfplay_train_smoke", SCRIPT)
selfplay_train_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(selfplay_train_smoke)


@pytest.mark.parametrize("value", [None, "", "1", "2", "random"])
def test_guard_rejects_unpinned_hash_seed(value, monkeypatch):
    if value is None:
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    else:
        monkeypatch.setenv("PYTHONHASHSEED", value)
    with pytest.raises(SystemExit) as excinfo:
        selfplay_train_smoke._require_pinned_hash_seed()
    assert "PYTHONHASHSEED=0" in str(excinfo.value)


def test_guard_accepts_pinned_hash_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    selfplay_train_smoke._require_pinned_hash_seed()


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


def test_no_trainer_or_population_jobs_used():
    """Confirms the script does not route through the ASU-hardcoded
    Trainer/population_jobs path (see docs/REFERENCE_AUDIT.md). Checks real
    code (imports and calls), not the docstring's prose explaining why."""
    source = SCRIPT.read_text(encoding="utf-8")
    import_lines = [
        line for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    assert not any("Trainer" in line for line in import_lines)
    assert not any("population_jobs" in line for line in import_lines)
    assert not any("generate_population_games" in line for line in import_lines)
    assert "Trainer(" not in source
    assert "population_jobs(" not in source
    assert "generate_population_games(" not in source


def test_opponent_pool_is_limited_to_self_and_fixed_a_b_c():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "FP_AGENT_CLASSES[:3]" in source


def test_does_not_import_adapters_arena_or_training_modules():
    """adapters.py and training.py import ASU_FROZEN_TEACHER at module level;
    arena.py imports adapters.py. Importing any of the three loads ASU as a
    side effect even if it's never called — see docs/REFERENCE_AUDIT.md and
    docs/DECISIONS.md's 2026-08-11 ASU-import-free correction. Only real
    import statement lines are checked, not prose."""
    source = SCRIPT.read_text(encoding="utf-8")
    import_lines = [
        line for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert hits == [], f"found forbidden ASU-coupled module import(s): {hits}"
    # also catch relative-import forms used from inside the monopoly_bench package
    assert not any(
        line.strip() in ("from .adapters import", "from .arena import", "from .training import")
        or line.strip().startswith(("from .adapters import", "from .arena import", "from .training import"))
        for line in import_lines
    )


def test_asu_module_guard_detects_loaded_modules(monkeypatch):
    sentinel_names = ["ASU_FROZEN_TEACHER", "ASU_FROZEN_TEACHER.core"]
    for name in sentinel_names:
        monkeypatch.setitem(sys.modules, name, object())
    try:
        loaded = selfplay_train_smoke._loaded_asu_modules()
    finally:
        for name in sentinel_names:
            sys.modules.pop(name, None)
    assert loaded == sentinel_names


def test_asu_module_guard_clean_when_absent():
    for name in list(sys.modules):
        assert not (name == "ASU_FROZEN_TEACHER" or name.startswith("ASU_FROZEN_TEACHER."))
    assert selfplay_train_smoke._loaded_asu_modules() == []


# ── Clean-git-tree guard (mocked subprocess, no real git calls) ───────────


class _FakeCompletedProcess:
    def __init__(self, stdout: str):
        self.stdout = stdout


def test_clean_tree_guard_returns_head_sha_when_clean(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["git", "status"]:
            return _FakeCompletedProcess(stdout="")
        if args[:2] == ["git", "rev-parse"]:
            return _FakeCompletedProcess(stdout="deadbeef1234567890deadbeef1234567890dead\n")
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(selfplay_train_smoke.subprocess, "run", fake_run)
    sha = selfplay_train_smoke._require_clean_git_tree()
    assert sha == "deadbeef1234567890deadbeef1234567890dead"
    assert calls == [
        ["git", "status", "--porcelain"],
        ["git", "rev-parse", "HEAD"],
    ]


def test_clean_tree_guard_raises_when_dirty(monkeypatch):
    def fake_run(args, **kwargs):
        if args[:2] == ["git", "status"]:
            return _FakeCompletedProcess(stdout=" M scripts/selfplay_train_smoke.py\n")
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(selfplay_train_smoke.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as excinfo:
        selfplay_train_smoke._require_clean_git_tree()
    assert "not clean" in str(excinfo.value)


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
