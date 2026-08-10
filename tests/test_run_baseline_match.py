"""Focused test for the PYTHONHASHSEED guard in scripts/run_baseline_match.py.

Loads the script as a module without triggering its heavy imports (those are
deferred to main(), after the guard), so these tests run fast and do not
require torch/numpy to be installed.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_baseline_match.py"

_spec = importlib.util.spec_from_file_location("run_baseline_match", SCRIPT)
run_baseline_match = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_baseline_match)


@pytest.mark.parametrize("value", [None, "", "1", "2", "random"])
def test_guard_rejects_unpinned_hash_seed(value, monkeypatch):
    if value is None:
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    else:
        monkeypatch.setenv("PYTHONHASHSEED", value)
    with pytest.raises(SystemExit) as excinfo:
        run_baseline_match._require_pinned_hash_seed()
    assert "PYTHONHASHSEED=0" in str(excinfo.value)


def test_guard_accepts_pinned_hash_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    run_baseline_match._require_pinned_hash_seed()


def test_cli_fails_fast_without_pinned_hash_seed():
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--seed", "1"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "PYTHONHASHSEED=0" in result.stderr


def test_cli_fails_fast_with_wrong_pinned_hash_seed():
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "1"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--seed", "1"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "PYTHONHASHSEED=0" in result.stderr
