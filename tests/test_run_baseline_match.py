"""Focused test for the PYTHONHASHSEED guard in scripts/run_baseline_match.py.

Loads the script as a module without triggering its heavy imports (those are
deferred to main(), after the guard), so these tests run fast and do not
require torch/numpy to be installed.
"""

from __future__ import annotations

import argparse
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


# ── Multi-seed parsing ──────────────────────────────────────────────────────


def test_single_seed_token_matches_legacy_behavior():
    assert run_baseline_match._parse_seed_token("42") == [42]


def test_negative_seed_token_is_not_treated_as_a_range():
    assert run_baseline_match._parse_seed_token("-5") == [-5]


def test_range_token_expands_inclusive():
    assert run_baseline_match._parse_seed_token("10000-10009") == list(
        range(10000, 10010)
    )


def test_single_element_range_token():
    assert run_baseline_match._parse_seed_token("7-7") == [7]


@pytest.mark.parametrize("token", ["abc", "10-", "10-5-3", "10009-10000"])
def test_invalid_seed_tokens_raise(token):
    with pytest.raises(argparse.ArgumentTypeError):
        run_baseline_match._parse_seed_token(token)


def test_expand_seeds_default_matches_legacy_single_seed():
    assert run_baseline_match._expand_seeds(["42"]) == [42]


def test_expand_seeds_mixes_plain_and_range_tokens_and_dedupes():
    assert run_baseline_match._expand_seeds(["1", "2", "1-3"]) == [1, 2, 3]


def test_expand_seeds_range_token():
    assert run_baseline_match._expand_seeds(["10000-10009"]) == list(
        range(10000, 10010)
    )
