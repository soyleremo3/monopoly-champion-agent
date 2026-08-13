"""Tests for scripts/preflight_a96_friend_match.py:
- _version_matches() handles exact matches and expected local-version-
  identifier suffixes (e.g. torch's "+cpu") without accepting a genuinely
  different version
- individual check functions fail closed on bad input (wrong Python
  version, missing/wrong checkpoint, submodule not present)
- against the REAL local A96 checkpoint + real environment (skipped
  cleanly if the checkpoint is absent): the full preflight succeeds and
  prints "A96 FRIEND MATCH CORE READY"
- source-level ASU-import guard
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "preflight_a96_friend_match.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location("preflight_a96_friend_match", SCRIPT)
module = importlib.util.module_from_spec(_spec)
sys.modules["preflight_a96_friend_match"] = module
_spec.loader.exec_module(module)


def test_source_never_imports_asu():
    import_lines = [
        line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("ASU_FROZEN_TEACHER", "monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert not hits, hits


# ── _version_matches ─────────────────────────────────────────────────────


def test_version_matches_exact():
    assert module._version_matches("2.13.0", "2.13.0") is True


def test_version_matches_local_identifier_suffix():
    assert module._version_matches("2.13.0+cpu", "2.13.0") is True


def test_version_matches_rejects_different_base_version():
    assert module._version_matches("2.14.0", "2.13.0") is False


def test_version_matches_rejects_non_plus_suffix():
    assert module._version_matches("2.13.0extra", "2.13.0") is False


# ── individual checks fail closed ────────────────────────────────────────


def test_check_python_version_fails_on_mismatch(monkeypatch):
    monkeypatch.setattr(module.platform, "python_version", lambda: "3.11.0")
    check = module.check_python_version()
    assert check.passed is False


def test_check_python_version_passes_on_match(monkeypatch):
    monkeypatch.setattr(module.platform, "python_version", lambda: module.EXPECTED_PYTHON_VERSION)
    check = module.check_python_version()
    assert check.passed is True


def test_check_submodule_sha_fails_closed_when_submodule_not_present():
    check = module.check_submodule_sha(submodule_present=False)
    assert check.passed is False
    assert "skipped" in check.detail


def test_check_checkpoint_exists_fails_on_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "resolve_checkpoint_path", lambda arg: tmp_path / "missing.pt")
    check, path = module.check_checkpoint_exists()
    assert check.passed is False
    assert path == tmp_path / "missing.pt"


def test_check_checkpoint_sha_skips_when_not_exists():
    check = module.check_checkpoint_sha(Path("/irrelevant.pt"), exists=False)
    assert check.passed is False
    assert "skipped" in check.detail


def test_check_load_and_metadata_skips_all_when_checkpoint_not_ok(tmp_path):
    agent, checks = module.check_load_and_metadata(tmp_path / "irrelevant.pt", checkpoint_ok=False)
    assert agent is None
    assert all(not c.passed for c in checks)
    assert all("skipped" in c.detail for c in checks)


def test_check_inference_skips_when_agent_none():
    check = module.check_inference(None)
    assert check.passed is False
    assert "skipped" in check.detail


# ── full preflight against the REAL local checkpoint ────────────────────


def test_full_preflight_success_path_against_real_checkpoint(capsys):
    if not module.resolve_checkpoint_path(None).is_file():
        import pytest

        pytest.skip("real A96 checkpoint not present locally (gitignored artifact)")
    exit_code = module.main()
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "A96 FRIEND MATCH CORE READY" in captured.out
