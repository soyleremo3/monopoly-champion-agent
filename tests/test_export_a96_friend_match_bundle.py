"""Tests for scripts/export_a96_friend_match_bundle.py:
- refuses (SystemExit) a source checkpoint with the wrong hash, and a
  missing source checkpoint, before writing anything
- against the REAL local A96 checkpoint (skipped cleanly if absent):
  the exported file is byte-identical to the source, and manifest.json
  carries the correct pinned hashes/metadata
- source-level ASU-import guard
All file-producing tests redirect DIST_DIR to a pytest tmp_path so they
never touch the real (gitignored) dist/ directory.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "export_a96_friend_match_bundle.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import monopolyzero_common as common  # noqa: E402

_spec = importlib.util.spec_from_file_location("export_a96_friend_match_bundle", SCRIPT)
module = importlib.util.module_from_spec(_spec)
sys.modules["export_a96_friend_match_bundle"] = module
_spec.loader.exec_module(module)


def test_source_never_imports_asu():
    import_lines = [
        line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("ASU_FROZEN_TEACHER", "monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert not hits, hits


def test_export_refuses_missing_source_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "DIST_DIR", tmp_path / "dist_out")
    with pytest.raises(SystemExit, match="missing"):
        module.export_bundle(source_checkpoint=tmp_path / "does_not_exist.pt")
    assert not (tmp_path / "dist_out").exists()


def test_export_refuses_wrong_source_checkpoint_hash(tmp_path, monkeypatch):
    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    wrong_checkpoint = tmp_path / "wrong.pt"
    agent.save(str(wrong_checkpoint))

    monkeypatch.setattr(module, "DIST_DIR", tmp_path / "dist_out")
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        module.export_bundle(source_checkpoint=wrong_checkpoint)
    assert not (tmp_path / "dist_out").exists()


def _skip_if_no_real_checkpoint():
    from a96_friend_match_agent import DEFAULT_CHECKPOINT_PATH

    if not DEFAULT_CHECKPOINT_PATH.is_file():
        pytest.skip(f"real A96 checkpoint not present locally (gitignored artifact): {DEFAULT_CHECKPOINT_PATH}")


def test_export_real_checkpoint_is_byte_identical(tmp_path, monkeypatch):
    _skip_if_no_real_checkpoint()
    monkeypatch.setattr(module, "DIST_DIR", tmp_path / "dist_out")

    exported_path = module.export_bundle()

    assert exported_path.is_file()
    assert exported_path.read_bytes() == module.DEFAULT_CHECKPOINT_PATH.read_bytes()
    assert module._file_sha256(exported_path) == module.A96_CHECKPOINT_SHA256


def test_export_manifest_has_correct_hashes_and_metadata(tmp_path, monkeypatch):
    _skip_if_no_real_checkpoint()
    monkeypatch.setattr(module, "DIST_DIR", tmp_path / "dist_out")

    exported_path = module.export_bundle()
    manifest = json.loads((module.DIST_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["checkpoint_filename"] == module.A96_CHECKPOINT_FILENAME
    assert manifest["checkpoint_sha256"] == module.A96_CHECKPOINT_SHA256
    assert manifest["actor_sha256"] == module.A96_ACTOR_SHA256
    assert manifest["reference_submodule_sha"] == module.A96_REFERENCE_SUBMODULE_SHA
    assert manifest["entrypoint"] == module.ENTRYPOINT
    assert manifest["checkpoint_size_bytes"] == exported_path.stat().st_size
    assert manifest["python_version_expected"] == module.EXPECTED_PYTHON_VERSION
    assert manifest["dependency_pins"] == module.DEPENDENCY_PINS
    assert isinstance(manifest["source_main_commit_sha"], str) and len(manifest["source_main_commit_sha"]) >= 7
    assert "state" in manifest["harness_must_provide"] and "legal_action_ids" in manifest["harness_must_provide"]


def test_export_writes_readme_with_key_usage_instructions(tmp_path, monkeypatch):
    _skip_if_no_real_checkpoint()
    monkeypatch.setattr(module, "DIST_DIR", tmp_path / "dist_out")

    module.export_bundle()
    readme = (module.DIST_DIR / "README.txt").read_text(encoding="utf-8")

    assert "main" in readme
    assert "git submodule update --init" in readme
    assert "pip install -r requirements.txt" in readme
    assert "A96_CHECKPOINT_PATH" in readme
    assert "A96FriendMatchAgent" in readme
    assert "act(state, legal_action_ids)" in readme
