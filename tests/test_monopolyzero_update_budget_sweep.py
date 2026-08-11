"""Focused tests for scripts/monopolyzero_update_budget_sweep.py: the
integrity-check gate (mocked filesystem, no real replay/checkpoint needed)
and config/import-graph checks. Does not run the actual training sweep (see
the experiment log for that result instead).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_update_budget_sweep.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_update_budget_sweep", SCRIPT)
sweep_module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_update_budget_sweep"] = sweep_module
_spec.loader.exec_module(sweep_module)


def test_config_matches_task_spec():
    assert sweep_module.UPDATE_BUDGETS == (100, 500, 1000)
    assert sweep_module.GLOBAL_SEED == 42
    assert sweep_module.BATCH_SIZE == 64
    assert sweep_module.EXPECTED_POSITIONS == 37_772
    assert sweep_module.EXPECTED_BASELINE_SHA256 == (
        "22ae2abea60478355f61ce0404a31de3b52fb97769d4f99f7e04f4c673629370"
    )


def test_verify_reused_artifacts_raises_when_replay_metadata_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_module, "REPLAY_DIR", tmp_path / "no_replay")
    monkeypatch.setattr(sweep_module, "BASELINE_CHECKPOINT", tmp_path / "no_checkpoint.pt")
    with pytest.raises(SystemExit) as excinfo:
        sweep_module.verify_reused_artifacts()
    assert "missing replay metadata" in str(excinfo.value)
    assert "missing baseline checkpoint" in str(excinfo.value)


def test_verify_reused_artifacts_raises_on_position_count_mismatch(tmp_path, monkeypatch):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    (replay_dir / "metadata.json").write_text(json.dumps({"size": 999}), encoding="utf-8")
    checkpoint = tmp_path / "baseline.pt"
    checkpoint.write_bytes(b"not a real checkpoint")

    monkeypatch.setattr(sweep_module, "REPLAY_DIR", replay_dir)
    monkeypatch.setattr(sweep_module, "BASELINE_CHECKPOINT", checkpoint)
    with pytest.raises(SystemExit) as excinfo:
        sweep_module.verify_reused_artifacts()
    assert "replay size mismatch" in str(excinfo.value)
    assert "SHA-256 mismatch" in str(excinfo.value)


def test_verify_reused_artifacts_passes_when_everything_matches(tmp_path, monkeypatch):
    import hashlib

    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    (replay_dir / "metadata.json").write_text(
        json.dumps({"size": sweep_module.EXPECTED_POSITIONS}), encoding="utf-8"
    )
    checkpoint = tmp_path / "baseline.pt"
    content = b"fake checkpoint bytes"
    checkpoint.write_bytes(content)
    fake_sha = hashlib.sha256(content).hexdigest()

    monkeypatch.setattr(sweep_module, "REPLAY_DIR", replay_dir)
    monkeypatch.setattr(sweep_module, "BASELINE_CHECKPOINT", checkpoint)
    monkeypatch.setattr(sweep_module, "EXPECTED_BASELINE_SHA256", fake_sha)

    result = sweep_module.verify_reused_artifacts()
    assert result["replay_size"] == sweep_module.EXPECTED_POSITIONS
    assert result["baseline_sha256"] == fake_sha


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


def test_uses_shared_monopolyzero_common_module():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import monopolyzero_common as common" in source


def test_no_self_play_game_generation():
    """This sweep must not call play_local_game — it only reuses the
    existing replay buffer, generating zero new games."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "play_local_game" not in source


def test_each_budget_starts_fresh_from_baseline_not_resumed():
    source = SCRIPT.read_text(encoding="utf-8")
    # load_inference(BASELINE_CHECKPOINT) must be inside the per-budget loop
    # body, not called once outside it, so each budget starts fresh.
    loop_start = source.index("for budget in UPDATE_BUDGETS")
    per_budget_body = source[loop_start:]
    assert "MonopolyZeroNet.load_inference(BASELINE_CHECKPOINT)" in per_budget_body
    assert per_budget_body.count("MonopolyZeroNet.load_inference(BASELINE_CHECKPOINT)") == 1
