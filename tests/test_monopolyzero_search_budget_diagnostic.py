"""Focused tests for scripts/monopolyzero_search_budget_diagnostic.py: the
checkpoint-integrity gate, the disagreement/value-delta/visit-share math
helpers (pure functions over a fake per_state list, no engine/model
needed), and config/import-graph checks. Does not run the actual 200-state
x 3-tier diagnostic (see the experiment log for that result instead).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_search_budget_diagnostic.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_search_budget_diagnostic", SCRIPT)
diag_module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_search_budget_diagnostic"] = diag_module
_spec.loader.exec_module(diag_module)


def test_config_matches_task_spec():
    assert diag_module.EXPECTED_CHECKPOINT_SHA256 == (
        "152c0a0f6136d1fc91e74973ac245b2f72774694c424d2a48854514ed2848383"
    )
    assert diag_module.HELD_OUT_SEEDS == (31000, 31001, 31002, 31003, 31004)
    assert diag_module.TARGET_STATES == 200
    assert diag_module.SIMULATION_TIERS == (4, 16, 32)
    assert diag_module.MAX_DEPTH == 16


def test_verify_checkpoint_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(diag_module, "CHECKPOINT_PATH", tmp_path / "nope.pt")
    with pytest.raises(SystemExit) as excinfo:
        diag_module.verify_checkpoint()
    assert "missing checkpoint" in str(excinfo.value)


def test_verify_checkpoint_raises_on_sha_mismatch(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"some bytes")
    monkeypatch.setattr(diag_module, "CHECKPOINT_PATH", checkpoint)
    with pytest.raises(SystemExit) as excinfo:
        diag_module.verify_checkpoint()
    assert "SHA-256 mismatch" in str(excinfo.value)


def test_verify_checkpoint_passes_when_hash_matches(tmp_path, monkeypatch):
    import hashlib

    checkpoint = tmp_path / "checkpoint.pt"
    content = b"some checkpoint bytes"
    checkpoint.write_bytes(content)
    monkeypatch.setattr(diag_module, "CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(diag_module, "EXPECTED_CHECKPOINT_SHA256", hashlib.sha256(content).hexdigest())
    assert diag_module.verify_checkpoint() == hashlib.sha256(content).hexdigest()


def _fake_state(chosen_4, chosen_16, chosen_32, value_4=0.0, value_16=0.0, value_32=0.0, share_4=1.0):
    return {
        "tiers": {
            4: {"chosen_action": chosen_4, "root_value_self": value_4, "chosen_visit_share": share_4},
            16: {"chosen_action": chosen_16, "root_value_self": value_16, "chosen_visit_share": share_4},
            32: {"chosen_action": chosen_32, "root_value_self": value_32, "chosen_visit_share": share_4},
        }
    }


def test_disagreement_rate_all_agree_is_zero():
    states = [_fake_state(1, 1, 1) for _ in range(10)]
    assert diag_module.disagreement_rate(states, 4, 16) == 0.0


def test_disagreement_rate_all_disagree_is_one():
    states = [_fake_state(1, 2, 3) for _ in range(10)]
    assert diag_module.disagreement_rate(states, 4, 16) == 1.0
    assert diag_module.disagreement_rate(states, 16, 32) == 1.0
    assert diag_module.disagreement_rate(states, 4, 32) == 1.0


def test_disagreement_rate_partial():
    states = [_fake_state(1, 1, 1) for _ in range(8)] + [_fake_state(1, 2, 2) for _ in range(2)]
    assert diag_module.disagreement_rate(states, 4, 16) == pytest.approx(0.2)


def test_disagreement_rate_empty_is_zero():
    assert diag_module.disagreement_rate([], 4, 16) == 0.0


def test_mean_abs_value_delta():
    states = [
        _fake_state(1, 1, 1, value_4=0.5, value_16=0.6, value_32=0.6),
        _fake_state(1, 1, 1, value_4=0.3, value_16=0.3, value_32=0.9),
    ]
    assert diag_module.mean_abs_value_delta(states, 4, 16) == pytest.approx(0.05)
    assert diag_module.mean_abs_value_delta(states, 16, 32) == pytest.approx(0.3)


def test_mean_abs_value_delta_empty_is_none():
    assert diag_module.mean_abs_value_delta([], 4, 16) is None


def test_mean_chosen_visit_share():
    states = [_fake_state(1, 1, 1, share_4=0.8), _fake_state(1, 1, 1, share_4=0.4)]
    assert diag_module.mean_chosen_visit_share(states, 4) == pytest.approx(0.6)


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


def test_no_training_update_call():
    """This is a pure inference/search diagnostic - it must never call the
    training-update step."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "local_training_update" not in source


def test_does_not_persist_raw_state_arrays_in_output_record():
    """The dict appended to per_state (the one that ends up in the JSON
    payload) must stay small (action ids, floats) - it must not include the
    raw env_clone snapshot itself."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "per_state.append(" in source
    start = source.index("per_state.append(")
    end = source.index(")\n", start)
    # find the matching close of the dict literal, not just the first ")\n"
    depth = 0
    idx = start
    while True:
        if source[idx] == "(":
            depth += 1
        elif source[idx] == ")":
            depth -= 1
            if depth == 0:
                end = idx
                break
        idx += 1
    appended_block = source[start:end]
    assert "env_clone" not in appended_block
