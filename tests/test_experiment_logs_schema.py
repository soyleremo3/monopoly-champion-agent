"""Validates every logs/experiments/*.json entry against schema.json.

Requires the `jsonschema` package (small, pure Python, already installed in
the project venv alongside psutil/pytest for the same reason).
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs" / "experiments"
SCHEMA_PATH = LOGS_DIR / "schema.json"

_schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
_log_paths = sorted(LOGS_DIR.glob("[0-9][0-9][0-9]-*.json"))


def test_schema_itself_is_valid_draft7():
    jsonschema.Draft7Validator.check_schema(_schema)


def test_at_least_one_log_entry_exists():
    assert _log_paths, f"no NNN-*.json entries found under {LOGS_DIR}"


@pytest.mark.parametrize("path", _log_paths, ids=lambda path: path.stem)
def test_log_entry_matches_schema(path):
    entry = json.loads(path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft7Validator(_schema)
    errors = sorted(validator.iter_errors(entry), key=lambda error: list(error.path))
    assert not errors, "\n".join(
        f"{path.name}: {'.'.join(str(p) for p in error.path)}: {error.message}"
        for error in errors
    )


@pytest.mark.parametrize("path", _log_paths, ids=lambda path: path.stem)
def test_experiment_id_matches_filename(path):
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert entry["experiment_id"] == path.stem


@pytest.mark.parametrize("path", _log_paths, ids=lambda path: path.stem)
def test_raw_logs_paths_exist_when_declared(path):
    entry = json.loads(path.read_text(encoding="utf-8"))
    for raw_path in entry.get("raw_logs") or []:
        if raw_path is None:
            continue
        assert (REPO_ROOT / raw_path).is_file(), (
            f"{path.name} declares raw_logs entry {raw_path!r} that doesn't exist"
        )


def test_experiment_ids_are_unique():
    ids = [json.loads(path.read_text(encoding="utf-8"))["experiment_id"] for path in _log_paths]
    assert len(ids) == len(set(ids)), f"duplicate experiment_id(s): {ids}"


# ── code_commit_sha semantics (corrected 2026-08-12) ────────────────────────
# code_commit_sha must be the clean git HEAD SHA at run time, never "the
# commit that recorded the results". Entries 001-010 predate the
# clean-tree-before-run discipline, so they were corrected to null.

_PRE_DISCIPLINE_IDS = {
    "001-fixed-agent-engine-smoke",
    "002-ddqn-20-game-training-smoke",
    "003-ddqn-reproducibility-check",
    "004-ddqn-500-game-training",
    "005-ddqn-20-vs-500-paired-evaluation",
    "006-asu-evaluation-only-benchmark",
    "007-ppo-1-game-compatibility-checkpoint",
    "008-monopolyzero-inference-smoke",
    "009-monopolyzero-puct-sim-runtime",
    "010-selfplay-training-plumbing-smoke",
}


@pytest.mark.parametrize(
    "path",
    [path for path in _log_paths if path.stem in _PRE_DISCIPLINE_IDS],
    ids=lambda path: path.stem,
)
def test_pre_discipline_entries_have_null_code_commit_sha(path):
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert entry["code_commit_sha"] is None, (
        f"{path.name}: predates the clean-tree-before-run discipline, "
        "code_commit_sha must be null, not a 'recorded the results' commit"
    )
