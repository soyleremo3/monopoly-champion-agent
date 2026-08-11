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
