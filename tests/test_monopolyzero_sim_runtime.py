"""Focused test for the PYTHONHASHSEED guard in
scripts/monopolyzero_sim_runtime.py.

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
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_sim_runtime.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_sim_runtime", SCRIPT)
monopolyzero_sim_runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monopolyzero_sim_runtime)


@pytest.mark.parametrize("value", [None, "", "1", "2", "random"])
def test_guard_rejects_unpinned_hash_seed(value, monkeypatch):
    if value is None:
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    else:
        monkeypatch.setenv("PYTHONHASHSEED", value)
    with pytest.raises(SystemExit) as excinfo:
        monopolyzero_sim_runtime._require_pinned_hash_seed()
    assert "PYTHONHASHSEED=0" in str(excinfo.value)


def test_guard_accepts_pinned_hash_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monopolyzero_sim_runtime._require_pinned_hash_seed()


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
