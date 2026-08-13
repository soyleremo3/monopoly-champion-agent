"""Creates a LOCAL, gitignored friend-match export bundle for the
verified A96 champion, for manual upload to another machine. Never
commits anything - `dist/` is gitignored and this script never runs
`git add`.

Before copying anything: verifies the source checkpoint file exists,
verifies its exact SHA-256, loads it through `A96FriendMatchAgent`
(which itself verifies the exact actor SHA-256 via
`FrozenPurePPOPolicy.from_checkpoint`) - refuses to export (raises,
non-zero exit) on any mismatch rather than exporting an unverified file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from a96_friend_match_agent import (  # noqa: E402
    A96_ACTOR_SHA256,
    A96_CHECKPOINT_FILENAME,
    A96_CHECKPOINT_SHA256,
    A96_REFERENCE_SUBMODULE_SHA,
    A96FriendMatchAgent,
    DEFAULT_CHECKPOINT_PATH,
)

DIST_DIR = REPO_ROOT / "dist" / "a96_friend_match"
EXPORTED_CHECKPOINT_NAME = "a96_champion.pt"

DEPENDENCY_PINS = ["torch==2.13.0", "numpy==2.5.2", "psutil==7.2.2", "jsonschema==4.26.0", "pytest==9.1.1"]
EXPECTED_PYTHON_VERSION = "3.12.10"
ENTRYPOINT = "scripts.a96_friend_match_agent:A96FriendMatchAgent"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _current_git_head_sha() -> str | None:
    """Best-effort - not a clean-tree gate (this export tool must remain
    usable both before and after a commit, unlike an experiment run)."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip() or None
    except Exception:
        return None


def export_bundle(source_checkpoint: Path | None = None) -> Path:
    source_checkpoint = Path(source_checkpoint) if source_checkpoint is not None else DEFAULT_CHECKPOINT_PATH

    if not source_checkpoint.is_file():
        raise SystemExit(f"STOP: source checkpoint missing at {source_checkpoint}")

    actual_sha256 = _file_sha256(source_checkpoint)
    if actual_sha256 != A96_CHECKPOINT_SHA256:
        raise SystemExit(
            f"STOP: source checkpoint sha256 mismatch - got {actual_sha256}, "
            f"expected {A96_CHECKPOINT_SHA256}. Refusing to export an unverified checkpoint."
        )

    # Loads (and hash-gates the actor state) via the same wrapper a real
    # friend-match run would use - refuses to export if this fails.
    A96FriendMatchAgent(checkpoint_path=source_checkpoint)

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    exported_path = DIST_DIR / EXPORTED_CHECKPOINT_NAME
    shutil.copy2(source_checkpoint, exported_path)

    exported_sha256 = _file_sha256(exported_path)
    if exported_sha256 != A96_CHECKPOINT_SHA256:
        raise SystemExit(
            f"STOP: exported checkpoint sha256 mismatch after copy - got {exported_sha256}, "
            f"expected {A96_CHECKPOINT_SHA256}. The copy is not byte-identical; refusing to leave "
            "a corrupted export in place."
        )

    manifest = {
        "source_main_commit_sha": _current_git_head_sha(),
        "checkpoint_filename": A96_CHECKPOINT_FILENAME,
        "exported_checkpoint_filename": EXPORTED_CHECKPOINT_NAME,
        "checkpoint_sha256": A96_CHECKPOINT_SHA256,
        "actor_sha256": A96_ACTOR_SHA256,
        "checkpoint_size_bytes": exported_path.stat().st_size,
        "python_version_expected": EXPECTED_PYTHON_VERSION,
        "dependency_pins": DEPENDENCY_PINS,
        "reference_submodule_sha": A96_REFERENCE_SUBMODULE_SHA,
        "entrypoint": ENTRYPOINT,
        "harness_must_provide": "a state vector (as produced by the pinned reference engine's own state encoding) and legal_action_ids for the current decision - this bundle does not include a game engine or opponent logic.",
    }
    manifest_path = DIST_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    readme_path = DIST_DIR / "README.txt"
    readme_path.write_text(_README_TEXT, encoding="utf-8")

    return exported_path


_README_TEXT = """A96 friend-match bundle - usage
================================

1. Code: check out `main` from this project's repository, then
   initialize the reference submodule:
       git submodule update --init --recursive

2. Install the pinned dependencies (from the repo root):
       pip install -r requirements.txt

3. Point the agent at this bundle's checkpoint (copy a96_champion.pt
   wherever you like, then set the environment variable):
       A96_CHECKPOINT_PATH=/path/to/a96_champion.pt   (Linux/macOS)
       $env:A96_CHECKPOINT_PATH = "C:\\path\\to\\a96_champion.pt"   (PowerShell)

4. Instantiate the agent (from the repo root, or with scripts/ on
   sys.path):
       from a96_friend_match_agent import A96FriendMatchAgent
       agent = A96FriendMatchAgent()   # reads A96_CHECKPOINT_PATH if set

5. Call it once per decision:
       action = agent.act(state, legal_action_ids)

See manifest.json for the exact checkpoint/actor SHA-256, dependency
pins, and reference submodule SHA this bundle was built and verified
against. The calling harness (your game loop / the official competition
harness) is responsible for producing `state` and `legal_action_ids` -
this bundle contains no game engine or opponent logic of its own.
"""


def main() -> int:
    exported_path = export_bundle()
    print(f"EXPORT OK: {exported_path} ({exported_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
