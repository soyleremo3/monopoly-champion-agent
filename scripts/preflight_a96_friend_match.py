"""Fail-closed preflight for an A96 friend-match run: checks Python
version, dependency versions, reference submodule presence/pin,
checkpoint presence/hash, real actor hash after load, hybrid/fixed-mask
rejection state, and one deterministic legal-masked inference - then
prints a compact PASS/FAIL table. Prints "A96 FRIEND MATCH CORE READY"
and exits 0 only if every check passes; otherwise exits non-zero and
that line is never printed.

Uses only a tiny reference-engine state/legal-action set for the
inference check (SharedGame.new + one act() call) - never plays a full
game, never consumes a DEV/PROMOTION/FINAL_BLIND seed (the smoke seed
below is a plain, non-experiment integer, not drawn from any registered
pool).
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from a96_friend_match_agent import (  # noqa: E402
    A96_ACTOR_SHA256,
    A96_CHECKPOINT_SHA256,
    A96_REFERENCE_SUBMODULE_SHA,
    A96FriendMatchAgent,
    resolve_checkpoint_path,
)

EXPECTED_PYTHON_VERSION = "3.12.10"
EXPECTED_DEPENDENCY_VERSIONS = {"torch": "2.13.0", "numpy": "2.5.2", "psutil": "7.2.2", "jsonschema": "4.26.0", "pytest": "9.1.1"}
REFERENCE_SUBMODULE_DIR = REPO_ROOT / "references" / "DeepRL_Monopoly"
PREFLIGHT_SMOKE_SEED = 424242  # plain integer, not from any registered DEV/PROMOTION/FINAL_BLIND pool


class Check:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.detail = ""

    def ok(self, detail: str = "") -> "Check":
        self.passed = True
        self.detail = detail
        return self

    def fail(self, detail: str) -> "Check":
        self.passed = False
        self.detail = detail
        return self


def check_python_version() -> Check:
    check = Check("python_version")
    actual = platform.python_version()
    if actual == EXPECTED_PYTHON_VERSION:
        return check.ok(actual)
    return check.fail(f"got {actual}, expected {EXPECTED_PYTHON_VERSION}")


def _version_matches(actual: str, expected: str) -> bool:
    """Exact match, or `expected` followed by a local-version-identifier
    suffix (e.g. torch's CPU-only wheel reports "2.13.0+cpu" for the
    "2.13.0" pin - a standard, expected suffix, not a version mismatch)."""
    return actual == expected or actual.startswith(expected + "+")


def check_dependency_versions() -> Check:
    check = Check("dependency_versions")
    mismatches = []
    for package, expected in EXPECTED_DEPENDENCY_VERSIONS.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            mismatches.append(f"{package}: NOT INSTALLED")
            continue
        if not _version_matches(actual, expected):
            mismatches.append(f"{package}: got {actual}, expected {expected}")
    if mismatches:
        return check.fail("; ".join(mismatches))
    return check.ok(", ".join(f"{p}=={v}" for p, v in EXPECTED_DEPENDENCY_VERSIONS.items()))


def check_submodule_exists() -> Check:
    check = Check("reference_submodule_exists")
    if REFERENCE_SUBMODULE_DIR.is_dir() and any(REFERENCE_SUBMODULE_DIR.iterdir()):
        return check.ok(str(REFERENCE_SUBMODULE_DIR))
    return check.fail(f"missing or empty: {REFERENCE_SUBMODULE_DIR} (run: git submodule update --init --recursive)")


def check_submodule_sha(submodule_present: bool) -> Check:
    check = Check("reference_submodule_sha")
    if not submodule_present:
        return check.fail("skipped - submodule not present")
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REFERENCE_SUBMODULE_DIR, capture_output=True, text=True, check=True,
        )
        actual = result.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return check.fail(f"could not read submodule HEAD: {exc}")
    if actual == A96_REFERENCE_SUBMODULE_SHA:
        return check.ok(actual)
    return check.fail(f"got {actual}, expected {A96_REFERENCE_SUBMODULE_SHA}")


def check_checkpoint_exists() -> tuple[Check, Path]:
    check = Check("checkpoint_exists")
    path = resolve_checkpoint_path(None)
    if path.is_file():
        return check.ok(str(path)), path
    return check.fail(f"missing: {path}"), path


def check_checkpoint_sha(checkpoint_path: Path, exists: bool) -> Check:
    check = Check("checkpoint_sha256")
    if not exists:
        return check.fail("skipped - checkpoint missing")
    import hashlib

    actual = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if actual == A96_CHECKPOINT_SHA256:
        return check.ok(actual)
    return check.fail(f"got {actual}, expected {A96_CHECKPOINT_SHA256}")


def check_load_and_metadata(checkpoint_path: Path, checkpoint_ok: bool):
    """Returns (agent_or_None, checks: list[Check]). Loads twice by
    necessity: once via A96FriendMatchAgent (the actual real inference
    path used below), and once via a plain PPOAgent construction purely
    to read hybrid/fixed_action_mask metadata that
    frozen_ppo_inference.load_frozen_actor intentionally does not expose
    on its return value (it returns only the bare ActorNetwork) -
    frozen_ppo_inference.py is not modified to add that. Both loads use
    the identical checkpoint path and the reference's own unmodified
    PPOAgent.load; no second policy or decision path is introduced."""
    actor_sha_check = Check("actor_sha256")
    hybrid_check = Check("hybrid_is_false")
    mask_check = Check("fixed_action_mask_all_false")

    if not checkpoint_ok:
        actor_sha_check.fail("skipped - checkpoint not verified")
        hybrid_check.fail("skipped - checkpoint not verified")
        mask_check.fail("skipped - checkpoint not verified")
        return None, [actor_sha_check, hybrid_check, mask_check]

    try:
        agent_wrapper = A96FriendMatchAgent(checkpoint_path=checkpoint_path)
    except Exception as exc:  # noqa: BLE001
        actor_sha_check.fail(str(exc))
        hybrid_check.fail("skipped - load failed")
        mask_check.fail("skipped - load failed")
        return None, [actor_sha_check, hybrid_check, mask_check]
    actor_sha_check.ok(A96_ACTOR_SHA256)  # from_checkpoint already hard-gated this; reaching here means it matched

    import monopolyzero_common as common

    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    metadata_agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    metadata_agent.load(str(checkpoint_path))
    if metadata_agent.hybrid is False:
        hybrid_check.ok("False")
    else:
        hybrid_check.fail(f"got {metadata_agent.hybrid}")
    if not bool(metadata_agent.fixed_action_mask.any()):
        mask_check.ok("all False")
    else:
        mask_check.fail("fixed_action_mask has at least one True entry")

    return agent_wrapper, [actor_sha_check, hybrid_check, mask_check]


def check_inference(agent) -> Check:
    check = Check("one_action_legal_inference")
    if agent is None:
        return check.fail("skipped - agent not loaded")
    try:
        import monopolyzero_common as common

        common.ensure_reference_on_path()
        from monopoly_bench.engine import SharedGame

        game = SharedGame.new(PREFLIGHT_SMOKE_SEED, 200)
        seat = game.env.whose_turn()
        legal = tuple(game.env.get_allowed_actions(seat))
        state = game.env._get_state(seat)
        action = agent.act(state, legal)
        if action in legal:
            return check.ok(f"action={action}, legal_count={len(legal)}")
        return check.fail(f"ILLEGAL ACTION RETURNED: {action} not in {legal}")
    except Exception as exc:  # noqa: BLE001
        return check.fail(f"{type(exc).__name__}: {exc}")


def main() -> int:
    checks: list[Check] = []

    checks.append(check_python_version())
    checks.append(check_dependency_versions())

    submodule_exists_check = check_submodule_exists()
    checks.append(submodule_exists_check)
    checks.append(check_submodule_sha(submodule_exists_check.passed))

    checkpoint_exists_check, checkpoint_path = check_checkpoint_exists()
    checks.append(checkpoint_exists_check)
    checkpoint_sha_check = check_checkpoint_sha(checkpoint_path, checkpoint_exists_check.passed)
    checks.append(checkpoint_sha_check)

    agent, load_checks = check_load_and_metadata(checkpoint_path, checkpoint_sha_check.passed)
    checks.extend(load_checks)
    checks.append(check_inference(agent))

    name_width = max(len(c.name) for c in checks)
    print(f"{'CHECK'.ljust(name_width)}  RESULT  DETAIL")
    for c in checks:
        result = "PASS" if c.passed else "FAIL"
        print(f"{c.name.ljust(name_width)}  {result:<6}  {c.detail}")

    all_passed = all(c.passed for c in checks)
    if all_passed:
        print("\nA96 FRIEND MATCH CORE READY")
        return 0
    print("\nA96 FRIEND MATCH NOT READY")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
