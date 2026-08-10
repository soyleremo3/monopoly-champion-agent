"""Rerun a 4-player ppo-plus-v2 match (fixed agents and/or a checkpoint).

Thin wrapper around ASU_FROZEN_TEACHER.evaluate.evaluate_lineup from the
references/DeepRL_Monopoly submodule. No game logic is copied here; the
submodule is imported at runtime from its checked-out path.

Requires numpy + torch (CPU) installed in the active Python environment,
because importing monopoly_game_engine pulls in torch transitively even for
scripted-only lineups.

Refuses to run unless PYTHONHASHSEED=0 is set. See
docs/REFERENCE_AUDIT.md#critical-issue-found: without a pinned hash seed,
the reference engine's "seeded" games are not reproducible across separate
process launches (confirmed by running the same seed twice and diffing the
output).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "references" / "DeepRL_Monopoly"

DEFAULT_FOCUS = "fixed-d"
DEFAULT_OPPONENTS = ("fixed-a", "fixed-b", "fixed-c")

REQUIRED_HASH_SEED = "0"


def _require_pinned_hash_seed() -> None:
    """Refuse to run unless PYTHONHASHSEED is pinned to REQUIRED_HASH_SEED.

    Checked before any heavy import (torch/numpy/the reference submodule) so
    the failure is immediate and does not depend on those being installed.
    """
    actual = os.environ.get("PYTHONHASHSEED")
    if actual == REQUIRED_HASH_SEED:
        return
    shown = "unset" if actual is None else repr(actual)
    raise SystemExit(
        "run_baseline_match.py refuses to run: PYTHONHASHSEED must be "
        f"'{REQUIRED_HASH_SEED}' for reproducible results, got {shown}.\n"
        "Reason: references/DeepRL_Monopoly's seeded games are not "
        "reproducible across separate process launches without a pinned "
        "hash seed (see docs/REFERENCE_AUDIT.md#critical-issue-found).\n"
        "Re-run as:\n"
        f"  PYTHONHASHSEED={REQUIRED_HASH_SEED} python "
        f"{Path(__file__).name} ...\n"
        "(on Windows PowerShell: "
        f"$env:PYTHONHASHSEED='{REQUIRED_HASH_SEED}'; python "
        f"{Path(__file__).name} ...)"
    )


def main(argv: list[str] | None = None) -> int:
    _require_pinned_hash_seed()

    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))
    from ASU_FROZEN_TEACHER.evaluate import evaluate_lineup

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focus", default=DEFAULT_FOCUS)
    parser.add_argument(
        "--opponents", nargs=3, default=DEFAULT_OPPONENTS,
        metavar=("AGENT_A", "AGENT_B", "AGENT_C"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    result = evaluate_lineup(args.focus, tuple(args.opponents), (args.seed,))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
