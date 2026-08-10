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


def _parse_seed_token(token: str) -> list[int]:
    """Parse one --seed token: a plain int, or an inclusive "START-END" range.

    "42" -> [42]; "10000-10009" -> [10000, ..., 10009]. A leading "-" (a
    negative literal, e.g. "-5") is not treated as a range.
    """
    if "-" in token[1:]:
        start_str, _, end_str = token.partition("-")
        try:
            start, end = int(start_str), int(end_str)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid seed or seed range: {token!r}")
        if end < start:
            raise argparse.ArgumentTypeError(
                f"seed range end must be >= start: {token!r}"
            )
        return list(range(start, end + 1))
    try:
        return [int(token)]
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid seed or seed range: {token!r}")


def _expand_seeds(tokens: list[str]) -> list[int]:
    """Expand --seed tokens into an ordered, de-duplicated list of seeds."""
    seeds: list[int] = []
    for token in tokens:
        for seed in _parse_seed_token(token):
            if seed not in seeds:
                seeds.append(seed)
    return seeds


def main(argv: list[str] | None = None) -> int:
    _require_pinned_hash_seed()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", nargs="+", default=["42"], dest="seed",
        help="one or more seeds and/or START-END ranges, "
        "e.g. --seed 42  or  --seed 10000-10009  or  --seed 1 2 10000-10002",
    )
    parser.add_argument("--focus", default=DEFAULT_FOCUS)
    parser.add_argument(
        "--opponents", nargs=3, default=DEFAULT_OPPONENTS,
        metavar=("AGENT_A", "AGENT_B", "AGENT_C"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    seeds = _expand_seeds(args.seed)

    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))
    from ASU_FROZEN_TEACHER.evaluate import evaluate_lineup

    result = evaluate_lineup(args.focus, tuple(args.opponents), tuple(seeds))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
