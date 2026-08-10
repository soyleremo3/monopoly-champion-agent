"""Rerun the fixed-agent 4-player ppo-plus-v2 baseline match.

Thin wrapper around ASU_FROZEN_TEACHER.evaluate.evaluate_lineup from the
references/DeepRL_Monopoly submodule. No game logic is copied here; the
submodule is imported at runtime from its checked-out path.

Requires numpy + torch (CPU) installed in the active Python environment,
because importing monopoly_game_engine pulls in torch transitively even for
scripted-only lineups.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "references" / "DeepRL_Monopoly"
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

from ASU_FROZEN_TEACHER.evaluate import evaluate_lineup  # noqa: E402

DEFAULT_FOCUS = "fixed-d"
DEFAULT_OPPONENTS = ("fixed-a", "fixed-b", "fixed-c")


def main(argv: list[str] | None = None) -> int:
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
