"""Measure MaxNPUCT search runtime at several simulation counts, ASU-independent.

Reuses the exact state-construction code monopoly_bench.cli.smoke() uses
(SharedGame.new(23, max_rounds=2), same property-ownership setup, same
decision seed 101) so each simulation count searches from an identical
starting state. No ASU import anywhere in this script or its call graph
(MonopolyZeroNet, MaxNPUCT, SearchConfig, SharedGame carry no ASU coupling —
see docs/REFERENCE_AUDIT.md). Requires a PPO-compatible checkpoint at
references/DeepRL_Monopoly/artifacts/ppo_plus/ppo_hybrid_2000_v2.pt (the
same hardcoded default monopoly_bench.cli.smoke() uses).

Refuses to run unless PYTHONHASHSEED=0 is set, for the same reproducibility
reason as scripts/run_baseline_match.py (see
docs/REFERENCE_AUDIT.md#critical-issue-found).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "references" / "DeepRL_Monopoly"
DEFAULT_PPO = REFERENCE_ROOT / "artifacts" / "ppo_plus" / "ppo_hybrid_2000_v2.pt"

REQUIRED_HASH_SEED = "0"


def _require_pinned_hash_seed() -> None:
    actual = os.environ.get("PYTHONHASHSEED")
    if actual == REQUIRED_HASH_SEED:
        return
    shown = "unset" if actual is None else repr(actual)
    raise SystemExit(
        "monopolyzero_sim_runtime.py refuses to run: PYTHONHASHSEED must be "
        f"'{REQUIRED_HASH_SEED}', got {shown}. Re-run as:\n"
        f"  PYTHONHASHSEED={REQUIRED_HASH_SEED} python {Path(__file__).name}"
    )


def main(argv: list[str] | None = None) -> int:
    _require_pinned_hash_seed()

    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))
    from monopoly_bench.config import SearchConfig
    from monopoly_bench.engine import ActionType, SharedGame
    from monopoly_bench.model import MonopolyZeroNet
    from monopoly_bench.search import MaxNPUCT

    if not DEFAULT_PPO.is_file():
        raise SystemExit(f"Missing PPO-compatible checkpoint: {DEFAULT_PPO}")

    model = MonopolyZeroNet()
    model.load_ppo_actor(DEFAULT_PPO)

    results = []
    for simulations in (4, 16, 32):
        game = SharedGame.new(23, max_rounds=2)
        actor = game.env.whose_turn()
        property_ = game.env.properties[1]
        property_.owner = actor
        game.env.players[actor].properties.append(property_)

        result = MaxNPUCT(
            model, SearchConfig(simulations=simulations, max_depth=16)
        ).choose_action(game, actor, 101)

        legal = game.env.get_allowed_actions(actor)
        illegal = result.chosen_action not in legal
        results.append(
            {
                "simulations": simulations,
                "requested_simulations": simulations,
                "actual_simulations": result.simulations,
                "chosen_action": result.chosen_action,
                "legal": not illegal,
                "latency_s": result.latency_s,
            }
        )
        if illegal:
            raise RuntimeError(
                f"simulations={simulations}: search selected illegal action "
                f"{result.chosen_action}; legal={legal}"
            )

    payload = {"ruleset": "ppo-plus-v2", "decision_seed": 101, "results": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
