"""ASU-import-free paired evaluation: pre-training baseline vs. trained
MonopolyZero checkpoint, on the same held-out seeds, seat-rotated, vs.
fixed-a/b/c. Never touches ASU (no evaluate_lineup/ASU_FROZEN_TEACHER —
that CLI doesn't know about MonopolyZeroNet checkpoints anyway).

Built on scripts/monopolyzero_common.py — see that module's docstring for
the import-graph and source-similarity audit notes. No ASU import.

Refuses to run unless PYTHONHASHSEED=0 is set, refuses on a dirty git tree.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import monopolyzero_common as common  # noqa: E402

HELD_OUT_SEEDS = (30000, 30001, 30002, 30003, 30004)
SIMULATIONS = 4
MAX_DEPTH = 16
MAX_ROUNDS = 200

REPO_ROOT = common.REPO_ROOT
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "monopolyzero_strength_pilot"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_95_interval(wins: int, games: int) -> tuple[float, float]:
    """Standard closed-form Wilson score interval (Wilson, 1927) — a fixed
    textbook formula, not reference-specific code."""
    if games <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    phat = wins / games
    denominator = 1 + z * z / games
    center = phat + z * z / (2 * games)
    margin = z * math.sqrt(phat * (1 - phat) / games + z * z / (4 * games * games))
    return ((center - margin) / denominator, (center + margin) / denominator)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def _evaluate_checkpoint(model, search_config, fixed_pool, label: str) -> dict:
    from monopoly_bench.engine import NUM_PLAYERS

    per_game = []
    all_latencies: list[float] = []
    total_illegal = 0
    total_crashed = 0
    fallback_totals = {"fixed_a": 0, "fixed_b": 0, "fixed_c": 0}

    for seed in HELD_OUT_SEEDS:
        for focus_seat in range(NUM_PLAYERS):
            non_focus_seats = [seat for seat in range(NUM_PLAYERS) if seat != focus_seat]
            fixed_a = common.LocalFixedPolicy(fixed_pool[0])
            fixed_b = common.LocalFixedPolicy(fixed_pool[1])
            fixed_c = common.LocalFixedPolicy(fixed_pool[2])
            fixed_by_seat = dict(zip(non_focus_seats, (fixed_a, fixed_b, fixed_c)))

            policies = {focus_seat: common.build_local_search_policy(model, search_config, self_play=False)}
            policies.update(fixed_by_seat)

            outcome = common.play_local_game(
                game_id=seed * 10 + focus_seat, seed=seed, policies=policies,
                max_rounds=MAX_ROUNDS, record_seats=set(),
            )
            total_illegal += outcome.illegal_actions
            total_crashed += int(outcome.crashed)
            all_latencies.extend(outcome.search_latencies_s)
            fallback_totals["fixed_a"] += fixed_a.fallback_count
            fallback_totals["fixed_b"] += fixed_b.fallback_count
            fallback_totals["fixed_c"] += fixed_c.fallback_count

            per_game.append(
                {
                    "seed": seed,
                    "focus_seat": focus_seat,
                    "completed": outcome.completed,
                    "winner": outcome.winner,
                    "focus_won": (outcome.winner == focus_seat) if outcome.completed else None,
                    "rounds": outcome.final_round,
                    "decisions": outcome.decisions,
                    "net_worth": list(outcome.final_net_worth) if outcome.final_net_worth else None,
                    "focus_net_worth": outcome.final_net_worth[focus_seat] if outcome.final_net_worth else None,
                    "round_capped": outcome.final_round >= MAX_ROUNDS,
                    "illegal_actions": outcome.illegal_actions,
                    "crashed": outcome.crashed,
                    "error": outcome.error,
                }
            )

    completed_games = [game for game in per_game if game["completed"]]
    wins = sum(1 for game in completed_games if game["focus_won"])
    win_rate = wins / len(completed_games) if completed_games else None
    wilson = wilson_95_interval(wins, len(completed_games)) if completed_games else None
    mean_net_worth = (
        sum(game["focus_net_worth"] for game in completed_games) / len(completed_games)
        if completed_games else None
    )
    round_cap_rate = sum(1 for game in per_game if game["round_capped"]) / len(per_game)
    p95_latency = _percentile(all_latencies, 95)

    return {
        "label": label,
        "games": len(per_game),
        "completed_games": len(completed_games),
        "wins": wins,
        "win_rate": win_rate,
        "wilson_95": list(wilson) if wilson else None,
        "mean_net_worth": mean_net_worth,
        "round_cap_rate": round_cap_rate,
        "p95_search_latency_s": p95_latency,
        "total_illegal_actions": total_illegal,
        "total_crashed": total_crashed,
        "fixed_fallbacks": fallback_totals,
        "fixed_fallbacks_total": sum(fallback_totals.values()),
        "per_game": per_game,
    }


def main(argv: list[str] | None = None) -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)

    common.ensure_reference_on_path()
    import random

    import numpy as np
    import torch

    from monopoly_bench.config import SearchConfig
    from monopoly_bench.model import MonopolyZeroNet
    from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES

    baseline_path = ARTIFACT_DIR / "baseline_pretraining.pt"
    trained_path = ARTIFACT_DIR / "trained_32games.pt"
    for path in (baseline_path, trained_path):
        if not path.is_file():
            raise SystemExit(f"Missing checkpoint: {path} (run monopolyzero_strength_train.py first)")

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        search_config = SearchConfig(simulations=SIMULATIONS, max_depth=MAX_DEPTH)
        fixed_pool = {0: FP_AGENT_CLASSES[0], 1: FP_AGENT_CLASSES[1], 2: FP_AGENT_CLASSES[2]}

        baseline_model = MonopolyZeroNet.load_inference(baseline_path)
        baseline_result = _evaluate_checkpoint(baseline_model, search_config, fixed_pool, "baseline")

        trained_model = MonopolyZeroNet.load_inference(trained_path)
        trained_result = _evaluate_checkpoint(trained_model, search_config, fixed_pool, "trained")

        asu_modules_loaded = common.loaded_asu_modules()

    elapsed_s = time.perf_counter() - started

    improvement_supported = False
    if baseline_result["wilson_95"] and trained_result["wilson_95"]:
        # Non-overlapping Wilson intervals, trained above baseline.
        improvement_supported = trained_result["wilson_95"][0] > baseline_result["wilson_95"][1]

    payload = {
        "git_head_sha": git_head_sha,
        "config": {
            "held_out_seeds": list(HELD_OUT_SEEDS),
            "simulations": SIMULATIONS,
            "max_depth": MAX_DEPTH,
            "max_rounds": MAX_ROUNDS,
            "self_play": False,
            "opponents": ["fixed-a", "fixed-b", "fixed-c"],
            "seat_rotation": "4 seats x 5 seeds = 20 games per checkpoint",
        },
        "baseline_checkpoint_sha256": _sha256(baseline_path),
        "trained_checkpoint_sha256": _sha256(trained_path),
        "baseline": baseline_result,
        "trained": trained_result,
        "improvement_statistically_supported": improvement_supported,
        "asu_modules_loaded": asu_modules_loaded,
        "asu_modules_loaded_count": len(asu_modules_loaded),
        "elapsed_s": elapsed_s,
        "peak_rss_gib": rss.peak_gib,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded during evaluation: {asu_modules_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
