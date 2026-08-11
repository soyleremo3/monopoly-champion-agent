"""ASU-import-free paired evaluation across the update-budget sweep:
0 (baseline), 100, 500, and 1000 updates, all on the SAME held-out seeds,
seat-rotated, vs. fixed-a/b/c. Never touches ASU.

Built on scripts/monopolyzero_common.py — no monopoly_bench.adapters/.arena
/.training import. Refuses to run unless PYTHONHASHSEED=0 is set and the
git tree is clean.
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

HELD_OUT_SEEDS = tuple(range(30000, 30010))  # 30000-30009
SIMULATIONS = 4
MAX_DEPTH = 16
MAX_ROUNDS = 200

REPO_ROOT = common.REPO_ROOT
PILOT_DIR = REPO_ROOT / "artifacts" / "monopolyzero_strength_pilot"

CHECKPOINTS = {
    "budget_0_baseline": PILOT_DIR / "baseline_pretraining.pt",
    "budget_100": PILOT_DIR / "trained_updates_100.pt",
    "budget_500": PILOT_DIR / "trained_updates_500.pt",
    "budget_1000": PILOT_DIR / "trained_updates_1000.pt",
}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_95_interval(wins: int, games: int) -> tuple[float, float]:
    """Standard closed-form Wilson score interval (Wilson, 1927)."""
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


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def non_overlapping_improvement(worse: dict, better: dict) -> bool:
    if not (worse.get("wilson_95") and better.get("wilson_95")):
        return False
    return better["wilson_95"][0] > worse["wilson_95"][1]


def _evaluate_checkpoint(model, search_config, fixed_pool, label: str) -> dict:
    from monopoly_bench.engine import NUM_PLAYERS

    per_game = []
    all_latencies: list[float] = []
    total_illegal = 0
    total_crashed = 0
    fallback_totals = {"fixed_a": 0, "fixed_b": 0, "fixed_c": 0}
    wins_by_seat = {seat: 0 for seat in range(NUM_PLAYERS)}
    games_by_seat = {seat: 0 for seat in range(NUM_PLAYERS)}

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

            focus_net_worth = outcome.final_net_worth[focus_seat] if outcome.final_net_worth else None
            games_by_seat[focus_seat] += 1
            focus_won = (outcome.winner == focus_seat) if outcome.completed else None
            if focus_won:
                wins_by_seat[focus_seat] += 1

            per_game.append(
                {
                    "seed": seed,
                    "focus_seat": focus_seat,
                    "completed": outcome.completed,
                    "winner": outcome.winner,
                    "focus_won": focus_won,
                    "rounds": outcome.final_round,
                    "decisions": outcome.decisions,
                    "net_worth": list(outcome.final_net_worth) if outcome.final_net_worth else None,
                    "focus_net_worth": focus_net_worth,
                    "round_capped": outcome.final_round >= MAX_ROUNDS,
                    "focus_bankrupt": (focus_net_worth is not None and focus_net_worth <= 0.0),
                    "illegal_actions": outcome.illegal_actions,
                    "crashed": outcome.crashed,
                    "error": outcome.error,
                }
            )

    completed_games = [game for game in per_game if game["completed"]]
    wins = sum(1 for game in completed_games if game["focus_won"])
    win_rate = wins / len(completed_games) if completed_games else None
    wilson = wilson_95_interval(wins, len(completed_games)) if completed_games else None

    net_worths = [game["focus_net_worth"] for game in completed_games if game["focus_net_worth"] is not None]
    mean_net_worth = sum(net_worths) / len(net_worths) if net_worths else None
    median_net_worth = _median(net_worths)
    bankruptcy_rate = (
        sum(1 for value in net_worths if value <= 0.0) / len(net_worths) if net_worths else None
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
        "median_net_worth": median_net_worth,
        "bankruptcy_rate": bankruptcy_rate,
        "round_cap_rate": round_cap_rate,
        "p95_search_latency_s": p95_latency,
        "total_illegal_actions": total_illegal,
        "total_crashed": total_crashed,
        "fixed_fallbacks": fallback_totals,
        "fixed_fallbacks_total": sum(fallback_totals.values()),
        "wins_by_seat": wins_by_seat,
        "games_by_seat": games_by_seat,
        "per_game": per_game,
    }


def main(argv: list[str] | None = None) -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)

    for label, path in CHECKPOINTS.items():
        if not path.is_file():
            raise SystemExit(f"Missing checkpoint for {label}: {path}")

    common.ensure_reference_on_path()
    import random

    import numpy as np
    import torch

    from monopoly_bench.config import SearchConfig
    from monopoly_bench.model import MonopolyZeroNet
    from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        search_config = SearchConfig(simulations=SIMULATIONS, max_depth=MAX_DEPTH)
        fixed_pool = {0: FP_AGENT_CLASSES[0], 1: FP_AGENT_CLASSES[1], 2: FP_AGENT_CLASSES[2]}

        checkpoint_hashes = {}
        results = {}
        for label, path in CHECKPOINTS.items():
            checkpoint_hashes[label] = _sha256(path)
            model = MonopolyZeroNet.load_inference(path)
            results[label] = _evaluate_checkpoint(model, search_config, fixed_pool, label)

        asu_modules_loaded = common.loaded_asu_modules()

    elapsed_s = time.perf_counter() - started

    pairs = {
        "100_vs_500": non_overlapping_improvement(results["budget_100"], results["budget_500"]),
        "500_vs_1000": non_overlapping_improvement(results["budget_500"], results["budget_1000"]),
    }

    payload = {
        "git_head_sha": git_head_sha,
        "config": {
            "held_out_seeds": list(HELD_OUT_SEEDS),
            "simulations": SIMULATIONS,
            "max_depth": MAX_DEPTH,
            "max_rounds": MAX_ROUNDS,
            "self_play": False,
            "opponents": ["fixed-a", "fixed-b", "fixed-c"],
            "seat_rotation": "4 seats x 10 seeds = 40 games per checkpoint",
        },
        "checkpoint_sha256": checkpoint_hashes,
        "results": results,
        "paired_improvement_statistically_supported": pairs,
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
