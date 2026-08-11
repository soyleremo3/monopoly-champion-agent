"""ASU-import-free ablation: does PUCT search add real value over bare
policy-head inference, for the SAME 500-update checkpoint? Generates ZERO
new training data and trains nothing.

Two inference policies share one checkpoint:
  - POLICY_ONLY: legal mask + policy-head softmax + legal argmax, no
    search at all (common.build_local_policy_only).
  - PUCT_4: the existing 4-simulation, depth-16, self_play=False search
    (common.build_local_search_policy) already used by every prior
    MonopolyZero evaluation in this project.

Both are evaluated on the SAME held-out seeds, 4-seat rotation, vs.
fixed-a/b/c, max_rounds=200 (40 games each). During PUCT_4's games,
POLICY_ONLY is additionally shadow-queried at every non-forced decision
the checkpoint faces (common.play_local_game's shadow_policy hook) - asked
what it would have chosen at that exact frozen state, without acting on
it - so the action-disagreement rate is measured on literally the same
decision states PUCT_4's own win-rate numbers come from, not a separate
offline sample.

Built on scripts/monopolyzero_common.py - no monopoly_bench.adapters/.arena
/.training import, no ASU. Refuses to run unless PYTHONHASHSEED=0 is set
and the git tree is clean, and refuses to run unless the 500-update
checkpoint's SHA-256 matches the value recorded in 015/016's experiment
logs.
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

EXPECTED_CHECKPOINT_SHA256 = "152c0a0f6136d1fc91e74973ac245b2f72774694c424d2a48854514ed2848383"

HELD_OUT_SEEDS = tuple(range(32000, 32010))  # 32000-32009
SIMULATIONS = 4
MAX_DEPTH = 16
MAX_ROUNDS = 200

REPO_ROOT = common.REPO_ROOT
PILOT_DIR = REPO_ROOT / "artifacts" / "monopolyzero_strength_pilot"
CHECKPOINT_PATH = PILOT_DIR / "trained_updates_500.pt"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint() -> str:
    if not CHECKPOINT_PATH.is_file():
        raise SystemExit(
            f"monopolyzero_policy_only_vs_puct_eval.py refuses to run: "
            f"missing checkpoint {CHECKPOINT_PATH}"
        )
    actual = _sha256(CHECKPOINT_PATH)
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit(
            "monopolyzero_policy_only_vs_puct_eval.py refuses to run: "
            f"checkpoint SHA-256 mismatch. Got {actual}, expected "
            f"{EXPECTED_CHECKPOINT_SHA256}."
        )
    return actual


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


def disagreement_rate(shadow_decisions: list[dict]) -> float | None:
    if not shadow_decisions:
        return None
    mismatches = sum(1 for decision in shadow_decisions if not decision["agree"])
    return mismatches / len(shadow_decisions)


def _evaluate_policy(policy_builder, fixed_pool, label: str, *, shadow_builder=None) -> dict:
    from monopoly_bench.engine import NUM_PLAYERS

    per_game = []
    all_latencies: list[float] = []
    all_shadow_decisions: list[dict] = []
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

            policies = {focus_seat: policy_builder()}
            policies.update(fixed_by_seat)
            shadow_policy = shadow_builder() if shadow_builder else None

            outcome = common.play_local_game(
                game_id=seed * 10 + focus_seat, seed=seed, policies=policies,
                max_rounds=MAX_ROUNDS, record_seats=set(),
                shadow_policy=shadow_policy,
                shadow_seats={focus_seat} if shadow_policy is not None else None,
            )
            total_illegal += outcome.illegal_actions
            total_crashed += int(outcome.crashed)
            all_latencies.extend(outcome.search_latencies_s)
            all_shadow_decisions.extend(outcome.shadow_decisions)
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
        "decision_latency_s": {
            "p50": _percentile(all_latencies, 50),
            "p95": _percentile(all_latencies, 95),
            "mean": sum(all_latencies) / len(all_latencies) if all_latencies else None,
        },
        "total_illegal_actions": total_illegal,
        "total_crashed": total_crashed,
        "fixed_fallbacks": fallback_totals,
        "fixed_fallbacks_total": sum(fallback_totals.values()),
        "wins_by_seat": wins_by_seat,
        "games_by_seat": games_by_seat,
        "shadow_decisions": all_shadow_decisions,
        "per_game": per_game,
    }


def main(argv: list[str] | None = None) -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)
    checkpoint_sha256 = verify_checkpoint()

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

        model = MonopolyZeroNet.load_inference(CHECKPOINT_PATH)
        search_config = SearchConfig(simulations=SIMULATIONS, max_depth=MAX_DEPTH)
        fixed_pool = FP_AGENT_CLASSES[:3]

        policy_only_result = _evaluate_policy(
            lambda: common.build_local_policy_only(model), fixed_pool, "policy_only",
        )
        puct_4_result = _evaluate_policy(
            lambda: common.build_local_search_policy(model, search_config, self_play=False),
            fixed_pool, "puct_4",
            shadow_builder=lambda: common.build_local_policy_only(model),
        )

        asu_modules_loaded = common.loaded_asu_modules()

    elapsed_s = time.perf_counter() - started

    action_disagreement_rate = disagreement_rate(puct_4_result["shadow_decisions"])
    action_disagreement_states = len(puct_4_result["shadow_decisions"])
    puct_4_improvement_over_policy_only = non_overlapping_improvement(policy_only_result, puct_4_result)
    policy_only_improvement_over_puct_4 = non_overlapping_improvement(puct_4_result, policy_only_result)

    latency_p50 = policy_only_result["decision_latency_s"]["p50"]
    puct_latency_p50 = puct_4_result["decision_latency_s"]["p50"]
    latency_p95 = policy_only_result["decision_latency_s"]["p95"]
    puct_latency_p95 = puct_4_result["decision_latency_s"]["p95"]

    # The raw per-decision shadow-query list (one record per non-forced
    # decision across all 40 PUCT_4 games, tens of thousands of records) is
    # only needed to compute the aggregate rate above - drop it before
    # printing so raw stdout stays a small, diffable summary, not a
    # multi-megabyte per-decision dump. The aggregate rate/count carries the
    # full statistical content this experiment needs.
    policy_only_result_public = {k: v for k, v in policy_only_result.items() if k != "shadow_decisions"}
    puct_4_result_public = {k: v for k, v in puct_4_result.items() if k != "shadow_decisions"}

    payload = {
        "git_head_sha": git_head_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "config": {
            "held_out_seeds": list(HELD_OUT_SEEDS),
            "simulations": SIMULATIONS,
            "max_depth": MAX_DEPTH,
            "max_rounds": MAX_ROUNDS,
            "self_play": False,
            "opponents": ["fixed-a", "fixed-b", "fixed-c"],
            "seat_rotation": "4 seats x 10 seeds = 40 games per policy",
        },
        "results": {
            "policy_only": policy_only_result_public,
            "puct_4": puct_4_result_public,
        },
        "action_disagreement_rate": action_disagreement_rate,
        "action_disagreement_states": action_disagreement_states,
        "latency_p50_diff_s": (
            None if latency_p50 is None or puct_latency_p50 is None else puct_latency_p50 - latency_p50
        ),
        "latency_p95_diff_s": (
            None if latency_p95 is None or puct_latency_p95 is None else puct_latency_p95 - latency_p95
        ),
        "puct_4_improvement_over_policy_only_statistically_supported": puct_4_improvement_over_policy_only,
        "policy_only_improvement_over_puct_4_statistically_supported": policy_only_improvement_over_puct_4,
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
