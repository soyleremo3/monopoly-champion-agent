"""ASU-import-free, training-free offline PUCT search-budget diagnostic.

Generates ZERO new training data and trains nothing. Collects 200
deterministic, non-forced decision states from the 500-update checkpoint
(013/016's update-budget sweep) playing against fixed-a/b/c on held-out
seeds, then re-runs PUCT search on each EXACT frozen state at 4, 16, and 32
simulations (self_play=False, depth=16) to measure whether more simulation
budget actually changes the chosen action / value estimate, before ever
spending a full game-evaluation budget on it.

Refuses to run unless the 500-update checkpoint's SHA-256 matches the value
recorded in 016's experiment log — if it doesn't (or the file is missing),
stops before doing anything else.

Built on scripts/monopolyzero_common.py — no monopoly_bench.adapters/.arena
/.training import, no ASU. Refuses to run unless PYTHONHASHSEED=0 is set
and the git tree is clean.

Does not save per-state raw arrays (game states) to disk — only the small
derived comparison metrics (action ids, floats) are recorded, which stay
small even for 200 states x 3 simulation counts.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import monopolyzero_common as common  # noqa: E402

EXPECTED_CHECKPOINT_SHA256 = "152c0a0f6136d1fc91e74973ac245b2f72774694c424d2a48854514ed2848383"

HELD_OUT_SEEDS = (31000, 31001, 31002, 31003, 31004)
TARGET_STATES = 200
COLLECTION_SIMULATIONS = 4  # the search budget already in use, being tested for scaling
COLLECTION_MAX_ROUNDS = 50
SIMULATION_TIERS = (4, 16, 32)
MAX_DEPTH = 16

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
            f"monopolyzero_search_budget_diagnostic.py refuses to run: "
            f"missing checkpoint {CHECKPOINT_PATH}"
        )
    actual = _sha256(CHECKPOINT_PATH)
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit(
            "monopolyzero_search_budget_diagnostic.py refuses to run: "
            f"checkpoint SHA-256 mismatch. Got {actual}, expected "
            f"{EXPECTED_CHECKPOINT_SHA256}."
        )
    return actual


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def disagreement_rate(per_state: list[dict], tier_a: int, tier_b: int) -> float:
    if not per_state:
        return 0.0
    mismatches = sum(
        1 for state in per_state
        if state["tiers"][tier_a]["chosen_action"] != state["tiers"][tier_b]["chosen_action"]
    )
    return mismatches / len(per_state)


def mean_abs_value_delta(per_state: list[dict], tier_a: int, tier_b: int) -> float | None:
    if not per_state:
        return None
    deltas = [
        abs(state["tiers"][tier_a]["root_value_self"] - state["tiers"][tier_b]["root_value_self"])
        for state in per_state
    ]
    return sum(deltas) / len(deltas)


def mean_chosen_visit_share(per_state: list[dict], tier: int) -> float | None:
    shares = [
        state["tiers"][tier]["chosen_visit_share"]
        for state in per_state
        if state["tiers"][tier]["chosen_visit_share"] is not None
    ]
    return sum(shares) / len(shares) if shares else None


def _collect_states(model, fixed_pool):
    """Plays sims=4 self_play=False games (model vs fixed-a/b/c, model's
    seat rotating), snapshotting a clone of the game BEFORE each non-forced
    decision the model faces, until TARGET_STATES snapshots are collected
    or the seed/seat grid is exhausted."""
    from monopoly_bench.config import SearchConfig
    from monopoly_bench.engine import NUM_PLAYERS, clone_env

    collection_config = SearchConfig(simulations=COLLECTION_SIMULATIONS, max_depth=MAX_DEPTH)
    snapshots = []

    for seed in HELD_OUT_SEEDS:
        for focus_seat in range(NUM_PLAYERS):
            if len(snapshots) >= TARGET_STATES:
                return snapshots
            snapshots.extend(
                _play_and_snapshot(model, fixed_pool, collection_config, seed, focus_seat)
            )
    return snapshots


def _play_and_snapshot(model, fixed_pool, collection_config, seed: int, focus_seat: int):
    from monopoly_bench.engine import MAX_DECISIONS_PER_TURN, NUM_PLAYERS, SharedGame, clone_env

    game = SharedGame.new(seed, COLLECTION_MAX_ROUNDS)
    non_focus_seats = [seat for seat in range(NUM_PLAYERS) if seat != focus_seat]
    fixed_policies = {
        seat: common.LocalFixedPolicy(agent_class)
        for seat, agent_class in zip(non_focus_seats, fixed_pool)
    }
    search_policy = common.build_local_search_policy(model, collection_config, self_play=False)

    local_snapshots = []
    decision_budget = COLLECTION_MAX_ROUNDS * NUM_PLAYERS * MAX_DECISIONS_PER_TURN
    turn_index = 0
    while turn_index < decision_budget and not game.env.done and len(local_snapshots) < TARGET_STATES:
        actor = game.env.whose_turn()
        legal = tuple(game.env.get_allowed_actions(actor))
        if len(legal) == 1:
            action = legal[0]
        else:
            decision_seed = common._mix_decision_seed(seed, turn_index, actor)
            if actor == focus_seat:
                local_snapshots.append(
                    {
                        "seed": seed,
                        "focus_seat": focus_seat,
                        "turn_index": turn_index,
                        "decision_seed": decision_seed,
                        "env_clone": clone_env(game),
                    }
                )
                result = search_policy.choose(game, actor, decision_seed)
                action = result.chosen_action
            else:
                action = fixed_policies[actor].choose(game, actor, decision_seed)
            if action not in legal:
                raise RuntimeError(f"seat {actor} attempted illegal action {action} during collection")
        game.step(action)
        turn_index += 1
    return local_snapshots


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
    from monopoly_bench.search import MaxNPUCT
    from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        model = MonopolyZeroNet.load_inference(CHECKPOINT_PATH)
        fixed_pool = FP_AGENT_CLASSES[:3]

        snapshots = _collect_states(model, fixed_pool)
        states_collected = len(snapshots)

        per_state = []
        illegal_total = 0
        latencies_by_tier = {tier: [] for tier in SIMULATION_TIERS}

        for snapshot in snapshots:
            env_clone = snapshot["env_clone"]
            actor = snapshot["focus_seat"]
            decision_seed = snapshot["decision_seed"]
            legal = tuple(env_clone.get_allowed_actions(actor))

            per_tier = {}
            for simulations in SIMULATION_TIERS:
                search = MaxNPUCT(model, SearchConfig(simulations=simulations, max_depth=MAX_DEPTH), self_play=False)
                result = search.choose_action(env_clone, actor, decision_seed)
                if result.chosen_action not in legal:
                    illegal_total += 1
                total_visits = sum(result.visits.values())
                visit_share = (
                    result.visits.get(result.chosen_action, 0) / total_visits if total_visits else None
                )
                latencies_by_tier[simulations].append(result.latency_s)
                per_tier[simulations] = {
                    "chosen_action": result.chosen_action,
                    "root_value_self": result.root_value[0],
                    "root_value_full": list(result.root_value),
                    "chosen_visit_share": visit_share,
                    "latency_s": result.latency_s,
                }

            per_state.append(
                {
                    "seed": snapshot["seed"],
                    "focus_seat": snapshot["focus_seat"],
                    "turn_index": snapshot["turn_index"],
                    "tiers": per_tier,
                }
            )

        asu_modules_loaded = common.loaded_asu_modules()

    elapsed_s = time.perf_counter() - started

    disagreement = {
        "4_vs_16": disagreement_rate(per_state, 4, 16),
        "16_vs_32": disagreement_rate(per_state, 16, 32),
        "4_vs_32": disagreement_rate(per_state, 4, 32),
    }
    value_delta = {
        "4_vs_16_mean_abs": mean_abs_value_delta(per_state, 4, 16),
        "16_vs_32_mean_abs": mean_abs_value_delta(per_state, 16, 32),
        "4_vs_32_mean_abs": mean_abs_value_delta(per_state, 4, 32),
    }
    mean_visit_share = {tier: mean_chosen_visit_share(per_state, tier) for tier in SIMULATION_TIERS}
    latency_stats = {
        tier: {
            "p50": _percentile(latencies_by_tier[tier], 50),
            "p95": _percentile(latencies_by_tier[tier], 95),
            "mean": sum(latencies_by_tier[tier]) / len(latencies_by_tier[tier]) if latencies_by_tier[tier] else None,
        }
        for tier in SIMULATION_TIERS
    }

    kill_search_scaling = disagreement["4_vs_16"] < 0.05 and (
        value_delta["4_vs_16_mean_abs"] is not None and value_delta["4_vs_16_mean_abs"] < 0.05
    )

    payload = {
        "git_head_sha": git_head_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "config": {
            "held_out_seeds": list(HELD_OUT_SEEDS),
            "target_states": TARGET_STATES,
            "states_collected": states_collected,
            "collection_simulations": COLLECTION_SIMULATIONS,
            "collection_max_rounds": COLLECTION_MAX_ROUNDS,
            "simulation_tiers": list(SIMULATION_TIERS),
            "max_depth": MAX_DEPTH,
            "self_play": False,
            "opponents": ["fixed-a", "fixed-b", "fixed-c"],
        },
        "disagreement": disagreement,
        "root_value_self_mean_abs_delta": value_delta,
        "mean_chosen_action_visit_share": mean_visit_share,
        "search_latency_s": latency_stats,
        "illegal_actions_during_search": illegal_total,
        "asu_modules_loaded": asu_modules_loaded,
        "asu_modules_loaded_count": len(asu_modules_loaded),
        "kill_search_scaling_recommended": kill_search_scaling,
        "elapsed_s": elapsed_s,
        "peak_rss_gib": rss.peak_gib,
        "per_state": per_state,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded during diagnostic: {asu_modules_loaded}")
    if illegal_total:
        raise RuntimeError(f"Illegal action(s) detected during search diagnostic: {illegal_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
