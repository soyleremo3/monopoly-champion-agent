"""ASU-import-free, training-free horizon diagnostic.

Two independent, purely descriptive measurements (no gradient update ever
happens; both checkpoints are loaded read-only):

1. Round-50-leader vs. round-200(-or-terminal) winner agreement. Plays 32
   fresh games (16 self-play + 16 vs-fixed, seat-balanced, fresh seeds never
   used by any prior DEV/eval run in this project - see SELF_PLAY_SEEDS/
   VS_FIXED_SEEDS below) at max_rounds=200 with the exact 4-simulation,
   self_play=True search recipe that generated 013's training data (which
   itself only ran to max_rounds=50). Snapshots every player's net worth the
   instant the round counter first reaches 50, then lets the SAME game
   continue uninterrupted to round 200 or elimination, and reports whether
   the round-50 net-worth leader matches the eventual winner.

2. State-encoding ablation. From the same 32 games, collects up to 200
   non-forced decision states seen in rounds 1-50 (deterministic order: game
   plan order, then turn order - first 200 encountered). For each state,
   clones the game and flips ONLY env.max_rounds (200 -> 50), leaving every
   other field untouched, then verifies programmatically - not assumed -
   that at most one state-vector index changes (monopoly_game_engine/
   state.py's build_state_vector encodes round/max_rounds as a single
   scalar, min(round/max_rounds, 1.0), traced by hand to index 278 of the
   300-dim state vector; the script fails loudly if any other index moves).
   Measures, for baseline_pretraining.pt and the 500-update checkpoint
   separately: POLICY_ONLY chosen-action disagreement, policy-distribution
   total-variation divergence, and value-head mean absolute delta between
   the two encodings of the exact same underlying game state.

No arbitrary GO/KILL threshold: this script only measures and reports. The
call on what to do next is made separately, after the results are read.

Built on scripts/monopolyzero_common.py - no monopoly_bench.adapters/.arena
/.training import, no ASU. Refuses to run unless PYTHONHASHSEED=0 is set,
the git tree is clean, and both checkpoints' SHA-256 match the recorded
values.
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

BASELINE_CHECKPOINT_SHA256 = "22ae2abea60478355f61ce0404a31de3b52fb97769d4f99f7e04f4c673629370"
UPDATE500_CHECKPOINT_SHA256 = "152c0a0f6136d1fc91e74973ac245b2f72774694c424d2a48854514ed2848383"

SIMULATIONS = 4
MAX_DEPTH = 16
MAX_ROUNDS = 200
ROUND_SNAPSHOT = 50
GAMES_PER_CATEGORY = 16

ABLATION_TARGET_STATES = 200
ABLATION_MIN_ROUND = 1
ABLATION_MAX_ROUND = 50
# Verified by hand-tracing monopoly_game_engine/state.py::build_state_vector's
# idx arithmetic: this is the single scalar min(round/max_rounds, 1.0).
# Asserted at runtime below, never just assumed.
ROUND_MAX_ROUNDS_STATE_INDEX = 278

# Fresh seeds: no overlap with any prior DEV/eval seed pool in this project
# (013 training: self-play 10000-10015, vs-fixed 20000-20015; DDQN paired
# eval: 10000-10009; 017 search-budget diagnostic: 31000-31004; 018
# policy-only-vs-PUCT eval: 32000-32009).
SELF_PLAY_SEEDS = tuple(range(40000, 40000 + GAMES_PER_CATEGORY))
VS_FIXED_SEEDS = tuple(range(41000, 41000 + GAMES_PER_CATEGORY))

REPO_ROOT = common.REPO_ROOT
PILOT_DIR = REPO_ROOT / "artifacts" / "monopolyzero_strength_pilot"
BASELINE_CHECKPOINT = PILOT_DIR / "baseline_pretraining.pt"
UPDATE500_CHECKPOINT = PILOT_DIR / "trained_updates_500.pt"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path: Path, expected_sha256: str) -> str:
    if not path.is_file():
        raise SystemExit(
            f"monopolyzero_horizon_diagnostic.py refuses to run: missing checkpoint {path}"
        )
    actual = _sha256(path)
    if actual != expected_sha256:
        raise SystemExit(
            "monopolyzero_horizon_diagnostic.py refuses to run: checkpoint SHA-256 mismatch "
            f"for {path}. Got {actual}, expected {expected_sha256}."
        )
    return actual


def _mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _final_rank(final_net_worth: tuple, seat: int) -> int:
    """1 = highest final net worth, len(final_net_worth) = lowest."""
    ordered = sorted(range(len(final_net_worth)), key=lambda s: final_net_worth[s], reverse=True)
    return ordered.index(seat) + 1


def _net_worth_snapshot(game) -> dict:
    net_worth = tuple(float(p.net_worth()) for p in game.env.players)
    ordered = sorted(range(len(net_worth)), key=lambda s: net_worth[s], reverse=True)
    leader_seat = ordered[0]
    margin = net_worth[leader_seat] - (net_worth[ordered[1]] if len(ordered) > 1 else 0.0)
    return {
        "round": game.env.round,
        "net_worth": list(net_worth),
        "leader_seat": leader_seat,
        "margin": margin,
    }


def _build_game_plan() -> list[dict]:
    """32 fresh games: 16 self-play (model in every seat, recording_seat=0
    is an arbitrary-but-deterministic pick for ablation-state collection
    purposes only) + 16 vs-fixed (model rotates through every seat evenly,
    fixed-a/b/c fill the rest, recording_seat=focus_seat)."""
    plan = []
    for seed in SELF_PLAY_SEEDS:
        plan.append({"category": "self_play", "seed": seed, "focus_seat": None, "recording_seat": 0})
    for index, seed in enumerate(VS_FIXED_SEEDS):
        focus_seat = index % 4
        plan.append({"category": "vs_fixed", "seed": seed, "focus_seat": focus_seat, "recording_seat": focus_seat})
    return plan


def _play_one_game(model, fixed_pool, search_config, job: dict, ablation_snapshots: list, ablation_target: int) -> dict:
    """Plays one game turn-by-turn, recording a round-50 net-worth snapshot
    and (while under budget) cloned pre-decision states for the
    state-encoding ablation. Structurally its own loop (not a reuse of
    common.play_local_game, which has no round-snapshot/ablation-collection
    hooks) built from the same ASU-clean primitives."""
    from monopoly_bench.engine import MAX_DECISIONS_PER_TURN, NUM_PLAYERS, SharedGame, clone_env

    seed = job["seed"]
    category = job["category"]
    focus_seat = job["focus_seat"]
    recording_seat = job["recording_seat"]

    if category == "self_play":
        policies = {
            seat: common.build_local_search_policy(model, search_config, self_play=True)
            for seat in range(NUM_PLAYERS)
        }
    else:
        non_focus_seats = [seat for seat in range(NUM_PLAYERS) if seat != focus_seat]
        policies = {focus_seat: common.build_local_search_policy(model, search_config, self_play=True)}
        for seat, agent_class in zip(non_focus_seats, fixed_pool):
            policies[seat] = common.LocalFixedPolicy(agent_class)

    game = SharedGame.new(seed, MAX_ROUNDS)
    decision_budget = MAX_ROUNDS * NUM_PLAYERS * MAX_DECISIONS_PER_TURN

    round50_snapshot = None
    turn_index = 0
    illegal_actions = 0
    crashed = False
    error = None

    try:
        while turn_index < decision_budget and not game.env.done:
            actor = game.env.whose_turn()
            legal = tuple(game.env.get_allowed_actions(actor))
            if len(legal) == 1:
                action = legal[0]
            else:
                decision_seed = common._mix_decision_seed(seed, turn_index, actor)
                current_round = game.env.round
                if (
                    actor == recording_seat
                    and ABLATION_MIN_ROUND <= current_round <= ABLATION_MAX_ROUND
                    and len(ablation_snapshots) < ablation_target
                ):
                    ablation_snapshots.append(
                        {
                            "seed": seed,
                            "category": category,
                            "recording_seat": recording_seat,
                            "turn_index": turn_index,
                            "round": current_round,
                            "env_clone": clone_env(game),
                        }
                    )
                action, _, _, _ = common._invoke_policy(policies[actor], game, actor, decision_seed)
                if action not in legal:
                    raise RuntimeError(f"seat {actor} attempted illegal action {action}")
            game.step(action)
            turn_index += 1

            if round50_snapshot is None and game.env.round >= ROUND_SNAPSHOT:
                round50_snapshot = _net_worth_snapshot(game)
    except RuntimeError as exc:
        illegal_actions = 1
        error = str(exc)
    except Exception as exc:  # noqa: BLE001 - intentional fail-closed boundary
        crashed = True
        error = f"{type(exc).__name__}: {exc}"

    finished = game.env.done
    winner = game.env.winner() if finished else None
    final_net_worth = tuple(float(p.net_worth()) for p in game.env.players)

    fixed_fallbacks = {
        policy.name: policy.fallback_count
        for policy in policies.values()
        if getattr(policy, "kind", None) == "fixed"
    }

    return {
        "seed": seed,
        "category": category,
        "focus_seat": focus_seat,
        "recording_seat": recording_seat,
        "decisions": turn_index,
        "completed": finished,
        "winner": winner,
        "final_round": game.env.round,
        "final_net_worth": final_net_worth,
        "round50_snapshot": round50_snapshot,
        "finished_before_round_50": round50_snapshot is None,
        "illegal_actions": illegal_actions,
        "crashed": crashed,
        "error": error,
        "fixed_fallbacks": fixed_fallbacks,
    }


def _agreement_stats(games: list[dict]) -> dict:
    if not games:
        return {"games": 0, "agreements": 0, "agreement_rate": None}
    agreements = sum(1 for g in games if g["round50_snapshot"]["leader_seat"] == g["winner"])
    return {"games": len(games), "agreements": agreements, "agreement_rate": agreements / len(games)}


def _summarize_horizon(game_reports: list[dict]) -> dict:
    eligible = [g for g in game_reports if g["round50_snapshot"] is not None and g["completed"]]
    finished_before_50 = [g for g in game_reports if g["round50_snapshot"] is None]

    overall = _agreement_stats(eligible)
    self_play = _agreement_stats([g for g in eligible if g["category"] == "self_play"])
    vs_fixed = _agreement_stats([g for g in eligible if g["category"] == "vs_fixed"])

    rank_distribution: dict[str, int] = {}
    for g in eligible:
        rank = _final_rank(g["final_net_worth"], g["round50_snapshot"]["leader_seat"])
        key = str(rank)
        rank_distribution[key] = rank_distribution.get(key, 0) + 1

    agree_margins = [g["round50_snapshot"]["margin"] for g in eligible if g["round50_snapshot"]["leader_seat"] == g["winner"]]
    disagree_margins = [g["round50_snapshot"]["margin"] for g in eligible if g["round50_snapshot"]["leader_seat"] != g["winner"]]

    margin_buckets = ((0.0, 500.0), (500.0, 2000.0), (2000.0, float("inf")))
    bucketed = []
    for lo, hi in margin_buckets:
        bucket_games = [g for g in eligible if lo <= g["round50_snapshot"]["margin"] < hi]
        bucketed.append(
            {
                "margin_range": f"[{lo},{'inf' if hi == float('inf') else hi})",
                **_agreement_stats(bucket_games),
            }
        )

    return {
        "overall": overall,
        "self_play": self_play,
        "vs_fixed": vs_fixed,
        "round50_leader_final_rank_distribution": rank_distribution,
        "margin_mean_when_agree": _mean(agree_margins),
        "margin_mean_when_disagree": _mean(disagree_margins),
        "margin_median_when_agree": _median(agree_margins),
        "margin_median_when_disagree": _median(disagree_margins),
        "agreement_rate_by_margin_bucket": bucketed,
        "games_finished_before_round_50": len(finished_before_50),
        "games_finished_before_round_50_detail": [
            {"seed": g["seed"], "category": g["category"], "final_round": g["final_round"], "winner": g["winner"]}
            for g in finished_before_50
        ],
    }


def _summarize_fallbacks(game_reports: list[dict]) -> dict:
    totals = {"self_play": 0, "vs_fixed": 0}
    for g in game_reports:
        totals[g["category"]] += sum(g["fixed_fallbacks"].values())
    totals["total"] = totals["self_play"] + totals["vs_fixed"]
    return totals


def _run_state_encoding_ablation(model, snapshots: list[dict], label: str) -> dict:
    """For each frozen pre-decision state, flips ONLY env.max_rounds
    (200 -> 50) on a fresh clone and compares model output between the two
    encodings of the identical underlying game state. Fails loudly if any
    state-vector index other than ROUND_MAX_ROUNDS_STATE_INDEX changes -
    that would mean the isolation assumption is wrong, not something to
    silently tolerate."""
    import numpy as np

    from monopoly_bench.engine import clone_env

    per_state = []
    diff_index_union: set[int] = set()

    for snap in snapshots:
        env_200 = snap["env_clone"]
        actor = snap["recording_seat"]
        legal = tuple(env_200.get_allowed_actions(actor))

        state_200 = env_200._get_state(actor)

        env_50 = clone_env(env_200)
        env_50.max_rounds = 50
        state_50 = env_50._get_state(actor)

        diff_indices = np.flatnonzero(state_200 != state_50).tolist()
        unexpected = set(diff_indices) - {ROUND_MAX_ROUNDS_STATE_INDEX}
        if unexpected:
            raise RuntimeError(
                "State-encoding ablation isolation broken: unexpected state-vector "
                f"indices changed {sorted(unexpected)} when only env.max_rounds was "
                f"flipped (seed={snap['seed']}, round={snap['round']}, label={label})"
            )
        diff_index_union.update(diff_indices)

        priors_200, value_200 = model.predict(state_200, legal, actor)
        priors_50, value_50 = model.predict(state_50, legal, actor)

        action_200 = max(priors_200, key=priors_200.get)
        action_50 = max(priors_50, key=priors_50.get)

        tv_distance = 0.5 * sum(abs(priors_200[a] - priors_50.get(a, 0.0)) for a in legal)
        value_abs_delta = float(np.mean(np.abs(np.asarray(value_200) - np.asarray(value_50))))

        per_state.append(
            {
                "action_disagree": action_200 != action_50,
                "policy_tv_distance": tv_distance,
                "value_mean_abs_delta": value_abs_delta,
                "diff_index_count": len(diff_indices),
            }
        )

    action_disagreements = sum(1 for s in per_state if s["action_disagree"])
    return {
        "label": label,
        "states_used": len(per_state),
        "action_disagreement_rate": (action_disagreements / len(per_state)) if per_state else None,
        "policy_tv_distance_mean": _mean([s["policy_tv_distance"] for s in per_state]),
        "value_mean_abs_delta_mean": _mean([s["value_mean_abs_delta"] for s in per_state]),
        "state_vector_diff_indices_union": sorted(diff_index_union),
        "states_with_no_index_diff": sum(1 for s in per_state if s["diff_index_count"] == 0),
    }


def main(argv: list[str] | None = None) -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)
    baseline_sha256 = verify_checkpoint(BASELINE_CHECKPOINT, BASELINE_CHECKPOINT_SHA256)
    update500_sha256 = verify_checkpoint(UPDATE500_CHECKPOINT, UPDATE500_CHECKPOINT_SHA256)

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
        fixed_pool = FP_AGENT_CLASSES[:3]

        baseline_model = MonopolyZeroNet.load_inference(BASELINE_CHECKPOINT)

        game_plan = _build_game_plan()
        ablation_snapshots: list[dict] = []
        game_reports = []
        for job in game_plan:
            report = _play_one_game(
                baseline_model, fixed_pool, search_config, job, ablation_snapshots, ABLATION_TARGET_STATES
            )
            game_reports.append(report)

        total_illegal = sum(g["illegal_actions"] for g in game_reports)
        total_crashed = sum(int(g["crashed"]) for g in game_reports)

        if total_illegal or total_crashed:
            payload = {
                "status": "FAILED_DURING_GAME_GENERATION",
                "git_head_sha": git_head_sha,
                "games": game_reports,
                "total_illegal_actions": total_illegal,
                "total_crashed": total_crashed,
            }
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
            raise RuntimeError(f"Stopping before ablation: crashed={total_crashed} illegal={total_illegal}")

        horizon_summary = _summarize_horizon(game_reports)
        fallback_totals = _summarize_fallbacks(game_reports)

        update500_model = MonopolyZeroNet.load_inference(UPDATE500_CHECKPOINT)
        ablation_baseline = _run_state_encoding_ablation(baseline_model, ablation_snapshots, "baseline_pretraining")
        ablation_update500 = _run_state_encoding_ablation(update500_model, ablation_snapshots, "trained_updates_500")

        asu_modules_loaded = common.loaded_asu_modules()

    elapsed_s = time.perf_counter() - started

    games_public = [
        {key: value for key, value in g.items()} for g in game_reports
    ]

    payload = {
        "git_head_sha": git_head_sha,
        "config": {
            "baseline_checkpoint_path": str(BASELINE_CHECKPOINT),
            "baseline_checkpoint_sha256": baseline_sha256,
            "update500_checkpoint_path": str(UPDATE500_CHECKPOINT),
            "update500_checkpoint_sha256": update500_sha256,
            "games_total": len(game_plan),
            "self_play_games": GAMES_PER_CATEGORY,
            "vs_fixed_games": GAMES_PER_CATEGORY,
            "self_play_seeds": list(SELF_PLAY_SEEDS),
            "vs_fixed_seeds": list(VS_FIXED_SEEDS),
            "seed_freshness_note": (
                "no overlap with any prior DEV/eval seed pool in this project "
                "(013: self-play 10000-10015, vs-fixed 20000-20015; DDQN paired "
                "eval: 10000-10009; 017: 31000-31004; 018: 32000-32009)"
            ),
            "simulations": SIMULATIONS,
            "max_depth": MAX_DEPTH,
            "self_play_for_search_seats": True,
            "max_rounds": MAX_ROUNDS,
            "round_snapshot": ROUND_SNAPSHOT,
            "opponents": ["fixed-a", "fixed-b", "fixed-c"],
            "ablation_target_states": ABLATION_TARGET_STATES,
            "ablation_round_range": [ABLATION_MIN_ROUND, ABLATION_MAX_ROUND],
            "ablation_state_vector_index": ROUND_MAX_ROUNDS_STATE_INDEX,
            "no_training": True,
            "no_arbitrary_go_kill_threshold": True,
            "asu_involved": False,
        },
        "games": games_public,
        "horizon": horizon_summary,
        "fixed_fallbacks": fallback_totals,
        "ablation_states_collected": len(ablation_snapshots),
        "ablation": {
            "baseline_pretraining": ablation_baseline,
            "trained_updates_500": ablation_update500,
        },
        "total_illegal_actions": total_illegal,
        "total_crashed": total_crashed,
        "asu_modules_loaded": asu_modules_loaded,
        "asu_modules_loaded_count": len(asu_modules_loaded),
        "elapsed_s": elapsed_s,
        "peak_rss_gib": rss.peak_gib,
    }

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded during diagnostic: {asu_modules_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
