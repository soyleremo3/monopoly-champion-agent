"""ASU-import-free value-GENERALIZATION probe: fixes two methodological
gaps in `020-monopolyzero-value-learnability-probe`.

020's "validation" split doubled as its own early-stopping monitor, so its
reported accuracy/CE was a model-selection number, not an unbiased estimate
of generalization to unseen games (see docs/DECISIONS.md's GO entry for
020). 020 also sampled "first N" non-forced decisions per (game, seat,
round-bucket) cell, which can cluster all samples near the start of a
bucket instead of spreading across it.

This script fixes both: a proper three-way TRAIN(64 games) / SELECTION(16
games) / TEST(16 games) split where TEST is touched exactly once (never for
early stopping, hyperparameter/temperature selection, or model choice), and
deterministic quantile-spread sampling (min/median/max of each cell's
available occurrences) with full provenance instead of "first N". Uses a
FRESH 96-game dataset (seeds 42100-42195) - does not reuse 020's 64 games.

Reuses monopolyzero_value_learnability_probe.py's ValueProbe architecture,
training loop, and pure metric helpers directly (same architecture, as
instructed) rather than duplicating them - see that module for their
definitions. Adds: quantile-spread sampling, a probabilistic net-worth-leader
baseline (temperature-scaled softmax, fit on TRAIN+SELECTION only, never
TEST), and a game-block (not state-level) bootstrap for paired CE/Brier/
accuracy differences, since states from the same game are correlated and
must not be treated as independent samples.

Does NOT touch MonopolyZeroNet's weights, does NOT run PUCT/search, does
NOT evaluate win rate, and makes NO strength/policy claim. Computes no
promotion/GO-KILL verdict of its own.

Refuses to run unless PYTHONHASHSEED=0 is set, the git tree is clean, the
checkpoint SHA-256 matches, and every self-play seed is DEV-scoped.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluation_protocol as ep  # noqa: E402
import monopolyzero_common as common  # noqa: E402
import monopolyzero_value_learnability_probe as probe_v1  # noqa: E402

MAX_ROUNDS = probe_v1.MAX_ROUNDS
ROUND_BUCKETS = probe_v1.ROUND_BUCKETS
CHECKPOINT_PATH = probe_v1.CHECKPOINT_PATH
BASELINE_CHECKPOINT_SHA256 = probe_v1.BASELINE_CHECKPOINT_SHA256

GAMES_TOTAL = 96
SELF_PLAY_SEEDS = tuple(range(42100, 42100 + GAMES_TOTAL))
TRAIN_SEEDS = SELF_PLAY_SEEDS[:64]
SELECTION_SEEDS = SELF_PLAY_SEEDS[64:80]
TEST_SEEDS = SELF_PLAY_SEEDS[80:96]

SAMPLES_PER_CELL = 3  # quantile-spread: min/median/max occurrence per (game, seat, bucket)

LEARNING_CURVE_TRAIN_GAME_COUNTS = (16, 32, 64)

TEMPERATURE_GRID = (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0)

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0


def verify_checkpoint(path: Path = CHECKPOINT_PATH, expected_sha256: str = BASELINE_CHECKPOINT_SHA256) -> str:
    return probe_v1.verify_checkpoint(path, expected_sha256)


def _median(values: list) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def quantile_indices(count: int, n: int) -> list[int]:
    """Deterministic spread selection: for count <= n, take everything.
    Otherwise pick n indices evenly spaced from 0 to count-1 inclusive
    (for n=3: min, ~median, max) - avoids clustering all samples near the
    start of a bucket's occurrences, unlike 020's "first N"."""
    if count == 0:
        return []
    if count <= n:
        return list(range(count))
    if n == 1:
        return [count // 2]
    return sorted({round(i * (count - 1) / (n - 1)) for i in range(n)})


# ── self-play game generation with quantile-spread sampling ────────────


def _generate_games(model):
    from monopoly_bench.engine import MAX_DECISIONS_PER_TURN, NUM_PLAYERS, SharedGame, clone_env

    policy = common.build_local_policy_only(model)  # stateless, safe to share across all 4 seats

    games = []
    for seed in SELF_PLAY_SEEDS:
        game = SharedGame.new(seed, MAX_ROUNDS)
        decision_budget = MAX_ROUNDS * NUM_PLAYERS * MAX_DECISIONS_PER_TURN
        candidates = {seat: {label: [] for label, _, _ in ROUND_BUCKETS} for seat in range(NUM_PLAYERS)}

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
                    current_round = game.env.round
                    bucket = probe_v1.bucket_for_round(current_round)
                    if bucket is not None:
                        candidates[actor][bucket].append(
                            {"turn_index": turn_index, "round": current_round, "env_clone": clone_env(game)}
                        )
                    decision_seed = common._mix_decision_seed(seed, turn_index, actor)
                    action, _, _, _ = common._invoke_policy(policy, game, actor, decision_seed)
                    if action not in legal:
                        raise RuntimeError(f"seat {actor} attempted illegal action {action}")
                game.step(action)
                turn_index += 1
        except RuntimeError as exc:
            illegal_actions = 1
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - intentional fail-closed boundary
            crashed = True
            error = f"{type(exc).__name__}: {exc}"

        finished = game.env.done
        winner = game.env.winner() if finished else None

        sampled = {seat: {} for seat in range(NUM_PLAYERS)}
        for seat in range(NUM_PLAYERS):
            for label, _, _ in ROUND_BUCKETS:
                pool = candidates[seat][label]
                picked_idx = quantile_indices(len(pool), SAMPLES_PER_CELL)
                sampled[seat][label] = [pool[i] for i in picked_idx]

        games.append(
            {
                "seed": seed, "sampled": sampled, "completed": finished, "winner": winner,
                "decisions": turn_index, "final_round": game.env.round,
                "illegal_actions": illegal_actions, "crashed": crashed, "error": error,
            }
        )
    return games


def _build_records(games):
    """Actor-relative TRUE full-horizon winner label (same convention as
    020), plus actor-relative net worth (needed by the probabilistic
    leader baseline) and full sampling provenance (seed/seat/round/
    turn_index/bucket)."""
    from monopoly_bench.engine import NUM_PLAYERS, actor_order

    records = []
    for g in games:
        winner = g["winner"]
        for seat in range(NUM_PLAYERS):
            order = actor_order(seat)
            for bucket_label, snapshots in g["sampled"][seat].items():
                for snapshot in snapshots:
                    env_clone = snapshot["env_clone"]
                    state_vector = env_clone._get_state(seat)
                    net_worth_absolute = tuple(float(p.net_worth()) for p in env_clone.players)
                    net_worth_relative = tuple(net_worth_absolute[pid] for pid in order)
                    leader_absolute = max(range(NUM_PLAYERS), key=lambda s: net_worth_absolute[s])
                    records.append(
                        {
                            "seed": g["seed"], "seat": seat, "round": snapshot["round"],
                            "turn_index": snapshot["turn_index"], "bucket": bucket_label,
                            "state": state_vector, "net_worth_relative": net_worth_relative,
                            "true_class_relative": order.index(winner),
                            "leader_class_relative": order.index(leader_absolute),
                        }
                    )
    return records


def _bucket_provenance(records) -> dict:
    result = {}
    for label, _, _ in ROUND_BUCKETS:
        bucket_records = [r for r in records if r["bucket"] == label]
        rounds = [r["round"] for r in bucket_records]
        result[label] = {
            "count": len(bucket_records),
            "unique_games": len({r["seed"] for r in bucket_records}),
            "median_round": _median(rounds),
            "min_round": min(rounds) if rounds else None,
            "max_round": max(rounds) if rounds else None,
        }
    return result


# ── probabilistic net-worth-leader baseline ─────────────────────────────


def probabilistic_leader_probs(net_worth_relative_batch, temperature: float):
    import numpy as np

    net_worth = np.asarray(net_worth_relative_batch, dtype=np.float64)
    logits = net_worth / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def fit_probabilistic_leader_temperature(net_worth_relative_batch, true_classes, grid=TEMPERATURE_GRID):
    """Fit ONLY on the data passed in (caller must pass TRAIN+SELECTION
    pooled, never TEST) - picks the grid temperature minimizing
    cross-entropy. Deterministic (fixed grid, no RNG)."""
    per_grid = []
    best_temperature = None
    best_ce = float("inf")
    for temperature in grid:
        probs = probabilistic_leader_probs(net_worth_relative_batch, temperature)
        ce = probe_v1.cross_entropy(probs, true_classes)
        per_grid.append({"temperature": temperature, "cross_entropy": ce})
        if ce < best_ce:
            best_ce = ce
            best_temperature = temperature
    return best_temperature, per_grid


# ── game-block (not state-level) paired bootstrap ───────────────────────


def game_block_bootstrap_metric_diff(per_game_records: dict, *, n_resamples: int = BOOTSTRAP_RESAMPLES, bootstrap_seed: int = BOOTSTRAP_SEED) -> dict:
    """Resamples whole TEST games (blocks), not individual states, since
    states from the same game are correlated (same board draw). For each
    resample, pools all states from the resampled games and computes
    (learned - probabilistic_leader) for CE/Brier/accuracy. Deterministic
    given the same bootstrap_seed."""
    import numpy as np

    game_ids = sorted(per_game_records)
    n_games = len(game_ids)
    if n_games == 0:
        empty = {"point": None, "ci_95": None}
        return {
            "cross_entropy_diff": dict(empty), "brier_diff": dict(empty), "accuracy_diff": dict(empty),
            "n_games": 0, "n_resamples": n_resamples, "bootstrap_seed": bootstrap_seed,
        }

    def _pooled_diffs(ids):
        learned_probs, baseline_probs, true_classes = [], [], []
        for gid in ids:
            for rec in per_game_records[gid]:
                learned_probs.append(rec["learned_probs"])
                baseline_probs.append(rec["baseline_probs"])
                true_classes.append(rec["true_class"])
        learned_probs = np.asarray(learned_probs)
        baseline_probs = np.asarray(baseline_probs)
        true_classes = np.asarray(true_classes)
        ce_diff = probe_v1.cross_entropy(learned_probs, true_classes) - probe_v1.cross_entropy(baseline_probs, true_classes)
        brier_diff = probe_v1.brier_score(learned_probs, true_classes) - probe_v1.brier_score(baseline_probs, true_classes)
        acc_diff = probe_v1.top1_accuracy(learned_probs, true_classes) - probe_v1.top1_accuracy(baseline_probs, true_classes)
        return ce_diff, brier_diff, acc_diff

    point_ce, point_brier, point_acc = _pooled_diffs(game_ids)

    rng = np.random.default_rng(bootstrap_seed)
    ce_diffs = np.empty(n_resamples)
    brier_diffs = np.empty(n_resamples)
    acc_diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        sampled_positions = rng.integers(0, n_games, size=n_games)
        sampled_ids = [game_ids[j] for j in sampled_positions]
        ce_diffs[i], brier_diffs[i], acc_diffs[i] = _pooled_diffs(sampled_ids)

    def _ci(point, arr):
        return {"point": float(point), "ci_95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]}

    return {
        "cross_entropy_diff": _ci(point_ce, ce_diffs),
        "brier_diff": _ci(point_brier, brier_diffs),
        "accuracy_diff": _ci(point_acc, acc_diffs),
        "n_games": n_games,
        "n_resamples": n_resamples,
        "bootstrap_seed": bootstrap_seed,
        "note": "diff = learned_value_probe - probabilistic_net_worth_leader; unit of resampling is the GAME, not the state, since states from one game are correlated",
    }


def _learning_curve_subset_seeds(train_seeds: tuple, game_count: int) -> tuple:
    return train_seeds[:game_count]


def main(argv: list[str] | None = None) -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)
    checkpoint_sha256 = verify_checkpoint()
    ep.require_seed_scope(SELF_PLAY_SEEDS, ep.SEED_CLASS_DEV, context="monopolyzero_value_generalization_probe.py")

    common.ensure_reference_on_path()
    import random

    import numpy as np
    import torch

    from monopoly_bench.model import MonopolyZeroNet
    from monopoly_game_engine.state import STATE_DIM

    if STATE_DIM != probe_v1.VALUE_PROBE_INPUT_DIM:
        raise SystemExit(f"State dim mismatch: engine reports {STATE_DIM}, probe expects {probe_v1.VALUE_PROBE_INPUT_DIM}")

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        model = MonopolyZeroNet.load_inference(CHECKPOINT_PATH)
        model.eval()

        games = _generate_games(model)
        total_illegal = sum(g["illegal_actions"] for g in games)
        total_crashed = sum(int(g["crashed"]) for g in games)
        if total_illegal or total_crashed:
            payload = {
                "status": "FAILED_DURING_GAME_GENERATION",
                "git_head_sha": git_head_sha,
                "games": [{k: v for k, v in g.items() if k != "sampled"} for g in games],
                "total_illegal_actions": total_illegal,
                "total_crashed": total_crashed,
            }
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
            raise RuntimeError(f"Stopping: crashed={total_crashed} illegal={total_illegal}")

        records = _build_records(games)

        train_seed_set = set(TRAIN_SEEDS)
        selection_seed_set = set(SELECTION_SEEDS)
        test_seed_set = set(TEST_SEEDS)
        overlaps = {
            "train_selection": sorted(train_seed_set & selection_seed_set),
            "train_test": sorted(train_seed_set & test_seed_set),
            "selection_test": sorted(selection_seed_set & test_seed_set),
        }
        if any(overlaps.values()):
            raise RuntimeError(f"Leakage guard failed: seed overlap between splits {overlaps}")

        train_records = [r for r in records if r["seed"] in train_seed_set]
        selection_records = [r for r in records if r["seed"] in selection_seed_set]
        test_records = [r for r in records if r["seed"] in test_seed_set]

        train_ids = {(r["seed"], r["seat"], r["turn_index"]) for r in train_records}
        selection_ids = {(r["seed"], r["seat"], r["turn_index"]) for r in selection_records}
        test_ids = {(r["seed"], r["seat"], r["turn_index"]) for r in test_records}
        state_overlaps = {
            "train_selection": len(train_ids & selection_ids),
            "train_test": len(train_ids & test_ids),
            "selection_test": len(selection_ids & test_ids),
        }
        if any(state_overlaps.values()):
            raise RuntimeError(f"Leakage guard failed: state overlap between splits {state_overlaps}")

        def _arrays(recs):
            x = np.stack([r["state"] for r in recs]).astype(np.float32) if recs else np.zeros((0, STATE_DIM), dtype=np.float32)
            y = np.array([r["true_class_relative"] for r in recs], dtype=np.int64)
            leader_y = np.array([r["leader_class_relative"] for r in recs], dtype=np.int64)
            nw = np.array([r["net_worth_relative"] for r in recs], dtype=np.float64) if recs else np.zeros((0, 4))
            buckets = [r["bucket"] for r in recs]
            return x, y, leader_y, nw, buckets

        train_x, train_y, train_leader_y, train_nw, train_buckets = _arrays(train_records)
        selection_x, selection_y, selection_leader_y, selection_nw, selection_buckets = _arrays(selection_records)
        test_x, test_y, test_leader_y, test_nw, test_buckets = _arrays(test_records)

        # ── temperature fit: TRAIN + SELECTION only, never TEST ──
        pooled_nw = np.concatenate([train_nw, selection_nw], axis=0)
        pooled_y = np.concatenate([train_y, selection_y], axis=0)
        fitted_temperature, temperature_grid_results = fit_probabilistic_leader_temperature(pooled_nw, pooled_y)

        # ── learning curve on SELECTION; keep the 64-game model as final ──
        learning_curve = []
        final_model = None
        for game_count in LEARNING_CURVE_TRAIN_GAME_COUNTS:
            subset_seeds = set(_learning_curve_subset_seeds(TRAIN_SEEDS, game_count))
            subset_mask = [r["seed"] in subset_seeds for r in train_records]
            subset_x = train_x[subset_mask]
            subset_y = train_y[subset_mask]
            subset_model, subset_stats = probe_v1.train_value_probe(subset_x, subset_y, selection_x, selection_y)
            subset_selection_probs = probe_v1.value_probe_predict_proba(subset_model, selection_x)
            learning_curve.append(
                {
                    "train_games": game_count,
                    "train_states": int(sum(subset_mask)),
                    "epochs_run": subset_stats["epochs_run"],
                    "selection_cross_entropy": probe_v1.cross_entropy(subset_selection_probs, selection_y),
                    "selection_brier_score": probe_v1.brier_score(subset_selection_probs, selection_y),
                    "selection_top1_accuracy": probe_v1.top1_accuracy(subset_selection_probs, selection_y),
                }
            )
            if game_count == 64:
                final_model = subset_model
                final_training_stats = subset_stats

        # ── TEST: touched exactly once, only for the final 64-game model ──
        test_learned_probs = probe_v1.value_probe_predict_proba(final_model, test_x)
        test_learned_eval = probe_v1.evaluate_predictor(test_learned_probs, test_y, test_buckets)

        test_uniform_probs = probe_v1.uniform_baseline_probs(len(test_records))
        test_uniform_eval = probe_v1.evaluate_predictor(test_uniform_probs, test_y, test_buckets)

        test_hard_leader_probs = probe_v1.leader_baseline_probs(test_leader_y)
        test_hard_leader_eval = probe_v1.evaluate_predictor(test_hard_leader_probs, test_y, test_buckets)

        test_probabilistic_leader_probs = probabilistic_leader_probs(test_nw, fitted_temperature)
        test_probabilistic_leader_eval = probe_v1.evaluate_predictor(test_probabilistic_leader_probs, test_y, test_buckets)

        # ── game-block bootstrap: learned vs. probabilistic leader, TEST only ──
        per_game_records: dict[int, list[dict]] = {}
        for i, rec in enumerate(test_records):
            per_game_records.setdefault(rec["seed"], []).append(
                {
                    "true_class": int(test_y[i]),
                    "learned_probs": test_learned_probs[i],
                    "baseline_probs": test_probabilistic_leader_probs[i],
                }
            )
        bootstrap_result = game_block_bootstrap_metric_diff(per_game_records)

        bucket_provenance = {
            "train": _bucket_provenance(train_records),
            "selection": _bucket_provenance(selection_records),
            "test": _bucket_provenance(test_records),
        }

        asu_modules_loaded = common.loaded_asu_modules()

    elapsed_s = time.perf_counter() - started

    payload = {
        "git_head_sha": git_head_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "config": {
            "self_play_seeds": list(SELF_PLAY_SEEDS),
            "train_seeds": list(TRAIN_SEEDS),
            "selection_seeds": list(SELECTION_SEEDS),
            "test_seeds": list(TEST_SEEDS),
            "max_rounds": MAX_ROUNDS,
            "policy": "POLICY_ONLY (no PUCT/search)",
            "opponents": "none - all 4 seats are the same checkpoint (clean self-play, zero fixed agents)",
            "round_buckets": [label for label, _, _ in ROUND_BUCKETS],
            "samples_per_cell": SAMPLES_PER_CELL,
            "sampling_method": "quantile-spread (min/median/max of available occurrences per game/seat/bucket), not first-N",
            "value_probe": {
                "input_dim": probe_v1.VALUE_PROBE_INPUT_DIM, "hidden_dim": probe_v1.VALUE_PROBE_HIDDEN_DIM,
                "output_dim": probe_v1.VALUE_PROBE_OUTPUT_DIM, "architecture": "same as 020 (300 -> 256 -> 4 MLP)",
            },
            "temperature_grid": list(TEMPERATURE_GRID),
            "fitted_temperature": fitted_temperature,
            "temperature_fit_data": "TRAIN + SELECTION pooled only, never TEST",
            "test_touch_policy": "TEST evaluated exactly once, only for the final 64-train-game model - never used for early stopping, hyperparameter/temperature selection, or model choice",
            "learning_curve_train_game_counts": list(LEARNING_CURVE_TRAIN_GAME_COUNTS),
            "learning_curve_evaluated_on": "SELECTION (not TEST) for all points, including the 64-game point",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "no_policy_training": True,
            "no_puct": True,
            "no_win_rate_evaluation": True,
            "no_arbitrary_go_kill_threshold": True,
            "reuses_020_games": False,
            "asu_involved": False,
        },
        "integrity": {
            "games_total": len(games),
            "total_illegal_actions": total_illegal,
            "total_crashed": total_crashed,
            "fixed_fallbacks": 0,
        },
        "leakage_guard": {
            "seed_overlaps": overlaps,
            "state_overlaps": state_overlaps,
        },
        "dataset": {
            "train_states": len(train_records),
            "selection_states": len(selection_records),
            "test_states": len(test_records),
        },
        "bucket_provenance": bucket_provenance,
        "temperature_grid_results": temperature_grid_results,
        "learning_curve": learning_curve,
        "final_value_probe_training": final_training_stats,
        "test_results": {
            "uniform": test_uniform_eval,
            "hard_net_worth_leader": test_hard_leader_eval,
            "probabilistic_net_worth_leader": test_probabilistic_leader_eval,
            "learned_value_probe": test_learned_eval,
        },
        "game_block_bootstrap_learned_vs_probabilistic_leader": bootstrap_result,
        "state_level_n_warning": "State counts above are NOT independent samples (multiple states per game are correlated) - use game_block_bootstrap_learned_vs_probabilistic_leader for uncertainty, not a state-level CI.",
        "asu_modules_loaded": asu_modules_loaded,
        "asu_modules_loaded_count": len(asu_modules_loaded),
        "elapsed_s": elapsed_s,
        "peak_rss_gib": rss.peak_gib,
    }

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded during probe: {asu_modules_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
