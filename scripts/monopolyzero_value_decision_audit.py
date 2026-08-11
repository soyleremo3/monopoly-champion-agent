"""Post-hoc, decision-critical audit of `021-monopolyzero-value-generalization-probe`.
No new self-play randomness, no new/different model, no PUCT/policy run.

021 never persisted per-state records (only aggregate metrics went into its
experiment log/stdout, per this project's small-raw-stdout convention) -
so there is no file to "just read" for the deeper per-state segmentation
this audit needs (leader margin, current-player rank, legal-action count,
decision phase). The only way to get real per-state data is to run 021's
EXACT deterministic pipeline again: same seeds, same config, same
ValueProbe training recipe, all imported directly from
scripts/monopolyzero_value_generalization_probe.py (`probe_v2`) and
scripts/monopolyzero_value_learnability_probe.py (`probe_v1`) rather than
redefined here. This adds zero new randomness and trains no new/different
model - it reproduces the SAME model 021 already reported on.

That claim is not just asserted: after regenerating, this script recomputes
the same four TEST-set summary metrics (uniform / hard leader /
probabilistic leader / learned ValueProbe) and RECONCILES them against
021's own logged values (`logs/experiments/021-*.json`) within a tight
numerical tolerance - it refuses to proceed with any audit conclusion if
they don't match, rather than silently presenting new/different results as
if they were 021's.

On top of the reconciled TEST set, this script adds four post-hoc
diagnostic axes never captured by 021: the probabilistic leader's own
top1-vs-top2 margin (quartile-binned), the deciding player's own
current net-worth rank, the legal-action count, and the decision phase
(`env.phase`). All segment-level breakdowns here are DIAGNOSTIC-ONLY /
exploratory subgroup analysis, not a confirmatory statistical test (021's
game-block bootstrap is the one confirmatory statistic this project
stands behind for this comparison). The margin-quartile and
current-player-rank axes are additionally OUTCOME-ADJACENT - both are
computed from the same net-worth signal the probabilistic leader baseline
itself is scored on, so a clean-looking accuracy-vs-margin relationship is
partly definitional, not fresh generalization evidence.

021's TEST set is used here ONLY for this descriptive audit - nothing is
fit, tuned, or selected against it (the temperature was already fixed by
021 on TRAIN+SELECTION; this script reconciles against that same fixed
value, never refits it).

Ends with a real A/B decision (unlike 013-021, which deliberately avoided
one): (A) drop the learned-value path for now and move to the
decision/policy win-rate phase, or (B) propose a new value hypothesis -
only if a segment clears a pre-stated bar (declared below, before running:
minimum segment size and minimum accuracy margin) showing the learned
ValueProbe meaningfully beating the leader baseline somewhere.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluation_protocol as ep  # noqa: E402
import monopolyzero_common as common  # noqa: E402
import monopolyzero_value_generalization_probe as probe_v2  # noqa: E402
import monopolyzero_value_learnability_probe as probe_v1  # noqa: E402

REPO_ROOT = common.REPO_ROOT
PRIOR_EXPERIMENT_LOG = REPO_ROOT / "logs" / "experiments" / "021-monopolyzero-value-generalization-probe.json"
RECONCILIATION_TOLERANCE = 1e-6

# Pre-stated (declared before running) bar for recommending B instead of A.
VALUE_PROBE_WIN_MIN_SEGMENT_STATES = 20
VALUE_PROBE_WIN_MIN_ACCURACY_MARGIN = 0.05  # learned ValueProbe must beat the leader by >=5 points

OUTCOME_ADJACENT_AXES = frozenset({"margin_quartile", "current_player_rank"})


def _quantile_summary(values) -> dict:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None}
    return {
        "min": float(np.min(arr)), "q25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)), "q75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def margin_quartile_label(margin: float, q25: float, median: float, q75: float) -> str:
    if margin <= q25:
        return "Q1(low margin)"
    if margin <= median:
        return "Q2"
    if margin <= q75:
        return "Q3"
    return "Q4(high margin)"


def segment_report(records: list[dict], segment_key_fn) -> dict:
    """Groups `records` (each needs seed/true_class/leader_probs/
    learned_probs) by segment_key_fn(record) and reports leader vs. learned
    ValueProbe accuracy/CE/Brier per segment, plus count and unique-game
    count. Diagnostic-only - no CI is computed per segment."""
    import numpy as np

    groups: dict = defaultdict(list)
    for record in records:
        groups[segment_key_fn(record)].append(record)

    report = {}
    for key, recs in groups.items():
        learned_probs = np.array([r["learned_probs"] for r in recs])
        leader_probs = np.array([r["leader_probs"] for r in recs])
        true_classes = np.array([r["true_class"] for r in recs])
        report[str(key)] = {
            "count": len(recs),
            "unique_games": len({r["seed"] for r in recs}),
            "leader": {
                "top1_accuracy": probe_v1.top1_accuracy(leader_probs, true_classes),
                "cross_entropy": probe_v1.cross_entropy(leader_probs, true_classes),
                "brier_score": probe_v1.brier_score(leader_probs, true_classes),
            },
            "learned_value_probe": {
                "top1_accuracy": probe_v1.top1_accuracy(learned_probs, true_classes),
                "cross_entropy": probe_v1.cross_entropy(learned_probs, true_classes),
                "brier_score": probe_v1.brier_score(learned_probs, true_classes),
            },
        }
    return report


def distribution(records: list[dict], key_fn) -> dict:
    counts = Counter(key_fn(record) for record in records)
    total = len(records)
    return {
        str(key): {"count": count, "fraction": (count / total) if total else None}
        for key, count in sorted(counts.items(), key=lambda kv: str(kv[0]))
    }


def find_value_probe_advantage_segments(
    all_segment_reports: dict,
    *, min_count: int = VALUE_PROBE_WIN_MIN_SEGMENT_STATES,
    min_accuracy_margin: float = VALUE_PROBE_WIN_MIN_ACCURACY_MARGIN,
) -> list[dict]:
    wins = []
    for axis_name, segments in all_segment_reports.items():
        for segment_label, stats in segments.items():
            if stats["count"] < min_count:
                continue
            accuracy_margin = stats["learned_value_probe"]["top1_accuracy"] - stats["leader"]["top1_accuracy"]
            if accuracy_margin >= min_accuracy_margin:
                wins.append(
                    {
                        "axis": axis_name, "segment": segment_label, "count": stats["count"],
                        "outcome_adjacent_axis": axis_name in OUTCOME_ADJACENT_AXES,
                        "leader_accuracy": stats["leader"]["top1_accuracy"],
                        "learned_accuracy": stats["learned_value_probe"]["top1_accuracy"],
                        "accuracy_margin": accuracy_margin,
                    }
                )
    return wins


# ── deterministic re-derivation of 021's exact TEST pipeline, + provenance ──


def _generate_games_with_provenance(model):
    from monopoly_bench.engine import MAX_DECISIONS_PER_TURN, NUM_PLAYERS, SharedGame, clone_env

    policy = common.build_local_policy_only(model)
    games = []
    for seed in probe_v2.SELF_PLAY_SEEDS:
        game = SharedGame.new(seed, probe_v2.MAX_ROUNDS)
        decision_budget = probe_v2.MAX_ROUNDS * NUM_PLAYERS * MAX_DECISIONS_PER_TURN
        candidates = {seat: {label: [] for label, _, _ in probe_v2.ROUND_BUCKETS} for seat in range(NUM_PLAYERS)}

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
                            {
                                "turn_index": turn_index, "round": current_round, "env_clone": clone_env(game),
                                "legal_action_count": len(legal), "decision_type": game.env.phase,
                            }
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
            for label, _, _ in probe_v2.ROUND_BUCKETS:
                pool = candidates[seat][label]
                picked_idx = probe_v2.quantile_indices(len(pool), probe_v2.SAMPLES_PER_CELL)
                sampled[seat][label] = [pool[i] for i in picked_idx]

        games.append(
            {
                "seed": seed, "sampled": sampled, "completed": finished, "winner": winner,
                "decisions": turn_index, "final_round": game.env.round,
                "illegal_actions": illegal_actions, "crashed": crashed, "error": error,
            }
        )
    return games


def _build_records_with_provenance(games):
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
                    current_player_rank = (
                        sorted(range(NUM_PLAYERS), key=lambda i: net_worth_relative[i], reverse=True).index(0) + 1
                    )
                    records.append(
                        {
                            "seed": g["seed"], "seat": seat, "round": snapshot["round"],
                            "turn_index": snapshot["turn_index"], "bucket": bucket_label,
                            "state": state_vector, "net_worth_relative": net_worth_relative,
                            "true_class_relative": order.index(winner),
                            "leader_class_relative": order.index(leader_absolute),
                            "legal_action_count": snapshot["legal_action_count"],
                            "decision_type": snapshot["decision_type"],
                            "current_player_rank": current_player_rank,
                        }
                    )
    return records


def main(argv: list[str] | None = None) -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)
    checkpoint_sha256 = probe_v1.verify_checkpoint()
    ep.require_seed_scope(probe_v2.SELF_PLAY_SEEDS, ep.SEED_CLASS_DEV, context="monopolyzero_value_decision_audit.py")

    if not PRIOR_EXPERIMENT_LOG.is_file():
        raise SystemExit(f"monopolyzero_value_decision_audit.py refuses to run: missing {PRIOR_EXPERIMENT_LOG}")
    with PRIOR_EXPERIMENT_LOG.open(encoding="utf-8") as handle:
        prior_log = json.load(handle)
    prior_test_overall = prior_log["metrics"]["test_overall"]
    prior_fitted_temperature = prior_log["algorithm_config"]["fitted_temperature"]

    common.ensure_reference_on_path()
    import random

    import numpy as np
    import torch

    from monopoly_bench.model import MonopolyZeroNet
    from monopoly_game_engine.state import STATE_DIM

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        model = MonopolyZeroNet.load_inference(probe_v1.CHECKPOINT_PATH)
        model.eval()

        games = _generate_games_with_provenance(model)
        total_illegal = sum(g["illegal_actions"] for g in games)
        total_crashed = sum(int(g["crashed"]) for g in games)
        if total_illegal or total_crashed:
            raise RuntimeError(
                f"Stopping: crashed={total_crashed} illegal={total_illegal} - unexpected, since this is the "
                "identical deterministic pipeline 021 already ran cleanly"
            )

        records = _build_records_with_provenance(games)
        train_seed_set = set(probe_v2.TRAIN_SEEDS)
        selection_seed_set = set(probe_v2.SELECTION_SEEDS)
        test_seed_set = set(probe_v2.TEST_SEEDS)

        train_records = [r for r in records if r["seed"] in train_seed_set]
        selection_records = [r for r in records if r["seed"] in selection_seed_set]
        test_records = [r for r in records if r["seed"] in test_seed_set]

        def _arrays(recs):
            x = np.stack([r["state"] for r in recs]).astype(np.float32) if recs else np.zeros((0, STATE_DIM), dtype=np.float32)
            y = np.array([r["true_class_relative"] for r in recs], dtype=np.int64)
            nw = np.array([r["net_worth_relative"] for r in recs], dtype=np.float64) if recs else np.zeros((0, 4))
            return x, y, nw

        train_x, train_y, train_nw = _arrays(train_records)
        selection_x, selection_y, selection_nw = _arrays(selection_records)
        test_x, test_y, test_nw = _arrays(test_records)

        pooled_nw = np.concatenate([train_nw, selection_nw], axis=0)
        pooled_y = np.concatenate([train_y, selection_y], axis=0)
        fitted_temperature, _ = probe_v2.fit_probabilistic_leader_temperature(pooled_nw, pooled_y)
        if fitted_temperature != prior_fitted_temperature:
            raise RuntimeError(
                f"Reconciliation failed: regenerated temperature {fitted_temperature} != 021's logged "
                f"{prior_fitted_temperature} - this is not 021's data, refusing to proceed"
            )

        final_model, _training_stats = probe_v1.train_value_probe(train_x, train_y, selection_x, selection_y)

        test_leader_y = np.array([r["leader_class_relative"] for r in test_records], dtype=np.int64)
        test_learned_probs = probe_v1.value_probe_predict_proba(final_model, test_x)
        test_uniform_probs = probe_v1.uniform_baseline_probs(len(test_records))
        test_hard_leader_probs = probe_v1.leader_baseline_probs(test_leader_y)
        test_probabilistic_leader_probs = probe_v2.probabilistic_leader_probs(test_nw, fitted_temperature)
        test_buckets = [r["bucket"] for r in test_records]

        reconciliation = {}
        for label, probs in (
            ("uniform", test_uniform_probs),
            ("hard_net_worth_leader", test_hard_leader_probs),
            ("probabilistic_net_worth_leader", test_probabilistic_leader_probs),
            ("learned_value_probe", test_learned_probs),
        ):
            evaluated = probe_v1.evaluate_predictor(probs, test_y, test_buckets)["overall"]
            prior = prior_test_overall[label]
            deltas = {
                "cross_entropy": abs(evaluated["cross_entropy"] - prior["cross_entropy"]),
                "brier_score": abs(evaluated["brier_score"] - prior["brier_score"]),
                "top1_accuracy": abs(evaluated["top1_accuracy"] - prior["top1_accuracy"]),
            }
            matches = all(delta < RECONCILIATION_TOLERANCE for delta in deltas.values())
            reconciliation[label] = {"matches_021": matches, "deltas": deltas}
            if not matches:
                raise RuntimeError(
                    f"Reconciliation failed for {label}: regenerated TEST metrics do not match 021's logged "
                    f"values within {RECONCILIATION_TOLERANCE} (deltas={deltas}) - refusing to draw any audit "
                    "conclusion from data that isn't verifiably 021's"
                )

        audit_records = []
        for i, r in enumerate(test_records):
            leader_probs_row = test_probabilistic_leader_probs[i]
            sorted_probs = sorted(leader_probs_row, reverse=True)
            margin = float(sorted_probs[0] - sorted_probs[1])
            audit_records.append(
                {
                    "seed": r["seed"], "bucket": r["bucket"], "current_player_rank": r["current_player_rank"],
                    "legal_action_count": r["legal_action_count"], "decision_type": r["decision_type"],
                    "margin": margin, "true_class": int(test_y[i]),
                    "leader_probs": leader_probs_row, "learned_probs": test_learned_probs[i],
                    "leader_correct": bool(np.argmax(leader_probs_row) == test_y[i]),
                }
            )

        margin_summary = _quantile_summary([rec["margin"] for rec in audit_records])
        for rec in audit_records:
            rec["margin_quartile"] = margin_quartile_label(
                rec["margin"], margin_summary["q25"], margin_summary["median"], margin_summary["q75"]
            )

        segment_reports = {
            "round_bucket": segment_report(audit_records, lambda r: r["bucket"]),
            "margin_quartile": segment_report(audit_records, lambda r: r["margin_quartile"]),
            "current_player_rank": segment_report(audit_records, lambda r: r["current_player_rank"]),
            "decision_type": segment_report(audit_records, lambda r: r["decision_type"]),
            "legal_action_count": segment_report(audit_records, lambda r: r["legal_action_count"]),
        }

        leader_wrong_records = [r for r in audit_records if not r["leader_correct"]]
        leader_wrong_distribution = {
            "count": len(leader_wrong_records),
            "fraction_of_test": (len(leader_wrong_records) / len(audit_records)) if audit_records else None,
            "by_round_bucket": distribution(leader_wrong_records, lambda r: r["bucket"]),
            "by_current_player_rank": distribution(leader_wrong_records, lambda r: r["current_player_rank"]),
            "by_decision_type": distribution(leader_wrong_records, lambda r: r["decision_type"]),
            "by_legal_action_count": distribution(leader_wrong_records, lambda r: r["legal_action_count"]),
            "by_margin_quartile": distribution(leader_wrong_records, lambda r: r["margin_quartile"]),
        }

        value_probe_advantage_segments = find_value_probe_advantage_segments(segment_reports)
        final_decision = "B" if value_probe_advantage_segments else "A"

        asu_modules_loaded = common.loaded_asu_modules()

    elapsed_s = time.perf_counter() - started

    decision = {
        "final_decision": final_decision,
        "option_a": "Drop the learned-value path for now; move to the decision/policy win-rate phase.",
        "option_b": "Propose a new value hypothesis - only warranted if a segment below clears the bar.",
        "rule": (
            f"Recommend B only if at least one segment (across round_bucket, margin_quartile, "
            f"current_player_rank, decision_type, legal_action_count) has >= {VALUE_PROBE_WIN_MIN_SEGMENT_STATES} "
            f"states AND the learned ValueProbe's top-1 accuracy exceeds the probabilistic leader's by >= "
            f"{VALUE_PROBE_WIN_MIN_ACCURACY_MARGIN:.0%}; otherwise A. Rule fixed in source before this audit ran."
        ),
        "qualifying_segments": value_probe_advantage_segments,
    }

    payload = {
        "git_head_sha": git_head_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "config": {
            "prior_experiment": "021-monopolyzero-value-generalization-probe",
            "reconciliation_tolerance": RECONCILIATION_TOLERANCE,
            "value_probe_win_min_segment_states": VALUE_PROBE_WIN_MIN_SEGMENT_STATES,
            "value_probe_win_min_accuracy_margin": VALUE_PROBE_WIN_MIN_ACCURACY_MARGIN,
            "outcome_adjacent_axes": sorted(OUTCOME_ADJACENT_AXES),
            "no_new_self_play": True,
            "no_new_or_different_model": True,
            "no_puct": True,
            "no_new_hyperparameter_or_temperature_selection_against_test": True,
            "asu_involved": False,
        },
        "reconciliation_against_021": reconciliation,
        "margin_summary": margin_summary,
        "segment_reports": segment_reports,
        "leader_wrong_distribution": leader_wrong_distribution,
        "decision": decision,
        "dataset": {"test_states": len(test_records), "leader_wrong_states": len(leader_wrong_records)},
        "asu_modules_loaded": asu_modules_loaded,
        "asu_modules_loaded_count": len(asu_modules_loaded),
        "elapsed_s": elapsed_s,
        "peak_rss_gib": rss.peak_gib,
    }

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded during audit: {asu_modules_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
