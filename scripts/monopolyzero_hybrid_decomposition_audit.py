"""Decomposes 023's +21.25-point HYBRID_COMPAT(BOTH) win-rate improvement
into its BUY_PROPERTY and ACCEPT_TRADE components, and checks whether the
effect survives against a repaired (non-crippled) peer population.

Two contexts:

CONTEXT 1 — crippled peers (023's exact setup): other 3 seats POLICY_ONLY.
  023's BASELINE (POLICY_ONLY) and BOTH (HYBRID_COMPAT, enable_buy=True,
  enable_trade=True — 023's only configuration) arms are deterministically
  regenerated here: same seeds, same checkpoint, same
  monopolyzero_common.play_local_game/build_local_policy_only/
  build_local_hybrid_compat_policy calls 023 itself used (imported and
  reused from 023's own module, not redefined), zero new randomness. Their
  aggregate stats are reconciled against 023's own logged values before
  anything else is computed — refusing to proceed on any mismatch. This is
  NOT a new/different measurement of those two arms; it is 023's exact
  result, mechanically reproduced because 023's log only persisted
  aggregates, not the per-game records this task's paired comparisons
  need. Two arms run fresh: BUY_ONLY (enable_buy=True, enable_trade=False)
  and TRADE_ONLY (enable_buy=False, enable_trade=True).

CONTEXT 2 — repaired peers: the other 3 seats are HYBRID_COMPAT(BOTH)
  instead of POLICY_ONLY — does the BUY_PROPERTY/ACCEPT_TRADE effect still
  show up when opponents are not also crippled? Four arms, all fresh:
  POLICY_ONLY, BUY_ONLY, TRADE_ONLY, and BOTH focus seats, all against 3
  HYBRID_COMPAT(BOTH) peers, same seed/seat pairing throughout. Since the
  BOTH arm's focus seat uses the exact same policy configuration as its 3
  peers, one self-play game per seed (not 4 separately-focused games)
  yields all 4 seats' paired records — see `_run_self_play_uniform_arm`.

For both contexts: PRIMARY seed-block paired randomization + seed-block
bootstrap (win-rate and net-worth diff) for BUY_ONLY/TRADE_ONLY/BOTH
against that context's own POLICY_ONLY reference arm, SECONDARY seat-level
McNemar, plus a recovered-fraction and additive-vs-synergy read on how much
of BOTH's total effect BUY_ONLY and TRADE_ONLY each individually account
for.

No training. No automatic GO/KILL. Built on scripts/monopolyzero_common.py
and scripts/evaluation_protocol.py; no monopoly_bench.adapters/.arena/
.training import, no ASU. Refuses to run unless PYTHONHASHSEED=0, the git
tree is clean, the checkpoint SHA-256 matches, and every seed is
registered in the DEV pool (reuses 023's 43000-43019 range — no new
registration).
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
import monopolyzero_hybrid_bootstrap_isolation_audit as audit_v1  # noqa: E402

RECONCILIATION_TOLERANCE = 1e-9  # deterministic reproduction — exact match expected
PRIOR_EXPERIMENT_LOG = audit_v1.REPO_ROOT / "logs" / "experiments" / "023-hybrid-bootstrap-isolation-audit.json"

SEEDS = audit_v1.SEEDS  # reused 43000-43019 (023's DEV registration), no new registration
MAX_ROUNDS = audit_v1.MAX_ROUNDS
NUM_SEATS = audit_v1.NUM_SEATS
CHECKPOINT_PATH = audit_v1.CHECKPOINT_PATH
BASELINE_CHECKPOINT_SHA256 = audit_v1.BASELINE_CHECKPOINT_SHA256

ARM_POLICY_ONLY = "policy_only"
ARM_BUY_ONLY = "buy_only"
ARM_TRADE_ONLY = "trade_only"
ARM_BOTH = "both"

SYNERGY_ADDITIVE_THRESHOLD_PP = 0.05  # descriptive label threshold only, not a GO/KILL rule


def _focus_policy_factory(model, arm: str):
    if arm == ARM_POLICY_ONLY:
        return lambda: common.build_local_policy_only(model)
    if arm == ARM_BUY_ONLY:
        return lambda: common.build_local_hybrid_compat_policy(model, enable_buy=True, enable_trade=False)
    if arm == ARM_TRADE_ONLY:
        return lambda: common.build_local_hybrid_compat_policy(model, enable_buy=False, enable_trade=True)
    if arm == ARM_BOTH:
        return lambda: common.build_local_hybrid_compat_policy(model, enable_buy=True, enable_trade=True)
    raise ValueError(f"unknown arm: {arm}")


def _run_rotation_arm(seeds, focus_factory, peer_policy_factory, *, max_rounds: int = MAX_ROUNDS) -> dict:
    """Standard 20-seed x 4-focus-seat rotation: a fresh focus-policy
    instance per game (so any internal .log stays game-scoped), a fresh
    peer-policy instance per non-focus seat per game (safe even when
    peer_policy_factory returns a stateless POLICY_ONLY singleton — its
    .log growth, if any, is simply never read)."""
    per_game: list[dict] = []
    records: list[dict] = []
    latencies: list[float] = []
    intervention_log: list[dict] = []
    total_illegal = 0
    total_crashed = 0
    incomplete = 0

    for seed in seeds:
        for focus_seat in range(NUM_SEATS):
            focus_policy = focus_factory()
            policies = {seat: peer_policy_factory() for seat in range(NUM_SEATS) if seat != focus_seat}
            policies[focus_seat] = focus_policy
            outcome = common.play_local_game(
                game_id=seed * 10 + focus_seat, seed=seed, policies=policies,
                max_rounds=max_rounds, record_seats=set(),
            )
            total_illegal += outcome.illegal_actions
            total_crashed += int(outcome.crashed)
            incomplete += int(not outcome.completed)
            latencies.extend(outcome.search_latencies_s)

            won = bool(outcome.completed and outcome.winner == focus_seat)
            net_worth = outcome.final_net_worth[focus_seat] if outcome.final_net_worth else 0.0
            records.append({"seed": seed, "seat": focus_seat, "win": won, "net_worth": net_worth})
            per_game.append(
                {
                    "seed": seed, "focus_seat": focus_seat, "completed": outcome.completed,
                    "winner": outcome.winner, "focus_won": won, "rounds": outcome.final_round,
                    "decisions": outcome.decisions, "focus_net_worth": net_worth,
                    "round_capped": outcome.final_round >= max_rounds,
                }
            )
            hybrid_log = getattr(focus_policy, "log", None)
            if hybrid_log is not None:
                for idx, entry in enumerate(hybrid_log):
                    tagged = dict(entry)
                    tagged.update(seed=seed, focus_seat=focus_seat, decision_index=idx)
                    intervention_log.append(tagged)

    return {
        "per_game": per_game, "records": records, "latencies": latencies,
        "intervention_log": intervention_log, "total_illegal": total_illegal,
        "total_crashed": total_crashed, "incomplete": incomplete,
        "games_played": len(seeds) * NUM_SEATS,  # one physical game per (seed, focus_seat)
    }


def _run_self_play_uniform_arm(seeds, arm, model, *, max_rounds: int = MAX_ROUNDS) -> dict:
    """When the focus arm's policy configuration is IDENTICAL to its
    peers' (context 2's BOTH arm: focus HYBRID_COMPAT(BOTH) vs. 3
    HYBRID_COMPAT(BOTH) peers), one self-play game per seed yields all 4
    seats' paired records. Decision-seed mixing
    (monopolyzero_common._mix_decision_seed) depends only on (seed,
    turn_index, seat), never on which seat is labeled 'focus', so a game
    with seed S and 4 identical per-seat policy instances is
    mathematically identical, seat-for-seat, to running that same seed as
    4 separately-focused rotation games — avoids replaying the same
    trajectory 4 times."""
    factory = _focus_policy_factory(model, arm)
    per_game: list[dict] = []
    records: list[dict] = []
    latencies: list[float] = []
    intervention_log: list[dict] = []
    total_illegal = 0
    total_crashed = 0
    incomplete = 0

    for seed in seeds:
        seat_policies = {seat: factory() for seat in range(NUM_SEATS)}
        outcome = common.play_local_game(
            game_id=seed * 10, seed=seed, policies=seat_policies,
            max_rounds=max_rounds, record_seats=set(),
        )
        total_illegal += outcome.illegal_actions
        total_crashed += int(outcome.crashed)
        incomplete += int(not outcome.completed)
        latencies.extend(outcome.search_latencies_s)

        for seat in range(NUM_SEATS):
            won = bool(outcome.completed and outcome.winner == seat)
            net_worth = outcome.final_net_worth[seat] if outcome.final_net_worth else 0.0
            records.append({"seed": seed, "seat": seat, "win": won, "net_worth": net_worth})
            per_game.append(
                {
                    "seed": seed, "focus_seat": seat, "completed": outcome.completed,
                    "winner": outcome.winner, "focus_won": won, "rounds": outcome.final_round,
                    "decisions": outcome.decisions, "focus_net_worth": net_worth,
                    "round_capped": outcome.final_round >= max_rounds,
                }
            )
            hybrid_log = getattr(seat_policies[seat], "log", None)
            if hybrid_log is not None:
                for idx, entry in enumerate(hybrid_log):
                    tagged = dict(entry)
                    tagged.update(seed=seed, focus_seat=seat, decision_index=idx)
                    intervention_log.append(tagged)

    return {
        "per_game": per_game, "records": records, "latencies": latencies,
        "intervention_log": intervention_log, "total_illegal": total_illegal,
        "total_crashed": total_crashed, "incomplete": incomplete,
        "games_played": len(seeds),  # one physical self-play game per seed, not per (seed, seat)
    }


def reconcile_arm_against_023(regenerated_summary: dict, logged_summary: dict, *, label: str) -> dict:
    """Compares only deterministic fields (never latency, which is
    inherently wall-clock/machine-load-dependent and not reproducible)."""
    fields = ("games", "wins", "win_rate", "bankruptcy_rate", "mean_net_worth", "median_net_worth", "round_cap_rate")
    deltas: dict[str, float | None] = {}
    for field in fields:
        regenerated_value = regenerated_summary.get(field)
        logged_value = logged_summary.get(field)
        if regenerated_value is None or logged_value is None:
            deltas[field] = None
        else:
            deltas[field] = abs(regenerated_value - logged_value)

    def _stringify_seat_keys(wins_by_seat: dict | None) -> dict | None:
        # JSON object keys are always strings, so 023's logged wins_by_seat
        # round-tripped through json.loads has string keys ("0","1",...)
        # while a freshly computed summary has int keys (0,1,...) - normalize
        # both to strings before comparing, or every regenerated arm would
        # spuriously fail reconciliation despite identical values.
        if wins_by_seat is None:
            return None
        return {str(seat): count for seat, count in wins_by_seat.items()}

    wins_by_seat_match = _stringify_seat_keys(regenerated_summary.get("wins_by_seat")) == _stringify_seat_keys(
        logged_summary.get("wins_by_seat")
    )
    numeric_deltas = [delta for delta in deltas.values() if delta is not None]
    max_delta = max(numeric_deltas) if numeric_deltas else 0.0
    matches = (max_delta <= RECONCILIATION_TOLERANCE) and wins_by_seat_match

    result = {"matches_023": matches, "max_delta": max_delta, "wins_by_seat_match": wins_by_seat_match, "deltas": deltas}
    if not matches:
        raise RuntimeError(f"Reconciliation FAILED for {label} against 023's log: {result}")
    return result


def recovery_and_synergy(*, baseline: dict, buy_only: dict, trade_only: dict, both: dict, metric: str) -> dict:
    """metric: 'win_rate' or 'mean_net_worth' — computes how much of BOTH's
    total effect (vs. that context's own POLICY_ONLY reference arm)
    BUY_ONLY and TRADE_ONLY individually recover, and whether BOTH's
    effect is approximately the sum of the two individual effects
    (additive) or not (synergy/redundancy)."""
    base_value = baseline[metric]
    buy_value = buy_only[metric]
    trade_value = trade_only[metric]
    both_value = both[metric]

    buy_effect = buy_value - base_value
    trade_effect = trade_value - base_value
    both_effect = both_value - base_value
    sum_of_individual = buy_effect + trade_effect
    interaction = both_effect - sum_of_individual

    if metric == "win_rate":
        threshold = SYNERGY_ADDITIVE_THRESHOLD_PP
    else:
        threshold = None  # no pre-stated label threshold for net-worth units

    if threshold is not None:
        if abs(interaction) < threshold:
            interaction_read = "approximately additive"
        elif interaction > 0:
            interaction_read = "super-additive (positive synergy)"
        else:
            interaction_read = "sub-additive (redundant/overlapping)"
    else:
        interaction_read = None

    return {
        "metric": metric,
        f"baseline_{metric}": base_value,
        f"buy_only_{metric}": buy_value,
        f"trade_only_{metric}": trade_value,
        f"both_{metric}": both_value,
        "buy_only_effect": buy_effect,
        "trade_only_effect": trade_effect,
        "both_effect": both_effect,
        "buy_only_recovers_fraction_of_both": (buy_effect / both_effect) if both_effect else None,
        "trade_only_recovers_fraction_of_both": (trade_effect / both_effect) if both_effect else None,
        "sum_of_individual_effects": sum_of_individual,
        "both_minus_sum_of_individual": interaction,
        "interaction_read": interaction_read,
    }


def main(argv: list[str] | None = None) -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)

    ep.require_seed_scope(SEEDS, ep.SEED_CLASS_DEV, context="monopolyzero_hybrid_decomposition_audit.py")

    baseline_checkpoint_sha256 = audit_v1.verify_baseline_checkpoint()

    if not PRIOR_EXPERIMENT_LOG.is_file():
        raise SystemExit(f"Missing prior experiment log: {PRIOR_EXPERIMENT_LOG}")
    prior_log = json.loads(PRIOR_EXPERIMENT_LOG.read_text(encoding="utf-8"))
    prior_results = prior_log["metrics"]["results"]

    common.ensure_reference_on_path()
    import random

    import numpy as np
    import torch

    from monopoly_bench.engine import NUM_PLAYERS
    from monopoly_bench.model import MonopolyZeroNet

    if NUM_PLAYERS != NUM_SEATS:
        raise RuntimeError(f"Expected {NUM_SEATS} players, engine reports {NUM_PLAYERS}")

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        model = MonopolyZeroNet.load_inference(CHECKPOINT_PATH)
        model.eval()
        policy_only = common.build_local_policy_only(model)

        crippled_peer_factory = lambda: policy_only  # noqa: E731 - stateless, safe to share
        repaired_peer_factory = lambda: common.build_local_hybrid_compat_policy(model)  # noqa: E731

        # ── CONTEXT 1: crippled peers (023's exact setup) ──
        ctx1_runs = {
            ARM_POLICY_ONLY: _run_rotation_arm(SEEDS, _focus_policy_factory(model, ARM_POLICY_ONLY), crippled_peer_factory),
            ARM_BOTH: _run_rotation_arm(SEEDS, _focus_policy_factory(model, ARM_BOTH), crippled_peer_factory),
            ARM_BUY_ONLY: _run_rotation_arm(SEEDS, _focus_policy_factory(model, ARM_BUY_ONLY), crippled_peer_factory),
            ARM_TRADE_ONLY: _run_rotation_arm(SEEDS, _focus_policy_factory(model, ARM_TRADE_ONLY), crippled_peer_factory),
        }

        # ── CONTEXT 2: repaired (HYBRID_COMPAT BOTH) peers ──
        ctx2_runs = {
            ARM_POLICY_ONLY: _run_rotation_arm(SEEDS, _focus_policy_factory(model, ARM_POLICY_ONLY), repaired_peer_factory),
            ARM_BUY_ONLY: _run_rotation_arm(SEEDS, _focus_policy_factory(model, ARM_BUY_ONLY), repaired_peer_factory),
            ARM_TRADE_ONLY: _run_rotation_arm(SEEDS, _focus_policy_factory(model, ARM_TRADE_ONLY), repaired_peer_factory),
            ARM_BOTH: _run_self_play_uniform_arm(SEEDS, ARM_BOTH, model),
        }

        asu_modules_loaded = common.loaded_asu_modules()

    elapsed_s = time.perf_counter() - started

    total_illegal = sum(run["total_illegal"] for run in (*ctx1_runs.values(), *ctx2_runs.values()))
    total_crashed = sum(run["total_crashed"] for run in (*ctx1_runs.values(), *ctx2_runs.values()))
    incomplete_games = sum(run["incomplete"] for run in (*ctx1_runs.values(), *ctx2_runs.values()))

    if total_illegal or total_crashed or incomplete_games:
        payload = {
            "status": "FAILED_DURING_GAME_GENERATION", "git_head_sha": git_head_sha,
            "total_illegal_actions": total_illegal, "total_crashed": total_crashed,
            "incomplete_games": incomplete_games,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        raise RuntimeError(
            f"Stopping before any stats: illegal={total_illegal} crashed={total_crashed} incomplete={incomplete_games}"
        )

    ctx1_summaries = {
        arm: audit_v1._arm_summary(f"ctx1_{arm}", run["per_game"], run["latencies"])
        for arm, run in ctx1_runs.items()
    }
    ctx2_summaries = {
        arm: audit_v1._arm_summary(f"ctx2_{arm}", run["per_game"], run["latencies"])
        for arm, run in ctx2_runs.items()
    }

    reconciliation = {
        "ctx1_policy_only_vs_023_baseline": reconcile_arm_against_023(
            ctx1_summaries[ARM_POLICY_ONLY], prior_results["baseline_policy_only"], label="ctx1 POLICY_ONLY vs 023 BASELINE"
        ),
        "ctx1_both_vs_023_candidate": reconcile_arm_against_023(
            ctx1_summaries[ARM_BOTH], prior_results["candidate_hybrid_compat"], label="ctx1 BOTH vs 023 CANDIDATE"
        ),
    }

    ctx1_intervention = {
        arm: audit_v1.intervention_audit(run["intervention_log"])
        for arm, run in ctx1_runs.items() if arm != ARM_POLICY_ONLY
    }
    ctx2_intervention = {
        arm: audit_v1.intervention_audit(run["intervention_log"])
        for arm, run in ctx2_runs.items() if arm != ARM_POLICY_ONLY
    }

    def _paired(baseline_records, candidate_records):
        return ep.paired_evaluation_summary(
            baseline_records=baseline_records, candidate_records=candidate_records,
            baseline_fallbacks=0, candidate_fallbacks=0, expected_seats=NUM_SEATS,
        )

    ctx1_paired = {
        "buy_only_vs_baseline": _paired(ctx1_runs[ARM_POLICY_ONLY]["records"], ctx1_runs[ARM_BUY_ONLY]["records"]),
        "trade_only_vs_baseline": _paired(ctx1_runs[ARM_POLICY_ONLY]["records"], ctx1_runs[ARM_TRADE_ONLY]["records"]),
        "both_vs_baseline": _paired(ctx1_runs[ARM_POLICY_ONLY]["records"], ctx1_runs[ARM_BOTH]["records"]),
    }
    ctx2_paired = {
        "buy_only_vs_policy_only": _paired(ctx2_runs[ARM_POLICY_ONLY]["records"], ctx2_runs[ARM_BUY_ONLY]["records"]),
        "trade_only_vs_policy_only": _paired(ctx2_runs[ARM_POLICY_ONLY]["records"], ctx2_runs[ARM_TRADE_ONLY]["records"]),
        "both_vs_policy_only": _paired(ctx2_runs[ARM_POLICY_ONLY]["records"], ctx2_runs[ARM_BOTH]["records"]),
    }

    ctx1_decomposition = {
        "win_rate": recovery_and_synergy(
            baseline=ctx1_summaries[ARM_POLICY_ONLY], buy_only=ctx1_summaries[ARM_BUY_ONLY],
            trade_only=ctx1_summaries[ARM_TRADE_ONLY], both=ctx1_summaries[ARM_BOTH], metric="win_rate",
        ),
        "mean_net_worth": recovery_and_synergy(
            baseline=ctx1_summaries[ARM_POLICY_ONLY], buy_only=ctx1_summaries[ARM_BUY_ONLY],
            trade_only=ctx1_summaries[ARM_TRADE_ONLY], both=ctx1_summaries[ARM_BOTH], metric="mean_net_worth",
        ),
    }
    ctx2_decomposition = {
        "win_rate": recovery_and_synergy(
            baseline=ctx2_summaries[ARM_POLICY_ONLY], buy_only=ctx2_summaries[ARM_BUY_ONLY],
            trade_only=ctx2_summaries[ARM_TRADE_ONLY], both=ctx2_summaries[ARM_BOTH], metric="win_rate",
        ),
        "mean_net_worth": recovery_and_synergy(
            baseline=ctx2_summaries[ARM_POLICY_ONLY], buy_only=ctx2_summaries[ARM_BUY_ONLY],
            trade_only=ctx2_summaries[ARM_TRADE_ONLY], both=ctx2_summaries[ARM_BOTH], metric="mean_net_worth",
        ),
    }

    payload = {
        "status": "OK",
        "git_head_sha": git_head_sha,
        "config": {
            "seeds": list(SEEDS), "max_rounds": MAX_ROUNDS,
            "context_1": "crippled peers - other 3 seats POLICY_ONLY (023's exact setup); POLICY_ONLY/BOTH regenerated+reconciled against 023's log, BUY_ONLY/TRADE_ONLY fresh",
            "context_2": "repaired peers - other 3 seats HYBRID_COMPAT(BOTH); all 4 arms fresh, BOTH arm uses 20 self-play games (not 80) since focus==peer policy",
            "games_run": {
                # physical games actually executed (not paired-record count -
                # the self-play-optimized context_2 BOTH arm plays 20 games
                # but yields 80 records, 4 seats each; see per-arm breakdown)
                "context_1": sum(run["games_played"] for run in ctx1_runs.values()),
                "context_2": sum(run["games_played"] for run in ctx2_runs.values()),
                "context_1_by_arm": {arm: run["games_played"] for arm, run in ctx1_runs.items()},
                "context_2_by_arm": {arm: run["games_played"] for arm, run in ctx2_runs.items()},
            },
            "asu_involved": False,
        },
        "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
        "prior_experiment_log": str(PRIOR_EXPERIMENT_LOG.relative_to(audit_v1.REPO_ROOT)),
        "reconciliation_against_023": reconciliation,
        "context_1_crippled_peers": {
            "results": ctx1_summaries,
            "paired_evaluation": ctx1_paired,
            "intervention_audit": ctx1_intervention,
            "decomposition": ctx1_decomposition,
        },
        "context_2_repaired_peers": {
            "results": ctx2_summaries,
            "paired_evaluation": ctx2_paired,
            "intervention_audit": ctx2_intervention,
            "decomposition": ctx2_decomposition,
        },
        "total_illegal_actions": total_illegal,
        "total_crashed": total_crashed,
        "incomplete_games": incomplete_games,
        "asu_modules_loaded": asu_modules_loaded,
        "asu_modules_loaded_count": len(asu_modules_loaded),
        "elapsed_s": elapsed_s,
        "peak_rss_gib": rss.peak_gib,
    }

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded during evaluation: {asu_modules_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
