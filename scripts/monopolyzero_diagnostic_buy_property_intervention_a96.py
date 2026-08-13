"""Isolated diagnostic (NOT a champion gate, NOT auto-promoting): does a
narrow, non-ASU BUY_PROPERTY runtime override fix A96's documented
BUY=0 pathology without degrading its proven PPO-family strength?

Two arms only - PURE_A96 (unmodified) and BUY_SIMPLE (BUY_PROPERTY gated
by monopoly_game_engine.agent_ppo.fixed_buy_decision, not ASU-derived).
The originally-scoped third arm (BUY_SAFETY, gated by
ASU_FROZEN_TEACHER.core.safety_breakdown) was dropped before any code was
written: CLAUDE.md's ASU section forbids ASU as a runtime fallback "under
any circumstance", documented as a competition-rules requirement, not a
style preference - the user, asked explicitly, chose to drop that arm
rather than override the rule. Zero ASU_FROZEN_TEACHER import anywhere in
this file. See docs/DIAGNOSTIC_BUY_PROPERTY_INTERVENTION_A96.md for the
full pre-registered design, seed provenance, and decision rule, written
before this runner is ever executed.

Minimal reuse: BUY_SIMPLE's non-BUY-opportunity code path is byte-for-byte
the same mask/forward-pass/counter-bookkeeping shape as
screen.build_masked_argmax_policy (copied, not reinvented, since that
function has no hook for a conditional per-decision action override).
Fixed-lineup context reuses monopolyzero_common.LocalFixedPolicy +
play_local_game unmodified (same as the prior A96-vs-80-vs-FPAgents
diagnostic). Statistics reuse evaluation_protocol.pair_records +
paired_seed_block_bootstrap and screen.summarize() unmodified. The
integrity check reuses play_local_game's existing shadow_policy mechanism
unmodified.

Checkpoints are loaded from the MAIN checkout's gitignored artifacts
directory by absolute path - this worktree has no artifacts/ of its own,
and none are copied/rebuilt/regenerated here.
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
import monopolyzero_pure_ppo_strength_screen as screen  # noqa: E402
from monopolyzero_challenger_gate_96_vs_champion_32_64_128 import load_and_verify  # noqa: E402

MAIN_CHECKOUT_ARTIFACT_DIR = Path(
    r"C:\Users\Emrullah Soyler\Desktop\Exposure Academy Projects\Second Week\Monoply"
    r"\artifacts\monopolyzero_pure_ppo_learnability_gate"
)

DEV_SEED_BASE, N_SEEDS = 53000, 12

CHECKPOINTS = {
    "A96": ("candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt",
            "78585ed4e2400d024633ee2878d8b88243f7f4a9f498d9019a73e44b3a830f51",
            "2bd1e9bad3d6e0100e033507f72f15b5088d8f8fb2f04dba1ae9d759b5a34a40"),
    "former_80": ("candidate_ppo_80.pt", "e47c8d4559c1d74cfceffe114fa069a8e7a2845ba60af68be0b0ae4bd37e1dae",
                  "7442f99e51619b7f6b53bc662e74d360e0a1f9cf2365b8dcb003d5a0cc3eda11"),
}

ARM_NAMES = ("PURE_A96", "BUY_SIMPLE")
CONTEXTS = ("clean_ppo", "structural_stress")


# ── BUY_SIMPLE policy: byte-for-byte same non-BUY mechanics as build_masked_argmax_policy ──


def build_buy_simple_policy(actor, device, counters, env_holder, opportunity_log):
    import torch

    from monopoly_bench.engine import action_family
    from monopoly_game_engine.actions import ACTION_SPACE_SIZE, ActionType
    from monopoly_game_engine.agent_ppo import fixed_buy_decision

    buy_id = int(ActionType.BUY_PROPERTY)
    accept_id = int(ActionType.ACCEPT_TRADE)
    decline_id = int(ActionType.DECLINE_TRADE)
    trade_offer_families = ("buy_trade", "sell_trade", "exch_trade")

    class _BuySimplePolicy:
        kind = "policy_only"

        def choose(self, game, seat: int, decision_seed: int):
            started = time.perf_counter()
            env = game.env
            if env_holder is not None:
                env_holder[0] = env
            legal = tuple(env.get_allowed_actions(seat))
            opportunity = buy_id in legal
            opportunity_log.append(opportunity)

            if opportunity and fixed_buy_decision(env, seat):
                chosen_action = buy_id
            else:
                maskable_legal = tuple(a for a in legal if a != buy_id) if opportunity else legal
                state = env._get_state(seat)
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool, device=device)
                mask_t[0, list(maskable_legal)] = True
                with torch.inference_mode():
                    log_probs = actor(state_t, mask_t)
                chosen_action = int(torch.argmax(log_probs, dim=-1).item())
            latency_s = time.perf_counter() - started

            if counters is not None:
                counters["decisions"] += 1
                if buy_id in legal:
                    counters["buy_property_opportunities"] += 1
                    counters["buy_property_chosen"] += int(chosen_action == buy_id)
                if accept_id in legal:
                    counters["accept_trade_opportunities"] += 1
                    counters["accept_trade_chosen"] += int(chosen_action == accept_id)
                if decline_id in legal:
                    counters["decline_trade_opportunities"] += 1
                    counters["decline_trade_chosen"] += int(chosen_action == decline_id)
                if action_family(chosen_action) in trade_offer_families:
                    counters["trade_offer_chosen"] += 1

            return common._PolicyOnlyResult(
                chosen_action=chosen_action, visits={chosen_action: 1}, q_values={},
                root_value=None, simulations=0, latency_s=latency_s,
            )

    return _BuySimplePolicy()


def make_focus_policy(arm: str, actor, device, counters, env_holder, opportunity_log: list | None):
    if arm == "PURE_A96":
        return screen.build_masked_argmax_policy(actor, device, counters, env_holder)
    if arm == "BUY_SIMPLE":
        return build_buy_simple_policy(actor, device, counters, env_holder, opportunity_log)
    raise ValueError(f"unknown arm {arm!r}")


def assert_shadow_integrity(opportunity_log: list[bool], shadow_decisions: list[dict]) -> None:
    if len(opportunity_log) != len(shadow_decisions):
        raise RuntimeError(
            f"shadow-integrity check: decision count mismatch - opportunity_log has "
            f"{len(opportunity_log)} entries, shadow_decisions has {len(shadow_decisions)}"
        )
    for had_opportunity, shadow in zip(opportunity_log, shadow_decisions):
        if not had_opportunity and not shadow["agree"]:
            raise RuntimeError(
                f"shadow-integrity check FAILED: non-BUY-opportunity decision at turn "
                f"{shadow['turn_index']} diverged from PURE_A96 (actual={shadow['actual_action']}, "
                f"shadow={shadow['shadow_action']}) - BUY_SIMPLE must be action-identical to "
                f"PURE_A96 whenever BUY_PROPERTY is not legal"
            )


# ── context A: clean PPO (focus arm vs 3 copies of the former champion) ──


def play_one_clean_context_game(*, game_id: int, seed: int, arm: str, focus_actor, baseline_actor,
                                 focus_seat: int, device, max_rounds: int) -> dict:
    counters_by_seat = {seat: screen._new_counters() for seat in range(screen.NUM_SEATS)}
    env_holder: list = [None]
    opportunity_log: list[bool] = []

    policies = {}
    for seat in range(screen.NUM_SEATS):
        if seat == focus_seat:
            policies[seat] = make_focus_policy(arm, focus_actor, device, counters_by_seat[seat], env_holder, opportunity_log)
        else:
            policies[seat] = screen.build_masked_argmax_policy(baseline_actor, device, counters_by_seat[seat], env_holder)

    shadow_policy = screen.build_masked_argmax_policy(focus_actor, device, None, None) if arm == "BUY_SIMPLE" else None
    shadow_seats = {focus_seat} if shadow_policy is not None else set()

    outcome = common.play_local_game(
        game_id=game_id, seed=seed, policies=policies, max_rounds=max_rounds,
        record_seats=set(), shadow_policy=shadow_policy, shadow_seats=shadow_seats,
    )
    if shadow_policy is not None:
        assert_shadow_integrity(opportunity_log, outcome.shadow_decisions)

    return _finish_game(game_id, seed, focus_seat, outcome, env_holder, counters_by_seat, fixed_agent_fallbacks=None)


# ── context B: structural stress (focus arm vs FPAgentA/B/C) ──────────


def play_one_stress_context_game(*, game_id: int, seed: int, arm: str, focus_actor,
                                  focus_seat: int, device, max_rounds: int) -> dict:
    from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC

    counters_by_seat = {seat: screen._new_counters() for seat in range(screen.NUM_SEATS)}
    env_holder: list = [None]
    opportunity_log: list[bool] = []

    other_seats = [seat for seat in range(screen.NUM_SEATS) if seat != focus_seat]
    fixed_policies = {seat: common.LocalFixedPolicy(cls) for seat, cls in zip(other_seats, (FPAgentA, FPAgentB, FPAgentC))}
    policies = {focus_seat: make_focus_policy(arm, focus_actor, device, counters_by_seat[focus_seat], env_holder, opportunity_log)}
    policies.update(fixed_policies)

    shadow_policy = screen.build_masked_argmax_policy(focus_actor, device, None, None) if arm == "BUY_SIMPLE" else None
    shadow_seats = {focus_seat} if shadow_policy is not None else set()

    outcome = common.play_local_game(
        game_id=game_id, seed=seed, policies=policies, max_rounds=max_rounds,
        record_seats=set(), shadow_policy=shadow_policy, shadow_seats=shadow_seats,
    )
    if shadow_policy is not None:
        assert_shadow_integrity(opportunity_log, outcome.shadow_decisions)

    fallback_counts = {policy.name: policy.fallback_count for policy in fixed_policies.values()}
    return _finish_game(game_id, seed, focus_seat, outcome, env_holder, counters_by_seat, fixed_agent_fallbacks=fallback_counts)


def _finish_game(game_id, seed, focus_seat, outcome, env_holder, counters_by_seat, *, fixed_agent_fallbacks) -> dict:
    final_env = env_holder[0]
    per_seat: dict[int, dict] = {}
    for seat in range(screen.NUM_SEATS):
        bankrupt = properties_owned = None
        if outcome.completed and final_env is not None:
            player = final_env.players[seat]
            bankrupt = bool(player.bankrupt)
            properties_owned = len(player.properties)
        per_seat[seat] = {
            "is_candidate": seat == focus_seat,
            "win": bool(outcome.completed and outcome.winner == seat),
            "net_worth": float(outcome.final_net_worth[seat]) if outcome.completed and outcome.final_net_worth else None,
            "bankrupt": bankrupt,
            "properties_owned": properties_owned,
            **counters_by_seat[seat],
        }
    return {
        "game_id": game_id, "seed": seed, "focus_seat": focus_seat,
        "completed": outcome.completed, "winner": outcome.winner, "decisions": outcome.decisions,
        "final_round": outcome.final_round, "round_cap_hit": bool(outcome.completed and outcome.final_round >= 200),
        "illegal_actions": outcome.illegal_actions, "crashed": outcome.crashed, "error": outcome.error,
        "per_seat": per_seat, "fixed_agent_fallbacks": fixed_agent_fallbacks,
        "focus_latencies_s": list(outcome.search_latencies_s),
    }


# ── aggregation ──────────────────────────────────────────────────────────


def summarize_arm_context(games: list[dict]) -> dict:
    summary = screen.summarize(games)
    all_latencies = [latency for game in games for latency in game["focus_latencies_s"]]
    summary["focus_inference_latency_s_mean"] = (sum(all_latencies) / len(all_latencies)) if all_latencies else None
    summary["focus_inference_latency_s_max"] = max(all_latencies) if all_latencies else None
    if games[0]["fixed_agent_fallbacks"] is not None:
        totals: dict[str, int] = {}
        for game in games:
            for name, count in game["fixed_agent_fallbacks"].items():
                totals[name] = totals.get(name, 0) + count
        summary["fixed_agent_fallback_counts"] = totals
        summary["fixed_agent_fallback_total"] = sum(totals.values())
    return summary


def _to_records(games: list[dict]) -> list[dict]:
    return [
        {"seed": game["seed"], "seat": game["focus_seat"],
         "win": game["per_seat"][game["focus_seat"]]["win"],
         "net_worth": game["per_seat"][game["focus_seat"]]["net_worth"]}
        for game in games
    ]


def main() -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)
    common.ensure_reference_on_path()

    seeds = screen._seed_range(DEV_SEED_BASE, N_SEEDS)
    ep.require_seed_scope(seeds, ep.SEED_CLASS_DEV, context="diagnostic_buy_property_intervention_a96.py")

    agents, hash_report = {}, {}
    for key, (filename, ck_sha, actor_sha) in CHECKPOINTS.items():
        agent, verified = load_and_verify(MAIN_CHECKOUT_ARTIFACT_DIR / filename, ck_sha, actor_sha)
        agents[key] = agent
        hash_report[key] = verified

    import torch

    device = torch.device("cpu")
    games: dict[str, dict[str, list[dict]]] = {arm: {} for arm in ARM_NAMES}
    started = time.perf_counter()
    with common.RssMonitor() as rss:
        for arm in ARM_NAMES:
            clean_games, stress_games = [], []
            game_id = 0
            for seed in seeds:
                for focus_seat in range(screen.NUM_SEATS):
                    game_id += 1
                    clean_games.append(play_one_clean_context_game(
                        game_id=game_id, seed=seed, arm=arm, focus_actor=agents["A96"].actor,
                        baseline_actor=agents["former_80"].actor, focus_seat=focus_seat,
                        device=device, max_rounds=screen.MAX_ROUNDS,
                    ))
            for seed in seeds:
                for focus_seat in range(screen.NUM_SEATS):
                    game_id += 1
                    stress_games.append(play_one_stress_context_game(
                        game_id=game_id, seed=seed, arm=arm, focus_actor=agents["A96"].actor,
                        focus_seat=focus_seat, device=device, max_rounds=screen.MAX_ROUNDS,
                    ))
            games[arm]["clean_ppo"] = clean_games
            games[arm]["structural_stress"] = stress_games
    elapsed_s = time.perf_counter() - started

    summaries = {arm: {ctx: summarize_arm_context(games[arm][ctx]) for ctx in CONTEXTS} for arm in ARM_NAMES}

    paired: dict[str, dict] = {}
    for ctx in CONTEXTS:
        paired_records = ep.pair_records(
            baseline_records=_to_records(games["PURE_A96"][ctx]),
            candidate_records=_to_records(games["BUY_SIMPLE"][ctx]),
            expected_seats=4,
        )
        paired[ctx] = ep.paired_seed_block_bootstrap(paired_records, n_resamples=2000, bootstrap_seed=0)

    clean_ci = paired["clean_ppo"]["win_rate_diff"]["ci_95"]
    if clean_ci[0] > 0.0:
        primary_verdict = "IMPROVED_WITHOUT_DEGRADATION"
    elif clean_ci[1] < 0.0:
        primary_verdict = "DEGRADED"
    else:
        primary_verdict = "NO_CLEAR_DIFFERENCE"

    total_illegal = sum(g["illegal_actions"] for arm in ARM_NAMES for ctx in CONTEXTS for g in games[arm][ctx])
    total_crashes = sum(1 for arm in ARM_NAMES for ctx in CONTEXTS for g in games[arm][ctx] if g["crashed"])

    asu_modules_loaded = common.loaded_asu_modules()
    payload = {
        "git_head_sha": git_head_sha, "elapsed_s": elapsed_s, "peak_rss_gib": rss.peak_gib,
        "asu_modules_loaded": asu_modules_loaded, "seeds": seeds,
        "hash_verification": hash_report,
        "summaries": summaries,
        "paired_buy_simple_minus_pure_a96": paired,
        "primary_clean_context_verdict": primary_verdict,
        "total_illegal_actions": total_illegal, "total_crashes": total_crashes,
        "buy_safety_arm": "NOT_RUN - dropped per user decision (ASU runtime-fallback rule conflict), see docs/DIAGNOSTIC_BUY_PROPERTY_INTERVENTION_A96.md",
        "games": games,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded: {asu_modules_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
