"""Isolated diagnostic (NOT a champion gate): does A96 (032's frozen
96-game A_lr1e-4 checkpoint) show any evidence of degradation vs. the
80-game champion when both face a STRUCTURALLY DIFFERENT opponent
distribution - the reference's own rule-based FPAgentA/FPAgentB/FPAgentC
(never used as an opponent in 028/030/031/033/034's PPO-vs-PPO designs)?

Minimal reuse, no new evaluation framework: one focus seat via
monopolyzero_pure_ppo_strength_screen.build_masked_argmax_policy
(unmodified) + monopolyzero_common.LocalFixedPolicy driving FPAgentA/B/C
on the three non-focus seats (assigned in ascending player-id order, same
convention as the reference's own train.py::run_episode) through
monopolyzero_common.play_local_game (unmodified). Per-checkpoint summary
statistics reuse screen.summarize() unmodified; the primary paired
statistic reuses evaluation_protocol.pair_records +
paired_seed_block_bootstrap unmodified.

Checkpoints are loaded from the MAIN checkout's gitignored artifacts
directory by absolute path (this worktree has no artifacts/ of its own) -
never copied, rebuilt, or regenerated. See
docs/DIAGNOSTIC_A96_VS_80_VS_FIXED_LINEUP.md for the full pre-registered
design and decision rule, written before this runner is ever executed.
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

# Absolute path into the MAIN checkout's gitignored artifacts directory -
# this worktree's own artifacts/ does not contain these files and never
# will (never copied/rebuilt here). Read-only reference, per this task.
MAIN_CHECKOUT_ARTIFACT_DIR = Path(
    r"C:\Users\Emrullah Soyler\Desktop\Exposure Academy Projects\Second Week\Monoply"
    r"\artifacts\monopolyzero_pure_ppo_learnability_gate"
)

DEV_SEED_BASE, N_SEEDS = 44000, 8

CHECKPOINTS = {
    "A96": ("candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt",
            "78585ed4e2400d024633ee2878d8b88243f7f4a9f498d9019a73e44b3a830f51",
            "2bd1e9bad3d6e0100e033507f72f15b5088d8f8fb2f04dba1ae9d759b5a34a40"),
    "80": ("candidate_ppo_80.pt", "e47c8d4559c1d74cfceffe114fa069a8e7a2845ba60af68be0b0ae4bd37e1dae",
           "7442f99e51619b7f6b53bc662e74d360e0a1f9cf2365b8dcb003d5a0cc3eda11"),
}


def fixed_lineup_seats(focus_seat: int) -> list[int]:
    """Non-focus seats in ascending player-id order - the exact convention
    the reference's own train.py::run_episode uses to assign
    FPAgentA/B/C, so FPAgentA always lands on the lowest-id non-focus
    seat, FPAgentB the next, FPAgentC the last."""
    return [seat for seat in range(screen.NUM_SEATS) if seat != focus_seat]


def play_one_fixed_lineup_game(*, game_id: int, seed: int, focus_actor, focus_seat: int, device, max_rounds: int) -> dict:
    from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC

    counters = screen._new_counters()
    env_holder: list = [None]
    other_seats = fixed_lineup_seats(focus_seat)
    fixed_policies = {seat: common.LocalFixedPolicy(cls) for seat, cls in zip(other_seats, (FPAgentA, FPAgentB, FPAgentC))}
    policies = {focus_seat: screen.build_masked_argmax_policy(focus_actor, device, counters, env_holder)}
    policies.update(fixed_policies)

    outcome = common.play_local_game(game_id=game_id, seed=seed, policies=policies, max_rounds=max_rounds, record_seats=set())

    final_env = env_holder[0]
    per_seat: dict[int, dict] = {}
    for seat in range(screen.NUM_SEATS):
        bankrupt = properties_owned = None
        if outcome.completed and final_env is not None:
            player = final_env.players[seat]
            bankrupt = bool(player.bankrupt)
            properties_owned = len(player.properties)
        seat_counters = counters if seat == focus_seat else screen._new_counters()
        per_seat[seat] = {
            "is_candidate": seat == focus_seat,
            "win": bool(outcome.completed and outcome.winner == seat),
            "net_worth": float(outcome.final_net_worth[seat]) if outcome.completed and outcome.final_net_worth else None,
            "bankrupt": bankrupt,
            "properties_owned": properties_owned,
            **seat_counters,
        }

    return {
        "game_id": game_id, "seed": seed, "focus_seat": focus_seat,
        "completed": outcome.completed, "winner": outcome.winner, "decisions": outcome.decisions,
        "final_round": outcome.final_round, "round_cap_hit": bool(outcome.completed and outcome.final_round >= max_rounds),
        "illegal_actions": outcome.illegal_actions, "crashed": outcome.crashed, "error": outcome.error,
        "per_seat": per_seat,
        "fixed_agent_fallbacks": {policy.name: policy.fallback_count for policy in fixed_policies.values()},
        "focus_latencies_s": list(outcome.search_latencies_s),
    }


def run_checkpoint(name: str, actor, seeds: list[int], device) -> list[dict]:
    games = []
    game_id = 0
    for seed in seeds:
        for focus_seat in range(screen.NUM_SEATS):
            game_id += 1
            games.append(play_one_fixed_lineup_game(
                game_id=game_id, seed=seed, focus_actor=actor, focus_seat=focus_seat,
                device=device, max_rounds=screen.MAX_ROUNDS,
            ))
    return games


def summarize_checkpoint(games: list[dict]) -> dict:
    """Reuses screen.summarize() unmodified for the per-checkpoint
    descriptive statistics (win rate, Wilson CI, net worth, bankruptcy,
    round-cap, BUY/ACCEPT/DECLINE behavior, integrity), then adds the
    diagnostic-only fields screen.summarize() has no concept of: latency
    and fixed-agent fallback counts, both read straight off the raw game
    records without touching the policy or the reused summarizer."""
    summary = screen.summarize(games)
    all_latencies = [latency for game in games for latency in game["focus_latencies_s"]]
    fallback_totals: dict[str, int] = {}
    for game in games:
        for name, count in game["fixed_agent_fallbacks"].items():
            fallback_totals[name] = fallback_totals.get(name, 0) + count
    summary["focus_inference_latency_s_mean"] = (sum(all_latencies) / len(all_latencies)) if all_latencies else None
    summary["focus_inference_latency_s_max"] = max(all_latencies) if all_latencies else None
    summary["fixed_agent_fallback_counts"] = fallback_totals
    summary["fixed_agent_fallback_total"] = sum(fallback_totals.values())
    return summary


def main() -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)
    common.ensure_reference_on_path()

    seeds = screen._seed_range(DEV_SEED_BASE, N_SEEDS)
    ep.require_seed_scope(seeds, ep.SEED_CLASS_DEV, context="diagnostic_a96_vs_80_vs_fpagents.py")

    agents, hash_report = {}, {}
    for key, (filename, ck_sha, actor_sha) in CHECKPOINTS.items():
        agent, verified = load_and_verify(MAIN_CHECKOUT_ARTIFACT_DIR / filename, ck_sha, actor_sha)
        agents[key] = agent
        hash_report[key] = verified

    import torch

    device = torch.device("cpu")
    games_by_checkpoint: dict[str, list[dict]] = {}
    started = time.perf_counter()
    with common.RssMonitor() as rss:
        for name, agent in agents.items():
            games_by_checkpoint[name] = run_checkpoint(name, agent.actor, seeds, device)
    elapsed_s = time.perf_counter() - started

    checkpoint_summaries = {name: summarize_checkpoint(games) for name, games in games_by_checkpoint.items()}

    def _to_records(games: list[dict]) -> list[dict]:
        return [
            {"seed": game["seed"], "seat": game["focus_seat"],
             "win": game["per_seat"][game["focus_seat"]]["win"],
             "net_worth": game["per_seat"][game["focus_seat"]]["net_worth"]}
            for game in games
        ]

    paired = ep.pair_records(
        baseline_records=_to_records(games_by_checkpoint["80"]),
        candidate_records=_to_records(games_by_checkpoint["A96"]),
        expected_seats=4,
    )
    paired_bootstrap = ep.paired_seed_block_bootstrap(paired, n_resamples=2000, bootstrap_seed=0)
    win_rate_diff_ci = paired_bootstrap["win_rate_diff"]["ci_95"]

    total_illegal = sum(g["illegal_actions"] for games in games_by_checkpoint.values() for g in games)
    total_crashes = sum(1 for games in games_by_checkpoint.values() for g in games if g["crashed"])
    integrity_red_flag = bool(total_illegal or total_crashes)
    win_rate_red_flag = bool(win_rate_diff_ci is not None and win_rate_diff_ci[1] < 0.0)
    verdict = "RED_FLAG" if (integrity_red_flag or win_rate_red_flag) else "NO_CLEAR_DEGRADATION"

    asu_modules_loaded = common.loaded_asu_modules()
    payload = {
        "git_head_sha": git_head_sha, "elapsed_s": elapsed_s, "peak_rss_gib": rss.peak_gib,
        "asu_modules_loaded": asu_modules_loaded, "seeds": seeds,
        "hash_verification": hash_report,
        "checkpoint_summaries": checkpoint_summaries,
        "paired_a96_minus_80_win_rate_bootstrap": paired_bootstrap,
        "integrity_red_flag": integrity_red_flag, "win_rate_red_flag": win_rate_red_flag,
        "total_illegal_actions": total_illegal, "total_crashes": total_crashes,
        "verdict": verdict,
        "games": games_by_checkpoint,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded: {asu_modules_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
