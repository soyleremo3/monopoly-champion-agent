"""Direct win-rate sanity screen: experiment 027's final PURE PPO
(hybrid=False) candidate checkpoint vs. its own untrained initialization -
same architecture, same seed-derived construction, no training here.

`027` asked whether 32 games of pure PPO moved BUY_PROPERTY/ACCEPT_TRADE at
all in a legal-action-rank/greedy-choice sense on a fixed diagnostic set,
and KILLed further scaling because ACCEPT_TRADE's rank regressed. That
regression is not, on its own, evidence of weaker play (a rational policy
should decline bad trades), so this script asks a different, more direct
question: did the checkpoint get stronger in real games against its own
starting point?

Design (pre-registered, see docs/EXPERIMENTS.md's 2026-08-12 "028" entry):
  - One CANDIDATE focus seat (027's final checkpoint) + three identical
    copies of the UNTRAINED baseline (027's own pre-training PPOAgent
    initialization, bit-exact reconstructed from the same seed(0)/
    np.random.seed(0)/torch.manual_seed(0) sequence 027's gate script used,
    right before constructing `PPOAgent(player_id=0, hybrid=False,
    device="cpu")`). No scripted fixed agents anywhere in this design - so
    there is no fixed-agent fallback-substitution path to contaminate a run.
  - Both checkpoint's/baseline's full actor state_dict SHA-256 are checked
    against the exact hashes recorded in `027`'s own experiment log BEFORE
    any game is played - a mismatch stops the run rather than silently
    retraining or reconstructing a different network.
  - Every seat's action selection is a single deterministic masked argmax
    over `ActorNetwork.forward`'s log-softmax output (this project's own
    thin wrapper here, NOT `PPOAgent.choose_action`, which samples via
    `torch.multinomial` - deterministic evaluation needs argmax, not
    sampling). Zero fixed BUY/TRADE rule, zero MCTS, zero DDQN, zero ASU.
  - `monopolyzero_common.ensure_reference_on_path()` installs this
    project's actor-relative property-owner state-encoding patch (see
    docs/DECISIONS.md's 2026-08-12 entry) globally, so it is active for
    every seat's `env._get_state(seat)` call, not just seat 0.
  - 20 fresh DEV seeds x 4 focus-seat rotations = 80 games, `max_rounds=200`
    (this project's current reference evaluation horizon).

Primary statistic: a seed-block bootstrap 95% CI (resampled at the SEED
level - the 4 seat-rotation games sharing one seed share a board draw, so
the seed, not the individual game, is the unit of exchangeability) for
`candidate_win_rate - 0.25` (the symmetric 4-player null), via
`evaluation_protocol.seed_block_bootstrap_vs_null`. GO/KILL/INCONCLUSIVE is
computed from that CI using the rule pre-registered in
docs/EXPERIMENTS.md, BEFORE this script is ever run against real results.

Reuses `monopolyzero_common.play_local_game` (this project's own,
independently-expressed game loop - see that module's docstring) for the
per-game turn resolution, illegal-action/crash handling, winner, and
final-round bookkeeping. This script owns only: the deterministic
masked-argmax policy wrapper, per-seat action/opportunity counters, reading
final bankruptcy/property-ownership off the (in-place-mutated) env object
after the game loop returns, and the summary statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluation_protocol as ep  # noqa: E402
import monopolyzero_common as common  # noqa: E402
from monopolyzero_pure_ppo_learnability_gate import _full_actor_sha256  # noqa: E402

DEV_SEED_BASE = 46000
N_SEEDS = 20
NUM_SEATS = 4
MAX_ROUNDS = 200
NULL_WIN_RATE = 0.25

DEFAULT_CANDIDATE_CHECKPOINT = (
    common.REPO_ROOT / "artifacts" / "monopolyzero_pure_ppo_learnability_gate" / "candidate_ppo.pt"
)

# Pinned provenance from experiment 027 (logs/experiments/027-*.json) - a
# mismatch on any of these three is a STOP, not a silent retrain/rebuild.
EXPECTED_CHECKPOINT_SHA256 = "85194f337182c2d698519966cb8c19d3c1e701102b5fb280cd69d4c23ed8a113"
EXPECTED_CANDIDATE_ACTOR_SHA256 = "e6c142d143b5430d02b37cd3be34fa39a7d8d2f282bdb1eda8677a6109861b5b"
EXPECTED_BASELINE_ACTOR_SHA256 = "ce2ffeaf655ef5745ef8f835f3f999a5baa8beac93b5c70ed22d46f4fed1e64c"


# ── pure helpers ─────────────────────────────────────────────────────────


def _seed_range(base: int, count: int) -> list[int]:
    return list(range(base, base + count))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


# ── deterministic masked-argmax policy (no sampling, no fixed rule) ────────


def build_masked_argmax_policy(actor, device, counters: dict | None, env_holder: list | None):
    """One seat's inference policy: legal-masked argmax over `actor`'s own
    `forward` log-softmax - the deterministic-evaluation analogue of
    `ActorNetwork.get_action` (which instead samples via
    `torch.multinomial`, appropriate for training exploration, not
    evaluation). `actor` may be shared across seats/games (stateless
    forward pass only, `.eval()` already called by the caller).

    `counters`, when given, is updated in place with per-seat opportunity/
    chosen counts for BUY_PROPERTY/ACCEPT_TRADE/DECLINE_TRADE and any
    trade-offer-family action (`monopoly_bench.engine.action_family`, so no
    action-space magic numbers are duplicated here).

    `env_holder`, when given, is a shared single-element list all seats in
    one game write into on every call - since `SharedGame.env` is mutated
    in place for the life of one game (never reassigned), whichever seat's
    wrapper last ran still holds a reference to the SAME env object after
    the whole game finishes, letting the caller read final
    bankruptcy/property-ownership without `play_local_game` needing to
    expose the env object itself.
    """
    import torch

    from monopoly_bench.engine import action_family
    from monopoly_game_engine.actions import ACTION_SPACE_SIZE, ActionType

    buy_id = int(ActionType.BUY_PROPERTY)
    accept_id = int(ActionType.ACCEPT_TRADE)
    decline_id = int(ActionType.DECLINE_TRADE)
    trade_offer_families = ("buy_trade", "sell_trade", "exch_trade")

    class _MaskedArgmaxPolicy:
        kind = "policy_only"

        def choose(self, game, seat: int, decision_seed: int):
            started = time.perf_counter()
            env = game.env
            if env_holder is not None:
                env_holder[0] = env

            legal = tuple(env.get_allowed_actions(seat))
            state = env._get_state(seat)
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool, device=device)
            mask_t[0, list(legal)] = True
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

    return _MaskedArgmaxPolicy()


def _new_counters() -> dict:
    return {
        "decisions": 0,
        "buy_property_opportunities": 0, "buy_property_chosen": 0,
        "accept_trade_opportunities": 0, "accept_trade_chosen": 0,
        "decline_trade_opportunities": 0, "decline_trade_chosen": 0,
        "trade_offer_chosen": 0,
    }


# ── one game ─────────────────────────────────────────────────────────────


def play_one_game(
    *, game_id: int, seed: int, candidate_actor, baseline_actor, focus_seat: int, device, max_rounds: int,
) -> dict:
    counters_by_seat = {seat: _new_counters() for seat in range(NUM_SEATS)}
    env_holder: list = [None]
    policies = {
        seat: build_masked_argmax_policy(
            candidate_actor if seat == focus_seat else baseline_actor,
            device, counters_by_seat[seat], env_holder,
        )
        for seat in range(NUM_SEATS)
    }

    outcome = common.play_local_game(
        game_id=game_id, seed=seed, policies=policies, max_rounds=max_rounds, record_seats=set(),
    )

    final_env = env_holder[0]
    per_seat: dict[int, dict] = {}
    for seat in range(NUM_SEATS):
        bankrupt = None
        properties_owned = None
        if outcome.completed and final_env is not None:
            player = final_env.players[seat]
            bankrupt = bool(player.bankrupt)
            properties_owned = len(player.properties)
        per_seat[seat] = {
            "is_candidate": seat == focus_seat,
            "win": bool(outcome.completed and outcome.winner == seat),
            "net_worth": (
                float(outcome.final_net_worth[seat])
                if outcome.completed and outcome.final_net_worth else None
            ),
            "bankrupt": bankrupt,
            "properties_owned": properties_owned,
            **counters_by_seat[seat],
        }

    return {
        "game_id": game_id,
        "seed": seed,
        "focus_seat": focus_seat,
        "completed": outcome.completed,
        "winner": outcome.winner,
        "decisions": outcome.decisions,
        "final_round": outcome.final_round,
        "round_cap_hit": bool(outcome.completed and outcome.final_round >= max_rounds),
        "illegal_actions": outcome.illegal_actions,
        "crashed": outcome.crashed,
        "error": outcome.error,
        "per_seat": per_seat,
    }


# ── candidate/baseline actor construction + hash gates ─────────────────────


def load_candidate_agent(checkpoint_path: Path):
    """Loads 027's final checkpoint into a fresh PPOAgent(player_id=0,
    hybrid=False) and REQUIRES its full actor hash to match 027's own
    logged value before returning - never silently proceeds on a mismatch.
    """
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.load(str(checkpoint_path))
    agent.actor.eval()
    if agent.hybrid is not False or bool(agent.fixed_action_mask.any()):
        raise RuntimeError("candidate PPOAgent is not pure hybrid=False - refusing to run")

    actual = _full_actor_sha256(agent.actor)
    if actual != EXPECTED_CANDIDATE_ACTOR_SHA256:
        raise SystemExit(
            "STOP: candidate actor full_actor_sha256 mismatch - "
            f"got {actual}, expected {EXPECTED_CANDIDATE_ACTOR_SHA256} (027's logged value). "
            "Refusing to proceed on an unverified checkpoint."
        )
    return agent


def build_untrained_baseline_agent():
    """Bit-exact reconstruction of 027's own untrained PPOAgent(hybrid=False)
    initialization: the identical `random.seed(0)` / `np.random.seed(0)` /
    `torch.manual_seed(0)` sequence 027's gate script (`main()`) ran
    immediately before constructing `PPOAgent(player_id=0, hybrid=False,
    device="cpu")`, with nothing else run in between that could consume
    the RNG streams differently. REQUIRES the resulting full actor hash to
    match 027's own logged pre-training value before returning."""
    import random

    import numpy as np
    import torch

    from monopoly_game_engine.agent_ppo import PPOAgent

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.actor.eval()
    if agent.hybrid is not False or bool(agent.fixed_action_mask.any()):
        raise RuntimeError("baseline PPOAgent is not pure hybrid=False - refusing to run")

    actual = _full_actor_sha256(agent.actor)
    if actual != EXPECTED_BASELINE_ACTOR_SHA256:
        raise SystemExit(
            "STOP: reconstructed baseline actor full_actor_sha256 mismatch - "
            f"got {actual}, expected {EXPECTED_BASELINE_ACTOR_SHA256} (027's logged pre-training value). "
            "Refusing to proceed on an unverified baseline reconstruction."
        )
    return agent


# ── orchestration ────────────────────────────────────────────────────────


def run_screen(*, seeds: list[int], checkpoint_path: Path, device_name: str) -> dict:
    import torch

    device = torch.device(device_name)

    candidate_agent = load_candidate_agent(checkpoint_path)
    baseline_agent = build_untrained_baseline_agent()

    games: list[dict] = []
    game_id = 0
    for seed in seeds:
        for focus_seat in range(NUM_SEATS):
            game_id += 1
            games.append(
                play_one_game(
                    game_id=game_id, seed=seed,
                    candidate_actor=candidate_agent.actor, baseline_actor=baseline_agent.actor,
                    focus_seat=focus_seat, device=device, max_rounds=MAX_ROUNDS,
                )
            )

    return {
        "candidate_actor_sha256": _full_actor_sha256(candidate_agent.actor),
        "baseline_actor_sha256": _full_actor_sha256(baseline_agent.actor),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "games": games,
    }


def _action_rate_summary(bucket: dict) -> dict:
    def _rate(chosen: int, opportunities: int) -> float | None:
        return (chosen / opportunities) if opportunities else None

    return {
        "buy_property_opportunities": bucket["buy_opp"],
        "buy_property_rate_when_legal": _rate(bucket["buy_chosen"], bucket["buy_opp"]),
        "accept_trade_opportunities": bucket["accept_opp"],
        "accept_trade_rate_when_legal": _rate(bucket["accept_chosen"], bucket["accept_opp"]),
        "decline_trade_opportunities": bucket["decline_opp"],
        "decline_trade_rate_when_legal": _rate(bucket["decline_chosen"], bucket["decline_opp"]),
        "trade_offers_total": bucket["trade_offer"],
    }


def summarize(games: list[dict]) -> dict:
    """Aggregates per-game records into the pre-registered primary
    statistic (seed-block bootstrap CI for candidate_win_rate - 0.25) plus
    secondary descriptive stats. Raises loudly on any illegal action or
    crash anywhere in the screen rather than silently excluding
    contaminated games from the statistics - this evaluation is supposed
    to have zero of both by construction (legal-masked argmax can only
    select a legal action; a genuine crash would mean something is broken,
    not something to average around)."""
    total_illegal = sum(game["illegal_actions"] for game in games)
    total_crashes = sum(1 for game in games if game["crashed"])
    if total_illegal or total_crashes:
        raise RuntimeError(
            f"Integrity failure: illegal_actions={total_illegal}, crashes={total_crashes} "
            "across the screen - refusing to compute strength statistics on contaminated games."
        )

    candidate_records = []
    candidate_net_worths: list[float] = []
    baseline_net_worths: list[float] = []
    candidate_bankrupt = 0
    baseline_bankrupt = 0
    baseline_seat_games = 0
    round_cap_hits = 0
    buckets = {
        "candidate": {"buy_opp": 0, "buy_chosen": 0, "accept_opp": 0, "accept_chosen": 0,
                      "decline_opp": 0, "decline_chosen": 0, "trade_offer": 0},
        "baseline": {"buy_opp": 0, "buy_chosen": 0, "accept_opp": 0, "accept_chosen": 0,
                     "decline_opp": 0, "decline_chosen": 0, "trade_offer": 0},
    }

    for game in games:
        focus_seat = game["focus_seat"]
        if game["round_cap_hit"]:
            round_cap_hits += 1
        for seat, stats in game["per_seat"].items():
            bucket_name = "candidate" if seat == focus_seat else "baseline"
            bucket = buckets[bucket_name]
            bucket["buy_opp"] += stats["buy_property_opportunities"]
            bucket["buy_chosen"] += stats["buy_property_chosen"]
            bucket["accept_opp"] += stats["accept_trade_opportunities"]
            bucket["accept_chosen"] += stats["accept_trade_chosen"]
            bucket["decline_opp"] += stats["decline_trade_opportunities"]
            bucket["decline_chosen"] += stats["decline_trade_chosen"]
            bucket["trade_offer"] += stats["trade_offer_chosen"]
            if bucket_name == "candidate":
                candidate_net_worths.append(stats["net_worth"])
                candidate_bankrupt += int(bool(stats["bankrupt"]))
            else:
                baseline_net_worths.append(stats["net_worth"])
                baseline_bankrupt += int(bool(stats["bankrupt"]))
                baseline_seat_games += 1
        candidate_records.append({"seed": game["seed"], "win": game["per_seat"][focus_seat]["win"]})

    bootstrap = ep.seed_block_bootstrap_vs_null(
        candidate_records, null_rate=NULL_WIN_RATE, n_resamples=2000, bootstrap_seed=0,
    )
    ci = bootstrap["win_rate_diff_from_null"]["ci_95"]
    if ci is None:
        verdict = "INCONCLUSIVE"
    else:
        lower, upper = ci
        if lower > 0.0:
            verdict = "GO"
        elif upper <= 0.0:
            verdict = "KILL"
        else:
            verdict = "INCONCLUSIVE"

    n_games = len(games)
    candidate_wins = sum(1 for record in candidate_records if record["win"])
    candidate_mean_nw = _mean(candidate_net_worths)
    baseline_mean_nw = _mean(baseline_net_worths)

    return {
        "n_games": n_games,
        "n_seed_blocks": bootstrap["n_seed_blocks"],
        "candidate_wins": candidate_wins,
        "candidate_win_rate": candidate_wins / n_games,
        "null_win_rate": NULL_WIN_RATE,
        "candidate_wilson_95": list(ep.wilson_95_interval(candidate_wins, n_games)),
        "primary_seed_block_bootstrap_vs_25pct_null": bootstrap,
        "verdict": verdict,
        "round_cap_hits": round_cap_hits,
        "round_cap_rate": round_cap_hits / n_games,
        "candidate_mean_net_worth": candidate_mean_nw,
        "baseline_mean_net_worth": baseline_mean_nw,
        "candidate_net_worth_advantage": (
            (candidate_mean_nw - baseline_mean_nw)
            if candidate_mean_nw is not None and baseline_mean_nw is not None else None
        ),
        "candidate_bankruptcy_rate": candidate_bankrupt / n_games,
        "baseline_bankruptcy_rate": (baseline_bankrupt / baseline_seat_games) if baseline_seat_games else None,
        "candidate_trade_offers_per_game": buckets["candidate"]["trade_offer"] / n_games,
        "baseline_trade_offers_per_game": (
            buckets["baseline"]["trade_offer"] / baseline_seat_games if baseline_seat_games else None
        ),
        "candidate_action_behavior": _action_rate_summary(buckets["candidate"]),
        "baseline_action_behavior": _action_rate_summary(buckets["baseline"]),
        "integrity": {"illegal_actions": total_illegal, "crashes": total_crashes},
    }


# ── CLI ──────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-base", type=int, default=DEV_SEED_BASE)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CANDIDATE_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)

    seeds = _seed_range(args.seed_base, args.n_seeds)
    ep.require_seed_scope(seeds, ep.SEED_CLASS_DEV, context="monopolyzero_pure_ppo_strength_screen.py")

    common.ensure_reference_on_path()

    if not args.checkpoint.is_file():
        raise SystemExit(f"STOP: candidate checkpoint not found at {args.checkpoint}")
    actual_checkpoint_sha256 = _file_sha256(args.checkpoint)
    if actual_checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit(
            "STOP: candidate checkpoint SHA-256 mismatch - "
            f"got {actual_checkpoint_sha256}, expected {EXPECTED_CHECKPOINT_SHA256}. "
            "Refusing to silently retrain/reconstruct."
        )

    def _emit(payload: dict) -> None:
        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
        print(text)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        run_result = run_screen(seeds=seeds, checkpoint_path=args.checkpoint, device_name=args.device)
    elapsed_s = time.perf_counter() - started

    summary = summarize(run_result["games"])
    asu_modules_loaded = common.loaded_asu_modules()

    payload = {
        "status": "OK",
        "git_head_sha": git_head_sha,
        "elapsed_s": elapsed_s,
        "peak_rss_gib": rss.peak_gib,
        "seeds": seeds,
        "n_games": len(run_result["games"]),
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": run_result["checkpoint_sha256"],
        "candidate_actor_sha256": run_result["candidate_actor_sha256"],
        "baseline_actor_sha256": run_result["baseline_actor_sha256"],
        "asu_modules_loaded": asu_modules_loaded,
        "summary": summary,
        "games": run_result["games"],
    }
    _emit(payload)
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded during screen run: {asu_modules_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
