"""Experiment 035 (pre-registered): EVALUATION-ONLY ASU robustness check for
the current PURE PPO champion (`candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt`,
promoted in `034`) - does it hold up against ASU_FROZEN_TEACHER opponents,
never seen during training?

Per `CLAUDE.md`'s ASU-restrictions section: ASU sits ONLY as a fixed
evaluation opponent here. Zero training, zero optimizer/backprop, zero
gradient steps. No ASU action/value/rollout output is ever written to a
replay buffer or used as a label - `play_local_game`/`screen.summarize`
below only ever read win/loss, net worth, and legal-action-rate counters
off the finished game, exactly like every other paired-evaluation screen in
this project.

Design (see `docs/EXPERIMENTS.md`'s `035` entry for the full pre-
registration and the ASURolloutV1 cost-calibration finding this design
responds to):
  - One CHAMPION focus seat (the frozen 96-game A_lr1e-4 checkpoint, hash-
    gated on both checkpoint and full-actor SHA-256 before any game) +
    three copies of ONE ASU_FROZEN_TEACHER policy per game, rotated across
    all 4 physical seats.
  - Champion action selection: the SAME deterministic legal-masked argmax
    `monopolyzero_pure_ppo_strength_screen.build_masked_argmax_policy` uses
    for every other champion-gate script in this project - reused
    unmodified, not reimplemented here.
  - ASU action selection: `ASU_FROZEN_TEACHER.{ASUValueV1,ASURolloutV1}`'s
    own `choose_action(env)`, called directly (frozen, deterministic, no
    sampling) through a small local adapter (`_ASUOpponentPolicy` below)
    that only exists to (a) satisfy `monopolyzero_common.play_local_game`'s
    per-seat `.choose(game, seat, decision_seed)` contract and (b) record
    the same BUY_PROPERTY/ACCEPT_TRADE/DECLINE_TRADE opportunity/chosen
    counters `build_masked_argmax_policy` records for the champion seat, so
    both sides are measured identically. No fallback/substitution layer -
    an ASU decision that somehow isn't legal is a hard integrity failure
    (`play_local_game` raises), never silently patched.
  - Both `references/DeepRL_Monopoly`'s pinned SHA and
    `ASU_FROZEN_TEACHER.spec.FROZEN_SPEC_HASH` are checked against the
    values this project's own prior ASU benchmark (`006`) already recorded,
    before any game - a mismatch is a hard STOP, since this run's entire
    purpose is measuring against the SAME frozen ASU spec `006` measured
    zero-illegal-action, zero-crash behavior for.
  - `ASURolloutV1` (family A) was measured, before this script's real run,
    to cost roughly 2-3 hours/game (up to ~18.5s for a single decision;
    see `docs/EXPERIMENTS.md`'s `035` entry for the full calibration) -
    ~40-50h for the pre-registered 16-game family, which the user
    explicitly chose to skip rather than run. This script still supports
    `--asu-policy asu-rollout-v1` (the infra is identical either way) so a
    future, explicitly-budgeted run can use it, but the registered `035`
    result covers `asu-value-v1` only.
"""

from __future__ import annotations

import argparse
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
from monopolyzero_pure_ppo_learnability_gate import _full_actor_sha256  # noqa: E402

DEV_SEED_BASE = 53000
N_SEEDS = 4
NUM_SEATS = screen.NUM_SEATS
MAX_ROUNDS = screen.MAX_ROUNDS

ARTIFACT_DIR = common.REPO_ROOT / "artifacts" / "monopolyzero_pure_ppo_learnability_gate"
CHAMPION_CHECKPOINT = ARTIFACT_DIR / "candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt"
EXPECTED_CHAMPION_CHECKPOINT_SHA256 = "78585ed4e2400d024633ee2878d8b88243f7f4a9f498d9019a73e44b3a830f51"
EXPECTED_CHAMPION_ACTOR_SHA256 = "2bd1e9bad3d6e0100e033507f72f15b5088d8f8fb2f04dba1ae9d759b5a34a40"

# Recorded by 006 (logs/experiments/006-asu-evaluation-only-benchmark.json) -
# this run must be measuring the exact same frozen ASU spec, on the same
# pinned reference submodule SHA, or it stops rather than silently drifting.
EXPECTED_REFERENCE_SUBMODULE_SHA = "afd9205761317e196d77f679921c35fb04c7ab96"
EXPECTED_ASU_FROZEN_SPEC_HASH = "9ab1907e0de1af4b253ce36ffa107c4fdb5e2b913858ef87c362423e61a6fd74"

ASU_POLICY_IDS = ("asu-value-v1", "asu-rollout-v1")


def load_and_verify_champion(path: Path):
    from monopoly_game_engine.agent_ppo import PPOAgent

    if not path.is_file():
        raise SystemExit(f"STOP: champion checkpoint missing at {path}")
    checkpoint_sha256 = screen._file_sha256(path)
    if checkpoint_sha256 != EXPECTED_CHAMPION_CHECKPOINT_SHA256:
        raise SystemExit(
            f"STOP: {path.name} checkpoint sha256 mismatch - got {checkpoint_sha256}, "
            f"expected {EXPECTED_CHAMPION_CHECKPOINT_SHA256}. Refusing to reconstruct/retrain."
        )
    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.load(str(path))
    agent.actor.eval()
    if agent.hybrid is not False or bool(agent.fixed_action_mask.any()):
        raise RuntimeError(f"{path.name}: not a pure hybrid=False PPOAgent - refusing to run")
    actor_sha256 = _full_actor_sha256(agent.actor)
    if actor_sha256 != EXPECTED_CHAMPION_ACTOR_SHA256:
        raise SystemExit(
            f"STOP: {path.name} actor sha256 mismatch - got {actor_sha256}, "
            f"expected {EXPECTED_CHAMPION_ACTOR_SHA256}. Refusing to reconstruct/retrain."
        )
    return agent, {"checkpoint_sha256": checkpoint_sha256, "actor_sha256": actor_sha256}


def verify_asu_frozen_spec() -> str:
    from ASU_FROZEN_TEACHER.spec import FROZEN_SPEC_HASH

    if FROZEN_SPEC_HASH != EXPECTED_ASU_FROZEN_SPEC_HASH:
        raise SystemExit(
            f"STOP: ASU_FROZEN_TEACHER.spec.FROZEN_SPEC_HASH mismatch - got {FROZEN_SPEC_HASH}, "
            f"expected {EXPECTED_ASU_FROZEN_SPEC_HASH} (006's recorded value). "
            "The pinned reference submodule may have drifted - refusing to run."
        )
    return FROZEN_SPEC_HASH


def resolve_asu_class(policy_id: str):
    from ASU_FROZEN_TEACHER import ASURolloutV1, ASUValueV1

    if policy_id == "asu-value-v1":
        return ASUValueV1
    if policy_id == "asu-rollout-v1":
        return ASURolloutV1
    raise ValueError(f"Unknown ASU policy id {policy_id!r}; use one of {ASU_POLICY_IDS}")


class _ASUOpponentPolicy:
    """Adapts a frozen `ASU_FROZEN_TEACHER` teacher's `choose_action(env)`
    into `monopolyzero_common.play_local_game`'s per-seat `.choose(game,
    seat, decision_seed)` contract, recording the SAME opportunity/chosen
    counters `build_masked_argmax_policy` records for the champion seat.
    `decision_seed` is accepted but unused - ASU's own `decide()` is
    internally deterministic (dice-seed enumeration, fixed rollout seeds),
    not driven by an externally injected seed, exactly like
    `ASU_FROZEN_TEACHER.evaluate.py`'s own agent adapters."""

    kind = "asu"

    def __init__(self, asu_class, seat: int, counters: dict, env_holder: list):
        self.agent = asu_class(seat)
        self.seat = seat
        self.counters = counters
        self.env_holder = env_holder

    def choose(self, game, seat: int, decision_seed: int) -> int:
        from monopoly_bench.engine import action_family, unwrap
        from monopoly_game_engine.actions import ActionType

        env = unwrap(game)
        self.env_holder[0] = env
        legal = tuple(env.get_allowed_actions(seat))
        chosen_action = int(self.agent.choose_action(env))

        buy_id = int(ActionType.BUY_PROPERTY)
        accept_id = int(ActionType.ACCEPT_TRADE)
        decline_id = int(ActionType.DECLINE_TRADE)
        trade_offer_families = ("buy_trade", "sell_trade", "exch_trade")

        self.counters["decisions"] += 1
        if buy_id in legal:
            self.counters["buy_property_opportunities"] += 1
            self.counters["buy_property_chosen"] += int(chosen_action == buy_id)
        if accept_id in legal:
            self.counters["accept_trade_opportunities"] += 1
            self.counters["accept_trade_chosen"] += int(chosen_action == accept_id)
        if decline_id in legal:
            self.counters["decline_trade_opportunities"] += 1
            self.counters["decline_trade_chosen"] += int(chosen_action == decline_id)
        if action_family(chosen_action) in trade_offer_families:
            self.counters["trade_offer_chosen"] += 1

        return chosen_action


def play_one_game_vs_asu(
    *, game_id: int, seed: int, champion_actor, asu_class, focus_seat: int, device, max_rounds: int,
) -> dict:
    """Same return schema as `screen.play_one_game`, so `screen.summarize`
    can be reused unmodified for the aggregate statistics."""
    counters_by_seat = {seat: screen._new_counters() for seat in range(NUM_SEATS)}
    env_holder: list = [None]
    policies = {}
    for seat in range(NUM_SEATS):
        if seat == focus_seat:
            policies[seat] = screen.build_masked_argmax_policy(
                champion_actor, device, counters_by_seat[seat], env_holder,
            )
        else:
            policies[seat] = _ASUOpponentPolicy(asu_class, seat, counters_by_seat[seat], env_holder)

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


def per_seat_win_breakdown(games: list[dict]) -> dict[str, dict]:
    """Champion win rate broken down by which PHYSICAL seat it occupied -
    diagnostic-only positional-bias check, not the primary statistic."""
    breakdown = {str(seat): {"games": 0, "wins": 0} for seat in range(NUM_SEATS)}
    for game in games:
        seat = game["focus_seat"]
        breakdown[str(seat)]["games"] += 1
        breakdown[str(seat)]["wins"] += int(game["per_seat"][seat]["win"])
    for stats in breakdown.values():
        stats["win_rate"] = (stats["wins"] / stats["games"]) if stats["games"] else None
    return breakdown


def run_family(*, seeds: list[int], champion_actor, asu_class, device) -> list[dict]:
    games: list[dict] = []
    game_id = 0
    for seed in seeds:
        for focus_seat in range(NUM_SEATS):
            game_id += 1
            games.append(
                play_one_game_vs_asu(
                    game_id=game_id, seed=seed, champion_actor=champion_actor, asu_class=asu_class,
                    focus_seat=focus_seat, device=device, max_rounds=MAX_ROUNDS,
                )
            )
    return games


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asu-policy", choices=ASU_POLICY_IDS, default="asu-value-v1")
    parser.add_argument("--seed-base", type=int, default=DEV_SEED_BASE)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)

    seeds = screen._seed_range(args.seed_base, args.n_seeds)
    ep.require_seed_scope(seeds, ep.SEED_CLASS_DEV, context="monopolyzero_a96_vs_asu_robustness_eval.py")

    common.ensure_reference_on_path()

    reference_sha = common.runtime_fingerprint()["reference_submodule_sha"]
    if reference_sha != EXPECTED_REFERENCE_SUBMODULE_SHA:
        raise SystemExit(
            f"STOP: references/DeepRL_Monopoly HEAD is {reference_sha}, "
            f"expected {EXPECTED_REFERENCE_SUBMODULE_SHA}. Refusing to run against a drifted reference."
        )
    frozen_spec_hash = verify_asu_frozen_spec()

    champion_agent, champion_hashes = load_and_verify_champion(CHAMPION_CHECKPOINT)
    asu_class = resolve_asu_class(args.asu_policy)

    import torch

    device = torch.device(args.device)

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        games = run_family(seeds=seeds, champion_actor=champion_agent.actor, asu_class=asu_class, device=device)
    elapsed_s = time.perf_counter() - started

    summary = screen.summarize(games)
    seat_breakdown = per_seat_win_breakdown(games)
    asu_modules_loaded = common.loaded_asu_modules()
    if not asu_modules_loaded:
        raise RuntimeError("Expected ASU_FROZEN_TEACHER to be loaded for this ASU-opponent run, but it was not.")

    payload = {
        "status": "OK",
        "git_head_sha": git_head_sha,
        "reference_submodule_sha": reference_sha,
        "asu_frozen_spec_hash": frozen_spec_hash,
        "asu_policy": args.asu_policy,
        "elapsed_s": elapsed_s,
        "peak_rss_gib": rss.peak_gib,
        "seeds": seeds,
        "n_games": len(games),
        "champion_checkpoint_path": str(CHAMPION_CHECKPOINT),
        "champion_hashes": champion_hashes,
        "asu_modules_loaded": asu_modules_loaded,
        "summary": summary,
        "per_seat_win_breakdown": seat_breakdown,
        "games": games,
    }
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
