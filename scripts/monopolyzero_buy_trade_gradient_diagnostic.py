"""Read-only diagnostic: does this project's own ASU-independent training
step actually expose BUY_PROPERTY/ACCEPT_TRADE/DECLINE_TRADE to gradient
signal under real MaxNPUCT self-play - or does PUCT's prior-proportional
exploration term (see search.py's `_select`) starve them of MCTS visits
regardless of the `mandatory`-widening guarantee (engine.py's
`action_family`/`progressive_actions`)?

Zero ASU, zero HYBRID_COMPAT, zero fixed-rule action selection anywhere -
every seat is `monopolyzero_common.build_local_search_policy`-shaped pure
MaxNPUCT self-play (self_play=True, same as real self-play generation would
use). No weight update, no optimizer step, no autograd pass anywhere in the
main diagnostic path (the gradient-signal number below is the closed-form
softmax cross-entropy identity `predicted_prob - target_share`, verified
against a real gradient computation in this script's own test suite, not
assumed).

Reuses the DEV seeds already spent by experiment 023 (43000-43019, via
`monopolyzero_hybrid_bootstrap_isolation_audit.SEEDS`) - consumes no new
scored DEV seed range.

Per opportunity, records: raw policy prior, prior rank among legal actions,
MCTS visit count, visit share (the exact policy-target this project's own
training-update step or the reference's `train_step` would compute for that
logit), the action root search actually chose, the plain-greedy POLICY_ONLY
action on the identical state, a BUY_PROPERTY-vs-best-alternative
comparison, an ACCEPT_TRADE-vs-DECLINE_TRADE comparison, and the
gradient-signal sign/magnitude for each side.

Cost warning, not yet resolved: full MaxNPUCT self-play (SearchConfig()
defaults, 128 simulations/decision) is far more expensive per decision than
the POLICY_ONLY/HYBRID_COMPAT arms 023/024 used (zero search there). A
1-seed smoke test with a small SearchConfig(simulations=4, max_depth=16)
and max_rounds=5 completes in seconds (see
tests/test_monopolyzero_buy_trade_gradient_diagnostic.py). With no flags,
`main()` reproduces the exact original hardcoded config (SearchConfig() +
MAX_ROUNDS=200, all 20 reused seeds) - NOT benchmarked, should not be run
as one long invocation without first measuring cost. `--seed-start`/
`--seed-count`/`--simulations`/`--max-rounds` (all optional) exist
specifically to run a controlled small benchmark first, e.g.:

    PYTHONHASHSEED=0 python scripts/monopolyzero_buy_trade_gradient_diagnostic.py \
      --seed-start 43000 --seed-count 1 --simulations 128 --max-rounds 20

`--output PATH` additionally writes the same structured JSON payload to
PATH (stdout is always printed regardless). A custom `--seed-start`/
`--seed-count` range still goes through the same DEV-pool-only guard
(`evaluation_protocol.require_seed_scope`) as the default 43000-43019
range - no seed outside a registered DEV/PROMOTION/FINAL_BLIND range is
ever accepted, and this diagnostic only ever requests DEV.

Produces no automatic conclusion about which fix (if any) is needed - see
this session's PLAN. Built on scripts/monopolyzero_common.py and
scripts/evaluation_protocol.py; no monopoly_bench.adapters/.arena/.training
import, no ASU. Refuses to run unless PYTHONHASHSEED=0 is set, the git tree
is clean, the baseline checkpoint SHA-256 matches, and every self-play seed
is registered in the DEV pool (`evaluation_protocol.require_seed_scope`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluation_protocol as ep  # noqa: E402
import monopolyzero_common as common  # noqa: E402
import monopolyzero_hybrid_bootstrap_isolation_audit as audit_v1  # noqa: E402

SEEDS = audit_v1.SEEDS  # reused 43000-43019 (023) - no new DEV seeds consumed
MAX_ROUNDS = audit_v1.MAX_ROUNDS
NUM_SEATS = audit_v1.NUM_SEATS
CHECKPOINT_PATH = audit_v1.CHECKPOINT_PATH
BASELINE_CHECKPOINT_SHA256 = audit_v1.BASELINE_CHECKPOINT_SHA256
verify_baseline_checkpoint = audit_v1.verify_baseline_checkpoint
_median = audit_v1._median


def _sign(value: float | None) -> str | None:
    if value is None:
        return None
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _prior_rank(action: int, legal: tuple[int, ...], raw_priors: dict[int, float]) -> int:
    ordered = sorted(legal, key=lambda candidate: (-raw_priors[candidate], candidate))
    return ordered.index(action) + 1


def _build_opportunity_record(
    *, seat: int, legal: tuple[int, ...], raw_priors: dict[int, float], result,
    buy_id: int, accept_id: int, decline_id: int,
) -> dict:
    """Builds one per-decision record for a BUY_PROPERTY and/or ACCEPT_TRADE
    opportunity. `result` is a search.py SearchResult-shaped object (visits,
    chosen_action, simulations) from a real MaxNPUCT.choose_action() call on
    this exact state - never mutated, only read."""
    visits: dict[int, int] = result.visits
    total_visits = sum(visits.values())

    def share(action: int) -> float:
        return (visits.get(action, 0) / total_visits) if total_visits else 0.0

    is_buy_opportunity = buy_id in legal
    is_trade_opportunity = accept_id in legal
    policy_only_action = max(raw_priors, key=raw_priors.get)

    record: dict = {
        "seat": seat,
        "is_buy_opportunity": is_buy_opportunity,
        "is_trade_opportunity": is_trade_opportunity,
        "root_chosen_action": result.chosen_action,
        "policy_only_action": policy_only_action,
        "simulations": result.simulations,
        "total_visits": total_visits,
        "forced_single_legal": result.simulations == 0,
    }

    if is_buy_opportunity:
        buy_prior = raw_priors[buy_id]
        buy_share = share(buy_id)
        buy_signal = buy_prior - buy_share
        alt_candidates = tuple(action for action in legal if action != buy_id)
        alternative_action = (
            max(alt_candidates, key=lambda action: (visits.get(action, 0), raw_priors.get(action, 0.0), -action))
            if alt_candidates else None
        )
        record["buy"] = {
            "raw_prior": buy_prior,
            "prior_rank": _prior_rank(buy_id, legal, raw_priors),
            "visit_count": visits.get(buy_id, 0),
            "target_share": buy_share,
            "gradient_signal": buy_signal,
            "gradient_sign": _sign(buy_signal),
            "alternative_action": alternative_action,
            "alternative_visit_count": (visits.get(alternative_action, 0) if alternative_action is not None else None),
            "alternative_target_share": (share(alternative_action) if alternative_action is not None else None),
            "alternative_raw_prior": (raw_priors.get(alternative_action) if alternative_action is not None else None),
        }

    if is_trade_opportunity:
        accept_prior = raw_priors[accept_id]
        accept_share = share(accept_id)
        accept_signal = accept_prior - accept_share
        decline_legal = decline_id in legal
        decline_prior = raw_priors.get(decline_id) if decline_legal else None
        decline_share = share(decline_id) if decline_legal else None
        decline_signal = (decline_prior - decline_share) if decline_legal else None
        record["trade"] = {
            "accept_raw_prior": accept_prior,
            "accept_prior_rank": _prior_rank(accept_id, legal, raw_priors),
            "accept_visit_count": visits.get(accept_id, 0),
            "accept_target_share": accept_share,
            "accept_gradient_signal": accept_signal,
            "accept_gradient_sign": _sign(accept_signal),
            "decline_legal": decline_legal,
            "decline_raw_prior": decline_prior,
            "decline_prior_rank": (_prior_rank(decline_id, legal, raw_priors) if decline_legal else None),
            "decline_visit_count": (visits.get(decline_id, 0) if decline_legal else None),
            "decline_target_share": decline_share,
            "decline_gradient_signal": decline_signal,
            "decline_gradient_sign": _sign(decline_signal),
        }

    return record


def build_gradient_diagnostic_policy(model, search_config):
    """A `kind = "search"` policy (same contract `play_local_game` already
    understands) that delegates every decision to a real
    `monopoly_bench.search.MaxNPUCT` instance, and additionally - only on
    BUY_PROPERTY/ACCEPT_TRADE opportunities, to avoid doubling every other
    decision's cost - runs one extra `model.predict()` call on the identical
    pre-search state to capture the raw (pre-Dirichlet-noise) policy prior,
    appending a record to `self.log`. Build one fresh instance per game (its
    `self.log` is game-scoped, matching how this project's HYBRID_COMPAT
    diagnostic policy scopes its own log per game) - safe to share across
    all 4 seats of that one game, since every record tags its own `seat`."""
    from monopoly_bench.search import MaxNPUCT
    from monopoly_game_engine.actions import ActionType

    buy_id = int(ActionType.BUY_PROPERTY)
    accept_id = int(ActionType.ACCEPT_TRADE)
    decline_id = int(ActionType.DECLINE_TRADE)

    class _GradientDiagnosticPolicy:
        kind = "search"

        def __init__(self):
            self._engine = MaxNPUCT(model, search_config, self_play=True)
            self.log: list[dict] = []
            self.total_searches = 0

        def choose(self, game, seat: int, decision_seed: int):
            self.total_searches += 1
            env = game.env
            legal = tuple(env.get_allowed_actions(seat))
            is_buy_opportunity = buy_id in legal
            is_trade_opportunity = accept_id in legal

            raw_priors = None
            if is_buy_opportunity or is_trade_opportunity:
                state = env._get_state(seat)
                raw_priors, _ = model.predict(state, legal, seat)

            result = self._engine.choose_action(game, seat, decision_seed)

            if raw_priors is not None:
                self.log.append(
                    _build_opportunity_record(
                        seat=seat, legal=legal, raw_priors=raw_priors, result=result,
                        buy_id=buy_id, accept_id=accept_id, decline_id=decline_id,
                    )
                )
            return result

    return _GradientDiagnosticPolicy()


def run_diagnostic(seeds, model, search_config, max_rounds: int) -> dict:
    """Plays one pure self-play game per seed (all 4 seats searched by the
    same diagnostic policy - 1 seed = 1 physical game, all 4 seats' data),
    collecting every BUY/TRADE opportunity record. No training, no weight
    change, no optimizer anywhere in this function."""
    common.ensure_reference_on_path()
    from monopoly_game_engine.actions import ActionType

    buy_id = int(ActionType.BUY_PROPERTY)
    accept_id = int(ActionType.ACCEPT_TRADE)
    decline_id = int(ActionType.DECLINE_TRADE)

    per_game: list[dict] = []
    all_records: list[dict] = []
    latencies: list[float] = []
    total_illegal = 0
    total_crashed = 0
    incomplete = 0

    for seed in seeds:
        policy = build_gradient_diagnostic_policy(model, search_config)
        policies = {seat: policy for seat in range(NUM_SEATS)}
        outcome = common.play_local_game(
            game_id=seed, seed=seed, policies=policies,
            max_rounds=max_rounds, record_seats=set(),
        )
        total_illegal += outcome.illegal_actions
        total_crashed += int(outcome.crashed)
        incomplete += int(not outcome.completed)
        latencies.extend(outcome.search_latencies_s)

        for idx, entry in enumerate(policy.log):
            tagged = dict(entry)
            tagged["seed"] = seed
            tagged["decision_index"] = idx
            all_records.append(tagged)

        per_game.append(
            {
                "seed": seed,
                "completed": outcome.completed,
                "decisions": outcome.decisions,
                "search_calls": policy.total_searches,
                "buy_opportunities": sum(1 for entry in policy.log if entry["is_buy_opportunity"]),
                "trade_opportunities": sum(1 for entry in policy.log if entry["is_trade_opportunity"]),
            }
        )

    return {
        "per_game": per_game,
        "records": all_records,
        "latencies": latencies,
        "total_illegal": total_illegal,
        "total_crashed": total_crashed,
        "incomplete": incomplete,
        "buy_id": buy_id,
        "accept_id": accept_id,
        "decline_id": decline_id,
    }


def buy_opportunity_aggregate(records: list[dict], *, buy_action_id: int) -> dict:
    buy_records = [record for record in records if record["is_buy_opportunity"]]
    n = len(buy_records)
    priors = [record["buy"]["raw_prior"] for record in buy_records]
    ranks = [record["buy"]["prior_rank"] for record in buy_records]
    shares = [record["buy"]["target_share"] for record in buy_records]
    signals = [record["buy"]["gradient_signal"] for record in buy_records]
    visit_zero = sum(1 for record in buy_records if record["buy"]["visit_count"] == 0)
    share_zero = sum(1 for record in buy_records if record["buy"]["target_share"] == 0.0)
    positive = sum(1 for signal in signals if signal > 0)
    negative = sum(1 for signal in signals if signal < 0)
    zero = sum(1 for signal in signals if signal == 0)
    flips_toward = sum(
        1 for record in buy_records
        if record["policy_only_action"] != buy_action_id and record["root_chosen_action"] == buy_action_id
    )
    flips_away = sum(
        1 for record in buy_records
        if record["policy_only_action"] == buy_action_id and record["root_chosen_action"] != buy_action_id
    )
    return {
        "opportunities": n,
        "visit_zero_rate": _rate(visit_zero, n),
        "target_share_zero_rate": _rate(share_zero, n),
        "mean_raw_prior": _mean(priors),
        "median_raw_prior": _median(priors),
        "mean_prior_rank": _mean(ranks),
        "median_prior_rank": _median(ranks),
        "mean_target_share": _mean(shares),
        "median_target_share": _median(shares),
        "mean_gradient_signal": _mean(signals),
        "median_gradient_signal": _median(signals),
        "positive_gradient_rate": _rate(positive, n),
        "negative_gradient_rate": _rate(negative, n),
        "zero_gradient_rate": _rate(zero, n),
        "search_flipped_toward_buy": flips_toward,
        "search_flipped_away_from_buy": flips_away,
        "root_chosen_buy_rate": _rate(
            sum(1 for record in buy_records if record["root_chosen_action"] == buy_action_id), n
        ),
        "policy_only_chose_buy_rate": _rate(
            sum(1 for record in buy_records if record["policy_only_action"] == buy_action_id), n
        ),
    }


def trade_response_aggregate(records: list[dict], *, side: str, action_id: int) -> dict:
    """`side`: "accept" or "decline" - same aggregate shape for both, read
    from the matching `record["trade"][f"{side}_*"]` fields. `decline`
    additionally filters to records where DECLINE_TRADE was actually legal
    (should be virtually always true whenever ACCEPT_TRADE is, but not
    assumed)."""
    if side not in ("accept", "decline"):
        raise ValueError(f"side must be 'accept' or 'decline', got {side!r}")
    trade_records = [record for record in records if record["is_trade_opportunity"]]
    if side == "decline":
        trade_records = [record for record in trade_records if record["trade"]["decline_legal"]]
    n = len(trade_records)
    priors = [record["trade"][f"{side}_raw_prior"] for record in trade_records]
    ranks = [record["trade"][f"{side}_prior_rank"] for record in trade_records]
    visit_counts = [record["trade"][f"{side}_visit_count"] for record in trade_records]
    shares = [record["trade"][f"{side}_target_share"] for record in trade_records]
    signals = [record["trade"][f"{side}_gradient_signal"] for record in trade_records]
    visit_zero = sum(1 for value in visit_counts if value == 0)
    share_zero = sum(1 for value in shares if value == 0.0)
    positive = sum(1 for value in signals if value is not None and value > 0)
    negative = sum(1 for value in signals if value is not None and value < 0)
    zero = sum(1 for value in signals if value is not None and value == 0)
    flips_toward = sum(
        1 for record in trade_records
        if record["policy_only_action"] != action_id and record["root_chosen_action"] == action_id
    )
    flips_away = sum(
        1 for record in trade_records
        if record["policy_only_action"] == action_id and record["root_chosen_action"] != action_id
    )
    return {
        "opportunities": n,
        "visit_zero_rate": _rate(visit_zero, n),
        "target_share_zero_rate": _rate(share_zero, n),
        "mean_raw_prior": _mean(priors),
        "median_raw_prior": _median(priors),
        "mean_prior_rank": _mean(ranks),
        "median_prior_rank": _median(ranks),
        "mean_target_share": _mean(shares),
        "median_target_share": _median(shares),
        "mean_gradient_signal": _mean(signals),
        "median_gradient_signal": _median(signals),
        "positive_gradient_rate": _rate(positive, n),
        "negative_gradient_rate": _rate(negative, n),
        "zero_gradient_rate": _rate(zero, n),
        f"search_flipped_toward_{side}": flips_toward,
        f"search_flipped_away_from_{side}": flips_away,
        "root_chosen_rate": _rate(sum(1 for record in trade_records if record["root_chosen_action"] == action_id), n),
        "policy_only_chose_rate": _rate(
            sum(1 for record in trade_records if record["policy_only_action"] == action_id), n
        ),
    }


def _resolve_seeds(seed_start: int | None, seed_count: int | None) -> list[int]:
    """--seed-start/--seed-count must be given together or not at all;
    omitting both reproduces the exact previous hardcoded seed range."""
    if (seed_start is None) != (seed_count is None):
        raise SystemExit("--seed-start and --seed-count must be given together")
    if seed_start is None:
        return list(SEEDS)
    return list(range(seed_start, seed_start + seed_count))


def build_arg_parser() -> argparse.ArgumentParser:
    """Every flag is optional - omitting all of them reproduces the exact
    previous hardcoded behavior (SEEDS=43000-43019, MAX_ROUNDS=200,
    SearchConfig() defaults, stdout only). Added purely so a controlled
    small run (e.g. 1 seed, low simulation count, short max_rounds) can be
    benchmarked before committing to the full 20-seed/128-simulation/
    200-round invocation - same discipline as scripts/colab_shard_runner.py's
    --benchmark-games-before-shard workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-start", type=int, default=None,
        help="first seed of a custom range; requires --seed-count too. Must be DEV-registered.",
    )
    parser.add_argument(
        "--seed-count", type=int, default=None,
        help="how many consecutive seeds from --seed-start; requires --seed-start too.",
    )
    parser.add_argument(
        "--simulations", type=int, default=None,
        help="MaxNPUCT simulations per decision (default: SearchConfig()'s own default, 128).",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=None,
        help=f"max rounds per self-play game (default: {MAX_ROUNDS}).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="path to also write the raw structured JSON payload to (stdout is always printed regardless).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    seeds = _resolve_seeds(args.seed_start, args.seed_count)
    max_rounds = args.max_rounds if args.max_rounds is not None else MAX_ROUNDS

    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)
    ep.require_seed_scope(seeds, ep.SEED_CLASS_DEV, context="monopolyzero_buy_trade_gradient_diagnostic.py")
    baseline_checkpoint_sha256 = verify_baseline_checkpoint()

    common.ensure_reference_on_path()
    import random

    import numpy as np
    import torch

    from monopoly_bench.config import SearchConfig
    from monopoly_bench.engine import NUM_PLAYERS
    from monopoly_bench.model import MonopolyZeroNet

    if NUM_PLAYERS != NUM_SEATS:
        raise RuntimeError(f"Expected {NUM_SEATS} players, engine reports {NUM_PLAYERS}")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    model = MonopolyZeroNet.load_inference(CHECKPOINT_PATH)
    model.eval()
    search_config = SearchConfig(simulations=args.simulations) if args.simulations is not None else SearchConfig()

    def _emit(payload: dict) -> None:
        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
        print(text)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        result = run_diagnostic(seeds, model, search_config, max_rounds)
        asu_modules_loaded = common.loaded_asu_modules()
    elapsed_s = time.perf_counter() - started

    if result["total_illegal"] or result["total_crashed"] or result["incomplete"]:
        payload = {
            "status": "FAILED_DURING_GAME_GENERATION",
            "git_head_sha": git_head_sha,
            "total_illegal_actions": result["total_illegal"],
            "total_crashed": result["total_crashed"],
            "incomplete_games": result["incomplete"],
        }
        _emit(payload)
        raise RuntimeError(
            "Stopping before any stats: illegal={} crashed={} incomplete={}".format(
                result["total_illegal"], result["total_crashed"], result["incomplete"]
            )
        )

    payload = {
        "status": "OK",
        "git_head_sha": git_head_sha,
        "config": {
            "seeds": list(seeds),
            "seed_count": len(seeds),
            "seeds_reused_from": (
                "023 (43000-43019) - no new DEV seeds consumed"
                if seeds == list(SEEDS) else "custom seed range - validated against the registered DEV pool"
            ),
            "max_rounds": max_rounds,
            "search_config": asdict(search_config),
            "policy": "pure MaxNPUCT self-play, all 4 seats, zero ASU, zero HYBRID_COMPAT, zero fixed rule",
            "physical_games": len(result["per_game"]),
        },
        "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
        "buy_opportunities": buy_opportunity_aggregate(result["records"], buy_action_id=result["buy_id"]),
        "accept_trade_opportunities": trade_response_aggregate(
            result["records"], side="accept", action_id=result["accept_id"]
        ),
        "decline_trade_opportunities": trade_response_aggregate(
            result["records"], side="decline", action_id=result["decline_id"]
        ),
        "per_game": result["per_game"],
        "total_illegal_actions": result["total_illegal"],
        "total_crashed": result["total_crashed"],
        "incomplete_games": result["incomplete"],
        "asu_modules_loaded": asu_modules_loaded,
        "asu_modules_loaded_count": len(asu_modules_loaded),
        "elapsed_s": elapsed_s,
        "peak_rss_gib": rss.peak_gib,
    }
    _emit(payload)
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded during evaluation: {asu_modules_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
