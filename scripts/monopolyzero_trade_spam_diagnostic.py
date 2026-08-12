"""Causal diagnostic for 028's extreme trade-offer spam (~330/game). FROZEN
algorithm: no PPOAgent/ActorNetwork/reward/state/action-space/env/MCTS
changes here - this only compares two INFERENCE-TIME masks over 027's own
frozen checkpoint. Arm A = 028's exact policy. Arm B = same checkpoint,
same deterministic policy, with only the candidate seat's outgoing
trade-offer families (buy_trade/sell_trade/exch_trade) removed from its own
legal mask (ACCEPT_TRADE/DECLINE_TRADE/BUY_PROPERTY untouched). Diagnostic
only - does not authorize shipping the masked policy. See
docs/EXPERIMENTS.md's 029 entry for the pre-registered decision rule.
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

# Reused DEV seeds (first 12 of 028's already-registered 46000-46019 block -
# DEV pool is documented as freely reusable; no evaluation_protocol.py edit).
SEEDS = screen._seed_range(46000, 12)
TRADE_OFFER_FAMILIES = ("buy_trade", "sell_trade", "exch_trade")


def run_arm(*, candidate_agent, baseline_agent, exclude_families, device):
    games = []
    for game_id, (seed, focus_seat) in enumerate(
        ((s, seat) for s in SEEDS for seat in range(screen.NUM_SEATS)), start=1
    ):
        games.append(screen.play_one_game(
            game_id=game_id, seed=seed, candidate_actor=candidate_agent.actor,
            baseline_actor=baseline_agent.actor, focus_seat=focus_seat, device=device,
            max_rounds=screen.MAX_ROUNDS, candidate_exclude_families=exclude_families,
        ))
    return games


def _records(games):
    return [
        {"seed": g["seed"], "seat": g["focus_seat"], "win": g["per_seat"][g["focus_seat"]]["win"],
         "net_worth": g["per_seat"][g["focus_seat"]]["net_worth"]}
        for g in games
    ]


def main() -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)
    ep.require_seed_scope(SEEDS, ep.SEED_CLASS_DEV, context="monopolyzero_trade_spam_diagnostic.py")
    common.ensure_reference_on_path()
    import torch

    device = torch.device("cpu")
    candidate_agent = screen.load_candidate_agent(screen.DEFAULT_CANDIDATE_CHECKPOINT)
    baseline_agent = screen.build_untrained_baseline_agent()

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        games_a = run_arm(candidate_agent=candidate_agent, baseline_agent=baseline_agent, exclude_families=(), device=device)
        games_b = run_arm(candidate_agent=candidate_agent, baseline_agent=baseline_agent, exclude_families=TRADE_OFFER_FAMILIES, device=device)
    elapsed_s = time.perf_counter() - started

    summary_a = screen.summarize(games_a)
    summary_b = screen.summarize(games_b)
    paired = ep.paired_evaluation_summary(
        baseline_records=_records(games_a), candidate_records=_records(games_b),
        baseline_fallbacks=0, candidate_fallbacks=0,
    )
    ci = paired["primary"]["seed_block_bootstrap"]["win_rate_diff"]["ci_95"]
    spam_reduced = summary_b["candidate_trade_offers_per_game"] < summary_a["candidate_trade_offers_per_game"]
    if spam_reduced and ci[0] > 0.0:
        verdict = "GO"
    elif ci[1] <= 0.0:
        verdict = "KILL"
    else:
        verdict = "INCONCLUSIVE"

    payload = {
        "status": "OK", "git_head_sha": git_head_sha, "elapsed_s": elapsed_s, "peak_rss_gib": rss.peak_gib,
        "seeds": SEEDS, "n_games_per_arm": len(games_a),
        "candidate_actor_sha256": screen._full_actor_sha256(candidate_agent.actor),
        "baseline_actor_sha256": screen._full_actor_sha256(baseline_agent.actor),
        "asu_modules_loaded": common.loaded_asu_modules(),
        "arm_a_original": summary_a, "arm_b_no_trade_offer": summary_b,
        "trade_spam_reduced": spam_reduced, "paired_evaluation": paired, "verdict": verdict,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if payload["asu_modules_loaded"]:
        raise RuntimeError(f"ASU modules loaded: {payload['asu_modules_loaded']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
