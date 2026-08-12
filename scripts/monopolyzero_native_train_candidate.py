"""Small, ASU-independent MonopolyZero training-candidate runner.

Loads `baseline_pretraining.pt` (the same checkpoint every prior diagnostic
in this project has used - PPO-hybrid actor copy, no further training
applied), plays pure MaxNPUCT self-play (all 4 seats the same model,
`self_play=True`) to collect real MCTS-visit-derived positions, then runs
this project's own native training-update step
(`monopolyzero_common.local_training_update` - an independently-written
equivalent of the reference's `training.py::train_step`, never imports
`monopoly_bench.training` at all, so `ASU_FROZEN_TEACHER` never loads) over
those positions. Mutates the checkpoint's weights in place and saves the
result as a new candidate checkpoint.

Zero ASU, zero HYBRID_COMPAT, zero fixed-rule BUY/TRADE action selection
anywhere - every self-play seat is plain `MaxNPUCT` search
(`monopolyzero_common.build_local_search_policy`), the exact same policy
shape 023/024's self-play arms and the BUY/TRADE gradient diagnostic used.

Before AND after the training update(s), measures the plain-greedy
(argmax, no search - `MonopolyZeroNet.predict`, the same "POLICY_ONLY"
behavior 023 measured) BUY_PROPERTY/ACCEPT_TRADE selection rate over the
exact same collected positions - the direct question this whole line of
diagnostics has been building toward: did this training step actually move
greedy behavior at all, on the states it was trained on?

Persists three things when actually run: (1) the raw stdout printed to the
console (shell-redirect to a `logs/experiments/raw/*.txt` file when
invoking for real, same convention as every prior experiment in this repo -
this script does not write that file itself); (2) a structured JSON payload
(printed to stdout always, additionally written to `--output PATH` if
given); (3) the trained candidate checkpoint's own SHA-256, recorded inside
that JSON payload.

CLI (all flags optional; every default reproduces a small-but-real run over
the 023 seed pool - nothing here is a large/long invocation by default):
`--seed-start`/`--seed-count` (paired, DEV-pool-only via
`evaluation_protocol.require_seed_scope` - same guard for a custom range as
the default one), `--simulations` (default 32), `--max-rounds` (default
200), `--updates` (default 10 native training-update steps over a fresh
replay buffer built from the collected positions), `--output` (structured
JSON path), `--checkpoint-output` (where to save the trained candidate
checkpoint; default `artifacts/monopolyzero_native_train_candidate/candidate.pt`,
gitignored like every other checkpoint in this repo).

No resume, no sharding - out of scope for this first small run (see the
Colab shard runner if that's ever needed for a much larger training pass).
Refuses to run unless PYTHONHASHSEED=0 is set, the git tree is clean, the
baseline checkpoint SHA-256 matches, and every self-play seed is registered
in the DEV pool.
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

DEFAULT_SIMULATIONS = 32
DEFAULT_UPDATES = 10
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

REPLAY_DIR = common.REPO_ROOT / "artifacts" / "monopolyzero_native_train_candidate" / "replay"
DEFAULT_CHECKPOINT_OUTPUT = common.REPO_ROOT / "artifacts" / "monopolyzero_native_train_candidate" / "candidate.pt"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_seeds(seed_start: int | None, seed_count: int | None) -> list[int]:
    """--seed-start/--seed-count must be given together or not at all;
    omitting both reuses the exact 023 seed pool."""
    if (seed_start is None) != (seed_count is None):
        raise SystemExit("--seed-start and --seed-count must be given together")
    if seed_start is None:
        return list(SEEDS)
    return list(range(seed_start, seed_start + seed_count))


def _legal_from_mask(mask) -> tuple[int, ...]:
    return tuple(int(index) for index, flag in enumerate(mask) if flag)


def _greedy_action(model, state, legal: tuple[int, ...], actor_id: int) -> int:
    """Plain policy-head argmax, no search - the exact POLICY_ONLY behavior
    `monopolyzero_common.build_local_policy_only` uses and 023 measured."""
    priors, _ = model.predict(state, legal, actor_id)
    return max(priors, key=priors.get)


def opportunity_greedy_stats(positions, model, *, buy_id: int, accept_id: int) -> dict:
    """For every collected self-play position where BUY_PROPERTY/
    ACCEPT_TRADE was legal, checks whether `model`'s current greedy choice
    is that action. Called once before training and once after (same model
    object, same positions) to measure whether the update moved greedy
    behavior at all on the states it was actually trained on."""
    buy_total = 0
    buy_chosen = 0
    accept_total = 0
    accept_chosen = 0
    for position in positions:
        legal = _legal_from_mask(position.legal_mask)
        if buy_id in legal:
            buy_total += 1
            if _greedy_action(model, position.state, legal, position.actor_id) == buy_id:
                buy_chosen += 1
        if accept_id in legal:
            accept_total += 1
            if _greedy_action(model, position.state, legal, position.actor_id) == accept_id:
                accept_chosen += 1
    return {
        "buy_property_opportunities": buy_total,
        "buy_property_greedy_chosen": buy_chosen,
        "buy_property_greedy_rate": (buy_chosen / buy_total) if buy_total else None,
        "accept_trade_opportunities": accept_total,
        "accept_trade_greedy_chosen": accept_chosen,
        "accept_trade_greedy_rate": (accept_chosen / accept_total) if accept_total else None,
    }


def generate_self_play_positions(seeds, model, search_config, max_rounds: int) -> dict:
    """Pure MaxNPUCT self-play (all 4 seats the same model, self_play=True) -
    zero ASU, zero HYBRID_COMPAT, zero fixed rule. Returns per-game reports
    plus the collected ReplayPosition list, never trains anything."""
    per_game: list[dict] = []
    all_positions: list = []
    total_illegal = 0
    total_crashed = 0
    incomplete = 0

    for seed in seeds:
        policy = common.build_local_search_policy(model, search_config, self_play=True)
        policies = {seat: policy for seat in range(NUM_SEATS)}
        outcome = common.play_local_game(
            game_id=seed, seed=seed, policies=policies,
            max_rounds=max_rounds, record_seats=set(range(NUM_SEATS)),
        )
        total_illegal += outcome.illegal_actions
        total_crashed += int(outcome.crashed)
        incomplete += int(not outcome.completed)
        all_positions.extend(outcome.positions)
        per_game.append(
            {
                "seed": seed, "completed": outcome.completed, "decisions": outcome.decisions,
                "positions_collected": len(outcome.positions), "winner": outcome.winner,
                "illegal_actions": outcome.illegal_actions, "crashed": outcome.crashed,
            }
        )

    return {
        "per_game": per_game, "positions": all_positions,
        "total_illegal": total_illegal, "total_crashed": total_crashed, "incomplete": incomplete,
    }


def train_candidate(model, positions, *, updates: int, batch_size: int, seed: int) -> list[dict]:
    """Runs `updates` native training-update steps
    (`monopolyzero_common.local_training_update`) over a fresh replay
    buffer built from `positions`. Mutates `model` in place; returns each
    update's stats dict."""
    import numpy as np
    import torch

    from monopoly_bench.storage import ReplayBuffer

    if REPLAY_DIR.exists():
        for child in REPLAY_DIR.glob("*"):
            child.unlink()
    else:
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    replay = ReplayBuffer(REPLAY_DIR, capacity=max(len(positions), 1), create=True)
    replay.append_many(positions)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)

    stats: list[dict] = []
    for _ in range(updates):
        batch = replay.sample(min(batch_size, len(replay)), rng)
        stats.append(common.local_training_update(model, optimizer, batch, gradient_clip=GRADIENT_CLIP))
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    """Every flag is optional - omitting all of them still runs a small
    (but real) training candidate over the reused 023 seed pool at the
    real target simulation count/max_rounds. Added specifically so a
    controlled 1-seed benchmark can be run before committing to a larger
    seed count - same discipline as the BUY/TRADE gradient diagnostic's own
    CLI and scripts/colab_shard_runner.py's benchmark-before-shard
    workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-start", type=int, default=None,
        help="first seed of a custom range; requires --seed-count too. Must be DEV-registered.",
    )
    parser.add_argument(
        "--seed-count", type=int, default=None,
        help="how many consecutive seeds from --seed-start; requires --seed-start too.",
    )
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="path to also write the raw structured JSON payload to (stdout is always printed regardless).",
    )
    parser.add_argument("--checkpoint-output", type=Path, default=DEFAULT_CHECKPOINT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    seeds = _resolve_seeds(args.seed_start, args.seed_count)

    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)
    ep.require_seed_scope(seeds, ep.SEED_CLASS_DEV, context="monopolyzero_native_train_candidate.py")
    baseline_checkpoint_sha256 = verify_baseline_checkpoint()

    common.ensure_reference_on_path()
    import random

    import numpy as np
    import torch

    from monopoly_bench.config import SearchConfig
    from monopoly_bench.engine import NUM_PLAYERS
    from monopoly_bench.model import MonopolyZeroNet
    from monopoly_game_engine.actions import ActionType

    if NUM_PLAYERS != NUM_SEATS:
        raise RuntimeError(f"Expected {NUM_SEATS} players, engine reports {NUM_PLAYERS}")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    model = MonopolyZeroNet.load_inference(CHECKPOINT_PATH)
    model.eval()
    search_config = SearchConfig(simulations=args.simulations)
    buy_id = int(ActionType.BUY_PROPERTY)
    accept_id = int(ActionType.ACCEPT_TRADE)

    def _emit(payload: dict) -> None:
        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
        print(text)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        generation = generate_self_play_positions(seeds, model, search_config, args.max_rounds)

        if generation["total_illegal"] or generation["total_crashed"] or generation["incomplete"]:
            payload = {
                "status": "FAILED_DURING_SELF_PLAY",
                "git_head_sha": git_head_sha,
                "total_illegal_actions": generation["total_illegal"],
                "total_crashed": generation["total_crashed"],
                "incomplete_games": generation["incomplete"],
                "per_game": generation["per_game"],
            }
            _emit(payload)
            raise RuntimeError("Stopping before training: self-play game(s) failed - see payload")

        positions = generation["positions"]
        if not positions:
            payload = {
                "status": "NO_POSITIONS_COLLECTED", "git_head_sha": git_head_sha,
                "per_game": generation["per_game"],
            }
            _emit(payload)
            raise RuntimeError("Stopping before training: no self-play positions collected")

        before_stats = opportunity_greedy_stats(positions, model, buy_id=buy_id, accept_id=accept_id)

        train_stats = train_candidate(model, positions, updates=args.updates, batch_size=BATCH_SIZE, seed=0)

        model.eval()
        after_stats = opportunity_greedy_stats(positions, model, buy_id=buy_id, accept_id=accept_id)

        args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        model.save_inference(
            args.checkpoint_output,
            {
                "source": "monopolyzero_native_train_candidate.py",
                "base_checkpoint_sha256": baseline_checkpoint_sha256,
                "seeds": seeds,
                "updates": args.updates,
            },
        )
        checkpoint_sha256 = _sha256(args.checkpoint_output)

        asu_modules_loaded = common.loaded_asu_modules()
    elapsed_s = time.perf_counter() - started

    payload = {
        "status": "OK",
        "git_head_sha": git_head_sha,
        "config": {
            "seeds": seeds,
            "seed_count": len(seeds),
            "seeds_reused_from": (
                "023 (43000-43019) - no new DEV seeds consumed"
                if seeds == list(SEEDS) else "custom seed range - validated against the registered DEV pool"
            ),
            "max_rounds": args.max_rounds,
            "search_config": asdict(search_config),
            "updates": args.updates,
            "batch_size": BATCH_SIZE,
            "policy": "pure MaxNPUCT self-play, all 4 seats, zero ASU, zero HYBRID_COMPAT, zero fixed rule",
        },
        "base_checkpoint_sha256": baseline_checkpoint_sha256,
        "per_game": generation["per_game"],
        "positions_collected": len(positions),
        "greedy_selection_before_training": before_stats,
        "greedy_selection_after_training": after_stats,
        "train_stats": train_stats,
        "checkpoint_output": str(args.checkpoint_output),
        "checkpoint_sha256": checkpoint_sha256,
        "asu_modules_loaded": asu_modules_loaded,
        "asu_modules_loaded_count": len(asu_modules_loaded),
        "elapsed_s": elapsed_s,
        "peak_rss_gib": rss.peak_gib,
    }
    _emit(payload)
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded during training: {asu_modules_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
