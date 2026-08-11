"""ASU-import-free MonopolyZero strength-pilot trainer.

Reusable runner built on scripts/monopolyzero_common.py — never imports
monopoly_bench.adapters/.arena/.training (see that module's docstring and
docs/DECISIONS.md's 2026-08-11 "later" correction). Saves a pre-training
("baseline") and a post-training ("trained") checkpoint via
MonopolyZeroNet.save_inference (ASU-free), both under artifacts/ (gitignored,
never committed).

Fixed for this pilot: 32 games, seed 42, CPU, 4 PUCT simulations,
max_rounds=50, opponent mix seat-balanced self-play + fixed-a/b/c, no ASU.
Optimizer hyperparameters (lr, weight_decay, gradient_clip) are read
directly from monopoly_bench.config.TrainingConfig's defaults (ASU-free
dataclass, just parameters); batch_size and update count are this project's
own choice for a small pilot (not the frozen-v1 full-scale defaults, which
assume a much larger generation) — both logged exactly below.

Refuses to run unless PYTHONHASHSEED=0 is set, refuses on a dirty git tree,
and fails loudly (no evaluation afterward) on a crash, an illegal action, a
non-finite loss, or any ASU module ending up in sys.modules.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import monopolyzero_common as common  # noqa: E402

GAMES_TOTAL = 32
GLOBAL_SEED = 42
SIMULATIONS = 4
MAX_ROUNDS = 50
BATCH_SIZE = 64
UPDATES = 100
MAX_DEPTH = 16

REPO_ROOT = common.REPO_ROOT
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "monopolyzero_strength_pilot"
REPLAY_DIR = ARTIFACT_DIR / "replay"
BASELINE_CHECKPOINT = ARTIFACT_DIR / "baseline_pretraining.pt"
TRAINED_CHECKPOINT = ARTIFACT_DIR / "trained_32games.pt"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_game_plan(games_total: int) -> list[dict]:
    """32 games: half self-play (our model in every seat), half vs-fixed
    (our model rotates through every seat evenly, fixed-a/b/c fill the
    rest). A plain, own-written seat rotation — not imported from
    monopoly_bench.arena's balanced_single_seats/balanced_team_seats,
    which live in the ASU-coupled arena.py module."""
    half = games_total // 2
    plan = []
    for index in range(half):
        plan.append({"category": "self_play", "seed": 10_000 + index, "focus_seat": None})
    for index in range(half):
        plan.append({"category": "vs_fixed", "seed": 20_000 + index, "focus_seat": index % 4})
    return plan


def main(argv: list[str] | None = None) -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)

    common.ensure_reference_on_path()
    import random

    import numpy as np
    import torch

    from monopoly_bench.config import SearchConfig, TrainingConfig
    from monopoly_bench.engine import NUM_PLAYERS
    from monopoly_bench.model import MonopolyZeroNet
    from monopoly_bench.storage import ReplayBuffer
    from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES

    training_defaults = TrainingConfig()

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        random.seed(GLOBAL_SEED)
        np.random.seed(GLOBAL_SEED)
        torch.manual_seed(GLOBAL_SEED)

        model = MonopolyZeroNet()
        if not common.DEFAULT_PPO.is_file():
            raise SystemExit(f"Missing PPO-compatible checkpoint: {common.DEFAULT_PPO}")
        model.load_ppo_actor(common.DEFAULT_PPO)
        model.eval()

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_inference(
            BASELINE_CHECKPOINT,
            {"stage": "pre_training", "games_trained": 0, "seed": GLOBAL_SEED},
        )
        baseline_sha256 = _sha256(BASELINE_CHECKPOINT)

        search_config = SearchConfig(simulations=SIMULATIONS, max_depth=MAX_DEPTH)
        fixed_pool = {
            0: FP_AGENT_CLASSES[0], 1: FP_AGENT_CLASSES[1], 2: FP_AGENT_CLASSES[2],
        }

        game_plan = _build_game_plan(GAMES_TOTAL)
        game_reports = []
        all_positions = []
        total_illegal = 0
        total_crashed = 0

        for job in game_plan:
            if job["category"] == "self_play":
                policies = {
                    seat: common.build_local_search_policy(model, search_config, self_play=True)
                    for seat in range(NUM_PLAYERS)
                }
                record_seats = set(range(NUM_PLAYERS))
            else:
                focus_seat = job["focus_seat"]
                non_focus_seats = [seat for seat in range(NUM_PLAYERS) if seat != focus_seat]
                policies = {focus_seat: common.build_local_search_policy(model, search_config, self_play=True)}
                for slot, seat in enumerate(non_focus_seats):
                    policies[seat] = common.LocalFixedPolicy(list(fixed_pool.values())[slot])
                record_seats = {focus_seat}

            outcome = common.play_local_game(
                game_id=job["seed"], seed=job["seed"], policies=policies,
                max_rounds=MAX_ROUNDS, record_seats=record_seats,
            )
            total_illegal += outcome.illegal_actions
            total_crashed += int(outcome.crashed)
            all_positions.extend(outcome.positions)

            fixed_fallbacks = {
                policy.name: policy.fallback_count
                for policy in policies.values()
                if getattr(policy, "kind", None) == "fixed"
            }
            game_reports.append(
                {
                    "category": job["category"],
                    "seed": job["seed"],
                    "focus_seat": job["focus_seat"],
                    "completed": outcome.completed,
                    "decisions": outcome.decisions,
                    "winner": outcome.winner,
                    "positions_collected": len(outcome.positions),
                    "illegal_actions": outcome.illegal_actions,
                    "crashed": outcome.crashed,
                    "error": outcome.error,
                    "fixed_fallbacks": fixed_fallbacks,
                }
            )

            if outcome.crashed or outcome.illegal_actions:
                break  # stop generating further games; the run fails below anyway

        if total_crashed or total_illegal:
            payload = {
                "status": "FAILED_DURING_GAME_GENERATION",
                "git_head_sha": git_head_sha,
                "games": game_reports,
                "total_illegal_actions": total_illegal,
                "total_crashed": total_crashed,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            raise RuntimeError(
                f"Stopping before training: crashed={total_crashed} illegal={total_illegal}"
            )
        if not all_positions:
            raise RuntimeError("No positions collected from any game")

        if REPLAY_DIR.exists():
            for child in REPLAY_DIR.glob("*"):
                child.unlink()
        else:
            REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        replay = ReplayBuffer(REPLAY_DIR, capacity=max(len(all_positions), 1), create=True)
        written_indices = replay.append_many(all_positions)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_defaults.learning_rate,
            weight_decay=training_defaults.weight_decay,
        )

        rng = np.random.default_rng(GLOBAL_SEED)
        update_stats = []
        loss_all_finite = True
        for update_index in range(UPDATES):
            batch = replay.sample(min(BATCH_SIZE, len(all_positions)), rng)
            stats = common.local_training_update(
                model, optimizer, batch, gradient_clip=training_defaults.gradient_clip
            )
            finite = all(
                torch.isfinite(torch.tensor(value)).item()
                for key, value in stats.items()
                if key in ("loss", "policy_loss", "value_loss", "gradient_norm")
            )
            if not finite:
                loss_all_finite = False
            update_stats.append(stats)
            if not finite:
                break  # stop immediately on non-finite loss, per task instructions

        asu_modules_loaded = common.loaded_asu_modules()

        if loss_all_finite and not asu_modules_loaded:
            model.eval()
            model.save_inference(
                TRAINED_CHECKPOINT,
                {
                    "stage": "post_training",
                    "games_trained": GAMES_TOTAL,
                    "updates": len(update_stats),
                    "seed": GLOBAL_SEED,
                },
            )
            trained_sha256 = _sha256(TRAINED_CHECKPOINT)
        else:
            trained_sha256 = None

    elapsed_s = time.perf_counter() - started

    final_losses = update_stats[-1] if update_stats else None
    payload = {
        "status": "OK" if (loss_all_finite and not asu_modules_loaded) else "FAILED",
        "git_head_sha": git_head_sha,
        "config": {
            "games_total": GAMES_TOTAL,
            "global_seed": GLOBAL_SEED,
            "simulations": SIMULATIONS,
            "max_depth": MAX_DEPTH,
            "max_rounds": MAX_ROUNDS,
            "batch_size": BATCH_SIZE,
            "updates_requested": UPDATES,
            "learning_rate": training_defaults.learning_rate,
            "weight_decay": training_defaults.weight_decay,
            "gradient_clip": training_defaults.gradient_clip,
            "optimizer_source": "monopoly_bench.config.TrainingConfig defaults (learning_rate/weight_decay/gradient_clip only)",
            "batch_size_and_updates_source": "this project's own pilot-scale choice, not TrainingConfig's frozen-v1 defaults (batch_size=256, updates_per_generation=1000, sized for a 32-games-per-generation full run, not a one-shot 32-game pilot)",
            "opponent_mix": "16 self-play games (model controls all 4 seats) + 16 vs-fixed games (model rotates through seats 0-3, 4 times each; fixed-a/b/c fill the other 3 seats)",
        },
        "baseline_checkpoint": {"path": str(BASELINE_CHECKPOINT), "sha256": baseline_sha256},
        "trained_checkpoint": {"path": str(TRAINED_CHECKPOINT) if trained_sha256 else None, "sha256": trained_sha256},
        "games": game_reports,
        "positions_collected_total": len(all_positions),
        "replay_buffer_size_after_append": len(replay),
        "replay_indices_written": len(written_indices),
        "updates_completed": len(update_stats),
        "loss_all_finite": loss_all_finite,
        "final_update_stats": final_losses,
        "total_illegal_actions": total_illegal,
        "total_crashed": total_crashed,
        "asu_modules_loaded": asu_modules_loaded,
        "asu_modules_loaded_count": len(asu_modules_loaded),
        "elapsed_s": elapsed_s,
        "peak_rss_gib": rss.peak_gib,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "OK":
        raise RuntimeError(f"Training pilot failed: {payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
