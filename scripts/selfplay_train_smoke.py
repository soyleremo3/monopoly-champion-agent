"""ASU-import-free MonopolyZero self-play training-plumbing smoke.

Proves the training loop's plumbing works — NOT policy strength. Plays a
handful of short games with a hand-picked opponent pool (self-copy and
fixed-a/b/c only, never ASU), collects the resulting search positions into a
ReplayBuffer, and runs exactly one training update.

Thin CLI over scripts/monopolyzero_common.py, which holds the actual
ASU-import-free game loop, search/fixed-agent wrappers, and training-update
step (see that module's docstring for the import-graph and
source-similarity audit notes). No self-play scaling, no ASU collection, no
large training, no Modal, no LLM — this is a single small smoke, per
docs/DECISIONS.md.

Refuses to run unless PYTHONHASHSEED=0 is set, and refuses to run on a dirty
git working tree (git_head_sha in the output is the clean HEAD used).
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

MAX_ROUNDS = 5
SIMULATIONS = 4
SEEDS = {"self_play_1": 501, "self_play_2": 502, "vs_fixed": 503}
BATCH_SIZE = 8
GLOBAL_SEED = 42
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

REPO_ROOT = common.REPO_ROOT
REPLAY_DIR = REPO_ROOT / "artifacts" / "monopolyzero_smoke" / "replay"


def main(argv: list[str] | None = None) -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)

    common.ensure_reference_on_path()
    import random

    import numpy as np
    import torch

    from monopoly_bench.config import SearchConfig
    from monopoly_bench.engine import NUM_PLAYERS
    from monopoly_bench.model import MonopolyZeroNet
    from monopoly_bench.storage import ReplayBuffer
    from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        random.seed(GLOBAL_SEED)
        np.random.seed(GLOBAL_SEED)
        torch.manual_seed(GLOBAL_SEED)

        model = MonopolyZeroNet()
        if common.DEFAULT_PPO.is_file():
            model.load_ppo_actor(common.DEFAULT_PPO)
            bootstrap = "ppo"
        else:
            bootstrap = "random_init"
        model.eval()

        search_config = SearchConfig(simulations=SIMULATIONS, max_depth=16)

        games = []
        for label in ("self_play_1", "self_play_2"):
            policies = {
                seat: common.build_local_search_policy(model, search_config, self_play=True)
                for seat in range(NUM_PLAYERS)
            }
            outcome = common.play_local_game(
                game_id=SEEDS[label], seed=SEEDS[label], policies=policies,
                max_rounds=MAX_ROUNDS, record_seats=set(range(NUM_PLAYERS)),
            )
            games.append((label, outcome))

        fixed_a, fixed_b, fixed_c = (common.LocalFixedPolicy(cls) for cls in FP_AGENT_CLASSES[:3])
        policies = {
            0: common.build_local_search_policy(model, search_config, self_play=True),
            1: fixed_a, 2: fixed_b, 3: fixed_c,
        }
        outcome = common.play_local_game(
            game_id=SEEDS["vs_fixed"], seed=SEEDS["vs_fixed"], policies=policies,
            max_rounds=MAX_ROUNDS, record_seats={0},
        )
        games.append(("vs_fixed", outcome))

        fixed_adapter_fallbacks = {
            "fixed_a": fixed_a.fallback_count,
            "fixed_b": fixed_b.fallback_count,
            "fixed_c": fixed_c.fallback_count,
        }

        game_reports = []
        total_illegal = 0
        total_crashes = 0
        all_positions = []
        for label, outcome in games:
            total_illegal += outcome.illegal_actions
            total_crashes += int(outcome.crashed)
            all_positions.extend(outcome.positions)
            game_reports.append(
                {
                    "label": label,
                    "seed": SEEDS[label],
                    "completed": outcome.completed,
                    "decisions": outcome.decisions,
                    "winner": outcome.winner,
                    "positions_collected": len(outcome.positions),
                    "illegal_actions": outcome.illegal_actions,
                    "crashes": int(outcome.crashed),
                    "error": outcome.error,
                }
            )

        if total_crashes or total_illegal:
            raise RuntimeError(
                f"Aborting before training: crashes={total_crashes} "
                f"illegal_actions={total_illegal}; see game_reports"
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

        rng = np.random.default_rng(42)
        batch = replay.sample(min(BATCH_SIZE, len(all_positions)), rng)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        before = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }
        stats = common.local_training_update(model, optimizer, batch, gradient_clip=GRADIENT_CLIP)
        changed_params = [
            name
            for name, parameter in model.named_parameters()
            if not torch.equal(before[name], parameter.detach())
        ]

        loss_finite = all(
            torch.isfinite(torch.tensor(value)).item()
            for key, value in stats.items()
            if key in ("loss", "policy_loss", "value_loss", "gradient_norm")
        )

        asu_modules_loaded = common.loaded_asu_modules()

    elapsed_s = time.perf_counter() - started

    payload = {
        "git_head_sha": git_head_sha,
        "global_seed": GLOBAL_SEED,
        "bootstrap": bootstrap,
        "games": game_reports,
        "positions_collected_total": len(all_positions),
        "replay_buffer_size_after_append": len(replay),
        "replay_indices_written": len(written_indices),
        "batch_size_sampled": int(batch["states"].shape[0]),
        "train_step_stats": stats,
        "loss_finite": loss_finite,
        "parameters_changed_count": len(changed_params),
        "parameters_total_count": len(before),
        "total_illegal_actions": total_illegal,
        "total_crashes": total_crashes,
        "fixed_adapter_fallbacks": fixed_adapter_fallbacks,
        "fixed_adapter_fallbacks_total": sum(fixed_adapter_fallbacks.values()),
        "asu_modules_loaded": asu_modules_loaded,
        "asu_modules_loaded_count": len(asu_modules_loaded),
        "elapsed_s": elapsed_s,
        "peak_rss_gib": rss.peak_gib,
    }

    if asu_modules_loaded:
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise RuntimeError(
            f"ASU modules loaded during a supposedly ASU-independent run: {asu_modules_loaded}"
        )

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
