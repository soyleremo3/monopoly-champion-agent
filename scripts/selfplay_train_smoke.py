"""ASU-independent MonopolyZero self-play training-plumbing smoke.

Proves the training loop's plumbing works — NOT policy strength. Plays a
handful of short games with a hand-picked opponent pool (self-copy and
fixed-a/b/c only, never ASU), collects the resulting search positions into a
ReplayBuffer, and runs exactly one train_step update.

Deliberately does NOT use monopoly_bench.training.Trainer or
population_jobs: population_jobs hardcodes ASU into part of every
generation's opponent pool with no way to disable it (see
docs/REFERENCE_AUDIT.md), so it is not usable under the ASU-evaluation-only
policy in CLAUDE.md. This script only imports the ASU-independent public
building blocks (MonopolyZeroNet, MaxNPUCT/SearchAdapter, arena.play_game,
ReplayBuffer, train_step) and FP_AGENT_CLASSES[:3] (fixed-a/b/c) directly.

No self-play scaling, no ASU collection, no large training, no Modal, no
LLM — this is a single small smoke, per docs/DECISIONS.md.

Refuses to run unless PYTHONHASHSEED=0 is set, same reproducibility reason
as every other script in this directory.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "references" / "DeepRL_Monopoly"
DEFAULT_PPO = REFERENCE_ROOT / "artifacts" / "ppo_plus" / "ppo_hybrid_2000_v2.pt"
REPLAY_DIR = REPO_ROOT / "artifacts" / "monopolyzero_smoke" / "replay"

REQUIRED_HASH_SEED = "0"

MAX_ROUNDS = 5
SIMULATIONS = 4
SEEDS = {"self_play_1": 501, "self_play_2": 502, "vs_fixed": 503}
BATCH_SIZE = 8


def _require_pinned_hash_seed() -> None:
    actual = os.environ.get("PYTHONHASHSEED")
    if actual == REQUIRED_HASH_SEED:
        return
    shown = "unset" if actual is None else repr(actual)
    raise SystemExit(
        "selfplay_train_smoke.py refuses to run: PYTHONHASHSEED must be "
        f"'{REQUIRED_HASH_SEED}', got {shown}. Re-run as:\n"
        f"  PYTHONHASHSEED={REQUIRED_HASH_SEED} python {Path(__file__).name}"
    )


class _RssMonitor:
    """Polls this process's RSS on a background thread; reports the peak."""

    def __init__(self, interval_s: float = 0.02):
        import psutil

        self._process = psutil.Process(os.getpid())
        self._interval_s = interval_s
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._peak = max(self._peak, self._process.memory_info().rss)
            except Exception:
                pass
            time.sleep(self._interval_s)

    def __enter__(self) -> "_RssMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def peak_gib(self) -> float:
        return self._peak / 1024**3


def main(argv: list[str] | None = None) -> int:
    _require_pinned_hash_seed()

    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))
    import numpy as np
    import torch

    from monopoly_bench.adapters import FixedAdapter, SearchAdapter
    from monopoly_bench.arena import play_game
    from monopoly_bench.config import SearchConfig
    from monopoly_bench.engine import NUM_PLAYERS
    from monopoly_bench.model import MonopolyZeroNet
    from monopoly_bench.storage import ReplayBuffer
    from monopoly_bench.training import train_step
    from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES

    started = time.perf_counter()
    with _RssMonitor() as rss:
        model = MonopolyZeroNet()
        if DEFAULT_PPO.is_file():
            model.load_ppo_actor(DEFAULT_PPO)
            bootstrap = "ppo"
        else:
            bootstrap = "random_init"
        model.eval()

        search_config = SearchConfig(simulations=SIMULATIONS, max_depth=16)

        games = []

        # Two pure self-play games: our model occupies every seat (self-copy).
        for label in ("self_play_1", "self_play_2"):
            policies = {
                seat: SearchAdapter(model, search_config, self_play=True)
                for seat in range(NUM_PLAYERS)
            }
            result = play_game(
                game_id=SEEDS[label],
                seed=SEEDS[label],
                policies=policies,
                max_rounds=MAX_ROUNDS,
                record_seats=set(range(NUM_PLAYERS)),
            )
            games.append((label, result))

        # One game against the fixed-a/b/c opponent pool, our model in seat 0.
        fixed_a, fixed_b, fixed_c = (
            FixedAdapter(agent_class) for agent_class in FP_AGENT_CLASSES[:3]
        )
        policies = {
            0: SearchAdapter(model, search_config, self_play=True),
            1: fixed_a,
            2: fixed_b,
            3: fixed_c,
        }
        result = play_game(
            game_id=SEEDS["vs_fixed"],
            seed=SEEDS["vs_fixed"],
            policies=policies,
            max_rounds=MAX_ROUNDS,
            record_seats={0},
        )
        games.append(("vs_fixed", result))

        game_reports = []
        total_illegal = 0
        total_crashes = 0
        all_positions = []
        for label, result in games:
            total_illegal += result.illegal_actions
            total_crashes += result.crashes
            all_positions.extend(result.positions)
            game_reports.append(
                {
                    "label": label,
                    "seed": result.seed,
                    "completed": result.completed,
                    "decisions": result.decisions,
                    "winner": result.winner,
                    "positions_collected": len(result.positions),
                    "illegal_actions": result.illegal_actions,
                    "crashes": result.crashes,
                    "error": result.error,
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

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        before = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }
        stats = train_step(model, optimizer, scaler, batch, gradient_clip=1.0)
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

    elapsed_s = time.perf_counter() - started

    payload = {
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
        "elapsed_s": elapsed_s,
        "peak_rss_gib": rss.peak_gib,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
