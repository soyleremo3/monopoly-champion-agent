"""ASU-import-free MonopolyZero self-play training-plumbing smoke.

Proves the training loop's plumbing works — NOT policy strength. Plays a
handful of short games with a hand-picked opponent pool (self-copy and
fixed-a/b/c only, never ASU), collects the resulting search positions into a
ReplayBuffer, and runs exactly one training update.

Import graph, corrected 2026-08-11 (later revision): an earlier version of
this script imported monopoly_bench.adapters, monopoly_bench.arena, and
monopoly_bench.training. Those modules never CALL ASU, but adapters.py does
`from ASU_FROZEN_TEACHER import ASURolloutV1, ASUValueV1` at module level,
and training.py does `from ASU_FROZEN_TEACHER import FROZEN_SPEC_HASH` at
module level — so merely importing either one loads ASU_FROZEN_TEACHER into
sys.modules as a side effect, even though its output/teacher/label was never
used. arena.py has no ASU text of its own but imports `.adapters`, so
importing it is transitively the same problem.

This version imports only modules confirmed ASU-import-clean by grep and by
reading their own import statements: monopoly_bench.engine, .model, .search,
.storage, .config, .contracts, and monopoly_game_engine.agents_fixed (see
docs/REFERENCE_AUDIT.md). The game loop, the search-policy wrapper, the
fixed-agent wrapper, and the training update step below are this project's
own implementation — built from the reference's own PUBLIC low-level
primitives (SharedGame, MaxNPUCT, ReplayBuffer, FP_AGENT_CLASSES) and their
observed contracts, not copied from adapters.py/arena.py/training.py. A
runtime guard at the end fails the run if ASU_FROZEN_TEACHER (or any
submodule of it) ever appears in sys.modules.

No self-play scaling, no ASU collection, no large training, no Modal, no
LLM — this is a single small smoke, per docs/DECISIONS.md.

Refuses to run unless PYTHONHASHSEED=0 is set, and refuses to run on a dirty
git working tree (git_head_sha in the output is the clean HEAD used).

Reproducibility: MonopolyZeroNet's value_head is never overwritten by
load_ppo_actor, so it (and any search-time exploration noise) needs Python/
NumPy/torch seeded explicitly before construction — GLOBAL_SEED does that,
separate from the existing per-game/per-decision seeding (SEEDS below).
"""

from __future__ import annotations

import json
import os
import subprocess
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
GLOBAL_SEED = 42
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0


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


def _require_clean_git_tree() -> str:
    """Refuses to run on a dirty tree; returns the clean HEAD SHA otherwise."""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    if status.stdout.strip():
        raise SystemExit(
            "selfplay_train_smoke.py refuses to run: working tree is not "
            "clean (git status --porcelain reported changes below). "
            "code_commit_sha must be an unambiguous clean HEAD — commit or "
            "stash first, then re-run.\n" + status.stdout
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return head.stdout.strip()


def _loaded_asu_modules() -> list[str]:
    return sorted(
        name for name in sys.modules
        if name == "ASU_FROZEN_TEACHER" or name.startswith("ASU_FROZEN_TEACHER.")
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
    git_head_sha = _require_clean_git_tree()

    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))
    import random

    import numpy as np
    import torch

    # ASU-import-clean modules only: no adapters, arena, or training here.
    from monopoly_bench.config import SearchConfig
    from monopoly_bench.contracts import ReplayPosition
    from monopoly_bench.engine import (
        NUM_PLAYERS,
        MAX_DECISIONS_PER_TURN,
        ActionType,
        SharedGame,
        actor_order,
        legal_mask,
        terminal_value,
        unwrap,
    )
    from monopoly_bench.model import MonopolyZeroNet
    from monopoly_bench.search import MaxNPUCT
    from monopoly_bench.storage import ReplayBuffer
    from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES

    # ── Our own search-policy wrapper (ASU-free: MaxNPUCT has zero ASU refs) ──
    class LocalSearchPolicy:
        def __init__(self, model: MonopolyZeroNet, config: SearchConfig, *, self_play: bool):
            self._search = MaxNPUCT(model, config, self_play=self_play)

        def choose(self, game, player_id: int, decision_seed: int):
            return self._search.choose_action(game, player_id, decision_seed)

    # ── Our own fixed-agent wrapper (behavior matched, code independent) ──────
    class LocalFixedPolicy:
        def __init__(self, agent_class):
            self.agent_class = agent_class
            self.name = agent_class.__name__
            self.fallback_count = 0

        def choose(self, game, player_id: int, decision_seed: int) -> int:
            env = unwrap(game)
            outer_state = random.getstate()
            try:
                random.seed(decision_seed)
                action = int(self.agent_class(player_id).choose_action(env))
            finally:
                random.setstate(outer_state)
            legal = tuple(env.get_allowed_actions(player_id))
            if action not in legal:
                self.fallback_count += 1
                action = int(ActionType.END_TURN) if int(ActionType.END_TURN) in legal else legal[0]
            return action

    # ── Our own small game loop (behavior matched to a fail-closed match) ─────
    def play_local_game(*, game_id: int, seed: int, policies: dict, max_rounds: int, record_seats: set[int]):
        game = SharedGame.new(seed, max_rounds)
        decisions = 0
        positions: list[ReplayPosition] = []
        illegal_actions = 0
        crashes = 0
        error = None
        try:
            for step in range(max_rounds * NUM_PLAYERS * MAX_DECISIONS_PER_TURN):
                if game.env.done:
                    break
                actor = game.env.whose_turn()
                legal = tuple(game.env.get_allowed_actions(actor))
                if len(legal) == 1:
                    action = legal[0]
                else:
                    decision_seed = seed * 1_000_003 + step * 17 + actor
                    policy = policies[actor]
                    if isinstance(policy, LocalSearchPolicy):
                        result = policy.choose(game, actor, decision_seed)
                        action = result.chosen_action
                        if actor in record_seats:
                            positions.append(
                                ReplayPosition(
                                    state=game.env._get_state(actor),
                                    legal_mask=legal_mask(legal),
                                    visits=result.visits,
                                    q_values=result.q_values,
                                    selected_action=action,
                                    value=result.root_value,
                                    actor_id=actor,
                                    game_id=game_id,
                                )
                            )
                    else:
                        action = policy.choose(game, actor, decision_seed)
                    if action not in legal:
                        illegal_actions += 1
                        raise RuntimeError(f"seat {actor} chose illegal action {action}")
                game.step(action)
                decisions = step + 1
        except Exception as exc:
            crashes += 1
            error = f"{type(exc).__name__}: {exc}"
            return {
                "completed": False, "winner": None, "decisions": decisions,
                "positions": positions, "illegal_actions": illegal_actions,
                "crashes": crashes, "error": error,
            }

        completed = game.env.done
        winner = game.env.winner() if completed else None
        if winner is not None:
            outcome = [0.0] * NUM_PLAYERS
            outcome[winner] = 1.0
            for position in positions:
                position.outcome = tuple(outcome)
        return {
            "completed": completed, "winner": winner, "decisions": decisions,
            "positions": positions, "illegal_actions": illegal_actions,
            "crashes": crashes, "error": error,
        }

    # ── Our own training update: MCTS visit-policy + real winner value target ─
    def local_training_update(model: MonopolyZeroNet, optimizer, batch: dict) -> dict:
        device = next(model.parameters()).device
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
        masks = torch.as_tensor(batch["legal_masks"], dtype=torch.bool, device=device)
        outcomes = torch.as_tensor(batch["outcomes"], dtype=torch.float32, device=device)
        actors = torch.as_tensor(batch["actors"], dtype=torch.long, device=device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
        counts = torch.as_tensor(batch["visit_counts"], dtype=torch.float32, device=device)
        lengths = torch.as_tensor(batch["lengths"], dtype=torch.long, device=device)

        # Reorder each position's physical winner-outcome vector into
        # actor-relative order (own seat first), matching how the model's
        # value head is defined (see MonopolyZeroNet.predict/forward).
        relative_targets = torch.stack(
            [outcomes[i, list(actor_order(int(actors[i])))] for i in range(outcomes.shape[0])]
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, values = model(states, masks)

        value_loss = -(relative_targets * values.clamp_min(1e-8).log()).sum(dim=1).mean()

        slot_index = torch.arange(actions.shape[1], device=device).unsqueeze(0)
        visit_weights = counts * (slot_index < lengths.unsqueeze(1))
        totals = visit_weights.sum(dim=1, keepdim=True)
        if (totals <= 0).any():
            raise ValueError("A sampled replay position has no recorded MCTS visits")
        visit_weights = visit_weights / totals
        log_probs = torch.log_softmax(logits, dim=1).gather(1, actions.clamp_min(0))
        log_probs = torch.where(visit_weights > 0, log_probs, torch.zeros_like(log_probs))
        policy_loss = -(visit_weights * log_probs).sum(dim=1).mean()

        loss = policy_loss + value_loss
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        return {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "gradient_norm": float(gradient_norm),
        }

    started = time.perf_counter()
    with _RssMonitor() as rss:
        random.seed(GLOBAL_SEED)
        np.random.seed(GLOBAL_SEED)
        torch.manual_seed(GLOBAL_SEED)

        model = MonopolyZeroNet()
        if DEFAULT_PPO.is_file():
            model.load_ppo_actor(DEFAULT_PPO)
            bootstrap = "ppo"
        else:
            bootstrap = "random_init"
        model.eval()

        search_config = SearchConfig(simulations=SIMULATIONS, max_depth=16)

        games = []
        for label in ("self_play_1", "self_play_2"):
            policies = {
                seat: LocalSearchPolicy(model, search_config, self_play=True)
                for seat in range(NUM_PLAYERS)
            }
            result = play_local_game(
                game_id=SEEDS[label], seed=SEEDS[label], policies=policies,
                max_rounds=MAX_ROUNDS, record_seats=set(range(NUM_PLAYERS)),
            )
            games.append((label, result))

        fixed_a, fixed_b, fixed_c = (LocalFixedPolicy(cls) for cls in FP_AGENT_CLASSES[:3])
        policies = {
            0: LocalSearchPolicy(model, search_config, self_play=True),
            1: fixed_a, 2: fixed_b, 3: fixed_c,
        }
        result = play_local_game(
            game_id=SEEDS["vs_fixed"], seed=SEEDS["vs_fixed"], policies=policies,
            max_rounds=MAX_ROUNDS, record_seats={0},
        )
        games.append(("vs_fixed", result))

        fixed_adapter_fallbacks = {
            "fixed_a": fixed_a.fallback_count,
            "fixed_b": fixed_b.fallback_count,
            "fixed_c": fixed_c.fallback_count,
        }

        game_reports = []
        total_illegal = 0
        total_crashes = 0
        all_positions: list[ReplayPosition] = []
        for label, result in games:
            total_illegal += result["illegal_actions"]
            total_crashes += result["crashes"]
            all_positions.extend(result["positions"])
            game_reports.append(
                {
                    "label": label,
                    "seed": SEEDS[label],
                    "completed": result["completed"],
                    "decisions": result["decisions"],
                    "winner": result["winner"],
                    "positions_collected": len(result["positions"]),
                    "illegal_actions": result["illegal_actions"],
                    "crashes": result["crashes"],
                    "error": result["error"],
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
        stats = local_training_update(model, optimizer, batch)
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

        asu_modules_loaded = _loaded_asu_modules()

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
