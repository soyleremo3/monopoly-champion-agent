"""Resumes 027's frozen 32-game PURE PPO checkpoint for 96 ADDITIONAL games
(128 total, via the reference's own unmodified PPOAgent/train() - same
call pattern as 027, no architecture/reward/action-space change), saves it
SEPARATELY (the 32-game checkpoint is never overwritten), then directly
strength-tests it against the frozen 32-game checkpoint by reusing 028's
exact masked-argmax policy/play_one_game/summarize and win-rate-vs-25%-
null bootstrap rule unmodified - no new evaluation framework. See
docs/EXPERIMENTS.md's 030 entry for the seed-derivation math."""

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

ADDITIONAL_GAMES = 96
EPISODE_SEED_START = 47000
# train()'s formula: episode_seed = seed_arg + absolute_game - 1; absolute_game
# starts at games_trained(32) + 1 = 33 for this resumed run.
BASE_SEED_ARG = EPISODE_SEED_START - 32
OUTPUT_CHECKPOINT = (
    common.REPO_ROOT / "artifacts" / "monopolyzero_pure_ppo_learnability_gate" / "candidate_ppo_128.pt"
)
EVAL_SEED_BASE, N_EVAL_SEEDS = 48000, 20


def resume_training(train_seeds: list[int]) -> tuple:
    common.ensure_reference_on_path()
    from monopoly_game_engine.train import train as reference_train

    agent = screen.load_candidate_agent(screen.DEFAULT_CANDIDATE_CHECKPOINT)
    starting_games = agent.games_trained
    if starting_games != 32:
        raise SystemExit(f"STOP: expected games_trained==32 before resuming, got {starting_games}")

    history = reference_train(
        agent, is_ppo=True, hybrid=False, n_games=ADDITIONAL_GAMES, log_every=1, seed=BASE_SEED_ARG,
    )
    actual_seeds = [BASE_SEED_ARG + g - 1 for g in history["games"]]
    if actual_seeds != train_seeds:
        raise RuntimeError(f"Episode seed mismatch: got {actual_seeds}, expected {train_seeds}")
    expected_final = starting_games + ADDITIONAL_GAMES
    if agent.games_trained != expected_final:
        raise SystemExit(f"STOP: training did not reach games_trained=={expected_final}, got {agent.games_trained}")

    agent.actor.eval()
    OUTPUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(OUTPUT_CHECKPOINT))
    return agent, actual_seeds


def strength_test(candidate_agent, eval_seeds: list[int], device) -> dict:
    baseline_agent = screen.load_candidate_agent(screen.DEFAULT_CANDIDATE_CHECKPOINT)  # frozen 32-game
    games = [
        screen.play_one_game(
            game_id=game_id, seed=seed, candidate_actor=candidate_agent.actor,
            baseline_actor=baseline_agent.actor, focus_seat=focus_seat, device=device, max_rounds=screen.MAX_ROUNDS,
        )
        for game_id, (seed, focus_seat) in enumerate(
            ((s, seat) for s in eval_seeds for seat in range(screen.NUM_SEATS)), start=1
        )
    ]
    return screen.summarize(games)


def main() -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)
    train_seeds = list(range(EPISODE_SEED_START, EPISODE_SEED_START + ADDITIONAL_GAMES))
    eval_seeds = screen._seed_range(EVAL_SEED_BASE, N_EVAL_SEEDS)
    ep.require_seed_scope(train_seeds, ep.SEED_CLASS_DEV, context="resume_128_strength_test.py (train)")
    ep.require_seed_scope(eval_seeds, ep.SEED_CLASS_DEV, context="resume_128_strength_test.py (eval)")

    import torch

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        agent, actual_seeds = resume_training(train_seeds)
        eval_summary = strength_test(agent, eval_seeds, torch.device("cpu"))
    elapsed_s = time.perf_counter() - started

    asu_modules_loaded = common.loaded_asu_modules()
    payload = {
        "status": "OK", "git_head_sha": git_head_sha, "elapsed_s": elapsed_s, "peak_rss_gib": rss.peak_gib,
        "episode_seed_first": actual_seeds[0], "episode_seed_last": actual_seeds[-1],
        "games_trained_start": 32, "games_trained_end": agent.games_trained,
        "checkpoint_output": str(OUTPUT_CHECKPOINT),
        "checkpoint_output_sha256": screen._file_sha256(OUTPUT_CHECKPOINT),
        "actor_sha256_128game": screen._full_actor_sha256(agent.actor),
        "eval_seeds": eval_seeds, "asu_modules_loaded": asu_modules_loaded,
        "strength_test_summary": eval_summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded: {asu_modules_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
