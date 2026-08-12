"""Resumes 027's frozen 32-game PURE PPO checkpoint (candidate_ppo.pt) along
the SAME continuation trajectory as experiment 030 (episode seeds
47000-47095, base seed arg 46968), but checkpointed every 16 games instead
of once at the end, to find whether an early-stop point between 32 and 128
games beats the 32-game champion. Training is split into 6 sequential
`train()` calls (32->48->64->80->96->112->128) on the SAME in-memory
PPOAgent - no disk round-trip between segments, and the reference's own
per-game full RNG reseed (episode_seed = seed_arg + absolute_game - 1)
makes this bit-exact equivalent to 030's single continuous 96-game call.
The 128-game actor hash is verified against 030's own logged value as a
reproducibility gate before any strength conclusion is drawn.

Selection screen and confirmation both reuse 028/030's exact
play_one_game/summarize (unmodified) - one candidate focus seat vs three
frozen copies of the hash-gated 32-game champion, deterministic masked
argmax, no new evaluation framework. See docs/EXPERIMENTS.md's 031 entry.
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

CHECKPOINT_STAGES = (48, 64, 80, 96, 112, 128)
EPISODE_SEED_START, ADDITIONAL_GAMES = 47000, 96
BASE_SEED_ARG = EPISODE_SEED_START - 32  # train()'s formula: episode_seed = seed_arg + absolute_game - 1
EXPECTED_128_ACTOR_SHA256 = "ad4e9a6d066aab3b2ff5c8414b40669cae51b4bde6d49c42a1ce8128985dd224"
CHECKPOINT_DIR = common.REPO_ROOT / "artifacts" / "monopolyzero_pure_ppo_learnability_gate"

SELECTION_SEED_BASE, N_SELECTION_SEEDS = 49000, 8
CONFIRM_SEED_BASE, N_CONFIRM_SEEDS = 49100, 20
SELECTION_WIN_RATE_THRESHOLD = 0.25


def resume_and_checkpoint() -> dict:
    common.ensure_reference_on_path()
    from monopoly_game_engine.train import train as reference_train

    agent = screen.load_candidate_agent(screen.DEFAULT_CANDIDATE_CHECKPOINT)
    if agent.games_trained != 32:
        raise SystemExit(f"STOP: expected games_trained==32 before resuming, got {agent.games_trained}")

    all_train_seeds = list(range(EPISODE_SEED_START, EPISODE_SEED_START + ADDITIONAL_GAMES))
    stages: dict[int, dict] = {}
    prev_total, seed_cursor = 32, 0
    for total in CHECKPOINT_STAGES:
        n_games = total - prev_total
        history = reference_train(agent, is_ppo=True, hybrid=False, n_games=n_games, log_every=1, seed=BASE_SEED_ARG)
        actual = [BASE_SEED_ARG + g - 1 for g in history["games"]]
        expected = all_train_seeds[seed_cursor: seed_cursor + n_games]
        if actual != expected:
            raise RuntimeError(f"Episode seed mismatch at stage {total}: got {actual}, expected {expected}")
        if agent.games_trained != total:
            raise SystemExit(f"STOP: training did not reach games_trained=={total}, got {agent.games_trained}")
        agent.actor.eval()
        out_path = CHECKPOINT_DIR / f"candidate_ppo_{total}.pt"
        agent.save(str(out_path))
        stages[total] = {"path": str(out_path), "checkpoint_sha256": screen._file_sha256(out_path),
                          "actor_sha256": screen._full_actor_sha256(agent.actor)}
        prev_total, seed_cursor = total, seed_cursor + n_games
    return stages


def eval_candidate(checkpoint_path: Path, eval_seeds: list[int], device) -> dict:
    from monopoly_game_engine.agent_ppo import PPOAgent

    candidate = PPOAgent(player_id=0, hybrid=False, device="cpu")
    candidate.load(str(checkpoint_path))
    candidate.actor.eval()
    baseline = screen.load_candidate_agent(screen.DEFAULT_CANDIDATE_CHECKPOINT)

    games = [
        screen.play_one_game(game_id=gid, seed=s, candidate_actor=candidate.actor, baseline_actor=baseline.actor,
                              focus_seat=seat, device=device, max_rounds=screen.MAX_ROUNDS)
        for gid, (s, seat) in enumerate(((s, seat) for s in eval_seeds for seat in range(screen.NUM_SEATS)), start=1)
    ]
    return screen.summarize(games)


def main() -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)

    train_seeds = list(range(EPISODE_SEED_START, EPISODE_SEED_START + ADDITIONAL_GAMES))
    selection_seeds = screen._seed_range(SELECTION_SEED_BASE, N_SELECTION_SEEDS)
    confirm_seeds = screen._seed_range(CONFIRM_SEED_BASE, N_CONFIRM_SEEDS)
    ep.require_seed_scope(train_seeds, ep.SEED_CLASS_DEV, context="early_stop_search.py (train)")
    ep.require_seed_scope(selection_seeds, ep.SEED_CLASS_DEV, context="early_stop_search.py (selection)")
    ep.require_seed_scope(confirm_seeds, ep.SEED_CLASS_DEV, context="early_stop_search.py (confirm)")

    import torch

    device = torch.device("cpu")
    result: dict = {}
    started = time.perf_counter()
    with common.RssMonitor() as rss:
        stages = resume_and_checkpoint()
        gate_pass = stages[128]["actor_sha256"] == EXPECTED_128_ACTOR_SHA256
        result["checkpoint_stages"] = stages
        result["hash_gate_128_pass"] = gate_pass

        if not gate_pass:
            result["status"] = "STOP_NON_DETERMINISM"
        else:
            selection = {
                total: eval_candidate(Path(stages[total]["path"]), selection_seeds, device)
                for total in (48, 64, 80, 96, 112)
            }
            result["selection"] = selection
            result["selection_seeds"] = selection_seeds
            best_total = max(selection, key=lambda t: (
                selection[t]["candidate_win_rate"], selection[t]["candidate_mean_net_worth"], -t))
            best = selection[best_total]

            if best["candidate_win_rate"] <= SELECTION_WIN_RATE_THRESHOLD:
                result["status"] = "KILL_NO_CANDIDATE_ABOVE_THRESHOLD"
                result["champion_remains"] = "32-game"
            else:
                confirmation = eval_candidate(Path(stages[best_total]["path"]), confirm_seeds, device)
                result["selected_checkpoint"] = best_total
                result["confirm_seeds"] = confirm_seeds
                result["confirmation"] = confirmation
                ci = confirmation["primary_seed_block_bootstrap_vs_25pct_null"]["win_rate_diff_from_null"]["ci_95"]
                if ci[0] > 0.0:
                    result["status"] = "GO_PROMOTE_EARLY_STOP_CANDIDATE"
                    result["champion_remains"] = f"{best_total}-game"
                elif ci[1] <= 0.0:
                    result["status"] = "KILL"
                    result["champion_remains"] = "32-game"
                else:
                    result["status"] = "INCONCLUSIVE"
                    result["champion_remains"] = "32-game"
    elapsed_s = time.perf_counter() - started

    asu_modules_loaded = common.loaded_asu_modules()
    payload = {
        "git_head_sha": git_head_sha, "elapsed_s": elapsed_s, "peak_rss_gib": rss.peak_gib,
        "asu_modules_loaded": asu_modules_loaded, "train_seed_first": EPISODE_SEED_START,
        "train_seed_last": EPISODE_SEED_START + ADDITIONAL_GAMES - 1, **result,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded: {asu_modules_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
