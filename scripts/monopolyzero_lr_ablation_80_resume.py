"""Tests whether a lower optimizer LR prevents the post-80-game PPO policy
collapse documented in 030/031 (BUY_PROPERTY 98.5%+ -> 0.0% by game 128).
Resumes the hash-gated frozen 80-game champion (candidate_ppo_80.pt) for 16
MORE games under two LR conditions - A: lr=1e-4, B: lr=3e-5 (both below the
PPOAgent default 3e-4 used by every prior run) - identical seed_arg so both
conditions train on the SAME 16 episodes, isolating LR as the only
difference. LR is changed by mutating the loaded agent's own `opt.param_groups`
in this runner only - PPOAgent itself is never edited. Selection/confirmation
reuse 028/030/031's exact play_one_game/summarize wholesale, candidate vs
three frozen copies of the 80-game champion (not the untrained init or the
32-game checkpoint). See docs/EXPERIMENTS.md's 032 entry.
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
from monopolyzero_pure_ppo_learnability_gate import _full_actor_sha256  # noqa: E402

CHAMPION_PATH = common.REPO_ROOT / "artifacts" / "monopolyzero_pure_ppo_learnability_gate" / "candidate_ppo_80.pt"
EXPECTED_CHAMPION_CHECKPOINT_SHA256 = "e47c8d4559c1d74cfceffe114fa069a8e7a2845ba60af68be0b0ae4bd37e1dae"
TRAIN_SEED_START, ADDITIONAL_GAMES = 52000, 16
BASE_SEED_ARG = TRAIN_SEED_START - 81 + 1  # train(): episode_seed = seed_arg + absolute_game - 1
CONDITIONS = {"A_lr1e-4": 1e-4, "B_lr3e-5": 3e-5}
SELECTION_SEED_BASE, N_SELECTION_SEEDS = 52100, 8
CONFIRM_SEED_BASE, N_CONFIRM_SEEDS = 52200, 20
SELECTION_WIN_RATE_THRESHOLD = 0.25


def load_verified_champion():
    from monopoly_game_engine.agent_ppo import PPOAgent
    checkpoint_sha256 = screen._file_sha256(CHAMPION_PATH)
    if checkpoint_sha256 != EXPECTED_CHAMPION_CHECKPOINT_SHA256:
        raise SystemExit(f"STOP: champion checkpoint sha256 mismatch - got {checkpoint_sha256}, expected {EXPECTED_CHAMPION_CHECKPOINT_SHA256}. Refusing to reconstruct/retrain.")
    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.load(str(CHAMPION_PATH))
    agent.actor.eval()
    if agent.games_trained != 80:
        raise SystemExit(f"STOP: expected games_trained==80, got {agent.games_trained}")
    return agent


def train_candidate(lr: float, out_path: Path) -> dict:
    common.ensure_reference_on_path()
    from monopoly_game_engine.train import train as reference_train
    agent = load_verified_champion()
    for group in agent.opt.param_groups:
        group["lr"] = lr
    history = reference_train(agent, is_ppo=True, hybrid=False, n_games=ADDITIONAL_GAMES, log_every=1, seed=BASE_SEED_ARG)
    expected = list(range(TRAIN_SEED_START, TRAIN_SEED_START + ADDITIONAL_GAMES))
    actual = [BASE_SEED_ARG + g - 1 for g in history["games"]]
    if actual != expected:
        raise RuntimeError(f"Episode seed mismatch: got {actual}, expected {expected}")
    if agent.games_trained != 80 + ADDITIONAL_GAMES:
        raise SystemExit(f"STOP: training did not reach games_trained==96, got {agent.games_trained}")
    agent.actor.eval()
    agent.save(str(out_path))
    return {"path": str(out_path), "checkpoint_sha256": screen._file_sha256(out_path), "actor_sha256": _full_actor_sha256(agent.actor)}


def eval_candidate(checkpoint_path: Path, eval_seeds: list[int], baseline_actor, device) -> dict:
    from monopoly_game_engine.agent_ppo import PPOAgent
    candidate = PPOAgent(player_id=0, hybrid=False, device="cpu")
    candidate.load(str(checkpoint_path))
    candidate.actor.eval()
    games = [
        screen.play_one_game(game_id=gid, seed=s, candidate_actor=candidate.actor, baseline_actor=baseline_actor, focus_seat=seat, device=device, max_rounds=screen.MAX_ROUNDS)
        for gid, (s, seat) in enumerate(((s, seat) for s in eval_seeds for seat in range(screen.NUM_SEATS)), start=1)
    ]
    return screen.summarize(games)


def main() -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)
    train_seeds = list(range(TRAIN_SEED_START, TRAIN_SEED_START + ADDITIONAL_GAMES))
    selection_seeds, confirm_seeds = screen._seed_range(SELECTION_SEED_BASE, N_SELECTION_SEEDS), screen._seed_range(CONFIRM_SEED_BASE, N_CONFIRM_SEEDS)
    for seeds, ctx in ((train_seeds, "train"), (selection_seeds, "selection"), (confirm_seeds, "confirm")):
        ep.require_seed_scope(seeds, ep.SEED_CLASS_DEV, context=f"lr_ablation_80_resume.py ({ctx})")

    import torch
    device = torch.device("cpu")
    result: dict = {}
    started = time.perf_counter()
    with common.RssMonitor() as rss:
        champion = load_verified_champion()
        candidates = {name: train_candidate(lr, CHAMPION_PATH.parent / f"candidate_ppo_80_lr_ablation_{name}_96.pt") for name, lr in CONDITIONS.items()}
        result["candidates"] = candidates
        selection = {name: eval_candidate(Path(info["path"]), selection_seeds, champion.actor, device) for name, info in candidates.items()}
        result["selection"], result["selection_seeds"] = selection, selection_seeds
        best_name = max(selection, key=lambda n: (selection[n]["candidate_win_rate"], selection[n]["candidate_mean_net_worth"]))
        best = selection[best_name]

        if best["candidate_win_rate"] <= SELECTION_WIN_RATE_THRESHOLD:
            result["status"], result["champion_remains"] = "KILL_NO_CANDIDATE_ABOVE_THRESHOLD", "80-game"
        else:
            confirmation = eval_candidate(Path(candidates[best_name]["path"]), confirm_seeds, champion.actor, device)
            result["selected_candidate"], result["confirm_seeds"], result["confirmation"] = best_name, confirm_seeds, confirmation
            ci = confirmation["primary_seed_block_bootstrap_vs_25pct_null"]["win_rate_diff_from_null"]["ci_95"]
            if ci[0] > 0.0:
                result["status"], result["champion_remains"] = "GO_PROMOTE_LR_ABLATION_CANDIDATE", f"{best_name}-96-game"
            else:
                result["status"], result["champion_remains"] = "NOT_GO_KEEP_CHAMPION", "80-game"
    elapsed_s = time.perf_counter() - started

    asu_modules_loaded = common.loaded_asu_modules()
    payload = {"git_head_sha": git_head_sha, "elapsed_s": elapsed_s, "peak_rss_gib": rss.peak_gib, "asu_modules_loaded": asu_modules_loaded,
               "train_seed_first": TRAIN_SEED_START, "train_seed_last": TRAIN_SEED_START + ADDITIONAL_GAMES - 1, **result}
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded: {asu_modules_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
