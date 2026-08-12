"""First PROMOTION/generalization gate for the 80-game DEV champion
(031's selected checkpoint, candidate_ppo_80.pt). Tests whether it beats
three DIFFERENT frozen PPO opponent families (32/64/128-game checkpoints,
all from 027/031), not just the single 32-game predecessor it was selected
against. Reuses 028/031's exact play_one_game/summarize wholesale - no new
evaluation framework. PROMOTION seeds 50000-50019 (fresh, never used).
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

ARTIFACT_DIR = common.REPO_ROOT / "artifacts" / "monopolyzero_pure_ppo_learnability_gate"
PROMOTION_SEED_BASE, N_PROMOTION_SEEDS = 50000, 20

# (name, checkpoint filename, expected checkpoint sha256, expected actor sha256)
CHECKPOINTS = {
    "champion_80": ("candidate_ppo_80.pt", "e47c8d4559c1d74cfceffe114fa069a8e7a2845ba60af68be0b0ae4bd37e1dae", None),
    "opponent_32": ("candidate_ppo.pt", "85194f337182c2d698519966cb8c19d3c1e701102b5fb280cd69d4c23ed8a113",
                     "e6c142d143b5430d02b37cd3be34fa39a7d8d2f282bdb1eda8677a6109861b5b"),
    "opponent_64": ("candidate_ppo_64.pt", "9dcd29045c5fa1d9c0f853f109a78b347b99ef6c2ef428d39066ef7d97ded2d1",
                     "df7114bed6598df0137a540c3485c945ca8ce3b14bc771b7d5a3bc109b610c6c"),
    "opponent_128": ("candidate_ppo_128.pt", "78be2735b14c0563e2c558a8a56cfc20c30d05b225acde29d31d1a2c0975fbcb",
                      "ad4e9a6d066aab3b2ff5c8414b40669cae51b4bde6d49c42a1ce8128985dd224"),
}


def load_and_verify(path: Path, expected_checkpoint_sha256: str, expected_actor_sha256: str | None):
    from monopoly_game_engine.agent_ppo import PPOAgent

    if not path.is_file():
        raise SystemExit(f"STOP: checkpoint missing at {path}")
    checkpoint_sha256 = screen._file_sha256(path)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise SystemExit(f"STOP: {path.name} checkpoint sha256 mismatch - got {checkpoint_sha256}, "
                          f"expected {expected_checkpoint_sha256}. Refusing to reconstruct/retrain.")
    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.load(str(path))
    agent.actor.eval()
    if agent.hybrid is not False or bool(agent.fixed_action_mask.any()):
        raise RuntimeError(f"{path.name}: not a pure hybrid=False PPOAgent - refusing to run")
    from monopolyzero_pure_ppo_learnability_gate import _full_actor_sha256
    actor_sha256 = _full_actor_sha256(agent.actor)
    if expected_actor_sha256 is not None and actor_sha256 != expected_actor_sha256:
        raise SystemExit(f"STOP: {path.name} actor sha256 mismatch - got {actor_sha256}, "
                          f"expected {expected_actor_sha256}. Refusing to reconstruct/retrain.")
    return agent, {"checkpoint_sha256": checkpoint_sha256, "actor_sha256": actor_sha256}


def main() -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)
    common.ensure_reference_on_path()

    seeds = screen._seed_range(PROMOTION_SEED_BASE, N_PROMOTION_SEEDS)
    ep.require_seed_scope(seeds, ep.SEED_CLASS_PROMOTION, context="promotion_gate_80_vs_32_64_128.py")

    agents, hash_report = {}, {}
    for key, (filename, ck_sha, actor_sha) in CHECKPOINTS.items():
        agent, verified = load_and_verify(ARTIFACT_DIR / filename, ck_sha, actor_sha)
        agents[key] = agent
        hash_report[key] = verified

    import torch

    device = torch.device("cpu")
    families = {"A_vs_32": agents["opponent_32"], "B_vs_64": agents["opponent_64"], "C_vs_128": agents["opponent_128"]}
    lineups: dict[str, list[dict]] = {}
    game_id = 0
    started = time.perf_counter()
    with common.RssMonitor() as rss:
        for family_name, opponent in families.items():
            games = []
            for seed in seeds:
                for focus_seat in range(screen.NUM_SEATS):
                    game_id += 1
                    games.append(screen.play_one_game(
                        game_id=game_id, seed=seed, candidate_actor=agents["champion_80"].actor,
                        baseline_actor=opponent.actor, focus_seat=focus_seat, device=device,
                        max_rounds=screen.MAX_ROUNDS,
                    ))
            lineups[family_name] = games
    elapsed_s = time.perf_counter() - started

    family_summaries = {name: screen.summarize(games) for name, games in lineups.items()}
    all_games = lineups["A_vs_32"] + lineups["B_vs_64"] + lineups["C_vs_128"]
    aggregate = screen.summarize(all_games)

    agg_ci = aggregate["primary_seed_block_bootstrap_vs_25pct_null"]["win_rate_diff_from_null"]["ci_95"]
    family_upper_bounds = [
        s["primary_seed_block_bootstrap_vs_25pct_null"]["win_rate_diff_from_null"]["ci_95"][1]
        for s in family_summaries.values()
    ]
    if agg_ci[0] > 0.0 and all(ub > 0.0 for ub in family_upper_bounds):
        verdict = "PROMOTION_GO"
    elif agg_ci[1] <= 0.0:
        verdict = "PROMOTION_KILL"
    else:
        verdict = "PROMOTION_INCONCLUSIVE"

    asu_modules_loaded = common.loaded_asu_modules()
    payload = {
        "git_head_sha": git_head_sha, "elapsed_s": elapsed_s, "peak_rss_gib": rss.peak_gib,
        "asu_modules_loaded": asu_modules_loaded, "promotion_seeds": seeds,
        "hash_verification": hash_report, "lineup_summaries": family_summaries,
        "aggregate_summary": aggregate, "verdict": verdict,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded: {asu_modules_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
