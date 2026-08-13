"""Experiment 034 (pre-registered, not yet run): PROMOTION-scoped
challenger gate for 032's frozen 96-game A_lr1e-4 checkpoint
(candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt) against the current
80-game champion PLUS the same 32/64/128-game opponent families 033
already used - the minimal adaptation of 033's promotion runner
(scripts/monopolyzero_promotion_gate_80_vs_32_64_128.py): same
play_one_game/summarize path, same hash-gate pattern, one more opponent
family (the 80-game champion itself), a different candidate, and a
stricter decision rule (see main()). Fresh PROMOTION seeds 50020-50039 -
reserved for this experiment, not yet consumed as of pre-registration.
See docs/EXPERIMENTS.md's 034 entry for the full pre-registered design
and decision rule, written BEFORE this runner is ever executed.
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
PROMOTION_SEED_BASE, N_PROMOTION_SEEDS = 50020, 20

# (name, checkpoint filename, expected checkpoint sha256, expected actor sha256)
CHECKPOINTS = {
    "challenger_96": ("candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt",
                       "78585ed4e2400d024633ee2878d8b88243f7f4a9f498d9019a73e44b3a830f51",
                       "2bd1e9bad3d6e0100e033507f72f15b5088d8f8fb2f04dba1ae9d759b5a34a40"),
    "champion_80": ("candidate_ppo_80.pt", "e47c8d4559c1d74cfceffe114fa069a8e7a2845ba60af68be0b0ae4bd37e1dae",
                     "7442f99e51619b7f6b53bc662e74d360e0a1f9cf2365b8dcb003d5a0cc3eda11"),
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


def decide_verdict(vs80_ci: list[float], agg_ci: list[float], family_upper_bounds: list[float]) -> str:
    """Pre-registered decision rule (docs/EXPERIMENTS.md's 034 entry) as a
    pure function so it can be validated against synthetic CIs before this
    runner is ever executed against real games."""
    if vs80_ci[0] > 0.0 and agg_ci[0] > 0.0 and all(ub > 0.0 for ub in family_upper_bounds):
        return "CHALLENGER_PROMOTION_GO"
    if vs80_ci[1] <= 0.0 or agg_ci[1] <= 0.0:
        return "CHALLENGER_PROMOTION_KILL_KEEP_80_GAME_CHAMPION"
    return "CHALLENGER_PROMOTION_INCONCLUSIVE_KEEP_80_GAME_CHAMPION"


def main() -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)
    common.ensure_reference_on_path()

    seeds = screen._seed_range(PROMOTION_SEED_BASE, N_PROMOTION_SEEDS)
    ep.require_seed_scope(seeds, ep.SEED_CLASS_PROMOTION, context="challenger_gate_96_vs_champion_32_64_128.py")

    agents, hash_report = {}, {}
    for key, (filename, ck_sha, actor_sha) in CHECKPOINTS.items():
        agent, verified = load_and_verify(ARTIFACT_DIR / filename, ck_sha, actor_sha)
        agents[key] = agent
        hash_report[key] = verified

    import torch

    device = torch.device("cpu")
    families = {"vs_80": agents["champion_80"], "vs_32": agents["opponent_32"],
                "vs_64": agents["opponent_64"], "vs_128": agents["opponent_128"]}
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
                        game_id=game_id, seed=seed, candidate_actor=agents["challenger_96"].actor,
                        baseline_actor=opponent.actor, focus_seat=focus_seat, device=device,
                        max_rounds=screen.MAX_ROUNDS,
                    ))
            lineups[family_name] = games
    elapsed_s = time.perf_counter() - started

    family_summaries = {name: screen.summarize(games) for name, games in lineups.items()}
    all_games = lineups["vs_80"] + lineups["vs_32"] + lineups["vs_64"] + lineups["vs_128"]
    aggregate = screen.summarize(all_games)

    def _ci(summary: dict) -> list[float]:
        return summary["primary_seed_block_bootstrap_vs_25pct_null"]["win_rate_diff_from_null"]["ci_95"]

    vs80_ci = _ci(family_summaries["vs_80"])
    agg_ci = _ci(aggregate)
    family_upper_bounds = [_ci(s)[1] for s in family_summaries.values()]
    verdict = decide_verdict(vs80_ci, agg_ci, family_upper_bounds)

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
