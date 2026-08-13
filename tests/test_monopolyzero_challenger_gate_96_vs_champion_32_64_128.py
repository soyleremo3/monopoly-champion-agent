"""Tests for scripts/monopolyzero_challenger_gate_96_vs_champion_32_64_128.py
(experiment 034, pre-registered - NOT yet run against real games):
- source-level ASU-import guard
- CHECKPOINTS dict structure (5 entries, all fields present, hash strings
  well-formed) and PROMOTION seed range 50020-50039 (fresh, disjoint from
  033's already-consumed 50000-50019, from DEV, and from FINAL_BLIND)
- load_and_verify(): hash-mismatch STOP behavior (SystemExit, not a silent
  retrain), tiny real-engine smoke
- decide_verdict(): the pre-registered decision rule as a pure function,
  validated against synthetic CIs (GO / KILL via vs80 / KILL via aggregate /
  INCONCLUSIVE) - this is the only way to validate the rule before this
  runner is ever executed, per this experiment's pre-registration.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_challenger_gate_96_vs_champion_32_64_128.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import evaluation_protocol as ep  # noqa: E402
import monopolyzero_common as common  # noqa: E402

_spec = importlib.util.spec_from_file_location("monopolyzero_challenger_gate_96_vs_champion_32_64_128", SCRIPT)
module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_challenger_gate_96_vs_champion_32_64_128"] = module
_spec.loader.exec_module(module)


def test_source_never_imports_asu():
    import_lines = [
        line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("ASU_FROZEN_TEACHER", "monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert not hits, hits


# ── checkpoint registry ──────────────────────────────────────────────────


def test_checkpoints_dict_has_five_entries_with_well_formed_hashes():
    assert set(module.CHECKPOINTS) == {"challenger_96", "champion_80", "opponent_32", "opponent_64", "opponent_128"}
    for name, (filename, checkpoint_sha256, actor_sha256) in module.CHECKPOINTS.items():
        assert filename.endswith(".pt")
        assert len(checkpoint_sha256) == 64 and all(c in "0123456789abcdef" for c in checkpoint_sha256)
        assert actor_sha256 is not None
        assert len(actor_sha256) == 64 and all(c in "0123456789abcdef" for c in actor_sha256)


def test_challenger_hash_matches_032s_logged_a96_candidate():
    filename, checkpoint_sha256, actor_sha256 = module.CHECKPOINTS["challenger_96"]
    assert filename == "candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt"
    assert checkpoint_sha256 == "78585ed4e2400d024633ee2878d8b88243f7f4a9f498d9019a73e44b3a830f51"
    assert actor_sha256 == "2bd1e9bad3d6e0100e033507f72f15b5088d8f8fb2f04dba1ae9d759b5a34a40"


# ── seed pool ─────────────────────────────────────────────────────────────


def test_promotion_seeds_are_fresh_and_disjoint_from_033_and_other_pools():
    seeds = set(range(module.PROMOTION_SEED_BASE, module.PROMOTION_SEED_BASE + module.N_PROMOTION_SEEDS))
    assert seeds == set(range(50020, 50040))
    already_consumed_by_033 = set(range(50000, 50020))
    assert seeds.isdisjoint(already_consumed_by_033)
    assert seeds.isdisjoint(ep.DEV_SEEDS)
    assert seeds.isdisjoint(ep.FINAL_BLIND_SEEDS)
    for seed in seeds:
        assert ep.classify_seed(seed) == ep.SEED_CLASS_PROMOTION


def test_promotion_seed_scope_guard_accepts_034_seeds():
    seeds = list(range(module.PROMOTION_SEED_BASE, module.PROMOTION_SEED_BASE + module.N_PROMOTION_SEEDS))
    ep.require_seed_scope(seeds, ep.SEED_CLASS_PROMOTION, context="test")  # must not raise


# ── load_and_verify(): hash-gate STOP behavior ───────────────────────────


def test_load_and_verify_raises_on_checkpoint_sha_mismatch(tmp_path):
    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    path = tmp_path / "checkpoint.pt"
    agent.save(str(path))

    with pytest.raises(SystemExit, match="checkpoint sha256 mismatch"):
        module.load_and_verify(path, "0" * 64, None)


def test_load_and_verify_raises_on_actor_sha_mismatch(tmp_path):
    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    path = tmp_path / "checkpoint.pt"
    agent.save(str(path))
    checkpoint_sha256 = module.screen._file_sha256(path)

    with pytest.raises(SystemExit, match="actor sha256 mismatch"):
        module.load_and_verify(path, checkpoint_sha256, "0" * 64)


def test_load_and_verify_passes_and_returns_agent_when_hashes_match(tmp_path):
    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent
    from monopolyzero_pure_ppo_learnability_gate import _full_actor_sha256

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    path = tmp_path / "checkpoint.pt"
    agent.save(str(path))
    checkpoint_sha256 = module.screen._file_sha256(path)
    actor_sha256 = _full_actor_sha256(agent.actor)

    loaded, verified = module.load_and_verify(path, checkpoint_sha256, actor_sha256)
    assert verified == {"checkpoint_sha256": checkpoint_sha256, "actor_sha256": actor_sha256}
    assert loaded.hybrid is False


# ── decide_verdict(): the pre-registered rule, pure-function validation ──


def test_decide_verdict_go_when_all_three_conditions_hold():
    verdict = module.decide_verdict(
        vs80_ci=[0.05, 0.30], agg_ci=[0.10, 0.25], family_upper_bounds=[0.30, 0.20, 0.25, 0.15],
    )
    assert verdict == "CHALLENGER_PROMOTION_GO"


def test_decide_verdict_kill_when_vs80_ci_upper_bound_at_or_below_zero():
    verdict = module.decide_verdict(
        vs80_ci=[-0.10, 0.0], agg_ci=[0.05, 0.20], family_upper_bounds=[0.0, 0.20, 0.25, 0.15],
    )
    assert verdict == "CHALLENGER_PROMOTION_KILL_KEEP_80_GAME_CHAMPION"


def test_decide_verdict_kill_when_aggregate_ci_upper_bound_at_or_below_zero():
    verdict = module.decide_verdict(
        vs80_ci=[0.05, 0.30], agg_ci=[-0.05, -0.01], family_upper_bounds=[0.30, 0.20, 0.25, 0.15],
    )
    assert verdict == "CHALLENGER_PROMOTION_KILL_KEEP_80_GAME_CHAMPION"


def test_decide_verdict_inconclusive_when_neither_go_nor_kill_condition_holds():
    # aggregate lower bound fails the GO check, but its upper bound is still
    # positive, so this is not the KILL condition either.
    verdict = module.decide_verdict(
        vs80_ci=[0.05, 0.30], agg_ci=[-0.02, 0.15], family_upper_bounds=[0.30, 0.20, 0.25, 0.15],
    )
    assert verdict == "CHALLENGER_PROMOTION_INCONCLUSIVE_KEEP_80_GAME_CHAMPION"


def test_decide_verdict_inconclusive_when_one_family_upper_bound_at_or_below_zero_but_no_kill_trigger():
    # A family CI upper bound <= 0 blocks GO (condition 3), but since
    # neither vs80 nor aggregate upper bound is <= 0, this is not KILL.
    verdict = module.decide_verdict(
        vs80_ci=[0.05, 0.30], agg_ci=[0.10, 0.25], family_upper_bounds=[0.30, 0.20, -0.01, 0.15],
    )
    assert verdict == "CHALLENGER_PROMOTION_INCONCLUSIVE_KEEP_80_GAME_CHAMPION"
