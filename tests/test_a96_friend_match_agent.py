"""Tests for scripts/a96_friend_match_agent.py:
- pinned A96 constants match the exact recorded 034 provenance
- checkpoint resolution order (explicit arg > env var > default path)
- fail-closed on missing checkpoint / wrong checkpoint hash (no real A96
  checkpoint needed for these - any freshly-constructed checkpoint that
  doesn't match the pinned hashes proves the gate rejects it)
- against the REAL local A96 checkpoint (skipped cleanly if absent, e.g.
  a fresh Colab clone): wrapper act() agrees exactly with a direct
  FrozenPurePPOPolicy built the same way, and never returns an illegal
  action across several scenarios
- source-level guards: no ASU import, no BUY/fixed-rule override logic
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "a96_friend_match_agent.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import monopolyzero_common as common  # noqa: E402
from frozen_ppo_inference import ChecksumMismatchError, FrozenPurePPOPolicy  # noqa: E402

# ChecksumMismatchError is captured HERE, at collection time, from the SAME
# sys.modules["frozen_ppo_inference"] entry that a96_friend_match_agent.py's
# own exec_module (right below) will bind to internally - other test files
# in this project reload "frozen_ppo_inference" via their own
# importlib.util.spec_from_file_location(..., "frozen_ppo_inference", ...),
# which overwrites sys.modules under that name. Re-importing the exception
# class inside a test FUNCTION (at run time, after all files have already
# collected) would silently grab a LATER reload's class object - a
# different identity than the one this module's own code actually raises -
# so pytest.raises() would never match it. Capturing it once here, in the
# same statement block as the module.exec_module() call below, keeps both
# bound to the identical class.
_spec = importlib.util.spec_from_file_location("a96_friend_match_agent", SCRIPT)
module = importlib.util.module_from_spec(_spec)
sys.modules["a96_friend_match_agent"] = module
_spec.loader.exec_module(module)


def test_source_never_imports_asu():
    import_lines = [
        line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("ASU_FROZEN_TEACHER", "monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert not hits, hits


def test_source_has_no_buy_override_or_hybrid_compat_logic():
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = ("fixed_buy_decision", "HYBRID_COMPAT", "build_local_hybrid_compat_policy", "hybrid=True", "safety_breakdown")
    hits = [name for name in forbidden if name in source]
    assert hits == [], hits


# ── pinned constants ──────────────────────────────────────────────────────


def test_pinned_constants_match_034_provenance():
    assert module.A96_CHECKPOINT_FILENAME == "candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt"
    assert module.A96_CHECKPOINT_SHA256 == "78585ed4e2400d024633ee2878d8b88243f7f4a9f498d9019a73e44b3a830f51"
    assert module.A96_ACTOR_SHA256 == "2bd1e9bad3d6e0100e033507f72f15b5088d8f8fb2f04dba1ae9d759b5a34a40"
    assert module.A96_REFERENCE_SUBMODULE_SHA == "afd9205761317e196d77f679921c35fb04c7ab96"
    assert len(module.A96_CHECKPOINT_SHA256) == 64
    assert len(module.A96_ACTOR_SHA256) == 64


# ── checkpoint resolution order ─────────────────────────────────────────


def test_resolve_checkpoint_path_explicit_argument_wins(monkeypatch):
    monkeypatch.setenv(module.CHECKPOINT_PATH_ENV_VAR, "/env/path.pt")
    resolved = module.resolve_checkpoint_path("/explicit/path.pt")
    assert resolved == Path("/explicit/path.pt")


def test_resolve_checkpoint_path_env_var_used_when_no_explicit_arg(monkeypatch):
    monkeypatch.setenv(module.CHECKPOINT_PATH_ENV_VAR, "/env/path.pt")
    resolved = module.resolve_checkpoint_path(None)
    assert resolved == Path("/env/path.pt")


def test_resolve_checkpoint_path_default_when_nothing_set(monkeypatch):
    monkeypatch.delenv(module.CHECKPOINT_PATH_ENV_VAR, raising=False)
    resolved = module.resolve_checkpoint_path(None)
    assert resolved == module.DEFAULT_CHECKPOINT_PATH


# ── fail-closed behavior ─────────────────────────────────────────────────


def test_agent_construction_fails_closed_on_missing_checkpoint(tmp_path):
    missing = tmp_path / "nope.pt"
    with pytest.raises(FileNotFoundError):
        module.A96FriendMatchAgent(checkpoint_path=missing)


def test_agent_construction_fails_closed_on_wrong_checkpoint_hash(tmp_path):
    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    path = tmp_path / "wrong.pt"
    agent.save(str(path))

    with pytest.raises(ChecksumMismatchError, match="checkpoint file SHA-256 mismatch"):
        module.A96FriendMatchAgent(checkpoint_path=path)


# ── against the REAL local A96 checkpoint (skips cleanly if absent) ─────


def _skip_if_no_real_checkpoint():
    if not module.DEFAULT_CHECKPOINT_PATH.is_file():
        pytest.skip(f"real A96 checkpoint not present locally (gitignored artifact): {module.DEFAULT_CHECKPOINT_PATH}")


def test_wrapper_action_agrees_with_direct_frozen_policy():
    _skip_if_no_real_checkpoint()
    common.ensure_reference_on_path()
    from monopoly_bench.engine import SharedGame

    agent = module.A96FriendMatchAgent()
    direct_policy = FrozenPurePPOPolicy.from_checkpoint(
        module.DEFAULT_CHECKPOINT_PATH,
        expected_checkpoint_sha256=module.A96_CHECKPOINT_SHA256,
        expected_actor_sha256=module.A96_ACTOR_SHA256,
    )

    game = SharedGame.new(555555, 200)
    seat = game.env.whose_turn()
    legal = tuple(game.env.get_allowed_actions(seat))
    state = game.env._get_state(seat)

    wrapper_action = agent.act(state, legal)
    direct_action = direct_policy.act(state, legal)
    assert wrapper_action == direct_action


def test_wrapper_action_always_legal_across_multiple_scenarios():
    _skip_if_no_real_checkpoint()
    common.ensure_reference_on_path()
    from monopoly_bench.engine import SharedGame

    agent = module.A96FriendMatchAgent()
    for seed in (111111, 222222, 333333):
        game = SharedGame.new(seed, 200)
        seat = game.env.whose_turn()
        legal = tuple(game.env.get_allowed_actions(seat))
        state = game.env._get_state(seat)
        action = agent.act(state, legal)
        assert action in legal


def test_env_var_checkpoint_path_used_by_real_agent(monkeypatch):
    _skip_if_no_real_checkpoint()
    monkeypatch.setenv(module.CHECKPOINT_PATH_ENV_VAR, str(module.DEFAULT_CHECKPOINT_PATH))
    agent = module.A96FriendMatchAgent()
    assert agent.checkpoint_path == module.DEFAULT_CHECKPOINT_PATH
