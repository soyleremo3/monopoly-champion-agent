"""Tests for scripts/monopolyzero_common.py: guards, the decision-seed mix
(regression-tested to differ from the reference's formula), the dense
visit-target scatter's numeric correctness, and LocalFixedPolicy's
fallback behavior — all without needing a real game/model where possible.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "monopolyzero_common.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_common", MODULE_PATH)
common = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_common"] = common
_spec.loader.exec_module(common)


# ── PYTHONHASHSEED guard ─────────────────────────────────────────────────


@pytest.mark.parametrize("value", [None, "", "1", "2", "random"])
def test_guard_rejects_unpinned_hash_seed(value, monkeypatch):
    if value is None:
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    else:
        monkeypatch.setenv("PYTHONHASHSEED", value)
    with pytest.raises(SystemExit) as excinfo:
        common.require_pinned_hash_seed("some_script.py")
    assert "PYTHONHASHSEED=0" in str(excinfo.value)


def test_guard_accepts_pinned_hash_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    common.require_pinned_hash_seed("some_script.py")


# ── clean-git-tree guard (mocked subprocess) ─────────────────────────────


class _FakeCompletedProcess:
    def __init__(self, stdout: str):
        self.stdout = stdout


def test_clean_tree_guard_returns_head_sha_when_clean(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["git", "status"]:
            return _FakeCompletedProcess(stdout="")
        if args[:2] == ["git", "rev-parse"]:
            return _FakeCompletedProcess(stdout="deadbeef1234567890deadbeef1234567890dead\n")
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(common.subprocess, "run", fake_run)
    sha = common.require_clean_git_tree("some_script.py")
    assert sha == "deadbeef1234567890deadbeef1234567890dead"
    assert calls == [["git", "status", "--porcelain"], ["git", "rev-parse", "HEAD"]]


def test_clean_tree_guard_raises_when_dirty(monkeypatch):
    def fake_run(args, **kwargs):
        if args[:2] == ["git", "status"]:
            return _FakeCompletedProcess(stdout=" M scripts/monopolyzero_common.py\n")
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(common.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as excinfo:
        common.require_clean_git_tree("some_script.py")
    assert "not clean" in str(excinfo.value)


# ── ASU-module sys.modules guard ─────────────────────────────────────────


def test_asu_module_guard_detects_loaded_modules(monkeypatch):
    sentinel_names = ["ASU_FROZEN_TEACHER", "ASU_FROZEN_TEACHER.core"]
    for name in sentinel_names:
        monkeypatch.setitem(sys.modules, name, object())
    assert common.loaded_asu_modules() == sentinel_names


def test_asu_module_guard_clean_when_absent():
    for name in list(sys.modules):
        assert not (name == "ASU_FROZEN_TEACHER" or name.startswith("ASU_FROZEN_TEACHER."))
    assert common.loaded_asu_modules() == []


# ── module itself imports no ASU-coupled reference modules ──────────────


def test_module_does_not_import_adapters_arena_or_training():
    source = MODULE_PATH.read_text(encoding="utf-8")
    import_lines = [
        line for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert hits == [], f"found forbidden ASU-coupled module import(s): {hits}"


# ── decision-seed mixing: deterministic, and provably not the reference's ──
# formula (seed * 1_000_003 + step * 17 + player_id from arena.py) ──────────


def test_mix_decision_seed_is_deterministic():
    a = common._mix_decision_seed(42, 7, 2)
    b = common._mix_decision_seed(42, 7, 2)
    assert a == b


def test_mix_decision_seed_varies_with_each_input():
    base = common._mix_decision_seed(42, 7, 2)
    assert common._mix_decision_seed(43, 7, 2) != base
    assert common._mix_decision_seed(42, 8, 2) != base
    assert common._mix_decision_seed(42, 7, 3) != base


@pytest.mark.parametrize("seed,turn,seat", [(42, 7, 2), (501, 0, 0), (503, 250, 3)])
def test_mix_decision_seed_differs_from_reference_formula(seed, turn, seat):
    reference_formula_value = seed * 1_000_003 + turn * 17 + seat
    assert common._mix_decision_seed(seed, turn, seat) != reference_formula_value


def test_mix_decision_seed_is_a_valid_nonnegative_int32ish_value():
    value = common._mix_decision_seed(503, 999, 3)
    assert isinstance(value, int)
    assert 0 <= value <= 0x7FFFFFFF


# ── dense visit-target scatter: numeric correctness ──────────────────────


def test_scatter_visit_targets_normalizes_sparse_counts():
    import torch

    actions = torch.tensor([[5, 9, 0]], dtype=torch.long)
    counts = torch.tensor([[3.0, 1.0, 0.0]], dtype=torch.float32)
    lengths = torch.tensor([2], dtype=torch.long)
    dense = common._scatter_visit_targets(actions, counts, lengths, num_actions=12)
    assert dense.shape == (1, 12)
    assert dense[0, 5].item() == pytest.approx(0.75)
    assert dense[0, 9].item() == pytest.approx(0.25)
    assert dense[0].sum().item() == pytest.approx(1.0)
    untouched = [i for i in range(12) if i not in (5, 9)]
    assert all(dense[0, i].item() == 0.0 for i in untouched)


def test_dense_target_cross_entropy_is_finite_with_masked_inf_logits():
    """Regression test: a first version of this function computed
    targets * log_softmax(logits) directly, which produced NaN whenever a
    masked (illegal) position had logit=-inf and target=0, since
    0 * -inf = NaN in IEEE754. This reproduces that exact shape (most
    positions masked out, matching a real legal_mask over a 2958-action
    space) and asserts the loss is finite."""
    import torch

    num_actions = 12
    logits = torch.full((1, num_actions), float("-inf"))
    logits[0, 3] = 0.5
    logits[0, 7] = 1.5
    targets = torch.zeros((1, num_actions))
    targets[0, 3] = 1.0

    loss = common._dense_target_cross_entropy(logits, targets)
    assert torch.isfinite(loss).item()


def test_dense_target_cross_entropy_matches_hand_computed_value():
    import torch

    logits = torch.tensor([[0.0, float("-inf"), 1.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    loss = common._dense_target_cross_entropy(logits, targets)
    expected = -torch.log_softmax(logits, dim=1)[0, 0]
    assert loss.item() == pytest.approx(expected.item())


def test_scatter_visit_targets_raises_on_all_zero_row():
    import torch

    actions = torch.zeros((1, 3), dtype=torch.long)
    counts = torch.zeros((1, 3), dtype=torch.float32)
    lengths = torch.zeros((1,), dtype=torch.long)
    with pytest.raises(ValueError):
        common._scatter_visit_targets(actions, counts, lengths, num_actions=12)


# ── LocalFixedPolicy: fallback behavior with a fake agent, no real engine ──


class _FakeEnv:
    def __init__(self, legal):
        self._legal = legal

    def get_allowed_actions(self, seat):
        return self._legal


def _make_fake_agent_class(fixed_action):
    class _FakeAgent:
        def __init__(self, player_id):
            self.player_id = player_id

        def choose_action(self, env):
            return fixed_action

    return _FakeAgent


def test_local_fixed_policy_passes_through_legal_action():
    common.ensure_reference_on_path()
    policy = common.LocalFixedPolicy(_make_fake_agent_class(fixed_action=7))
    env = _FakeEnv(legal=(3, 7, 9))
    action = policy.choose(env, seat=0, decision_seed=1)
    assert action == 7
    assert policy.fallback_count == 0


def test_local_fixed_policy_substitutes_and_counts_illegal_action():
    from monopoly_bench.engine import ActionType

    common.ensure_reference_on_path()
    policy = common.LocalFixedPolicy(_make_fake_agent_class(fixed_action=999))
    legal = (3, 9, int(ActionType.END_TURN))
    env = _FakeEnv(legal=legal)
    action = policy.choose(env, seat=0, decision_seed=1)
    assert action == int(ActionType.END_TURN)
    assert policy.fallback_count == 1


def test_local_fixed_policy_falls_back_to_first_legal_without_end_turn():
    common.ensure_reference_on_path()
    policy = common.LocalFixedPolicy(_make_fake_agent_class(fixed_action=999))
    env = _FakeEnv(legal=(11, 22, 33))
    action = policy.choose(env, seat=0, decision_seed=1)
    assert action == 11
    assert policy.fallback_count == 1


# ── POLICY_ONLY: direct policy-head inference, no search ─────────────────


class _FakeEnvWithState:
    def __init__(self, legal, state="state"):
        self._legal = legal
        self._state = state

    def get_allowed_actions(self, seat):
        return self._legal

    def _get_state(self, seat):
        return self._state


class _FakeGame:
    def __init__(self, env):
        self.env = env


class _FakeModel:
    """predict() returns a higher prior on the last legal action, so the
    argmax is deterministic and easy to assert on."""

    def __init__(self):
        self.calls = []

    def predict(self, state, legal_actions, actor_id):
        self.calls.append((state, tuple(legal_actions), actor_id))
        legal = tuple(legal_actions)
        priors = {action: 0.1 for action in legal}
        priors[legal[-1]] = 0.9
        return priors, [0.5]


def test_build_local_policy_only_has_policy_only_kind():
    policy = common.build_local_policy_only(_FakeModel())
    assert policy.kind == "policy_only"


def test_build_local_policy_only_picks_legal_argmax():
    model = _FakeModel()
    policy = common.build_local_policy_only(model)
    game = _FakeGame(_FakeEnvWithState(legal=(3, 7, 9)))
    result = policy.choose(game, seat=2, decision_seed=123)
    assert result.chosen_action == 9
    assert result.visits == {9: 1}
    assert result.root_value == [0.5]
    assert result.simulations == 0
    assert result.latency_s >= 0.0
    assert model.calls == [("state", (3, 7, 9), 2)]


def test_build_local_policy_only_is_deterministic_regardless_of_decision_seed():
    model = _FakeModel()
    policy = common.build_local_policy_only(model)
    game = _FakeGame(_FakeEnvWithState(legal=(1, 2, 3)))
    first = policy.choose(game, seat=0, decision_seed=1).chosen_action
    second = policy.choose(game, seat=0, decision_seed=999999).chosen_action
    assert first == second == 3


# ── _invoke_policy: normalizes search/policy_only/fixed return shapes ────


class _FakeFixedKindPolicy:
    kind = "fixed"

    def choose(self, game, seat, decision_seed):
        return 42


def test_invoke_policy_normalizes_fixed_kind_to_plain_int_with_no_latency():
    action, latency_s, result, kind = common._invoke_policy(
        _FakeFixedKindPolicy(), game=None, seat=0, decision_seed=0
    )
    assert (action, latency_s, result, kind) == (42, None, None, "fixed")


def test_invoke_policy_normalizes_policy_only_kind_to_result_object():
    model = _FakeModel()
    policy = common.build_local_policy_only(model)
    game = _FakeGame(_FakeEnvWithState(legal=(1, 2)))
    action, latency_s, result, kind = common._invoke_policy(policy, game, seat=0, decision_seed=0)
    assert kind == "policy_only"
    assert action == result.chosen_action == 2
    assert latency_s == result.latency_s


# ── play_local_game: shadow_policy hook signature ─────────────────────────


def test_play_local_game_accepts_shadow_policy_kwargs():
    import inspect

    params = inspect.signature(common.play_local_game).parameters
    assert "shadow_policy" in params
    assert params["shadow_policy"].default is None
    assert "shadow_seats" in params
    assert params["shadow_seats"].default is None


def test_local_game_outcome_has_shadow_decisions_field():
    outcome = common.LocalGameOutcome(completed=True, winner=0, decisions=1)
    assert outcome.shadow_decisions == []
