"""Tests for scripts/frozen_ppo_inference.py - the submission-safe frozen
pure-PPO inference layer.

Coverage:
- source-level + runtime ASU-import guards
- import-time side effects (fresh subprocess: no torch/numpy/reference
  import merely from `import frozen_ppo_inference`)
- exact agreement with the existing evaluation harness's masked-argmax
  policy (scripts/monopolyzero_pure_ppo_strength_screen.py::
  build_masked_argmax_policy) across many randomized legal-action-mask
  scenarios - THE most important test here
- determinism (same inputs twice -> same output)
- never selects an illegal action, across many randomized scenarios
- checkpoint-file hash gate: rejects a corrupted/wrong checkpoint
- actor-source hash gate: rejects on a source-hash mismatch
- rejects a hybrid/fixed-rule checkpoint (HYBRID_COMPAT-style override
  path), never silently falls back to it
- torch.argmax's own tie-break behavior (empirically verified, not assumed)
- FrozenPurePPOPolicy.act(): legality guarantee, determinism, decision trace
- OfficialSubmissionAdapter: explicit TBD stub, raises on use
"""

from __future__ import annotations

import importlib.util
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "frozen_ppo_inference.py"
STRENGTH_SCREEN_SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_pure_ppo_strength_screen.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import monopolyzero_common as common  # noqa: E402

_spec = importlib.util.spec_from_file_location("frozen_ppo_inference", SCRIPT)
module = importlib.util.module_from_spec(_spec)
sys.modules["frozen_ppo_inference"] = module
_spec.loader.exec_module(module)

_ss_spec = importlib.util.spec_from_file_location(
    "monopolyzero_pure_ppo_strength_screen_for_equivalence_test", STRENGTH_SCREEN_SCRIPT
)
strength_screen = importlib.util.module_from_spec(_ss_spec)
sys.modules[_ss_spec.name] = strength_screen
_ss_spec.loader.exec_module(strength_screen)


# ── source-level + runtime ASU guards ───────────────────────────────────


def test_source_never_imports_asu():
    import_lines = [
        line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = (
        "ASU_FROZEN_TEACHER", "monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training",
    )
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert not hits, hits


def test_no_asu_modules_loaded_after_import_and_use(tmp_path):
    """Runtime check (not just source grep): after importing this module
    AND exercising its loader/act path against a real checkpoint, no
    ASU_FROZEN_TEACHER module is present in sys.modules."""
    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    ckpt_path = tmp_path / "ckpt.pt"
    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.save(str(ckpt_path))

    expected_checkpoint_sha256 = module.file_sha256(ckpt_path)
    expected_actor_source_sha256 = module.actor_source_sha256()

    policy = module.FrozenPurePPOPolicy.from_checkpoint(
        ckpt_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_actor_source_sha256=expected_actor_source_sha256,
    )
    policy.act([0.0] * 300, [1])

    assert common.loaded_asu_modules() == []


# ── import-time side effects (fresh subprocess) ─────────────────────────


def test_fresh_subprocess_import_has_no_heavy_side_effects():
    """Merely `import`-ing frozen_ppo_inference must not pull in torch,
    numpy, or the reference submodule's packages - those are all supposed
    to be lazy, function-local imports. Run in a fresh subprocess so this
    process's already-imported torch/numpy (from other tests) can't mask a
    real regression."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT / 'scripts')!r})\n"
        "import frozen_ppo_inference\n"
        "heavy = [m for m in ('torch', 'numpy', 'monopoly_game_engine', 'monopoly_bench') if m in sys.modules]\n"
        "print('HEAVY_MODULES=' + ','.join(heavy))\n"
        "asu = [m for m in sys.modules if m == 'ASU_FROZEN_TEACHER' or m.startswith('ASU_FROZEN_TEACHER.')]\n"
        "print('ASU_MODULES=' + ','.join(asu))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    heavy_line = next(line for line in result.stdout.splitlines() if line.startswith("HEAVY_MODULES="))
    asu_line = next(line for line in result.stdout.splitlines() if line.startswith("ASU_MODULES="))
    assert heavy_line == "HEAVY_MODULES=", heavy_line
    assert asu_line == "ASU_MODULES=", asu_line


def test_fresh_subprocess_full_flow_matches_in_process(tmp_path):
    """Loads a real checkpoint and calls act() inside a fresh subprocess -
    catches any import-order-dependent behavior that wouldn't show up when
    this module was already partially imported by earlier tests."""
    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    ckpt_path = tmp_path / "ckpt.pt"
    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.save(str(ckpt_path))
    checkpoint_sha256 = module.file_sha256(ckpt_path)
    actor_source_sha256 = module.actor_source_sha256()

    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT / 'scripts')!r})\n"
        "import frozen_ppo_inference as m\n"
        f"policy = m.FrozenPurePPOPolicy.from_checkpoint({str(ckpt_path)!r}, "
        f"expected_checkpoint_sha256={checkpoint_sha256!r}, "
        f"expected_actor_source_sha256={actor_source_sha256!r})\n"
        "action = policy.act([0.0] * 300, [1, 2, 3])\n"
        "assert action in (1, 2, 3), action\n"
        "print('OK', action)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().startswith("OK")


# ── torch.argmax tie-break (empirically verified, not assumed) ─────────


def test_torch_argmax_ties_break_to_lowest_index():
    import torch

    row = torch.tensor([[1.0, 5.0, 5.0, 5.0, 0.0]])
    assert int(torch.argmax(row, dim=-1).item()) == 1

    masked = torch.tensor([[float("-inf"), 5.0, 5.0, float("-inf"), float("-inf")]])
    assert int(torch.argmax(masked, dim=-1).item()) == 1


# ── exact agreement with the existing eval harness (real ActorNetwork) ──


class _FixedStateEnv:
    """Duck-typed stand-in matching exactly what
    build_masked_argmax_policy's choose() calls: get_allowed_actions(seat)
    and _get_state(seat). Returns a caller-supplied fixed state/legal set
    instead of always zeros, so scenarios can vary."""

    def __init__(self, legal_actions, state):
        self._legal = legal_actions
        self._state = state

    def get_allowed_actions(self, seat):
        return self._legal

    def _get_state(self, seat):
        return self._state


class _FixedStateGame:
    def __init__(self, env):
        self.env = env


def _random_scenarios(n: int, *, seed: int, action_space_size: int, state_dim: int):
    rng = random.Random(seed)
    scenarios = []
    for _ in range(n):
        k = rng.randint(1, 12)
        legal = tuple(sorted(rng.sample(range(action_space_size), k)))
        state = [rng.uniform(-3.0, 3.0) for _ in range(state_dim)]
        scenarios.append((state, legal))
    return scenarios


def _build_real_actor(hidden_dim: int = 8, torch_seed: int = 0):
    import torch

    common.ensure_reference_on_path()
    from monopoly_game_engine.networks import ActorNetwork

    torch.manual_seed(torch_seed)
    actor = ActorNetwork(hidden_dim=hidden_dim)
    actor.eval()
    return actor


def test_exact_agreement_with_existing_eval_harness_masked_argmax():
    """THE most important test: for many randomized (state, legal_action_ids)
    scenarios and a real ActorNetwork, frozen_ppo_inference.masked_argmax_action
    must select the IDENTICAL action as
    monopolyzero_pure_ppo_strength_screen.build_masked_argmax_policy - the
    policy already used to produce experiments 027-033."""
    import torch

    from monopoly_game_engine.actions import ACTION_SPACE_SIZE
    from monopoly_game_engine.state import STATE_DIM

    actor = _build_real_actor()
    device = torch.device("cpu")

    scenarios = _random_scenarios(200, seed=1234, action_space_size=ACTION_SPACE_SIZE, state_dim=STATE_DIM)

    existing_policy = strength_screen.build_masked_argmax_policy(actor, device, counters=None, env_holder=None)

    for state, legal in scenarios:
        env = _FixedStateEnv(legal, state)
        game = _FixedStateGame(env)
        expected = existing_policy.choose(game, seat=0, decision_seed=0).chosen_action

        actual = module.masked_argmax_action(actor, device, state, legal)
        assert actual == expected, (state, legal, expected, actual)

        via_policy_object = module.FrozenPurePPOPolicy(actor, device="cpu").act(state, legal)
        assert via_policy_object == expected


def test_exact_agreement_across_multiple_actor_seeds():
    """Same equivalence check but across a few differently-seeded actors,
    so agreement isn't an artifact of one particular weight init."""
    import torch

    from monopoly_game_engine.actions import ACTION_SPACE_SIZE
    from monopoly_game_engine.state import STATE_DIM

    device = torch.device("cpu")
    scenarios = _random_scenarios(20, seed=99, action_space_size=ACTION_SPACE_SIZE, state_dim=STATE_DIM)

    for torch_seed in (0, 1, 2, 7):
        actor = _build_real_actor(hidden_dim=8, torch_seed=torch_seed)
        existing_policy = strength_screen.build_masked_argmax_policy(actor, device, counters=None, env_holder=None)
        for state, legal in scenarios:
            env = _FixedStateEnv(legal, state)
            game = _FixedStateGame(env)
            expected = existing_policy.choose(game, seat=0, decision_seed=0).chosen_action
            actual = module.masked_argmax_action(actor, device, state, legal)
            assert actual == expected


# ── determinism ──────────────────────────────────────────────────────────


def test_masked_argmax_action_is_deterministic():
    import torch

    from monopoly_game_engine.actions import ACTION_SPACE_SIZE
    from monopoly_game_engine.state import STATE_DIM

    actor = _build_real_actor()
    device = torch.device("cpu")
    state = [0.1 * i for i in range(STATE_DIM)]
    legal = (0, 3, 7, ACTION_SPACE_SIZE - 1)

    results = {module.masked_argmax_action(actor, device, state, legal) for _ in range(10)}
    assert len(results) == 1


def test_act_is_deterministic():
    from monopoly_game_engine.state import STATE_DIM

    actor = _build_real_actor()
    policy = module.FrozenPurePPOPolicy(actor, device="cpu")
    state = [0.2 * i for i in range(STATE_DIM)]
    legal = (1, 4, 9)

    first = policy.act(state, legal)
    for _ in range(10):
        assert policy.act(state, legal) == first


# ── never selects an illegal action (sampled scenarios) ─────────────────


def test_act_never_selects_illegal_action_across_sampled_scenarios():
    from monopoly_game_engine.actions import ACTION_SPACE_SIZE
    from monopoly_game_engine.state import STATE_DIM

    actor = _build_real_actor()
    policy = module.FrozenPurePPOPolicy(actor, device="cpu")
    scenarios = _random_scenarios(300, seed=555, action_space_size=ACTION_SPACE_SIZE, state_dim=STATE_DIM)

    for state, legal in scenarios:
        chosen = policy.act(state, legal)
        assert chosen in legal


def test_act_with_single_legal_action_is_forced():
    from monopoly_game_engine.state import STATE_DIM

    actor = _build_real_actor()
    policy = module.FrozenPurePPOPolicy(actor, device="cpu")
    state = [0.0] * STATE_DIM
    assert policy.act(state, [17]) == 17
    assert policy.act(state, [0]) == 0


def test_act_rejects_empty_legal_action_ids():
    from monopoly_game_engine.state import STATE_DIM

    actor = _build_real_actor()
    policy = module.FrozenPurePPOPolicy(actor, device="cpu")
    with pytest.raises(ValueError):
        policy.act([0.0] * STATE_DIM, [])


# ── checkpoint file hash gate ────────────────────────────────────────────


def _save_real_checkpoint(path: Path, *, hybrid: bool = False, player_id: int = 0):
    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=player_id, hybrid=hybrid, device="cpu")
    agent.save(str(path))
    return agent


def test_load_frozen_actor_rejects_corrupted_checkpoint(tmp_path):
    ckpt_path = tmp_path / "ckpt.pt"
    _save_real_checkpoint(ckpt_path)
    expected_checkpoint_sha256 = module.file_sha256(ckpt_path)
    expected_actor_source_sha256 = module.actor_source_sha256()

    # Corrupt the file after computing the expected hash from the original.
    raw = bytearray(ckpt_path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    ckpt_path.write_bytes(bytes(raw))

    with pytest.raises(module.ChecksumMismatchError, match="checkpoint file SHA-256 mismatch"):
        module.load_frozen_actor(
            ckpt_path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_actor_source_sha256=expected_actor_source_sha256,
        )


def test_load_frozen_actor_rejects_wrong_checkpoint(tmp_path):
    """A structurally valid, real checkpoint - just not the one whose hash
    was pinned as expected."""
    ckpt_path = tmp_path / "ckpt.pt"
    _save_real_checkpoint(ckpt_path)

    with pytest.raises(module.ChecksumMismatchError, match="checkpoint file SHA-256 mismatch"):
        module.load_frozen_actor(
            ckpt_path,
            expected_checkpoint_sha256="0" * 64,
            expected_actor_source_sha256=module.actor_source_sha256(),
        )


def test_load_frozen_actor_rejects_nonexistent_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        module.load_frozen_actor(
            tmp_path / "does_not_exist.pt",
            expected_checkpoint_sha256="0" * 64,
            expected_actor_source_sha256="0" * 64,
        )


def test_load_frozen_actor_succeeds_on_matching_hashes(tmp_path):
    ckpt_path = tmp_path / "ckpt.pt"
    _save_real_checkpoint(ckpt_path)
    actor = module.load_frozen_actor(
        ckpt_path,
        expected_checkpoint_sha256=module.file_sha256(ckpt_path),
        expected_actor_source_sha256=module.actor_source_sha256(),
    )
    common.ensure_reference_on_path()
    from monopoly_game_engine.networks import ActorNetwork

    assert isinstance(actor, ActorNetwork)
    assert actor.training is False  # .eval() was called


# ── actor-source hash gate ──────────────────────────────────────────────


def test_actor_source_sha256_is_a_real_hash():
    value = module.actor_source_sha256()
    assert isinstance(value, str)
    assert len(value) == 64
    int(value, 16)  # raises if not valid hex


def test_load_frozen_actor_rejects_on_actor_source_hash_mismatch(tmp_path):
    ckpt_path = tmp_path / "ckpt.pt"
    _save_real_checkpoint(ckpt_path)

    with pytest.raises(module.ChecksumMismatchError, match="ActorNetwork class source-code SHA-256 mismatch"):
        module.load_frozen_actor(
            ckpt_path,
            expected_checkpoint_sha256=module.file_sha256(ckpt_path),
            expected_actor_source_sha256="0" * 64,
        )


# ── hybrid / fixed-rule override rejection ──────────────────────────────


def test_load_frozen_actor_rejects_hybrid_checkpoint(tmp_path):
    """A checkpoint saved with hybrid=True (the HYBRID_COMPAT-style
    fixed-rule carve-out this project treats as diagnostic-only, never a
    submission candidate - see docs/DECISIONS.md) must be rejected, not
    silently loaded as if it were pure PPO."""
    ckpt_path = tmp_path / "hybrid_ckpt.pt"
    _save_real_checkpoint(ckpt_path, hybrid=True)

    # PPOAgent.load's own metadata check (hybrid mismatch) fires before
    # this module's own redundant hybrid/fixed_action_mask assertion -
    # either way, this must not return a usable actor.
    with pytest.raises((ValueError, RuntimeError)):
        module.load_frozen_actor(
            ckpt_path,
            expected_checkpoint_sha256=module.file_sha256(ckpt_path),
            expected_actor_source_sha256=module.actor_source_sha256(),
        )


def test_module_never_imports_hybrid_compat_or_fixed_rule_paths():
    """Static check that this module never references the reference's
    fixed-rule decision functions or the project's own HYBRID_COMPAT
    diagnostic builder - those are diagnostic-upper-bound-only, never a
    primary decision path, per CLAUDE.md's hybrid-approach rule and
    docs/DECISIONS.md's 2026-08-12 entry. Name-based (not code-usage-based
    like the hybrid=True check below) because these names have no
    legitimate reason to appear anywhere in this module, including prose -
    unlike "hybrid=True", which the docstrings must legitimately discuss
    (as something rejected)."""
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "fixed_buy_decision",
        "fixed_accept_trade_decision",
        "build_local_hybrid_compat_policy",
        "HybridCompatPolicy",
    )
    hits = [name for name in forbidden if name in source]
    assert not hits, hits


def test_module_never_calls_anything_with_hybrid_true():
    """AST-based check (not a naive substring search, which would also
    flag this module's own docstring prose explaining that hybrid=True
    checkpoints are REJECTED): no Call node anywhere in this module's
    actual code passes a `hybrid=True` keyword argument."""
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "hybrid" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    hits.append(ast.dump(node))
    assert not hits, hits


def test_load_frozen_actor_never_accepts_a_hybrid_kwarg():
    """load_frozen_actor's signature must not expose a way for a caller to
    request hybrid=True - the frozen inference layer is pure-PPO only,
    structurally, not just by default."""
    import inspect

    sig = inspect.signature(module.load_frozen_actor)
    assert "hybrid" not in sig.parameters


# ── decision trace ───────────────────────────────────────────────────────


def test_decision_trace_disabled_by_default():
    from monopoly_game_engine.state import STATE_DIM

    actor = _build_real_actor()
    policy = module.FrozenPurePPOPolicy(actor, device="cpu")
    policy.act([0.0] * STATE_DIM, [1, 2, 3])
    assert policy.trace_log == []
    assert policy.last_trace() is None


def test_decision_trace_enabled_records_turn_and_top_candidates():
    from monopoly_game_engine.state import STATE_DIM

    actor = _build_real_actor()
    policy = module.FrozenPurePPOPolicy(actor, device="cpu", enable_trace=True)
    legal = (2, 5, 9, 11)
    chosen = policy.act([0.05] * STATE_DIM, legal, turn=42, trace_top_k=2)

    trace = policy.last_trace()
    assert trace is not None
    assert trace.turn == 42
    assert trace.chosen_action == chosen
    assert trace.legal_action_ids == legal
    assert len(trace.top_candidates) == 2
    assert trace.top_candidates[0][0] == chosen  # best-scored candidate is the chosen one
    # scores are sorted best-first
    assert trace.top_candidates[0][1] >= trace.top_candidates[1][1]
    for action_id, _score in trace.top_candidates:
        assert action_id in legal


# ── official adapter: explicit TBD ──────────────────────────────────────


def test_official_submission_adapter_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        module.OfficialSubmissionAdapter()


# ── full_actor_state_sha256 equivalence with the existing helper ───────


def test_full_actor_state_sha256_matches_existing_learnability_gate_helper():
    import importlib.util as _ilu

    gate_script = REPO_ROOT / "scripts" / "monopolyzero_pure_ppo_learnability_gate.py"
    gate_spec = _ilu.spec_from_file_location("monopolyzero_pure_ppo_learnability_gate_for_equiv_test", gate_script)
    gate_module = _ilu.module_from_spec(gate_spec)
    sys.modules[gate_spec.name] = gate_module
    gate_spec.loader.exec_module(gate_module)

    actor = _build_real_actor(hidden_dim=8, torch_seed=3)
    assert module.full_actor_state_sha256(actor) == gate_module._full_actor_sha256(actor)
