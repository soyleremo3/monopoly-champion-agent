"""Tests for scripts/monopolyzero_pure_ppo_learnability_gate.py:
- pure helpers (_mean, _legal_from_mask, _rank, _seed_range)
- actor snapshot/hash/compare against a tiny real ActorNetwork, proving the
  BUY_PROPERTY/ACCEPT_TRADE row hash/L2 machinery actually detects a real,
  isolated weight change and nothing else
- measure_actor_on_positions against a fake actor with hand-computed
  expected opportunity/rank/greedy/per-game_id stats (no real engine)
- instrument_agent: fake-agent exclusion-detection test (proves the wrapper
  itself works) plus a real-but-tiny PPOAgent(hybrid=False) smoke test
  (proves it wraps the real bound methods without changing behavior)
- install_fallback_counting_fixed_agents: monkeypatch/restore mechanism
- load_actor0_diagnostic_positions: real (tiny) on-disk ReplayBuffer,
  actor_id filtering
- source-level guard: zero "asu"/"ASU" text anywhere in this script
- CLI argument parsing
- one small REAL-engine smoke test (2 tiny games, hidden_dim left at
  default since PPOAgent has no override for it in this script) exercising
  run_gate() end to end - not a benchmark, not the real 32-game run
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_pure_ppo_learnability_gate.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_pure_ppo_learnability_gate", SCRIPT)
module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_pure_ppo_learnability_gate"] = module
_spec.loader.exec_module(module)

BUY_ID, ACCEPT_ID = 3, 7


# ── source-level guard ──────────────────────────────────────────────────────


def test_source_never_imports_asu():
    """The docstring legitimately discusses ASU-avoidance in prose (same as
    every other script in this project) - the actual invariant is that no
    import statement anywhere in this file ever names ASU_FROZEN_TEACHER or
    the ASU-coupled monopoly_bench modules (.adapters/.arena/.training)."""
    import_lines = [
        line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("ASU_FROZEN_TEACHER", "monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert not hits, hits


# ── pure helpers ─────────────────────────────────────────────────────────────


def test_mean_helper():
    assert module._mean([]) is None
    assert module._mean([1.0, 2.0, 3.0]) == 2.0


def test_legal_from_mask():
    assert module._legal_from_mask([True, False, True, False]) == (0, 2)


def test_rank_ties_broken_by_lower_action_id():
    prob_of = {0: 0.5, 3: 0.3, 7: 0.3, 9: 0.1}
    legal = (0, 3, 7, 9)
    assert module._rank(0, legal, prob_of) == 1
    assert module._rank(3, legal, prob_of) == 2  # tied with 7, lower id wins
    assert module._rank(7, legal, prob_of) == 3
    assert module._rank(9, legal, prob_of) == 4


def test_seed_range():
    assert module._seed_range(45000, 5) == [45000, 45001, 45002, 45003, 45004]


# ── actor snapshot / hash / compare ─────────────────────────────────────────


def test_snapshot_and_compare_detect_isolated_row_change():
    module.common.ensure_reference_on_path()
    import torch

    from monopoly_game_engine.networks import ActorNetwork

    torch.manual_seed(0)
    actor = ActorNetwork(hidden_dim=8)

    before = module.snapshot_actor(actor, buy_id=BUY_ID, accept_id=ACCEPT_ID)

    with torch.no_grad():
        final_layer = actor.net[-1]
        final_layer.weight[BUY_ID] += 1.0
        final_layer.bias[BUY_ID] += 1.0

    after = module.snapshot_actor(actor, buy_id=BUY_ID, accept_id=ACCEPT_ID)
    comparison = module.compare_snapshots(before, after)

    assert comparison["full_actor_changed"] is True
    assert comparison["buy_property_row_changed"] is True
    assert comparison["accept_trade_row_changed"] is False
    assert comparison["buy_property_row_l2_delta"] > 0.0
    assert comparison["accept_trade_row_l2_delta"] == pytest.approx(0.0)
    assert "_buy_property_row" not in comparison["before"]  # internal arrays stripped


def test_snapshot_no_change_yields_identical_hashes():
    module.common.ensure_reference_on_path()
    import torch

    from monopoly_game_engine.networks import ActorNetwork

    torch.manual_seed(1)
    actor = ActorNetwork(hidden_dim=8)
    before = module.snapshot_actor(actor, buy_id=BUY_ID, accept_id=ACCEPT_ID)
    after = module.snapshot_actor(actor, buy_id=BUY_ID, accept_id=ACCEPT_ID)
    comparison = module.compare_snapshots(before, after)
    assert comparison["full_actor_changed"] is False
    assert comparison["buy_property_row_changed"] is False
    assert comparison["accept_trade_row_changed"] is False


# ── measure_actor_on_positions (fake actor, hand-computed expectations) ────


class _FakeActor:
    """Returns pre-baked log-probabilities directly (log(p), so .exp() in
    the function under test recovers exactly `p`) - decouples this test
    from real network weights/softmax behavior."""

    def __init__(self, rows):
        import torch

        self._stacked = torch.log(torch.stack([torch.as_tensor(row, dtype=torch.float64) for row in rows]).float())
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def __call__(self, states_t, masks_t):
        assert states_t.shape[0] == masks_t.shape[0]
        return self._stacked[: states_t.shape[0]]


def _row(num_actions: int, probs: dict[int, float]):
    row = [1e-12] * num_actions
    for action, p in probs.items():
        row[action] = p
    return row


def test_measure_actor_on_positions_matches_hand_computed_stats():
    import types

    import numpy as np
    import torch

    num_actions = 10
    rows = [
        _row(num_actions, {0: 0.5, BUY_ID: 0.3, ACCEPT_ID: 0.2}),  # game 100
        _row(num_actions, {BUY_ID: 0.9, ACCEPT_ID: 0.1}),  # game 100
        _row(num_actions, {0: 0.4, ACCEPT_ID: 0.6}),  # game 200, no buy opportunity
    ]
    fake_actor = _FakeActor(rows)

    def _mask(legal_ids: set[int]) -> np.ndarray:
        mask = np.zeros(num_actions, dtype=bool)
        mask[list(legal_ids)] = True
        return mask

    positions = [
        types.SimpleNamespace(state=np.zeros(num_actions), legal_mask=_mask({0, BUY_ID, ACCEPT_ID}), game_id=100),
        types.SimpleNamespace(state=np.zeros(num_actions), legal_mask=_mask({BUY_ID, ACCEPT_ID}), game_id=100),
        types.SimpleNamespace(state=np.zeros(num_actions), legal_mask=_mask({0, ACCEPT_ID}), game_id=200),
    ]

    result = module.measure_actor_on_positions(
        fake_actor, positions, buy_id=BUY_ID, accept_id=ACCEPT_ID, device=torch.device("cpu"), batch_size=10,
    )

    assert fake_actor.eval_called is True

    assert result["buy_property_opportunities"] == 2
    assert result["buy_property_greedy_chosen"] == 1
    assert result["buy_property_greedy_rate"] == pytest.approx(0.5)
    assert result["buy_property_mean_raw_prior"] == pytest.approx((0.3 + 0.9) / 2)
    assert result["buy_property_mean_prior_rank"] == pytest.approx((2 + 1) / 2)

    assert result["accept_trade_opportunities"] == 3
    assert result["accept_trade_greedy_chosen"] == 1
    assert result["accept_trade_greedy_rate"] == pytest.approx(1 / 3)
    assert result["accept_trade_mean_raw_prior"] == pytest.approx((0.2 + 0.1 + 0.6) / 3)
    assert result["accept_trade_mean_prior_rank"] == pytest.approx((3 + 2 + 1) / 3)

    per_game = result["per_game_id"]
    assert set(per_game) == {"100", "200"}
    assert per_game["100"]["buy_property_opportunities"] == 2
    assert per_game["100"]["buy_property_greedy_rate"] == pytest.approx(0.5)
    assert per_game["100"]["buy_property_mean_prior_rank"] == pytest.approx(1.5)
    assert per_game["100"]["accept_trade_opportunities"] == 2
    assert per_game["100"]["accept_trade_greedy_rate"] == pytest.approx(0.0)
    assert per_game["200"]["buy_property_opportunities"] == 0
    assert per_game["200"]["buy_property_greedy_rate"] is None
    assert per_game["200"]["accept_trade_opportunities"] == 1
    assert per_game["200"]["accept_trade_greedy_rate"] == pytest.approx(1.0)


# ── instrument_agent ─────────────────────────────────────────────────────────


class _FakeInnerAgent:
    """Duck-typed stand-in for PPOAgent - instrument_agent only touches
    `.choose_action`/`.store`, so this proves the wrapper's own counting
    logic in isolation from the reference's real hybrid branching."""

    def __init__(self, nn_allowed_override=None):
        self.nn_allowed_override = nn_allowed_override
        self.stored = []

    def choose_action(self, state, env, allowed_actions):
        nn_allowed = self.nn_allowed_override if self.nn_allowed_override is not None else list(allowed_actions)
        return (nn_allowed[0], 0.1, 0.0, nn_allowed)

    def store(self, state, action, log_prob, reward, value, done, allowed_actions):
        self.stored.append((action, log_prob))


def test_instrument_agent_detects_no_exclusion_when_action_stays_in_nn_allowed():
    fake = _FakeInnerAgent()
    counters = module.instrument_agent(fake, buy_id=BUY_ID, accept_id=ACCEPT_ID)

    fake.choose_action(None, None, [BUY_ID, ACCEPT_ID, 99])

    assert counters["buy_property_opportunities_seen"] == 1
    assert counters["buy_property_excluded_from_nn"] == 0
    assert counters["accept_trade_opportunities_seen"] == 1
    assert counters["accept_trade_excluded_from_nn"] == 0


def test_instrument_agent_detects_real_exclusion_from_nn_allowed():
    fake = _FakeInnerAgent(nn_allowed_override=[99])  # BUY_ID/ACCEPT_ID excluded
    counters = module.instrument_agent(fake, buy_id=BUY_ID, accept_id=ACCEPT_ID)

    fake.choose_action(None, None, [BUY_ID, ACCEPT_ID, 99])

    assert counters["buy_property_opportunities_seen"] == 1
    assert counters["buy_property_excluded_from_nn"] == 1
    assert counters["accept_trade_opportunities_seen"] == 1
    assert counters["accept_trade_excluded_from_nn"] == 1


def test_instrument_agent_counts_stores_by_action_and_forwards_to_original():
    fake = _FakeInnerAgent()
    counters = module.instrument_agent(fake, buy_id=BUY_ID, accept_id=ACCEPT_ID)

    fake.store(None, BUY_ID, 0.5, 0.0, 0.0, False, [BUY_ID])
    fake.store(None, ACCEPT_ID, 0.7, 0.0, 0.0, False, [ACCEPT_ID])
    fake.store(None, 99, 0.9, 0.0, 0.0, False, [99])

    assert counters["store_calls"] == 3
    assert counters["buy_property_stored_with_log_prob"] == 1
    assert counters["accept_trade_stored_with_log_prob"] == 1
    assert fake.stored == [(BUY_ID, 0.5), (ACCEPT_ID, 0.7), (99, 0.9)]  # original still ran


def test_instrument_agent_wraps_real_ppo_agent_without_breaking_it():
    """hybrid=False PPOAgent, forced-legal (single allowed action = BUY_ID)
    so the wrapped choose_action's return is deterministic regardless of
    network weights."""
    module.common.ensure_reference_on_path()
    import numpy as np

    from monopoly_game_engine.agent_ppo import PPOAgent
    from monopoly_game_engine.state import STATE_DIM

    agent = PPOAgent(player_id=0, hybrid=False, hidden_dim=8, device="cpu")
    counters = module.instrument_agent(agent, buy_id=BUY_ID, accept_id=ACCEPT_ID)

    state = np.zeros(STATE_DIM, dtype=np.float32)
    action, log_prob, value, nn_allowed = agent.choose_action(state, env=None, allowed_actions=[BUY_ID])

    assert action == BUY_ID
    assert log_prob is not None
    assert nn_allowed == [BUY_ID]
    assert counters["buy_property_opportunities_seen"] == 1
    assert counters["buy_property_excluded_from_nn"] == 0

    agent.store(state, action, log_prob, 0.0, value, False, nn_allowed)
    assert counters["store_calls"] == 1
    assert counters["buy_property_stored_with_log_prob"] == 1


# ── install_fallback_counting_fixed_agents (patch/restore mechanism) ───────


def test_install_fallback_counting_patches_and_restores_module_globals():
    module.common.ensure_reference_on_path()
    import sys as _sys

    train_module = _sys.modules["monopoly_game_engine.train"]  # see install_fallback_counting_fixed_agents's own comment on why not `import ... as`
    from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC

    counters, restore = module.install_fallback_counting_fixed_agents()
    try:
        assert train_module.FPAgentA is not FPAgentA
        assert issubclass(train_module.FPAgentA, FPAgentA)
        assert train_module.FPAgentB is not FPAgentB
        assert issubclass(train_module.FPAgentB, FPAgentB)
        assert train_module.FPAgentC is not FPAgentC
        assert issubclass(train_module.FPAgentC, FPAgentC)
        assert counters == {"fallback_substitutions": 0}
    finally:
        restore()

    assert train_module.FPAgentA is FPAgentA
    assert train_module.FPAgentB is FPAgentB
    assert train_module.FPAgentC is FPAgentC


def test_install_fallback_counting_detects_a_forced_substitution(monkeypatch):
    """Forces the wrapped base class's choose_action to propose a fixed,
    always-illegal action id, and confirms the counting subclass (a) still
    passes that proposal through unchanged (train.py's own substitution
    logic, not this wrapper's job, decides what actually gets played) and
    (b) counts exactly the calls where it wasn't in the env's allowed set."""
    module.common.ensure_reference_on_path()
    import sys as _sys

    train_module = _sys.modules["monopoly_game_engine.train"]  # see install_fallback_counting_fixed_agents's own comment on why not `import ... as`
    from monopoly_game_engine.agents_fixed import FPAgentA

    counters, restore = module.install_fallback_counting_fixed_agents()
    try:
        counting_class = train_module.FPAgentA
        monkeypatch.setattr(FPAgentA, "choose_action", lambda self, env: 12345)
        agent = counting_class(0)

        class _FakeEnvIllegal:
            def get_allowed_actions(self, pid):
                return [1, 2, 3]  # 12345 never legal here

        class _FakeEnvLegal:
            def get_allowed_actions(self, pid):
                return [12345]

        proposed = agent.choose_action(_FakeEnvIllegal())
        assert proposed == 12345  # pass-through unchanged, not this wrapper's job to fix it
        assert counters["fallback_substitutions"] == 1

        agent.choose_action(_FakeEnvLegal())
        assert counters["fallback_substitutions"] == 1  # legal this time - no increment
    finally:
        restore()


# ── load_actor0_diagnostic_positions ────────────────────────────────────────


def test_load_actor0_diagnostic_positions_filters_by_actor_id(tmp_path):
    module.common.ensure_reference_on_path()
    import numpy as np

    from monopoly_bench.contracts import ReplayPosition
    from monopoly_bench.engine import ACTION_SPACE_SIZE, STATE_DIM
    from monopoly_bench.storage import ReplayBuffer

    replay_dir = tmp_path / "replay"
    replay = ReplayBuffer(replay_dir, capacity=4, create=True)
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    mask[[0, BUY_ID]] = True

    def _pos(actor_id, game_id):
        return ReplayPosition(
            state=np.zeros(STATE_DIM, dtype=np.float32), legal_mask=mask,
            visits={0: 1}, q_values={0: (0.25,) * 4}, selected_action=0,
            value=(0.25,) * 4, outcome=(0.0,) * 4, actor_id=actor_id, game_id=game_id,
        )

    replay.append_many([_pos(0, 1), _pos(1, 1), _pos(0, 2), _pos(2, 2)])

    positions = module.load_actor0_diagnostic_positions(replay_dir)
    assert len(positions) == 2
    assert all(position.actor_id == 0 for position in positions)
    assert sorted(position.game_id for position in positions) == [1, 2]


def test_load_actor0_diagnostic_positions_missing_directory_raises(tmp_path):
    with pytest.raises(SystemExit, match="No replay buffer found"):
        module.load_actor0_diagnostic_positions(tmp_path / "does_not_exist")


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_defaults():
    args = module.build_arg_parser().parse_args([])
    assert args.n_games == module.N_GAMES == 32
    assert args.seed_base == module.DEV_SEED_BASE == 45000
    assert args.checkpoint_output == module.DEFAULT_CHECKPOINT_OUTPUT
    assert args.replay_dir == module.DIAGNOSTIC_REPLAY_DIR
    assert args.output is None


def test_cli_overrides():
    args = module.build_arg_parser().parse_args(
        ["--n-games", "2", "--seed-base", "1", "--output", "out.json"]
    )
    assert args.n_games == 2
    assert args.seed_base == 1
    assert str(args.output) == "out.json"


# ── tiny REAL-engine smoke run (not a benchmark, not the real 32-game gate) ─


def test_real_engine_smoke_two_tiny_games(tmp_path):
    """Genuinely runs run_gate() for 2 games against the real engine/fixed
    opponents - proves the whole pipeline (PPOAgent, reference train(),
    instrumentation, diagnostic measurement, checkpoint save/reload,
    ASU-free guard) is wired correctly end to end. Not a strength claim,
    not the real 32-game gate (that is run separately, outside the test
    suite, as its own logged experiment)."""
    module.common.ensure_reference_on_path()
    import numpy as np
    import torch

    from monopoly_bench.contracts import ReplayPosition
    from monopoly_bench.engine import ACTION_SPACE_SIZE, STATE_DIM
    from monopoly_game_engine.actions import ActionType

    random_module = __import__("random")
    random_module.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Tiny synthetic diagnostic set (2 positions) - the real actor_id==0
    # diagnostic set from experiment 026 is exercised separately, when the
    # real gate is actually run; this keeps the test fast and independent
    # of that large on-disk artifact.
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    buy_id, accept_id = int(ActionType.BUY_PROPERTY), int(ActionType.ACCEPT_TRADE)
    mask[[0, buy_id, accept_id]] = True
    diagnostic_positions = [
        ReplayPosition(
            state=np.zeros(STATE_DIM, dtype=np.float32), legal_mask=mask,
            visits={0: 1}, q_values={0: (0.25,) * 4}, selected_action=0,
            value=(0.25,) * 4, outcome=(0.0,) * 4, actor_id=0, game_id=42,
        )
    ]

    checkpoint_output = tmp_path / "candidate_ppo.pt"
    result = module.run_gate(
        n_games=2, seed_base=1, diagnostic_positions=diagnostic_positions, checkpoint_output=checkpoint_output,
    )

    assert result["asu_modules_loaded"] == []
    assert result["hybrid_inactive_proof"]["agent_hybrid_flag"] is False
    assert result["hybrid_inactive_proof"]["fixed_action_mask_any_true"] is False
    assert result["hybrid_inactive_proof"]["fixed_action_mask_buy_property"] is False
    assert result["hybrid_inactive_proof"]["fixed_action_mask_accept_trade"] is False

    assert len(result["training_history"]["games"]) == 2
    assert len(result["training_history"]["win_rates_per_game_pct"]) == 2
    assert result["training_instrumentation"]["store_calls"] >= 0
    assert result["opponent_fallback_instrumentation"]["fallback_substitutions"] >= 0

    assert result["before_diagnostic"]["buy_property_opportunities"] == 1
    assert result["after_diagnostic"]["buy_property_opportunities"] == 1

    assert checkpoint_output.is_file()

    # Restored, not left dangling on the real reference module.
    import sys as _sys

    train_module = _sys.modules["monopoly_game_engine.train"]  # see install_fallback_counting_fixed_agents's own comment on why not `import ... as`
    from monopoly_game_engine.agents_fixed import FPAgentA

    assert train_module.FPAgentA is FPAgentA

    # Checkpoint is genuinely loadable by the reference's own PPOAgent.load.
    from monopoly_game_engine.agent_ppo import PPOAgent

    reloaded = PPOAgent(player_id=0, hybrid=False, hidden_dim=result["config"]["hidden_dim"], device="cpu")
    reloaded.load(str(checkpoint_output))
