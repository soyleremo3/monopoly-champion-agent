"""Tests for scripts/monopolyzero_pure_ppo_strength_screen.py:
- pure helpers (_seed_range, _mean, _file_sha256)
- source-level ASU-import guard
- build_masked_argmax_policy: deterministic (no sampling) legal-masked
  argmax against a fake actor with hand-computed expectations, plus
  opportunity/chosen counter bookkeeping and the shared env_holder
  mechanism
- load_candidate_agent / build_untrained_baseline_agent: hash-mismatch STOP
  behavior (SystemExit, not a silent retrain), and that the untrained
  baseline reconstruction is bit-exact reproducible
- summarize(): hand-computed stats from synthetic per-game records,
  integrity-failure raise on any illegal action/crash, GO/KILL/INCONCLUSIVE
  boundary logic
- one small REAL-engine smoke game (tiny hand-built actor, not the real
  checkpoint) exercising play_one_game() end to end
- CLI argument parsing
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_pure_ppo_strength_screen.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import monopolyzero_common as common  # noqa: E402

_spec = importlib.util.spec_from_file_location("monopolyzero_pure_ppo_strength_screen", SCRIPT)
module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_pure_ppo_strength_screen"] = module
_spec.loader.exec_module(module)


# ── source-level guard ──────────────────────────────────────────────────


def test_source_never_imports_asu():
    import_lines = [
        line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("ASU_FROZEN_TEACHER", "monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert not hits, hits


def test_source_never_uses_sampling_action_selection():
    """This screen must use deterministic masked argmax everywhere, never
    PPOAgent.choose_action's torch.multinomial sampling path."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "torch.multinomial(" not in source
    assert "torch.argmax" in source


# ── pure helpers ─────────────────────────────────────────────────────────


def test_seed_range():
    assert module._seed_range(46000, 5) == [46000, 46001, 46002, 46003, 46004]


def test_mean_helper():
    assert module._mean([]) is None
    assert module._mean([2.0, 4.0]) == 3.0


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "f.bin"
    path.write_bytes(b"hello world")
    assert module._file_sha256(path) == hashlib.sha256(b"hello world").hexdigest()


# ── build_masked_argmax_policy (fake actor, hand-computed) ─────────────


class _FakeGame:
    def __init__(self, env):
        self.env = env


class _FakeEnv:
    """Duck-typed stand-in exposing exactly what the policy under test
    calls: get_allowed_actions(seat) and _get_state(seat)."""

    def __init__(self, legal_actions, state_dim=10):
        self._legal = legal_actions
        self._state_dim = state_dim

    def get_allowed_actions(self, seat):
        return self._legal

    def _get_state(self, seat):
        import numpy as np

        return np.zeros(self._state_dim, dtype=np.float32)


class _FakeActor:
    """Returns fixed logits regardless of input state, but DOES apply the
    given legal-action mask the same way the real ActorNetwork.forward
    does (masked_fill -inf on disallowed actions) - decoupled from real
    network weights/state, but exercises the same masking contract the
    policy wrapper under test relies on. Callable(state_t, mask_t) ->
    log_probs tensor of shape (1, num_actions)."""

    def __init__(self, logits_row):
        import torch

        self._row = torch.as_tensor(logits_row, dtype=torch.float32).unsqueeze(0)
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def __call__(self, state_t, mask_t):
        return self._row.masked_fill(~mask_t, float("-inf"))


def test_masked_argmax_policy_picks_highest_prob_legal_action():
    import torch

    common.ensure_reference_on_path()
    from monopoly_game_engine.actions import ACTION_SPACE_SIZE, ActionType

    buy_id = int(ActionType.BUY_PROPERTY)
    row = [0.0] * ACTION_SPACE_SIZE
    row[buy_id] = 5.0  # highest logit/log-prob among legal actions
    row[0] = 10.0  # highest overall, but NOT legal - must be ignored
    actor = _FakeActor(row)

    legal = (0 + 1, buy_id, 2)  # action 0 (highest score) deliberately excluded from legal set
    env = _FakeEnv(legal)
    game = _FakeGame(env)

    policy = module.build_masked_argmax_policy(actor, torch.device("cpu"), counters=None, env_holder=None)
    result = policy.choose(game, seat=0, decision_seed=0)

    assert result.chosen_action == buy_id
    assert policy.kind == "policy_only"


def test_masked_argmax_policy_updates_counters_and_env_holder():
    import torch

    common.ensure_reference_on_path()
    from monopoly_game_engine.actions import ACTION_SPACE_SIZE, ActionType

    buy_id = int(ActionType.BUY_PROPERTY)
    accept_id = int(ActionType.ACCEPT_TRADE)
    decline_id = int(ActionType.DECLINE_TRADE)

    row = [0.0] * ACTION_SPACE_SIZE
    row[accept_id] = 5.0  # will be chosen over decline
    actor = _FakeActor(row)

    legal = (buy_id, accept_id, decline_id)
    env = _FakeEnv(legal)
    game = _FakeGame(env)

    counters = module._new_counters()
    env_holder = [None]
    policy = module.build_masked_argmax_policy(actor, torch.device("cpu"), counters, env_holder)
    result = policy.choose(game, seat=1, decision_seed=0)

    assert result.chosen_action == accept_id
    assert counters["decisions"] == 1
    assert counters["buy_property_opportunities"] == 1
    assert counters["buy_property_chosen"] == 0
    assert counters["accept_trade_opportunities"] == 1
    assert counters["accept_trade_chosen"] == 1
    assert counters["decline_trade_opportunities"] == 1
    assert counters["decline_trade_chosen"] == 0
    assert env_holder[0] is env  # shared reference captured for later reads


def test_masked_argmax_policy_counts_trade_offer_family():
    import torch

    common.ensure_reference_on_path()
    from monopoly_game_engine.actions import ACTION_SPACE_SIZE, OFFSETS

    trade_offer_action = OFFSETS["buy_trade"]  # first action in the buy_trade family
    row = [0.0] * ACTION_SPACE_SIZE
    row[trade_offer_action] = 5.0
    actor = _FakeActor(row)

    env = _FakeEnv((trade_offer_action, 1))
    game = _FakeGame(env)
    counters = module._new_counters()
    policy = module.build_masked_argmax_policy(actor, torch.device("cpu"), counters, env_holder=None)
    policy.choose(game, seat=0, decision_seed=0)

    assert counters["trade_offer_chosen"] == 1


# ── candidate/baseline hash gates ───────────────────────────────────────


def test_load_candidate_agent_raises_system_exit_on_hash_mismatch(monkeypatch, tmp_path):
    common.ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    fake_checkpoint = tmp_path / "candidate_ppo.pt"
    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    agent.save(str(fake_checkpoint))  # a real, structurally valid checkpoint - just not 027's

    monkeypatch.setattr(module, "EXPECTED_CANDIDATE_ACTOR_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="candidate actor full_actor_sha256 mismatch"):
        module.load_candidate_agent(fake_checkpoint)


def test_build_untrained_baseline_agent_raises_system_exit_on_hash_mismatch(monkeypatch):
    common.ensure_reference_on_path()
    monkeypatch.setattr(module, "EXPECTED_BASELINE_ACTOR_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="reconstructed baseline actor full_actor_sha256 mismatch"):
        module.build_untrained_baseline_agent()


def test_build_untrained_baseline_agent_is_bit_exact_reproducible():
    common.ensure_reference_on_path()
    agent_a = module.build_untrained_baseline_agent()
    agent_b = module.build_untrained_baseline_agent()
    hash_a = module._full_actor_sha256(agent_a.actor)
    hash_b = module._full_actor_sha256(agent_b.actor)
    assert hash_a == hash_b
    assert hash_a == module.EXPECTED_BASELINE_ACTOR_SHA256  # pinned to 027's own logged value


# ── summarize(): synthetic per-game records ─────────────────────────────


def _fake_game(seed, focus_seat, *, winner, illegal=0, crashed=False, round_cap=True,
                candidate_nw=1000.0, baseline_nw=1000.0, candidate_bankrupt=False, baseline_bankrupt=False):
    def _seat_stats(seat):
        is_candidate = seat == focus_seat
        return {
            "is_candidate": is_candidate,
            "win": seat == winner,
            "net_worth": candidate_nw if is_candidate else baseline_nw,
            "bankrupt": candidate_bankrupt if is_candidate else baseline_bankrupt,
            "properties_owned": 3,
            "decisions": 10,
            "buy_property_opportunities": 2, "buy_property_chosen": 1 if is_candidate else 0,
            "accept_trade_opportunities": 4, "accept_trade_chosen": 2 if is_candidate else 1,
            "decline_trade_opportunities": 4, "decline_trade_chosen": 0,
            "trade_offer_chosen": 5,
        }

    return {
        "game_id": 0, "seed": seed, "focus_seat": focus_seat,
        "completed": True, "winner": winner, "decisions": 40,
        "final_round": 200 if round_cap else 100,
        "round_cap_hit": round_cap,
        "illegal_actions": illegal, "crashed": crashed, "error": None,
        "per_seat": {seat: _seat_stats(seat) for seat in range(4)},
    }


def test_summarize_raises_on_illegal_action():
    games = [_fake_game(46000, 0, winner=0, illegal=1)]
    with pytest.raises(RuntimeError, match="Integrity failure"):
        module.summarize(games)


def test_summarize_raises_on_crash():
    games = [_fake_game(46000, 0, winner=None, crashed=True)]
    with pytest.raises(RuntimeError, match="Integrity failure"):
        module.summarize(games)


def test_summarize_all_candidate_wins_verdict_go():
    """20 seed blocks x 4 rotations, candidate always wins - win rate 100%
    vs. 25% null should produce a CI entirely above zero -> GO."""
    games = [
        _fake_game(46000 + seed_index, focus_seat, winner=focus_seat)
        for seed_index in range(20) for focus_seat in range(4)
    ]
    summary = module.summarize(games)
    assert summary["candidate_win_rate"] == pytest.approx(1.0)
    assert summary["verdict"] == "GO"
    lower, upper = summary["primary_seed_block_bootstrap_vs_25pct_null"]["win_rate_diff_from_null"]["ci_95"]
    assert lower > 0.0


def test_summarize_candidate_never_wins_verdict_kill():
    games = [
        _fake_game(46000 + seed_index, focus_seat, winner=(focus_seat + 1) % 4)
        for seed_index in range(20) for focus_seat in range(4)
    ]
    summary = module.summarize(games)
    assert summary["candidate_win_rate"] == pytest.approx(0.0)
    assert summary["verdict"] == "KILL"


def _win_pattern_games(seed, wins):
    """`wins`: 4 bools, one per focus-seat rotation - True means that
    rotation's game is set up so the candidate seat wins it."""
    games = []
    for focus_seat, win in enumerate(wins):
        winner = focus_seat if win else (focus_seat + 1) % 4
        games.append(_fake_game(seed, focus_seat, winner=winner))
    return games


def test_summarize_mixed_evidence_straddling_null_is_inconclusive():
    """Overall win rate is exactly the 25% null (point diff = 0), but built
    from two very different block win rates (50% and 0%) so the seed-block
    bootstrap CI has real spread straddling zero - neither GO's strict
    lower>0 nor KILL's upper<=0 can hold."""
    games = []
    for i in range(10):
        games += _win_pattern_games(46000 + i, [True, True, False, False])  # 2/4 = 0.5
    for i in range(10, 20):
        games += _win_pattern_games(46000 + i, [False, False, False, False])  # 0/4 = 0.0
    summary = module.summarize(games)
    assert summary["candidate_win_rate"] == pytest.approx(0.25)
    assert summary["verdict"] == "INCONCLUSIVE"


def test_summarize_net_worth_and_bankruptcy_aggregation():
    games = [_fake_game(46000, 0, winner=0, candidate_nw=2000.0, baseline_nw=500.0, candidate_bankrupt=False, baseline_bankrupt=True)]
    summary = module.summarize(games)
    assert summary["candidate_mean_net_worth"] == pytest.approx(2000.0)
    assert summary["baseline_mean_net_worth"] == pytest.approx(500.0)
    assert summary["candidate_net_worth_advantage"] == pytest.approx(1500.0)
    assert summary["candidate_bankruptcy_rate"] == pytest.approx(0.0)
    assert summary["baseline_bankruptcy_rate"] == pytest.approx(1.0)  # all 3 baseline seats bankrupt


def test_summarize_action_behavior_matches_hand_computed_rates():
    games = [_fake_game(46000, 0, winner=0)]
    summary = module.summarize(games)
    candidate_behavior = summary["candidate_action_behavior"]
    assert candidate_behavior["buy_property_opportunities"] == 2
    assert candidate_behavior["buy_property_rate_when_legal"] == pytest.approx(0.5)
    assert candidate_behavior["accept_trade_rate_when_legal"] == pytest.approx(0.5)
    assert candidate_behavior["trade_offers_total"] == 5


def test_summarize_round_cap_rate():
    games = [_fake_game(46000, 0, winner=0, round_cap=True), _fake_game(46001, 0, winner=0, round_cap=False)]
    summary = module.summarize(games)
    assert summary["round_cap_hits"] == 1
    assert summary["round_cap_rate"] == pytest.approx(0.5)


# ── tiny REAL-engine smoke game ─────────────────────────────────────────


def test_real_engine_smoke_one_game():
    """Plays one real (tiny hidden_dim, not the real checkpoint) game end
    to end through play_one_game() - proves the whole plumbing (policy
    wrapper x4 seats, play_local_game, final env read via env_holder,
    per-seat stat extraction) is wired correctly against the real engine.
    Not a strength claim, not the real 80-game screen."""
    import torch

    common.ensure_reference_on_path()
    from monopoly_game_engine.networks import ActorNetwork

    torch.manual_seed(0)
    candidate_actor = ActorNetwork(hidden_dim=8)
    torch.manual_seed(1)
    baseline_actor = ActorNetwork(hidden_dim=8)
    candidate_actor.eval()
    baseline_actor.eval()

    result = module.play_one_game(
        game_id=1, seed=46000, candidate_actor=candidate_actor, baseline_actor=baseline_actor,
        focus_seat=0, device=torch.device("cpu"), max_rounds=5,
    )

    assert result["completed"] is True
    assert result["illegal_actions"] == 0
    assert result["crashed"] is False
    assert set(result["per_seat"]) == {0, 1, 2, 3}
    for seat, stats in result["per_seat"].items():
        assert stats["is_candidate"] == (seat == 0)
        assert isinstance(stats["bankrupt"], bool)
        assert isinstance(stats["properties_owned"], int)
        assert stats["net_worth"] is not None
    assert result["winner"] in (0, 1, 2, 3)
    assert result["round_cap_hit"] == (result["final_round"] >= 5)


# ── CLI ──────────────────────────────────────────────────────────────────


def test_cli_defaults():
    args = module.build_arg_parser().parse_args([])
    assert args.seed_base == module.DEV_SEED_BASE == 46000
    assert args.n_seeds == module.N_SEEDS == 20
    assert args.checkpoint == module.DEFAULT_CANDIDATE_CHECKPOINT
    assert args.device == "cpu"
    assert args.output is None


def test_cli_overrides():
    args = module.build_arg_parser().parse_args(["--seed-base", "1", "--n-seeds", "2", "--output", "out.json"])
    assert args.seed_base == 1
    assert args.n_seeds == 2
    assert str(args.output) == "out.json"
