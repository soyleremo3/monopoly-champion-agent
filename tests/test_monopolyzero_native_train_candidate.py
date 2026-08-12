"""Tests for scripts/monopolyzero_native_train_candidate.py:
- pure helper logic (_resolve_seeds, _legal_from_mask, _greedy_action,
  opportunity_greedy_stats) with synthetic fake positions/models, no engine
- generate_self_play_positions accumulation via a monkeypatched engine
- source-level guards: no ASU import, no monopoly_bench.training import
  (ASU_FROZEN_TEACHER only loads via that module - never allowed here),
  no fixed-rule/HYBRID_COMPAT usage, uses the native local_training_update
- CLI argument parsing
- main() wiring fully monkeypatched (no git-tree/real-checkpoint/real-engine
  dependency) - verifies --output/--checkpoint-output/config actually
  reflect what was requested
- one small REAL-engine smoke test (1 seed, tiny SearchConfig, short
  max_rounds, 2 training updates) - not a benchmark, not a long training run
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_native_train_candidate.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_native_train_candidate", SCRIPT)
module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_native_train_candidate"] = module
_spec.loader.exec_module(module)

BUY_ID, ACCEPT_ID = 3, 7


def _source_import_lines() -> list[str]:
    source = SCRIPT.read_text(encoding="utf-8")
    return [line for line in source.splitlines() if line.strip().startswith(("import ", "from "))]


def _mask(num_actions: int, legal_ids: set[int]) -> list[bool]:
    return [index in legal_ids for index in range(num_actions)]


# ── config reuse from 023 - not redefined ───────────────────────────────


def test_reuses_023_config_not_redefined():
    assert module.SEEDS == module.audit_v1.SEEDS == tuple(range(43000, 43020))
    assert module.MAX_ROUNDS == module.audit_v1.MAX_ROUNDS == 200
    assert module.NUM_SEATS == module.audit_v1.NUM_SEATS == 4
    assert module.CHECKPOINT_PATH == module.audit_v1.CHECKPOINT_PATH
    assert module.BASELINE_CHECKPOINT_SHA256 == module.audit_v1.BASELINE_CHECKPOINT_SHA256
    assert module.verify_baseline_checkpoint is module.audit_v1.verify_baseline_checkpoint
    source = SCRIPT.read_text(encoding="utf-8")
    assert "SEEDS = tuple(range(" not in source
    assert "def verify_baseline_checkpoint(" not in source


# ── source-level guards ─────────────────────────────────────────────────


def test_does_not_import_adapters_arena_training_or_asu():
    import_lines = _source_import_lines()
    forbidden = ("monopoly_bench.adapters", "monopoly_bench.arena", "monopoly_bench.training")
    hits = [line for line in import_lines if any(name in line for name in forbidden)]
    assert hits == [], f"found forbidden ASU-coupled module import(s): {hits}"
    assert not any("ASU_FROZEN_TEACHER" in line for line in import_lines)


def test_does_not_use_fixed_rule_or_hybrid_compat():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "fixed_buy_decision" not in source
    assert "fixed_accept_trade_decision" not in source
    assert "build_local_hybrid_compat_policy" not in source


def test_uses_native_local_training_update():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "common.local_training_update(" in source


def test_uses_replay_buffer_and_search_policy():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from monopoly_bench.storage import ReplayBuffer" in source
    assert "common.build_local_search_policy(" in source
    assert "self_play=True" in source


# ── _legal_from_mask / _greedy_action ──────────────────────────────────


def test_legal_from_mask():
    assert module._legal_from_mask([False, True, False, True, True]) == (1, 3, 4)


class _PreferModel:
    """Fake model: greedy-picks `preferred_id` whenever legal, else the
    smallest legal action id."""

    def __init__(self, preferred_id: int):
        self.preferred_id = preferred_id

    def predict(self, state, legal, actor_id):
        if self.preferred_id in legal:
            priors = {action: (1.0 if action == self.preferred_id else 0.01) for action in legal}
        else:
            best = min(legal)
            priors = {action: (1.0 if action == best else 0.01) for action in legal}
        return priors, (0.25, 0.25, 0.25, 0.25)


def test_greedy_action_returns_argmax():
    model = _PreferModel(ACCEPT_ID)
    assert module._greedy_action(model, state=None, legal=(BUY_ID, ACCEPT_ID, 1), actor_id=0) == ACCEPT_ID


# ── opportunity_greedy_stats ─────────────────────────────────────────────


def test_opportunity_greedy_stats_counts_and_rates():
    positions = [
        types.SimpleNamespace(legal_mask=_mask(10, {BUY_ID, 1}), state=None, actor_id=0),  # buy only, chosen
        types.SimpleNamespace(legal_mask=_mask(10, {BUY_ID, ACCEPT_ID}), state=None, actor_id=1),  # both
        types.SimpleNamespace(legal_mask=_mask(10, {1, 2}), state=None, actor_id=2),  # neither
    ]
    model = _PreferModel(BUY_ID)  # always prefers BUY_ID over ACCEPT_ID when both legal
    stats = module.opportunity_greedy_stats(positions, model, buy_id=BUY_ID, accept_id=ACCEPT_ID)
    assert stats["buy_property_opportunities"] == 2
    assert stats["buy_property_greedy_chosen"] == 2
    assert stats["buy_property_greedy_rate"] == pytest.approx(1.0)
    assert stats["accept_trade_opportunities"] == 1
    assert stats["accept_trade_greedy_chosen"] == 0  # BUY_ID beat ACCEPT_ID in the shared position
    assert stats["accept_trade_greedy_rate"] == pytest.approx(0.0)


def test_opportunity_greedy_stats_empty_positions_rates_are_none():
    stats = module.opportunity_greedy_stats([], _PreferModel(BUY_ID), buy_id=BUY_ID, accept_id=ACCEPT_ID)
    assert stats["buy_property_opportunities"] == 0
    assert stats["buy_property_greedy_rate"] is None
    assert stats["accept_trade_opportunities"] == 0
    assert stats["accept_trade_greedy_rate"] is None


# ── generate_self_play_positions (monkeypatched engine) ──────────────────


def test_generate_self_play_positions_accumulates_across_seeds(monkeypatch):
    fake_position = object()
    calls = []

    def fake_build_search_policy(model, search_config, self_play):
        assert self_play is True
        return "fake-policy"

    def fake_play_local_game(*, game_id, seed, policies, max_rounds, record_seats):
        calls.append(seed)
        assert record_seats == set(range(module.NUM_SEATS))
        return types.SimpleNamespace(
            illegal_actions=0, crashed=False, completed=True, decisions=42,
            winner=0, positions=[fake_position, fake_position],
        )

    monkeypatch.setattr(module.common, "build_local_search_policy", fake_build_search_policy)
    monkeypatch.setattr(module.common, "play_local_game", fake_play_local_game)

    result = module.generate_self_play_positions([100, 101], model=None, search_config=None, max_rounds=5)

    assert calls == [100, 101]
    assert len(result["positions"]) == 4
    assert len(result["per_game"]) == 2
    assert result["per_game"][0]["positions_collected"] == 2
    assert result["total_illegal"] == 0
    assert result["total_crashed"] == 0
    assert result["incomplete"] == 0


def test_generate_self_play_positions_flags_illegal_and_incomplete(monkeypatch):
    monkeypatch.setattr(module.common, "build_local_search_policy", lambda model, search_config, self_play: "policy")
    monkeypatch.setattr(
        module.common, "play_local_game",
        lambda **kwargs: types.SimpleNamespace(
            illegal_actions=1, crashed=False, completed=False, decisions=1, winner=None, positions=[],
        ),
    )
    result = module.generate_self_play_positions([1], model=None, search_config=None, max_rounds=5)
    assert result["total_illegal"] == 1
    assert result["incomplete"] == 1


# ── _resolve_seeds ──────────────────────────────────────────────────────


def test_resolve_seeds_defaults_to_023_range():
    assert module._resolve_seeds(None, None) == list(module.SEEDS)


def test_resolve_seeds_custom_range():
    assert module._resolve_seeds(43005, 3) == [43005, 43006, 43007]


def test_resolve_seeds_requires_both_args_together():
    with pytest.raises(SystemExit):
        module._resolve_seeds(43005, None)
    with pytest.raises(SystemExit):
        module._resolve_seeds(None, 3)


# ── CLI argument parsing ─────────────────────────────────────────────────


def test_arg_parser_defaults():
    parser = module.build_arg_parser()
    args = parser.parse_args([])
    assert args.seed_start is None
    assert args.seed_count is None
    assert args.simulations == module.DEFAULT_SIMULATIONS == 32
    assert args.max_rounds == module.MAX_ROUNDS == 200
    assert args.updates == module.DEFAULT_UPDATES
    assert args.output is None
    assert args.checkpoint_output == module.DEFAULT_CHECKPOINT_OUTPUT


def test_arg_parser_parses_all_flags():
    parser = module.build_arg_parser()
    args = parser.parse_args(
        [
            "--seed-start", "43000", "--seed-count", "1",
            "--simulations", "32", "--max-rounds", "20", "--updates", "2",
            "--output", "out.json", "--checkpoint-output", "cand.pt",
        ]
    )
    assert args.seed_start == 43000
    assert args.seed_count == 1
    assert args.simulations == 32
    assert args.max_rounds == 20
    assert args.updates == 2
    assert args.output == Path("out.json")
    assert args.checkpoint_output == Path("cand.pt")


# ── main() wiring (fully monkeypatched - no git-tree/real engine cost) ──


def test_main_writes_output_and_checkpoint_with_requested_config(monkeypatch, tmp_path):
    module.common.ensure_reference_on_path()
    from monopoly_bench.model import MonopolyZeroNet

    monkeypatch.setattr(module.common, "require_pinned_hash_seed", lambda name: None)
    monkeypatch.setattr(module.common, "require_clean_git_tree", lambda name: "a" * 40)
    monkeypatch.setattr(module.ep, "require_seed_scope", lambda seeds, seed_class, *, context: None)
    monkeypatch.setattr(module, "verify_baseline_checkpoint", lambda: "b" * 64)
    monkeypatch.setattr(module.common, "loaded_asu_modules", lambda: [])

    class _FakeModel:
        def eval(self):
            return self

        def save_inference(self, path, metadata):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"fake-checkpoint-bytes")

    monkeypatch.setattr(MonopolyZeroNet, "load_inference", classmethod(lambda cls, path: _FakeModel()))

    fake_positions = [object(), object()]
    fake_generation = {
        "per_game": [
            {"seed": 43000, "completed": True, "decisions": 5, "positions_collected": 2,
             "winner": 0, "illegal_actions": 0, "crashed": False}
        ],
        "positions": fake_positions, "total_illegal": 0, "total_crashed": 0, "incomplete": 0,
    }
    monkeypatch.setattr(
        module, "generate_self_play_positions",
        lambda seeds, model, search_config, max_rounds: fake_generation,
    )

    stats_calls = []

    def fake_opportunity_greedy_stats(positions, model, *, buy_id, accept_id):
        stats_calls.append(positions)
        return {
            "buy_property_opportunities": 1, "buy_property_greedy_chosen": 0, "buy_property_greedy_rate": 0.0,
            "accept_trade_opportunities": 1, "accept_trade_greedy_chosen": 0, "accept_trade_greedy_rate": 0.0,
        }

    monkeypatch.setattr(module, "opportunity_greedy_stats", fake_opportunity_greedy_stats)
    monkeypatch.setattr(
        module, "train_candidate",
        lambda model, positions, *, updates, batch_size, seed: (
            [{"loss": 1.0, "policy_loss": 0.5, "value_loss": 0.5, "gradient_norm": 0.1}] * updates
        ),
    )

    output_path = tmp_path / "out.json"
    checkpoint_path = tmp_path / "candidate.pt"

    exit_code = module.main(
        [
            "--seed-start", "43000", "--seed-count", "1",
            "--simulations", "32", "--max-rounds", "20", "--updates", "3",
            "--output", str(output_path), "--checkpoint-output", str(checkpoint_path),
        ]
    )

    assert exit_code == 0
    assert checkpoint_path.is_file()
    assert output_path.is_file()

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["status"] == "OK"
    assert written["config"]["seeds"] == [43000]
    assert written["config"]["max_rounds"] == 20
    assert written["config"]["updates"] == 3
    assert written["config"]["search_config"]["simulations"] == 32
    assert len(written["train_stats"]) == 3
    assert written["checkpoint_sha256"] == hashlib.sha256(b"fake-checkpoint-bytes").hexdigest()
    assert len(stats_calls) == 2  # before + after


def test_main_stops_before_training_on_self_play_failure(monkeypatch, tmp_path):
    module.common.ensure_reference_on_path()
    from monopoly_bench.model import MonopolyZeroNet

    monkeypatch.setattr(module.common, "require_pinned_hash_seed", lambda name: None)
    monkeypatch.setattr(module.common, "require_clean_git_tree", lambda name: "a" * 40)
    monkeypatch.setattr(module.ep, "require_seed_scope", lambda seeds, seed_class, *, context: None)
    monkeypatch.setattr(module, "verify_baseline_checkpoint", lambda: "b" * 64)

    monkeypatch.setattr(MonopolyZeroNet, "load_inference", classmethod(lambda cls, path: types.SimpleNamespace(eval=lambda: None)))

    monkeypatch.setattr(
        module, "generate_self_play_positions",
        lambda seeds, model, search_config, max_rounds: {
            "per_game": [], "positions": [], "total_illegal": 1, "total_crashed": 0, "incomplete": 0,
        },
    )
    called = {"train": False}
    monkeypatch.setattr(
        module, "train_candidate",
        lambda *a, **k: called.__setitem__("train", True),
    )

    with pytest.raises(RuntimeError, match="Stopping before training"):
        module.main(["--seed-start", "43000", "--seed-count", "1"])

    assert called["train"] is False


# ── tiny REAL-engine smoke run (not a benchmark, not a long training run) ─


def test_real_engine_smoke_one_seed_tiny_training_and_checkpoint_roundtrip(tmp_path):
    """Genuinely plays 1 seed of self-play (tiny SearchConfig, short
    max_rounds), runs 2 native training updates, measures before/after
    greedy BUY/ACCEPT rates on the same positions, and round-trips a saved
    checkpoint through save_inference/load_inference + SHA-256. Calls the
    sub-functions directly (like the gradient diagnostic's own real smoke
    test) rather than main(), to avoid main()'s git-clean-tree requirement
    during local dev-loop testing. Not a benchmark, not a long training run."""
    module.common.ensure_reference_on_path()
    import random

    import numpy as np
    import torch

    from monopoly_bench.config import SearchConfig
    from monopoly_bench.model import MonopolyZeroNet
    from monopoly_game_engine.actions import ActionType

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    if not module.CHECKPOINT_PATH.is_file():
        pytest.skip("baseline_pretraining.pt not present in this environment")

    model = MonopolyZeroNet.load_inference(module.CHECKPOINT_PATH)
    model.eval()
    search_config = SearchConfig(simulations=4, max_depth=16)
    buy_id = int(ActionType.BUY_PROPERTY)
    accept_id = int(ActionType.ACCEPT_TRADE)

    generation = module.generate_self_play_positions([43000], model, search_config, max_rounds=5)
    assert generation["total_illegal"] == 0
    assert generation["total_crashed"] == 0
    assert generation["incomplete"] == 0
    positions = generation["positions"]
    assert positions

    before_stats = module.opportunity_greedy_stats(positions, model, buy_id=buy_id, accept_id=accept_id)

    before_params = {name: p.detach().clone() for name, p in model.named_parameters()}
    train_stats = module.train_candidate(model, positions, updates=2, batch_size=8, seed=0)
    assert len(train_stats) == 2
    for update in train_stats:
        for key in ("loss", "policy_loss", "value_loss", "gradient_norm"):
            assert torch.isfinite(torch.tensor(update[key]))
    changed = [
        name for name, p in model.named_parameters()
        if not torch.equal(before_params[name], p.detach())
    ]
    assert changed  # at least one parameter actually moved

    model.eval()
    after_stats = module.opportunity_greedy_stats(positions, model, buy_id=buy_id, accept_id=accept_id)
    assert set(after_stats) == set(before_stats)

    checkpoint_path = tmp_path / "candidate.pt"
    model.save_inference(checkpoint_path, {"source": "test"})
    sha = module._sha256(checkpoint_path)
    assert len(sha) == 64

    reloaded = MonopolyZeroNet.load_inference(checkpoint_path)
    reloaded.eval()
    reloaded_params = dict(reloaded.named_parameters())
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), reloaded_params[name].detach())
