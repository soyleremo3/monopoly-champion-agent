"""Tests for scripts/monopolyzero_a96_vs_asu_robustness_eval.py:
- pure helpers (resolve_asu_class, per_seat_win_breakdown)
- hash-gate STOP behavior (champion checkpoint/actor mismatch, ASU frozen
  spec mismatch) - SystemExit, never a silent proceed
- one small REAL-engine smoke game (tiny hidden_dim actor, not the real
  checkpoint; real ASUValueV1 as the three opponents) exercising
  play_one_game_vs_asu() end to end through the real engine
- CLI argument parsing
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "monopolyzero_a96_vs_asu_robustness_eval.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import monopolyzero_common as common  # noqa: E402

_spec = importlib.util.spec_from_file_location("monopolyzero_a96_vs_asu_robustness_eval", SCRIPT)
module = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_a96_vs_asu_robustness_eval"] = module
_spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def _no_asu_module_leakage():
    """This is the one test file in the suite that legitimately imports
    real ASU_FROZEN_TEACHER submodules (this script's whole purpose is
    using ASU as an opponent). Without cleanup, those imports stick in
    `sys.modules` for the rest of the pytest PROCESS (module caching is
    process-wide, not per-file), which would break every other file's
    "ASU_FROZEN_TEACHER is not loaded" guard test
    (test_monopolyzero_common.py::test_asu_module_guard_clean_when_absent
    and the ASU-free-guard assertions in other gate/screen scripts' own
    smoke tests) if they happen to run later in the same session. Strips
    only the ASU_FROZEN_TEACHER entries THIS test added, after it runs."""
    before = set(sys.modules)
    yield
    for name in list(sys.modules):
        if name not in before and (name == "ASU_FROZEN_TEACHER" or name.startswith("ASU_FROZEN_TEACHER.")):
            del sys.modules[name]


# ── pure helpers ─────────────────────────────────────────────────────────


def test_resolve_asu_class_returns_expected_classes():
    common.ensure_reference_on_path()
    from ASU_FROZEN_TEACHER import ASURolloutV1, ASUValueV1

    assert module.resolve_asu_class("asu-value-v1") is ASUValueV1
    assert module.resolve_asu_class("asu-rollout-v1") is ASURolloutV1


def test_resolve_asu_class_rejects_unknown_id():
    with pytest.raises(ValueError):
        module.resolve_asu_class("not-a-real-policy")


def test_per_seat_win_breakdown_counts_by_physical_seat():
    games = [
        {"focus_seat": 0, "per_seat": {0: {"win": True}}},
        {"focus_seat": 0, "per_seat": {0: {"win": False}}},
        {"focus_seat": 1, "per_seat": {1: {"win": True}}},
    ]
    breakdown = module.per_seat_win_breakdown(games)
    assert breakdown["0"] == {"games": 2, "wins": 1, "win_rate": 0.5}
    assert breakdown["1"] == {"games": 1, "wins": 1, "win_rate": 1.0}
    assert breakdown["2"] == {"games": 0, "wins": 0, "win_rate": None}
    assert breakdown["3"] == {"games": 0, "wins": 0, "win_rate": None}


# ── hash gates: STOP, never silent proceed ─────────────────────────────────


def test_load_and_verify_champion_raises_system_exit_on_checkpoint_hash_mismatch(tmp_path):
    fake_checkpoint = tmp_path / "fake.pt"
    fake_checkpoint.write_bytes(b"not a real checkpoint")
    with pytest.raises(SystemExit, match="checkpoint sha256 mismatch"):
        module.load_and_verify_champion(fake_checkpoint)


def test_load_and_verify_champion_raises_system_exit_on_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="checkpoint missing"):
        module.load_and_verify_champion(tmp_path / "does_not_exist.pt")


def test_verify_asu_frozen_spec_raises_system_exit_on_mismatch(monkeypatch):
    monkeypatch.setattr(module, "EXPECTED_ASU_FROZEN_SPEC_HASH", "0" * 64)
    with pytest.raises(SystemExit, match="FROZEN_SPEC_HASH mismatch"):
        module.verify_asu_frozen_spec()


def test_verify_asu_frozen_spec_passes_on_pinned_reference():
    common.ensure_reference_on_path()
    assert module.verify_asu_frozen_spec() == module.EXPECTED_ASU_FROZEN_SPEC_HASH


# ── real-engine smoke ────────────────────────────────────────────────────


def test_real_engine_smoke_one_game_vs_asu_value_v1():
    """Plays one real (tiny hidden_dim, not the real checkpoint) game end
    to end through play_one_game_vs_asu() with REAL ASUValueV1 opponents -
    proves the _ASUOpponentPolicy adapter is wired correctly against the
    real engine (legal actions only, counters populated, env_holder
    shared). Not a strength claim, not the real 16-game family."""
    import torch

    common.ensure_reference_on_path()
    from ASU_FROZEN_TEACHER import ASUValueV1
    from monopoly_game_engine.networks import ActorNetwork

    torch.manual_seed(0)
    champion_actor = ActorNetwork(hidden_dim=8)
    champion_actor.eval()

    result = module.play_one_game_vs_asu(
        game_id=1, seed=53000, champion_actor=champion_actor, asu_class=ASUValueV1,
        focus_seat=0, device=torch.device("cpu"), max_rounds=3,
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
        assert stats["decisions"] >= 1
    assert result["winner"] in (0, 1, 2, 3)
    assert result["round_cap_hit"] == (result["final_round"] >= 3)


# ── CLI ──────────────────────────────────────────────────────────────────


def test_cli_defaults():
    args = module.build_arg_parser().parse_args([])
    assert args.asu_policy == "asu-value-v1"
    assert args.seed_base == module.DEV_SEED_BASE == 53000
    assert args.n_seeds == module.N_SEEDS == 4
    assert args.device == "cpu"
    assert args.output is None


def test_cli_overrides():
    args = module.build_arg_parser().parse_args(
        ["--asu-policy", "asu-rollout-v1", "--seed-base", "1", "--n-seeds", "2", "--output", "out.json"]
    )
    assert args.asu_policy == "asu-rollout-v1"
    assert args.seed_base == 1
    assert args.n_seeds == 2
    assert args.output == Path("out.json")


def test_cli_rejects_unknown_asu_policy():
    with pytest.raises(SystemExit):
        module.build_arg_parser().parse_args(["--asu-policy", "not-real"])
