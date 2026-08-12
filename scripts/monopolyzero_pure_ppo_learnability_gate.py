"""PURE PPO BUY_PROPERTY/ACCEPT_TRADE learnability gate (hybrid=False).

Every prior BUY/TRADE finding in this project (`023`/`024`/`025`/`026`)
measured `POLICY_ONLY`/`HYBRID_COMPAT` inference over a checkpoint whose
actor was bootstrapped from a **hybrid** PPO run - one where
`BUY_PROPERTY`/`ACCEPT_TRADE` were always handled by fixed rules
(`agent_ppo.py`'s `fixed_action_mask`) and therefore never received a PPO
gradient update at all. This script asks the more basic question those
findings never actually tested: if PPO is allowed to touch these two
actions from the start (`hybrid=False`), does gradient descent move them at
all over a small number of games?

Uses the reference's own, unmodified, ASU-free training primitives -
`monopoly_game_engine.agent_ppo.PPOAgent` and
`monopoly_game_engine.train.train()` - exactly as shipped (grepped clean of
"asu"/"ASU" at every module in this import chain: `agent_ppo.py`,
`train.py`, `networks.py`, `agents_fixed.py`, `env.py`, `state.py`,
`actions.py`, `constants.py`). Zero MCTS, zero DDQN, zero HYBRID_COMPAT,
zero fixed BUY/TRADE rule (`hybrid=False` means `PPOAgent.fixed_action_mask`
is all-`False` and both hybrid interception branches in `choose_action` are
structurally unreachable - see the source, not just this docstring).

This project owns exactly three small pieces of instrumentation around
those unmodified reference primitives, none of which change their
behavior:
  1. `_instrument_agent` wraps the `PPOAgent` instance's own
     `choose_action`/`store` bound methods (assigned as new instance
     attributes, the original reference class is untouched) to record,
     from real runtime data, whether `BUY_PROPERTY`/`ACCEPT_TRADE` were
     ever excluded from the neural candidate set when legal, and whether
     they were ever actually stored into a PPO transition (which by
     construction only happens with a non-`None` log_prob - see Bug 1 in
     `agent_ppo.py`'s own docstring).
  2. `_install_fallback_counting_fixed_agents` monkeypatches
     `monopoly_game_engine.train`'s own `FPAgentA`/`FPAgentB`/`FPAgentC`
     module-global bindings (not the classes themselves) to small
     subclasses that additionally count when the fixed opponent proposed
     something `train.py`'s own `run_episode` had to silently substitute -
     the vanilla reference `train()` does not expose this number itself,
     unlike `monopoly_bench.play_local_game`'s `fallback_count`. Restored
     after the run. Same "patch the module-global name, call time resolves
     it" mechanism `monopolyzero_common.patch_actor_relative_owner_encoding`
     already uses for the actor-relative owner-encoding fix, applied here
     to a second, unrelated reference module.
  3. `measure_actor_on_positions` runs the trained actor's own masked
     `log_softmax` forward pass (`ActorNetwork.forward`, unedited) over a
     fixed, read-only diagnostic set - never a backward pass, never used as
     training data.

Diagnostic set: ONLY `actor_id == 0` positions from experiment `026`'s
55,215-position replay
(`artifacts/monopolyzero_native_train_candidate/buy_trade_learnability_v2/replay`).
`026` is PRE-FIX diagnostic-only data for training purposes (see
`docs/DECISIONS.md`'s 2026-08-12 owner-encoding-fix entry) - but
`agent_id == 0`'s state encoding is byte-for-byte IDENTICAL before and
after that fix (`actor_order(0) == (0, 1, 2, 3)`, the identity ordering the
bug coincidentally already produced for physical id 0), so this specific
subset is safe to read as a corrected-state-consistent diagnostic input.
Never written to, never gradient-trained on here.

The learner occupies `player_id=0` for the SAME reason: not because the
fix matters for its own live game states (it doesn't, for player 0), but
so the training run and the diagnostic read use one consistent pipeline
end to end. `monopolyzero_common.ensure_reference_on_path()` (which installs
the actor-relative owner-encoding patch) is called before any reference
import, exactly like every other script in this project.

Training: 32 full games, `PPOAgent(player_id=0, hybrid=False, device="cpu")`,
a fixed DEV seed, against the reference's own `FPAgentA`/`FPAgentB`/
`FPAgentC` fixed opponents (`train()`'s hardcoded, unmodified opponent
pool - `max_rounds=200` is likewise hardcoded inside `train()` itself, not
a parameter this script re-specifies). `log_every=1` so `train()`'s own
history dict returns genuinely per-game (not windowed) win/trade/property
figures - reusing its existing aggregation rather than duplicating it.

This is a small learnability gate, not a strength claim: no evaluation,
no A/B, no promotion/GO-KILL verdict computed by this script itself (same
discipline as every other diagnostic in this project) - see
docs/EXPERIMENTS.md for the pre-registered decision rule this gate's output
is meant to be read against.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluation_protocol as ep  # noqa: E402
import monopolyzero_common as common  # noqa: E402

DEV_SEED_BASE = 45000
N_GAMES = 32

DIAGNOSTIC_REPLAY_DIR = (
    common.REPO_ROOT
    / "artifacts"
    / "monopolyzero_native_train_candidate"
    / "buy_trade_learnability_v2"
    / "replay"
)
DEFAULT_CHECKPOINT_OUTPUT = (
    common.REPO_ROOT / "artifacts" / "monopolyzero_pure_ppo_learnability_gate" / "candidate_ppo.pt"
)


# ── pure helpers (no engine/torch import at module scope) ──────────────────


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _legal_from_mask(mask) -> tuple[int, ...]:
    return tuple(int(index) for index, flag in enumerate(mask) if flag)


def _rank(action: int, legal: tuple[int, ...], prob_of: dict[int, float]) -> int:
    """1-indexed rank of `action`'s probability among `legal` (rank 1 =
    highest), ties broken by lower action id - same convention
    `monopolyzero_native_train_candidate._prior_rank` uses."""
    ordered = sorted(legal, key=lambda candidate: (-prob_of[candidate], candidate))
    return ordered.index(action) + 1


def _seed_range(base: int, count: int) -> list[int]:
    return list(range(base, base + count))


# ── actor snapshot/hash proof ───────────────────────────────────────────────


def _full_actor_sha256(actor) -> str:
    import hashlib

    digest = hashlib.sha256()
    state_dict = actor.state_dict()
    for key in sorted(state_dict):
        digest.update(key.encode("utf-8"))
        digest.update(state_dict[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _action_row(actor, action_id: int):
    """The final Linear layer's weight row + bias scalar for one action id -
    the only parameters a hybrid-PPO fixed_action_mask would have kept
    permanently untrained for BUY_PROPERTY/ACCEPT_TRADE (see 023's finding).
    Returns (weight_row: np.ndarray[hidden_dim], bias: float)."""
    final_layer = actor.net[-1]
    weight_row = final_layer.weight.detach().cpu().numpy()[action_id].copy()
    bias = float(final_layer.bias.detach().cpu().numpy()[action_id])
    return weight_row, bias


def _row_sha256(weight_row, bias: float) -> str:
    import hashlib

    import numpy as np

    digest = hashlib.sha256()
    digest.update(weight_row.tobytes())
    digest.update(np.float64(bias).tobytes())
    return digest.hexdigest()


def snapshot_actor(actor, *, buy_id: int, accept_id: int) -> dict:
    import numpy as np

    buy_row, buy_bias = _action_row(actor, buy_id)
    accept_row, accept_bias = _action_row(actor, accept_id)
    return {
        "full_actor_sha256": _full_actor_sha256(actor),
        "buy_property_row_sha256": _row_sha256(buy_row, buy_bias),
        "accept_trade_row_sha256": _row_sha256(accept_row, accept_bias),
        "_buy_property_row": buy_row,
        "_buy_property_bias": buy_bias,
        "_accept_trade_row": accept_row,
        "_accept_trade_bias": accept_bias,
    }


def compare_snapshots(before: dict, after: dict) -> dict:
    import numpy as np

    buy_delta = float(
        np.linalg.norm(
            np.concatenate([after["_buy_property_row"] - before["_buy_property_row"],
                             [after["_buy_property_bias"] - before["_buy_property_bias"]]])
        )
    )
    accept_delta = float(
        np.linalg.norm(
            np.concatenate([after["_accept_trade_row"] - before["_accept_trade_row"],
                             [after["_accept_trade_bias"] - before["_accept_trade_bias"]]])
        )
    )
    return {
        "full_actor_changed": before["full_actor_sha256"] != after["full_actor_sha256"],
        "buy_property_row_changed": before["buy_property_row_sha256"] != after["buy_property_row_sha256"],
        "accept_trade_row_changed": before["accept_trade_row_sha256"] != after["accept_trade_row_sha256"],
        "buy_property_row_l2_delta": buy_delta,
        "accept_trade_row_l2_delta": accept_delta,
        "before": {k: v for k, v in before.items() if not k.startswith("_")},
        "after": {k: v for k, v in after.items() if not k.startswith("_")},
    }


# ── diagnostic measurement over a fixed, read-only position set ────────────


def load_actor0_diagnostic_positions(replay_dir: Path) -> list:
    """Loads experiment 026's replay and returns ONLY actor_id==0 records.
    Read-only: never appended to, never used as a training source."""
    from monopoly_bench.storage import ReplayBuffer

    if not (replay_dir / "metadata.json").is_file():
        raise SystemExit(f"No replay buffer found at {replay_dir} (missing metadata.json)")
    replay = ReplayBuffer(replay_dir, create=False)
    return [position for position in replay.records() if position.actor_id == 0]


def measure_actor_on_positions(
    actor, positions: list, *, buy_id: int, accept_id: int, device, batch_size: int = 1024,
) -> dict:
    """Masked log_softmax forward pass only (no sampling, no backward).
    Returns overall + per-game_id BUY_PROPERTY/ACCEPT_TRADE stats:
    opportunity count, mean raw probability, mean legal-action rank, and
    greedy-choice rate."""
    import numpy as np
    import torch

    buy_total = 0
    buy_greedy = 0
    buy_priors: list[float] = []
    buy_ranks: list[int] = []
    accept_total = 0
    accept_greedy = 0
    accept_priors: list[float] = []
    accept_ranks: list[int] = []
    per_game: dict[int, dict] = {}

    def _game_bucket(game_id: int) -> dict:
        return per_game.setdefault(
            game_id,
            {
                "buy_property_opportunities": 0, "buy_property_greedy_chosen": 0,
                "buy_property_priors": [], "buy_property_ranks": [],
                "accept_trade_opportunities": 0, "accept_trade_greedy_chosen": 0,
                "accept_trade_priors": [], "accept_trade_ranks": [],
            },
        )

    actor.eval()
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            chunk = positions[start:start + batch_size]
            states_t = torch.as_tensor(
                np.stack([position.state for position in chunk]), dtype=torch.float32, device=device,
            )
            masks_t = torch.as_tensor(
                np.stack([position.legal_mask for position in chunk]), dtype=torch.bool, device=device,
            )
            log_probs = actor(states_t, masks_t)
            probs = log_probs.exp()

            for row, position in enumerate(chunk):
                legal = _legal_from_mask(position.legal_mask)
                row_probs = probs[row]
                prob_of = {action: float(row_probs[action]) for action in legal}
                greedy_action = max(prob_of, key=prob_of.get)
                bucket = _game_bucket(position.game_id)

                if buy_id in legal:
                    prior = prob_of[buy_id]
                    rank = _rank(buy_id, legal, prob_of)
                    is_greedy = greedy_action == buy_id
                    buy_total += 1
                    buy_priors.append(prior)
                    buy_ranks.append(rank)
                    buy_greedy += int(is_greedy)
                    bucket["buy_property_opportunities"] += 1
                    bucket["buy_property_priors"].append(prior)
                    bucket["buy_property_ranks"].append(rank)
                    bucket["buy_property_greedy_chosen"] += int(is_greedy)

                if accept_id in legal:
                    prior = prob_of[accept_id]
                    rank = _rank(accept_id, legal, prob_of)
                    is_greedy = greedy_action == accept_id
                    accept_total += 1
                    accept_priors.append(prior)
                    accept_ranks.append(rank)
                    accept_greedy += int(is_greedy)
                    bucket["accept_trade_opportunities"] += 1
                    bucket["accept_trade_priors"].append(prior)
                    bucket["accept_trade_ranks"].append(rank)
                    bucket["accept_trade_greedy_chosen"] += int(is_greedy)

    per_game_summary = {}
    for game_id, bucket in sorted(per_game.items()):
        per_game_summary[str(game_id)] = {
            "buy_property_opportunities": bucket["buy_property_opportunities"],
            "buy_property_greedy_rate": (
                bucket["buy_property_greedy_chosen"] / bucket["buy_property_opportunities"]
                if bucket["buy_property_opportunities"] else None
            ),
            "buy_property_mean_raw_prior": _mean(bucket["buy_property_priors"]),
            "buy_property_mean_prior_rank": _mean(bucket["buy_property_ranks"]),
            "accept_trade_opportunities": bucket["accept_trade_opportunities"],
            "accept_trade_greedy_rate": (
                bucket["accept_trade_greedy_chosen"] / bucket["accept_trade_opportunities"]
                if bucket["accept_trade_opportunities"] else None
            ),
            "accept_trade_mean_raw_prior": _mean(bucket["accept_trade_priors"]),
            "accept_trade_mean_prior_rank": _mean(bucket["accept_trade_ranks"]),
        }

    return {
        "buy_property_opportunities": buy_total,
        "buy_property_greedy_chosen": buy_greedy,
        "buy_property_greedy_rate": (buy_greedy / buy_total) if buy_total else None,
        "buy_property_mean_raw_prior": _mean(buy_priors),
        "buy_property_mean_prior_rank": _mean(buy_ranks),
        "accept_trade_opportunities": accept_total,
        "accept_trade_greedy_chosen": accept_greedy,
        "accept_trade_greedy_rate": (accept_greedy / accept_total) if accept_total else None,
        "accept_trade_mean_raw_prior": _mean(accept_priors),
        "accept_trade_mean_prior_rank": _mean(accept_ranks),
        "per_game_id": per_game_summary,
    }


# ── training-time instrumentation (wraps instance methods only) ────────────


def instrument_agent(agent, *, buy_id: int, accept_id: int) -> dict:
    """Wraps `agent`'s own bound `choose_action`/`store` methods as new
    instance attributes (the PPOAgent class itself is untouched) to record,
    from real runtime data:
      - whether BUY_PROPERTY/ACCEPT_TRADE were ever excluded from the
        neural candidate set (`nn_allowed`) when legal - proves item 2 of
        the pre-registered decision rule from actual training, not just
        `hybrid=False`'s structural guarantee.
      - how many times each action was actually stored into a PPO
        transition, which by `agent_ppo.py`'s own Bug-1 fix only ever
        happens with a non-None log_prob - proves both actions "entered
        neural PPO transitions with non-null log_prob at least once when
        selected", if it happened.
    Returns the mutable counters dict (updated in place as training runs).
    """
    counters = {
        "buy_property_opportunities_seen": 0,
        "buy_property_excluded_from_nn": 0,
        "accept_trade_opportunities_seen": 0,
        "accept_trade_excluded_from_nn": 0,
        "store_calls": 0,
        "buy_property_stored_with_log_prob": 0,
        "accept_trade_stored_with_log_prob": 0,
    }

    original_choose_action = agent.choose_action
    original_store = agent.store

    def choose_action(state, env, allowed_actions):
        result = original_choose_action(state, env, allowed_actions)
        _action, _log_prob, _value, nn_allowed = result
        if buy_id in allowed_actions:
            counters["buy_property_opportunities_seen"] += 1
            if buy_id not in nn_allowed:
                counters["buy_property_excluded_from_nn"] += 1
        if accept_id in allowed_actions:
            counters["accept_trade_opportunities_seen"] += 1
            if accept_id not in nn_allowed:
                counters["accept_trade_excluded_from_nn"] += 1
        return result

    def store(state, action, log_prob, reward, value, done, allowed_actions):
        counters["store_calls"] += 1
        if action == buy_id:
            counters["buy_property_stored_with_log_prob"] += 1
        elif action == accept_id:
            counters["accept_trade_stored_with_log_prob"] += 1
        return original_store(state, action, log_prob, reward, value, done, allowed_actions)

    agent.choose_action = choose_action
    agent.store = store
    return counters


def install_fallback_counting_fixed_agents():
    """Monkeypatches `monopoly_game_engine.train`'s own module-global
    `FPAgentA`/`FPAgentB`/`FPAgentC` bindings (not the reference classes -
    those stay untouched) to subclasses that additionally count when
    `train.py`'s own run_episode had to substitute the fixed agent's
    proposed action for an illegal one. `train()` reads these names as
    bare globals at call time (`fp_classes = [FPAgentA, FPAgentB,
    FPAgentC]`), the same "patch the module-global name" mechanism
    `monopolyzero_common.patch_actor_relative_owner_encoding` already uses
    for the owner-encoding fix - applied here to a second, unrelated
    reference module. Returns (counters dict, restore callable).
    """
    import sys

    from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC

    # NOT `import monopoly_game_engine.train as train_module`: that resolves
    # via attribute access on the `monopoly_game_engine` package, and the
    # package's own __init__.py does `from .train import train`, which
    # overwrites the package's `train` attribute from "the train.py
    # submodule" to "the train() function" as a side effect - `import a.b as
    # x` would then silently bind `x` to that function, not the module.
    # sys.modules keyed by the full dotted path is unambiguous.
    train_module = sys.modules["monopoly_game_engine.train"]

    counters = {"fallback_substitutions": 0}
    originals = (train_module.FPAgentA, train_module.FPAgentB, train_module.FPAgentC)

    def _counting(base_class):
        class _Counting(base_class):
            def choose_action(self, env):
                proposed = super().choose_action(env)
                allowed = env.get_allowed_actions(self.player_id)
                if proposed not in allowed:
                    counters["fallback_substitutions"] += 1
                return proposed

        return _Counting

    train_module.FPAgentA = _counting(FPAgentA)
    train_module.FPAgentB = _counting(FPAgentB)
    train_module.FPAgentC = _counting(FPAgentC)

    def restore() -> None:
        train_module.FPAgentA, train_module.FPAgentB, train_module.FPAgentC = originals

    return counters, restore


# ── gate orchestration ──────────────────────────────────────────────────────


def run_gate(*, n_games: int, seed_base: int, diagnostic_positions: list, checkpoint_output: Path) -> dict:
    """Runs the full gate: snapshot -> before-diagnostic -> N-game pure PPO
    training (hybrid=False, player_id=0, vs FPAgentA/B/C) -> after-diagnostic
    -> snapshot comparison. Never touches `diagnostic_positions` as training
    data - only forward passes."""
    import torch

    from monopoly_game_engine.agent_ppo import PPOAgent
    from monopoly_game_engine.actions import ActionType
    from monopoly_game_engine.train import train as reference_train

    buy_id = int(ActionType.BUY_PROPERTY)
    accept_id = int(ActionType.ACCEPT_TRADE)
    device = torch.device("cpu")

    agent = PPOAgent(player_id=0, hybrid=False, device="cpu")
    if agent.hybrid is not False or bool(agent.fixed_action_mask.any()):
        raise RuntimeError("PPOAgent was not constructed with hybrid disabled - refusing to run")

    before_snapshot = snapshot_actor(agent.actor, buy_id=buy_id, accept_id=accept_id)
    before_diagnostic = measure_actor_on_positions(
        agent.actor, diagnostic_positions, buy_id=buy_id, accept_id=accept_id, device=device,
    )

    training_counters = instrument_agent(agent, buy_id=buy_id, accept_id=accept_id)
    fallback_counters, restore_fixed_agents = install_fallback_counting_fixed_agents()

    training_started = time.perf_counter()
    try:
        history = reference_train(
            agent, is_ppo=True, hybrid=False, n_games=n_games, log_every=1, seed=seed_base,
        )
    finally:
        restore_fixed_agents()
    training_elapsed_s = time.perf_counter() - training_started

    agent.actor.eval()
    after_snapshot = snapshot_actor(agent.actor, buy_id=buy_id, accept_id=accept_id)
    after_diagnostic = measure_actor_on_positions(
        agent.actor, diagnostic_positions, buy_id=buy_id, accept_id=accept_id, device=device,
    )

    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(checkpoint_output))

    asu_modules_loaded = common.loaded_asu_modules()

    win_rates = history.get("win_rates", [])
    wins_per_game = [1 if rate > 0 else 0 for rate in win_rates]

    return {
        "config": {
            "player_id": agent.player_id,
            "hybrid": agent.hybrid,
            "device": str(agent.device),
            "n_games": n_games,
            "seed_base": seed_base,
            "episode_seeds": _seed_range(seed_base, n_games),
            "max_rounds": 200,
            "opponents": "FPAgentA/FPAgentB/FPAgentC (reference monopoly_game_engine.agents_fixed, unmodified)",
            "n_steps": agent.n_steps, "n_epochs": agent.n_epochs, "batch_size": agent.batch_size,
            "gamma": agent.gamma, "lam": agent.lam, "clip_eps": agent.clip_eps,
            "entropy_coef": agent.entropy_coef, "value_coef": agent.value_coef,
            "max_grad_norm": agent.max_grad_norm, "win_loss_bonus": agent.win_loss_bonus,
            "hidden_dim": agent.hidden_dim,
        },
        "hybrid_inactive_proof": {
            "agent_hybrid_flag": agent.hybrid,
            "fixed_action_mask_any_true": bool(agent.fixed_action_mask.any()),
            "fixed_action_mask_buy_property": bool(agent.fixed_action_mask[buy_id]),
            "fixed_action_mask_accept_trade": bool(agent.fixed_action_mask[accept_id]),
        },
        "diagnostic_set": {
            "source": "experiment 026 replay, actor_id==0 positions only",
            "replay_dir": str(DIAGNOSTIC_REPLAY_DIR),
            "positions": len(diagnostic_positions),
            "game_ids": sorted({int(position.game_id) for position in diagnostic_positions}),
        },
        "before_diagnostic": before_diagnostic,
        "after_diagnostic": after_diagnostic,
        "actor_change_proof": compare_snapshots(before_snapshot, after_snapshot),
        "training_instrumentation": training_counters,
        "opponent_fallback_instrumentation": fallback_counters,
        "training_history": {
            "games": history.get("games", []),
            "win_rates_per_game_pct": win_rates,
            "wins_per_game": wins_per_game,
            "total_wins": sum(wins_per_game),
            "avg_properties_acquired_per_game": history.get("avg_properties_acquired", []),
            "avg_trades_initiated_per_game": history.get("avg_trades_initiated", []),
            "avg_trades_accepted_per_game": history.get("avg_trades_accepted", []),
            "avg_trades_declined_per_game": history.get("avg_trades_declined", []),
            "stopped_early": bool(history.get("stopped_early", False)),
            "stop_reason": history.get("stop_reason"),
            "games_completed_this_run": history.get("games_completed_this_run"),
        },
        "training_elapsed_s": training_elapsed_s,
        "checkpoint_output": str(checkpoint_output),
        "asu_modules_loaded": asu_modules_loaded,
        "asu_modules_loaded_count": len(asu_modules_loaded),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-games", type=int, default=N_GAMES)
    parser.add_argument("--seed-base", type=int, default=DEV_SEED_BASE)
    parser.add_argument("--checkpoint-output", type=Path, default=DEFAULT_CHECKPOINT_OUTPUT)
    parser.add_argument("--replay-dir", type=Path, default=DIAGNOSTIC_REPLAY_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)

    episode_seeds = _seed_range(args.seed_base, args.n_games)
    ep.require_seed_scope(
        episode_seeds, ep.SEED_CLASS_DEV, context="monopolyzero_pure_ppo_learnability_gate.py",
    )

    common.ensure_reference_on_path()
    import random

    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    def _emit(payload: dict) -> None:
        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
        print(text)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")

    diagnostic_positions = load_actor0_diagnostic_positions(args.replay_dir)
    if not diagnostic_positions:
        payload = {
            "status": "NO_DIAGNOSTIC_POSITIONS", "git_head_sha": git_head_sha,
            "replay_dir": str(args.replay_dir),
        }
        _emit(payload)
        raise RuntimeError(f"Stopping: no actor_id==0 positions found in {args.replay_dir}")

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        result = run_gate(
            n_games=args.n_games, seed_base=args.seed_base,
            diagnostic_positions=diagnostic_positions, checkpoint_output=args.checkpoint_output,
        )
    elapsed_s = time.perf_counter() - started

    payload = {
        "status": "OK",
        "git_head_sha": git_head_sha,
        "elapsed_s": elapsed_s,
        "peak_rss_gib": rss.peak_gib,
        **result,
    }
    _emit(payload)
    if result["asu_modules_loaded"]:
        raise RuntimeError(f"ASU modules loaded during gate run: {result['asu_modules_loaded']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
