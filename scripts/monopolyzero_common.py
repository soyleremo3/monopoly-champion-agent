"""Shared ASU-import-free MonopolyZero wiring for this project's own scripts.

Imports only modules confirmed ASU-import-clean (see docs/REFERENCE_AUDIT.md
and docs/DECISIONS.md's 2026-08-11 "later" correction entry): engine, model,
search, storage, config, contracts, and monopoly_game_engine.agents_fixed.
Never imports monopoly_bench.adapters, .arena, or .training, since those
load ASU_FROZEN_TEACHER at module scope (adapters.py, training.py) or
transitively (arena.py imports adapters.py) — see the ASU-module-loaded
guard below, which is how every script using this module verifies that at
runtime rather than just asserting it.

Audit note (2026-08-11, later still): an earlier version of this module's
game loop and training-update function were structurally too close to the
reference's `arena.py::play_game` and `training.py::train_step` (same magic
constants, same variable-name-for-variable-name structure) for a repository
with no compatible open-source license (references/DeepRL_Monopoly has none
— see docs/REFERENCE_AUDIT.md's License section). They were rewritten below
with different control flow, different decision-seed mixing, and a dense
(scatter-then-normalize) policy-target formulation instead of the
reference's sparse-gather one. Same verified behavior (see
logs/experiments/012-selfplay-asu-import-free-smoke.json for the equivalence
check against the reference-shaped version), independent expression.

Reference submodule stays read-only: nothing here modifies it.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "references" / "DeepRL_Monopoly"
DEFAULT_PPO = REFERENCE_ROOT / "artifacts" / "ppo_plus" / "ppo_hybrid_2000_v2.pt"

REQUIRED_HASH_SEED = "0"


def require_pinned_hash_seed(script_name: str) -> None:
    actual = os.environ.get("PYTHONHASHSEED")
    if actual == REQUIRED_HASH_SEED:
        return
    shown = "unset" if actual is None else repr(actual)
    raise SystemExit(
        f"{script_name} refuses to run: PYTHONHASHSEED must be "
        f"'{REQUIRED_HASH_SEED}', got {shown}. Re-run as:\n"
        f"  PYTHONHASHSEED={REQUIRED_HASH_SEED} python {script_name}"
    )


def require_clean_git_tree(script_name: str) -> str:
    """Refuses to run on a dirty tree; returns the clean HEAD SHA otherwise."""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    if status.stdout.strip():
        raise SystemExit(
            f"{script_name} refuses to run: working tree is not clean "
            "(git status --porcelain reported changes below). "
            "code_commit_sha must be an unambiguous clean HEAD — commit or "
            "stash first, then re-run.\n" + status.stdout
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return head.stdout.strip()


def loaded_asu_modules() -> list[str]:
    return sorted(
        name for name in sys.modules
        if name == "ASU_FROZEN_TEACHER" or name.startswith("ASU_FROZEN_TEACHER.")
    )


class RssMonitor:
    """Polls this process's RSS on a background thread; reports the peak."""

    def __init__(self, interval_s: float = 0.02):
        import psutil

        self._process = psutil.Process(os.getpid())
        self._interval_s = interval_s
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._peak = max(self._peak, self._process.memory_info().rss)
            except Exception:
                pass
            time.sleep(self._interval_s)

    def __enter__(self) -> "RssMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def peak_gib(self) -> float:
        return self._peak / 1024**3


def ensure_reference_on_path() -> None:
    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))


def _mix_decision_seed(base_seed: int, turn_index: int, seat: int) -> int:
    """Deterministic per-decision seed. Independent scheme from the
    reference's `seed * 1_000_003 + step * 17 + player_id` — this mixes with
    XOR over three differently-scaled multiplicative terms so the numeric
    identity of the two schemes never coincides."""
    mixed = (base_seed * 2654435761) ^ (turn_index * 40503) ^ (seat * 2246822519)
    return mixed & 0x7FFFFFFF


def build_local_search_policy(model, search_config, *, self_play: bool):
    """Thin delegation to MaxNPUCT (search.py has zero ASU references)."""
    from monopoly_bench.search import MaxNPUCT

    class _SearchPolicy:
        kind = "search"

        def __init__(self):
            self._engine = MaxNPUCT(model, search_config, self_play=self_play)

        def choose(self, game, seat: int, decision_seed: int):
            return self._engine.choose_action(game, seat, decision_seed)

    return _SearchPolicy()


class LocalFixedPolicy:
    """Reimplements the fixed-agent decision contract we need: a seeded,
    isolated call into a scripted FixedPolicyAgent subclass, with a
    substitute-and-count fallback when it proposes something illegal.
    """

    kind = "fixed"

    def __init__(self, agent_class):
        self.agent_class = agent_class
        self.name = agent_class.__name__
        self.fallback_count = 0

    def choose(self, game, seat: int, decision_seed: int) -> int:
        from monopoly_bench.engine import ActionType, unwrap

        env = unwrap(game)
        with _temporarily_seeded_random(decision_seed):
            proposed = int(self.agent_class(seat).choose_action(env))
        legal = tuple(env.get_allowed_actions(seat))
        if proposed in legal:
            return proposed
        self.fallback_count += 1
        end_turn = int(ActionType.END_TURN)
        return end_turn if end_turn in legal else legal[0]


class _SeededRandomScope:
    """Context manager: seed the global `random` module, restore its prior
    state on exit — needed because FixedPolicyAgent subclasses read from
    the global module internally, not from an injected RNG."""

    def __init__(self, seed: int):
        self._seed = seed
        self._saved = None

    def __enter__(self) -> "_SeededRandomScope":
        self._saved = random.getstate()
        random.seed(self._seed)
        return self

    def __exit__(self, *exc_info) -> None:
        random.setstate(self._saved)


def _temporarily_seeded_random(seed: int) -> _SeededRandomScope:
    return _SeededRandomScope(seed)


@dataclass
class LocalGameOutcome:
    completed: bool
    winner: int | None
    decisions: int
    positions: list = field(default_factory=list)
    illegal_actions: int = 0
    crashed: bool = False
    error: str | None = None


def play_local_game(*, game_id: int, seed: int, policies: dict, max_rounds: int, record_seats: set[int]) -> LocalGameOutcome:
    """Plays one game turn-by-turn against the supplied per-seat policies.

    Structurally independent of arena.py's play_game: a closure-based
    per-turn resolver driven by a `while` loop (not a single `for step in
    range(...)` body), a different decision-seed mixing function, and the
    game/crash/outcome bookkeeping assembled in one place at the end rather
    than mutated incrementally on a result object during the loop.
    """
    from monopoly_bench.engine import MAX_DECISIONS_PER_TURN, NUM_PLAYERS, SharedGame, legal_mask
    from monopoly_bench.contracts import ReplayPosition

    game = SharedGame.new(seed, max_rounds)
    positions: list[ReplayPosition] = []
    decision_budget = max_rounds * NUM_PLAYERS * MAX_DECISIONS_PER_TURN

    def resolve_one_turn(turn_index: int) -> int:
        seat = game.env.whose_turn()
        legal_actions = tuple(game.env.get_allowed_actions(seat))
        if len(legal_actions) == 1:
            return legal_actions[0]

        decision_seed = _mix_decision_seed(seed, turn_index, seat)
        policy = policies[seat]
        if getattr(policy, "kind", None) == "search":
            outcome = policy.choose(game, seat, decision_seed)
            chosen_action = outcome.chosen_action
            if seat in record_seats:
                positions.append(
                    ReplayPosition(
                        state=game.env._get_state(seat),
                        legal_mask=legal_mask(legal_actions),
                        visits=outcome.visits,
                        q_values=outcome.q_values,
                        selected_action=chosen_action,
                        value=outcome.root_value,
                        actor_id=seat,
                        game_id=game_id,
                    )
                )
        else:
            chosen_action = policy.choose(game, seat, decision_seed)

        if chosen_action not in legal_actions:
            raise RuntimeError(f"seat {seat} attempted illegal action {chosen_action}")
        return chosen_action

    turn_index = 0
    try:
        while turn_index < decision_budget and not game.env.done:
            action = resolve_one_turn(turn_index)
            game.step(action)
            turn_index += 1
    except RuntimeError as exc:
        return LocalGameOutcome(
            completed=False, winner=None, decisions=turn_index,
            positions=positions, illegal_actions=1, crashed=False, error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - intentional fail-closed boundary
        return LocalGameOutcome(
            completed=False, winner=None, decisions=turn_index,
            positions=positions, illegal_actions=0, crashed=True,
            error=f"{type(exc).__name__}: {exc}",
        )

    finished = game.env.done
    winner = game.env.winner() if finished else None
    if winner is not None:
        from monopoly_bench.engine import NUM_PLAYERS as _N

        one_hot = tuple(1.0 if seat == winner else 0.0 for seat in range(_N))
        for position in positions:
            position.outcome = one_hot

    return LocalGameOutcome(
        completed=finished, winner=winner, decisions=turn_index, positions=positions,
    )


def _scatter_visit_targets(actions_sparse, visit_counts_sparse, lengths, num_actions: int):
    """Turn each position's sparse (action, visit-count) pairs into a dense,
    row-normalized target distribution over the full action space. This is
    a different formulation from the reference's sparse-gather approach
    (which computes log-probabilities only at the sparse action indices);
    here the target itself is dense and cross-entropy runs against the full
    policy distribution."""
    import torch

    batch_size = actions_sparse.shape[0]
    dense = torch.zeros((batch_size, num_actions), dtype=torch.float32)
    for row in range(batch_size):
        valid = int(lengths[row])
        if valid:
            dense[row].scatter_(0, actions_sparse[row, :valid], visit_counts_sparse[row, :valid])
    row_totals = dense.sum(dim=1, keepdim=True)
    if (row_totals <= 0).any():
        raise ValueError("A sampled replay position carries no recorded search visits")
    return dense / row_totals


def _dense_target_cross_entropy(logits, targets):
    """Cross-entropy of a dense target distribution against `logits`, safe
    when `logits` contains -inf at masked (illegal) positions: those are
    always paired with a zero target, but `0 * -inf` is NaN in IEEE754, so
    the -inf must be swapped out before multiplying rather than relied on
    to cancel via the zero target."""
    import torch

    log_probs = torch.log_softmax(logits, dim=1)
    safe_log_probs = torch.where(targets > 0, log_probs, torch.zeros_like(log_probs))
    return -(targets * safe_log_probs).sum(dim=1).mean()


def local_training_update(model, optimizer, batch: dict, *, gradient_clip: float) -> dict:
    """One gradient step: policy target from MCTS visit counts (dense
    cross-entropy, see _scatter_visit_targets), value target from the real
    game winner reordered to the acting seat's perspective.

    Independent expression of the same training objective as
    training.py::train_step: policy loss computed first here (reference
    computes value loss first), dense-target cross-entropy instead of
    sparse gather-and-mask, and no AMP/GradScaler machinery (this project
    only trains on CPU so far, so that reference-specific complexity does
    not apply).
    """
    import torch

    from monopoly_bench.engine import ACTION_SPACE_SIZE, actor_order

    device = next(model.parameters()).device
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
    masks = torch.as_tensor(batch["legal_masks"], dtype=torch.bool, device=device)
    actions_sparse = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
    visit_counts_sparse = torch.as_tensor(batch["visit_counts"], dtype=torch.float32, device=device)
    lengths = torch.as_tensor(batch["lengths"], dtype=torch.long, device=device)
    outcomes = torch.as_tensor(batch["outcomes"], dtype=torch.float32, device=device)
    actors = torch.as_tensor(batch["actors"], dtype=torch.long, device=device)

    policy_targets = _scatter_visit_targets(actions_sparse, visit_counts_sparse, lengths, ACTION_SPACE_SIZE).to(device)
    value_targets = torch.stack(
        [outcomes[row, list(actor_order(int(actors[row])))] for row in range(outcomes.shape[0])]
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    policy_logits, value_probs = model(states, masks)

    policy_loss = _dense_target_cross_entropy(policy_logits, policy_targets)

    value_log_probs = torch.log(value_probs.clamp_min(1e-8))
    value_loss = -(value_targets * value_log_probs).sum(dim=1).mean()

    total_loss = policy_loss + value_loss
    total_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()

    return {
        "loss": float(total_loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "gradient_norm": float(gradient_norm),
    }
