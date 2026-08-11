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


@dataclass
class _PolicyOnlyResult:
    """Same field shape as search.py's SearchResult contract (chosen_action,
    visits, q_values, root_value, simulations, latency_s) so callers that
    generically read `.chosen_action`/`.latency_s` don't need to special-case
    which kind of policy produced it. visits/q_values are placeholders (no
    search happened), never used to build a ReplayPosition."""

    chosen_action: int
    visits: dict
    q_values: dict
    root_value: object
    simulations: int
    latency_s: float


def build_local_policy_only(model):
    """Direct policy-head inference: no MCTS/search at all. Calls
    MonopolyZeroNet.predict() (model.py's own public forward-pass API —
    legal-masked softmax over the raw policy logits) and takes the legal
    argmax. This is the cheapest possible policy for a given checkpoint,
    used to measure whether PUCT search earns its computational cost over
    the bare policy head."""

    class _PolicyOnlyPolicy:
        kind = "policy_only"

        def choose(self, game, seat: int, decision_seed: int) -> "_PolicyOnlyResult":
            started = time.perf_counter()
            env = game.env
            legal = tuple(env.get_allowed_actions(seat))
            priors, value = model.predict(env._get_state(seat), legal, seat)
            chosen_action = max(priors, key=priors.get)
            latency_s = time.perf_counter() - started
            return _PolicyOnlyResult(
                chosen_action=chosen_action,
                visits={chosen_action: 1},
                q_values={},
                root_value=value,
                simulations=0,
                latency_s=latency_s,
            )

    return _PolicyOnlyPolicy()


def build_local_hybrid_compat_policy(model):
    """Diagnostic-only policy — NOT a final-submission candidate. Reproduces
    the ORIGINAL hybrid-PPO agent's BUY_PROPERTY/ACCEPT_TRADE fixed-rule
    carve-out on top of otherwise-plain POLICY_ONLY inference, by
    runtime-importing the reference's own fixed-decision functions (never
    copied into this repo — see
    references/DeepRL_Monopoly/monopoly_game_engine/agent_ppo.py's
    `fixed_buy_decision`/`fixed_accept_trade_decision`, both zero-ASU
    functions taking only the raw env + player id).

    Exists to isolate whether `build_local_policy_only`'s flat legal-masked
    argmax (which has no such carve-out) silently lets the neural policy
    head pick BUY_PROPERTY/ACCEPT_TRADE off logits that PPO training's
    `fixed_action_mask` (agent_ppo.py:186-189) permanently excluded from
    every gradient update — i.e. logits still sitting at whatever
    random-initialization value `ActorNetwork` started at, never trained by
    either the original hybrid-PPO run or this project's own `load_ppo_actor`
    copy (a pure weight copy, no training).

    Mirrors `PPOAgent.choose_action`'s hybrid branching exactly: BUY_PROPERTY
    checked first (bought via rule if legal and the rule says yes; otherwise
    falls through with BUY_PROPERTY removed from the neural candidate set —
    original hybrid semantics, not "always buy"/"never buy"), then an
    incoming-trade response checked next (decided directly via rule,
    accept/decline, whenever ACCEPT_TRADE is legal AND a pending trade
    targeting this seat is actually found — same double-condition the
    reference agent uses, not just legality).

    Every decision is additionally logged on `self.log` (one dict per call)
    recording the opportunity/intervention/disagreement-with-plain-
    POLICY_ONLY audit trail this diagnostic exists to produce — see the
    per-key comments below. A fresh instance should be built per game (like
    `LocalFixedPolicy`), not reused across games, so `self.log` stays
    game-scoped for the caller to tag with seed/seat afterward.
    """
    from monopoly_game_engine.actions import ActionType
    from monopoly_game_engine.agent_ppo import fixed_accept_trade_decision, fixed_buy_decision

    fixed_ids = {int(ActionType.BUY_PROPERTY), int(ActionType.ACCEPT_TRADE)}

    class _HybridCompatPolicy:
        kind = "hybrid_compat"

        def __init__(self):
            self.log: list[dict] = []

        def choose(self, game, seat: int, decision_seed: int) -> "_PolicyOnlyResult":
            started = time.perf_counter()
            env = game.env
            legal = tuple(env.get_allowed_actions(seat))
            state = env._get_state(seat)

            # What a plain POLICY_ONLY policy would do on this identical
            # state — computed unconditionally so every decision (not just
            # opportunity ones) has a POLICY_ONLY comparison point, which is
            # also how the integrity check (100% agreement outside
            # opportunity states) is verified downstream.
            policy_only_priors, policy_only_value = model.predict(state, legal, seat)
            policy_only_action = max(policy_only_priors, key=policy_only_priors.get)

            is_buy_opportunity = int(ActionType.BUY_PROPERTY) in legal
            is_trade_opportunity = int(ActionType.ACCEPT_TRADE) in legal
            trade_pending_found = None
            decision_kind = "no_opportunity"
            chosen_action = None
            value = policy_only_value

            if is_buy_opportunity and fixed_buy_decision(env, seat):
                chosen_action = int(ActionType.BUY_PROPERTY)
                decision_kind = "buy_property_rule_bought"

            if chosen_action is None and is_trade_opportunity:
                pending = next(
                    (offer for offer in env.pending_trades.values() if offer.to_player == seat),
                    None,
                )
                trade_pending_found = pending is not None
                if pending is not None:
                    accept = fixed_accept_trade_decision(env, seat)
                    chosen_action = int(ActionType.ACCEPT_TRADE) if accept else int(ActionType.DECLINE_TRADE)
                    decision_kind = "trade_response_rule_accept" if accept else "trade_response_rule_decline"

            fixed_rule_fired = chosen_action is not None
            nn_legal = tuple(action for action in legal if action not in fixed_ids)

            if chosen_action is None:
                # Original hybrid semantics: BUY_PROPERTY/ACCEPT_TRADE are
                # permanently excluded from the neural candidate set,
                # regardless of why we got here (buy declined by rule, or an
                # ACCEPT_TRADE-legal-but-no-pending-found edge case).
                candidate_legal = nn_legal if nn_legal else legal
                if candidate_legal == legal:
                    priors = policy_only_priors
                else:
                    priors, value = model.predict(state, candidate_legal, seat)
                    decision_kind = "candidate_set_narrowed_neural_pick"
                chosen_action = max(priors, key=priors.get)

            intervened = fixed_rule_fired or (nn_legal != legal)
            latency_s = time.perf_counter() - started

            self.log.append(
                {
                    "is_buy_opportunity": is_buy_opportunity,
                    "is_trade_opportunity": is_trade_opportunity,
                    "trade_pending_found": trade_pending_found,
                    "decision_kind": decision_kind,
                    "intervened": intervened,
                    "policy_only_action": policy_only_action,
                    "hybrid_compat_action": chosen_action,
                    "disagrees_with_policy_only": chosen_action != policy_only_action,
                    "policy_only_prob_buy": (
                        policy_only_priors.get(int(ActionType.BUY_PROPERTY)) if is_buy_opportunity else None
                    ),
                    "policy_only_chose_buy": (
                        (policy_only_action == int(ActionType.BUY_PROPERTY)) if is_buy_opportunity else None
                    ),
                    "policy_only_prob_accept_trade": (
                        policy_only_priors.get(int(ActionType.ACCEPT_TRADE)) if is_trade_opportunity else None
                    ),
                    "policy_only_chose_accept_trade": (
                        (policy_only_action == int(ActionType.ACCEPT_TRADE)) if is_trade_opportunity else None
                    ),
                }
            )

            return _PolicyOnlyResult(
                chosen_action=chosen_action,
                visits={chosen_action: 1},
                q_values={},
                root_value=value,
                simulations=0,
                latency_s=latency_s,
            )

    return _HybridCompatPolicy()


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
    final_round: int = 0
    final_net_worth: tuple = ()
    search_latencies_s: list = field(default_factory=list)
    shadow_decisions: list = field(default_factory=list)


def _invoke_policy(policy, game, seat: int, decision_seed: int):
    """Runs one policy's choose() and normalizes the return shape: (action,
    latency_s|None, result|None, kind). `result` carries visits/q_values/
    root_value for kinds that have them (search, policy_only, hybrid_compat)
    and is None for plain-int-returning kinds (fixed)."""
    kind = getattr(policy, "kind", None)
    if kind in ("search", "policy_only", "hybrid_compat"):
        result = policy.choose(game, seat, decision_seed)
        return result.chosen_action, result.latency_s, result, kind
    return policy.choose(game, seat, decision_seed), None, None, kind


def play_local_game(
    *,
    game_id: int,
    seed: int,
    policies: dict,
    max_rounds: int,
    record_seats: set[int],
    shadow_policy=None,
    shadow_seats: set[int] | None = None,
) -> LocalGameOutcome:
    """Plays one game turn-by-turn against the supplied per-seat policies.

    Structurally independent of arena.py's play_game: a closure-based
    per-turn resolver driven by a `while` loop (not a single `for step in
    range(...)` body), a different decision-seed mixing function, and the
    game/crash/outcome bookkeeping assembled in one place at the end rather
    than mutated incrementally on a result object during the loop.

    `shadow_policy`, when given, is asked what it would have chosen at every
    non-forced decision made by a seat in `shadow_seats` — queried on the
    exact same pre-step state as the real decision, but its answer is only
    recorded (LocalGameOutcome.shadow_decisions), never acted on. Lets two
    policies be compared on literally the same decision states a real
    evaluation game visits, without playing two divergent games.
    """
    from monopoly_bench.engine import MAX_DECISIONS_PER_TURN, NUM_PLAYERS, SharedGame, legal_mask
    from monopoly_bench.contracts import ReplayPosition

    game = SharedGame.new(seed, max_rounds)
    positions: list[ReplayPosition] = []
    search_latencies_s: list[float] = []
    shadow_decisions: list[dict] = []
    shadow_seats = shadow_seats or set()
    decision_budget = max_rounds * NUM_PLAYERS * MAX_DECISIONS_PER_TURN

    def resolve_one_turn(turn_index: int) -> int:
        seat = game.env.whose_turn()
        legal_actions = tuple(game.env.get_allowed_actions(seat))
        if len(legal_actions) == 1:
            return legal_actions[0]

        decision_seed = _mix_decision_seed(seed, turn_index, seat)
        policy = policies[seat]
        chosen_action, latency_s, result, kind = _invoke_policy(policy, game, seat, decision_seed)
        if latency_s is not None:
            search_latencies_s.append(latency_s)
        if kind == "search" and seat in record_seats:
            positions.append(
                ReplayPosition(
                    state=game.env._get_state(seat),
                    legal_mask=legal_mask(legal_actions),
                    visits=result.visits,
                    q_values=result.q_values,
                    selected_action=chosen_action,
                    value=result.root_value,
                    actor_id=seat,
                    game_id=game_id,
                )
            )

        if shadow_policy is not None and seat in shadow_seats:
            shadow_action, _, _, _ = _invoke_policy(shadow_policy, game, seat, decision_seed)
            shadow_decisions.append(
                {
                    "turn_index": turn_index,
                    "seat": seat,
                    "actual_action": chosen_action,
                    "shadow_action": shadow_action,
                    "agree": chosen_action == shadow_action,
                }
            )

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
            final_round=game.env.round, search_latencies_s=search_latencies_s,
            shadow_decisions=shadow_decisions,
        )
    except Exception as exc:  # noqa: BLE001 - intentional fail-closed boundary
        return LocalGameOutcome(
            completed=False, winner=None, decisions=turn_index,
            positions=positions, illegal_actions=0, crashed=True,
            error=f"{type(exc).__name__}: {exc}",
            final_round=game.env.round, search_latencies_s=search_latencies_s,
            shadow_decisions=shadow_decisions,
        )

    finished = game.env.done
    winner = game.env.winner() if finished else None
    if winner is not None:
        from monopoly_bench.engine import NUM_PLAYERS as _N

        one_hot = tuple(1.0 if seat == winner else 0.0 for seat in range(_N))
        for position in positions:
            position.outcome = one_hot

    net_worth = tuple(float(player.net_worth()) for player in game.env.players)

    return LocalGameOutcome(
        completed=finished, winner=winner, decisions=turn_index, positions=positions,
        final_round=game.env.round, final_net_worth=net_worth,
        search_latencies_s=search_latencies_s, shadow_decisions=shadow_decisions,
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
