"""Submission-safe internal inference layer for this project's frozen
pure-PPO (`hybrid=False`) Monopoly actor.

WHAT THIS IS: the smallest useful wrapper that (1) loads an actor network's
weights from a checkpoint file only after its hash gates pass (checkpoint
file + actor state, both required; actor source, optional - see below),
and (2) serves deterministic legal-masked-argmax `act(state, legal_action_ids)`
decisions from it. Nothing here is a new decision policy - the action
selection is a byte-for-byte reuse of this project's own existing,
already-in-use evaluation-time policy contract (see
`masked_argmax_decision`/`masked_argmax_action` below and their docstrings
for exactly which lines they reuse from
`scripts/monopolyzero_pure_ppo_strength_screen.py`, the script that
produced experiments 027-033).

Hash gates:
  1. checkpoint file SHA-256 (`file_sha256`) - the .pt file's raw bytes.
     REQUIRED (`expected_checkpoint_sha256`).
  2. full actor `state_dict` SHA-256 (`full_actor_state_sha256`) - computed
     AFTER loading, over the actual live weights. This is this project's
     established "actor SHA-256": the exact algorithm already computed and
     logged by every experiment/promotion runner that hash-gates a
     checkpoint (`monopolyzero_pure_ppo_strength_screen.py`'s
     `load_candidate_agent`/`build_untrained_baseline_agent`,
     `monopolyzero_promotion_gate_80_vs_32_64_128.py`,
     `monopolyzero_challenger_gate_96_vs_champion_32_64_128.py`,
     `monopolyzero_lr_ablation_80_resume.py`, ...) - all of which import
     `_full_actor_sha256` from
     `scripts/monopolyzero_pure_ppo_learnability_gate.py` rather than each
     reimplementing it. This module does the same: `full_actor_state_sha256`
     is a thin delegating wrapper around that exact imported function, not a
     reimplementation. REQUIRED (`expected_actor_sha256`).
  3. `ActorNetwork` class source-code SHA-256 (`actor_source_sha256`) -
     `inspect.getsource` over the live imported class. A weights-only
     checkpoint/state hash can never catch a code-level change to the
     network's architecture or forward pass (e.g. someone editing
     `references/DeepRL_Monopoly/monopoly_game_engine/networks.py`'s
     `ActorNetwork.forward`, or - since that submodule is supposed to stay
     read-only per CLAUDE.md - a stray local edit to it). OPTIONAL
     additional gate (`expected_actor_source_sha256`, default `None` =
     skipped) - it supplements, and never substitutes for, the required
     actor-state-hash gate above.
A mismatch on gate 1, on gate 2, or on gate 3 (when supplied) raises
`ChecksumMismatchError` immediately - never a silent retrain/reconstruct/
fallback, matching this project's existing hash-gate discipline.

Which exact checkpoint counts as "the" frozen submission agent is TBD as of
this writing (docs/EXPERIMENTS.md's `033` PROMOTION gate went GO; `034`,
which would compare that champion against further challengers, is
pre-registered but NOT YET RUN as of this writing - not touched, not run,
not disturbed by this module or its tests). Per CLAUDE.md ("Do not make
unverified assumptions ... If unknown, mark TBD and confirm before relying
on it"), this module does NOT hardcode a pinned checkpoint path or
expected-hash values for a specific champion - `expected_checkpoint_sha256`
and `expected_actor_sha256` are required caller-supplied parameters, and
`expected_actor_source_sha256` is an optional caller-supplied parameter.
Wire in the specific pinned values once the champion checkpoint is
finalized.

Hybrid/fixed-rule rejection: `load_frozen_actor` always constructs
`PPOAgent(hybrid=False)` (never accepts a caller-supplied `hybrid` flag) -
loading a checkpoint saved with `hybrid=True` fails structurally inside
`PPOAgent.load`'s own metadata check before this module's own redundant
`fixed_action_mask`-must-be-all-False assertion even runs. `HYBRID_COMPAT`
(the fixed BUY_PROPERTY/ACCEPT_TRADE rule carve-out used only as a
diagnostic upper-bound elsewhere in this project - see docs/DECISIONS.md's
2026-08-12 entry) is structurally unreachable from this module: it is never
imported, never constructed, never an available code path here.

No ASU import anywhere in this module - see test_frozen_ppo_inference.py's
source-level guard and its fresh-subprocess `sys.modules` check.

Official competition submission API: explicitly TBD, see
`OfficialSubmissionAdapter` at the bottom of this file. Its shape is not
guessed.

Import-time side effects: this module imports only the standard library,
plus `scripts/monopolyzero_pure_ppo_learnability_gate.py`'s
`_full_actor_sha256` (for the required actor-state hash gate, reused rather
than reimplemented - see above), at module scope. That import is itself
cheap: `monopolyzero_pure_ppo_learnability_gate.py` and the two project
modules it imports (`evaluation_protocol.py`, `monopolyzero_common.py`)
only import the standard library at THEIR module scope too - `torch`,
`numpy`, and every `monopoly_game_engine`/`monopoly_bench` import stay
lazy, inside functions/methods, everywhere in this chain. So merely
`import`-ing this module still never loads torch, numpy, or the reference
submodule (see test_frozen_ppo_inference.py's fresh-subprocess import
test).
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# This project's own established "actor SHA-256" algorithm - reused
# verbatim (not reimplemented) exactly like every experiment/promotion
# runner that hash-gates a checkpoint already does
# (monopolyzero_pure_ppo_strength_screen.py,
# monopolyzero_promotion_gate_80_vs_32_64_128.py,
# monopolyzero_challenger_gate_96_vs_champion_32_64_128.py,
# monopolyzero_lr_ablation_80_resume.py, ...).
from monopolyzero_pure_ppo_learnability_gate import _full_actor_sha256  # noqa: E402

DEFAULT_PLAYER_ID = 0  # matches every pure-PPO (hybrid=False) checkpoint
                        # this project has produced so far (027's candidate,
                        # etc. - see scripts/monopolyzero_pure_ppo_learnability_gate.py)
DEFAULT_HIDDEN_DIM = 256  # this project's only actor hidden_dim in real use


class ChecksumMismatchError(RuntimeError):
    """Raised when a checkpoint file's SHA-256, the loaded actor's full
    state_dict SHA-256, or (if supplied) the live ActorNetwork class's
    source-code SHA-256 does not match a caller-supplied expected value.
    Always raised BEFORE any inference is served - never a silent
    retrain/reconstruct/fallback."""


class IllegalActionSelectedError(RuntimeError):
    """Raised if `masked_argmax_decision`'s own legal-mask construction were
    ever somehow bypassed and an illegal action selected. Structurally
    unreachable given how the mask is built below - kept as a defensive
    invariant check, not a code path expected to ever fire."""


def _ensure_reference_on_path() -> None:
    """Lazy, idempotent path setup - defers to this project's own already
    ASU-audited `monopolyzero_common.ensure_reference_on_path()` (see that
    module's docstring) rather than duplicating its submodule-path/monkey-
    patch logic here."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import monopolyzero_common as common

    common.ensure_reference_on_path()


# ── hash gates ───────────────────────────────────────────────────────────


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def actor_source_sha256() -> str:
    """SHA-256 of the live `ActorNetwork` class's source text
    (`inspect.getsource`), from `references/DeepRL_Monopoly`'s
    `monopoly_game_engine/networks.py`. Guards against a code-level change
    to the architecture/forward pass that a weights-only checkpoint hash
    cannot see."""
    import inspect

    _ensure_reference_on_path()
    from monopoly_game_engine.networks import ActorNetwork

    source = inspect.getsource(ActorNetwork)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def full_actor_state_sha256(actor) -> str:
    """SHA-256 over an actor's full `state_dict` (sorted keys, key bytes +
    tensor bytes each) - this project's canonical "actor SHA-256". Pure
    delegation to (NOT a reimplementation of)
    `scripts/monopolyzero_pure_ppo_learnability_gate.py::_full_actor_sha256`
    - the exact function every experiment/promotion runner in this project
    already imports for this purpose. Exists under this name purely so
    this module's public API doesn't expose a private (`_`-prefixed)
    function from another script."""
    return _full_actor_sha256(actor)


# ── frozen checkpoint loader ────────────────────────────────────────────


def load_frozen_actor(
    checkpoint_path: Path | str,
    *,
    expected_checkpoint_sha256: str,
    expected_actor_sha256: str,
    expected_actor_source_sha256: str | None = None,
    player_id: int = DEFAULT_PLAYER_ID,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    device: str = "cpu",
):
    """Loads a frozen pure-PPO (`hybrid=False`) actor network from
    `checkpoint_path`. Returns the actor (`ActorNetwork`, already
    `.eval()`'d) only if every gate below passes; raises otherwise. Never
    returns a partially-verified or unverified actor.

    Order of checks (each one a hard stop, in this order):
      1. `checkpoint_path` exists (`FileNotFoundError` otherwise).
      2. checkpoint file SHA-256 == `expected_checkpoint_sha256`
         (`ChecksumMismatchError` otherwise) - checked BEFORE any attempt
         to deserialize the file, so a corrupted file fails with a clear
         hash-mismatch message rather than an opaque torch/pickle error.
      3. `PPOAgent(player_id, hybrid=False, hidden_dim, device).load(...)` -
         reuses `references/DeepRL_Monopoly`'s own unedited checkpoint
         loader verbatim for the actual weight/metadata loading (its own
         state_dim/action_dim/hybrid/player_id/hidden_dim structural
         checks apply here too - a checkpoint saved with `hybrid=True`, a
         different `player_id`, or a different `hidden_dim` is rejected by
         that call with `ValueError`, before this function's own step 4).
      4. `agent.hybrid is False` and `fixed_action_mask` has no `True`
         entries (`RuntimeError` otherwise) - redundant with step 3 for the
         checkpoint-hybrid-flag case specifically, kept as an explicit,
         self-documenting invariant matching
         `monopolyzero_pure_ppo_strength_screen.py::load_candidate_agent`'s
         own belt-and-suspenders check.
      5. loaded actor's full state_dict SHA-256 (`full_actor_state_sha256`,
         this project's established "actor SHA-256") ==
         `expected_actor_sha256` (`ChecksumMismatchError` otherwise) -
         REQUIRED. This is the primary actor-identity gate: it verifies the
         actual live weights, the same quantity every experiment/promotion
         runner in this project already computes and logs.
      6. OPTIONAL: if `expected_actor_source_sha256` is not `None`, live
         `ActorNetwork` class source SHA-256 (`actor_source_sha256`) must
         also match (`ChecksumMismatchError` otherwise). Supplements step 5
         - guards against a code-level change to the architecture/forward
         pass that a weights-only hash cannot see - but never substitutes
         for it; skipped entirely (no check performed) when left `None`.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    actual_checkpoint_hash = file_sha256(checkpoint_path)
    if actual_checkpoint_hash != expected_checkpoint_sha256:
        raise ChecksumMismatchError(
            "REJECTED: checkpoint file SHA-256 mismatch - "
            f"got {actual_checkpoint_hash}, expected {expected_checkpoint_sha256}. "
            "Refusing to load an unverified checkpoint file."
        )

    _ensure_reference_on_path()
    from monopoly_game_engine.agent_ppo import PPOAgent

    agent = PPOAgent(player_id=player_id, hybrid=False, hidden_dim=hidden_dim, device=device)
    agent.load(str(checkpoint_path))
    agent.actor.eval()

    if agent.hybrid is not False or bool(agent.fixed_action_mask.any()):
        raise RuntimeError(
            "REJECTED: loaded PPOAgent is not pure hybrid=False (fixed_action_mask "
            "has an entry set) - a hybrid/fixed-rule checkpoint is a diagnostic "
            "upper-bound only per docs/DECISIONS.md's 2026-08-12 entry, never a "
            "submission candidate. Refusing to serve inference from it."
        )

    actual_actor_hash = full_actor_state_sha256(agent.actor)
    if actual_actor_hash != expected_actor_sha256:
        raise ChecksumMismatchError(
            "REJECTED: actor state_dict SHA-256 mismatch - "
            f"got {actual_actor_hash}, expected {expected_actor_sha256}. "
            "Refusing to serve inference from an unverified actor."
        )

    if expected_actor_source_sha256 is not None:
        actual_source_hash = actor_source_sha256()
        if actual_source_hash != expected_actor_source_sha256:
            raise ChecksumMismatchError(
                "REJECTED: ActorNetwork class source-code SHA-256 mismatch - "
                f"got {actual_source_hash}, expected {expected_actor_source_sha256}. "
                "The actor network implementation has changed since this hash "
                "was pinned; refusing to run inference against an unverified "
                "network architecture/forward pass."
            )

    return agent.actor


# ── deterministic legal-masked argmax action selection ─────────────────


def masked_argmax_decision(actor, device, state, legal_action_ids: Sequence[int]):
    """Runs the `ActorNetwork` forward pass EXACTLY ONCE and returns
    `(chosen_action, log_probs, latency_s)`:
      - `chosen_action`: the deterministic legal-masked argmax action id.
      - `log_probs`: the resulting `(ACTION_SPACE_SIZE,)` log-prob tensor
        (already squeezed out of its batch dim), so a caller that also
        wants a decision trace can score every legal candidate directly
        from this tensor - WITHOUT running a second forward pass.
      - `latency_s`: wall-clock time (`time.perf_counter`) spent inside the
        single `actor(state_t, mask_t)` call only - tensor/mask
        construction and the argmax are outside the timed window.

    Byte-for-byte the same tensor construction and selection as
    `scripts/monopolyzero_pure_ppo_strength_screen.py`'s
    `build_masked_argmax_policy`'s inner `choose()` method - the policy
    that actually produced experiments 027-033's evaluation results:
    build a `(1, ACTION_SPACE_SIZE)` boolean legal-action mask, run
    `ActorNetwork.forward`'s masked log-softmax under `torch.inference_mode()`,
    and take `torch.argmax` over the result. Never
    `ActorNetwork.get_action`/`PPOAgent.choose_action` - both of those
    SAMPLE via `torch.multinomial`, which is correct for training
    exploration but not for deterministic evaluation/submission play.

    No explicit custom tie-break logic is layered on top: `torch.argmax`
    itself deterministically returns the FIRST (lowest-index) maximal
    element (verified for this project's exact torch build in
    test_frozen_ppo_inference.py::test_torch_argmax_ties_break_to_lowest_index),
    which already matches the "lowest action id" tie-break convention this
    project uses elsewhere.
    """
    import time

    import torch

    from monopoly_game_engine.actions import ACTION_SPACE_SIZE

    if not legal_action_ids:
        raise ValueError("masked_argmax_decision() received no legal action ids")

    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool, device=device)
    mask_t[0, list(legal_action_ids)] = True

    started = time.perf_counter()
    with torch.inference_mode():
        log_probs = actor(state_t, mask_t)
    latency_s = time.perf_counter() - started

    chosen_action = int(torch.argmax(log_probs, dim=-1).item())
    return chosen_action, log_probs.squeeze(0), latency_s


def masked_argmax_action(actor, device, state, legal_action_ids: Sequence[int]) -> int:
    """Convenience wrapper over `masked_argmax_decision` for callers that
    only need the chosen action id. Still exactly one forward pass -
    delegates entirely to `masked_argmax_decision`, no separate
    implementation of the tensor/mask/argmax logic."""
    chosen_action, _log_probs, _latency_s = masked_argmax_decision(actor, device, state, legal_action_ids)
    return chosen_action


@dataclass(frozen=True)
class DecisionTrace:
    """Compact per-decision trace: enough to audit what the policy did and
    why, without recording anything large (no full logit vector). Built
    entirely from the single forward pass `act()` already ran for action
    selection - never triggers a second `ActorNetwork` forward."""

    turn: int | None
    chosen_action: int
    legal_action_ids: tuple[int, ...]
    legal_action_count: int
    inference_latency_ms: float  # wall-clock time of the single actor forward only
    top_candidates: tuple[tuple[int, float], ...]  # (action_id, log_prob), best first


class FrozenPurePPOPolicy:
    """Submission-safe inference wrapper around one hash-verified, frozen
    pure-PPO actor network.

    Construct via `FrozenPurePPOPolicy.from_checkpoint` for any real use -
    it runs both hash gates via `load_frozen_actor` before this object can
    exist. The bare `__init__` accepts an already-loaded actor directly
    (used by this module's own tests to exercise `act()` against
    hand-built actors without needing a checkpoint file); it does NOT
    itself verify anything, so callers using `__init__` directly are
    opting out of the hash gates.
    """

    def __init__(self, actor, *, device: str = "cpu", enable_trace: bool = False):
        self._actor = actor
        self._device = device
        self._enable_trace = enable_trace
        self.trace_log: list[DecisionTrace] = []

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,
        expected_checkpoint_sha256: str,
        expected_actor_sha256: str,
        expected_actor_source_sha256: str | None = None,
        player_id: int = DEFAULT_PLAYER_ID,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        device: str = "cpu",
        enable_trace: bool = False,
    ) -> "FrozenPurePPOPolicy":
        actor = load_frozen_actor(
            checkpoint_path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_actor_sha256=expected_actor_sha256,
            expected_actor_source_sha256=expected_actor_source_sha256,
            player_id=player_id,
            hidden_dim=hidden_dim,
            device=device,
        )
        return cls(actor, device=device, enable_trace=enable_trace)

    def act(
        self,
        state,
        legal_action_ids: Sequence[int],
        *,
        turn: int | None = None,
        trace_top_k: int = 5,
    ) -> int:
        """Returns exactly one action id, guaranteed to be a member of
        `legal_action_ids`. No sampling: identical `(state, legal_action_ids)`
        always returns the identical action (see
        `masked_argmax_decision`'s docstring). Runs the `ActorNetwork`
        forward pass EXACTLY ONCE regardless of whether tracing is enabled
        - both the selected action and (when tracing is on) the decision
        trace are built from that single forward pass's output."""
        legal_action_ids = tuple(int(a) for a in legal_action_ids)
        if not legal_action_ids:
            raise ValueError("act() received no legal_action_ids")

        chosen, log_probs, latency_s = masked_argmax_decision(
            self._actor, self._device, state, legal_action_ids
        )

        if chosen not in legal_action_ids:
            # Structurally unreachable given masked_argmax_decision's own
            # mask construction (only legal_action_ids are ever unmasked,
            # so argmax cannot land elsewhere) - kept as an explicit,
            # never-silently-proceed invariant check.
            raise IllegalActionSelectedError(
                f"masked_argmax_decision returned {chosen}, which is not in "
                f"legal_action_ids={legal_action_ids} - refusing to return "
                "an illegal action."
            )

        if self._enable_trace:
            self.trace_log.append(
                self._build_trace(log_probs, legal_action_ids, chosen, turn, trace_top_k, latency_s)
            )

        return chosen

    def _build_trace(self, log_probs, legal_action_ids, chosen, turn, top_k, latency_s) -> DecisionTrace:
        """Builds a DecisionTrace purely by reading `log_probs` (already
        computed by `act()`'s single forward pass) - no new tensor
        construction, no second `ActorNetwork` call."""
        scored = sorted(
            ((int(a), float(log_probs[a].item())) for a in legal_action_ids),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return DecisionTrace(
            turn=turn,
            chosen_action=chosen,
            legal_action_ids=legal_action_ids,
            legal_action_count=len(legal_action_ids),
            inference_latency_ms=latency_s * 1000.0,
            top_candidates=tuple(scored[:top_k]),
        )

    def last_trace(self) -> DecisionTrace | None:
        return self.trace_log[-1] if self.trace_log else None


# ── official competition adapter: explicitly TBD ────────────────────────


class OfficialSubmissionAdapter:
    """Explicit TBD stub for the official competition's runtime submission
    API.

    This project does not yet have a confirmed official competition
    adapter contract (per-turn invocation shape/signature, how legal
    actions are communicated, timing/resource limits, process/entrypoint
    convention, etc.). Per CLAUDE.md ("Do not make unverified assumptions
    about rules, environment, or APIs. If unknown, mark TBD and confirm
    before relying on it."), guessing that shape here would be a direct
    rule violation - so this class intentionally implements nothing and
    raises on any attempted use.

    When the official API is confirmed, wire it in here: translate
    whatever the official per-turn call provides into
    `(state, legal_action_ids)` for `FrozenPurePPOPolicy.act`, and
    translate the returned action id back into whatever shape the official
    API expects. TBD.
    """

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Official competition submission API is not yet specified (TBD) - "
            "see OfficialSubmissionAdapter's docstring in "
            "scripts/frozen_ppo_inference.py. Do not guess its shape; confirm "
            "the official adapter contract before implementing this."
        )
