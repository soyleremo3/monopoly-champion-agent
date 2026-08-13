"""Dedicated, A96-pinned friend-match wrapper around the generic,
champion-agnostic `frozen_ppo_inference.FrozenPurePPOPolicy`.

NOT the official competition adapter - `frozen_ppo_inference.py`'s
`OfficialSubmissionAdapter` stays an explicit TBD stub (unmodified, not
touched by this module) until the official per-turn API contract is
confirmed; guessing it is out of scope here per CLAUDE.md's
no-unverified-assumptions rule.

This module MAY pin A96's identity (checkpoint filename + both required
hashes) - the generic `FrozenPurePPOPolicy`/`load_frozen_actor` in
`frozen_ppo_inference.py` must not and does not; this wrapper is the one
deliberately champion-specific layer, per this task's own instruction.

`A96FriendMatchAgent` instantiates ONLY
`FrozenPurePPOPolicy.from_checkpoint(...)` - no second policy, no
fallback policy, no sampling, no BUY_PROPERTY override (see
scripts/monopolyzero_diagnostic_buy_property_intervention_a96.py's
isolated, never-merged BUY_SIMPLE experiment - not reused here), no ASU
import anywhere in this file. A checkpoint/hash mismatch propagates
`FrozenPurePPOPolicy.from_checkpoint`'s own exception unchanged - this
wrapper adds no try/except around it, so a bad checkpoint fails closed,
never silently substituted.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from frozen_ppo_inference import FrozenPurePPOPolicy  # noqa: E402

# Pinned A96 identity - verified against 034's own logged provenance
# (logs/experiments/034-challenger-gate-96-vs-champion-32-64-128.json's
# "challenger_96" entry) before being wired in here.
A96_CHECKPOINT_FILENAME = "candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt"
A96_CHECKPOINT_SHA256 = "78585ed4e2400d024633ee2878d8b88243f7f4a9f498d9019a73e44b3a830f51"
A96_ACTOR_SHA256 = "2bd1e9bad3d6e0100e033507f72f15b5088d8f8fb2f04dba1ae9d759b5a34a40"
A96_REFERENCE_SUBMODULE_SHA = "afd9205761317e196d77f679921c35fb04c7ab96"

DEFAULT_CHECKPOINT_PATH = (
    REPO_ROOT / "artifacts" / "monopolyzero_pure_ppo_learnability_gate" / A96_CHECKPOINT_FILENAME
)
CHECKPOINT_PATH_ENV_VAR = "A96_CHECKPOINT_PATH"


def resolve_checkpoint_path(checkpoint_path: Path | str | None) -> Path:
    """Checkpoint resolution order (first match wins), never silently
    substituted:
      1. explicit `checkpoint_path` argument
      2. `A96_CHECKPOINT_PATH` environment variable
      3. this project's own local champion path under artifacts/
    """
    if checkpoint_path is not None:
        return Path(checkpoint_path)
    env_value = os.environ.get(CHECKPOINT_PATH_ENV_VAR)
    if env_value:
        return Path(env_value)
    return DEFAULT_CHECKPOINT_PATH


class A96FriendMatchAgent:
    """Loads the pinned A96 checkpoint (hash-gated, fail-closed) and
    serves `act(state, legal_action_ids)` by delegating directly to
    `FrozenPurePPOPolicy.act` - no logic of its own beyond checkpoint
    resolution and construction."""

    def __init__(self, checkpoint_path: Path | str | None = None, *, device: str = "cpu", enable_trace: bool = False):
        resolved_path = resolve_checkpoint_path(checkpoint_path)
        self._checkpoint_path = resolved_path
        self._policy = FrozenPurePPOPolicy.from_checkpoint(
            resolved_path,
            expected_checkpoint_sha256=A96_CHECKPOINT_SHA256,
            expected_actor_sha256=A96_ACTOR_SHA256,
            device=device,
            enable_trace=enable_trace,
        )

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    def act(self, state, legal_action_ids: Sequence[int], *, turn: int | None = None) -> int:
        return self._policy.act(state, legal_action_ids, turn=turn)

    def last_trace(self):
        return self._policy.last_trace()
