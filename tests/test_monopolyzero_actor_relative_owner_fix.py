"""Regression tests for the actor-relative property-owner encoding fix.

monopoly_game_engine.state.build_state_vector's property-owner one-hot
indexed by *physical* player id (`owner_vec[prop.owner] = 1.0`), unlike
every other per-player section of the same 300-dim vector (turn order,
bankrupt, jail_turns, debt_creditor, ...), which all go through the
engine's canonical actor-relative order:

    order = [agent_id] + [i for i in range(NUM_PLAYERS) if i != agent_id]

i.e. the acting player first, then the other physical player ids in
ascending order -- not an arbitrary/rotation-invariant relabeling. For
agent_id != 0 the owner one-hot mislabeled which relative seat owns a
property.

references/DeepRL_Monopoly is a pinned, read-only submodule (never edited in
place -- see CLAUDE.md and docs/REFERENCE_AUDIT.md), so the fix lives in
scripts/monopolyzero_common.py as a runtime monkeypatch
(patch_actor_relative_owner_encoding, wired into ensure_reference_on_path)
rather than in state.py itself. See docs/DECISIONS.md's 2026-08-12 entry.

Test design note (correction from an earlier version of this file): a
previous test asserted that arbitrary physical-seat permutations of "the
same relative situation" produce byte-identical full 300-dim observations.
That is NOT the correct invariant, because the engine's actor_order depends
on the *absolute* physical ids of the other players (ascending order), not
just their position relative to the agent -- relabeling individuals to
different physical seats changes which physical ids are "the others",
which changes `order` for every per-player section, not just ownership.
The correct invariant, proven below, is internal frame consistency: within
one single build_state_vector call, property-owner slot k must name the
same physical player as player-feature slot k, both under this call's own
actor_order(agent_id).

Environment note: two of the tests below (search-adapter and replay-
recorder coverage) exercise MaxNPUCT/play_local_game against the real
engine. `MaxNPUCT.choose_action` unconditionally calls
`numpy.random.default_rng`, which on this machine is currently blocked by a
local Application Control policy (`ImportError: DLL load failed while
importing _mt19937: An Application Control policy has blocked this file`)
-- a pre-existing, unrelated environment issue (confirmed by the same
failure on `tests/test_monopolyzero_buy_trade_gradient_diagnostic.py
::test_real_engine_smoke_one_seed_tiny_search`, which predates this fix and
touches none of this code). To keep these two tests meaningful on this
machine without depending on that unrelated breakage:
  - the search-adapter test calls `MaxNPUCT._evaluate` directly (the exact
    method `choose_action`/`_simulate` call for both the root node and
    every descendant node -- see search.py lines 265, 198, and 214) instead
    of `choose_action`, so it never touches `numpy.random`.
  - the replay-recorder test uses a minimal `kind="search"` stub policy
    (real `play_local_game` code path, fake search internals) instead of a
    real `MaxNPUCT`-backed policy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "monopolyzero_common.py"

_spec = importlib.util.spec_from_file_location("monopolyzero_common", MODULE_PATH)
common = importlib.util.module_from_spec(_spec)
sys.modules["monopolyzero_common"] = common
_spec.loader.exec_module(common)


def _load_pristine_unpatched_state_module():
    """A fresh exec of monopoly_game_engine/state.py, independent of
    whatever patch_actor_relative_owner_encoding() has already done to the
    cached sys.modules copy -- the real original, not an inlined guess."""
    common.ensure_reference_on_path()
    state_path = common.REFERENCE_ROOT / "monopoly_game_engine" / "state.py"
    spec = importlib.util.spec_from_file_location("monopoly_game_engine.state", state_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _owner_section_bounds(num_players: int, prop_index: int) -> tuple[int, int]:
    section_start = 4 * num_players  # end of the 16-dim player-features block
    stride = 8  # owner_onehot(5) + mortgaged(1) + is_monopoly(1) + improvement_fraction(1)
    start = section_start + prop_index * stride
    return start, start + 5


def test_owner_encoded_in_actor_relative_slot_not_physical_slot():
    """Owner mapping assertion (kept, not weakened): agent_id=2, property
    owned by physical player 0 must land in actor-relative owner slot 1
    (order=[2,0,1,3] -> index(0)==1), not physical slot 0 (the bug)."""
    common.ensure_reference_on_path()
    from monopoly_game_engine.constants import NUM_PLAYERS, PROPERTY_IDS
    from monopoly_game_engine.state import Player, Property, build_state_vector

    players = [Player(i) for i in range(NUM_PLAYERS)]
    properties = {sq: Property(sq) for sq in PROPERTY_IDS}
    properties[PROPERTY_IDS[0]].owner = 0  # physical player 0 owns it

    state = build_state_vector(players, properties, agent_id=2, env=None)

    start, end = _owner_section_bounds(NUM_PLAYERS, 0)
    owner_slice = state[start:end]

    assert owner_slice[1] == 1.0
    assert owner_slice[0] == 0.0
    assert owner_slice[2] == 0.0 and owner_slice[3] == 0.0 and owner_slice[4] == 0.0


def test_owner_slot_matches_actor_relative_player_slot_for_all_actors():
    """The required proof, generalized over actor_id 0..3 and every
    physical owner: property-owner slot k must equal actor_order(agent_id)
    exactly, AND -- the actual internal-frame-consistency check -- slot k
    must name the same physical player as player-feature slot k, cross-
    checked against the player-features block, which is a separate,
    untouched-by-this-fix code path (state.py lines 138-146). This is not
    circular: if the fix's `order.index(owner)` were wrong (off-by-one,
    wrong sort direction, ...), this cross-check against the independently
    correct player-features block would catch it even if the owner-only
    assertion above happened to still look plausible."""
    common.ensure_reference_on_path()
    from monopoly_game_engine.constants import NUM_PLAYERS, PROPERTY_IDS
    from monopoly_game_engine.state import STATE_DIM, Player, Property, build_state_vector

    cash_fingerprint = {i: 1000.0 + i * 111.0 for i in range(NUM_PLAYERS)}

    for agent_id in range(NUM_PLAYERS):
        for owner in range(NUM_PLAYERS):
            players = [Player(i) for i in range(NUM_PLAYERS)]
            for i, p in enumerate(players):
                p.cash = cash_fingerprint[i]
            properties = {sq: Property(sq) for sq in PROPERTY_IDS}
            properties[PROPERTY_IDS[0]].owner = owner

            state = build_state_vector(players, properties, agent_id=agent_id, env=None)

            assert state.shape == (STATE_DIM,)  # layout/dims unchanged

            start, end = _owner_section_bounds(NUM_PLAYERS, 0)
            owner_slice = state[start:end]
            assert owner_slice.sum() == 1.0
            owner_slot = int(np.argmax(owner_slice))

            expected_order = [agent_id] + [i for i in range(NUM_PLAYERS) if i != agent_id]
            assert expected_order[owner_slot] == owner, (agent_id, owner, owner_slot)

            player_slot_cash_fraction = state[owner_slot * 4 + 1]
            expected_fraction = min(cash_fingerprint[owner] / 5000.0, 1.0)
            assert abs(player_slot_cash_fraction - expected_fraction) < 1e-6


def test_agent_id_zero_encoding_byte_for_byte_unchanged_by_fix():
    """agent_id=0's output must be identical to the real, pristine, never-
    monkeypatched original -- not just self-consistent."""
    common.ensure_reference_on_path()
    from monopoly_game_engine.constants import NUM_PLAYERS, PROPERTY_IDS
    from monopoly_game_engine.state import build_state_vector as patched_build_state_vector

    pristine = _load_pristine_unpatched_state_module()

    players = [pristine.Player(i) for i in range(NUM_PLAYERS)]
    players[1].cash = 1234
    players[2].in_jail = True
    properties = {sq: pristine.Property(sq) for sq in PROPERTY_IDS}
    properties[PROPERTY_IDS[0]].owner = 1
    properties[PROPERTY_IDS[3]].owner = 3
    properties[PROPERTY_IDS[5]].owner = 0

    original_output = pristine.build_state_vector(players, properties, agent_id=0, env=None)
    patched_output = patched_build_state_vector(players, properties, agent_id=0, env=None)

    assert np.array_equal(original_output, patched_output)

    # Sanity check that the harness itself isn't vacuous: away from
    # agent_id=0, pristine and patched must actually differ on this exact
    # ownership config (owner=3 is not at physical/relative slot 3 once the
    # agent isn't seat 0), otherwise this test would pass even if the patch
    # were a no-op.
    original_agent2 = pristine.build_state_vector(players, properties, agent_id=2, env=None)
    patched_agent2 = patched_build_state_vector(players, properties, agent_id=2, env=None)
    assert not np.array_equal(original_agent2, patched_agent2)


def test_owner_fix_touches_only_owner_coordinates_for_actors_1_to_3():
    """For actors 1-3, every non-owner coordinate must remain unchanged;
    only the 28 five-wide owner one-hot slices may differ from the
    pristine, unpatched original."""
    common.ensure_reference_on_path()
    from monopoly_game_engine.constants import NUM_PLAYERS, PROPERTY_IDS
    from monopoly_game_engine.state import build_state_vector as patched_build_state_vector

    pristine = _load_pristine_unpatched_state_module()

    for agent_id in (1, 2, 3):
        players = [pristine.Player(i) for i in range(NUM_PLAYERS)]
        players[1].cash = 1234
        players[2].in_jail = True
        properties = {sq: pristine.Property(sq) for sq in PROPERTY_IDS}
        properties[PROPERTY_IDS[0]].owner = 1
        properties[PROPERTY_IDS[3]].owner = 3
        properties[PROPERTY_IDS[5]].owner = 0

        original = pristine.build_state_vector(players, properties, agent_id=agent_id, env=None)
        patched = patched_build_state_vector(players, properties, agent_id=agent_id, env=None)

        assert original.shape == patched.shape  # dimension/layout unchanged

        owner_mask = np.zeros_like(original, dtype=bool)
        for prop_index in range(len(PROPERTY_IDS)):
            start, end = _owner_section_bounds(NUM_PLAYERS, prop_index)
            owner_mask[start:end] = True

        assert np.array_equal(original[~owner_mask], patched[~owner_mask]), agent_id
        # Non-vacuous: this ownership config must actually trigger a
        # remapping for these actors, otherwise the assertion above would
        # pass even for a no-op fix.
        assert not np.array_equal(original[owner_mask], patched[owner_mask]), agent_id


def test_patch_is_idempotent_and_does_not_double_remap():
    """Requirement: do not double-remap states that are already stored
    corrected. patch_actor_relative_owner_encoding() is guarded by a marker
    on the wrapped function; calling it (or ensure_reference_on_path())
    again must not wrap the already-patched function a second time, which
    would remap an already-actor-relative owner encoding right back into
    something wrong."""
    common.ensure_reference_on_path()
    from monopoly_game_engine.constants import NUM_PLAYERS, PROPERTY_IDS
    from monopoly_game_engine.state import Player, Property, build_state_vector

    players = [Player(i) for i in range(NUM_PLAYERS)]
    properties = {sq: Property(sq) for sq in PROPERTY_IDS}
    properties[PROPERTY_IDS[0]].owner = 2

    once = build_state_vector(players, properties, agent_id=3, env=None)

    common.patch_actor_relative_owner_encoding()
    common.patch_actor_relative_owner_encoding()
    common.ensure_reference_on_path()

    import monopoly_game_engine.state as state_module

    twice = state_module.build_state_vector(players, properties, agent_id=3, env=None)

    assert np.array_equal(once, twice)


def test_search_adapter_receives_corrected_observations_at_root_and_descendants():
    """MaxNPUCT must receive corrected state at both the root node and
    descendant nodes, without editing reference search.py. Both are built
    by the same `MaxNPUCT._evaluate` method (search.py lines 265 for root,
    198/214 for descendants), which always fetches its state via
    `env._get_state(actor)` -- the same single monkeypatched call site
    everything else in this fix goes through. No project-owned model/
    predict adapter is needed: proven here by calling `_evaluate` directly
    on both a live env and a transitioned (descendant) env and checking
    every observation it produced against an independently-computed
    expected owner mapping.
    """
    common.ensure_reference_on_path()
    from monopoly_bench.config import SearchConfig
    from monopoly_bench.engine import SharedGame, transition
    from monopoly_bench.search import MaxNPUCT
    from monopoly_game_engine.actions import ActionType
    from monopoly_game_engine.constants import NUM_PLAYERS, PROPERTY_IDS
    import monopoly_game_engine.env as env_module

    game = SharedGame.new(seed=7, max_rounds=5)
    # Walk forward with "always take the first legal action" until a real,
    # non-forced decision (>1 legal action) appears.
    for _ in range(60):
        seat = game.env.whose_turn()
        legal = tuple(game.env.get_allowed_actions(seat))
        if len(legal) > 1:
            break
        game.step(legal[0])
    else:
        raise AssertionError("Never reached a non-forced decision within 60 steps")

    # Decorate with ownership that would expose the bug (owner != agent,
    # owner != physical slot it'd land in under the old physical-index bug).
    game.env.properties[PROPERTY_IDS[0]].owner = 2
    game.env.properties[PROPERTY_IDS[27]].owner = (seat + 1) % NUM_PLAYERS

    calls: list[tuple[int, dict, np.ndarray]] = []
    patched_fn = env_module.build_state_vector

    def _spy(players, properties_dict, agent_id, env=None):
        owners_snapshot = {sq: properties_dict[sq].owner for sq in PROPERTY_IDS}
        state = patched_fn(players, properties_dict, agent_id, env)
        calls.append((agent_id, owners_snapshot, state.copy()))
        return state

    class _StubModel:
        def predict(self, state, legal_actions, actor_id):
            legal_ids = tuple(int(action) for action in legal_actions)
            priors = {action: 1.0 / len(legal_ids) for action in legal_ids}
            return priors, np.zeros(NUM_PLAYERS, dtype=np.float32)

    env_module.build_state_vector = _spy
    try:
        engine = MaxNPUCT(_StubModel(), SearchConfig(simulations=1), self_play=False)
        engine._evaluate(game.env)  # root node

        roll = int(ActionType.ROLL_DICE)
        non_roll_actions = [action for action in legal if action != roll]
        assert non_roll_actions, "test needs a non-ROLL_DICE legal action to avoid an explicit dice outcome"
        next_env = transition(game.env, non_roll_actions[0])
        if not next_env.done:
            engine._evaluate(next_env)  # descendant node
    finally:
        env_module.build_state_vector = patched_fn

    assert len(calls) >= 2  # at least root + one descendant/internal call

    for agent_id, owners_snapshot, state in calls:
        order = common._actor_relative_order(agent_id, NUM_PLAYERS)
        for prop_index, square_id in enumerate(PROPERTY_IDS):
            owner = owners_snapshot[square_id]
            start, end = _owner_section_bounds(NUM_PLAYERS, prop_index)
            owner_slice = state[start:end]
            expected = np.zeros(5, dtype=np.float32)
            if owner is not None:
                expected[order.index(owner)] = 1.0
            assert np.array_equal(owner_slice, expected), (agent_id, square_id, owner)


def test_replay_recorder_stores_corrected_observations():
    """ReplayPosition.state recorded by play_local_game must contain
    corrected state, so local_training_update trains on corrected
    semantics. Uses a minimal kind="search" stub policy (not a real
    MaxNPUCT) so this test exercises play_local_game's own recording code
    (`state=game.env._get_state(seat)` in scripts/monopolyzero_common.py)
    without depending on MaxNPUCT's `numpy.random` usage.
    """
    common.ensure_reference_on_path()
    from monopoly_game_engine.actions import ActionType
    from monopoly_game_engine.constants import NUM_PLAYERS, PROPERTY_IDS
    import monopoly_game_engine.env as env_module

    calls: list[tuple[int, dict, np.ndarray]] = []
    patched_fn = env_module.build_state_vector

    def _spy(players, properties_dict, agent_id, env=None):
        owners_snapshot = {sq: properties_dict[sq].owner for sq in PROPERTY_IDS}
        state = patched_fn(players, properties_dict, agent_id, env)
        calls.append((agent_id, owners_snapshot, state))
        return state

    class _AlwaysBuySearchPolicy:
        """kind='search' so play_local_game records a ReplayPosition for
        every decision; always buys when legal so ownership actually
        changes hands during the game, giving a non-vacuous owner mapping
        to check."""

        kind = "search"

        def choose(self, game, seat, decision_seed):
            env = game.env
            legal = tuple(env.get_allowed_actions(seat))
            buy = int(ActionType.BUY_PROPERTY)
            action = buy if buy in legal else legal[0]
            return common._PolicyOnlyResult(
                chosen_action=action, visits={action: 1}, q_values={},
                root_value=(0.0, 0.0, 0.0, 0.0), simulations=0, latency_s=0.0,
            )

    env_module.build_state_vector = _spy
    try:
        outcome = common.play_local_game(
            game_id=1,
            seed=7,
            policies={seat: _AlwaysBuySearchPolicy() for seat in range(NUM_PLAYERS)},
            max_rounds=8,
            record_seats={0, 1, 2, 3},
        )
    finally:
        env_module.build_state_vector = patched_fn

    assert not outcome.crashed
    assert outcome.positions

    nonzero_owner_checks = 0
    for position in outcome.positions:
        found = next((entry for entry in calls if entry[2] is position.state), None)
        assert found is not None, "ReplayPosition.state must be the exact array build_state_vector produced"
        agent_id, owners_snapshot, state = found
        assert agent_id == position.actor_id

        order = common._actor_relative_order(agent_id, NUM_PLAYERS)
        for prop_index, square_id in enumerate(PROPERTY_IDS):
            owner = owners_snapshot[square_id]
            start, end = _owner_section_bounds(NUM_PLAYERS, prop_index)
            owner_slice = state[start:end]
            expected = np.zeros(5, dtype=np.float32)
            if owner is not None:
                expected[order.index(owner)] = 1.0
                nonzero_owner_checks += 1
            assert np.array_equal(owner_slice, expected), (agent_id, square_id, owner)

    assert nonzero_owner_checks > 0  # non-vacuous: at least one property was actually owned
