# Rules Specification

Status: unverified — this file must only contain confirmed competition rules.
Unknown items are marked `TBD` until confirmed against an official source.

## Competition Format

- Organizer: TBD
- Number of players per match: TBD
- Board variant (standard US Monopoly vs. other): TBD
- Starting cash: TBD
- Time/turn limits: TBD
- Win condition (bankruptcy, turn limit, score-based): TBD

## Game Rules In Effect

- Free Parking pot rule: TBD
- Auction on declined property purchase: TBD
- Trading rules (allowed/disallowed, timing): TBD
- Mortgage/unmortgage rules: TBD
- House/hotel building rules: **even building CONFIRMED** per official
  competition guidance (2026-08-11) — houses/hotels must be built evenly
  across a color group (cannot add a 2nd house to one property in the group
  before every property in that group has at least 1, and so on). This is
  the one and only confirmed special game rule so far; do not extend it to
  even-*selling* or anything else without separate confirmation. Consistent
  with (but not sourced from) the reference engine's implementation —
  `monopoly_game_engine/env.py::_improve_actions` gates both `improve_house`
  and `improve_hotel` on `self._is_least_developed(prop)`; see
  `docs/REFERENCE_AUDIT.md`. Building shortage (finite house/hotel bank)
  remains TBD as an official rule, though the reference engine also models
  one (32 houses / 12 hotels).
- Jail rules (doubles, pay to leave, card usage): TBD
- Bankruptcy/elimination handling: TBD

## Agent Interface / Environment

- Game engine / simulator to be used: TBD
- API or protocol the agent must implement: TBD
- Observation space: TBD
- Action space: TBD
- Reward signal (if RL): TBD

## Notes

Do not infer or fabricate any rule above from `DeepRL_Monopoly` or other references — that repo is a technical reference only, not the competition's ruleset. See `docs/REFERENCES.md`.
