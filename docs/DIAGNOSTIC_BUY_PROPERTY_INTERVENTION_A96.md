# Diagnostic (isolated branch only): BUY_PROPERTY runtime intervention on A96

**Branch**: `diagnostic/buy-intervention-a96`, based on `main@0eae9bb`
(the commit that recorded `034`'s `CHALLENGER_PROMOTION_GO` and promoted
A96 - `candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt` - to champion).
**Status of this note**: written and committed BEFORE any game of this
diagnostic is played. Not part of `main` - an isolated, never-merged DEV
experiment. Cannot auto-promote anything.

## Scope change from the original task, agreed with the user before writing any code

The original task specified three arms: `PURE_A96`, `BUY_SIMPLE`, and
`BUY_SAFETY` (the latter gating BUY_PROPERTY on
`ASU_FROZEN_TEACHER.core.safety_breakdown().passed`). `CLAUDE.md`'s ASU
section states: *"ASU may never be used as a runtime fallback or as the
core/final competition agent, under any circumstance"* - documented as
having been **re-locked** specifically to close a prior "ASU as
teacher/expert signal" loophole, on the record as a **competition-rules
requirement**, not a project preference. `BUY_SAFETY` would have used
`safety_breakdown` (ASU-derived, even if a pure financial-safety
calculator rather than a value/policy/rollout call) to gate a live
action inside the champion's real decision path - which reads as exactly
the kind of runtime use that rule forbids unconditionally. Asked
explicitly, the user chose: **run `PURE_A96` and `BUY_SIMPLE` only, drop
`BUY_SAFETY` entirely.** No `ASU_FROZEN_TEACHER` code is imported
anywhere in this diagnostic. Every "safety_breakdown"/"ASU helper call
count" field the original task asked for is reported as **N/A - arm not
run** in the final report, not silently omitted.

## Hypothesis

A96 has a well-documented `BUY_PROPERTY` pathology (collapsed to ~0-10%
of legal opportunities since `030`; `0.0%` against the fixed lineup in
the prior isolated diagnostic). Does a narrow, non-ASU, deterministic
runtime override - `monopoly_game_engine.agent_ppo.fixed_buy_decision`
(already used by the reference's own training loop, not ASU-derived) -
fix that pathology without degrading A96's proven strength against its
own PPO lineage?

## Design

**Arms** (both frozen A96 weights - no training, no weight
modification anywhere):

- **`PURE_A96`**: exact current champion behavior -
  `monopolyzero_pure_ppo_strength_screen.build_masked_argmax_policy`,
  completely unmodified.
- **`BUY_SIMPLE`**: identical to `PURE_A96` at every decision EXCEPT
  when `BUY_PROPERTY` is legal. At that opportunity: call
  `fixed_buy_decision(env, seat)`. If `True`, choose `BUY_PROPERTY`
  directly. If `False`, remove `BUY_PROPERTY` from the legal-action mask
  and run the SAME masked-argmax forward pass over the remaining legal
  actions (same network, same mask mechanics, same counter-bookkeeping
  convention as `build_masked_argmax_policy` - copied verbatim, not
  reimplemented differently). No other action family's logic changes.

**Integrity check**: every non-BUY-opportunity decision must be
action-identical to `PURE_A96` on the identical state. Enforced via
`monopolyzero_common.play_local_game`'s existing `shadow_policy`
mechanism (unmodified) - `PURE_A96`'s own masked-argmax policy is
queried as the shadow at every `BUY_SIMPLE` decision (same pre-step
state, same decision seed, answer recorded not acted on). `BUY_SIMPLE`'s
own policy additionally logs, per decision, whether `BUY_PROPERTY` was
in the legal set. After each game: for every decision where that log is
`False`, the shadow's `agree` field MUST be `True` - asserted at
runtime (`RuntimeError` on violation), not just reasoned about
structurally.

**Contexts** (same 2 arms, same seeds, both contexts):

- **A) Clean PPO context**: focus arm vs. three frozen copies of the
  FORMER 80-game champion (`candidate_ppo_80.pt`) - no fixed agents, no
  fallback-substitution path, a clean read. **This is where the primary
  decision statistic is computed.**
- **B) Structural stress context**: focus arm vs. `FPAgentA`/`FPAgentB`/
  `FPAgentC` (reused `monopolyzero_common.LocalFixedPolicy`, unmodified) -
  every scripted fallback recorded. **Diagnostic only** - fallback
  contamination means this context alone can never promote or kill a
  policy, per this project's existing `fallback_contamination` framing
  (`evaluation_protocol.py`).

**Checkpoints** (referenced from the MAIN checkout's gitignored
`artifacts/monopolyzero_pure_ppo_learnability_gate/` by absolute path -
never copied/rebuilt/regenerated in this worktree - hash-gated,
checkpoint AND actor SHA-256, before any game):

- A96 (`candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt`): checkpoint
  `78585ed4e2400d024633ee2878d8b88243f7f4a9f498d9019a73e44b3a830f51`,
  actor `2bd1e9bad3d6e0100e033507f72f15b5088d8f8fb2f04dba1ae9d759b5a34a40`.
- Former 80-game champion (`candidate_ppo_80.pt`): checkpoint
  `e47c8d4559c1d74cfceffe114fa069a8e7a2845ba60af68be0b0ae4bd37e1dae`,
  actor `7442f99e51619b7f6b53bc662e74d360e0a1f9cf2365b8dcb003d5a0cc3eda11`.

**Seeds**: fresh DEV seeds `53000-53011` (12 seeds), verified before this
note - `classify_seed` returns `SEED_CLASS_DEV` for every one (part of
no registered `DEV_SEED_RANGES` sub-range - a genuinely new range, the
next available slot past `52200-52219`); zero hits in any
`logs/experiments/*.json`'s `seeds` field; disjoint from `PROMOTION_SEEDS`
and `FINAL_BLIND_SEEDS`. 4 seat rotations x 12 seeds x 2 arms x 2 contexts
= **192 physical games**. `max_rounds=200`, no PUCT/MCTS.

## Pre-registered decision rule

**Primary question**: can `BUY_SIMPLE` improve the `BUY_PROPERTY`
pathology without clear degradation in the clean PPO context (context A)?

**Primary statistic** (context A only): paired seed-block bootstrap of
`(BUY_SIMPLE_win_rate - PURE_A96_win_rate)`, seed as the resampling
block - `evaluation_protocol.pair_records` (pairing by `(seed, seat)`,
`expected_seats=4`) + `.paired_seed_block_bootstrap`, unmodified.

- Read as **improved-without-degradation** if the CI lower bound is
  `> 0` (or the point estimate is non-negative and the CI does not
  clearly favor `PURE_A96`).
- Read as **degraded** if the CI upper bound is `< 0`.
- Otherwise **no clear difference**.
- Context B's paired statistic is computed and reported the same way
  but is **diagnostic only** - it cannot by itself promote or kill
  either arm, regardless of its own CI, because of fallback
  contamination.
- Rule fixed before running, not altered after seeing results. **No
  auto-promotion of any arm from this DEV experiment** - any action on
  this result requires a separate, explicit decision.

## Integrity requirements

- A96 checkpoint + actor hash gate; former-champion checkpoint + actor
  hash gate - both before any game.
- Zero behavior change outside `BUY_PROPERTY` opportunity states -
  enforced by the shadow-policy assertion above.
- Zero ASU import anywhere in this diagnostic (source-level guard +
  runtime `monopolyzero_common.loaded_asu_modules()` check, the same
  mechanism every script in this project already uses) - and,
  specifically, `ASUValueV1`/`ASURolloutV1` are asserted never
  instantiated (trivially true: neither name appears anywhere in this
  diagnostic's source).
- No training/optimizer calls anywhere.
- No `PROMOTION`/`FINAL_BLIND` seed consumed.

## Isolation

New branch/worktree based on `main@0eae9bb`; `main` and
`feat/frozen-ppo-inference-layer` are never checked out, reset, pulled,
committed, or otherwise modified. This branch is never merged as part of
this task.
