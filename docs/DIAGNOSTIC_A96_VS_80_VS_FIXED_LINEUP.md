# Diagnostic (isolated branch only): A96 vs. 80-game champion, both vs. FPAgentA/B/C

**Branch**: `diagnostic/034-a96-vs-80-vs-fpagents`, based on `main@4b26df3`.
**Status of this note**: written and committed BEFORE any game of this
diagnostic is played, per this project's pre-registration discipline. Not
part of `main` — this is a parallel, isolated diagnostic run alongside
(not instead of) the already-completed PROMOTION-scoped `034` gate on
`main`, and does not alter that gate's result or provenance.

## Hypothesis

`034` (on `main`) found `A96` (`032`'s frozen 96-game `A_lr1e-4`
checkpoint) beats the 80-game champion across four PPO-opponent families,
all built from the SAME underlying pure-PPO recipe/opponent-generation
lineage. Does that improvement generalize to a **structurally different
opponent distribution** — the reference's own rule-based `FPAgentA`/
`FPAgentB`/`FPAgentC` (the fixed heuristic agents `027`'s original
training used), never used as an opponent in any of `028`/`030`/`031`/
`033`/`034`'s PPO-vs-PPO evaluation designs?

This is a **diagnostic**, not a champion gate: no promotion/kill decision
authority. It cannot reverse `034`'s already-pre-registered
`CHALLENGER_PROMOTION_GO` result.

## Design

- One focus seat (the checkpoint under test: `A96` or `80`, deterministic
  legal-masked argmax — reuses
  `monopolyzero_pure_ppo_strength_screen.build_masked_argmax_policy`
  unmodified) + the reference's own `FPAgentA`/`FPAgentB`/`FPAgentC` on
  the three non-focus seats, assigned to the non-focus seats in
  ascending player-id order (same convention `train.py`'s own
  `run_episode` uses) — reuses `monopolyzero_common.LocalFixedPolicy`
  (already tested in `tests/test_monopolyzero_common.py`) to drive each
  fixed agent through `monopolyzero_common.play_local_game`, unmodified.
- DEV seeds `44000`-`44007` (8 seeds) — verified below.
- 4 focus-seat rotations per seed = 32 games for `A96`, the SAME 8 seeds
  x 4 rotations = 32 games for `80` = **64 physical games total**.
- `max_rounds=200`. No PUCT/MCTS, no hybrid-compat rule, no training, no
  checkpoint modification.
- Checkpoints referenced directly from the MAIN checkout's
  `artifacts/monopolyzero_pure_ppo_learnability_gate/` directory by
  absolute path (gitignored, not present in this worktree, never
  copied/rebuilt/regenerated) — hash-gated (checkpoint SHA-256 AND actor
  SHA-256) against the already-recorded values before any game:
  - `A96` (`candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt`): checkpoint
    `78585ed4e2400d024633ee2878d8b88243f7f4a9f498d9019a73e44b3a830f51`,
    actor `2bd1e9bad3d6e0100e033507f72f15b5088d8f8fb2f04dba1ae9d759b5a34a40`.
  - `80` (`candidate_ppo_80.pt`): checkpoint
    `e47c8d4559c1d74cfceffe114fa069a8e7a2845ba60af68be0b0ae4bd37e1dae`,
    actor `7442f99e51619b7f6b53bc662e74d360e0a1f9cf2365b8dcb003d5a0cc3eda11`.

## Seed verification (done BEFORE writing this note)

`evaluation_protocol.classify_seed(s) == SEED_CLASS_DEV` for every
`s` in `44000-44007` (part of the registered `44000-44999` "Colab infra"
DEV pool, `docs/EVALUATION_PROTOCOL.md`'s DEV registry table). Confirmed
genuinely unconsumed: no `logs/experiments/*.json` `seeds` field contains
any value in `44000-44007` (checked programmatically across every log);
the only repo hits for the literal digit sequence are unrelated
floating-point noise in `020`'s/`028`'s JSON (e.g. `...0059440012...`,
a probability value, not a seed) and `docs/COLAB_RUNBOOK.md`'s own
*example* commands (a runbook procedure, never actually executed against
this exact sub-range as of this note). Disjoint from
`evaluation_protocol.PROMOTION_SEEDS` and `.FINAL_BLIND_SEEDS`.

## Pre-registered decision rule (diagnostic only — no promotion authority)

Primary statistic: **paired seed-block bootstrap for
`A96_win_rate - 80_win_rate`**, seed as the resampling block — reuses
`evaluation_protocol.pair_records` (pairing by `(seed, seat)`,
`expected_seats=4`) + `evaluation_protocol.paired_seed_block_bootstrap`
unmodified, with `A96` as `candidate_records` and `80` as
`baseline_records` (so `win_rate_diff` is exactly `A96 - 80`).

- **`RED_FLAG`** if the 95% paired bootstrap CI upper bound is `< 0`
  (strictly less than zero — note this is a different threshold
  convention than the `<= 0` KILL rules used by `033`/`034`'s champion
  gates; this diagnostic's threshold is exactly as specified by the task
  that commissioned it), **OR** if any illegal action or crash occurs
  anywhere in the 64 games (immediate reliability RED_FLAG regardless of
  win rate).
- **`NO_CLEAR_DEGRADATION`** otherwise.
- This rule will not be altered after seeing results. Behavior metrics
  (BUY/ACCEPT/DECLINE rates), net worth, bankruptcy, and round-cap rate
  are recorded but are diagnostic-only and cannot override this rule.
- **This diagnostic may NOT automatically reverse `034`'s already
  pre-registered `CHALLENGER_PROMOTION_GO` decision on `main`** — any
  action on this result requires a separate, explicit decision.

## Isolation

- New branch `diagnostic/034-a96-vs-80-vs-fpagents`, new worktree, based
  on `main@4b26df3`. `main`'s own checkout is never checked out, reset,
  pulled, committed, or otherwise modified by this diagnostic.
  `feat/frozen-ppo-inference-layer` is not touched.
- No training, no checkpoint modification, no ASU import anywhere in the
  chain, no PROMOTION or FINAL_BLIND seed consumed.
- This branch is never merged as part of this task.
