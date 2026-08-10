# Plan

## Status: Reference validated, first compliant training milestone in progress

No agent code of our own exists yet — everything so far calls the
`references/DeepRL_Monopoly` submodule's existing entry points
(`ASU_FROZEN_TEACHER.evaluate`, `tools/train_and_save.py`) under the ASU
restrictions in `CLAUDE.md`. This document tracks phases as they are defined.

## Phase 0 — Project Setup (done)

- [x] Init repo, create GitHub private repo
- [x] Scaffold docs structure
- [x] Register `DeepRL_Monopoly` as reference submodule

## Phase 1 — Reference validation (done)

- [x] Audit the reference: dependencies, test suite, license — see
      `docs/REFERENCE_AUDIT.md`
- [x] Engine smoke test: fixed-vs-fixed 4-player match runs end to end, no
      crashes, no illegal actions — see `docs/BASELINE.md` (reclassified from
      an earlier "baseline" framing: 62 compatibility fallbacks in that run
      invalidate any win-rate reading, so it proves the engine runs, not
      relative agent strength)
- [x] Found and root-caused a reference reproducibility gap: seeded games are
      not stable across separate process launches unless `PYTHONHASHSEED=0`
      is pinned. `scripts/run_baseline_match.py` now refuses to run without it
- [x] First compliant DDQN training smoke: 20 games, CPU, zero ASU coupling
      (verified by code inspection + grep) — see `docs/EXPERIMENTS.md`
      (2026-08-10 entry). Two Windows dependency gaps found and fixed
      (`psutil`, `PYTHONIOENCODING=utf-8`), no submodule edits
- [x] Multi-seed support added to `run_baseline_match.py` (plain seeds and
      `START-END` ranges), `--seed` single-value usage unchanged

## Phase 2 — First DDQN milestone (current)

- [ ] Reproducibility check: two independent 20-game DDQN training runs, same
      seed, same environment — checkpoint tensors (online/target net,
      optimizer, replay buffer), epsilon, steps, and games trained must match
      exactly; deterministic history fields must match, timing/memory fields
      may differ. Stop on any mismatch — do not proceed to a larger run on an
      unverified-reproducible trainer
- [ ] Resume the 20-game checkpoint to 500 total games (CPU, seed 42), after
      preserving the 20-game checkpoint as a separate, gitignored milestone
      file (never committed — checkpoints are local artifacts only)
- [ ] Greedy paired evaluation of the 20-game and 500-game checkpoints on
      held-out seeds `10000-10009`, seat-rotated, vs. `fixed-a/b/c` — win
      rate + Wilson interval, mean net worth, round-cap rate, fallbacks by
      policy. No improvement claim unless the paired result actually supports
      one
- [ ] Record everything (SHA-256, epsilon, games trained, runtime, all of the
      above) in `docs/EXPERIMENTS.md`

## Next measurable milestones (after Phase 2 lands)

- A DDQN checkpoint trained long enough to beat `fixed-a/b/c` at better than
  chance on held-out seeds, with a paired-seed statistical comparison (Wilson
  interval, not a single-seed anecdote) against the 500-game checkpoint as
  the new reference point.
- Decide, with evidence from that comparison, whether more DDQN training
  games or a different approach is the better next step — this decision must
  itself go through `docs/DECISIONS.md`, not be assumed here.
- ASU may be added as an **evaluation opponent only** (never a data/label
  source, per `CLAUDE.md`) once there is a checkpoint worth evaluating against
  it.
- `docs/RULES_SPEC.md` still needs the official competition ruleset filled
  in — unblocked independently of the training track above, still `TBD`.

## Future Phases (TBD)

Not yet planned — depends on:
- Official competition rules and constraints (see `docs/RULES_SPEC.md`)
- Environment/interface the agent must implement against
- Results of Phase 2's paired evaluation (drives whether we keep scaling DDQN
  games, change algorithm, or change reward shaping)
- Baseline agent requirements before any LLM usage is considered (per `CLAUDE.md`)

Phases will be added here incrementally as they are decided, not speculated in advance.
