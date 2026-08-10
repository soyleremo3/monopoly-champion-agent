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

## Phase 2 — First DDQN milestone (done)

- [x] Reproducibility check: two independent 20-game DDQN training runs, same
      seed, same environment — checkpoint tensors (online/target net,
      optimizer, replay buffer), epsilon, steps, and games trained matched
      exactly (bit-exact via `torch.equal`); deterministic history fields
      matched too. **Passed** — see `docs/EXPERIMENTS.md` (2026-08-11 entry)
      and the new `scripts/compare_ddqn_checkpoints.py`. Note: raw checkpoint
      file SHA-256 differs between reproducible runs (`torch.save` container
      non-determinism) — that is expected and not itself a mismatch signal
- [x] Resumed the 20-game checkpoint to 500 total games (CPU, seed 42, 111.4
      min wall time), after preserving the 20-game checkpoint as
      `artifacts/training_smoke/milestones/ddqn_hybrid_20_v2_milestone.pt`
      (gitignored, not committed)
- [x] Greedy paired evaluation of the 20-game and 500-game checkpoints on
      held-out seeds `10000-10009` (never used in training), seat-rotated,
      vs. `fixed-a/b/c`. Win rate: 20-game 0/40 (Wilson [0%, 8.76%]),
      500-game 1/40 (Wilson [0.44%, 12.88%]) — intervals overlap almost
      completely. **No improvement claim made**: 500 games at `epsilon≈0.78`
      is not enough training for a measurable skill signal yet
- [x] Recorded SHA-256, epsilon, games trained, runtime, win rates, Wilson
      intervals, mean net worth, round-cap rate, fallbacks-by-policy in
      `docs/EXPERIMENTS.md`

## Next measurable milestones (Phase 3, not yet started)

- A DDQN training run long enough to get `epsilon` meaningfully below the
  ~0.78 reached at 500 games — the paper's own reference run used 10,000
  games (`references/DeepRL_Monopoly/PPO_PLUS_RULES.md`) — then re-run the
  exact same paired-seed evaluation protocol (seeds `10000-10009`,
  seat-rotated, vs. `fixed-a/b/c`) against both the 20-game and 500-game
  checkpoints as reference points, again requiring a statistically supported
  Wilson-interval separation before claiming any improvement.
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
