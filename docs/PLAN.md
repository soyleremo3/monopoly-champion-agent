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

## Competition strategy (locked 2026-08-11, see `docs/DECISIONS.md`)

- Primary objective: **overall/generalizable win-rate**, not beating any one
  fixed opponent (including ASU).
- Hybrid RL + explicit deterministic edge-case algorithms are allowed, but
  the trained model must make the large majority of ordinary decisions —
  algorithms are a targeted supplement for rare, well-defined edge cases.
- **ASU is evaluation-only, locked, no exceptions**: fixed evaluation
  opponent and anti-ASU robustness benchmark. Never a training label, never
  imitated/distilled/bootstrapped-from, never a runtime fallback, never the
  core/final agent, never our Modal training/deployment model. See
  `CLAUDE.md`'s ASU Restrictions section for the full, current rule set —
  this supersedes an earlier same-day version of that section that had
  (incorrectly) allowed ASU as a teacher.
- `docs/RULES_SPEC.md`'s one confirmed special rule: even building
  (color-group houses/hotels must be built evenly). Everything else in that
  file stays `TBD` until separately confirmed — do not infer more rules from
  the reference engine.

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
- `docs/RULES_SPEC.md` still needs the rest of the official competition
  ruleset filled in beyond even-building — unblocked independently of the
  training track above, still `TBD`.

## MonopolyZero (`monopoly_bench`) — ASU-independent parts only

Investigated 2026-08-11 (read-only, submodule untouched, nothing run beyond
what's already logged in `docs/EXPERIMENTS.md`) — see `docs/REFERENCE_AUDIT.md`
for the full write-up. Summary of what's usable under the ASU re-lock:

- **Usable as-is (no ASU coupling)**: `MonopolyZeroNet` (policy/value
  network, `monopoly_bench/model.py`), `MaxNPUCT` (PUCT/Max-N search,
  `monopoly_bench/search.py` — zero ASU references, verified by grep),
  checkpoint/replay storage (`monopoly_bench/storage.py` — zero ASU
  references), the generic `arena.play_game` mechanism (zero ASU references
  itself; `monopoly_bench/ladder.py` chooses to call it with an ASU opponent
  for gating, which is compliant evaluation use, not training).
- **Not usable as shipped**: `Trainer`/`monopoly_bench train`'s bootstrap
  (`bootstrap_asu_expert`/`expert_train_step` — direct ASU imitation, now
  banned) and its self-play population generation
  (`training.py::population_jobs` hardcodes `asu_count = max(1,
  baseline_count // 2)` into every generation's "baseline" opponent slice,
  with no config flag to exclude ASU). Using `Trainer.run_generation` as-is
  would seat ASU as an opponent in self-play games that feed our replay
  buffer and training updates — too close to the re-locked line to use
  without modification, so it's out until we build our own opponent-pool
  wiring that excludes ASU entirely.
- `monopoly_bench/cli.py`'s `smoke` subcommand does **not** use `Trainer` at
  all — it only loads a PPO checkpoint into `MonopolyZeroNet` and runs one
  `MaxNPUCT.choose_action` call (4 simulations, depth 16). Zero ASU
  involvement, zero training (pure inference). This is the smallest
  ASU-independent smoke available, but requires a PPO checkpoint
  (`artifacts/ppo_plus/ppo_hybrid_2000_v2.pt` by default) that we do not
  currently have — we've only trained DDQN checkpoints so far, and DDQN
  weights are not PPO-actor-compatible (different network class).

### Smallest ASU-independent MonopolyZero smoke plan (not yet started)

1. Train a small PPO checkpoint purely for architectural compatibility (not
   policy quality) — e.g. `tools/train_and_save.py --algo ppo --games 20`,
   mirroring the already-validated DDQN 20-game smoke recipe
   (`PYTHONHASHSEED=0`, `PYTHONIOENCODING=utf-8`, `psutil` installed,
   reproducibility check recommended the same way as the DDQN one before
   trusting it).
2. Run `python -m monopoly_bench smoke` against that checkpoint — inference
   only, no ASU, no training. Confirms `MonopolyZeroNet.load_ppo_actor` +
   `MaxNPUCT` work end to end on our machine.
3. Only after that passes: consider a hand-built, ASU-excluded self-play
   loop using `MaxNPUCT` + `arena.play_game` + a hand-picked opponent pool
   (fixed-a/b/c, and/or our own PPO/DDQN checkpoints — never
   `ASUAdapter`/`asu_value_v1`/`asu_rollout_v1`) — this is genuinely new
   wiring, not an existing entry point, and needs its own
   `docs/DECISIONS.md` entry and small-step plan before being built. Not
   started.

## Future Phases (TBD)

Not yet planned — depends on:
- Official competition rules and constraints (see `docs/RULES_SPEC.md`)
- Environment/interface the agent must implement against
- Results of Phase 2's paired evaluation (drives whether we keep scaling DDQN
  games, change algorithm, or change reward shaping)
- Baseline agent requirements before any LLM usage is considered (per `CLAUDE.md`)

Phases will be added here incrementally as they are decided, not speculated in advance.
