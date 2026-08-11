# Plan

## Status: MonopolyZero recipe/inference scaling paused pending methodology lockdown

This project has its own agent code now — `scripts/monopolyzero_common.py`
plus purpose-built training/eval/diagnostic runners (`scripts/monopolyzero_*.py`,
see the MonopolyZero section below), built on the ASU-clean primitives of the
`references/DeepRL_Monopoly` submodule (`MonopolyZeroNet`, `MaxNPUCT`,
engine/storage/config/contracts, `agents_fixed`) under the ASU restrictions in
`CLAUDE.md`. The reference submodule itself stays read-only, consumed via
import at its pinned SHA, never edited in place. This document tracks phases
as they are defined.

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

## Next measurable milestones (Phase 3, current)

- **DDQN long-run scaling is paused, not the current milestone.** After the
  500-game paired evaluation showed no statistically supported improvement
  over the 20-game checkpoint (`docs/DECISIONS.md`'s "Pause DDQN scaling
  past 500 games" entry), the decision was to pause further DDQN games
  rather than assume more of the same training helps. Resuming that track
  would need its own new `docs/DECISIONS.md` entry with a reason to expect
  it'll help this time — not assumed here.
- **Current MonopolyZero recipe scaling is paused** (training-update budget,
  PUCT simulation count, and the 32-game training pass itself all tested —
  see `docs/DECISIONS.md`'s "Pause current MonopolyZero recipe scaling
  entirely" entry).
- **The current PUCT/MCTS inference path is KILLed for the current
  recipe/checkpoint family only** — `018` showed it has no statistically
  supported win-rate advantage over bare policy-head inference on held-out
  games (2/40 vs 2/40, identical Wilson interval), at ~8x the per-decision
  latency, with only 2.93% action disagreement. See `docs/DECISIONS.md`'s
  "Kill the current PUCT/MCTS inference path" entry. **POLICY_ONLY
  (`common.build_local_policy_only`) is now the diagnostic/default
  inference path** for this checkpoint family — PUCT stays available for
  reference/comparison only, not as the assumed default in new scripts.
  This is not a permanent verdict on search itself: search adds value only
  when the value/policy head it searches over is informative enough to
  reward deeper lookahead, so PUCT is a reasonable thing to re-test once a
  materially better value model exists, not before.
- **The existing 37,772-position replay (from `013`) is DEPRECATED/KILLed
  for strength training** — not deleted, kept as historical data — per the
  horizon diagnostic (`019`, see `docs/EXPERIMENTS.md`). That replay was
  generated at `max_rounds=50`, and `019` measured a round-50 net-worth
  leader agreeing with the eventual winner only 59.4% overall (37.5% in
  clean self-play, 81.25% in vs-fixed but that subgroup is contaminated by
  90 fixed-agent fallbacks — see `docs/DECISIONS.md`'s `019` entry) — too
  weak a proxy signal to justify training on. Note the horizon this replay
  was checked against (`max_rounds=200`) is this project's own *current
  reference evaluation horizon*, not a confirmed official competition
  parameter — see the P0 blocker note below. The state-encoding ablation in
  the same experiment (isolating index 278, `round/max_rounds`) is a
  narrower, separate result and is not on its own sufficient grounds for a
  feature-removal decision — see `docs/DECISIONS.md`'s `019` entry for the
  sampling-methodology caveat.
- **The 500-update checkpoint (`trained_updates_500.pt`) is
  diagnostic/research-only** — it exists to compare training-update budgets
  and to drive ablations (`016`-`019`), not as a candidate for production
  use or further promotion testing as-is.
- **Official competition engine/rules/API remain a P0 blocker.** The
  `references/DeepRL_Monopoly` reference engine (`ppo-plus-v2` ruleset) is
  a technical reference only, per `CLAUDE.md` — it must never be treated as
  the official competition engine/ruleset/API. In particular,
  `max_rounds=200` (used throughout `014`-`019`) is this project's own
  *current reference evaluation horizon*, chosen for consistency across
  experiments — it is **not** a confirmed official competition parameter.
  The real competition horizon/rules/API remain `TBD` in
  `docs/RULES_SPEC.md` beyond even-building; that gap is independent of and
  blocks on top of every training/inference result recorded so far.
- **Evaluation methodology: GO.** `docs/EVALUATION_PROTOCOL.md` — DEV /
  PROMOTION / FINAL_BLIND seed pools with a scope-exclusive guard
  (`require_seed_scope`), and cluster-aware paired statistics in
  `scripts/evaluation_protocol.py`: a seed-block paired randomization test
  + seed-block bootstrap as the **primary** paired-comparison evidence
  (seat-level McNemar is kept only as a **secondary** diagnostic, since the
  4 seats per seed are a cluster, not independent trials) — is adopted for
  all evaluation going forward. Own test suite passing is what makes this a
  GO; it is methodology, not an experiment result, so it has no
  `logs/experiments/` entry of its own. Any new paired evaluation should
  use that protocol rather than a from-scratch Wilson-non-overlap check.
- **Current next milestone: a full-horizon value-learnability probe
  (`020`).** Before any new strength/policy training, measure whether the
  existing 300-dim state representation carries the true full-horizon
  (`max_rounds=200`) final winner in a learnable way at all — a small,
  separate, own-written `ValueProbe` supervised on real-final-winner labels
  from 64 clean `POLICY_ONLY` self-play games, compared against uniform and
  current-net-worth-leader baselines. Does not touch the policy network,
  run PUCT, or make any win-rate/strength claim — see
  `docs/EXPERIMENTS.md`/`logs/experiments/020-*.json` once run.
- **Not yet approved: a factorized action head, or training any new actor
  network.** Both are plausible next steps if `020` shows the state
  representation *does* carry a learnable full-horizon signal, but neither
  is decided or started here — either would need its own proposed
  experiment and a `docs/DECISIONS.md` entry after `020`'s results are
  read, not assumed from this milestone note.

## MonopolyZero (`monopoly_bench`) — ASU-independent parts only

Investigated 2026-08-11, then corrected the same day — see
`docs/REFERENCE_AUDIT.md` for the full write-up and
`docs/DECISIONS.md`'s "(later)" correction entry for exactly what changed.

- **Usable as-is (confirmed ASU-import-clean, not just "no ASU calls")**:
  `MonopolyZeroNet` (`monopoly_bench/model.py`), `MaxNPUCT` (PUCT/Max-N
  search, `monopoly_bench/search.py`), replay storage
  (`monopoly_bench/storage.py`), `monopoly_bench/engine.py`, `.config`,
  `.contracts`, and `monopoly_game_engine.agents_fixed` (`FP_AGENT_CLASSES`).
  Confirmed by reading each module's own import statements, not just
  grepping for the literal string "asu" in its body.
- **NOT ASU-import-free — do not import these in a training process, even
  though none of them ever call ASU**: `monopoly_bench.adapters` (`from
  ASU_FROZEN_TEACHER import ASURolloutV1, ASUValueV1` at module scope),
  `monopoly_bench.training` (`from ASU_FROZEN_TEACHER import
  FROZEN_SPEC_HASH` at module scope), and `monopoly_bench.arena` (imports
  `.adapters`, so it's the same problem transitively — an earlier version of
  this document called `arena.play_game` "ASU-independent," which was
  wrong at the import level; corrected 2026-08-11). Importing any of these
  loads `ASU_FROZEN_TEACHER` into `sys.modules` as a side effect. `Trainer`/
  `monopoly_bench train` is additionally unusable on its own merits: its
  bootstrap directly imitates ASU (`bootstrap_asu_expert`/
  `expert_train_step`, now banned), and its self-play population generation
  (`training.py::population_jobs`) hardcodes ASU into part of every
  generation's opponent pool with no disable flag.
- `monopoly_bench/cli.py`'s `smoke` subcommand loads a PPO checkpoint into
  `MonopolyZeroNet` and runs one `MaxNPUCT.choose_action` call — zero ASU
  involvement, zero training (pure inference). Done — see
  `docs/EXPERIMENTS.md`.
- This project's own ASU-import-free wiring lives in
  `scripts/monopolyzero_common.py`: a game loop, a search-policy wrapper, a
  fixed-agent wrapper, and a training-update step, all built from the
  ASU-clean primitives above with this project's own control flow/
  expression (not copied from `adapters.py`/`arena.py`/`training.py` — see
  the source-similarity audit note in that file's docstring and
  `docs/DECISIONS.md`). A runtime guard checks `sys.modules` for
  `ASU_FROZEN_TEACHER` at the end of every run using it.

### MonopolyZero progress

1. ✅ PPO-compatible checkpoint trained (architecture compatibility only) —
   `docs/EXPERIMENTS.md`.
2. ✅ `python -m monopoly_bench smoke` passed — inference path validated.
3. ✅ ASU-import-free self-play training-plumbing smoke (3 games, 1 update) —
   `docs/EXPERIMENTS.md`, `logs/experiments/012-*.json`.
4. ✅ Small strength pilot (32 games, paired evaluation against a
   pre-training baseline) — NO-SIGNAL (`013`/`014`).
5. ✅ Same-replay update-budget scaling (100/500/1000 updates) — NO-SIGNAL
   (`015`/`016`).
6. ✅ PUCT search-budget scaling (4/16/32 simulations) — KILL (`017`).
7. ✅ POLICY_ONLY vs PUCT_4 ablation — KILL the PUCT inference path, no
   search-vs-no-search advantage found (`018`).
8. ✅ Horizon diagnostic — round-50 net-worth leader only agreed with the
   round-200/terminal winner 59.4% overall (37.5% clean self-play, 81.25%
   vs-fixed but fallback-contaminated); `013`'s replay is DEPRECATED/KILLed
   for strength training as a result (`019`, see `docs/DECISIONS.md` and
   `docs/EXPERIMENTS.md`).
9. ✅ Methodology lockdown before any new training — DEV / PROMOTION /
   FINAL_BLIND seed-pool discipline (scope-exclusive guard) and cluster-aware
   paired statistics (seed-block randomization test + bootstrap as primary,
   seat-level McNemar demoted to secondary), replacing ad-hoc
   Wilson-interval comparisons as the promotion test. See
   `docs/EVALUATION_PROTOCOL.md` and `scripts/evaluation_protocol.py`.
   **GO.**
10. Current: full-horizon value-learnability probe (`020`) — does the
    existing 300-dim state representation carry the true `max_rounds=200`
    final winner in a learnable way, before any new policy/strength
    training is proposed. No policy change, no PUCT, no win-rate claim.

## Future Phases (TBD)

Not yet planned — depends on:
- Official competition rules and constraints (see `docs/RULES_SPEC.md`)
- Environment/interface the agent must implement against
- Results of Phase 2's paired evaluation (drives whether we keep scaling DDQN
  games, change algorithm, or change reward shaping)
- Baseline agent requirements before any LLM usage is considered (per `CLAUDE.md`)

Phases will be added here incrementally as they are decided, not speculated in advance.
