# Decisions Log

Record of significant project decisions. Append chronologically, most recent last.

## Format

```
## YYYY-MM-DD — <decision>

- Context:
- Decision:
- Alternatives considered:
- Reference checked (DeepRL_Monopoly / other):
```

## 2026-08-10 — Project scaffolding

- Context: Starting the Monopoly champion agent project from scratch.
- Decision: Set up repo structure, docs, and reference submodule before writing any agent/training code. Follow small-steps + test-before-advance workflow (see `CLAUDE.md`).
- Alternatives considered: N/A (initial setup).
- Reference checked: N/A.

## 2026-08-10 — Baseline via existing fixed/ASU path, no training

- Context: Needed proof the DeepRL_Monopoly reference actually runs locally, plus a reproducible first baseline match, without writing any agent or training code yet.
- Decision: Use the reference's existing `ASU_FROZEN_TEACHER.evaluate` CLI with an all-scripted fixed-agent lineup (no checkpoint needed, fastest available path) via a thin wrapper script (`scripts/run_baseline_match.py`) that imports the submodule at runtime instead of copying code. Installed CPU-only torch (no CUDA) plus numpy/pytest in an isolated venv outside both repos.
- Alternatives considered: `tools/play_game.py` (rejected — requires a trained model checkpoint we don't have and only defaults to 3 fixed opponents, not 4); training a checkpoint first (rejected — explicitly out of scope, "no checkpoint → don't train").
- Reference checked: `references/DeepRL_Monopoly` at `afd9205761317e196d77f679921c35fb04c7ab96`. Found and documented a critical reproducibility gap (`PYTHONHASHSEED` not pinned by the reference's seeding logic) — see `docs/REFERENCE_AUDIT.md`.

## 2026-08-11 — Pause DDQN scaling past 500 games; no improvement yet

- Context: Trained a DDQN checkpoint to 500 games (resumed from a 20-game milestone, seed 42, CPU), after first confirming the trainer is bit-exact reproducible (two independent 20-game runs matched exactly — checkpoint tensors, optimizer, replay buffer, epsilon, steps, games, deterministic history fields; see `docs/EXPERIMENTS.md` 2026-08-11 entry).
- Decision: **Pause further DDQN training for now.** Reproducibility passed, but the 500-game checkpoint's paired evaluation against the 20-game checkpoint on held-out seeds 10000-10009 (seat-rotated, vs. fixed-a/b/c) showed no statistically supported improvement: 20-game checkpoint won 0/40 (Wilson 95% CI [0%, 8.76%]), 500-game checkpoint won 1/40 (Wilson 95% CI [0.44%, 12.88%]) — the intervals overlap almost completely, so the one extra win is not distinguishable from noise. Both checkpoints are still far below fixed-b/fixed-c win rates. Per `CLAUDE.md`, no improvement claim is made, and simply running more of the same (more games at the same schedule) isn't justified yet without evidence it would help.
- Alternatives considered: keep scaling DDQN games blindly toward the paper's 10,000-game reference run (rejected for now — no evidence yet that more of the same training is the bottleneck vs. epsilon schedule, reward shaping, or algorithm choice); switch approach entirely (not yet decided, needs more investigation first).
- Reference checked: `references/DeepRL_Monopoly` at `afd9205761317e196d77f679921c35fb04c7ab96` (submodule unchanged, read-only). Epsilon at 500 games was still ~0.78 (decays toward a 0.05 floor), consistent with 500 games being early relative to the paper's reference run — this is a candidate explanation, not a confirmed cause.

## 2026-08-11 — ASU policy re-locked to evaluation-only per official competition guidance

- Context: Earlier the same day, `CLAUDE.md`'s ASU section was corrected to *allow* ASU as a teacher/expert signal for training our own model (imitation/distillation/bootstrap). New official competition guidance has since arrived that rules this out.
- Decision: **Revoke that correction.** ASU (`ASU_FROZEN_TEACHER`) may only be used as an evaluation opponent, from now on, with no exception:
  - No ASU output/action/value/rollout data may ever be a training label for any model we train.
  - ASU imitation, distillation, teacher-bootstrap, and output-cloning are banned outright.
  - `monopoly_bench collect-asu` and any ASU-guided training path must not be used. This includes `monopoly_bench`'s self-play population generation as shipped (`population_jobs` hardcodes ASU into part of every generation's opponent pool with no disable flag — see `docs/REFERENCE_AUDIT.md`), which must not be used as-is; an ASU-independent self-play setup would need to be built from lower-level primitives with ASU excluded from the opponent pool.
  - ASU may never be a runtime fallback or the core/final competition agent.
  - ASU may never be our Modal training/deployment model.
  - ASU remains an important evaluation opponent and an anti-ASU robustness benchmark — this role is unaffected and is part of the competition strategy below.
- Alternatives considered: keep the ASU-as-teacher allowance and treat the new guidance as advisory (rejected — the guidance is described as official competition rules, which take precedence over an internal project preference); partially allow ASU imitation only for a bootstrap phase (rejected — the new guidance draws no such exception, and a partial allowance would be easy to misapply later).
- Reference checked: `references/DeepRL_Monopoly` at `afd9205761317e196d77f679921c35fb04c7ab96` (submodule unchanged, read-only). No code was run under the now-revoked allowance between the two corrections — `monopoly_bench collect-asu`/`train` were never executed this session (only read-only code inspection, per the standing gate requiring an explicit `docs/DECISIONS.md` entry before running them, which was never made).

## 2026-08-11 — Competition strategy: overall win-rate, hybrid RL + edge-case algorithms, ASU as robustness benchmark

- Context: Official competition guidance clarifies the scoring target and what kinds of agent designs are in scope.
- Decision:
  - Primary objective is **overall/generalizable win-rate** — performing well across a broad range of opponents and situations, not just beating any one fixed baseline (including ASU).
  - A **hybrid approach is allowed**: our trained RL model may be combined with explicit, deterministic algorithms for well-defined edge cases (e.g. a clearly provably-optimal action in a narrow, identifiable situation). The trained model must make the large majority of ordinary decisions — deterministic algorithms are a targeted supplement, not the primary decision-maker, and must not become a way to route around training a real policy for common situations.
  - ASU is an important **evaluation opponent** and an **anti-ASU robustness benchmark** — i.e., part of what "generalizable win-rate" needs to be measured against is not overfitting to beat our own DDQN/fixed-agent evaluation seeds while remaining weak against a structurally different policy like ASU. ASU's role stays evaluation-only (see the ASU policy re-lock decision above) — this is a benchmark use, not a training-data use.
- Alternatives considered: pure end-to-end RL with no deterministic component (not rejected outright, still the default — the hybrid allowance is permissive, not a requirement to add algorithmic edge-case handling); optimizing against a single fixed opponent set (rejected — explicitly not the scoring target per this guidance).
- Reference checked: N/A (competition-rules decision, not a reference-repo finding). Confirmed as the one verified special game rule so far: even-building is enforced by the reference engine (`monopoly_game_engine/env.py::_improve_actions`, `_is_least_developed` gate) — see `docs/RULES_SPEC.md` and `docs/REFERENCE_AUDIT.md`. All other competition rules remain `TBD` and are not being finalized from the reference repo.

## 2026-08-11 — Try an ASU-independent custom self-play wiring smoke

- Context: The MonopolyZero inference path (checkpoint load, policy/value forward, legal-action masking, PUCT/Max-N search) was already validated ASU-free (`docs/EXPERIMENTS.md`'s "ASU-independent MonopolyZero inference smoke" entry). The next question is whether *training* plumbing — collecting search-derived positions into a replay buffer and running an update step — also works, without touching the reference's shipped `Trainer.run_generation` (not usable as-is: it hardcodes ASU into every generation's population, see `docs/REFERENCE_AUDIT.md`).
- Decision: Build a small, self-contained self-play smoke runner in our own repo (`scripts/`), using only the reference's ASU-independent public building blocks (`MonopolyZeroNet`, `MaxNPUCT`/`SearchAdapter`, `arena.play_game`, `ReplayBuffer`, `train_step`) — no `Trainer`, no `population_jobs`, no ASU import anywhere. Opponent pool limited to exactly: self-copy (our own model playing itself) and the three fixed policies `fixed-a`/`fixed-b`/`fixed-c` (`FP_AGENT_CLASSES[:3]` via `FixedAdapter`) — nothing else. 2-4 short games (small `max_rounds`, small simulation count), collect whatever positions those games produce, write at least one batch to a `ReplayBuffer`, and run **exactly one** `train_step` update.
- **Goal is training plumbing, not policy strength.** No claim of a stronger or trained model is being made or intended here — success is defined as: games complete with zero illegal actions and zero crashes, the replay buffer accepts the collected positions, the loss from the one update is finite, and at least one model parameter changed value after the update. Explicitly out of scope for this step: multi-generation training, large self-play volume, Modal, or any LLM component.
- Alternatives considered: modifying `Trainer`/`population_jobs` to add an ASU-exclusion flag (rejected — would mean editing `references/DeepRL_Monopoly`, which must stay read-only); skipping straight to a larger self-play run (rejected — same small-steps-first practice already used for the DDQN smoke-then-scale sequence).
- Reference checked: `references/DeepRL_Monopoly` at `afd9205761317e196d77f679921c35fb04c7ab96` (submodule unchanged, read-only, nothing edited). Only public classes/functions are imported at runtime, as with every other script in `scripts/` so far.

## 2026-08-11 (later) — Correct the self-play smoke's "ASU-independent" claim; rebuild with a genuinely ASU-import-free import graph

- Context: The decision above called `SearchAdapter`, `arena.play_game`, and `train_step` "ASU-independent public building blocks." That was wrong at the import level: `monopoly_bench/adapters.py` (home of `SearchAdapter`/`FixedAdapter`) does `from ASU_FROZEN_TEACHER import ASURolloutV1, ASUValueV1` at module scope, and `monopoly_bench/training.py` (home of `train_step`) does `from ASU_FROZEN_TEACHER import FROZEN_SPEC_HASH` at module scope — so merely `import`ing either module loads `ASU_FROZEN_TEACHER` into `sys.modules`, whether or not any ASU function is ever called. `monopoly_bench/arena.py` has no ASU text of its own but imports `.adapters`, so importing it has the same effect transitively. `scripts/selfplay_train_smoke.py` (both the original version and the reproducibility-fixed version from earlier the same day) imported all three, so `ASU_FROZEN_TEACHER` was loaded as a side effect in every run so far, even though its output was never used as a training label — the ASU-as-teacher/data-source ban was never actually violated, but the "ASU-independent" framing was overstated.
- Decision: Rewrite `scripts/selfplay_train_smoke.py` to import only confirmed ASU-import-clean modules — `monopoly_bench.engine`, `.model`, `.search`, `.storage`, `.config`, `.contracts`, and `monopoly_game_engine.agents_fixed` (each individually grepped for "asu" and their own import statements traced; `monopoly_bench/__init__.py` itself only imports `.config`/`.contracts`, both clean). Replace `SearchAdapter`, `FixedAdapter`, `arena.play_game`, and `train_step` with this project's own implementations built from those lower-level primitives (`MaxNPUCT`, `SharedGame`, `ReplayBuffer`, `FP_AGENT_CLASSES`) — matching their observed behavior, not copied from the reference source. Add a runtime guard that checks `sys.modules` for any `ASU_FROZEN_TEACHER` entry at the end of the run and fails if one is found, so the "ASU-independent" claim is verified every run, not just asserted.
- Verification: the rewritten script reproduced the exact same numbers (709 positions, same winners, same four loss values, 16/16 parameters changed) as the last run of the old, ASU-import-coupled implementation, on the same seed — confirming the reimplementation is behaviorally equivalent. `asu_modules_loaded` was `[]` (count 0) in two independent runs, and the two runs matched each other bit-for-bit outside timing/memory fields (reproducibility re-verified after the rewrite).
- Alternatives considered: leaving the "ASU-independent" claim as "no ASU output used" without correcting the import-level language (rejected — imprecise, and the whole point of `CLAUDE.md`'s ASU rules is to be exact about what "involves ASU" means); patching `adapters.py`/`training.py` to defer their ASU imports (rejected — would mean editing `references/DeepRL_Monopoly`, which must stay read-only).
- Reference checked: `references/DeepRL_Monopoly` at `afd9205761317e196d77f679921c35fb04c7ab96` (submodule unchanged, read-only). Historical results in `logs/experiments/010-*.json` and `011-*.json` are not deleted or edited — corrected via their own `notes` fields, per this project's no-silent-rewrite practice. See `logs/experiments/012-selfplay-asu-import-free-smoke.json`.

## 2026-08-11 (later still) — Pause same-replay update scaling; 500-update checkpoint is a nominal candidate only

*Date corrected 2026-08-11: this entry (and several other "2026-08-12"
labels across this project's docs/logs going back a couple of sessions)
used a date one day ahead of the actual system clock. Not retroactively
rewritten everywhere it appears — the underlying facts/data are unaffected,
only the calendar label was wrong in places. Fixed going forward from here.*

- Context: `016-monopolyzero-update-budget-sweep-paired-eval` trained three checkpoints (100/500/1000 updates) from the same pre-training baseline against the same fixed 37,772-position replay buffer (013's 32 games, zero new self-play), then paired-evaluated all four (0/100/500/1000) on held-out seeds `30000-30009`, 40 games each.
- Decision: **Pause scaling training updates on this same fixed replay buffer further.** Result was NO-SIGNAL: no update budget showed a statistically supported win-rate improvement over another (`100 vs 500` and `500 vs 1000` both failed the non-overlapping-Wilson-interval test). 500 updates had the nominally highest win rate (7.5%, 3/40) of the four, but its interval `[2.6%, 19.9%]` overlaps all the others, and 1000 updates fell back to baseline's win rate (2.5%) despite having the lowest training loss by far — consistent with the value head overfitting the fixed dataset rather than the policy generalizing. **500 updates is therefore a nominal research candidate only, not a proven-best checkpoint** — it should not be treated as "the good one" in any future comparison without new evidence.
- Alternatives considered: keep scaling updates further (2000, 5000, ...) on the same fixed buffer (rejected — the 100→500→1000 trend already shows diminishing-to-reversing win-rate returns while training loss keeps dropping, the textbook overfitting-on-fixed-data signature; more of the same is not expected to help); generate more self-play games before any further update-budget experiments (not decided here — flagged as the more likely lever per `docs/EXPERIMENTS.md`'s 2026-08-12 entry, but starting that needs its own decision when actually proposed, not assumed by this entry).
- Reference checked: `references/DeepRL_Monopoly` at `afd9205761317e196d77f679921c35fb04c7ab96` (submodule unchanged, read-only). No training code changed for this decision — it's a scope/direction call based on 016's results, not a new finding about the reference.

## 2026-08-11 (later still) — Pause current MonopolyZero recipe scaling entirely; no more of the same self-play data

- Context: three independent scaling levers have now each been tested against the 013 32-game self-play dataset / its derived checkpoints, and none produced a statistically supported improvement:
  1. **32-game strength pilot** (013/014): baseline-vs-trained paired eval on held-out seeds — NO-SIGNAL (win-rate difference not statistically supported; round-cap rate moved from 45% to 55% in the adverse direction, flagged but not itself statistically significant).
  2. **Same-replay training-update-budget scaling** (015/016): 100 vs 500 vs 1000 updates on the identical fixed 37,772-position replay buffer — NO-SIGNAL (`100 vs 500` and `500 vs 1000` both failed the non-overlapping-Wilson-interval test; 1000 updates fell back toward baseline despite the lowest training loss, the overfitting-on-fixed-data signature).
  3. **PUCT search-budget scaling** (017): 4 vs 16 vs 32 simulations on 200 frozen decision states from the 500-update checkpoint — KILL (4-vs-16 disagreement 1.5%, mean root-value delta 0.0386, both under the pre-registered 5%/0.05 kill thresholds; no full game eval run, per the task's own decision rule).
- Decision: **Pause scaling the current MonopolyZero recipe.** Neither training longer on the same data, nor searching harder at inference time, nor the original 32-game training pass itself has moved the needle. **No more self-play games will be generated from this same recipe/data source until a new lever is identified and its own experiment is proposed and logged.** This is a scope pause, not an abandonment — it stops further busywork on a data/recipe combination that has now been tested from three independent angles with the same NO-SIGNAL/KILL outcome.
- Alternatives considered: keep sweeping update budgets or simulation counts further past 1000/32 (rejected — each lever already shows either flat or reversing returns, more of the same is not expected to help); generate a larger/fresh self-play dataset immediately (not decided here — remains the most likely next lever per `docs/EXPERIMENTS.md`, but needs its own proposed experiment, not assumed by this pause entry); switch algorithm/architecture (not decided here — see the search-vs-policy-only ablation this entry accompanies, which bears directly on whether the *architecture* itself, not just its training budget, needs to change).
- Reference checked: `references/DeepRL_Monopoly` at `afd9205761317e196d77f679921c35fb04c7ab96` (submodule unchanged, read-only). This is a scope/direction call aggregating 013–017's results, not a new reference finding.

## 2026-08-11 (later still) — Kill the current PUCT/MCTS inference path; pivot next experiment toward search-free learning

- Context: `018-monopolyzero-policy-only-vs-puct-eval` isolated whether PUCT search adds real value over bare policy-head inference, on the same 500-update checkpoint, same held-out seeds `32000-32009`, 4-seat rotation vs fixed-a/b/c (40 games each, no new training/self-play, zero ASU). POLICY_ONLY (legal mask + policy-head softmax + legal argmax, no search) and PUCT_4 (the 4-simulation search used by every prior evaluation in this project) both won 2/40 (5.0%, identical Wilson-95 interval `[1.4%, 16.5%]`); neither showed a statistically supported advantage over the other. On the 31,535 non-forced decisions PUCT_4's own games actually visited, POLICY_ONLY would have chosen the same action 97.07% of the time, while costing ~7.7x-8.0x less latency per decision.
- Decision: **Kill the current PUCT/MCTS inference path as currently configured** — it is not earning its computational cost on this checkpoint. **The next experiment should pivot toward a search-free learning objective/architecture**, not further MCTS tuning (simulation count, depth, etc. — 017 already killed that direction too). This is now the third and most direct of three independent tests (013/014 training pilot, 015/016 update-budget scaling, 017 search-budget scaling, 018 search-vs-no-search) all converging on the same conclusion for this checkpoint family: the missing skill is not being unlocked by spending more compute at either training time or inference time within the current recipe.
- Alternatives considered: keep PUCT as the production inference path anyway on the theory that search should help in principle even if this measurement didn't show it (rejected — the whole point of this ablation was to test that assumption directly, on real held-out games, not asserted; a 2.93% disagreement rate with no win-rate difference is a clear negative result, not an inconclusive one); re-run with a larger seed sample to see if the 5.0%-vs-5.0% tie breaks (not decided here — the Wilson intervals here are already wide at n=40 per policy, so a larger sample is plausible future work, but it competes with the more promising open levers below rather than being an obvious next step on its own).
- Reference checked: `references/DeepRL_Monopoly` at `afd9205761317e196d77f679921c35fb04c7ab96` (submodule unchanged, read-only). No training or reference code changed for this decision.

## 2026-08-11 (later still x2) — Deprecate the 013 replay for strength training; horizon mismatch confirmed

- Context: `019-monopolyzero-horizon-diagnostic` measured whether a round-50
  net-worth leader (013's replay was generated at `max_rounds=50`) predicts
  the round-200-or-terminal winner, on 32 fresh held-out games (seeds
  `40000-40015`/`41000-41015`, never used before). Overall agreement was
  **19/32 (59.4%)**. Split by category: **self-play 6/16 (37.5%), clean** —
  no fixed-agent fallback substitutions possible in a self-play game, so
  this number reflects the model's own play only. **vs-fixed 13/16
  (81.25%), but contaminated**: 90 fixed-agent fallback substitutions
  occurred across those 16 games (`TheDealMaker`/`TheGambler`/`TheHoarder`
  proposing an illegal action and being substituted) — the vs-fixed
  agreement number is not a clean read of the checkpoint's own horizon
  behavior, since the opponents' actual policy was partly a fallback
  substitute rather than their real scripted decision. Separately, the same
  experiment ran a state-encoding ablation: cloning 200 non-forced
  decision states from rounds 1-50 and flipping only `env.max_rounds`
  (200 vs 50) confirmed, at runtime (not assumed), that exactly state-vector
  index 278 (`round/max_rounds`) changes and nothing else, with small but
  nonzero POLICY_ONLY action disagreement (1.5% baseline, 1.0% trained) and
  a notably larger value-head delta for the trained checkpoint (~7.5x
  baseline's).
- Decision: **The existing 37,772-position replay from `013` is
  DEPRECATED/KILLed as a source for any new strength-training run** — not
  deleted (kept as historical/reference data, referenced by path and
  provenance, same as any other artifact) — because a 59.4% overall / 37.5%
  clean-self-play round-50-leader-equals-final-winner rate is too weak a
  proxy signal for a `max_rounds=200`-evaluated agent to train on. **The
  state-index-278 ablation result, on its own, is explicitly NOT sufficient
  grounds for a feature-removal decision** (e.g. dropping or reweighting
  the round/max_rounds scalar from the state encoding): the 200-state
  sample was collected as the *first* 200 non-forced states encountered
  across the 32 games in fixed (game-order, then turn-order) sequence, not
  a round-stratified sample — so it is not guaranteed to represent rounds
  1-50 evenly, and a feature-level change should not be made from an
  un-stratified convenience sample. If that specific question (does the
  round/max_rounds scalar itself need to change) becomes worth answering
  later, it needs its own round-stratified experiment, not a reuse of this
  one's states.
- Alternatives considered: treat the 81.25% vs-fixed number as the headline
  figure (rejected — it's inflated by fallback contamination, not a clean
  signal; self-play's 37.5% is the trustworthy number here); make a
  feature-removal call on the state encoding directly from the index-278
  ablation (rejected — explicitly out of scope per the sampling caveat
  above); keep training on the 013 replay anyway on the theory that a weak
  proxy is still better than nothing (rejected — no evidence offered for
  that claim, and it contradicts this project's own "no unverified
  assumptions" rule in `CLAUDE.md`).
- Reference checked: `references/DeepRL_Monopoly` at `afd9205761317e196d77f679921c35fb04c7ab96` (submodule unchanged, read-only). No training or reference code changed for this decision. Note: `max_rounds=200` here is this project's own current reference evaluation horizon, not a confirmed official competition parameter — see `docs/PLAN.md`'s P0 blocker note.
