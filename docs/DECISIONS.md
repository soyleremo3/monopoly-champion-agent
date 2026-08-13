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

## 2026-08-11 (later still x3) — 300-dim state carries a learnable full-horizon signal (GO); 020's figure is not an unbiased generalization estimate

- Context: `020-monopolyzero-value-learnability-probe` trained a small,
  separate `ValueProbe` (300 → 256 → 4) on the TRUE full-horizon winner
  label from 64 clean `POLICY_ONLY` self-play games, and compared it
  against a uniform baseline and a current-net-worth-leader baseline. The
  `ValueProbe` clearly beat uniform on its held-out split (cross-entropy
  0.716 vs. 1.386, top-1 accuracy 69.2% vs. 32.2%) — real signal exists in
  the 300-dim state encoding for the full-horizon winner, not just noise.
- Decision: **GO** — the state representation does carry a learnable
  full-horizon-winner signal; that is a real, usable finding. Two things
  about how `020` was run need to be on the record alongside it, though,
  so this GO isn't misread as stronger than it is:
  1. **020's "validation" split doubled as its own early-stopping
     monitor** — training was stopped based on that exact split's loss,
     which makes it a model-selection set, not a held-out generalization
     test. Its reported 69.2% accuracy / 0.716 cross-entropy **must not be
     quoted as an unbiased estimate of how the `ValueProbe` generalizes to
     unseen games** — early-stopping-on-the-same-split-you-report-on
     optimistically biases the reported number by an unknown amount. A
     proper three-way train/selection/test split, with the test split
     touched exactly once, is needed before that number can be trusted —
     see `021`.
  2. **Correction to how `020` was summarized**: the net-worth-leader
     baseline beat the learned `ValueProbe` on accuracy (91.0% vs. 69.2%)
     and Brier score (0.179 vs. 0.427), but the learned `ValueProbe` was
     actually *better* than the leader baseline on cross-entropy (0.716
     vs. 1.238). "The leader baseline wins every metric" is not an
     accurate summary of `020`'s own numbers and should not be repeated —
     each metric answers a different question (cross-entropy rewards
     well-calibrated probabilities across the whole distribution; the hard
     leader baseline is a degenerate near-0/1 predictor that gets
     penalized hard on cross-entropy whenever it's wrong, even though its
     argmax is right more often).
- Alternatives considered: treat `020` as fully conclusive and move
  straight to policy/architecture changes (rejected — the early-stopping/
  test-set conflation means the generalization claim isn't actually
  supported yet, only the weaker "there is *some* learnable signal" claim
  is); discard `020` entirely as invalid (rejected — the uniform-vs-learned
  comparison inside `020` doesn't depend on the validation/test conflation,
  since uniform has no free parameters to overfit to that split; only the
  *generalization magnitude* claim is what's unsupported).
- Reference checked: `references/DeepRL_Monopoly` at `afd9205761317e196d77f679921c35fb04c7ab96` (submodule unchanged, read-only). No training or reference code changed for this decision.

## 2026-08-11 (later still x5) — Close 021: KILL the current learned-value-probe direction; move to the decision/policy win-rate phase

- Context: `021-monopolyzero-value-generalization-probe` gave an unbiased,
  properly-held-out TEST read (fixing `020`'s validation/test conflation):
  the learned `ValueProbe` lost to the probabilistic net-worth-leader
  baseline on every metric (cross-entropy 1.182 vs. 0.063, Brier 0.643 vs.
  0.032, accuracy 55.5% vs. 96.6%), with a game-block bootstrap (unit =
  TEST game, not state) confirming all three metric-difference 95% CIs
  exclude zero in the adverse direction (CE +1.119 `[+0.747, +1.555]`,
  Brier +0.612 `[+0.450, +0.762]`, accuracy −0.411 `[−0.503, −0.309]`,
  n=16 games). Before closing that out, `022-monopolyzero-value-decision-audit`
  did the due-diligence check this decision needed: a post-hoc,
  decision-critical audit of `021`'s exact TEST predictions (deterministically
  re-derived and reconciled bit-exact against `021`'s own logged numbers,
  zero new self-play/model/PUCT/temperature-fit) segmented by round bucket,
  leader-margin quartile, current-player rank, decision type, and
  legal-action count, explicitly searching for ANY segment where the
  learned probe meaningfully beat the leader. **None was found** - every
  segment across all five axes still favored the leader baseline, most by
  a wide margin (leader 86-100% accuracy vs. `ValueProbe` 35-77%
  everywhere), against a bar (≥20 states, ≥5-point accuracy margin) fixed
  before the audit ran.
- Decision: **KILL the current learned-value-probe direction** on this
  checkpoint/state-representation family - `021`'s point estimate plus its
  own confirmatory game-block bootstrap plus `022`'s exhaustive
  post-hoc segment search all converge on the same answer, with no
  qualifying counter-evidence anywhere. **Final call, per `022`'s
  pre-stated decision rule: option A** - drop the learned-value path for
  now and move to the decision/policy win-rate phase (the next milestone
  this project's actual competition objective, per `docs/PLAN.md`'s
  competition-strategy section, actually needs). Option B (propose a new
  value hypothesis) is explicitly not taken - `022` found no strong
  evidence to warrant it, and this project's own discipline (`CLAUDE.md`:
  "do not make unverified assumptions") rules out proposing a new
  hypothesis speculatively when the audit built specifically to look for
  supporting evidence came up empty.
- Alternatives considered: keep iterating on `ValueProbe` architecture/
  training procedure on the theory that more tuning would close the gap
  (rejected - the SELECTION learning curve in `021` was still improving at
  64 games with no plateau, so more *data* remains plausible future work,
  but `022` found no segment-level evidence that architecture/tuning
  specifically is the bottleneck rather than data volume or the ceiling of
  what a 300-dim net-worth-derived state can support); treat `022`'s
  round-26-onward 100% leader accuracy as grounds for a NEW value
  hypothesis built around net worth directly (not decided here - noted as
  a live possibility in `docs/EXPERIMENTS.md`'s `022` entry, but proposing
  it formally needs its own experiment design, not an inference from this
  closing decision).
- Reference checked: `references/DeepRL_Monopoly` at `afd9205761317e196d77f679921c35fb04c7ab96` (submodule unchanged, read-only). No training or reference code changed for this decision.

## 2026-08-11 (later still x7) — 023 corrections: bootstrap mismatch CONFIRMED, current POLICY_ONLY checkpoint KILL as a strength candidate

- Context: `023-hybrid-bootstrap-isolation-audit` isolated the effect of
  `baseline_pretraining.pt`'s bootstrap mismatch (its actor is copied in
  full from a hybrid-PPO checkpoint whose `BUY_PROPERTY`/`ACCEPT_TRADE`
  action-head rows never received a gradient update) under plain
  POLICY_ONLY inference, with a passing integrity gate and two PRIMARY
  cluster-aware statistics both excluding zero. Six corrections/scope
  clarifications follow directly from that result and must be on the
  record before any further decomposition (`024`) or strength work:
  1. **Bootstrap mismatch: CONFIRMED, not merely isolated.** The diagnostic
     `HYBRID_COMPAT` arm beat plain `POLICY_ONLY` by +21.25 points win
     rate (46.25% vs. 25.0%), seed-block paired randomization p=0.000122,
     bootstrap 95% CI `[0.1375, 0.2875]` excluding zero. This is a real,
     large, statistically supported effect, not noise.
  2. **The current `POLICY_ONLY` checkpoint is KILLed as a strength
     candidate.** `023`'s intervention audit showed plain `POLICY_ONLY`
     never once chooses `BUY_PROPERTY` or `ACCEPT_TRADE` (0/2,616 buy
     opportunities, 0/5,557 trade opportunities) despite assigning them
     non-trivial probability - a systematic "never buy property, never
     accept a trade" defect, not sampling noise. No plain-`POLICY_ONLY`
     checkpoint should be proposed as a submission/strength candidate
     until this is fixed at the representation/training level.
  3. **`HYBRID_COMPAT` is a diagnostic upper-bound only, not a submission
     candidate.** It exists to *measure the size of the problem* by
     runtime-importing the reference's own fixed rules - it is not a
     principled fix (no retraining happened, and depending on a
     hand-picked external heuristic for two action types is not a
     designed policy). A real fix requires retraining or otherwise
     correcting the `BUY_PROPERTY`/`ACCEPT_TRADE` representation itself;
     that is separate, not-yet-started work.
  4. **The 46.25% win rate is not a general strength claim.** It was
     measured with all three opponents also running the same broken
     no-buy/no-accept `POLICY_ONLY` policy - a crippled-peer population,
     not a neutral or competitive baseline. Whether the `BUY_PROPERTY`/
     `ACCEPT_TRADE` fix still matters against stronger peers is exactly
     what `024`'s "repaired peer population" context is for - not yet
     answered as of this entry.
  5. **`020`-`022`'s learned-value-probe KILL decision is re-scoped, not
     retracted.** All self-play games in `020`/`021`/`022` were generated
     under this same broken, no-buy/no-accept `POLICY_ONLY` distribution
     (`baseline_pretraining.pt`, zero PUCT). Their KILL conclusion (the
     probabilistic net-worth-leader beats the learned `ValueProbe` on
     every metric/segment) holds **for states drawn from that specific
     broken-policy distribution** - it is not yet established to hold for
     a state distribution generated by a policy that actually buys
     property and negotiates trades, since games under a fixed policy
     would look structurally different (more monopolies built, longer
     games, different net-worth trajectories). Re-running that
     value-learnability question after any `BUY_PROPERTY`/`ACCEPT_TRADE`
     fix is a live possibility, not decided here.
  6. **`baseline_pretraining.pt` provenance, restated precisely:** its PPO
     bootstrap (`references/DeepRL_Monopoly/artifacts/ppo_plus/ppo_hybrid_2000_v2.pt`)
     is this project's own 1-game/598-step local stand-in checkpoint
     (experiment `007`), generated solely to satisfy `load_ppo_actor`'s
     format/metadata check - it is **not** the upstream reference's
     documented "2000-game" hybrid-PPO artifact (confirmed by SHA-256
     mismatch against `references/DeepRL_Monopoly/TRAINING_RESULTS.md` in
     `023`). The filename's "2000" does not describe this project's
     checkpoint.
- Decision: record all six corrections above; no new action beyond that in
  this entry - `024` (BUY_PROPERTY/ACCEPT_TRADE contribution decomposition,
  crippled- and repaired-peer contexts) is the next step already in
  progress, not decided here.
- Alternatives considered: leave `023`'s result standing without an
  explicit KILL-as-candidate statement for the current checkpoint
  (rejected - `CLAUDE.md` requires tests/verified findings to gate
  advancement, and an unstated implication that the current checkpoint
  might still be viable is exactly the kind of unverified assumption that
  rule exists to prevent); retract `020`-`022`'s KILL decision outright
  instead of re-scoping it (rejected - the audit chain in `021`/`022` is
  internally sound for the distribution it actually measured; the open
  question is external validity to a *different* future distribution, not
  a flaw in what was measured).
- Reference checked: `references/DeepRL_Monopoly` at `afd9205761317e196d77f679921c35fb04c7ab96` (submodule unchanged, read-only). No training or reference code changed for this decision.

## 2026-08-12 — Fix confirmed actor-relative property-owner state bug; re-scope prior MCTS/value/BUY-TRADE findings to the pre-fix state family

- Context: `references/DeepRL_Monopoly/monopoly_game_engine/state.py::build_state_vector`'s
  300-dim observation encodes every other per-player section (player
  position/cash/jail block, turn order, bankrupt, jail_turns,
  debt_creditor, auction leader/bidders, incoming-trade sender, ...) through
  the engine's canonical actor-relative order,
  `order = [agent_id] + [i for i in range(NUM_PLAYERS) if i != agent_id]`
  (agent first, then the other physical player ids in ascending order) -
  **except** the 28 property-owner 5-dim one-hot slices, which indexed
  directly by raw physical `prop.owner` (`owner_vec[prop.owner] = 1.0`,
  state.py line 154). For any `agent_id != 0` this mislabeled which
  actor-relative seat owns a property, corrupting exactly the
  `BUY_PROPERTY`/`ACCEPT_TRADE`-adjacent part of the state representation
  for three of every four decisions.
- Decision: **Fix it as a project-owned runtime monkeypatch**, not an edit
  to the pinned read-only submodule. `scripts/monopolyzero_common.py` adds
  `patch_actor_relative_owner_encoding()` (wired into the existing
  `ensure_reference_on_path()`, so every script/test that already calls it
  picks the fix up automatically, idempotent via a marker attribute so
  repeated calls never double-wrap): it replaces
  `monopoly_game_engine.state.build_state_vector` with a wrapper that calls
  the original unchanged for the full 300-dim vector, then re-maps only the
  28 owner one-hot slices from physical-id indexing to
  `actor_order(agent_id).index(owner)` indexing, in place. `STATE_DIM`,
  action ids, checkpoint tensor shapes, and every other feature are
  byte-for-byte untouched.
  - **No separate search/predict adapter was needed for MaxNPUCT, despite
    the task's provisional allowance for one.** `monopoly_game_engine/env.py`
    has exactly one call site for `build_state_vector`
    (`_get_state`, env.py line 1076), reached via a bare module-global name
    lookup resolved at call time, not import time - so patching
    `env`'s module dict once covers every consumer: `POLICY_ONLY`/
    `HYBRID_COMPAT` (`model.predict(env._get_state(seat), ...)`),
    `MaxNPUCT._evaluate` for **both** the root node (`search.py:265`) and
    every descendant node (`search.py:198,214`, both go through
    `env._get_state(actor)` again), and `play_local_game`'s
    `ReplayPosition.state=game.env._get_state(seat)` recording call - all
    without touching `references/DeepRL_Monopoly/monopoly_bench/search.py`.
    Because there is exactly one function that ever builds a state vector,
    and no separate post-hoc remap step exists anywhere downstream, there
    is no code path that could double-remap an already-corrected state
    (verified by `test_patch_is_idempotent_and_does_not_double_remap`).
  - **Test-design correction** (superseding the original approach in an
    earlier uncommitted draft of this fix): a broad
    "arbitrary-physical-seat-permutation invariance of the full 300-dim
    vector" test is **wrong** and was removed. Because `order` sorts the
    *other* physical ids ascending (not "same relative position after any
    relabeling"), relabeling individuals to different physical seats
    changes `order` itself for every per-player section, not just
    ownership - two seatings of "the same relative situation" are not
    expected to produce identical vectors in general. The correct,
    narrower invariant - **internal frame consistency** - is that within
    one `build_state_vector` call, property-owner slot `k` must name the
    same physical player as player-feature slot `k`, both under that
    call's own `actor_order(agent_id)`. `tests/test_monopolyzero_actor_relative_owner_fix.py`
    now proves this directly (including a non-circular cross-check against
    the independently-built player-features block, not just the owner
    section against itself), plus: exact owner-mapping for actor_id 0-3,
    byte-for-byte-unchanged actor_id=0 output, non-owner-coordinate
    invariance for actors 1-3, patch idempotency, `MaxNPUCT` root+descendant
    coverage (via direct `_evaluate` calls - see environment note below),
    and `play_local_game`/`ReplayPosition` recording correctness.
  - **Environment note**: two of the new tests would otherwise exercise
    `MaxNPUCT.choose_action`/`play_local_game` end-to-end, but
    `numpy.random.default_rng` is currently blocked on this machine by a
    local Application Control policy (`ImportError: DLL load failed while
    importing _mt19937: An Application Control policy has blocked this
    file`) - a pre-existing environment issue, confirmed unrelated to this
    fix by re-running the full `tests/` suite against clean HEAD
    `2f4fa0f` before making any change: the identical 27 tests fail there
    too (582 passed/27 failed on `2f4fa0f`; 589 passed/27 failed with this
    fix's 7 new tests added, zero newly-broken, zero newly-fixed). Those 27
    failures span files with no relation to this fix (`evaluation_protocol.py`,
    `colab_shard_runner.py`, `monopolyzero_native_train_candidate.py`,
    `monopolyzero_value_generalization_probe.py`,
    `monopolyzero_value_learnability_probe.py`,
    `monopolyzero_buy_trade_gradient_diagnostic.py`), all via the same
    `numpy.random`/`_mt19937` import failure. This is a machine-level
    issue outside this repo's and this task's scope (not a security
    setting this project should be changing) - **`tests/` is not fully
    green on this machine right now, but every failure is this one
    pre-existing, unrelated cause, and the fix's own 7 new tests are
    designed to avoid it** (calling `MaxNPUCT._evaluate` directly instead
    of `choose_action`, and a `kind="search"` stub instead of a real
    `MaxNPUCT`-backed policy for the replay-recorder test).
  - **Prior findings built on the buggy owner encoding must be re-scoped**,
    not silently carried forward as if measured on corrected semantics:
    - `026`'s 55,215-position replay
      (`artifacts/monopolyzero_native_train_candidate/buy_trade_learnability_v2/replay`)
      is **PRE-FIX DIAGNOSTIC ONLY, NOT eligible for corrected-model
      strength training** - both its stored states and its MCTS visit
      targets were generated under the old, physical-id owner encoding.
    - `017`/`018`'s PUCT-search-budget and PUCT-vs-`POLICY_ONLY` KILL
      conclusions are re-scoped to the pre-fix checkpoint/state family they
      were actually measured on - not yet re-established for
      corrected-state search.
    - `021`/`022`'s learned-value-probe KILL conclusion is likewise
      re-scoped to the pre-fix state family - the probe was trained and
      evaluated entirely on states whose owner one-hot was mislabeled for
      3 of every 4 seats.
    - `023`/`024`'s qualitative finding - that plain `POLICY_ONLY` never
      chooses `BUY_PROPERTY`/`ACCEPT_TRADE` due to the PPO-bootstrap's
      untrained action-head rows - is **kept**, since it is a property of
      the checkpoint's weights, not of the state encoding. Their exact
      win-rate magnitudes (e.g. `023`'s +21.25-point `HYBRID_COMPAT` vs.
      `POLICY_ONLY` gap) are **not** carried forward as post-fix evidence -
      those numbers were measured on pre-fix states and have not been
      re-verified against corrected ones.
  - This task explicitly excluded training, replay generation, and
    evaluation - re-running any of `017`/`018`/`021`/`022`/`023`/`024`/`026`
    against corrected-state semantics is future work, not decided here.
- Alternatives considered: editing `references/DeepRL_Monopoly/monopoly_game_engine/state.py`
  directly (rejected - the submodule must stay read-only per `CLAUDE.md`);
  a project-owned `search.py`/model-predict wrapper class instead of a
  `build_state_vector` monkeypatch (rejected as unnecessary after tracing
  every consumer to the single `env._get_state` call site - would have
  been strictly more code for the same coverage, and would have needed to
  be threaded through `MaxNPUCT` at both root and descendant evaluation
  separately); keeping the original broad permutation-invariance test and
  trying to make the implementation satisfy it (rejected - the test's own
  premise doesn't hold for this engine's actual `actor_order` semantics, so
  making it pass would have required either a different, non-canonical
  ordering scheme or a fragile test-only special case, neither justified
  by anything in this task).
- Reference checked: `references/DeepRL_Monopoly` at `afd9205761317e196d77f679921c35fb04c7ab96` (submodule unchanged, read-only).

## 2026-08-13 — Champion promoted: frozen 96-game A_lr1e-4 checkpoint replaces the 80-game champion

- Context: `034` (pre-registered before running, `docs/EXPERIMENTS.md`) tested
  032's frozen 96-game `A_lr1e-4` checkpoint (`candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt`)
  against the current 80-game champion (`candidate_ppo_80.pt`) plus the
  same 32/64/128-game opponent families `033` used, on 20 fresh PROMOTION
  seeds (`50020-50039`) never touched by `032` or `033`. The run completed
  unattended (320/320 games, 0 illegal actions, 0 crashes, 0 ASU modules
  loaded, exit code 0) and was recovered from a stdout capture rather than
  re-run - see `logs/experiments/034-challenger-gate-96-vs-champion-32-64-128.json`
  for full provenance (raw SHA-256 verified byte-identical between the
  original capture and its committed copy).
- Decision: **Promote the frozen 96-game `A_lr1e-4` checkpoint to project
  champion.** All three conditions of the pre-registered `034` decision
  rule passed without alteration after seeing results: vs_80 CI
  `[+5.0, +32.5]`pts (lower bound `> 0`), aggregate CI `[+18.125, +29.0625]`pts
  (lower bound `> 0`), no family's CI upper bound `<= 0` (worst case
  `+26.25`pts). Verdict: `CHALLENGER_PROMOTION_GO`.
  - **Canonical champion checkpoint**: `candidate_ppo_80_lr_ablation_A_lr1e-4_96.pt`
    — checkpoint SHA-256 `78585ed4e2400d024633ee2878d8b88243f7f4a9f498d9019a73e44b3a830f51`,
    actor SHA-256 `2bd1e9bad3d6e0100e033507f72f15b5088d8f8fb2f04dba1ae9d759b5a34a40`.
  - The previous 80-game checkpoint (`candidate_ppo_80.pt`) **remains
    preserved unchanged** as the former champion / reference baseline —
    no checkpoint file was renamed, copied, rebuilt, or committed by this
    decision. This is a champion-identity/decision update recorded against
    existing frozen artifacts only.
  - **This promotion does NOT claim official-competition generalization.**
    Diagnostic (non-overriding, per `034`'s own pre-registered rule)
    behavior metrics show the new champion's aggregate round-cap rate is
    89.375% (most wins are round-200 net-worth tiebreaks, not bankruptcy
    victories) and its `BUY_PROPERTY`/`ACCEPT_TRADE` rates when legal are
    both 0.0% — the same collapse already documented for the prior
    champion in `033`. The official competition engine/API and turn/round
    horizon remain `TBD` (`docs/RULES_SPEC.md`); today's real competition
    test, if it runs, is the next highest-priority evidence on whether
    this strength generalizes beyond this project's own self-play family.
- Alternatives considered: keep the 80-game champion (rejected — the
  pre-registered `034` rule's GO conditions all passed, and per this
  project's pre-registration discipline the rule is not overridden after
  seeing results); wait for further PROMOTION-seed (`50040-50049`) or
  `FINAL_BLIND` evidence before promoting (rejected for now — those pools
  were not needed for `034`'s own pre-registered question and remain
  fresh/unconsumed; `50040-50049` is PROMOTION, `FINAL_BLIND` is reserved
  exclusively for the final model-selection read, see
  `docs/EVALUATION_PROTOCOL.md`).
- Reference checked: `references/DeepRL_Monopoly` at `afd9205761317e196d77f679921c35fb04c7ab96` (submodule unchanged, read-only). No core algorithm file, `evaluation_protocol.py`'s statistics, checkpoint files, the `034` runner, or its pre-registered decision rule was modified to record this decision.
