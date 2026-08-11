# Project Rules — Monopoly Champion Agent

These rules govern how work proceeds in this repo. They override default behavior.

## Development Process

- Work in small, incremental tasks. No large multi-feature commits.
- Do not make unverified assumptions about rules, environment, or APIs. If unknown, mark `TBD` and confirm before relying on it.
- Test every significant change before moving to the next step.
- Do not advance to the next phase/task until current tests pass.
- LLM components may only be adopted if an A/B test demonstrates measurable benefit over the baseline. No LLM-in-the-loop by default.
- Review game replays before making large agent-behavior changes.
- Primary objective is overall/generalizable win-rate, not performance against any single opponent. A hybrid approach is allowed: our trained model should make the large majority of ordinary decisions; explicit deterministic algorithms may only be used for rare, well-defined edge cases, not as the primary decision-maker. See `docs/DECISIONS.md` for the full competition-strategy record.

## Git Workflow

- Commit at the smallest meaningful unit of change.
- Push immediately after every commit.
- Prefer many small commits/pushes over few large ones, within a single session.
- Keep the working tree clean at the end of a session.

## Experiment Logging (mandatory)

*Corrected 2026-08-12: `code_commit_sha`'s meaning was originally wrong (see
below) — fixed here and in `logs/experiments/schema.json` and `README.md`.*

- Every training run, benchmark, reproducibility check, A/B test, or
  replay-derived result **must** get a `logs/experiments/NNN-slug.json`
  entry conforming to `logs/experiments/schema.json` — see
  `logs/experiments/README.md` for the standard. This is required, not
  optional, starting 2026-08-11.
- **The working tree must be `git status --porcelain`-clean before a
  meaningful experiment starts.** Commit and push any pending code changes
  first, then run the experiment against that clean HEAD.
- `code_commit_sha` in the log entry is that clean HEAD SHA (`git rev-parse
  HEAD`) captured *at the moment the experiment ran* — **not** the commit
  that later records the results in `docs/EXPERIMENTS.md`/the log itself.
  Which commit added a given log file is already discoverable via `git log`/
  `git blame` on that file, so it is not duplicated as a separate field.
- Write the JSON log entry as part of the same commit that records the
  experiment in `docs/EXPERIMENTS.md` — not a separate, later, or skipped
  step.
- Unknown or unmeasured values are `null` in the log — never guessed or
  estimated into a field that should hold a measured number. This includes
  `code_commit_sha` itself when the clean-tree discipline wasn't followed.
- Never commit checkpoints, replay buffers, or other large/binary artifacts
  into `logs/experiments/` (or anywhere else in this repo) — reference them
  by SHA-256 and local path instead. Small raw stdout/stderr `.txt` logs are
  fine and belong in `logs/experiments/raw/`.
- `tests/test_experiment_logs_schema.py` validates every entry against the
  schema — it must pass before a commit that adds or changes a log entry.

## Reference Usage

- Check the `references/DeepRL_Monopoly` submodule before major design/algorithm decisions — it is the primary technical reference (not an authority on official competition rules).
- Before copying any code from external repos (including `DeepRL_Monopoly`), check license/compliance. Do not copy without verifying license compatibility.

## Rules Specification

- `docs/RULES_SPEC.md` holds only verified competition rules. Do not fabricate rules — leave unknowns as `TBD`.

## ASU Restrictions

*Re-locked 2026-08-11 (later same day) per official competition guidance —
supersedes the "Corrected 2026-08-11" version of this section, which allowed
ASU as a teacher/expert signal. That is revoked; ASU-as-teacher is banned
again below, this time on the record as a competition-rules requirement, not
a project preference.*

- ASU (`ASU_FROZEN_TEACHER`) may **only** be used as an evaluation opponent —
  a fixed benchmark to play against for measuring our own model.
- ASU output/action/value/rollout data may **never** be a training label for
  any model we train. This includes value targets, policy targets, and
  reward shaping derived from ASU.
- ASU imitation, distillation, teacher-bootstrap, and output-cloning are
  **banned outright** — no exceptions, regardless of fraction, decay
  schedule, or "just for bootstrap" framing.
- Do not use `monopoly_bench collect-asu`, or any ASU-guided training path
  (`monopoly_bench train`'s default bootstrap, `bootstrap_asu_expert`,
  `expert_train_step` — see `docs/REFERENCE_AUDIT.md`'s MonopolyZero section
  for exactly which code paths these are). Self-play population generation
  that structurally seats ASU as an opponent (`monopoly_bench`'s
  `population_jobs`, which hardcodes ASU into part of every generation with
  no disable flag) also counts as an ASU-guided path and must not be used
  as-is; an ASU-independent self-play setup must exclude ASU from the
  opponent pool entirely, built from the reference's lower-level
  ASU-independent primitives instead (see `docs/REFERENCE_AUDIT.md`).
- ASU may never be used as a runtime fallback or as the core/final
  competition agent, under any circumstance.
- ASU may never be used as our Modal training/deployment model. Modal-hosted
  training and deployment run our own architecture only.
- `references/DeepRL_Monopoly` (the submodule) stays read-only research
  reference material at all times — consume it via import/CLI at its pinned
  SHA, never edit it in place.
- ASU is, and remains, an important **evaluation opponent and anti-ASU
  robustness benchmark** — see the competition-strategy decision in
  `docs/DECISIONS.md` for how this fits the overall win-rate goal.
