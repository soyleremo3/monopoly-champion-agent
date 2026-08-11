# Project Rules — Monopoly Champion Agent

These rules govern how work proceeds in this repo. They override default behavior.

## Development Process

- Work in small, incremental tasks. No large multi-feature commits.
- Do not make unverified assumptions about rules, environment, or APIs. If unknown, mark `TBD` and confirm before relying on it.
- Test every significant change before moving to the next step.
- Do not advance to the next phase/task until current tests pass.
- LLM components may only be adopted if an A/B test demonstrates measurable benefit over the baseline. No LLM-in-the-loop by default.
- Review game replays before making large agent-behavior changes.

## Git Workflow

- Commit at the smallest meaningful unit of change.
- Push immediately after every commit.
- Prefer many small commits/pushes over few large ones, within a single session.
- Keep the working tree clean at the end of a session.

## Reference Usage

- Check the `references/DeepRL_Monopoly` submodule before major design/algorithm decisions — it is the primary technical reference (not an authority on official competition rules).
- Before copying any code from external repos (including `DeepRL_Monopoly`), check license/compliance. Do not copy without verifying license compatibility.

## Rules Specification

- `docs/RULES_SPEC.md` holds only verified competition rules. Do not fabricate rules — leave unknowns as `TBD`.

## ASU Restrictions

*Corrected 2026-08-11 — the original version of this section banned using ASU
as a teacher/data source outright. That was wrong; corrected below.*

- ASU (`ASU_FROZEN_TEACHER`) may be used as a teacher/expert signal for
  training our own student model(s) — e.g. bootstrap supervision, policy/value
  targets, imitation data. This is permitted.
- ASU may never be, or become, this project's final/core competition agent.
  The competition entry must be our own trained model — not ASU itself
  (frozen or otherwise), and not a thin wrapper that just calls ASU at
  inference time.
- ASU may never be used as our Modal training/deployment model. Modal-hosted
  training and deployment run our own architecture; ASU stays a local,
  reference-only component, never the thing we train or deploy on Modal.
- ASU may always be used as a fixed evaluation opponent, independent of the
  above.
- `references/DeepRL_Monopoly` (the submodule) stays read-only research
  reference material at all times, regardless of how ASU is used — consume it
  via import/CLI at its pinned SHA, never edit it in place.
- Running `monopoly_bench train`, `collect-asu`, or `export-teacher` (or
  starting any MonopolyZero/teacher-bootstrap training) requires its own
  explicit decision logged in `docs/DECISIONS.md` first — permitted in
  principle now, but not assumed just because ASU-as-teacher is allowed.
