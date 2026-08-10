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

- Never copy, imitate, distill, label from, or reproduce ASU (`ASU_FROZEN_TEACHER`) outputs. No training signal, dataset, label, or policy weight may derive from ASU decisions, values, or rollouts.
- ASU may only be used as an evaluation opponent (a fixed benchmark to play against), never as a teacher, data source, or training target.
- Do not modify or advance `references/DeepRL_Monopoly` (the submodule). It is read-only research reference material; consume it via import/CLI at its pinned SHA, never edit it in place.
- Do not use `monopoly_bench train`, `monopoly_bench collect-asu`, or `monopoly_bench export-teacher`. These pipelines bootstrap from or distill ASU/PPO artifacts and are out of scope for this project's own agent development.
