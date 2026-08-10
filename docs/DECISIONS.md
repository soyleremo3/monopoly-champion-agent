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
