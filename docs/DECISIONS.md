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
