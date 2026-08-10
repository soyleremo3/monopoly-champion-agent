# Baseline Match

First reproducible baseline: a training-free, checkpoint-free 4-player match
on the `references/DeepRL_Monopoly` `ppo-plus-v2` engine, using only the
scripted fixed agents (fastest available path — no ASU lookahead, no neural
inference).

## Setup

- Dependencies: isolated venv at `C:/mpvenv`, `numpy`, `torch 2.13.0+cpu`
  (no CUDA), `pytest`. See [REFERENCE_AUDIT.md](REFERENCE_AUDIT.md).
- Runner: [scripts/run_baseline_match.py](../scripts/run_baseline_match.py) —
  a thin wrapper that adds `references/DeepRL_Monopoly` to `sys.path` and
  calls `ASU_FROZEN_TEACHER.evaluate.evaluate_lineup` (existing runnable path
  from the reference repo; no game logic copied).
- Lineup: `fixed-d` (TheBuilder) vs. `fixed-a` (TheHoarder), `fixed-b`
  (TheDealMaker), `fixed-c` (TheGambler) — all scripted, no checkpoint needed.
- The evaluate CLI runs a **seat-balanced paired block**: for one seed it
  plays 4 games, rotating which physical seat the "focus" agent (`fixed-d`)
  occupies. That block is what the command below produces.

## Command

```bash
PYTHONHASHSEED=0 python scripts/run_baseline_match.py --seed 42 --output docs/baseline_runs/baseline_seed42_run1.json
```

`PYTHONHASHSEED=0` is required for reproducibility — see the critical finding
in [REFERENCE_AUDIT.md](REFERENCE_AUDIT.md). Without it, one of the four
games in the block can diverge between runs.

Raw output saved at
[docs/baseline_runs/baseline_seed42_run1.json](baseline_runs/baseline_seed42_run1.json)
and
[docs/baseline_runs/baseline_seed42_run2.json](baseline_runs/baseline_seed42_run2.json).

## Result (seed 42, 4-game paired block)

| Game | Focus seat | Seats (0,1,2,3) | Winner | Rounds | Decisions | Game elapsed (s) |
|---|---|---|---|---|---|---|
| 1 | 0 | fixed-d, fixed-a, fixed-b, fixed-c | seat 0 (fixed-d) | 40 | 1,484 | 0.190 |
| 2 | 1 | fixed-a, fixed-d, fixed-b, fixed-c | seat 1 (fixed-d) | 92 | 2,498 | 0.315 |
| 3 | 2 | fixed-a, fixed-b, fixed-d, fixed-c | seat 1 (fixed-b) | 60 | 2,080 | 0.263 |
| 4 | 3 | fixed-a, fixed-b, fixed-c, fixed-d | seat 3 (fixed-d) | 200 (round cap) | 5,301 | 0.717 |

- No crashes. No illegal actions (the evaluator raises `RuntimeError` on any
  illegal action; none were raised — see `_run_game` in
  `ASU_FROZEN_TEACHER/evaluate.py`).
- `truncations: 0` for all 4 games (this field tracks hitting the evaluator's
  20,000-decision safety cap, not the engine's round cap). Game 4 hit the
  engine's 200-round cap with 2 players still active (seats 2 and 3; seats 0
  and 1 were already bankrupt); per `MonopolyEnv.winner()` the engine still
  returns a decided winner at the round cap via a net-worth tie-break
  (seat 3: 30,236.5 vs. seat 2: 17,903.5), it does not require elimination
  down to one player.
- `scripted_compatibility_fallbacks: 62` total across the block — expected,
  documented behavior (fixed agents sometimes return `END_TURN` when only
  liquidation/trade actions are legal; the evaluator's compatibility fallback
  picks the first legal action instead and counts it). Not an error.
- fixed-d (TheBuilder) won 3 of 4 seat rotations in this single seed —  not a
  performance claim, one seed is not a statistically meaningful sample.

## Runtime

- Full block (4 games), in-loop time: **~1.49–1.53 s** (`elapsed_seconds` in
  the JSON).
- Full process wall time (`python scripts/run_baseline_match.py`, includes
  interpreter startup + torch import): **~2.7 s** (`time -p`: real 2.68s /
  2.72s across the two runs below).

## Determinism check

Ran the exact command above twice, same seed (42), two separate process
launches, `PYTHONHASHSEED=0` exported both times:

```bash
export PYTHONHASHSEED=0
python scripts/run_baseline_match.py --seed 42 --output docs/baseline_runs/baseline_seed42_run1.json
python scripts/run_baseline_match.py --seed 42 --output docs/baseline_runs/baseline_seed42_run2.json
diff docs/baseline_runs/baseline_seed42_run1.json docs/baseline_runs/baseline_seed42_run2.json
```

Result: **identical** except for the wall-clock `elapsed_seconds` fields
(timing noise, expected). Every winner, round count, decision count, net
worth, and fallback count matched exactly between the two runs.

Without `PYTHONHASHSEED` pinned, the same two-run comparison **failed**: game
4 in the block diverged (winner seat 3 vs. no single winner / round 200 cap
differently resolved, 2,288 vs. 5,301 decisions). See
[REFERENCE_AUDIT.md](REFERENCE_AUDIT.md#critical-issue-found) for the root
cause.

## Re-running

```bash
python scripts/run_baseline_match.py --seed 42 --output docs/baseline_runs/<name>.json
```

Optional flags: `--focus`, `--opponents` (3 IDs), see `--help`. Always export
`PYTHONHASHSEED=0` first if you need the result to reproduce across separate
runs.
