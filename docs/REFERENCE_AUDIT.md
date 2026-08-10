# Reference Audit — DeepRL_Monopoly

Audit of `references/DeepRL_Monopoly` (submodule, pinned SHA
`afd9205761317e196d77f679921c35fb04c7ab96`, see [REFERENCES.md](REFERENCES.md)).
Goal: prove the reference runs locally and produce a reproducible baseline
match. No training or agent code was written for this project as part of
this audit.

## What was reviewed

- `README.md` — high-level pointers only (CFR/PPO/DDQN/SLM sections have no
  content yet in this repo, just reading-list links).
- `PPO_PLUS_RULES.md` — the `ppo-plus-v2` ruleset: 4 players, standard 40-space
  board, $1,500 start, $200 Go, taxes, jail, auctions, mortgages, houses/hotels
  with a finite 32-house/12-hotel bank. Explicitly **not** official Monopoly:
  no Chance/Community Chest card effects, no even-building enforcement, deeds
  sellable back to the bank at mortgage value, 200-round cap with net-worth
  tie-break. Do not treat this as our competition's ruleset — see
  [RULES_SPEC.md](RULES_SPEC.md), which stays `TBD` until verified separately.
- `TRAINING_RESULTS.md` — prior measured runs (PPO 2,000 games ≈31 min on a
  laptop GPU, weak win rate; CFR one full game ≈25 min). Confirms training is
  expensive and out of scope for this pass, consistent with the "no training"
  instruction.
- `tests/` (9 files, repo root) and `monopoly_bench/tests/` (4 files) — see
  results below.
- `tools/` — `train_and_save.py` (training), `play_game.py` (requires a model
  checkpoint, no fixed-only mode), `generate_stats.py`. None were run: the
  first two need training or a checkpoint we don't have.
- `monopoly_bench/` — an isolated MonopolyZero-style self-play/search pipeline
  (`arena.py`, `search.py`, `training.py`, `ladder.py`, ...). Not exercised
  beyond its test suite; its `train`/`gate`/`evaluate` CLI subcommands all
  assume a bootstrap PPO checkpoint we don't have.
- `ASU_FROZEN_TEACHER/` — deterministic, checkpoint-free heuristic policies
  (`asu-value-v1`, `asu-rollout-v1`) plus a seat-balanced `evaluate` CLI that
  also runs the six scripted `fixed-a`..`fixed-f` agents with **no checkpoint
  required**. This is the "existing runnable path" used for the baseline
  match below.

## Minimum dependencies

No `requirements.txt` / `pyproject.toml` ships in the reference repo.
Determined by reading imports directly:

- `numpy`
- `torch` (CPU-only build; imported by `monopoly_game_engine/__init__.py`
  even for scripted-only games, because that package eagerly imports the PPO
  and DDQN agent modules — so torch is unavoidable even though the fixed
  agents themselves never call it)
- `pytest` (to run the test suite only)

Installed in an isolated venv (`C:/mpvenv`, kept out of both repos), CPU-only
torch from `https://download.pytorch.org/whl/cpu` — no CUDA/GPU packages:

```
torch 2.13.0+cpu (torch.cuda.is_available() == False)
numpy 2.5.2
pytest 9.1.1
```

Nothing was installed into the submodule or committed to this repo; the venv
is local machine state only.

## Test results

Command (from `references/DeepRL_Monopoly`):

```bash
python -m pytest tests/ -q
python -m pytest monopoly_bench/tests -q
```

**`tests/` — 95 passed, 2 failed, 2 subtests passed (79.95s)**

Both failures are in `tests/test_gemma4_notebook.py` and are unrelated to the
game engine, ASU teacher, or baseline path used below:

1. `test_generic_hardware_reference_is_unchanged` — a SHA-256 pin on a
   notebook file has drifted from what the test expects (`d013c76b...` vs
   expected `a7e01e9d...`). Pre-existing content drift in the reference repo,
   not something this audit changed.
2. `test_launcher_validates_private_rclone_inputs` — asserts a config file's
   POSIX permission bits (`chmod 0o600`) are honored. Windows/NTFS does not
   enforce POSIX group/other permission bits the same way, so the guard
   condition trips. Environment-specific (Windows vs. POSIX), not a logic bug
   in the paths this audit depends on.

**`monopoly_bench/tests/` — 22 passed, 9 failed (2.50s)**

All 9 failures are `FileNotFoundError` on
`artifacts/ppo_plus/ppo_hybrid_2000_v2.pt` — a PPO checkpoint that only exists
after running training, which this audit deliberately did not do (`artifacts/`
is git-ignored upstream and was never present). Expected given the "no
training" constraint, not a defect in `monopoly_bench` itself.

## Baseline match — see [BASELINE.md](BASELINE.md)

Ran via `ASU_FROZEN_TEACHER.evaluate.evaluate_lineup` (imported from the
submodule, not copied) through [scripts/run_baseline_match.py](../scripts/run_baseline_match.py).

## Critical issue found

**Same-seed runs are not reproducible across separate process launches
unless `PYTHONHASHSEED` is pinned.**

Running `scripts/run_baseline_match.py --seed 42` twice, in two separate
`python` process invocations, produced **different game outcomes** for one of
the four paired-seat games in the block (different winner, round count —
63 vs. 200 — and decision count — 2,288 vs. 5,301), even though the other
three games in the block matched exactly. Root cause narrowed down by
inspecting `ASU_FROZEN_TEACHER/core.py`:

- Per-game dice/setup randomness *is* properly isolated: `_new_seeded_game`
  seeds the global `random` module with the caller's seed, builds the env, and
  `_PrivateGame.step()` swaps the process's global `random` state in and out
  around each `env.step()` call. That part is correctly deterministic.
- The remaining source is very likely Python's per-process hash
  randomization (`PYTHONHASHSEED`, randomized by default since Python 3.3)
  affecting the iteration order of some `set`/hash-keyed structure reachable
  from action-legality or decision logic. `agents_fixed.py` itself contains
  no direct `random.*` calls, so the divergence isn't coming from the
  scripted agents' own decision code.
- **Confirmed by experiment**: with `PYTHONHASHSEED=0` exported before both
  runs, two independent process launches with `--seed 42` produced
  byte-identical output except for wall-clock `elapsed_seconds` fields. See
  [BASELINE.md](BASELINE.md) for the diff.

**Practical implication for this project**: any reproducibility claim about
this reference engine (or anything built on top of it) must pin
`PYTHONHASHSEED` in addition to the game seed. This is a reference-repo
property, not something we patched — we did not modify submodule code.
Worth re-checking against a newer commit of `DeepRL_Monopoly` before relying
on seeded reproducibility for real experiments later.

## License / copying

No `LICENSE` file in `Darkosxl/DeepRL_Monopoly` (GitHub API reports
`"license": null`) — all rights reserved by default. Nothing from the
reference repo was copied into this repo; it is consumed only as a git
submodule and imported at runtime from its checked-out path
(`references/DeepRL_Monopoly`) by `scripts/run_baseline_match.py`.
