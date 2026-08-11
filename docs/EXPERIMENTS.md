# Experiments Log

Log of experiments run on the agent. Append entries chronologically, most recent last.

## Format

```
## YYYY-MM-DD — <short title>

- Hypothesis:
- Setup:
- Result:
- Conclusion / next step:
```

## 2026-08-10 — First compliant training smoke: 20-game hybrid DDQN (CPU)

- Hypothesis: the reference's own `tools/train_and_save.py` DDQN path can train
  a small, reproducible checkpoint using only environment reward and fixed
  opponents (no ASU coupling), and that checkpoint can be loaded back for
  inference — proving the training→checkpoint→inference loop works before any
  of our own agent code exists.
- ASU compliance check (task 5, by code inspection, not just claim):
  `tools/train_and_save.py` → `monopoly_game_engine.train_ddqn` →
  `monopoly_game_engine/train.py::train()`/`run_episode()`. Opponents are
  hardcoded to `FPAgentA, FPAgentB, FPAgentC` (`TheHoarder`, `TheDealMaker`,
  `TheGambler` — scripted fixed policies). Reward is
  `potential_delta()` = clipped `gamma * env._compute_reward(agent_pid) - Phi(prev)`
  (pure environment/net-worth potential shaping) plus a terminal win/loss
  bonus from `env.winner()`. Grepped the full call chain for any ASU
  reference:
  ```bash
  grep -rniE "asu" tools/train_and_save.py training_guard.py \
    monopoly_game_engine/__init__.py monopoly_game_engine/train.py \
    monopoly_game_engine/agent_ddqn.py monopoly_game_engine/networks.py \
    monopoly_game_engine/agents_fixed.py monopoly_game_engine/env.py
  ```
  Zero matches. Confirmed: no ASU import, data, or coupling anywhere in the
  DDQN training path.

### Environment

- Python 3.12.10, `torch 2.13.0+cpu` (`torch.cuda.is_available() == False`),
  `numpy 2.5.2`, `psutil 7.2.2` (newly added — see failures below).
- Isolated venv at `C:/mpvenv`, same one used for the engine smoke test.

### Command 1 (failed) — training

```bash
PYTHONHASHSEED=0 python references/DeepRL_Monopoly/tools/train_and_save.py \
  --algo ddqn --games 20 --device cpu --seed 42 --checkpoint-every 10 \
  --out artifacts/training_smoke/ddqn_hybrid_20_v2.pt
```

**Exact error:**

```
Memory watchdog stopped training: Process RSS is unavailable
Emergency checkpoint: artifacts\training_smoke\ddqn_hybrid_20_v2_emergency.pt
Training stopped early in 1.2s
Games completed: 0/20
```

Root cause: `references/DeepRL_Monopoly/training_guard.py:36-48`,
`MemoryWatchdog.rss_bytes()` requires `psutil`; without it, it falls back to
`/proc/self/statm` (Linux-only) then `resource.getrusage` (POSIX-only,
`resource` is `None` on Windows per that file's own import guard). Neither
fallback works on Windows without `psutil`, so it raises
`MemoryLimitReached("Process RSS is unavailable")` on the first watchdog
check, before game 1 starts. Our original dependency scoping (numpy, torch,
pytest) covered the evaluation path but not this training-only dependency.
**Fix**: installed `psutil` (small, pure-Python-wheel, no GPU, does not touch
the submodule) — `pip install psutil` in the venv.

### Command 2 (failed) — training, retried with psutil installed

Same command as above. **Exact error:**

```
UnicodeEncodeError: 'charmap' codec can't encode character '\u03b5' in position 29: character maps to <undefined>
```
(raised from `monopoly_game_engine/train.py:393-398`, printing the `ε=` epsilon
progress line; traceback bottoms out in
`Lib\encodings\cp1252.py`.)

Root cause: Windows console defaults to the `cp1252` codepage for stdout,
which cannot encode `ε` (U+03B5). **Fix**: set `PYTHONIOENCODING=utf-8`
before running (environment variable only, no submodule edit).

### Command 3 (succeeded) — training

```bash
PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python references/DeepRL_Monopoly/tools/train_and_save.py \
  --algo ddqn --games 20 --device cpu --seed 42 --checkpoint-every 10 \
  --out artifacts/training_smoke/ddqn_hybrid_20_v2.pt
```

- Wall time: **175.19s** real (`time -p`); script-reported "Training complete
  in 173.5s", mean **8.675s/game**.
- Games completed: **20/20**.
- Peak process RSS: **0.35 GiB**.
- Win rate: **0.0%** across all 20 games (epsilon 1.000 → 0.990 — still
  near-random exploration this early; expected and unremarkable for a
  20-game smoke, not a strength result).
- Reward per game (potential-shaped + win/loss bonus): ranged **-11.70 to
  -10.29** across the 20 games (see
  `artifacts/training_smoke/ddqn_hybrid_20_v2_history.json`, not committed —
  gitignored under `artifacts/`).
- No separate loss/TD-error is recorded by this reference's history schema —
  only `win_rates`, `rewards`, and trade/property counters are logged per
  `log_every` interval. Noting this as a schema gap rather than fabricating a
  loss figure; recovering it would require instrumenting
  `monopoly_game_engine/train.py`, which we won't do (no submodule edits).
- Checkpoint: `artifacts/training_smoke/ddqn_hybrid_20_v2.pt`,
  58,859,919 bytes (56.1 MiB). Not committed (gitignored).
- **Checkpoint SHA-256**:
  `47f3c177c1ae42449b8b3d1a34c253204329e15240a6345d78412f4f900716f4`
- Crashes: 2 (both environment/dependency issues above, both fixed without
  touching the submodule); 0 illegal actions (the training loop raises
  `ValueError` on an illegal learning-agent action — see
  `monopoly_game_engine/train.py:157-160` — none raised).

### Command 4 (succeeded) — seeded 4-player inference

Reused `scripts/run_baseline_match.py` (already PYTHONHASHSEED-guarded) with
the new checkpoint as focus and fixed agents as the only opponents (ASU not
involved):

```bash
PYTHONHASHSEED=0 python scripts/run_baseline_match.py \
  --focus "ddqn:<absolute path to>artifacts/training_smoke/ddqn_hybrid_20_v2.pt" \
  --opponents fixed-a fixed-b fixed-c --seed 42 \
  --output docs/baseline_runs/inference_ddqn_smoke.json
```

- Wall time: **9.18s** real.
- Checkpoint loaded successfully: `AgentFactory._torch_metadata` validated
  `format_version=3`, `ruleset=ppo-plus-v2`, `state_dim=300`,
  `action_dim=2958` before use. `checkpoint_hashes` in the output JSON
  independently reports SHA-256
  `47f3c177c1ae42449b8b3d1a34c253204329e15240a6345d78412f4f900716f4` —
  matches the value computed directly against the `.pt` file above.
- This CLI produces a seat-rotated 4-game block per seed (same pattern as the
  engine smoke). Primary reported game (focus_seat 0, DDQN in seat 0 vs.
  fixed-a/b/c in seats 1-3, seed 42): **winner seat 2 (fixed-b)**, 200 rounds
  (round cap, net-worth tie-break), **9,703 decisions**, elapsed 3.89s, 0
  illegal actions, 0 scripted-compatibility fallbacks in this specific game.
- Full 4-game block: DDQN won **0/4** seat rotations (0.0% — expected for a
  near-random, 20-game checkpoint; not a strength claim). Total
  `scripted_compatibility_fallbacks` across the block: **15 — all 15 from the
  fixed opponents (fixed-a/b/c); the DDQN checkpoint had 0 fallbacks in every
  game**. Correction: an earlier draft of this entry reported the 15 as an
  undifferentiated block total without attributing them; verified by reading
  `scripted_compatibility_fallbacks` per seat in
  `docs/baseline_runs/inference_ddqn_smoke.json` against each game's
  `policies` list — the DDQN checkpoint's own seat is `0` in all 4 games
  (game 1: `[0,0,0,0]`, seat 0 = ddqn; game 2: `[1,0,5,0]`, seat 1 = ddqn;
  game 3: `[1,0,0,0]`, seat 2 = ddqn; game 4: `[1,7,0,0]`, seat 3 = ddqn).
  The compatibility fallback only fires for the scripted-agent adapter (see
  `_ScriptedAdapter` in `ASU_FROZEN_TEACHER/evaluate.py`); the DDQN/neural
  adapter (`_NeuralAdapter`) always plays a masked argmax over legal actions
  and has no fallback path, so a 0-fallback DDQN result is expected by
  construction, not just an empirical observation. `truncations: 0`. No
  crashes, no illegal actions (the evaluator's `RuntimeError` guard on
  illegal actions was never triggered).

### Conclusion / next step

The compliant training→checkpoint→inference loop works end to end on CPU,
using only environment reward and fixed opponents, with zero ASU coupling
(confirmed by code inspection and grep, not just by not calling it). Two
environment/dependency gaps (`psutil`, console UTF-8 encoding) were found and
fixed without modifying the submodule; both are now implicit prerequisites
for running this reference's training path on Windows and should be added to
setup notes before any larger training run. The checkpoint itself is not a
meaningful policy (20 games, ~99% epsilon, 0% win rate) — this smoke proves
plumbing, not agent quality. No agent/training code of our own was written;
everything here calls the reference's existing entry points.

## 2026-08-11 — DDQN training reproducibility check, 500-game milestone, paired evaluation

- Hypothesis: (1) the reference's DDQN trainer is bit-exact reproducible
  given the same seed, so it's safe to build on; (2) resuming the 20-game
  checkpoint to 500 games produces a checkpoint that is measurably different
  from (and ideally stronger than) the 20-game one on held-out seeds.

### Reproducibility check (task 4)

Ran a second, independent 20-game DDQN training with identical parameters to
the original smoke, to a separate output path:

```bash
PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python references/DeepRL_Monopoly/tools/train_and_save.py \
  --algo ddqn --games 20 --device cpu --seed 42 --checkpoint-every 10 \
  --min-available-gib 1 \
  --out artifacts/training_smoke/ddqn_hybrid_20_v2_repro.pt
```

`--min-available-gib 1` (down from the 2 GiB default) was added only because
the memory watchdog interrupted two attempts on this machine at the default
threshold while ambient system RAM (Chrome, other apps — not this training
process, whose own peak RSS is ~0.35 GiB) was low; see the two failed
attempts and the user's explicit go-ahead to lower this specific flag,
earlier in this session. No other training parameter was changed.

Compared with a new tool,
[scripts/compare_ddqn_checkpoints.py](../scripts/compare_ddqn_checkpoints.py)
(deep-compares every field `DDQNAgent.save()` writes — metadata,
online/target network weights, optimizer state, replay buffer — via
`torch.equal`, i.e. bit-exact, not a tolerance; and separately compares only
the deterministic fields of the two `*_history.json` files, explicitly
ignoring `elapsed_seconds`, `seconds_per_game`, `peak_rss_gib`,
`peak_cuda_gib`, `training_segments`):

```bash
python scripts/compare_ddqn_checkpoints.py \
  artifacts/training_smoke/ddqn_hybrid_20_v2.pt artifacts/training_smoke/ddqn_hybrid_20_v2_repro.pt \
  --history-a artifacts/training_smoke/ddqn_hybrid_20_v2_history.json \
  --history-b artifacts/training_smoke/ddqn_hybrid_20_v2_repro_history.json
```

**Result: `MATCH`.** Every metadata field, both network state dicts, the
optimizer state, the replay buffer, `epsilon`, `step_count`, `games_trained`,
and every deterministic history field (`win_rates`, `rewards`, trade/property
counters, games-completed counters) were bit-identical between the two
independent runs.

**Noteworthy non-finding**: the two `.pt` files' raw SHA-256 differ
(`47f3c177...` vs `4a630a14...`) despite the comparator reporting a full
match. This is expected `torch.save` behavior — its pickle/zip container can
differ in non-content metadata (e.g. storage-key ordering) between two calls
even when every tensor and Python value it encodes is identical — so file
hash is not a valid reproducibility check for `.pt` files; the field-by-field
tensor/value comparison above is. Recording this so a future bare hash
mismatch on a "should be reproducible" checkpoint isn't mistaken for a bug.

**Conclusion: reproducibility passes.** Proceeded to the 500-game run per
plan. (Both `psutil` and `PYTHONIOENCODING=utf-8`, from the original smoke,
were still required and still used.)

### Preserving the 20-game checkpoint as a milestone (task 5)

Before resuming the main checkpoint past 20 games, copied it (and its
history) to a separate path, both gitignored, neither committed:

```
artifacts/training_smoke/milestones/ddqn_hybrid_20_v2_milestone.pt
artifacts/training_smoke/milestones/ddqn_hybrid_20_v2_milestone_history.json
```

Copy verified byte-for-byte identical to the original (SHA-256
`47f3c177c1ae42449b8b3d1a34c253204329e15240a6345d78412f4f900716f4` both
before and after copying).

### 500-game resume (task 5)

```bash
PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python references/DeepRL_Monopoly/tools/train_and_save.py \
  --algo ddqn --games 500 --device cpu --seed 42 --checkpoint-every 100 --resume \
  --out artifacts/training_smoke/ddqn_hybrid_20_v2.pt
```

Ran in the background; no parameters changed from what was specified going
in. RAM was rechecked before launch (3.49 GiB available, comfortably above
the default 2 GiB floor) and the run completed without any watchdog
interruption.

- Wall time: **6681.37s** real (**111.4 min**); script-reported "Training
  complete in 6679.6s", mean **13.916s/game** for the 480 games this run
  actually played (resume only trains the remaining games toward the 500
  total, matching `games_completed_this_run: 480`,
  `games_completed: 500`, `resumed_from_games: 0` — the merged history
  correctly folds both segments; `training_segments` has 2 entries, one per
  training call).
- Games completed: **500/500**. No crash, no interruption, no illegal
  action (an illegal DDQN action raises `ValueError` and aborts the process —
  exit code was 0).
- Peak process RSS this run: **0.36 GiB**.
- Win rate stayed **0.0%** in every logged 10-game window through game 500;
  epsilon decayed from ~0.99 to **0.7787520933134615** (still high — 500
  games is early-stage exploration for this trainer's 0.9995/game decay
  toward a 0.05 floor).
- Checkpoint: `artifacts/training_smoke/ddqn_hybrid_20_v2.pt`,
  62,151,119 bytes. Not committed (gitignored).
- **Checkpoint SHA-256**:
  `fee4f2952461364bf9fab6f1d545ea223f3634f131f25e55d5fc811e6c953a72`
- `games_trained: 500`, `step_count: 211097` (read directly from the
  checkpoint payload).

### Greedy paired evaluation on held-out seeds 10000-10009 (tasks 6-7)

Held-out seeds were never used in training (training used
`seed=42`-derived per-game seeds only). Evaluated both checkpoints with the
same seat-rotated, seed-paired protocol as the earlier engine/inference
smokes, now via the multi-seed range support added to
`scripts/run_baseline_match.py` in this session:

```bash
PYTHONHASHSEED=0 python scripts/run_baseline_match.py \
  --focus "ddqn:<path to checkpoint>" --opponents fixed-a fixed-b fixed-c \
  --seed 10000-10009 \
  --output docs/baseline_runs/eval_<20game|500game>_seeds10000-10009.json
```

10 seeds × 4 seat rotations = 40 games per checkpoint. "Greedy" here means
what the evaluator's `_NeuralAdapter` already does — masked argmax over
Q-values, no epsilon sampling — not an extra setting we had to add.

| | 20-game checkpoint | 500-game checkpoint |
|---|---|---|
| Wall time | 48.31s (46.75s in-loop) | 57.13s (55.48s in-loop) |
| DDQN win rate | 0/40 = **0.0%** | 1/40 = **2.5%** |
| DDQN Wilson 95% CI | **[0.0%, 8.76%]** | **[0.44%, 12.88%]** |
| DDQN mean net worth | 363.9 | 608.1 |
| fixed-a win rate | 0/40 = 0.0%, CI [0.0%, 8.76%] | 1/40 = 2.5%, CI [0.44%, 12.88%] |
| fixed-b win rate | 25/40 = 62.5%, CI [47.03%, 75.78%] | 25/40 = 62.5%, CI [47.03%, 75.78%] |
| fixed-c win rate | 15/40 = 37.5%, CI [24.22%, 52.97%] | 13/40 = 32.5%, CI [20.08%, 47.98%] |
| Round-cap rate (200 rounds) | 13/40 games | 13/40 games |
| Fallbacks by policy | ddqn: **0**, fixed-a: 32, fixed-b: 114, fixed-c: 68 (total 214) | ddqn: **0**, fixed-a: 28, fixed-b: 142, fixed-c: 97 (total 267) |
| Truncations / crashes / illegal actions | 0 / 0 / 0 | 0 / 0 / 0 |

DDQN fallbacks are 0 in both evaluations **by construction**, not
coincidence: `_NeuralAdapter.choose_action` in
`ASU_FROZEN_TEACHER/evaluate.py` always masks illegal actions to `-inf`
before `argmax`, so it can never select (or need a fallback for) an illegal
action — unlike `_ScriptedAdapter`, which wraps a fixed policy that can
propose an illegal action and needs the compatibility fallback. This matches
and generalizes the correction made earlier in this log for the first
inference smoke.

**No improvement claim**: the 500-game checkpoint won 1/40 vs. the 20-game
checkpoint's 0/40. The two Wilson 95% intervals ([0%, 8.76%] and
[0.44%, 12.88%]) overlap almost completely. One additional win out of 40
paired games is not a statistically supported improvement — it is
indistinguishable from noise at this sample size. **Conclusion: do not claim
the 500-game checkpoint is stronger than the 20-game checkpoint** based on
this evaluation. Both remain far below `fixed-b`/`fixed-c` (which are
unaffected fixed policies, so their similar win rates across both
evaluations are an internal consistency check, not a finding).

### Conclusion / next step

Reproducibility is solid (bit-exact given the same seed and environment),
so it's safe to keep extending this checkpoint's training. 500 games at
`epsilon≈0.78` is still too early in this trainer's decay schedule to expect
a measurable skill signal against fixed opponents — the paired evaluation
correctly shows no statistically supported difference from the 20-game
checkpoint, and that absence of a claim is itself the correct, honest
result, not a failure of the experiment. Next milestone should pick a games
target that gets `epsilon` meaningfully lower (it decays ~0.05%/game
multiplicatively, so reaching the 0.05 floor needs several thousand games,
consistent with the paper's 10,000-game reference run in
`references/DeepRL_Monopoly/PPO_PLUS_RULES.md`) before re-running this same
paired-seed evaluation protocol. Both checkpoints and all raw JSON outputs
from this entry are preserved locally (gitignored, not committed) for exact
re-comparison later.

## 2026-08-11 — ASU-value-v1 as evaluation opponent only (short benchmark)

- Hypothesis: `asu-value-v1` can be run as a **fixed evaluation opponent**
  (always permitted, unchanged by today's `CLAUDE.md` ASU-policy correction)
  against `fixed-a/b/c` on a held-out seed, cleanly — no crashes, no illegal
  actions, no compatibility fallbacks — and its per-game cost is worth
  measuring before considering it as a teacher signal.
- Setup: reused `scripts/run_baseline_match.py` unmodified (no ASU-specific
  code needed — `asu-value-v1` was already a supported `--focus`/`--opponents`
  value via the reference's own `ASU_FROZEN_TEACHER.evaluate` CLI).
  `PYTHONHASHSEED=0` enforced by the script's existing guard. Seed `20000`
  chosen fresh — not used in the DDQN training or either DDQN paired
  evaluation, so this is an independent held-out probe.

```bash
PYTHONHASHSEED=0 python scripts/run_baseline_match.py \
  --focus asu-value-v1 --opponents fixed-a fixed-b fixed-c \
  --seed 20000 \
  --output docs/baseline_runs/asu_eval_probe_seed20000.json
```

- Wall time: **2,126.6s (35.4 min) for 4 games** (one seed, full seat
  rotation) — **~531.7s/game average**. This is roughly **1,000x** slower
  than the fixed-vs-fixed engine smoke (which ran 4 games in ~1.5s) or the
  DDQN paired evaluations (~1.2-1.4s/game). The cost is inherent to ASU's
  heuristic (5-turn dice enumeration + 5-lap landing distribution evaluated
  for every legal action at every decision), not this project's code. Only
  **one seed** was run — going further (e.g. the same `10000-10009` range
  used for the DDQN evaluations) would cost on the order of 15-20 CPU-hours
  at this rate, which is why this is a short, single-seed probe rather than a
  matched 10-seed benchmark.
- Result: `asu-value-v1` won **3/4** games in this single seed's rotation
  (Wilson 95% CI **[30.1%, 95.4%]** — extremely wide at `n=4`, not a
  strength claim, just what one seed shows). Mean net worth 31,711.25 vs.
  fixed-a 0.0, fixed-b 6,072.5, fixed-c 0.0.
- **Fallbacks by policy**: `asu-value-v1`: **0** (every game); `fixed-a`: 4,
  `fixed-b`: 22, `fixed-c`: 10 (total 36 — matches the block total exactly).
  Same pattern already established for the DDQN checkpoints: ASU's adapter
  (`ASU_FROZEN_TEACHER.evaluate`'s ASU wrapper) never needs the scripted
  compatibility fallback; only the raw fixed-policy adapter does.
- **Illegal actions**: 0 — the evaluator's `RuntimeError` guard on illegal
  actions was never triggered (would have aborted the whole run non-zero;
  exit code was 0). **Crashes**: 0. **Truncations**: 0 (all 4 games completed
  normally; one hit the 200-round cap with a real net-worth-tie-break
  winner, same pattern documented for earlier smokes).
- ASU was used **only** as an evaluation opponent here, per `CLAUDE.md` — no
  ASU output was collected, imitated, distilled, or used as a training
  label. This does not touch or start the `monopoly_bench collect-asu`/
  `train` pipeline investigated separately in `docs/REFERENCE_AUDIT.md`.

### Conclusion / next step

ASU-value-v1 runs cleanly as a fixed evaluation opponent — no plumbing
concerns. Its cost (~530s/game blended, i.e. much worse per ASU-side
decision) is the real constraint on ever using it as a `collect-asu` teacher
signal at scale: see `docs/REFERENCE_AUDIT.md`'s MonopolyZero/ASU-teacher
section for the resulting cost extrapolation and the recommended smallest
next step (`python -m monopoly_bench smoke`, which needs a PPO bootstrap
checkpoint we don't have yet, before any ASU-teacher training is even
attempted).

## 2026-08-11 — ASU-independent MonopolyZero inference smoke (PPO-compatible checkpoint + `monopoly_bench smoke` + sim-count runtime)

- Hypothesis: the MonopolyZero inference path (`MonopolyZeroNet` +
  `MaxNPUCT` PUCT/Max-N search) can be exercised end to end — load, forward,
  legal-action masking, search — with **zero ASU involvement**, once a
  PPO-format-compatible checkpoint exists. Goal was checkpoint format/
  architecture compatibility only, not policy strength, and no self-play,
  ASU collection, large training, Modal, or LLM work was started.

### Step 1 — minimal PPO-compatible checkpoint (CPU, seed 42, 1 game)

`MonopolyZeroNet.load_ppo_actor` needs a real PPO-format checkpoint (DDQN
weights are not compatible — different network class). Trained the smallest
possible one purely for format/architecture compatibility:

```bash
PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python references/DeepRL_Monopoly/tools/train_and_save.py \
  --algo ppo --games 1 --device cpu --seed 42 --checkpoint-every 1 \
  --out references/DeepRL_Monopoly/artifacts/ppo_plus/ppo_hybrid_2000_v2.pt
```

Saved to the exact path `monopoly_bench/cli.py`'s `smoke` subcommand
hardcodes as `DEFAULT_PPO` (it takes no `--model`/`--checkpoint` flag) —
`references/DeepRL_Monopoly/artifacts/ppo_plus/ppo_hybrid_2000_v2.pt`. This
is inside the submodule's own working tree but under its own `.gitignore`
(`artifacts/`), so it is genuinely local/untracked, not a submodule
modification: `git -C references/DeepRL_Monopoly status --short` stayed
empty before and after, confirmed both times.

- Wall time: 12.88s real (7.4s reported training time; PPO on CPU is far
  faster per-game than DDQN at this scale — 1 game, not a general claim).
- Games completed: 1/1. Peak process RSS: 0.26 GiB. No crash.
- Checkpoint metadata (read directly): `format_version: 3`,
  `hidden_dim: 256` (matches `MonopolyZeroNet`'s default hidden_dim, verified
  by reading both classes' constructors before running anything), `hybrid:
  True`, `games_trained: 1`, `step_count: 598`.
- Size: 14,204,371 bytes.
- **SHA-256**: `1c825dcdd2c8d83651bd21100024ab2d0b8ce2ba276d701dceb3599536f615cb`
- No ASU import anywhere in this step — `tools/train_and_save.py`'s PPO path
  was already grepped clean of ASU references in the first DDQN smoke
  entry's compliance check, and this run used the identical code path with
  `--algo ppo` instead of `--algo ddqn`.

### Step 2 — `python -m monopoly_bench smoke`

```bash
cd references/DeepRL_Monopoly
PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python -m monopoly_bench smoke
```

Output:

```json
{
  "chosen_action": 1,
  "dice_outcomes": 36,
  "latency_s": 0.0707757,
  "ruleset": "ppo-plus-v2",
  "simulations": 4,
  "status": "ok"
}
```

- Wall time: 5.43s real (includes Python/torch startup; the search itself
  was 0.071s). **Peak RSS measured separately at 0.2206 GiB** (polled the
  child process + subprocesses via `psutil` every 20ms until exit).
- **`MonopolyZeroNet` loads**: `model.load_ppo_actor(DEFAULT_PPO)` inside
  `smoke()` did not raise — it validates `format_version`, `ruleset`,
  `state_dim`, `action_dim`, `hidden_dim` before loading, so this is a real
  compatibility check, not just "a file existed."
- **Policy/value forward works**: `MaxNPUCT` calls the model's forward pass
  internally on every simulation; the search completed and returned a
  result, which is not possible if forward had raised.
- **Legal-action masking works**: `smoke()` itself asserts
  `result.chosen_action in game.env.get_allowed_actions(actor)` and raises
  `RuntimeError("Smoke search selected an illegal action")` otherwise — it
  did not raise, so this was verified, not assumed.
- **Max-N PUCT with 4 simulations returns a valid action**: `simulations: 4`,
  `chosen_action: 1`, confirmed legal per the above.
- **Crash / illegal action / fallback**: none. Exit code 0, `"status":
  "ok"`. No fallback mechanism applies here — `MaxNPUCT`/`SearchAdapter`
  select only from the legal-action-masked policy output by construction
  (see `docs/REFERENCE_AUDIT.md`'s ASU-independent-parts breakdown), unlike
  the scripted-agent adapter, which is the only one that ever needs a
  compatibility fallback.
- Zero ASU coupling: `monopoly_bench/search.py` and `monopoly_bench/model.py`
  both grepped clean of "asu" (see `docs/REFERENCE_AUDIT.md`); `smoke()`
  itself never touches `Trainer`, `collect_asu_examples`, or
  `bootstrap_asu_expert`.

### Step 3 — 4/16/32-simulation runtime on the identical state

`monopoly_bench smoke` hardcodes `simulations=4` with no CLI override, so
[scripts/monopolyzero_sim_runtime.py](../scripts/monopolyzero_sim_runtime.py)
reuses the exact same state-construction code (`SharedGame.new(23,
max_rounds=2)`, same property-ownership setup, same decision seed `101`) to
measure 4/16/32 simulations from an identical starting state each time. No
ASU import (enforced by
`tests/test_monopolyzero_sim_runtime.py::test_no_asu_import_in_script_source`).

```bash
PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python scripts/monopolyzero_sim_runtime.py
```

| Simulations | Latency (s) | Chosen action | Legal |
|---|---|---|---|
| 4  | 0.0694 | 1 | yes |
| 16 | 0.1114 | 1 | yes |
| 32 | 0.2225 | 1 | yes |

Roughly linear scaling with simulation count (0.069 → 0.111 → 0.222s), as
expected for a search whose cost is dominated by per-simulation env
clone + forward pass. Same chosen action (`1`) at every simulation count on
this particular state — not a general claim that action choice is
insensitive to simulation count, just what this one state showed. All three
legal, no crash, no illegal action, no fallback (same reasoning as Step 2).

### Conclusion / next step

The full ASU-independent MonopolyZero inference path — checkpoint load,
policy/value forward, legal-action masking, PUCT/Max-N search — works end to
end on CPU with zero ASU involvement anywhere in the call graph. This
directly enables the plan recorded in `docs/PLAN.md`'s MonopolyZero section.
No self-play, no `collect-asu`, no large training, no Modal, no LLM — none
of that was started, per this task's explicit scope.

**Recommended next smallest training experiment** (not started): a tiny,
**hand-built self-play smoke** — a handful of games (e.g. 2-4) played via
`MaxNPUCT` + `arena.play_game` with a hand-picked, ASU-excluded opponent
pool (self-copies and/or `fixed-a/b/c`, never `ASUAdapter`), just enough to
prove positions can be collected into `ReplayBuffer` and one `train_step`
update runs without error — mirroring how the 20-game DDQN smoke validated
the DDQN training loop before scaling up. This needs its own
`docs/DECISIONS.md` entry before starting, since it means writing new
self-play wiring rather than calling an existing entry point (`Trainer.
run_generation` is not usable as-is — see `docs/REFERENCE_AUDIT.md`).

## 2026-08-11 — ASU-independent self-play training-plumbing smoke

- Hypothesis: a hand-built self-play loop (never touching `Trainer`/
  `population_jobs`, which hardcode ASU into every generation) can play a
  handful of short games, collect their search-derived positions into a
  `ReplayBuffer`, and run one real gradient update — proving training
  plumbing works, with zero ASU involvement anywhere in the call graph.
  Decision to attempt this logged first in `docs/DECISIONS.md` (2026-08-11,
  "Try an ASU-independent custom self-play wiring smoke"), per this
  session's instructions. **Goal was plumbing, not strength** — no claim
  about policy quality is made or implied by any number below.
- Setup: new
  [scripts/selfplay_train_smoke.py](../scripts/selfplay_train_smoke.py) +
  [tests/test_selfplay_train_smoke.py](../tests/test_selfplay_train_smoke.py)
  (10 tests: hash-seed guard, no-ASU-import check, no-`Trainer`/
  `population_jobs`-usage check, opponent-pool-is-limited check). Only
  imports ASU-independent public building blocks:
  `monopoly_bench.model.MonopolyZeroNet`,
  `monopoly_bench.adapters.{SearchAdapter, FixedAdapter}` (search.py and
  model.py both grepped clean of "asu" — see `docs/REFERENCE_AUDIT.md`),
  `monopoly_bench.arena.play_game` (itself ASU-free; zero "asu" references),
  `monopoly_bench.storage.ReplayBuffer`, `monopoly_bench.training.train_step`
  (the plain self-play loss, not `expert_train_step`), and
  `monopoly_game_engine.agents_fixed.FP_AGENT_CLASSES[:3]` (fixed-a/b/c).
  Bootstrapped the model from the PPO-compatible checkpoint already
  validated in the prior smoke entry (SHA-256
  `1c825dcdd2c8d83651bd21100024ab2d0b8ce2ba276d701dceb3599536f615cb`).
  `SearchConfig(simulations=4, max_depth=16)`, `max_rounds=5` per game (kept
  short, per the task).

### Games played (3, within the requested 2-4)

Opponent pool used: **only** self-copy (our own model in every seat) and
`fixed-a`/`fixed-b`/`fixed-c` — nothing else, verified by a dedicated test.

```bash
PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python scripts/selfplay_train_smoke.py
```

| Game | Seed | Opponent pool | Decisions | Positions collected | Winner | Illegal | Crash |
|---|---|---|---|---|---|---|---|
| self_play_1 | 501 | self-copy (all 4 seats) | 407 | 333 | seat 0 | 0 | 0 |
| self_play_2 | 502 | self-copy (all 4 seats) | 390 | 325 | seat 1 | 0 | 0 |
| vs_fixed | 503 | fixed-a, fixed-b, fixed-c (seats 1-3) | 270 | 63 | seat 3 | 0 | 0 |

- **721 training samples collected** total (333 + 325 + 63), all written to
  a fresh `ReplayBuffer` (`replay_buffer_size_after_append: 721`,
  `replay_indices_written: 721` — every position accepted, none rejected by
  the buffer's own shape/finite/legal-action validation in
  `ReplayBuffer._write`).
- All 3 games `completed: true`. **0 illegal actions, 0 crashes**, across
  all games (`arena.play_game` fails a game closed and records
  `illegal_actions`/`crashes` on any problem — both stayed 0 the whole way).

### One `train_step` update

Sampled one batch of 8 positions (`np.random.default_rng(42)`,
`ReplayBuffer.sample`, `batch_size_sampled: 8`) and called
`monopoly_bench.training.train_step` **exactly once**:

```json
{
  "loss": 3.2124853134155273,
  "policy_loss": 1.9503673315048218,
  "value_loss": 1.2621181011199951,
  "gradient_norm": 8.63224983215332
}
```

- **Loss finite**: yes (`loss_finite: true` — checked `torch.isfinite` on
  loss, policy_loss, value_loss, gradient_norm before reporting).
- **At least one parameter changed**: `parameters_changed_count: 16` out of
  `parameters_total_count: 16` — **every** parameter tensor in the model
  changed value after the single update (compared via `torch.equal` against
  a pre-update clone of every named parameter), not just one.
- No ASU output, label, or data was used anywhere in the batch —
  `train_step` (not `expert_train_step`) trains the policy head from each
  position's own MCTS visit counts and the value head from each game's real
  winner, both produced entirely by our own model's search and the engine's
  own rules.

### Runtime and RAM

- Elapsed (script-measured, games + buffer write + one update):
  **11.10s**. Full process wall time (`time -p`, includes Python/torch
  startup): **15.07s** real.
- Peak RSS (background `psutil` polling thread, 20ms interval, whole run):
  **0.291 GiB**.
- A second run (not saved as the committed artifact) produced very similar
  but not byte-identical numbers (721 vs. 697 positions, different winners) —
  expected, not a new reproducibility bug: `self_play=True` deliberately
  injects exploration randomness (temperature sampling / Dirichlet noise) as
  part of `MaxNPUCT`'s designed self-play behavior, unlike the strict
  per-step engine RNG isolation already documented for the fixed/DDQN/ASU
  evaluation paths. Not investigated further — out of scope for a plumbing
  smoke.

### Conclusion / next step

Training plumbing works end to end — games → positions → replay buffer →
one real gradient update — with zero ASU involvement anywhere (confirmed by
which functions were called, not just by absence of an import). No
multi-generation training, no ASU collection, no large self-play volume, no
Modal, no LLM — none of that was started, per this task's explicit scope.

**Recommended next smallest strength experiment** (not started): extend this
same script's pattern to noticeably more games (e.g. 50-100, still small)
and more than one `train_step` update (e.g. a few dozen), then run the
*existing* paired-seed evaluation protocol
(`scripts/run_baseline_match.py --seed 10000-10009` against `fixed-a/b/c`,
same Wilson-interval discipline already used for the DDQN checkpoints) to
see whether the resulting MonopolyZero checkpoint shows any measurable skill
signal yet — mirroring exactly how the DDQN checkpoint went from a 20-game
smoke to a 500-game milestone with a paired held-out evaluation. Needs its
own `docs/DECISIONS.md` entry first, per this project's standing practice.
