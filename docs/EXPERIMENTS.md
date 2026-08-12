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

## 2026-08-11 — Self-play smoke reproducibility fix and check (before any strength experiment)

- Hypothesis: the prior self-play smoke entry (above) attributed run-to-run
  variation entirely to `self_play=True`'s intentional exploration
  randomness. That explanation was incomplete: `MonopolyZeroNet`'s
  `value_head` is never overwritten by `load_ppo_actor` (only the
  PPO-compatible trunk/policy_head are), so it was left at an **unseeded**
  random init every run — a real, fixable reproducibility gap, not just
  designed exploration noise.
- Fix (`scripts/selfplay_train_smoke.py`, no submodule changes): seed
  Python's `random`, NumPy's global RNG, and `torch.manual_seed` once, right
  before `MonopolyZeroNet()` is constructed (`GLOBAL_SEED = 42`). The
  existing per-game/per-decision seeding (`SEEDS = {501, 502, 503}`, and the
  `decision_seed` `arena.play_game` derives internally) was left untouched,
  per the task's explicit instruction.
- Also added: a `_require_clean_git_tree()` guard (refuses to run on a dirty
  tree, returns `git rev-parse HEAD` otherwise) so the script's own output
  carries an unambiguous `git_head_sha` — this is what closed the
  `code_commit_sha` provenance gap described in the standard-fix entry
  above. And: `FixedAdapter.compatibility_fallbacks` is now read directly
  from each fixed-agent instance after the `vs_fixed` game and reported as
  `fixed_adapter_fallbacks`/`fixed_adapter_fallbacks_total`, instead of
  being left `null` (it was never actually zero-by-assumption; it just
  wasn't measured before).

### Reproducibility check

Ran the fixed script twice, independently, same seed, immediately after
committing the fix (clean tree both times, both runs against commit
`3b7db1f06bc60ca7f31497034efb42dd705e770e`):

```bash
PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python scripts/selfplay_train_smoke.py
PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python scripts/selfplay_train_smoke.py
```

**Result: PASSED.** `diff` between the two outputs, excluding `elapsed_s` and
`peak_rss_gib`, was empty (exit code 0):

| | Run 1 | Run 2 |
|---|---|---|
| Positions collected | 709 | 709 |
| Winners (self_play_1 / self_play_2 / vs_fixed) | seat 1 / seat 3 / seat 3 | seat 1 / seat 3 / seat 3 |
| Batch size sampled | 8 | 8 |
| `loss` | 2.804666519165039 | 2.804666519165039 |
| `policy_loss` | 1.4619874954223633 | 1.4619874954223633 |
| `value_loss` | 1.3426790237426758 | 1.3426790237426758 |
| `gradient_norm` | 9.985905647277832 | 9.985905647277832 |
| Parameters changed | 16/16 | 16/16 |
| `fixed_adapter_fallbacks` (a/b/c) | 0/0/0 | 0/0/0 |

Every deterministic field matched exactly, including all four loss figures
bit-for-bit. `FixedAdapter` had zero compatibility fallbacks in either run
(now a real measurement, not an unmeasured `null`).

### Conclusion / next step

The RNG-seeding fix closes the previously-documented non-determinism gap;
this self-play training-plumbing path is now verified bit-exact
reproducible, clearing the way to consider a strength experiment (per the
recommendation already on record in the prior entry: more games, more
`train_step` updates, then the existing paired-seed evaluation protocol).
See [logs/experiments/011-selfplay-smoke-reproducibility-check.json](../logs/experiments/011-selfplay-smoke-reproducibility-check.json)
for the full structured record.

## 2026-08-11 (later) — Self-play smoke made genuinely ASU-import-free

- Hypothesis/correction: the self-play smoke's "ASU-independent" claim (both
  entries above) was overstated. `scripts/selfplay_train_smoke.py` imported
  `monopoly_bench.adapters` and `monopoly_bench.arena` for `SearchAdapter`/
  `FixedAdapter`/`play_game`, and `monopoly_bench.training` for `train_step`.
  `adapters.py` does `from ASU_FROZEN_TEACHER import ASURolloutV1,
  ASUValueV1` at module scope; `training.py` does `from ASU_FROZEN_TEACHER
  import FROZEN_SPEC_HASH` at module scope; `arena.py` imports `.adapters`.
  So every prior run of this smoke loaded `ASU_FROZEN_TEACHER` into
  `sys.modules` as an import side effect — ASU output/teacher/label was
  never used (that ban was never actually violated), but "ASU-independent"
  was not an accurate description of the import graph. See
  `docs/DECISIONS.md`'s 2026-08-11 (later) correction entry.
- Fix: rewrote the script to import only confirmed ASU-import-clean modules
  — `monopoly_bench.engine`, `.model`, `.search`, `.storage`, `.config`,
  `.contracts`, `monopoly_game_engine.agents_fixed` — and replaced
  `SearchAdapter`/`FixedAdapter`/`arena.play_game`/`train_step` with this
  project's own implementations (`LocalSearchPolicy`, `LocalFixedPolicy`,
  `play_local_game`, `local_training_update`) built from the lower-level
  primitives (`MaxNPUCT`, `SharedGame`, `ReplayBuffer`, `FP_AGENT_CLASSES`)
  and their observed contracts — not copied from `adapters.py`/`arena.py`/
  `training.py`. Added a runtime guard: at the end of the run, `sys.modules`
  is checked for `ASU_FROZEN_TEACHER` (or any submodule); the run fails if
  one is present, so "ASU-independent" is verified every run, not asserted.

### Result

```bash
PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python scripts/selfplay_train_smoke.py
```

- `asu_modules_loaded: []`, `asu_modules_loaded_count: 0` — confirmed in
  both runs below by the script's own guard (which would have raised
  otherwise).
- Ran twice, independently, same seed, same clean HEAD
  (`f835a74dcb85acb7aa22a60ded5abaecb9d1fc90`):

| | Run 1 | Run 2 |
|---|---|---|
| Positions collected | 709 | 709 |
| Winners (self_play_1 / self_play_2 / vs_fixed) | seat 1 / seat 3 / seat 3 | seat 1 / seat 3 / seat 3 |
| `loss` | 2.804666519165039 | 2.804666519165039 |
| `policy_loss` / `value_loss` / `gradient_norm` | 1.4619874954223633 / 1.3426790237426758 / 9.985905647277832 | identical |
| Parameters changed | 16/16 | 16/16 |
| `fixed_adapter_fallbacks` (a/b/c) | 0/0/0 | 0/0/0 |
| Illegal actions / crashes | 0 / 0 | 0 / 0 |

`diff` between the two outputs, excluding `elapsed_s`/`peak_rss_gib`, was
empty — **reproducibility PASSED** again after the rewrite.

**Notably**, these numbers (709 positions, same winners, same four loss
values, 16/16 parameters changed) are numerically identical to the last run
of the *old*, ASU-import-coupled implementation on the same seed — strong
evidence the reimplementation preserves behavior exactly while removing the
import-time ASU dependency, not a coincidence.

### Historical entries corrected, not deleted

`logs/experiments/010-selfplay-training-plumbing-smoke.json` and
`011-selfplay-smoke-reproducibility-check.json` both predate this fix and
both loaded ASU as an import side effect. Their `notes` fields now carry an
explicit correction (their measured results — positions, winners, loss,
illegal actions, crashes — are unaffected and remain accurate; only the
import-cleanliness claim was overstated). Numbers were not rewritten or
removed. See
[logs/experiments/012-selfplay-asu-import-free-smoke.json](../logs/experiments/012-selfplay-asu-import-free-smoke.json)
for the full structured record of this fix and rerun.

### Conclusion / next step

The self-play training-plumbing smoke is now genuinely ASU-import-free,
bit-exact reproducible, and behaviorally verified equivalent to the prior
implementation. This clears the way for the previously-recommended next
step (more games, more `train_step` updates, then the existing paired-seed
evaluation protocol) without carrying forward the import-graph caveat.

## 2026-08-11 (later) — First MonopolyZero strength pilot: 32-game training

- Housekeeping first: `docs/PLAN.md` updated (DDQN long-scaling paused, this
  pilot is the current milestone), and a source-similarity audit of
  `scripts/selfplay_train_smoke.py`'s game loop / training-update code found
  real overlap with `arena.py::play_game` / `training.py::train_step`
  (identical magic constants, near line-for-line structure with renamed
  variables) — with no compatible license in the reference repo, that's a
  real risk. Refactored into `scripts/monopolyzero_common.py` with a
  different decision-seed mix, a different control-flow shape (closure +
  `while` loop vs. a single `for step in range(...)` body), and a dense
  scatter-then-normalize policy-target formulation instead of the
  reference's sparse gather-and-mask — same verified behavior, independent
  expression. See that module's docstring and `docs/DECISIONS.md`.
  Verification run after the refactor caught a real bug (dense policy
  cross-entropy produced `NaN` from `0 * -inf` on masked illegal-action
  logits) — fixed and regression-tested before proceeding to this pilot.

### Training

New reusable runners: `scripts/monopolyzero_strength_train.py` and
`scripts/monopolyzero_strength_eval.py`, both built on
`monopolyzero_common.py` — no `monopoly_bench.adapters`/`.arena`/`.training`
import anywhere, verified by the same `sys.modules` guard used since the
ASU-import-free self-play smoke.

```bash
PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python scripts/monopolyzero_strength_train.py
```

- **32/32 games completed** (16 self-play, our model in every seat; 16
  vs-fixed, our model seat-rotated evenly 0-3, `fixed-a/b/c` filling the
  rest), seed 42, CPU, 4 PUCT simulations, `max_rounds=50`.
- **0 illegal actions, 0 crashes.** 11 fixed-agent fallbacks total, all from
  `TheDealMaker` in one game (seed 20013) — not an error, the documented
  compatibility-fallback path.
- **37,772 positions** collected into the replay buffer (all accepted).
- **100/100 training updates finite** — final update: `loss=2.4687`,
  `policy_loss=1.6583`, `value_loss=0.8104`, `gradient_norm=3.1287`.
- Optimizer hyperparameters (`lr=3e-4`, `weight_decay=1e-4`,
  `gradient_clip=1.0`) read directly from `monopoly_bench.config
  .TrainingConfig`'s defaults (an ASU-free dataclass, just parameters).
  `batch_size=64` and `updates=100` are this project's own pilot-scale
  choice, not `TrainingConfig`'s frozen-v1 defaults (`batch_size=256`,
  `updates_per_generation=1000`, sized for 32-games-*per-generation* at full
  scale, not a one-shot 32-game pilot).
- Wall time **301.55s** (~5 min), peak RSS **0.665 GiB**.
- `asu_modules_loaded: []` (count 0) — confirmed by the script's own guard.
- Checkpoints saved and hashed, not committed (gitignored):
  baseline (pre-training) SHA-256
  `22ae2abea60478355f61ce0404a31de3b52fb97769d4f99f7e04f4c673629370`,
  trained (post-training) SHA-256
  `d00263f4a2ab3cdd73a4d2691bdfae42385292a153996a17198b0074433d0f93`.

None of the stop conditions triggered (no crash, no illegal action, no
non-finite loss, no ASU module load), so proceeding to the paired evaluation
— see the next entry.

Full structured record:
[logs/experiments/013-monopolyzero-strength-pilot-training.json](../logs/experiments/013-monopolyzero-strength-pilot-training.json).

### Paired evaluation: baseline vs. trained

```bash
PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python scripts/monopolyzero_strength_eval.py
```

Held-out seeds `30000-30004` (never used in training), 4-seat rotation ×
5 seeds = 20 games per checkpoint, vs. `fixed-a/b/c`, `max_rounds=200`,
`self_play=False` (greedy/deterministic — masked argmax over visit counts,
no temperature sampling or Dirichlet noise), same search config as training
(4 simulations, depth 16). **ASU benchmark not run**, per task instruction.

| | Baseline (pre-training) | Trained (32 games) |
|---|---|---|
| Win rate | 0/20 = **0.0%** | 1/20 = **5.0%** |
| Wilson 95% CI | **[0.0%, 16.1%]** | **[0.9%, 23.6%]** |
| Mean net worth | 2,450.4 | 3,759.3 |
| Round-cap rate | 45% | 55% |
| p95 search latency | 0.0132s | 0.0141s |
| Illegal actions / crashes | 0 / 0 | 0 / 0 |
| Fixed-agent fallbacks | 17 (fixed-b: 9, fixed-c: 8) | 17 (fixed-c: 17) |

Wall time **322.39s** (~5.4 min), peak RSS **0.206 GiB**.
`asu_modules_loaded_count: 0` for both checkpoints.

**NO-SIGNAL — no improvement claim.** The trained checkpoint's Wilson
interval `[0.9%, 23.6%]` overlaps almost completely with the baseline's
`[0.0%, 16.1%]`; the non-overlap test (`trained_lower > baseline_upper`)
fails. One extra win and a higher mean net worth over 20 games is not
distinguishable from noise at this sample size — same discipline already
applied to the DDQN 20-vs-500 comparison. **No statistically supported
regression either** on win rate/net worth (both moved in the same
direction, just not far enough to be significant) — **but this is not
"nothing got worse."** *Correction, 2026-08-12: round-cap rate moved from
45% (baseline) to 55% (trained) — the trained checkpoint's games hit the
200-round cap without a decisive winner more often, which is a move in the
undesirable direction. It wasn't tested for statistical significance (no
Wilson interval was computed for it, unlike win rate), so "statistically
supported regression" still correctly reads "none confirmed" — but the
original wording overclaimed by implying every metric was flat or better.
Historical numbers (45%/55%, win rates, net worth) are unchanged; only this
interpretive sentence is corrected. See
[logs/experiments/014-monopolyzero-strength-pilot-paired-eval.json](../logs/experiments/014-monopolyzero-strength-pilot-paired-eval.json)'s
`notes` field for the same correction attached to the structured record.*

Full structured record:
[logs/experiments/014-monopolyzero-strength-pilot-paired-eval.json](../logs/experiments/014-monopolyzero-strength-pilot-paired-eval.json).

### Conclusion / next step

Training plumbing works at pilot scale (32 games, 100 updates) with zero
ASU involvement, zero crashes/illegal actions, and both checkpoints hashed
and preserved locally for exact re-comparison. The paired evaluation found
no statistically supported skill difference yet — 32 games / 100 updates is
a small pilot, and this result (like the DDQN 20-vs-500 case) is the honest,
correct outcome to report, not a failure. A larger pilot (more games, more
updates) with the same paired-seed evaluation protocol is the natural next
step, and per this project's standing practice needs its own
`docs/DECISIONS.md` entry with a reason to expect it'll move the needle,
not just "more of the same."

## 2026-08-12 — Update-budget sweep: does more training on the SAME data help?

Correction first: the prior entry's "nothing got worse" claim was wrong —
`round_cap_rate` moved from 45% (baseline) to 55% (trained), an adverse
direction never tested for significance. Corrected in place above and in
`logs/experiments/014-*.json`'s `notes`; the 45%/55% numbers themselves are
unchanged.

**Question**: with zero new self-play games, does training for longer on
013's existing 32-game / 37,772-position replay buffer produce a measurably
different policy? Isolates training-update budget as the single variable.

### Integrity check (before touching anything)

Built into `scripts/monopolyzero_update_budget_sweep.py`'s
`verify_reused_artifacts()`, which runs before any training and refuses to
proceed on a mismatch (no separate standalone command — it's the first
thing the training command below does):

- Replay buffer present, `metadata.json` size **37,772** — matches.
- Baseline checkpoint present, SHA-256
  `22ae2abea60478355f61ce0404a31de3b52fb97769d4f99f7e04f4c673629370` —
  matches 013's recorded value exactly.
- **PASSED** — proceeded to training. (Had either check failed, the script
  raises `SystemExit` before any training call, so a mismatch would have
  produced zero new checkpoints, not a silent fallback.)

### Training: 100 / 500 / 1000 updates, zero new games

```bash
PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python scripts/monopolyzero_update_budget_sweep.py
```

Each budget starts fresh from the same baseline checkpoint (never resumes
from another budget's checkpoint) with the same sampling seed (42), so the
first 100 updates are byte-identical across all three runs and the first
500 identical between the 500- and 1000-update runs — verified directly:
the interval-logged loss values at updates 25/50/75/100 are bit-identical
across all three, and 125-500 identical between the 500 and 1000 runs.
`batch_size=64`, `lr=3e-4`, `weight_decay=1e-4`, `gradient_clip=1.0` (the
last three read directly from `TrainingConfig`'s defaults).

| Budget | Final loss | Final policy_loss | Final value_loss | Checkpoint SHA-256 (short) |
|---|---|---|---|---|
| 100  | 2.4687 | 1.6583 | 0.8104 | `1a428b96...` |
| 500  | 1.9708 | 1.6572 | 0.3136 | `152c0a0f...` |
| 1000 | 1.6482 | 1.5406 | 0.1076 | `06f65...` |

All updates finite across all three budgets — no stop condition triggered.
Total wall time for all three combined: **12.98s** (no game generation, so
dramatically faster than 013's 301.55s). `asu_modules_loaded_count: 0`.

**Loss trend**: total and value loss both drop steadily; value_loss in
particular falls sharply and close to monotonically (0.81 → 0.31 → 0.11).
policy_loss barely moves (1.658 → 1.657 → 1.541). This pattern — value loss
dropping fast on a *fixed*, *small* (37,772-position) dataset while policy
loss stays roughly flat — is consistent with the value head increasingly
fitting (or overfitting) the same 32 games' outcomes, not necessarily with
the policy generalizing better. Flagged as a concern to watch in the
evaluation below, not asserted as fact from the loss curve alone.

Full structured record:
[logs/experiments/015-monopolyzero-update-budget-sweep-training.json](../logs/experiments/015-monopolyzero-update-budget-sweep-training.json).

### Paired evaluation: baseline (0), 100, 500, 1000 updates

```bash
PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python scripts/monopolyzero_update_budget_eval.py
```

Held-out seeds `30000-30009` (10 seeds, never used in training), 4-seat
rotation × 10 seeds = 40 games/checkpoint = **160 games total**, vs.
`fixed-a/b/c`, `max_rounds=200`, `self_play=False`, same search config as
training (4 simulations, depth 16). ASU benchmark not run.

| | Baseline (0) | 100 updates | 500 updates | 1000 updates |
|---|---|---|---|---|
| Win rate | 1/40 = **2.5%** | 1/40 = **2.5%** | 3/40 = **7.5%** | 1/40 = **2.5%** |
| Wilson 95% CI | [0.4%, 12.9%] | [0.4%, 12.9%] | [2.6%, 19.9%] | [0.4%, 12.9%] |
| Mean net worth | 2,961.9 | 3,794.3 | 3,597.8 | 4,333.1 |
| Median net worth | 0.0 | 0.0 | 0.0 | 2,274.5 |
| Bankruptcy rate | 62.5% | 55.0% | 62.5% | 50.0% |
| Round-cap rate | 40.0% | 50.0% | 47.5% | 55.0% |
| p95 search latency | 0.0134s | 0.0139s | 0.0125s | 0.0126s |
| Illegal / crash | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Fixed fallbacks | 56 | 43 | 81 | 76 |

Wall time **1306.37s** (~21.8 min), peak RSS **0.210 GiB**.
`asu_modules_loaded_count: 0` across all four checkpoints.

**Paired comparisons** (non-overlapping-Wilson-interval test):
`100 vs 500`: **not statistically supported**. `500 vs 1000`: **not
statistically supported** — and worth stating plainly rather than just
reporting the negative: 1000's win rate (2.5%) is nominally *lower* than
500's (7.5%), the opposite of an improvement, even though 1000 had by far
the lowest training loss of the three. Not claimed as a statistically
supported regression either (the two Wilson intervals still overlap
substantially) — same discipline as the round-cap correction above: report
the direction honestly without overclaiming significance either way.

**GO / NO-SIGNAL / REGRESSION: NO-SIGNAL.** No update budget in this sweep
shows a statistically supported win-rate improvement over baseline or over
another budget. 500 updates had the nominally best win rate of the four,
but its interval overlaps all the others. Net worth trended upward with
more updates, but net worth is not a win-rate proxy and bankruptcy rates
stayed high (50-62.5%) throughout — most games are still lost outright, not
won narrowly.

Full structured record:
[logs/experiments/016-monopolyzero-update-budget-sweep-paired-eval.json](../logs/experiments/016-monopolyzero-update-budget-sweep-paired-eval.json).

### Conclusion / next step

Training-update budget alone, on this same fixed 32-game dataset, does not
produce a statistically detectable skill change in either direction across
100-1000 updates. Combined with 015's loss-curve observation (value loss
dropping sharply while policy loss stays flat, and 1000's win rate falling
back to baseline despite the lowest loss), the more likely lever is **more
self-play data**, not more updates on the same small replay buffer — larger
update budgets on a fixed 37,772-position set risk overfitting the value
head without improving the policy. Any next step scaling either games or
updates further needs its own `docs/DECISIONS.md` entry with a reason to
expect it'll help, per this project's standing practice — not assumed here.

## 2026-08-11 (later still) — Offline PUCT search-budget diagnostic: KILL

*Note: this and the entry above were originally dated 2026-08-12 in a few
places across this project; that was one day ahead of the actual system
clock. Corrected going forward, not rewritten everywhere retroactively —
see `docs/DECISIONS.md`'s matching correction note.*

**Question**: before spending a full game-evaluation budget on it, does
scaling PUCT simulation count (4 → 16 → 32) on the 500-update checkpoint
actually change decisions enough to be worth measuring in real games? Zero
new training, zero new self-play data, ASU untouched.

### Checkpoint integrity

Verified `artifacts/monopolyzero_strength_pilot/trained_updates_500.pt`
SHA-256 `152c0a0f6136d1fc91e74973ac245b2f72774694c424d2a48854514ed2848383`
matches 016's recorded value exactly — confirmed before running anything
(built into `scripts/monopolyzero_search_budget_diagnostic.py`'s
`verify_checkpoint()`, which raises `SystemExit` on any mismatch).

### Diagnostic

```bash
PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python scripts/monopolyzero_search_budget_diagnostic.py
```

Collected exactly **200 non-forced decision states** by playing sims=4,
`self_play=False` games (the 500-update checkpoint vs. `fixed-a/b/c`, focus
seat rotating) on held-out seeds `31000-31004`, snapshotting a clone of the
game before every decision where the checkpoint faced more than one legal
action. Then re-ran PUCT search on each *exact frozen state* at 4, 16, and
32 simulations (`self_play=False`, `depth=16`) — same state, same decision
seed, only simulation count varies.

| | 4 vs 16 | 16 vs 32 | 4 vs 32 |
|---|---|---|---|
| Chosen-action disagreement | **1.5%** (3/200) | 2.0% (4/200) | 3.5% (7/200) |
| Mean abs. root-value change (self) | 0.0386 | 0.0238 | 0.0492 |

| | 4 sims | 16 sims | 32 sims |
|---|---|---|---|
| Mean chosen-action visit share | 97.75% | 76.91% | 63.48% |
| Search latency mean / p50 / p95 | 0.0090s / 0.0083s / 0.0137s | 0.0396s / 0.0372s / 0.0689s | 0.0808s / 0.0755s / 0.1388s |

Illegal actions during search: **0**. `asu_modules_loaded_count: 0`. Wall
time for the whole diagnostic (collection + 600 searches): **27.95s**
(29.19s including process startup), peak RSS **0.206 GiB**.

Latency scales roughly linearly with simulation count (16 sims ≈ 5x the
p95 latency of 4 sims; 32 sims ≈ 10x) — exactly the cost a full 4-vs-16 or
4-vs-32 game evaluation would have paid per decision, for a chosen-action
difference this diagnostic already shows is small.

**GO / NO-SIGNAL / KILL: KILL.** 4-vs-16 disagreement (1.5%) is well under
the 5% kill threshold, and the mean root-value change (0.0386) is under the
0.05 threshold too — both conditions for killing search-budget scaling are
met. **No full game evaluation was run**, per the task's own decision rule
(not a shortcut — spending ~20+ minutes of game evaluation to confirm what
a 28-second, training-free diagnostic already shows clearly would have been
the wrong call).

Full structured record:
[logs/experiments/017-monopolyzero-search-budget-diagnostic.json](../logs/experiments/017-monopolyzero-search-budget-diagnostic.json).

### Conclusion / next step

More PUCT simulations per decision is not, on its own, a lever worth
pursuing for this checkpoint — the policy's decisions are already stable
under 4x-8x more search. Combined with the update-budget sweep's NO-SIGNAL
result above, neither "train longer on the same data" nor "search harder
at inference time" moved the needle for the 500-update checkpoint. The
remaining unexplored lever, per both entries, is more self-play data —
starting that needs its own `docs/DECISIONS.md` entry, not assumed here.

## 2026-08-11 (later still) — POLICY_ONLY vs PUCT_4: does search add value at all

017 asked whether *more* simulation budget helps (4 vs 16 vs 32) and found
it doesn't. This asks the more basic question underneath it: does search
help *at all*, compared to just reading the policy head off the same
checkpoint? Two inference policies, same 500-update checkpoint
(`152c0a0f...`, integrity-verified before running), same held-out seeds
`32000-32009`, 4-seat rotation, vs. fixed-a/b/c, `max_rounds=200`:

- **POLICY_ONLY**: `MonopolyZeroNet.predict()`'s legal-masked softmax,
  legal argmax. No MCTS.
- **PUCT_4**: the 4-simulation, depth-16, `self_play=False` search every
  prior MonopolyZero evaluation in this project has used.

40 games each. During PUCT_4's 40 games, POLICY_ONLY was additionally
shadow-queried at every non-forced decision the checkpoint faced — same
frozen pre-step state, answer recorded but never acted on — so the
disagreement rate below is measured on the *exact* decision states PUCT_4's
own win-rate numbers came from, not a separate offline sample.

| | POLICY_ONLY | PUCT_4 |
|---|---|---|
| Win rate (Wilson 95% CI) | 5.0% (2/40) `[1.4%, 16.5%]` | 5.0% (2/40) `[1.4%, 16.5%]` |
| Mean / median net worth | 3817.0 / 0.0 | 3919.7 / 2029.5 |
| Bankruptcy rate | 52.5% | 50.0% |
| Round-cap rate | 47.5% | 52.5% |
| Decision latency p50 / p95 | 1.28ms / 4.79ms | 9.83ms / 38.36ms |
| Illegal actions / crashes | 0 / 0 | 0 / 0 |
| Fixed-opponent fallbacks | 151 | 128 |
| Wins by seat | seat 1: 2, others: 0 | seat 1: 2, others: 0 |

**Action disagreement (PUCT_4 vs POLICY_ONLY, same 31,535 decision states
from PUCT_4's own games): 2.93%.** PUCT_4 costs ~7.7x more at p50 and ~8.0x
more at p95 per decision (+8.6ms / +33.6ms) for a chosen-action difference
under 3%.

Both policies won on the exact same two (seed, seat) pairs — not a
coincidence worth being suspicious of, and checked: per-game decision
counts differ slightly between the two policies on shared seeds (e.g. 6687
vs 5775 decisions on seed 32001/seat 0), confirming the trajectories really
do diverge where the policies disagree. At ~97% agreement, most games play
out almost identically, so most outcomes land the same way too — this is
the expected shape of a low-disagreement result, not evidence of a bug.

Neither direction is statistically supported:
`puct_4_improvement_over_policy_only_statistically_supported: false`,
`policy_only_improvement_over_puct_4_statistically_supported: false`.
Zero ASU imports (`asu_modules_loaded_count: 0`), zero new training or
self-play data generated. Wall time 666.8s (80 games total), peak RSS
0.21 GiB.

**GO / NO-SIGNAL / KILL: KILL the current PUCT/MCTS inference path.**
PUCT_4 shows no measurable or consistent advantage over POLICY_ONLY — per
the task's own decision rule, that's a kill. POLICY_ONLY is equal (not
worse), which per the same rule means the next experiment should pivot
toward a search-free learning objective/architecture rather than further
MCTS tuning — spending more inference-time compute on this checkpoint is
not where the missing skill is.

Full structured record:
[logs/experiments/018-monopolyzero-policy-only-vs-puct-eval.json](../logs/experiments/018-monopolyzero-policy-only-vs-puct-eval.json).

### Conclusion / next step

Combined with 013/014 (32-game pilot, NO-SIGNAL), 015/016 (update-budget
sweep, NO-SIGNAL), and 017 (search-budget scaling, KILL), this closes out
every scaling lever tried so far on the current recipe/checkpoint family:
training longer, searching harder, and searching at all versus not
searching, all land at the same place. See `docs/DECISIONS.md`'s
"Pause current MonopolyZero recipe scaling entirely" entry. The two open
levers going forward are (a) a genuinely larger/fresh self-play dataset,
and (b) a different learning objective/architecture that doesn't lean on
search at inference time — neither is started by this entry; each needs
its own proposed experiment and decision log.

## 2026-08-11 (later still x2) — Horizon diagnostic: round-50 leader vs. round-200 winner, and a state-encoding ablation

- Hypothesis: none pre-registered, purely descriptive. Two open questions
  before proposing any new strength-training run on top of 013's replay: (1)
  013's training data was generated at `max_rounds=50`, but the competition
  target is `max_rounds=200` — does a round-50 net-worth leader actually
  predict the round-200 (or terminal) winner well enough for a 50-round
  training signal to be a reasonable proxy? (2) Separately, does the
  `round/max_rounds` scalar baked into the state encoding itself change
  model output at a fixed game state, independent of anything about actual
  gameplay?
- Setup: **Part 1** — 32 fresh games (16 self-play + 16 vs-fixed,
  seat-balanced, `baseline_pretraining.pt`, 4 simulations, `self_play=True`,
  the same search recipe 013 used to generate its training data), but at
  `max_rounds=200` this time. Fresh seeds `40000-40015` /`41000-41015`, no
  overlap with any seed pool used anywhere else in this project. Snapshots
  every player's net worth the instant the round counter first reaches 50,
  then lets the same game continue uninterrupted to round 200 or
  elimination. **Part 2** — from the same 32 games, up to 200 non-forced
  decision states from rounds 1-50 (deterministic: game order, then turn
  order). Each state is cloned and only `env.max_rounds` is flipped
  (200 → 50); the script asserts at runtime (not just assumes) that at most
  one state-vector index changes, then compares `POLICY_ONLY` model output
  between the two encodings, for `baseline_pretraining.pt` and
  `trained_updates_500.pt` separately. No GO/KILL threshold was set for
  either part — this is a measurement, not a test with a pass/fail bar.
- Result:

  **Part 1 — round-50 leader vs. final winner**

  | | Games | Agreements | Agreement rate |
  |---|---|---|---|
  | Overall | 32 | 19 | 59.4% |
  | Self-play | 16 | 6 | 37.5% |
  | vs-fixed | 16 | 13 | 81.25% |

  All 32 games were still live at round 50 (`games_finished_before_round_50:
  0`), so every game is in the agreement sample. The round-50 leader's final
  rank: 1st in 19/32, 2nd in 6, 3rd in 5, 4th in 2. Agreement tracks
  round-50 margin size: 50.0% at margin `<500` and `<2000`, 68.75% at margin
  `>=2000`; mean margin was 4061.6 when the round-50 leader went on to win
  vs. 1590.5 when they didn't (median 3179.5 vs. 1196.0).

  **Part 2 — state-encoding ablation (round/max_rounds only)**

  | | baseline_pretraining | trained_updates_500 |
  |---|---|---|
  | States used | 200 | 200 |
  | POLICY_ONLY action disagreement | 1.5% | 1.0% |
  | Policy TV-distance (mean) | 0.00237 | 0.00497 |
  | Value-head mean abs delta | 0.00136 | 0.01019 |
  | State-vector indices that changed | `{278}` (every state) | `{278}` (every state) |

  Isolation held on every one of the 400 state x checkpoint comparisons —
  flipping `max_rounds` moved exactly index 278 (`min(round/max_rounds,
  1.0)`, hand-traced in `monopoly_game_engine/state.py::build_state_vector`)
  and nothing else, verified programmatically, not assumed. The trained
  checkpoint's *value head* is noticeably more sensitive to this single
  scalar than the untrained baseline's (~7.5x larger mean abs delta), even
  though its *chosen action* changes slightly less often.

  Zero illegal actions, zero crashes, zero ASU modules loaded
  (`asu_modules_loaded_count: 0`). Fixed-opponent fallbacks: 90, all from
  vs-fixed games' non-focus seats (self-play games have none). Wall time
  960.2s, peak RSS 0.21 GiB.

  Full structured record:
  [logs/experiments/019-monopolyzero-horizon-diagnostic.json](../logs/experiments/019-monopolyzero-horizon-diagnostic.json).

- Conclusion / next step: **No GO/KILL/NO-SIGNAL verdict is being called
  here** — per the task instructions this experiment measures only; the
  59.4% overall / 37.5% self-play / 81.25% vs-fixed agreement figures, and
  the 500-update checkpoint's larger value-head sensitivity to the round/50
  vs. round/200 encoding, are handed off as read results, not pre-judged. This
  is exactly the measurement `docs/PLAN.md`'s "37,772-position replay not
  approved for new strength training until a horizon/label audit passes"
  gate was waiting on — the gate itself stays in place until that call is
  made and logged separately.

## 2026-08-11 (later still x3) — Full-horizon value-learnability probe: does the 300-dim state carry the real winner

- Hypothesis: none pre-registered, purely descriptive. Before proposing any
  new policy/strength training, does the existing 300-dim state
  representation carry the TRUE `max_rounds=200` final winner in a
  learnable way at all — measured with the real full-horizon label this
  time, not `013`'s round-50-truncated proxy (`019` already showed that
  proxy is weak, 59.4% overall agreement with the real winner).
- Setup: 64 fresh, clean `POLICY_ONLY` self-play games (`baseline_pretraining.pt`,
  all 4 seats the same checkpoint, zero fixed agents, zero PUCT, seeds
  `42000-42063` newly registered as DEV). Game-level split: first 48 seeds
  TRAIN, last 16 VALIDATION, no state crosses the split. Round-stratified
  deterministic sampling (5 buckets × 4 seats × up to 3 states/game/cell =
  2088 TRAIN / 569 VALIDATION states), each labeled with that game's REAL
  eventual winner (actor-relative one-hot). A small, separate, own-written
  `ValueProbe` MLP (300 → 256 → 4, CPU) was trained purely as a
  supervised final-winner classifier, early-stopped on validation
  cross-entropy — `MonopolyZeroNet`'s own weights were never touched.
  Compared against a uniform baseline and a current-net-worth-leader
  baseline (ε-smoothed for finite cross-entropy) on both splits, overall
  and per round-bucket, plus a 25%/50%/100%-of-TRAIN-games learning curve.
- Result:

  | Validation (569 states) | Cross-entropy | Brier | Top-1 accuracy |
  |---|---|---|---|
  | Uniform | 1.386 | 0.750 | 32.2% |
  | Net-worth leader | 1.238 | 0.179 | 91.0% |
  | Learned ValueProbe | 0.716 | 0.427 | 69.2% |

  The ValueProbe clearly beats uniform (real signal exists), but
  overfits hard (train accuracy 96.8% / CE 0.142 vs. validation 69.2% /
  CE 0.716) and loses to the trivial net-worth-leader baseline on **every**
  validation round-bucket:

  | Bucket | Leader accuracy | ValueProbe accuracy |
  |---|---|---|
  | 1-25 | 73.4% | 70.3% |
  | 26-50 | **100.0%** | 64.8% |
  | 51-100 | **100.0%** | 60.0% |
  | 101-150 | **100.0%** | 71.6% |
  | 151-terminal | **100.0%** | 82.1% |

  The leader baseline hitting exactly 100.0% accuracy in 4 of 5 validation
  buckets is a striking snowball/monotonicity signature: in this 64-game
  `POLICY_ONLY` self-play sample, whoever leads on net worth by round ~26
  essentially always goes on to win — not a degenerate dataset artifact
  (all 4 relative winner classes occur on both splits; validation class
  balance ranges 16.7%-32.2%). Learning curve (25%/50%/100% of the 48
  TRAIN games): validation CE 1.090 → 0.925 → 0.716, accuracy 44.1% →
  55.9% → 69.2% — still improving at 48 games with no visible plateau, so
  whether the ValueProbe's gap behind the leader baseline reflects a small
  training set or a real information ceiling in the state representation
  is not resolved by this experiment. Leakage guards held (0 seed overlap,
  0 state overlap); integrity clean (0 illegal, 0 crashes, 0 fallbacks — no
  fixed agents were even seated — 0 ASU modules loaded). Wall time 683.0s,
  peak RSS 0.35 GiB.

  Full structured record:
  [logs/experiments/020-monopolyzero-value-learnability-probe.json](../logs/experiments/020-monopolyzero-value-learnability-probe.json).

- Conclusion / next step: **No GO/KILL verdict is being called here** —
  per the task instructions this experiment measures only. The numbers are
  handed off as read results: the state representation carries a real but
  apparently hard-to-learn-from-2000-samples signal, while a one-line
  net-worth-leader heuristic already gets it almost exactly right from
  round 26 onward in this sample. Whether that means (a) more training
  data would close the gap, (b) the ValueProbe architecture/training
  procedure needs work, or (c) the current state encoding isn't the
  bottleneck at all and the leader-heuristic result says something more
  fundamental about this checkpoint's self-play dynamics, is for a human
  to decide from here — not pre-judged by this entry.

  **Corrected 2026-08-11 (later still x3), see `docs/DECISIONS.md`'s GO
  entry for `020`:** (1) the "VALIDATION" split above doubled as the
  early-stopping monitor, so its 69.2% accuracy / 0.716 CE is a
  model-selection number, not an unbiased held-out generalization
  estimate — `021` re-runs this with a proper train/selection/test split.
  (2) "loses to the trivial net-worth-leader baseline on every validation
  round-bucket" above is about *accuracy* specifically (true, per-bucket)
  and should not be read as "loses on every metric" — the learned
  `ValueProbe` actually beat the leader baseline on cross-entropy overall
  (0.716 vs. 1.238).

## 2026-08-11 (later still x4) — Value-generalization probe: an unbiased TEST read after fixing 020's two gaps

- Hypothesis: none pre-registered, purely descriptive. `020` showed the
  300-dim state carries *some* learnable full-horizon signal, but its
  "validation" split doubled as its own early-stopping monitor, so 69.2%
  accuracy wasn't a trustworthy estimate of generalization to unseen games
  — and its "first N" sampling could cluster all samples near the start of
  a round bucket. Does a proper train/selection/**test** split (test
  touched exactly once) and quantile-spread sampling change the picture?
- Setup: 96 fresh `POLICY_ONLY` self-play games (seeds `42100-42195`, newly
  DEV-registered, **not** reusing `020`'s 64 games), split 64 TRAIN / 16
  SELECTION / 16 TEST at the game level. Quantile-spread sampling
  (min/median/max of each game/seat/bucket's occurrences, not first-N) with
  full round provenance. Same `ValueProbe` architecture as `020`, fit on
  TRAIN, early-stopped on SELECTION, evaluated on TEST exactly once with
  the final 64-train-game model — never for early stopping, hyperparameter/
  temperature selection, or model choice. Added a **probabilistic**
  net-worth-leader baseline (temperature-scaled softmax over relative net
  worth, temperature grid-fit on TRAIN+SELECTION pooled, never TEST) since
  the plain hard-leader baseline is a degenerate predictor for
  cross-entropy purposes. Uncertainty for the learned-vs-probabilistic-leader
  comparison uses a **game-block** bootstrap (resamples whole TEST games,
  not states, since a game's ~8 sampled states are correlated) rather than
  treating state count as an independent sample size.
- Result:

  **Untouched TEST (16 games, 759 states, evaluated once):**

  | | Cross-entropy | Brier | Top-1 accuracy |
  |---|---|---|---|
  | Uniform | 1.386 | 0.750 | 30.0% |
  | Hard leader (accuracy-diagnostic only) | 0.510 | 0.074 | 96.3% |
  | **Probabilistic leader** (T=500) | **0.063** | **0.032** | **96.6%** |
  | Learned ValueProbe | 1.182 | 0.643 | 55.5% |

  This time the learned `ValueProbe` loses to **both** leader baselines on
  **every** metric, including cross-entropy — the opposite of `020`'s
  contaminated read, where the probe had actually won on CE. Per-bucket,
  the probabilistic leader hits 100.0% accuracy in 4 of 5 buckets (same
  snowball signature `020` found) while the `ValueProbe` stays in the
  43-63% range everywhere:

  | Bucket | Probabilistic leader acc. | ValueProbe acc. | Unique TEST games |
  |---|---|---|---|
  | 1-25 | 86.2% | 58.7% | 16 |
  | 26-50 | 100.0% | 43.1% | 16 |
  | 51-100 | 100.0% | 55.3% | 16 |
  | 101-150 | 100.0% | 57.8% | 14 |
  | 151-terminal | 100.0% | 62.9% | 14 |

  **Game-block bootstrap** (learned − probabilistic leader, 16 TEST games,
  2000 resamples): cross-entropy diff **+1.119** `[+0.747, +1.555]`, Brier
  diff **+0.612** `[+0.450, +0.762]`, accuracy diff **−0.411** `[−0.503,
  −0.309]` — all three 95% CIs exclude zero in the adverse direction, so
  this isn't sampling noise at the game level.

  **Quantile-spread sampling confirmed working**: TEST bucket round
  provenance spans full ranges (e.g. `151-terminal`: min 151, median 174.5,
  max 199; `1-25`: min 1, median 9, max 25) instead of clustering near a
  bucket's start.

  **Learning curve** (SELECTION only, all 3 points — TEST not repeatedly
  consumed): 16 games CE 1.105/acc 46.2%, 32 games CE 1.015/acc 52.3%, 64
  games CE 0.814/acc 69.2% — still improving, no plateau. Notably, that
  64-game **SELECTION** accuracy (69.2%) came out numerically identical to
  `020`'s old validation figure (69.2%) — while the same model's real
  **TEST** accuracy is only 55.5%. That gap is exactly what `020` couldn't
  see, since its validation split played both roles at once.

  Leakage guards: 0 seed/state overlap across all three split pairs.
  Integrity: 96/96 games clean (0 illegal, 0 crashes, 0 fallbacks, 0 ASU).
  Wall time 1636.2s, peak RSS 0.50 GiB.

  Full structured record:
  [logs/experiments/021-monopolyzero-value-generalization-probe.json](../logs/experiments/021-monopolyzero-value-generalization-probe.json).

- Conclusion / next step: **No GO/KILL verdict is being called here** —
  per the task instructions this experiment measures only. What it
  resolves from `020`: the true generalization picture is materially worse
  for the `ValueProbe` than `020` suggested (55.5% real TEST accuracy vs.
  the 69.2% model-selection number), and a simple probabilistic net-worth
  heuristic remains far ahead of the learned probe on every metric, with
  game-block-bootstrap-confirmed statistical support this time, not just a
  point estimate. Whether the fix is more training data (the SELECTION
  learning curve still hadn't plateaued at 64 games), a different probe
  architecture/objective, or accepting that a cheap net-worth-based
  heuristic already captures most of the accessible signal in this
  checkpoint's self-play dynamics, is for a human to decide from here.

## 2026-08-11 (later still x5) — Value decision audit: post-hoc segmentation of 021, and a final A/B call

- Hypothesis: none pre-registered for the segments themselves (explicitly
  diagnostic/exploratory) - but this entry, unlike `013`-`021`, DOES end
  with a real decision, not a "human decides" deferral: is there ANY
  segment of `021`'s TEST set where the learned `ValueProbe` meaningfully
  beats the probabilistic net-worth-leader baseline, that would justify
  proposing a new value hypothesis instead of dropping the learned-value
  path?
- Setup: `021` never persisted per-state records (only aggregates), so
  this re-derived `021`'s exact deterministic pipeline once - same seeds,
  same split, same quantile sampling, same `ValueProbe` training recipe,
  imported directly from `021`'s and `020`'s own modules rather than
  redefined - to recover per-state fields `021` never captured (legal-action
  count, decision phase/`env.phase`) plus two new derived fields (the
  probabilistic leader's own top1-vs-top2 margin, and the deciding
  player's current net-worth rank). Zero new self-play randomness, no
  new/different model, no PUCT, no new temperature/hyperparameter fit
  against TEST. Before drawing ANY conclusion, the four TEST predictor
  summaries were reconciled against `021`'s own logged values - refusing to
  proceed if they didn't match exactly.
- Result: **Reconciliation: exact.** All four predictors (uniform, hard
  leader, probabilistic leader, learned `ValueProbe`) matched `021`'s
  logged cross-entropy/Brier/accuracy with **zero delta** - this is
  genuinely `021`'s data, not a new experiment wearing its numbers.

  **Segment search: nothing rescues the `ValueProbe`.** Across every axis
  (round bucket, leader-margin quartile, current-player rank, decision
  type, legal-action count), `value_probe_advantage_segments_found` is
  **empty** against the pre-stated bar (≥20 states, ≥5-point accuracy
  margin, fixed in source before running):

  | Axis | Leader accuracy range | ValueProbe accuracy range |
  |---|---|---|
  | Round bucket | 86.2% – 100.0% | 43.1% – 62.9% |
  | Margin quartile | 86.3% – 100.0% | 43.2% – 77.4% |
  | Current-player rank | 94.9% – 98.3% | 34.8% – 73.2% |
  | Decision type | 88.0% – 98.8% | 51.6% – 62.0% |
  | Legal-action count (condensed) | 87.9% – 99.3% | 48.9% – 66.7% |

  The leader hits **exactly 100.0% accuracy** in every round bucket from
  26 onward and in 3 of 4 margin quartiles; the `ValueProbe` never breaks
  78% anywhere. **Where the leader is wrong** (26/759 TEST states, 3.4%):
  100% of those states fall in round bucket 1-25 AND 100% fall in margin
  quartile Q1(low margin) - the leader only ever misses in the earliest,
  least net-worth-differentiated part of the game, and is still right
  86.2% of the time even there. (Margin-quartile and current-player-rank
  are flagged **outcome-adjacent** - both derive from the same net-worth
  signal the leader is scored on, so their clean accuracy gradient is
  partly definitional, not fresh evidence.)

  Zero illegal actions, zero crashes, zero ASU. Wall time 456.7s, peak RSS
  0.49 GiB.

  Full structured record:
  [logs/experiments/022-monopolyzero-value-decision-audit.json](../logs/experiments/022-monopolyzero-value-decision-audit.json).

- **Final decision: A.** Drop the learned-value path for now; move to the
  decision/policy win-rate phase. Per the rule fixed before this audit ran
  (B only if a segment clears the pre-stated bar), no segment came close -
  the largest `ValueProbe`-vs-leader gap in the learned probe's favor
  across all five axes was still a deficit, not an advantage. Combined
  with `021`'s game-block-bootstrap-confirmed result (learned strictly
  worse on CE/Brier/accuracy, 95% CIs excluding zero), this closes
  `021`'s question with a **KILL** on the current learned-value-probe
  direction for this checkpoint/representation - see `docs/DECISIONS.md`.

## 2026-08-11 (later still x6) — Hybrid-PPO bootstrap isolation audit: does the BUY_PROPERTY/ACCEPT_TRADE carve-out cost POLICY_ONLY strength

- Hypothesis: `baseline_pretraining.pt`'s actor is bootstrapped
  (`MonopolyZeroNet.load_ppo_actor`) from a **hybrid** PPO checkpoint whose
  training loop hands `BUY_PROPERTY`/`ACCEPT_TRADE` to fixed rules and never
  gradient-updates those two action-head rows
  (`references/DeepRL_Monopoly/monopoly_game_engine/agent_ppo.py`'s
  `fixed_action_mask`). `load_ppo_actor` copies the policy head in full,
  with no gating carried over, and `build_local_policy_only`'s flat
  legal-masked argmax has no BUY/TRADE special case either - so does
  POLICY_ONLY inference silently let the neural head pick those two action
  types off untrained logits, and does that cost real strength?
- Bootstrap provenance (no games played for this part): the local
  `ppo_hybrid_2000_v2.pt` checkpoint's SHA-256 does **not** match
  `TRAINING_RESULTS.md`'s documented SHA for that filename - expected,
  since experiment `007` generated its own minimal 1-game/598-step PPO
  checkpoint at that exact path purely to satisfy
  `load_ppo_actor`'s format/metadata check, never reproducing upstream's
  full training run. It **does** match `007`'s own logged SHA (confirmed:
  same artifact, not silently changed). The checkpoint's own payload,
  read at runtime, confirms `hybrid: true`, `games_trained: 1`,
  `step_count: 598` - the only numbers used to characterize "how trained"
  it is; no broader "untrained weights" claim is made. Independently, the
  `fixed_action_mask` lines were grepped verbatim from the pinned
  reference source (not inferred from a docstring): `BUY_PROPERTY`'s and
  `ACCEPT_TRADE`'s actor rows never receive a PPO gradient update in
  hybrid mode, regardless of games trained - they sit at random
  initialization in this checkpoint and after `load_ppo_actor`'s
  ungated full copy into `baseline_pretraining.pt`.
- Setup: a diagnostic-only `HYBRID_COMPAT` policy
  (`monopolyzero_common.build_local_hybrid_compat_policy`) restores the
  original hybrid-PPO carve-out on top of otherwise-plain POLICY_ONLY
  inference, by runtime-importing `fixed_buy_decision`/
  `fixed_accept_trade_decision` straight from the reference's
  `agent_ppo.py` (never copied). Registered new DEV seeds `43000`-`43019`.
  Clean paired screen: BASELINE (focus seat POLICY_ONLY) vs. CANDIDATE
  (focus seat `HYBRID_COMPAT`), other 3 seats POLICY_ONLY in both arms, 20
  seeds x 4 focus-seat rotation = 80 games/arm, `max_rounds=200`, zero
  fixed opponents (zero fallback-contamination risk), zero ASU. An
  integrity gate ran first: `play_local_game`'s `shadow_policy` hook
  queried a fresh POLICY_ONLY instance on the literal same pre-step state
  as every `HYBRID_COMPAT` focus-seat decision, and the script would have
  stopped before reporting anything on any disagreement outside a flagged
  BUY/TRADE opportunity.
- Result: **Isolation integrity: PASS**, 0 violations across all 80
  candidate games' 96,629 non-forced focus-seat decisions - BASELINE and
  CANDIDATE are isolated to exactly the fixed-action carve-out.

  **Intervention audit:** 8,173 of 96,629 decisions (8.5%) were BUY/TRADE
  opportunities (2,616 `BUY_PROPERTY`-legal, 5,557 genuine incoming-trade
  responses). At every single one, plain POLICY_ONLY's own argmax
  **never once** chose `BUY_PROPERTY` or `ACCEPT_TRADE`
  (0/2616, 0/5557 - 0.0% chosen-action frequency for both), despite
  assigning them non-trivial mean probability (7.9% buy, 10.7% accept).
  The untrained logits are competitive but never win the legal-set argmax
  against this checkpoint's heavily-trained alternatives at these
  decisions - so POLICY_ONLY's actual failure mode here is **"never buy
  property, never accept a trade,"** not random erratic picks.

  **Paired strength screen** (80 games/arm):

  | | BASELINE (POLICY_ONLY) | CANDIDATE (HYBRID_COMPAT) |
  |---|---|---|
  | Win rate | 25.0% (Wilson [16.8%, 35.5%]) | 46.25% (Wilson [35.7%, 57.1%]) |
  | Bankruptcy rate | 41.25% | 16.25% |
  | Mean net worth | 6,386.6 | 8,479.7 |
  | Round-cap rate | 75% | 85% |

  Both PRIMARY seed-block statistics agree and exclude zero: the paired
  randomization test's observed mean win-rate diff is **+0.2125**
  (p=0.000122, exact enumeration over 2^20 sign patterns), and the
  seed-block bootstrap's win-rate-diff 95% CI is **[0.1375, 0.2875]**
  (net-worth diff +2,093.09, 95% CI [1,020.47, 3,272.17]). Secondary
  seat-level McNemar (b=21, c=4, p=0.00091) is directionally consistent
  but not relied on alone (clustered seats).

  Zero illegal actions, zero crashes, zero fallbacks (no fixed agents),
  zero ASU. Wall time 436.8s, peak RSS 0.28 GiB.

  Full structured record:
  [logs/experiments/023-hybrid-bootstrap-isolation-audit.json](../logs/experiments/023-hybrid-bootstrap-isolation-audit.json).

- No promotion/GO-KILL verdict computed by the script itself, per this
  task's own instructions - the numbers above are for a human to read.

## 2026-08-11 (later still x7) — Hybrid decomposition audit: BUY vs. TRADE contribution, crippled vs. repaired peers

- Hypothesis: `023`'s +21.25-point `HYBRID_COMPAT` win-rate improvement
  bundles two independent fixes (`BUY_PROPERTY` and `ACCEPT_TRADE`). How
  much of that improvement does each contribute alone, do they combine
  additively or with synergy, and does the effect survive when the
  opponents are not also crippled `POLICY_ONLY` peers?
- Setup: `build_local_hybrid_compat_policy` was made configurable
  (`enable_buy`/`enable_trade`, both default `True` - unchanged `023`
  behavior) so BUY_ONLY/TRADE_ONLY/BOTH/NEITHER differ in nothing but
  which fixed-rule branch may fire. Reused `023`'s exact seeds
  (`43000`-`43019`, no new DEV registration), checkpoint, and module
  (imported as `audit_v1`, not redefined). **Context 1** (crippled peers,
  `023`'s exact setup): `023`'s `POLICY_ONLY`/`BOTH` arms were
  deterministically regenerated and reconciled bit-for-bit against `023`'s
  own logged values before anything else ran - `023`'s log only persisted
  aggregates, not the per-game records this decomposition's paired stats
  need; `BUY_ONLY`/`TRADE_ONLY` ran fresh. **Context 2** (repaired peers):
  all 4 arms fresh against 3 `HYBRID_COMPAT(BOTH)` peers instead of
  crippled `POLICY_ONLY` ones; the `BOTH` arm used one self-play game per
  seed (20, not 80) since its focus seat's policy config is identical to
  its peers'.
- Result: **Reconciliation: exact.** Context 1's regenerated
  `POLICY_ONLY`/`BOTH` arms matched `023`'s logged values with **zero
  delta** on every deterministic field.

  **Context 1 (crippled peers) - win rate:**

  | Arm | Win rate | vs. baseline | Randomization p | Bootstrap 95% CI |
  |---|---|---|---|---|
  | POLICY_ONLY (baseline) | 25.0% | — | — | — |
  | BUY_ONLY | 41.25% | +16.25pp | 0.00098 | [10.0, 22.5]pp |
  | TRADE_ONLY | 31.25% | +6.25pp | 0.0625 | [1.25, 11.25]pp |
  | BOTH | 46.25% | +21.25pp | 0.000122 | [13.75, 28.75]pp |

  `BUY_ONLY` alone recovers **76.5%** of `BOTH`'s win-rate improvement
  (92.8% of its net-worth improvement); `TRADE_ONLY` alone recovers
  **29.4%**. The two PRIMARY stats disagree on `TRADE_ONLY`'s
  significance: randomization p=0.0625 exceeds the conventional 0.05, the
  bootstrap CI barely excludes zero - reported as-is, no cherry-pick.
  `BUY_ONLY`'s +16.25pp plus `TRADE_ONLY`'s +6.25pp sum to +22.5pp, close
  to `BOTH`'s actual +21.25pp (delta −1.25pp) - **approximately additive**.

  **Context 2 (repaired peers) - win rate:**

  | Arm | Win rate | vs. baseline | Randomization p | Bootstrap 95% CI |
  |---|---|---|---|---|
  | POLICY_ONLY (baseline) | 7.5% | — | — | — |
  | BUY_ONLY | 11.25% | +3.75pp | 0.590 | [−5.0, 12.5]pp |
  | TRADE_ONLY | 10.0% | +2.5pp | 0.625 | [−2.5, 7.5]pp |
  | BOTH | 25.0% | +17.5pp | 0.00052 | [11.25, 22.5]pp |

  Against tougher, non-crippled opponents, every arm's win rate drops
  sharply. **Neither `BUY_ONLY` nor `TRADE_ONLY` alone clears statistical
  significance at this sample size (n=20 seed blocks)** - both CIs cross
  zero. Only `BOTH` does. `BOTH`'s +17.5pp substantially exceeds the sum
  of the two individual effects (+6.25pp, delta +11.25pp) - **super-additive
  / positive synergy**: against competent opponents, having both fixes
  together appears to matter far more than either alone, though neither
  alone is individually confirmed at this n.

  Intervention rates (share of non-forced focus-seat decisions where the
  carve-out fired) ranged 1.9%-10.3% across arms/contexts, highest for
  `BOTH` in both contexts (the only arm where either carve-out can fire on
  a given trajectory). Zero illegal actions, zero crashes, zero incomplete
  games across all 580 physical games (320 context 1 + 260 context 2).
  Wall time 7,379.5s (~2.05h), peak RSS 0.39 GiB.

  Full structured record:
  [logs/experiments/024-hybrid-decomposition-audit.json](../logs/experiments/024-hybrid-decomposition-audit.json).

- No promotion/GO-KILL verdict computed by the script itself, per this
  task's own instructions - the numbers above are for a human to read.

## 2026-08-12 — Learnability sweep: does native ASU-free training actually move BUY_PROPERTY/ACCEPT_TRADE?

- Hypothesis: `023`/`024` showed a fixed-rule `HYBRID_COMPAT` carve-out for
  `BUY_PROPERTY`/`ACCEPT_TRADE` measurably improves win rate against a
  `POLICY_ONLY` baseline. Before investing in a larger native (ASU-free)
  training run, does gradient descent over native MCTS-visit targets
  (`scripts/monopolyzero_native_train_candidate.py`, no ASU, no
  `HYBRID_COMPAT`, no fixed rule anywhere) actually move the *trained
  policy's own* `BUY_PROPERTY`/`ACCEPT_TRADE` behavior at all, at small
  update counts?
- Setup: `--reuse-replay` training-only mode - no new self-play, no new
  search, no seed consumed - four independent runs at 0/25/50/100 updates,
  each starting fresh from the same `baseline_pretraining.pt` (never
  chained), each reading the exact same on-disk replay (286 positions: 14
  `BUY_PROPERTY` opportunities, 40 `ACCEPT_TRADE` opportunities). Verified
  deterministic: the 25-update run's per-update losses are byte-identical
  to the 50-update run's first 25, and the 50-update run's to the
  100-update run's first 50. **Caveat surfaced, not hidden:** running this
  project's own existing test suite as a pre-flight check regenerated the
  on-disk replay via that suite's own fixed deterministic recipe (seed
  43000, already inside the registered DEV pool - no new seed) before this
  sweep started; position count matched the pre-existing replay exactly.
  Full detail in the JSON log's `algorithm_config.replay_provenance_caveat`.
- Result: Training **does** move `BUY_PROPERTY`/`ACCEPT_TRADE` prior mass
  and rank away from the 0-update baseline, but weakly and non-monotonically,
  and it mostly does not flip the greedy (argmax) decision:

  | Updates | BUY greedy rate | BUY mean prior | BUY mean rank | ACCEPT greedy rate | ACCEPT mean prior | ACCEPT mean rank |
  |---|---|---|---|---|---|---|
  | 0 | 0.0% | 0.222 | 2.93 | 0.0% | 0.133 | 6.53 |
  | 25 | 0.0% | 0.141 | 2.71 | 0.0% | 0.057 | 7.35 |
  | 50 | 0.0% | 0.186 | 2.50 | 0.0% | 0.075 | 4.85 |
  | 100 | **7.1%** (1/14) | 0.166 | 2.64 | **2.5%** (1/40) | 0.062 | 4.68 |

  Greedy rate for both actions stays exactly 0% through 50 updates and
  only reaches 1 of 14 (`BUY_PROPERTY`) / 1 of 40 (`ACCEPT_TRADE`)
  opportunities by 100 updates. Raw prior mass for **both** actions ends
  *lower* at 100 updates than at 0 updates, and `BUY_PROPERTY`'s prior
  dips at 25 updates before partially recovering - not a clean trend
  toward the target action. Meanwhile `value_loss` collapses to
  near-zero (1.335 → 0.010) far faster than `policy_loss` falls
  (1.413 → 1.033) on this fixed 286-position replay (batch size 32, so
  100 updates is >11x pass-equivalent exposure) - a small-dataset
  value-head-overfitting signature, not evidence the policy head is
  robustly learning these two actions. Zero ASU modules loaded across all
  four runs, zero illegal actions/crashes, zero new self-play games
  generated by this sweep itself.

  Full structured record:
  [logs/experiments/025-native-train-buy-trade-learnability-sweep.json](../logs/experiments/025-native-train-buy-trade-learnability-sweep.json).

- **No GO/KILL claim.** With only 14 `BUY_PROPERTY` and 40 `ACCEPT_TRADE`
  opportunities total, from a single short self-play game, one greedy
  flip in each category is low-power evidence. The honest read: some
  movement is visible, but it's small, non-monotonic, mostly below the
  greedy-decision threshold, and measured on a replay too small to be
  conclusive either way. A larger self-play replay (more seeds, more
  positions) is needed before this question has adequate statistical
  power - out of scope for this task (no new self-play data, no large
  strength evaluation, per instructions).
