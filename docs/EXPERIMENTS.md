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
