# Colab Runbook

How to run this project's evaluation/self-play code (`scripts/colab_shard_runner.py`
and `scripts/colab_merge_shards.py`) on a **fresh** Google Colab Free/Pro
runtime — from a cold "Runtime > Disconnect and delete runtime" state, with
nothing pre-installed and nothing assumed about the local filesystem.

**Runtime type: CPU is sufficient and recommended.** This project runs no
GPU-accelerated code (pure CPU inference/self-play), and CPU keeps the
existing determinism guarantees (seeded games, `PYTHONHASHSEED=0`) the
simplest to reason about — there is no reason to pay for/wait on a GPU
runtime here.

**Do not paste `PYTHONHASHSEED=0` into a plain Python cell as
`os.environ["PYTHONHASHSEED"] = "0"` and expect it to take effect** —
Python only reads that variable once, at interpreter *startup*; setting it
from inside an already-running process (the notebook's own kernel) does
nothing to that process's hash randomization. Every cell below that needs
it uses a `!`-prefixed shell command (`!PYTHONHASHSEED=0 python ...`),
which sets the variable for the **new** subprocess before it starts —
that's the only form that actually works, mirrored exactly from how
`monopolyzero_common.require_pinned_hash_seed` is used everywhere else in
this project.

## 1. Clone the repo

```python
!git clone https://github.com/soyleremo3/monopoly-champion-agent.git
%cd monopoly-champion-agent
```

`%cd` (a notebook magic, not `!cd`) is required so the working directory
persists into later cells — `!cd x` only affects that one shell subprocess.

## 2. Check out the exact commit

Pin to whichever commit you actually intend to run — do not rely on
whatever `main` happens to be at clone time. As of the commit that added
this runbook, the immediately preceding commit was
`e00c330b02b63390956d38fadff37ce9ed3984c1` — that value is already stale
by the time you read this (this doc's own commit, and everything after
it, moved `main` forward). Get the real one to use with
`git log --oneline -5` on your own machine, or use whatever SHA whoever
asked you to run this specifies:

```python
!git checkout <COMMIT_SHA>
```

## 3. Initialize the submodule

```python
!git submodule update --init --recursive
```

This pulls `references/DeepRL_Monopoly` at whatever SHA this repo's commit
pins it to (currently `afd9205761317e196d77f679921c35fb04c7ab96` — verify
with `!git -C references/DeepRL_Monopoly rev-parse HEAD` after this cell;
`colab_shard_runner.py` also records this SHA itself in every shard's
`metadata.json`, so it doesn't need to be hand-verified here, just sanity
checked once).

## 4. Install dependencies (single cell)

```python
!pip install -r requirements-colab.txt -q
```

`requirements-colab.txt` pins the exact versions this project's own dev
environment uses (`torch==2.13.0` CPU-only, `numpy==2.5.2`,
`psutil==7.2.2`, `jsonschema==4.26.0`, `pytest==9.1.1`) via
`--extra-index-url https://download.pytorch.org/whl/cpu` for the CPU-only
torch wheel, everything else from PyPI as normal.

## 5. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 6. Copy `baseline_pretraining.pt` from Drive to the artifact path

`artifacts/` is gitignored in this repo (checkpoints are never committed —
see `docs/EVALUATION_PROTOCOL.md`), so the checkpoint has to come from
somewhere else. Upload `baseline_pretraining.pt` to your own Drive first
(same file used by experiments `020`-`024` locally), then:

```python
import pathlib

dest_dir = pathlib.Path("artifacts/monopolyzero_strength_pilot")
dest_dir.mkdir(parents=True, exist_ok=True)

# EDIT this path to wherever you put the file in your own Drive:
DRIVE_CHECKPOINT_PATH = "/content/drive/MyDrive/monopoly-champion-agent/baseline_pretraining.pt"

!cp "{DRIVE_CHECKPOINT_PATH}" artifacts/monopolyzero_strength_pilot/baseline_pretraining.pt
```

The destination is exactly the path every script in this repo already
expects (`monopolyzero_common.REPO_ROOT / "artifacts" / "monopolyzero_strength_pilot"
/ "baseline_pretraining.pt"`, computed from `__file__` at import time — it
self-adjusts to wherever the repo was cloned, `/content/monopoly-champion-agent`
here, no path to edit in any script).

## 7. Verify the checkpoint SHA-256

```python
!sha256sum artifacts/monopolyzero_strength_pilot/baseline_pretraining.pt
```

Expected: `22ae2abea60478355f61ce0404a31de3b52fb97769d4f99f7e04f4c673629370`
(the same value every prior experiment log, `020` onward, records as
`model_checkpoint.sha256`). If it doesn't match, stop here — every script
below re-verifies this automatically at startup and will refuse to run on
a mismatch anyway, but checking it explicitly first saves a wasted Colab
session if the wrong file got uploaded.

## 8. Run the unit + smoke tests (optional but recommended once per fresh runtime)

```python
!PYTHONHASHSEED=0 python -m pytest tests/ -q
```

Confirms the fresh runtime's dependency versions and the checkout are
actually consistent with what this repo expects before spending compute on
a real shard.

## 9. 20-game benchmark

Always run this before committing to a shard size — per-game cost varies
with `context`/`arm` and with whatever CPU Colab happens to hand you this
session.

```python
!PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python scripts/colab_shard_runner.py \
  --seed-start 43000 --arm both --context repaired --benchmark-games 20
```

Uses seeds from the `43000`-`43019` range already spent by experiments
`023`/`024` (reusable here — a benchmark run produces no experiment-log
data and isn't scored, so replaying already-used seeds costs nothing).
**The `44000`-`44999` pool is reserved for real DEV shard data** (step 10)
— don't burn seeds from it on throwaway benchmark timing runs.

Prints `physical_games_completed`, `seat_records_completed`, and
`sec_per_physical_game`/`projected_seconds` for 100/500/1000 games — read
those before picking `--seed-count` for the real run below.
`sec_per_physical_game` is keyed off actual `play_local_game` calls, not
seat-records: in self-play-optimized mode (`context=repaired`,
`arm=both`) one physical game yields 4 seat-records, so a naive
records-based rate would look 4x faster than reality. Swap `--arm`/
`--context` to match what you actually intend to run; benchmark numbers
differ meaningfully between them (repaired-peer/HYBRID_COMPAT arms cost
roughly 1.5-2x a plain POLICY_ONLY decision — see `024`'s experiment log
for measured examples).

## 10. Real shard run

Pick a seed sub-range from the reserved Colab pool (`44000`-`44999`,
registered in `scripts/evaluation_protocol.py`'s `DEV_SEED_RANGES` — never
touch `PROMOTION_SEED_RANGE`/`FINAL_BLIND_SEED_RANGE`). Running several
Colab tabs/sessions in parallel as separate shards: give each one a
disjoint `--seed-start`/`--seed-count` slice of that pool.

```python
!PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 python scripts/colab_shard_runner.py \
  --seed-start 44000 --seed-count 50 --arm both --context repaired \
  --output-dir /content/drive/MyDrive/monopoly-champion-agent/shards/shard_44000_50 \
  --resume
```

Writing `--output-dir` directly onto the mounted Drive (rather than
`/content`) means progress already survives a runtime disconnect without a
separate copy step — `per_game.jsonl` is fsync'd after every completed
seed. `--resume` is always safe to pass, including on the very first
invocation (there's nothing to resume from yet, so it's a no-op then);
passing it on every run means a Colab disconnect mid-shard only costs the
one seed that was in flight, never re-plays finished ones.

If a runtime disconnects, reconnect, redo cells 1-8 (a fresh runtime has
nothing installed), then rerun this exact same cell (same `--seed-start`/
`--seed-count`/`--arm`/`--context`/`--output-dir`, `--resume` still set) —
it picks up where it left off. Resuming with a *different* arm/context/
checkpoint/commit than the original run is refused loudly (`RuntimeError:
--resume refuses to continue: shard config differs...`), not silently
mixed in.

To spread one run across multiple Colab sessions running in parallel,
repeat this cell in each session with a disjoint `--seed-start`/
`--seed-count` and its own `--output-dir` — e.g. a second session running
`--seed-start 44050 --seed-count 50 --output-dir .../shards/shard_44050_50`
alongside the one above covers seeds `44000`-`44099` between the two,
which is exactly what the merge example below expects.

## 11. Copy results to Drive

Skippable if `--output-dir` already pointed at Drive in step 10 (recommended).
Otherwise, if you ran with a local `/content` output dir:

```python
!cp -r /content/monopoly-champion-agent/<local-output-dir> \
  /content/drive/MyDrive/monopoly-champion-agent/shards/
```

## Outputs, per shard

Every `--output-dir` ends up with:

- `per_game.jsonl` — one JSON object per game (`seed`, `focus_seat`,
  `completed`, `winner`, `focus_won`, `rounds`, `decisions`,
  `focus_net_worth`, `round_capped`); appended and `fsync`'d after every
  completed seed, never after only part of a seed.
- `metadata.json` — `git_head_sha`, `checkpoint_sha256`, `submodule_sha`,
  `arm`, `context`, `seed_start`, `seed_count`, `seeds`,
  `self_play_optimized`, `games_per_seed`, `max_rounds`, plus
  `finished_status`/`elapsed_s_this_invocation` once the shard completes.
- `summary.json` — the full structured per-arm result (win rate, Wilson
  interval, bankruptcy rate, mean/median net worth, round-cap rate,
  per-seat wins), rebuilt from the complete `per_game.jsonl` every time the
  script exits (so it's always in sync, even mid-resume).
- `run_log.txt` — a plain-text mirror of everything printed to stdout
  (per-seed progress lines, illegal/crash/incomplete flags), written
  incrementally so it survives even if the notebook's own cell output gets
  truncated or lost across a disconnect.

## Merging shards

Once every shard for a run has finished (check each shard's
`metadata.json.finished_status == "OK"` first):

```python
!PYTHONHASHSEED=0 python scripts/colab_merge_shards.py \
  --shard-dirs /content/drive/MyDrive/monopoly-champion-agent/shards/shard_44000_50 \
               /content/drive/MyDrive/monopoly-champion-agent/shards/shard_44050_50 \
  --output-dir /content/drive/MyDrive/monopoly-champion-agent/shards/merged \
  --expected-seed-start 44000 --expected-seed-count 100
```

`--expected-seed-start`/`--expected-seed-count` are optional but
recommended — with them, the merge additionally verifies the combined seed
coverage exactly equals that declared full range (no gaps, no extras), not
just that the shards you happened to point it at are mutually consistent.
Without them, it still refuses (raises, does not silently proceed) on any
of: a shard missing games for a seed it declared, a shard with games for
an undeclared seed, two shards claiming an overlapping seed, shards whose
`arm`/`context`/`checkpoint_sha256`/`git_head_sha`/`games_per_seed`/
`max_rounds` disagree, or any duplicate `(seed, seat)` pair in the merged
result.

## No local-path assumptions, by construction

Every path in `scripts/colab_shard_runner.py`, `scripts/colab_merge_shards.py`,
and everything they import (`monopolyzero_common.py`,
`monopolyzero_hybrid_decomposition_audit.py`, `evaluation_protocol.py`) is
derived from `Path(__file__).resolve().parents[N]` at import time, not
hardcoded — the checkpoint path, the reference submodule path, and the
`REPO_ROOT` used for the git-clean-tree check all self-locate relative to
wherever the repo actually was cloned. Nothing in this repo references
`C:\`, `/home/<user>`, or any path specific to this project's own Windows
dev machine.
