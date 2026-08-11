# Experiment Logs

Standardized, structured, comparable log of every meaningful training,
benchmark, reproducibility-check, A/B-test, or self-play/replay run in this
project. This is the machine-readable companion to `docs/EXPERIMENTS.md`
(which stays the prose narrative — hypothesis, setup, reasoning); this
directory is the queryable/diffable record.

## Standard

- One JSON file per experiment: `NNN-kebab-case-slug.json`, zero-padded
  three-digit sequence number, in this directory (`logs/experiments/`).
- Must validate against [`schema.json`](schema.json) — see
  `tests/test_experiment_logs_schema.py` for the automated check that runs
  in CI/locally.
- Unknown or unmeasured values use JSON `null`. Never guess or estimate a
  number into a field that should hold a measured value — if it wasn't
  measured, it's `null`.
- Small raw stdout/stderr text logs (a few KB, not checkpoints/binaries) live
  in [`raw/`](raw/) alongside the JSON entries, named to match the
  experiment id (e.g. `002-ddqn-20-game-training-smoke_stdout.txt`), and are
  referenced by repo-relative path in the JSON's `raw_logs` array.
- **Never commit checkpoints, replay buffers, or other large/binary
  artifacts here or anywhere in this repo.** Reference them by SHA-256 (in
  `model_checkpoint.sha256`) and local path (`model_checkpoint.path`, which
  will not exist on another machine — that's expected, the hash is the
  portable identity). Large-but-small JSON result payloads (e.g. full
  evaluation output with per-game breakdowns) can stay where they already
  are under `docs/baseline_runs/` and be referenced by path in `raw_logs`;
  they aren't duplicated into this directory.
- This format is **mandatory** for every training run, benchmark,
  reproducibility check, A/B test, or replay-derived result from now on —
  see the rule in `CLAUDE.md`. Write the JSON log as part of the same
  commit that records the experiment in `docs/EXPERIMENTS.md`, not
  separately or later.

## Fields

See `schema.json` for the authoritative, validated definition. Summary:

| Field | Meaning |
|---|---|
| `experiment_id` | Matches the filename, `NNN-slug` |
| `experiment_name` | Human-readable title |
| `experiment_type` | One of a fixed enum (see schema) |
| `date` | `YYYY-MM-DD` |
| `code_commit_sha` | This repo's commit that recorded the experiment |
| `reference_submodule_sha` | Pinned `references/DeepRL_Monopoly` SHA at the time |
| `exact_commands` | Exact shell command(s), including env vars |
| `seeds` | Seed(s) used |
| `algorithm_config` | Free-form: algorithm, hyperparameters, opponent pool, ruleset |
| `model_checkpoint` | `{sha256, path, format_version}`, all `null` if no checkpoint involved |
| `counts` | Free-form: games, rounds, decisions, positions/samples, updates |
| `runtime` | `{wall_time_s, peak_rss_gib}` |
| `metrics` | Win/loss, win rates, Wilson intervals, net worth, etc., or `null` |
| `loss` | Training loss figures, or `null` |
| `integrity` | `{illegal_actions, fallbacks, crashes}` |
| `result` | One-paragraph conclusion — no unsupported improvement claims |
| `raw_logs` | Repo-relative paths to raw output, or `null` |
| `notes` | Optional free-text caveats |

## Backfilled history

Entries `001`-`010` backfill every verified experiment run before this
standard existed (2026-08-10 through 2026-08-11), reconstructed from
`docs/EXPERIMENTS.md`, `docs/BASELINE.md`, and the raw JSON/`.txt` artifacts
already committed under `docs/baseline_runs/`. Where a figure was never
actually measured (e.g. peak RSS wasn't polled for the very first engine
smoke), the field is `null` — not reconstructed or estimated.
