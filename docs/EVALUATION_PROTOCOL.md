# Evaluation Protocol

Locked 2026-08-11, alongside `docs/DECISIONS.md`'s "Deprecate the 013
replay for strength training" entry. Governs how this project selects
seeds for training/evaluation/diagnostics, and how a candidate checkpoint
gets compared against a baseline, from now on. Implemented in
`scripts/evaluation_protocol.py`; see that module's docstring for the
runnable functions this document describes.

## Why this exists

Every paired evaluation run so far (`005`, `014`, `016`, `018`) used
"non-overlapping Wilson intervals" as the de facto promotion test — did the
candidate's win-rate confidence interval clear the baseline's. That's a
single, fairly weak signal: it throws away the *pairing* (same seed, same
seat, same opponent draw) that a held-out-seed design actually gives you
for free, and a single boolean threshold invites exactly the kind of
overclaiming `CLAUDE.md` and this log already guard against elsewhere. This
document replaces that ad-hoc practice with two things: a disciplined seed
pool split (so "held-out" actually means something), and a proper paired
statistic (McNemar's exact test + a seed-block bootstrap) that uses the
pairing instead of discarding it.

## Seed pools

Three disjoint pools (verified disjoint by
`tests/test_evaluation_protocol.py::test_dev_promotion_final_blind_pools_are_pairwise_disjoint`):

| Pool | Purpose | Reuse policy |
|---|---|---|
| **DEV** | Every seed already consumed by a training run, paired evaluation, or diagnostic in this project, plus general iterative-development use (writing/debugging a new script, sanity-checking a change). | Freely reusable — it's already "seen", so nothing about held-out validity is lost by reusing it during ordinary development. |
| **PROMOTION** | Fresh, never-run seeds. | Spent *once* on a candidate that DEV-based iteration already suggests is genuinely promising — not for routine dev, not for "let's just check." |
| **FINAL_BLIND** | Fresh, never-run seeds, reserved exclusively for the final model-selection read. | Must never be touched by anything before that single final read — see the guard below. |

### DEV registry (as of this document; sourced from each experiment log's
own `seeds` field, `logs/experiments/001-019`)

| Range | Source | Note |
|---|---|---|
| `42` | `001`/`002`/`003`/`004`/`007`/`010`/`011`/`012`/`013`/`015` | recurring global/training RNG seed |
| `0` | `013` | reserved value, logged but unconsumed |
| `20000` | `006` | ASU evaluation-only benchmark |
| `23`, `101` | `008`/`009` | MonopolyZero inference/PUCT-runtime smoke |
| `501`-`503` | `010`/`011`/`012` | self-play training-plumbing smoke |
| `10000`-`10009` | `005` | DDQN 20-vs-500 paired eval held-out seeds |
| `10000`-`10015` | `013` | self-play game generation (documented in `013`'s `algorithm_config`, not its top-level `seeds` field — that field only logged `[42, 0]` for that run, a logging gap in that entry, not a reason to treat these seeds as unused) |
| `20000`-`20015` | `013` | vs-fixed game generation (same note as above) |
| `30000`-`30009` | `014`/`016` | MonopolyZero strength-pilot / update-budget-sweep paired eval |
| `31000`-`31004` | `017` | PUCT search-budget diagnostic |
| `32000`-`32009` | `018` | POLICY_ONLY vs PUCT_4 paired eval |
| `40000`-`40015`, `41000`-`41015` | `019` | horizon diagnostic (self-play / vs-fixed) |

All of the above are unioned into `evaluation_protocol.DEV_SEEDS`. Any new
DEV-scope run should extend `DEV_SEED_RANGES` in that module (with a
provenance note) rather than picking ad-hoc numbers.

### PROMOTION and FINAL_BLIND ranges

- `PROMOTION_SEED_RANGE = (50000, 50049)` — 50 fresh seeds, never run by
  anything as of this document.
- `FINAL_BLIND_SEED_RANGE = (90000, 90049)` — 50 fresh seeds, deliberately
  far from every other pool, never run by anything as of this document.

Neither range has been consumed by any script or experiment in this repo.
**No experiment may run against either range except the specific promotion
or final-selection event it's reserved for** — this document's existence
is the record of that constraint; `evaluation_protocol.require_non_final_blind()`
is the code-level enforcement for the FINAL_BLIND half of it (see below).

### The FINAL_BLIND guard

`evaluation_protocol.require_non_final_blind(seeds, context=...)` raises
`RuntimeError` if any seed in the given list falls in `FINAL_BLIND_SEEDS`.
Any future "normal" (DEV or PROMOTION-scope) evaluation entrypoint should
call this before running, the same way every script in this project already
calls `common.require_pinned_hash_seed()` / `common.require_clean_git_tree()`
before doing anything else. There is deliberately no equivalent guard
against PROMOTION seeds being used in DEV runs — PROMOTION seeds are meant
to be spent rarely, not never, and the cost of a wasted PROMOTION seed is
much lower than the cost of ever touching FINAL_BLIND before the final read.

## Paired evaluation statistics

`scripts/evaluation_protocol.py` provides:

- **`wilson_95_interval(wins, games)`** — the same closed-form Wilson
  interval this project has always used, kept as a purely **descriptive**
  per-arm statistic. It is explicitly no longer the promotion test itself.
- **`mcnemar_exact(paired_outcomes)`** — exact McNemar's test (via the
  binomial distribution on discordant pairs) over same-seed+seat
  baseline/candidate win pairs. Uses the pairing directly: only decision
  states where the two arms *disagree* on the outcome carry information.
- **`paired_seed_block_bootstrap(records, n_resamples, bootstrap_seed)`** —
  resamples whole seeds (blocks), not individual records, with replacement,
  to get 95% CIs on the win-rate difference and the net-worth difference.
  Block-level (not record-level) resampling preserves whatever
  within-seed/seat-rotation correlation the paired design creates.
  Deterministic given the same `bootstrap_seed` (uses a `numpy.random.Generator`
  instance, never global RNG state).
- **`fallback_contamination(baseline_fallbacks, candidate_fallbacks)`** — a
  separate, explicit flag. A vs-fixed arm with nonzero fixed-agent
  fallbacks (as `019`'s vs-fixed subgroup had — 90 fallbacks across 16
  games) is not a clean read of the checkpoint under test and must be
  reported as such, never silently folded into a win rate.
- **`paired_evaluation_summary(...)`** — assembles all of the above into one
  report. **It computes no promotion/GO/KILL boolean of its own.** The
  numbers (McNemar p-value, bootstrap CIs, per-arm Wilson, contamination
  flag) are handed to a human to read and decide, the same discipline this
  project already applies to every other diagnostic (see e.g. `017`'s and
  `019`'s "no arbitrary threshold" framing in `docs/EXPERIMENTS.md`).

## What this document does not do

It does not run anything. No new game, training run, or evaluation was
executed to produce this document or `scripts/evaluation_protocol.py` — see
`docs/DECISIONS.md`'s entry for this methodology change. The next script
that runs a paired evaluation should import from `evaluation_protocol`
rather than reimplementing Wilson-only comparison logic from scratch.
