# Evaluation Protocol

Locked 2026-08-11, alongside `docs/DECISIONS.md`'s "Deprecate the 013
replay for strength training" entry; tightened the same day (methodology
fix: scope-exclusive seed guard, cluster-aware primary statistic, explicit
input validation — see "Revision (same day)" below). Governs how this
project selects seeds for training/evaluation/diagnostics, and how a
candidate checkpoint gets compared against a baseline, from now on.
Implemented in `scripts/evaluation_protocol.py`; see that module's
docstring for the runnable functions this document describes.

## Revision (same day) — two methodological gaps fixed

1. **Seed guard was FINAL_BLIND-only, not scope-exclusive.**
   `require_non_final_blind()` only ever checked for FINAL_BLIND seeds — a
   DEV-scoped run could still consume a PROMOTION seed uncaught (and vice
   versa), quietly eroding what "held out for promotion" means.
   `require_seed_scope(seeds, scope, context)` replaces it as the standard
   for new code: a `DEV`-scoped call accepts only `DEV_SEEDS`, `PROMOTION`
   only `PROMOTION_SEEDS`, `FINAL_BLIND` only `FINAL_BLIND_SEEDS` — an
   unclassified seed, or a seed from any *other* scope, fails every check.
   `require_non_final_blind()` is kept for backward compatibility only.
2. **The primary paired statistic treated seats as independent trials.**
   The original `mcnemar_exact`-as-primary design implicitly assumed every
   (seed, seat) pair was an independent Bernoulli trial. It isn't: the 4
   seats rotated through one seed share a single board draw (property
   distribution, opponent behavior on that board) — they're a **cluster**,
   not 4 independent samples, and treating them as independent understates
   the true variance. McNemar is now explicitly **secondary/diagnostic
   only**, always surfaced under the `seat_level_mcnemar_secondary` key.
   The **primary** paired-comparison evidence is now
   `seed_block_paired_randomization_test` (a deterministic sign-flip test
   over per-*seed* win-rate differences) plus the existing seed-block
   bootstrap — both treat the seed, not the seat, as the unit of
   exchangeability/resampling.

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
| `42000`-`42063` | `020` | value-learnability probe: 64 clean POLICY_ONLY self-play games (baseline checkpoint, all 4 seats, zero fixed agents) |

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

### The seed-scope guard

`evaluation_protocol.require_seed_scope(seeds, scope, context=...)` is the
standard guard for any future evaluation entrypoint — call it before doing
anything else, the same way every script in this project already calls
`common.require_pinned_hash_seed()` / `common.require_clean_git_tree()`.
`scope` is one of `evaluation_protocol.SEED_CLASS_DEV`,
`SEED_CLASS_PROMOTION`, or `SEED_CLASS_FINAL_BLIND`; the call raises
`RuntimeError` naming every seed that isn't in exactly that scope's pool
(including a partial-violation batch — one bad seed among otherwise-fine
ones still fails the whole call, nothing is silently filtered down to the
safe subset). This makes each pool's isolation symmetric: a DEV run can no
more consume a PROMOTION seed than a PROMOTION run can consume FINAL_BLIND.

`require_non_final_blind(seeds, context=...)` still exists and still works
(raises only on FINAL_BLIND seeds) but is kept for backward compatibility
only — it does not catch a PROMOTION seed leaking into a DEV run. New code
should call `require_seed_scope` instead.

## Record pairing and input validation

Every stat below consumes *paired* records, not raw per-arm outcomes.
**`pair_records(baseline_records, candidate_records, expected_seats=4)`**
builds that pairing from two flat lists (each record: `{"seed", "seat",
"win", "net_worth"}`) and fails loudly rather than silently dropping
anything:

- a duplicate `(seed, seat)` within either arm's own records → `RuntimeError`
- a `(seed, seat)` present in one arm but missing from the other (an
  incomplete baseline/candidate pairing) → `RuntimeError`
- if `expected_seats` is given (default `4`), any seed whose seat set
  doesn't have exactly that many distinct seats → `RuntimeError`

`paired_evaluation_summary` calls `pair_records` first and additionally
re-checks that both arms' reported game counts equal the paired-record
count — structurally guaranteed by `pair_records` already, kept as an
explicit, self-documenting invariant rather than an implicit one.

## Paired evaluation statistics

`scripts/evaluation_protocol.py` provides, grouped the same way
`paired_evaluation_summary`'s output is grouped:

**Descriptive (not a test):**

- **`wilson_95_interval(wins, games)`** — the same closed-form Wilson
  interval this project has always used, kept as a purely descriptive
  per-arm statistic.

**Primary (the actual paired-comparison evidence — both seed-block, i.e.
both treat one seed's 4 rotated seats as one cluster/unit, not 4
independent samples):**

- **`seed_block_paired_randomization_test(paired_records, n_resamples,
  randomization_seed)`** — for each seed, computes the candidate-minus-
  baseline win-rate difference across that seed's seats (one number per
  seed block); the observed statistic is the mean of those per-seed
  differences. Under the null (no systematic difference), each block's
  sign is exchangeable, so the null distribution comes from flipping each
  block's sign independently: **exact enumeration** of all `2^n` sign
  patterns when there are at most 20 seed blocks, otherwise a
  **deterministic Monte Carlo** sample (seeded `numpy.random.Generator`,
  same `randomization_seed` always reproduces the same p-value). Reports
  the two-sided p-value and the observed mean win-rate difference.
- **`paired_seed_block_bootstrap(records, n_resamples, bootstrap_seed)`** —
  resamples whole seeds (blocks), not individual records, with replacement,
  to get 95% CIs on the win-rate difference and the net-worth difference.
  Deterministic given the same `bootstrap_seed`.

**Secondary (diagnostic cross-check only — do not use alone):**

- **`mcnemar_exact(paired_outcomes)`** — exact McNemar's test (via the
  binomial distribution on discordant pairs) over same-seed+seat
  baseline/candidate win pairs. Implicitly treats each pair as an
  independent trial, which the 4-seats-per-seed design violates (see the
  "Revision" section above) — always surfaced as
  `seat_level_mcnemar_secondary`, never as standalone promotion evidence.

**Always reported separately:**

- **`fallback_contamination(baseline_fallbacks, candidate_fallbacks)`** — a
  separate, explicit flag. A vs-fixed arm with nonzero fixed-agent
  fallbacks (as `019`'s vs-fixed subgroup had — 90 fallbacks across 16
  games) is not a clean read of the checkpoint under test and must be
  reported as such, never silently folded into a win rate.

**`paired_evaluation_summary(baseline_records=..., candidate_records=...,
baseline_fallbacks=..., candidate_fallbacks=..., ...)`** assembles all of
the above (pairing/validation included) into one report shaped
`{"descriptive": {...}, "primary": {...}, "secondary": {...},
"fallback_contamination": {...}}`. **It computes no promotion/GO/KILL
boolean of its own, anywhere in that structure.** The numbers are handed
to a human to read and decide, the same discipline this project already
applies to every other diagnostic (see e.g. `017`'s and `019`'s "no
arbitrary threshold" framing in `docs/EXPERIMENTS.md`).

## What this document does not do

It does not run anything. No new game, training run, or evaluation was
executed to produce this document or `scripts/evaluation_protocol.py` — see
`docs/DECISIONS.md`'s entry for this methodology change. The next script
that runs a paired evaluation should import from `evaluation_protocol`
rather than reimplementing Wilson-only comparison logic from scratch.
