"""Evaluation methodology shared across future paired-evaluation scripts:
DEV/PROMOTION/FINAL_BLIND seed-pool classification, and paired statistics
(per-arm Wilson interval, exact McNemar on same-seed+seat paired outcomes,
seed-block paired bootstrap for win-rate/net-worth difference CIs).

Not tied to any single model family or to monopoly_bench/ASU concerns —
pure Python + numpy, no engine/reference imports. See
docs/EVALUATION_PROTOCOL.md for the methodology this implements and the
full provenance of every seed range below.

Replaces "non-overlapping Wilson intervals" as the promotion test (see
docs/DECISIONS.md's 2026-08-11 "later still" entries): Wilson intervals
here stay purely descriptive per-arm; McNemar's exact test and the paired
bootstrap are the actual paired-comparison statistics. This module
deliberately computes no promotion/GO/KILL boolean of its own — that
call is made by a human reading the numbers, same discipline as every
other diagnostic in this project.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

# ── Seed pools ───────────────────────────────────────────────────────────
#
# DEV: every seed range already consumed by a training run, paired
# evaluation, or diagnostic in this project (logs/experiments/001-019),
# plus general iterative-development use. Safe to reuse freely during
# ordinary development/debugging - already "seen".
#
# PROMOTION: fresh, never-run seeds. Only spent on a candidate that DEV-based
# iteration already suggests is genuinely promising - not for routine dev.
#
# FINAL_BLIND: fresh, never-run seeds, reserved exclusively for the final
# model-selection read. Must never be touched by any run before that.
#
# DEV_SEED_RANGES sourced directly from each experiment log's own "seeds"
# field (logs/experiments/*.json), except where noted: 013's top-level
# "seeds" field only recorded [42, 0] (the training RNG seed and an unused
# reserved value), not its actual 32 per-game seeds - those are documented
# in 013's own algorithm_config note ("10_000 + index" / "20_000 + index"
# for index in range(16)) and are included below for that reason.

DEV_SEED_RANGES: tuple[tuple[int, int, str], ...] = (
    (42, 42, "001/002/003/004/007/010/011/012/013/015 - recurring global/training seed"),
    (0, 0, "013 - reserved value, logged but unconsumed by that script"),
    (20000, 20000, "006 - ASU evaluation-only benchmark"),
    (23, 23, "008/009 - MonopolyZero inference/PUCT-runtime smoke"),
    (101, 101, "008/009 - MonopolyZero inference/PUCT-runtime smoke"),
    (501, 503, "010/011/012 - self-play training-plumbing smoke"),
    (10000, 10009, "005 - DDQN 20-vs-500 paired evaluation held-out seeds"),
    (10000, 10015, "013 - self-play game generation (algorithm_config, not top-level seeds field)"),
    (20000, 20015, "013 - vs-fixed game generation (algorithm_config, not top-level seeds field)"),
    (30000, 30009, "014/016 - MonopolyZero strength-pilot / update-budget-sweep paired eval"),
    (31000, 31004, "017 - PUCT search-budget diagnostic"),
    (32000, 32009, "018 - POLICY_ONLY vs PUCT_4 paired eval"),
    (40000, 40015, "019 - horizon diagnostic self-play games"),
    (41000, 41015, "019 - horizon diagnostic vs-fixed games"),
)

# Fresh, disjoint from DEV_SEED_RANGES and from each other. Reserved here,
# not yet run by anything (verified by the disjointness test in
# tests/test_evaluation_protocol.py).
PROMOTION_SEED_RANGE: tuple[int, int] = (50000, 50049)
FINAL_BLIND_SEED_RANGE: tuple[int, int] = (90000, 90049)


def _expand_ranges(ranges: Iterable[tuple[int, int, ...]]) -> frozenset[int]:
    expanded: set[int] = set()
    for entry in ranges:
        lo, hi = entry[0], entry[1]
        expanded.update(range(lo, hi + 1))
    return frozenset(expanded)


DEV_SEEDS: frozenset[int] = _expand_ranges(DEV_SEED_RANGES)
PROMOTION_SEEDS: frozenset[int] = _expand_ranges([PROMOTION_SEED_RANGE])
FINAL_BLIND_SEEDS: frozenset[int] = _expand_ranges([FINAL_BLIND_SEED_RANGE])

SEED_CLASS_DEV = "dev"
SEED_CLASS_PROMOTION = "promotion"
SEED_CLASS_FINAL_BLIND = "final_blind"
SEED_CLASS_UNCLASSIFIED = "unclassified"


def classify_seed(seed: int) -> str:
    if seed in DEV_SEEDS:
        return SEED_CLASS_DEV
    if seed in PROMOTION_SEEDS:
        return SEED_CLASS_PROMOTION
    if seed in FINAL_BLIND_SEEDS:
        return SEED_CLASS_FINAL_BLIND
    return SEED_CLASS_UNCLASSIFIED


def require_non_final_blind(seeds: Iterable[int], *, context: str) -> None:
    """Guard for any 'normal' (non-final-selection) evaluation entrypoint:
    call this before running and it refuses if any seed is in
    FINAL_BLIND_SEEDS. FINAL_BLIND is reserved exclusively for the final
    model-selection read - it must never be spent by routine dev/promotion
    evaluation runs, accidentally or otherwise."""
    hits = sorted({seed for seed in seeds if seed in FINAL_BLIND_SEEDS})
    if hits:
        raise RuntimeError(
            f"{context} refuses to run: FINAL_BLIND seed(s) {hits} must not be "
            "consumed by a normal evaluation. FINAL_BLIND is reserved for the "
            "final model-selection read only - see docs/EVALUATION_PROTOCOL.md."
        )


# ── Per-arm descriptive statistic ───────────────────────────────────────


def wilson_95_interval(wins: int, games: int) -> tuple[float, float]:
    """Standard closed-form Wilson score interval (Wilson, 1927). Purely
    descriptive here - no promotion decision is made from this alone (see
    module docstring)."""
    if games <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    phat = wins / games
    denominator = 1 + z * z / games
    center = phat + z * z / (2 * games)
    margin = z * math.sqrt(phat * (1 - phat) / games + z * z / (4 * games * games))
    return ((center - margin) / denominator, (center + margin) / denominator)


# ── Paired exact McNemar test ───────────────────────────────────────────


def _binomial_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value: sum of all pmf values no
    larger than pmf(k), the standard exact-test definition (matches R's
    binom.test / scipy's binomtest default two-sided method)."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    threshold = pmf[k] * (1 + 1e-9)
    return min(1.0, sum(prob for prob in pmf if prob <= threshold))


def mcnemar_exact(paired_outcomes: Sequence[tuple[bool, bool]]) -> dict:
    """Exact McNemar's test via the binomial distribution on discordant
    pairs. `paired_outcomes` is a sequence of (baseline_win, candidate_win)
    booleans for the SAME seed+seat decision - i.e. genuinely paired data,
    not independent samples. b = candidate-only wins (discordant pairs
    favoring the candidate), c = baseline-only wins (discordant pairs
    favoring the baseline); concordant pairs (both win or both lose) carry
    no information under McNemar and are excluded from n."""
    b = sum(1 for baseline_win, candidate_win in paired_outcomes if candidate_win and not baseline_win)
    c = sum(1 for baseline_win, candidate_win in paired_outcomes if baseline_win and not candidate_win)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}
    p_value = _binomial_two_sided_p(min(b, c), n)
    return {"b": b, "c": c, "n_discordant": n, "p_value": p_value}


# ── Seed-block paired bootstrap ─────────────────────────────────────────


def paired_seed_block_bootstrap(
    records: Sequence[dict],
    *, n_resamples: int = 2000, bootstrap_seed: int = 0,
) -> dict:
    """Seed-block paired bootstrap for win-rate and net-worth differences.

    `records`: each dict needs {"seed", "baseline_win", "candidate_win",
    "baseline_net_worth", "candidate_net_worth"} for one paired seed+seat
    decision. Resampling is done at the SEED level (all records sharing a
    seed are resampled together as one block, with replacement) rather than
    per-record, to preserve whatever within-seed correlation the paired
    seat-rotation design creates - resampling individual records would
    treat seat-rotated outcomes on the same board as independent, which
    they are not.

    Deterministic: same `records` + same `bootstrap_seed` always produces
    the same CI (uses numpy's Generator, not global RNG state).
    """
    import numpy as np

    if not records:
        empty = {"point": None, "ci_95": None}
        return {
            "win_rate_diff": empty,
            "net_worth_diff": dict(empty),
            "n_seed_blocks": 0,
            "n_records": 0,
            "n_resamples": n_resamples,
            "bootstrap_seed": bootstrap_seed,
        }

    blocks: dict[int, list[dict]] = {}
    for record in records:
        blocks.setdefault(record["seed"], []).append(record)
    seed_list = sorted(blocks)
    n_blocks = len(seed_list)

    def _diffs(recs: list[dict]) -> tuple[float, float]:
        n = len(recs)
        win_rate_diff = (
            sum(1 for r in recs if r["candidate_win"]) / n
            - sum(1 for r in recs if r["baseline_win"]) / n
        )
        net_worth_diff = (
            sum(r["candidate_net_worth"] for r in recs) / n
            - sum(r["baseline_net_worth"] for r in recs) / n
        )
        return win_rate_diff, net_worth_diff

    point_win_diff, point_nw_diff = _diffs(list(records))

    rng = np.random.default_rng(bootstrap_seed)
    win_diffs = np.empty(n_resamples)
    nw_diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        sampled_indices = rng.integers(0, n_blocks, size=n_blocks)
        resampled: list[dict] = []
        for idx in sampled_indices:
            resampled.extend(blocks[seed_list[idx]])
        win_diffs[i], nw_diffs[i] = _diffs(resampled)

    return {
        "win_rate_diff": {
            "point": point_win_diff,
            "ci_95": [float(np.percentile(win_diffs, 2.5)), float(np.percentile(win_diffs, 97.5))],
        },
        "net_worth_diff": {
            "point": point_nw_diff,
            "ci_95": [float(np.percentile(nw_diffs, 2.5)), float(np.percentile(nw_diffs, 97.5))],
        },
        "n_seed_blocks": n_blocks,
        "n_records": len(records),
        "n_resamples": n_resamples,
        "bootstrap_seed": bootstrap_seed,
    }


# ── Fallback contamination flag ─────────────────────────────────────────


def fallback_contamination(baseline_fallbacks: int, candidate_fallbacks: int) -> dict:
    """A separate, explicit flag/metric - never silently folded into a win
    rate. Any nonzero fallback count means at least one non-focus seat's
    real scripted decision was replaced by a substitute action, so that
    arm's result is not a clean read of the checkpoint under test."""
    return {
        "baseline_fallbacks": baseline_fallbacks,
        "candidate_fallbacks": candidate_fallbacks,
        "total_fallbacks": baseline_fallbacks + candidate_fallbacks,
        "contaminated": bool(baseline_fallbacks or candidate_fallbacks),
    }


def paired_evaluation_summary(
    *,
    baseline_wins: int,
    baseline_games: int,
    candidate_wins: int,
    candidate_games: int,
    paired_outcomes: Sequence[tuple[bool, bool]],
    bootstrap_records: Sequence[dict],
    baseline_fallbacks: int,
    candidate_fallbacks: int,
    n_resamples: int = 2000,
    bootstrap_seed: int = 0,
) -> dict:
    """Assembles the full paired-evaluation report: per-arm Wilson (purely
    descriptive), exact McNemar on the paired outcomes, the seed-block
    bootstrap CIs, and the fallback-contamination flag. Deliberately does
    NOT compute a promote/GO/KILL boolean - Wilson non-overlap is no longer
    treated as the promotion test itself (see module docstring); a human
    reads these numbers and decides."""
    return {
        "baseline_wilson_95": list(wilson_95_interval(baseline_wins, baseline_games)),
        "candidate_wilson_95": list(wilson_95_interval(candidate_wins, candidate_games)),
        "mcnemar": mcnemar_exact(paired_outcomes),
        "bootstrap": paired_seed_block_bootstrap(
            bootstrap_records, n_resamples=n_resamples, bootstrap_seed=bootstrap_seed
        ),
        "fallback_contamination": fallback_contamination(baseline_fallbacks, candidate_fallbacks),
    }
