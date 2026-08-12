"""Evaluation methodology shared across future paired-evaluation scripts:
DEV/PROMOTION/FINAL_BLIND seed-pool classification with a scope-exclusive
guard, input-validated baseline/candidate record pairing, and paired
statistics (per-arm Wilson interval as descriptive-only, a seed-block
sign-flip randomization test + seed-block bootstrap as the PRIMARY paired
inference, and an exact seat-level McNemar test kept as SECONDARY/diagnostic
only - see the clustering note on that function).

Not tied to any single model family or to monopoly_bench/ASU concerns -
pure Python + numpy, no engine/reference imports. See
docs/EVALUATION_PROTOCOL.md for the methodology this implements and the
full provenance of every seed range below.

This module deliberately computes no promotion/GO/KILL boolean of its own
anywhere - that call is made by a human reading the numbers, same
discipline as every other diagnostic in this project.
"""

from __future__ import annotations

import math
from itertools import product
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
    (42000, 42063, "020 - value-learnability probe: 64 clean POLICY_ONLY self-play games"),
    (42100, 42195, "021 - value-generalization probe: 96 clean POLICY_ONLY self-play games (64 train / 16 selection / 16 test)"),
    (43000, 43019, "023 - hybrid-PPO BUY_PROPERTY/ACCEPT_TRADE fixed-action isolation screen: 20 seeds x 4 focus-seat rotation, POLICY_ONLY vs HYBRID_COMPAT paired games"),
    (44000, 44999, "Colab infra (docs/COLAB_RUNBOOK.md, scripts/colab_shard_runner.py) - reserved pool for future Colab-run shards; not yet consumed by a specific experiment as of registration"),
    (45000, 45031, "027 - pure PPO (hybrid=False) BUY_PROPERTY/ACCEPT_TRADE learnability gate: 32-game training run, player_id=0 vs FPAgentA/B/C"),
    (46000, 46019, "028 - pure PPO (hybrid=False) 027-candidate-vs-own-untrained-baseline strength screen: 20 seeds x 4 focus-seat rotation, deterministic masked-argmax PPO actor, candidate vs 3 copies of the untrained baseline (no fixed/ASU opponents)"),
    (47000, 47095, "030 - 027 checkpoint resumed 32->128 games: 96 additional training episode seeds (train()'s own episode_seed=seed_arg+absolute_game-1 formula, seed_arg=46968 chosen so this range results exactly)"),
    (48000, 48019, "030 - 128-game candidate vs 32-game baseline strength test: 20 seeds x 4 focus-seat rotation"),
    (49000, 49007, "031 - 32->128 early-stop checkpoint search: selection screen, 8 seeds x 4 focus-seat rotation, common set across all 5 candidates (48/64/80/96/112) vs 32-game champion"),
    (49100, 49119, "031 - early-stop checkpoint search: confirmation of the selected candidate, 20 seeds x 4 focus-seat rotation, vs 32-game champion"),
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

_SCOPE_POOLS: dict[str, frozenset[int]] = {
    SEED_CLASS_DEV: DEV_SEEDS,
    SEED_CLASS_PROMOTION: PROMOTION_SEEDS,
    SEED_CLASS_FINAL_BLIND: FINAL_BLIND_SEEDS,
}


def classify_seed(seed: int) -> str:
    if seed in DEV_SEEDS:
        return SEED_CLASS_DEV
    if seed in PROMOTION_SEEDS:
        return SEED_CLASS_PROMOTION
    if seed in FINAL_BLIND_SEEDS:
        return SEED_CLASS_FINAL_BLIND
    return SEED_CLASS_UNCLASSIFIED


def require_seed_scope(seeds: Iterable[int], scope: str, *, context: str) -> None:
    """The standard guard for any new evaluation entrypoint: every seed
    passed in must belong to EXACTLY the declared scope's pool - a DEV-scope
    call cannot consume a PROMOTION or FINAL_BLIND seed, a PROMOTION-scope
    call cannot consume a DEV or FINAL_BLIND seed, and an unclassified seed
    (never registered in any pool) fails every scope. This is stricter than
    `require_non_final_blind` below, which only ever checked the
    FINAL_BLIND half of this - new code should call this instead."""
    if scope not in _SCOPE_POOLS:
        raise ValueError(f"require_seed_scope: unknown scope {scope!r}, must be one of {sorted(_SCOPE_POOLS)}")
    allowed = _SCOPE_POOLS[scope]
    violations = sorted({seed for seed in seeds if seed not in allowed})
    if violations:
        raise RuntimeError(
            f"{context} refuses to run: seed(s) {violations} are not in the "
            f"{scope!r} pool. A {scope!r}-scoped run may only consume seeds "
            f"from that exact pool (unclassified seeds and seeds belonging "
            f"to a different scope both fail this check) - see "
            f"docs/EVALUATION_PROTOCOL.md."
        )


def require_non_final_blind(seeds: Iterable[int], *, context: str) -> None:
    """Kept for backward compatibility (only checks the FINAL_BLIND half of
    require_seed_scope's guarantee - a DEV run could still pass a
    PROMOTION seed through this one uncaught). New evaluation code should
    call `require_seed_scope` instead, which is scope-exclusive."""
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
    descriptive here - not the paired-comparison test (see module
    docstring and `seed_block_paired_randomization_test` below)."""
    if games <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    phat = wins / games
    denominator = 1 + z * z / games
    center = phat + z * z / (2 * games)
    margin = z * math.sqrt(phat * (1 - phat) / games + z * z / (4 * games * games))
    return ((center - margin) / denominator, (center + margin) / denominator)


# ── Record pairing + validation ─────────────────────────────────────────


def pair_records(
    baseline_records: Sequence[dict],
    candidate_records: Sequence[dict],
    *, expected_seats: int | None = 4,
) -> list[dict]:
    """Pairs flat per-arm game outcomes by (seed, seat) into the paired
    record shape every stat function below consumes. Each input record
    needs {"seed", "seat", "win" (bool), "net_worth" (float)}.

    Fails loudly (RuntimeError) rather than silently dropping/ignoring
    anything:
      - a duplicate (seed, seat) within either arm's own records
      - a (seed, seat) present in one arm but not the other (incomplete
        baseline/candidate pairing)
      - if `expected_seats` is given, any seed whose seat set doesn't have
        exactly that many distinct seats
    """

    def _index(records: Sequence[dict], label: str) -> dict[tuple[int, int], dict]:
        index: dict[tuple[int, int], dict] = {}
        for record in records:
            key = (record["seed"], record["seat"])
            if key in index:
                raise RuntimeError(f"pair_records: duplicate seed+seat {key} in {label} records")
            index[key] = record
        return index

    baseline_index = _index(baseline_records, "baseline")
    candidate_index = _index(candidate_records, "candidate")

    baseline_keys = set(baseline_index)
    candidate_keys = set(candidate_index)
    missing_in_candidate = sorted(baseline_keys - candidate_keys)
    missing_in_baseline = sorted(candidate_keys - baseline_keys)
    if missing_in_candidate or missing_in_baseline:
        raise RuntimeError(
            "pair_records: incomplete baseline/candidate pairing - "
            f"seed+seat present in baseline but missing in candidate: "
            f"{missing_in_candidate}; present in candidate but missing in "
            f"baseline: {missing_in_baseline}"
        )

    if expected_seats is not None:
        seats_by_seed: dict[int, set[int]] = {}
        for seed, seat in baseline_keys:
            seats_by_seed.setdefault(seed, set()).add(seat)
        bad = {seed: sorted(seats) for seed, seats in seats_by_seed.items() if len(seats) != expected_seats}
        if bad:
            raise RuntimeError(
                f"pair_records: expected exactly {expected_seats} seats per "
                f"seed, got a mismatched seat set for seed(s): {bad}"
            )

    paired: list[dict] = []
    for seed, seat in sorted(baseline_keys):
        baseline_record = baseline_index[(seed, seat)]
        candidate_record = candidate_index[(seed, seat)]
        paired.append(
            {
                "seed": seed,
                "seat": seat,
                "baseline_win": bool(baseline_record["win"]),
                "candidate_win": bool(candidate_record["win"]),
                "baseline_net_worth": float(baseline_record["net_worth"]),
                "candidate_net_worth": float(candidate_record["net_worth"]),
            }
        )
    return paired


# ── PRIMARY: seed-block paired randomization (sign-flip) test ─────────
#
# Treats each SEED as the unit of exchangeability (not each seat/game): the
# 4 seats rotated through the same seed share a board draw and are not
# independent trials, so the null hypothesis this test encodes is "the sign
# of each seed-block's candidate-minus-baseline win-rate difference is
# equally likely to be + or -", not "each seat's win/loss is an independent
# coin flip" (that second, weaker/wrong assumption is what seat-level
# McNemar implicitly makes - see its docstring below for why it's demoted
# to secondary).

EXACT_ENUMERATION_MAX_BLOCKS = 20


def _seed_block_win_rate_diffs(paired_records: Sequence[dict]) -> tuple[list[int], list[float]]:
    blocks: dict[int, list[dict]] = {}
    for record in paired_records:
        blocks.setdefault(record["seed"], []).append(record)
    seed_list = sorted(blocks)
    diffs = []
    for seed in seed_list:
        recs = blocks[seed]
        m = len(recs)
        candidate_rate = sum(1 for r in recs if r["candidate_win"]) / m
        baseline_rate = sum(1 for r in recs if r["baseline_win"]) / m
        diffs.append(candidate_rate - baseline_rate)
    return seed_list, diffs


def seed_block_paired_randomization_test(
    paired_records: Sequence[dict],
    *, n_resamples: int = 20000, randomization_seed: int = 0,
    exact_max_blocks: int = EXACT_ENUMERATION_MAX_BLOCKS,
) -> dict:
    """Deterministic paired sign-flip randomization test over seed blocks.

    For each seed, computes the candidate-minus-baseline win-rate
    difference across that seed's seats (one number per seed - the block
    unit). The observed statistic is the mean of these per-seed
    differences. Under the null (no systematic baseline/candidate
    difference), each block's sign is exchangeable, so the null
    distribution is built by flipping each block's sign independently:
    exact enumeration of all 2^n sign patterns when there are at most
    `exact_max_blocks` seeds (feasible up to n=20, 2^20 ~ 1e6 patterns),
    otherwise a deterministic Monte Carlo sample of `n_resamples` random
    sign patterns from a seeded `numpy.random.Generator` (same
    `randomization_seed` always reproduces the same p-value).

    Two-sided p-value: fraction of null-distribution statistics with
    |statistic| >= |observed statistic| (float-tolerant comparison).
    """
    seed_list, diffs = _seed_block_win_rate_diffs(paired_records)
    n = len(diffs)
    if n == 0:
        return {
            "n_seed_blocks": 0, "observed_mean_win_rate_diff": None,
            "method": None, "n_null_samples": 0,
            "randomization_seed": None, "p_value": None,
        }

    observed_stat = sum(diffs) / n

    if n <= exact_max_blocks:
        method = "exact_enumeration"
        null_stats = [sum(sign * diff for sign, diff in zip(signs, diffs)) / n for signs in product((1.0, -1.0), repeat=n)]
        used_randomization_seed = None
    else:
        import numpy as np

        method = "monte_carlo"
        rng = np.random.default_rng(randomization_seed)
        diffs_arr = np.asarray(diffs)
        signs_matrix = rng.choice(np.array([1.0, -1.0]), size=(n_resamples, n))
        null_stats = (signs_matrix * diffs_arr).mean(axis=1).tolist()
        used_randomization_seed = randomization_seed

    tolerance = 1e-9
    observed_abs = abs(observed_stat)
    extreme = sum(1 for stat in null_stats if abs(stat) >= observed_abs - tolerance)
    p_value = extreme / len(null_stats)

    return {
        "n_seed_blocks": n,
        "observed_mean_win_rate_diff": observed_stat,
        "method": method,
        "n_null_samples": len(null_stats),
        "randomization_seed": used_randomization_seed,
        "p_value": p_value,
    }


# ── PRIMARY: seed-block paired bootstrap ────────────────────────────────


def paired_seed_block_bootstrap(
    records: Sequence[dict],
    *, n_resamples: int = 2000, bootstrap_seed: int = 0,
) -> dict:
    """Seed-block paired bootstrap for win-rate and net-worth differences.

    `records`: paired records (see `pair_records`) - each needs {"seed",
    "baseline_win", "candidate_win", "baseline_net_worth",
    "candidate_net_worth"}. Resampling is done at the SEED level (all
    records sharing a seed are resampled together as one block, with
    replacement) rather than per-record, to preserve whatever within-seed
    correlation the paired seat-rotation design creates - resampling
    individual records would treat seat-rotated outcomes on the same board
    as independent, which they are not.

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


# ── PRIMARY (single-arm designs): seed-block bootstrap vs. a fixed null ─
#
# For designs where there is no separate baseline ARM played on the same
# seeds (e.g. one candidate seat embedded in a game otherwise filled with
# copies of a reference policy, rotated across all 4 seats) - there is
# nothing to pair against per `pair_records`. The natural comparison point
# instead is a fixed null win rate (1/num_seats for a symmetric-skill null),
# and the seed is still the correct resampling block: the 4 seat-rotation
# games sharing one seed share a board draw and are not independent trials,
# same reasoning as `paired_seed_block_bootstrap` above.


def seed_block_bootstrap_vs_null(
    records: Sequence[dict],
    *, null_rate: float, n_resamples: int = 2000, bootstrap_seed: int = 0,
) -> dict:
    """Seed-block bootstrap for (win_rate - null_rate), single arm.

    `records`: one row per physical game, each needs {"seed", "win" (bool)}
    - e.g. one row per focus-seat rotation. Resampling is done at the SEED
    level (all records sharing a seed are resampled together as one block,
    with replacement), matching `paired_seed_block_bootstrap`'s rationale.

    Deterministic: same `records` + same `bootstrap_seed` always produces
    the same CI (uses numpy's Generator, not global RNG state).
    """
    import numpy as np

    if not records:
        return {
            "win_rate_diff_from_null": {"point": None, "ci_95": None},
            "null_rate": null_rate,
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

    def _win_rate_diff(recs: list[dict]) -> float:
        n = len(recs)
        return sum(1 for r in recs if r["win"]) / n - null_rate

    point_diff = _win_rate_diff(list(records))

    rng = np.random.default_rng(bootstrap_seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        sampled_indices = rng.integers(0, n_blocks, size=n_blocks)
        resampled: list[dict] = []
        for idx in sampled_indices:
            resampled.extend(blocks[seed_list[idx]])
        diffs[i] = _win_rate_diff(resampled)

    return {
        "win_rate_diff_from_null": {
            "point": point_diff,
            "ci_95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        },
        "null_rate": null_rate,
        "n_seed_blocks": n_blocks,
        "n_records": len(records),
        "n_resamples": n_resamples,
        "bootstrap_seed": bootstrap_seed,
    }


# ── SECONDARY: seat-level exact McNemar (diagnostic only) ──────────────


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
    (baseline_win, candidate_win) pairs, one pair per seat-level decision.

    SECONDARY / diagnostic only - NOT valid as standalone promotion
    evidence. McNemar's test assumes each pair is an independent trial, but
    in this project's paired-evaluation design the 4 seats rotated through
    the same seed share one board draw (same property distribution, same
    opponent behavior on that board) - they are a CLUSTER, not 4
    independent samples. Treating them as independent understates the
    true variance and can make this test look more significant than the
    data actually supports. `seed_block_paired_randomization_test` and
    `paired_seed_block_bootstrap` (both seed-block, cluster-aware) are the
    PRIMARY paired-comparison statistics; this function is kept only as a
    secondary, seat-level cross-check, always labeled
    `seat_level_mcnemar_secondary` in `paired_evaluation_summary`'s output.
    """
    b = sum(1 for baseline_win, candidate_win in paired_outcomes if candidate_win and not baseline_win)
    c = sum(1 for baseline_win, candidate_win in paired_outcomes if baseline_win and not candidate_win)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}
    p_value = _binomial_two_sided_p(min(b, c), n)
    return {"b": b, "c": c, "n_discordant": n, "p_value": p_value}


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


# ── Top-level report ────────────────────────────────────────────────────


def paired_evaluation_summary(
    *,
    baseline_records: Sequence[dict],
    candidate_records: Sequence[dict],
    baseline_fallbacks: int,
    candidate_fallbacks: int,
    expected_seats: int | None = 4,
    n_resamples: int = 2000,
    bootstrap_seed: int = 0,
    n_randomization_samples: int = 20000,
    randomization_seed: int = 0,
) -> dict:
    """Assembles the full paired-evaluation report from raw per-arm records
    (each {"seed", "seat", "win", "net_worth"}).

    Validates the baseline/candidate pairing (see `pair_records`) before
    computing anything, then additionally re-checks that both arms'
    reported game counts match the paired-record count - structurally
    guaranteed by `pair_records` already, kept here too as an explicit,
    self-documenting invariant rather than an implicit one.

    Returns:
      - `descriptive`: per-arm Wilson 95% intervals only - not a test.
      - `primary`: seed-block paired randomization test + seed-block
        bootstrap win-rate/net-worth CIs - the actual paired-comparison
        evidence, both cluster-aware (seed = block).
      - `secondary`: seat-level exact McNemar - diagnostic only, see
        `mcnemar_exact`'s docstring for why it's not primary evidence.
      - `fallback_contamination`: explicit contamination flag.

    Deliberately computes NO promotion/GO/KILL boolean anywhere in this
    structure - a human reads these numbers and decides.
    """
    paired = pair_records(baseline_records, candidate_records, expected_seats=expected_seats)
    if len(baseline_records) != len(paired) or len(candidate_records) != len(paired):
        raise RuntimeError(
            "paired_evaluation_summary: baseline/candidate game counts do "
            f"not match the paired record count (baseline={len(baseline_records)}, "
            f"candidate={len(candidate_records)}, paired={len(paired)})"
        )

    baseline_wins = sum(1 for r in baseline_records if r["win"])
    baseline_games = len(baseline_records)
    candidate_wins = sum(1 for r in candidate_records if r["win"])
    candidate_games = len(candidate_records)

    paired_outcomes = [(r["baseline_win"], r["candidate_win"]) for r in paired]

    return {
        "descriptive": {
            "baseline_wilson_95": list(wilson_95_interval(baseline_wins, baseline_games)),
            "candidate_wilson_95": list(wilson_95_interval(candidate_wins, candidate_games)),
        },
        "primary": {
            "seed_block_paired_randomization": seed_block_paired_randomization_test(
                paired, n_resamples=n_randomization_samples, randomization_seed=randomization_seed
            ),
            "seed_block_bootstrap": paired_seed_block_bootstrap(
                paired, n_resamples=n_resamples, bootstrap_seed=bootstrap_seed
            ),
        },
        "secondary": {
            "seat_level_mcnemar_secondary": mcnemar_exact(paired_outcomes),
        },
        "fallback_contamination": fallback_contamination(baseline_fallbacks, candidate_fallbacks),
    }
