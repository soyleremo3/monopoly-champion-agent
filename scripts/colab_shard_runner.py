"""Colab-safe, resumable, single-shard game runner.

Runs one (context, arm) combination over a seed range, writing a per-game
JSONL that is fsync'd after every completed SEED (not every game) so a
disconnected/pre-empted Colab runtime can resume with `--resume` without
re-playing already-completed seeds. Designed to have zero local-path or
environment assumptions beyond a standard `git clone` of this repo: every
path is derived from `monopolyzero_common.REPO_ROOT`
(`Path(__file__).resolve().parents[1]`), which is correct wherever the repo
was cloned (`/content/...` on Colab, anything else locally).

Two modes:
  - `--benchmark-games N`: plays N games (from `--seed-start`, no output
    files) purely to measure `sec/game` and project total runtime for
    100/500/1000 games — meant to be run before committing to a real shard
    size on a fresh Colab runtime, where per-game cost is unknown upfront.
  - Normal shard mode: plays `--seed-count` seeds starting at
    `--seed-start`, writing `per_game.jsonl`, `metadata.json`,
    `summary.json`, and `run_log.txt` into `--output-dir`.

`context`: "crippled" (other 3 seats POLICY_ONLY) or "repaired" (other 3
seats HYBRID_COMPAT(BOTH)) — same two contexts `024`'s decomposition audit
used. `arm`: policy_only / buy_only / trade_only / both — same four arms.
When context=repaired and arm=both, the focus seat's policy configuration
is identical to its peers', so one self-play game per seed yields all 4
seats' paired records (see `monopolyzero_hybrid_decomposition_audit.py`'s
`_run_self_play_uniform_arm` docstring for why that's valid) — this runner
reuses that same optimization at seed granularity.

No training. Refuses to run unless PYTHONHASHSEED=0, the git tree is
clean, the checkpoint SHA-256 matches, and every requested seed is
registered in the DEV pool.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluation_protocol as ep  # noqa: E402
import monopolyzero_common as common  # noqa: E402
import monopolyzero_hybrid_decomposition_audit as decomp  # noqa: E402

MAX_ROUNDS = decomp.MAX_ROUNDS
NUM_SEATS = decomp.NUM_SEATS
CHECKPOINT_PATH = decomp.CHECKPOINT_PATH
BASELINE_CHECKPOINT_SHA256 = decomp.BASELINE_CHECKPOINT_SHA256

CONTEXT_CRIPPLED = "crippled"
CONTEXT_REPAIRED = "repaired"
CONTEXTS = (CONTEXT_CRIPPLED, CONTEXT_REPAIRED)
ARMS = (decomp.ARM_POLICY_ONLY, decomp.ARM_BUY_ONLY, decomp.ARM_TRADE_ONLY, decomp.ARM_BOTH)

BENCHMARK_PROJECTION_GAME_COUNTS = (100, 500, 1000)


def is_self_play_optimized(context: str, arm: str) -> bool:
    return context == CONTEXT_REPAIRED and arm == decomp.ARM_BOTH


def games_per_seed(context: str, arm: str) -> int:
    return 1 if is_self_play_optimized(context, arm) else NUM_SEATS


def build_peer_factory(context: str, model):
    if context == CONTEXT_CRIPPLED:
        policy_only = common.build_local_policy_only(model)
        return lambda: policy_only  # stateless, safe to share
    if context == CONTEXT_REPAIRED:
        return lambda: common.build_local_hybrid_compat_policy(model)
    raise ValueError(f"unknown context: {context}")


def get_submodule_sha() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(common.REFERENCE_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _play_one_seed(seed: int, context: str, arm: str, model, focus_factory, peer_factory) -> dict:
    """Plays exactly one seed's worth of games: 4 rotation games (one per
    focus seat) normally, or 1 self-play game (4 seats' records extracted)
    when `is_self_play_optimized(context, arm)`. Mirrors
    monopolyzero_hybrid_decomposition_audit.py's `_run_rotation_arm`/
    `_run_self_play_uniform_arm` at single-seed granularity, so a Colab
    disconnect never loses more than the seed in flight."""
    per_game: list[dict] = []
    records: list[dict] = []
    latencies: list[float] = []
    intervention_log: list[dict] = []
    total_illegal = 0
    total_crashed = 0
    incomplete = 0

    def _record_seat(seat: int, outcome, policy) -> None:
        won = bool(outcome.completed and outcome.winner == seat)
        net_worth = outcome.final_net_worth[seat] if outcome.final_net_worth else 0.0
        records.append({"seed": seed, "seat": seat, "win": won, "net_worth": net_worth})
        per_game.append(
            {
                "seed": seed, "focus_seat": seat, "completed": outcome.completed,
                "winner": outcome.winner, "focus_won": won, "rounds": outcome.final_round,
                "decisions": outcome.decisions, "focus_net_worth": net_worth,
                "round_capped": outcome.final_round >= MAX_ROUNDS,
            }
        )
        hybrid_log = getattr(policy, "log", None)
        if hybrid_log is not None:
            for idx, entry in enumerate(hybrid_log):
                tagged = dict(entry)
                tagged.update(seed=seed, focus_seat=seat, decision_index=idx)
                intervention_log.append(tagged)

    if is_self_play_optimized(context, arm):
        seat_policies = {seat: focus_factory() for seat in range(NUM_SEATS)}
        outcome = common.play_local_game(
            game_id=seed * 10, seed=seed, policies=seat_policies,
            max_rounds=MAX_ROUNDS, record_seats=set(),
        )
        total_illegal += outcome.illegal_actions
        total_crashed += int(outcome.crashed)
        incomplete += int(not outcome.completed)
        latencies.extend(outcome.search_latencies_s)
        for seat in range(NUM_SEATS):
            _record_seat(seat, outcome, seat_policies[seat])
    else:
        for focus_seat in range(NUM_SEATS):
            focus_policy = focus_factory()
            policies = {seat: peer_factory() for seat in range(NUM_SEATS) if seat != focus_seat}
            policies[focus_seat] = focus_policy
            outcome = common.play_local_game(
                game_id=seed * 10 + focus_seat, seed=seed, policies=policies,
                max_rounds=MAX_ROUNDS, record_seats=set(),
            )
            total_illegal += outcome.illegal_actions
            total_crashed += int(outcome.crashed)
            incomplete += int(not outcome.completed)
            latencies.extend(outcome.search_latencies_s)
            _record_seat(focus_seat, outcome, focus_policy)

    return {
        "per_game": per_game, "records": records, "latencies": latencies,
        "intervention_log": intervention_log, "total_illegal": total_illegal,
        "total_crashed": total_crashed, "incomplete": incomplete,
    }


def run_benchmark(*, n_games: int, seed_start: int, context: str, arm: str, model) -> dict:
    """Plays exactly `n_games` PHYSICAL games (real `play_local_game` calls),
    not seat-records. In self-play-optimized mode (context=repaired,
    arm=both) one physical game yields 4 seat-records, so counting records
    instead of physical games would under-count real per-game cost by 4x -
    `n_games` seat-records would only be `n_games / 4` physical games, and
    `sec_per_game` derived from that would be 4x too optimistic."""
    focus_factory = decomp._focus_policy_factory(model, arm)
    peer_factory = build_peer_factory(context, model)
    physical_games_per_seed = 1 if is_self_play_optimized(context, arm) else NUM_SEATS

    started = time.perf_counter()
    physical_games_completed = 0
    seat_records_completed = 0
    seed = seed_start
    while physical_games_completed < n_games:
        result = _play_one_seed(seed, context, arm, model, focus_factory, peer_factory)
        if result["total_illegal"] or result["total_crashed"]:
            raise RuntimeError(f"benchmark: illegal/crash at seed {seed} - {result}")
        physical_games_completed += physical_games_per_seed
        seat_records_completed += len(result["per_game"])
        seed += 1
    elapsed_s = time.perf_counter() - started

    sec_per_physical_game = elapsed_s / physical_games_completed if physical_games_completed else None
    projected_seconds = (
        {str(n): sec_per_physical_game * n for n in BENCHMARK_PROJECTION_GAME_COUNTS}
        if sec_per_physical_game else None
    )
    return {
        "mode": "benchmark", "context": context, "arm": arm,
        "seeds_used": seed - seed_start,
        "physical_games_completed": physical_games_completed,
        "seat_records_completed": seat_records_completed,
        "elapsed_s": elapsed_s, "sec_per_physical_game": sec_per_physical_game,
        "projected_seconds": projected_seconds,
        "self_play_optimized": is_self_play_optimized(context, arm),
    }


def _read_completed_seeds(jsonl_path: Path, required_games_per_seed: int) -> set[int]:
    if not jsonl_path.is_file():
        return set()
    counts: dict[int, int] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            counts[record["seed"]] = counts.get(record["seed"], 0) + 1
    return {seed for seed, count in counts.items() if count >= required_games_per_seed}


def run_shard(
    *, seed_start: int, seed_count: int, context: str, arm: str, output_dir: Path,
    resume: bool, model, git_head_sha: str, checkpoint_sha256: str, submodule_sha: str | None,
) -> dict:
    seeds = list(range(seed_start, seed_start + seed_count))
    ep.require_seed_scope(seeds, ep.SEED_CLASS_DEV, context="colab_shard_runner.py")

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "per_game.jsonl"
    metadata_path = output_dir / "metadata.json"
    summary_path = output_dir / "summary.json"
    log_path = output_dir / "run_log.txt"

    required_games_per_seed = games_per_seed(context, arm)
    new_metadata = {
        "git_head_sha": git_head_sha, "checkpoint_sha256": checkpoint_sha256,
        "submodule_sha": submodule_sha, "arm": arm, "context": context,
        "seed_start": seed_start, "seed_count": seed_count, "seeds": seeds,
        "self_play_optimized": is_self_play_optimized(context, arm),
        "games_per_seed": required_games_per_seed, "max_rounds": MAX_ROUNDS,
    }

    if resume and metadata_path.is_file():
        prior_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        mismatched = {
            key: (prior_metadata.get(key), new_metadata[key])
            for key in ("git_head_sha", "checkpoint_sha256", "arm", "context", "seed_start", "seed_count")
            if prior_metadata.get(key) != new_metadata[key]
        }
        if mismatched:
            raise RuntimeError(
                f"--resume refuses to continue: shard config differs from the prior run's metadata.json: {mismatched}"
            )
        completed_seeds = _read_completed_seeds(jsonl_path, required_games_per_seed)
    elif resume:
        # --resume given but metadata.json doesn't exist yet (first invocation) -
        # nothing to resume from and nothing to validate against, so this starts
        # fresh exactly like a non-resume run. Still reads any orphaned jsonl data
        # (defensive - normal fresh output_dir has none).
        completed_seeds = _read_completed_seeds(jsonl_path, required_games_per_seed)
    else:
        if jsonl_path.is_file() and jsonl_path.stat().st_size > 0:
            raise SystemExit(
                f"{jsonl_path} already exists and is non-empty; pass --resume to continue it "
                "or remove/rename it first to start a fresh shard"
            )
        completed_seeds = set()

    metadata_path.write_text(json.dumps(new_metadata, indent=2, sort_keys=True), encoding="utf-8")

    log_handle = log_path.open("a", encoding="utf-8")

    def _log(message: str) -> None:
        print(message)
        log_handle.write(message + "\n")
        log_handle.flush()

    _log(
        f"colab_shard_runner: seeds={seed_start}..{seed_start + seed_count - 1} arm={arm} "
        f"context={context} resume={resume} already_complete={len(completed_seeds)}/{len(seeds)}"
    )

    focus_factory = decomp._focus_policy_factory(model, arm)
    peer_factory = build_peer_factory(context, model)

    total_illegal = 0
    total_crashed = 0
    total_incomplete = 0
    stopped_early = False
    started = time.perf_counter()

    try:
        with jsonl_path.open("a", encoding="utf-8") as jsonl_handle:
            for seed in seeds:
                if seed in completed_seeds:
                    _log(f"seed {seed}: already complete, skipping")
                    continue
                seed_started = time.perf_counter()
                result = _play_one_seed(seed, context, arm, model, focus_factory, peer_factory)
                total_illegal += result["total_illegal"]
                total_crashed += result["total_crashed"]
                total_incomplete += result["incomplete"]
                for game in result["per_game"]:
                    jsonl_handle.write(json.dumps(game, sort_keys=True) + "\n")
                jsonl_handle.flush()
                os.fsync(jsonl_handle.fileno())
                seed_elapsed = time.perf_counter() - seed_started
                _log(
                    f"seed {seed}: {len(result['per_game'])} game(s) in {seed_elapsed:.1f}s "
                    f"(illegal={result['total_illegal']} crashed={result['total_crashed']} "
                    f"incomplete={result['incomplete']})"
                )
                if result["total_illegal"] or result["total_crashed"] or result["incomplete"]:
                    _log(f"seed {seed}: STOPPING shard - integrity violation, refusing to continue")
                    stopped_early = True
                    break
    finally:
        elapsed_s = time.perf_counter() - started

        all_per_game: list[dict] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    all_per_game.append(json.loads(line))

        status = "OK" if not (total_illegal or total_crashed or total_incomplete or stopped_early) else "FAILED"
        summary = decomp.audit_v1._arm_summary(f"{context}_{arm}", all_per_game, [])
        summary["status"] = status
        summary["context"] = context
        summary["arm"] = arm
        summary["seeds_requested"] = seeds
        summary["seeds_completed_this_run"] = len(seeds) - len(completed_seeds)
        summary["seeds_already_complete_before_this_run"] = len(completed_seeds)
        summary["total_illegal_actions_this_invocation"] = total_illegal
        summary["total_crashed_this_invocation"] = total_crashed
        summary["total_incomplete_this_invocation"] = total_incomplete
        summary["elapsed_s_this_invocation"] = elapsed_s
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")

        new_metadata["finished_status"] = status
        new_metadata["elapsed_s_this_invocation"] = elapsed_s
        metadata_path.write_text(json.dumps(new_metadata, indent=2, sort_keys=True), encoding="utf-8")

        log_handle.close()

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, default=None, help="required unless --benchmark-games is given")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--context", choices=CONTEXTS, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="required unless --benchmark-games is given")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark-games", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.benchmark_games is None and args.seed_count is None:
        raise SystemExit("--seed-count is required unless --benchmark-games is given")

    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)
    ep.require_seed_scope(
        range(args.seed_start, args.seed_start + max(args.seed_count or 0, args.benchmark_games or 0, 1)),
        ep.SEED_CLASS_DEV, context="colab_shard_runner.py",
    )
    checkpoint_sha256 = decomp.audit_v1.verify_baseline_checkpoint()
    submodule_sha = get_submodule_sha()

    common.ensure_reference_on_path()
    from monopoly_bench.engine import NUM_PLAYERS
    from monopoly_bench.model import MonopolyZeroNet

    if NUM_PLAYERS != NUM_SEATS:
        raise RuntimeError(f"Expected {NUM_SEATS} players, engine reports {NUM_PLAYERS}")

    model = MonopolyZeroNet.load_inference(CHECKPOINT_PATH)
    model.eval()

    if args.benchmark_games:
        result = run_benchmark(
            n_games=args.benchmark_games, seed_start=args.seed_start,
            context=args.context, arm=args.arm, model=model,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.output_dir is None:
        raise SystemExit("--output-dir is required for a real shard run (only optional with --benchmark-games)")

    summary = run_shard(
        seed_start=args.seed_start, seed_count=args.seed_count, context=args.context, arm=args.arm,
        output_dir=args.output_dir, resume=args.resume, model=model,
        git_head_sha=git_head_sha, checkpoint_sha256=checkpoint_sha256, submodule_sha=submodule_sha,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))

    asu_modules_loaded = common.loaded_asu_modules()
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded during evaluation: {asu_modules_loaded}")
    return 0 if summary.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
