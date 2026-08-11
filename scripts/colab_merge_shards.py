"""Merges multiple `colab_shard_runner.py` shard output directories into one
combined dataset — STOPs (raises) rather than silently proceeding on any of:

  - a shard whose per_game.jsonl is missing seat records for a seed it
    declared in its own metadata.json (the shard didn't finish, or was
    truncated), or has a partial (not exactly `seat_records_per_seed`,
    always 4) or duplicate seat-record set for a seed
  - a shard with games for a seed it never declared
  - two shards declaring an overlapping seed
  - shards whose metadata disagree on arm/context/checkpoint_sha256/
    git_head_sha/physical_games_per_seed/seat_records_per_seed/max_rounds
    (refusing to merge results from different setups or code versions)
  - (if --expected-seed-start/--expected-seed-count given) the merged seed
    coverage not exactly matching that declared full range - gaps or
    extras both fail
  - any duplicate (seed, seat) pair in the merged records, as a final
    safety net independent of the per-shard/per-seed checks above

Writes a merged `per_game.jsonl` and `summary.json` into `--output-dir` on
success. No training, no game-playing here - pure file/logic merging.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import colab_shard_runner as runner  # noqa: E402
import monopolyzero_hybrid_decomposition_audit as decomp  # noqa: E402

CONSISTENCY_FIELDS = (
    "arm", "context", "checkpoint_sha256", "git_head_sha",
    "physical_games_per_seed", "seat_records_per_seed", "max_rounds",
)


def _load_shard(shard_dir: Path) -> tuple[dict, list[dict]]:
    metadata_path = shard_dir / "metadata.json"
    jsonl_path = shard_dir / "per_game.jsonl"
    if not metadata_path.is_file():
        raise SystemExit(f"missing metadata.json in shard dir: {shard_dir}")
    if not jsonl_path.is_file():
        raise SystemExit(f"missing per_game.jsonl in shard dir: {shard_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    per_game: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                per_game.append(json.loads(line))
    return metadata, per_game


def merge_shards(
    shard_dirs: list[Path], *, expected_seed_start: int | None = None, expected_seed_count: int | None = None,
) -> tuple[list[dict], list[int], dict]:
    if not shard_dirs:
        raise SystemExit("no shard directories given")

    metadatas: list[tuple[Path, dict]] = []
    all_per_game: list[dict] = []
    seed_to_shard: dict[int, Path] = {}

    for shard_dir in shard_dirs:
        metadata, per_game = _load_shard(shard_dir)
        metadatas.append((shard_dir, metadata))

        declared_seeds = set(metadata["seeds"])
        required_seat_records_per_seed = metadata["seat_records_per_seed"]
        seed_seats: dict[int, list[int]] = {}
        for game in per_game:
            seed_seats.setdefault(game["seed"], []).append(game["focus_seat"])

        # Exact seat-set check (not a >= count check): a seed with only 1 of 4
        # seat records (the self-play-optimized-mode bug this replaced) or a
        # duplicate/out-of-range seat raises here rather than being silently
        # accepted as "complete enough".
        complete_seeds = runner.validate_seed_seat_completeness(
            seed_seats, required_seat_records_per_seed, source=f"shard {shard_dir}",
        )
        missing_in_shard = sorted(declared_seeds - complete_seeds)
        if missing_in_shard:
            raise RuntimeError(
                f"STOP: shard {shard_dir} declares seeds {sorted(declared_seeds)} in its metadata.json "
                f"but per_game.jsonl is missing/incomplete games for: {missing_in_shard} "
                "(the shard did not finish, or was truncated - do not merge a partial shard)"
            )
        extra_seeds = sorted(seed_seats.keys() - declared_seeds)
        if extra_seeds:
            raise RuntimeError(
                f"STOP: shard {shard_dir} has games for seed(s) {extra_seeds} it never declared in metadata.json"
            )

        for seed in declared_seeds:
            if seed in seed_to_shard:
                raise RuntimeError(
                    f"STOP: seed {seed} is declared by both {seed_to_shard[seed]} and {shard_dir} "
                    "(overlapping shards - each seed must belong to exactly one shard)"
                )
            seed_to_shard[seed] = shard_dir

        all_per_game.extend(per_game)

    reference_dir, reference_metadata = metadatas[0]
    for shard_dir, metadata in metadatas[1:]:
        mismatched = {
            field: (reference_metadata.get(field), metadata.get(field))
            for field in CONSISTENCY_FIELDS
            if reference_metadata.get(field) != metadata.get(field)
        }
        if mismatched:
            raise RuntimeError(
                f"STOP: shard {shard_dir}'s config does not match {reference_dir}'s - refusing to merge "
                f"shards from different setups/code versions: {mismatched}"
            )

    merged_seeds = sorted(seed_to_shard)
    if expected_seed_start is not None and expected_seed_count is not None:
        expected = set(range(expected_seed_start, expected_seed_start + expected_seed_count))
        missing = sorted(expected - set(merged_seeds))
        unexpected = sorted(set(merged_seeds) - expected)
        if missing or unexpected:
            raise RuntimeError(
                "STOP: merged seed coverage does not match the expected full range "
                f"[{expected_seed_start}, {expected_seed_start + expected_seed_count}): "
                f"missing={missing} unexpected={unexpected}"
            )

    seen_pairs: set[tuple[int, int]] = set()
    for game in all_per_game:
        pair = (game["seed"], game["focus_seat"])
        if pair in seen_pairs:
            raise RuntimeError(f"STOP: duplicate (seed, seat) pair {pair} across merged shards")
        seen_pairs.add(pair)

    return all_per_game, merged_seeds, reference_metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-seed-start", type=int, default=None)
    parser.add_argument("--expected-seed-count", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    all_per_game, merged_seeds, reference_metadata = merge_shards(
        args.shard_dirs, expected_seed_start=args.expected_seed_start, expected_seed_count=args.expected_seed_count,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_jsonl_path = args.output_dir / "per_game.jsonl"
    with merged_jsonl_path.open("w", encoding="utf-8") as handle:
        for game in sorted(all_per_game, key=lambda g: (g["seed"], g["focus_seat"])):
            handle.write(json.dumps(game, sort_keys=True) + "\n")

    summary = decomp.audit_v1._arm_summary(
        f"{reference_metadata.get('context')}_{reference_metadata.get('arm')}_merged", all_per_game, [],
    )
    summary["status"] = "OK"
    summary["arm"] = reference_metadata.get("arm")
    summary["context"] = reference_metadata.get("context")
    summary["checkpoint_sha256"] = reference_metadata.get("checkpoint_sha256")
    summary["git_head_sha"] = reference_metadata.get("git_head_sha")
    summary["shard_count"] = len(args.shard_dirs)
    summary["shard_dirs"] = [str(shard_dir) for shard_dir in args.shard_dirs]
    summary["merged_seeds"] = merged_seeds
    summary["seat_records"] = len(all_per_game)
    summary["physical_games"] = (
        len({game["seed"] for game in all_per_game})
        if reference_metadata.get("self_play_optimized") else len(all_per_game)
    )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
