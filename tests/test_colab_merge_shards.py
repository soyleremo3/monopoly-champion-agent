"""Tests for scripts/colab_merge_shards.py: pure file/logic merging - every
STOP condition (incomplete shard, undeclared seed, overlapping seeds,
config mismatch, gap/extra vs. an expected range, duplicate (seed, seat)
pair) plus the success path. No engine, no real games - just synthetic
metadata.json/per_game.jsonl fixture directories.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "colab_merge_shards.py"

_spec = importlib.util.spec_from_file_location("colab_merge_shards", SCRIPT)
merge_module = importlib.util.module_from_spec(_spec)
sys.modules["colab_merge_shards"] = merge_module
_spec.loader.exec_module(merge_module)


BASE_METADATA = {
    "arm": "policy_only", "context": "crippled", "checkpoint_sha256": "b" * 64,
    "git_head_sha": "a" * 40, "self_play_optimized": False,
    "physical_games_per_seed": 4, "seat_records_per_seed": 4, "max_rounds": 200,
}


def _make_shard(tmp_path, name, *, seeds, metadata_overrides=None, records_per_seed=4, seed_game_counts=None):
    shard_dir = tmp_path / name
    shard_dir.mkdir()
    metadata = dict(BASE_METADATA, seeds=list(seeds), **(metadata_overrides or {}))
    (shard_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    # records_per_seed controls only the synthetic per_game.jsonl line count here -
    # metadata's own seat_records_per_seed (what completeness is checked against)
    # comes from BASE_METADATA/metadata_overrides above, independently, so tests can
    # make the two disagree on purpose (e.g. a truncated/partial shard).
    counts = seed_game_counts or {seed: records_per_seed for seed in seeds}
    lines = []
    for seed, count in counts.items():
        for seat in range(count):
            lines.append({"seed": seed, "focus_seat": seat, "completed": True, "winner": seat, "focus_won": True, "focus_net_worth": 100.0, "round_capped": False, "rounds": 10, "decisions": 5})
    (shard_dir / "per_game.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return shard_dir


def test_merge_succeeds_on_two_disjoint_complete_shards(tmp_path):
    shard_a = _make_shard(tmp_path, "a", seeds=[1, 2])
    shard_b = _make_shard(tmp_path, "b", seeds=[3, 4])

    all_per_game, merged_seeds, reference_metadata = merge_module.merge_shards([shard_a, shard_b])
    assert merged_seeds == [1, 2, 3, 4]
    assert len(all_per_game) == 16  # 4 seeds x 4 games
    assert reference_metadata["arm"] == "policy_only"


def test_merge_stops_on_incomplete_shard(tmp_path):
    shard_a = _make_shard(tmp_path, "a", seeds=[1, 2], seed_game_counts={1: 4, 2: 0})  # seed 2: no records at all
    with pytest.raises(RuntimeError, match="missing/incomplete"):
        merge_module.merge_shards([shard_a])


def test_merge_stops_on_partial_seed_seat_set(tmp_path):
    """Regression: a seed with SOME but not all seat records (2 of 4 -
    e.g. a shard interrupted mid-seed) must STOP with a clear "partial"
    reason, not be silently accepted as complete or lumped in with the
    "entirely missing" case."""
    shard_a = _make_shard(tmp_path, "a", seeds=[1, 2], seed_game_counts={1: 4, 2: 2})  # seed 2: only 2 of 4
    with pytest.raises(RuntimeError, match="partial record set"):
        merge_module.merge_shards([shard_a])


def test_merge_stops_on_self_play_optimized_partial_seed(tmp_path):
    """Regression for the exact reported bug at the merge layer:
    context=repaired/arm=both declares physical_games_per_seed=1 but
    seat_records_per_seed is still 4 - a seed with only 1 of 4 seat
    records (what a crash right after the physical game finishes, before
    all 4 seats are extracted and written, would leave) must STOP, not be
    silently accepted because a shard is "self-play-optimized"."""
    shard_a = _make_shard(
        tmp_path, "a", seeds=[1, 2],
        metadata_overrides={
            "arm": "both", "context": "repaired", "self_play_optimized": True,
            "physical_games_per_seed": 1, "seat_records_per_seed": 4,
        },
        seed_game_counts={1: 4, 2: 1},  # seed 2: only 1 of 4 seat records
    )
    with pytest.raises(RuntimeError, match="partial record set"):
        merge_module.merge_shards([shard_a])


def test_merge_stops_on_undeclared_seed(tmp_path):
    shard_a = _make_shard(tmp_path, "a", seeds=[1], seed_game_counts={1: 4, 99: 4})  # 99 not declared
    with pytest.raises(RuntimeError, match="undeclared"):
        merge_module.merge_shards([shard_a])


def test_merge_stops_on_overlapping_seeds_across_shards(tmp_path):
    shard_a = _make_shard(tmp_path, "a", seeds=[1, 2])
    shard_b = _make_shard(tmp_path, "b", seeds=[2, 3])  # seed 2 in both
    with pytest.raises(RuntimeError, match="overlapping shards"):
        merge_module.merge_shards([shard_a, shard_b])


def test_merge_stops_on_config_mismatch(tmp_path):
    shard_a = _make_shard(tmp_path, "a", seeds=[1])
    shard_b = _make_shard(tmp_path, "b", seeds=[2], metadata_overrides={"arm": "buy_only"})
    with pytest.raises(RuntimeError, match="different setups"):
        merge_module.merge_shards([shard_a, shard_b])


def test_merge_stops_on_physical_games_per_seed_mismatch(tmp_path):
    """physical_games_per_seed/seat_records_per_seed replaced the old single
    games_per_seed consistency field - both must still be checked."""
    shard_a = _make_shard(tmp_path, "a", seeds=[1])
    shard_b = _make_shard(
        tmp_path, "b", seeds=[2],
        metadata_overrides={"physical_games_per_seed": 1, "self_play_optimized": True},
    )
    with pytest.raises(RuntimeError, match="different setups"):
        merge_module.merge_shards([shard_a, shard_b])


def test_merge_stops_on_gap_vs_expected_range(tmp_path):
    shard_a = _make_shard(tmp_path, "a", seeds=[1, 2])  # missing seed 3 relative to expected [1,4)
    with pytest.raises(RuntimeError, match="missing="):
        merge_module.merge_shards([shard_a], expected_seed_start=1, expected_seed_count=3)


def test_merge_stops_on_extra_vs_expected_range(tmp_path):
    shard_a = _make_shard(tmp_path, "a", seeds=[1, 2, 3])  # extra seed 3 vs expected [1,3)
    with pytest.raises(RuntimeError, match="unexpected="):
        merge_module.merge_shards([shard_a], expected_seed_start=1, expected_seed_count=2)


def test_merge_succeeds_when_matching_expected_range_exactly(tmp_path):
    shard_a = _make_shard(tmp_path, "a", seeds=[1, 2])
    shard_b = _make_shard(tmp_path, "b", seeds=[3, 4])
    all_per_game, merged_seeds, _ = merge_module.merge_shards(
        [shard_a, shard_b], expected_seed_start=1, expected_seed_count=4,
    )
    assert merged_seeds == [1, 2, 3, 4]


def test_merge_stops_on_duplicate_seed_seat_pair_within_a_shard(tmp_path):
    shard_dir = tmp_path / "dup"
    shard_dir.mkdir()
    metadata = dict(BASE_METADATA, seeds=[1])
    (shard_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    # Two lines claiming (seed=1, seat=0) - caught as a duplicate seat within a
    # single seed's records regardless of the overall line count (4 total).
    lines = [
        {"seed": 1, "focus_seat": 0}, {"seed": 1, "focus_seat": 0},
        {"seed": 1, "focus_seat": 1}, {"seed": 1, "focus_seat": 2},
    ]
    (shard_dir / "per_game.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate"):
        merge_module.merge_shards([shard_dir])


def test_merge_raises_on_missing_metadata_file(tmp_path):
    shard_dir = tmp_path / "no_meta"
    shard_dir.mkdir()
    (shard_dir / "per_game.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="missing metadata.json"):
        merge_module.merge_shards([shard_dir])


def test_merge_raises_on_missing_jsonl_file(tmp_path):
    shard_dir = tmp_path / "no_jsonl"
    shard_dir.mkdir()
    (shard_dir / "metadata.json").write_text(json.dumps(dict(BASE_METADATA, seeds=[1])), encoding="utf-8")
    with pytest.raises(SystemExit, match="missing per_game.jsonl"):
        merge_module.merge_shards([shard_dir])


def test_main_writes_merged_output(tmp_path):
    shard_a = _make_shard(tmp_path, "a", seeds=[1, 2])
    shard_b = _make_shard(tmp_path, "b", seeds=[3, 4])
    output_dir = tmp_path / "merged"

    exit_code = merge_module.main(
        ["--shard-dirs", str(shard_a), str(shard_b), "--output-dir", str(output_dir)]
    )
    assert exit_code == 0
    assert (output_dir / "per_game.jsonl").is_file()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "OK"
    assert summary["merged_seeds"] == [1, 2, 3, 4]
    assert summary["shard_count"] == 2
    assert summary["seat_records"] == 16
    assert summary["physical_games"] == 16  # rotation mode: 1:1 with seat records
    lines = (output_dir / "per_game.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 16


def test_main_self_play_optimized_computes_physical_games_from_distinct_seeds(tmp_path):
    """Regression: in a self-play-optimized merge, physical_games must be
    counted from distinct seeds (1 physical game/seed), not from
    len(all_per_game) (4 seat records/seed) - conflating the two was the
    reported 4x-too-optimistic counting bug."""
    shard_a = _make_shard(
        tmp_path, "a", seeds=[1, 2],
        metadata_overrides={
            "arm": "both", "context": "repaired", "self_play_optimized": True,
            "physical_games_per_seed": 1, "seat_records_per_seed": 4,
        },
    )
    output_dir = tmp_path / "merged_self_play"

    exit_code = merge_module.main(["--shard-dirs", str(shard_a), "--output-dir", str(output_dir)])
    assert exit_code == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["seat_records"] == 8  # 2 seeds x 4 seat records
    assert summary["physical_games"] == 2  # 2 seeds x 1 physical game
    assert summary["games"] == 8  # win_rate/etc still computed per seat-record - unchanged semantics
