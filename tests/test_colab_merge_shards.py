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
    "git_head_sha": "a" * 40, "games_per_seed": 4, "max_rounds": 200,
}


def _make_shard(tmp_path, name, *, seeds, metadata_overrides=None, games_per_seed=4, seed_game_counts=None):
    shard_dir = tmp_path / name
    shard_dir.mkdir()
    metadata = dict(BASE_METADATA, seeds=list(seeds), **(metadata_overrides or {}))
    (shard_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    counts = seed_game_counts or {seed: games_per_seed for seed in seeds}
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
    shard_a = _make_shard(tmp_path, "a", seeds=[1, 2], seed_game_counts={1: 4, 2: 2})  # seed 2 only 2 of 4
    with pytest.raises(RuntimeError, match="missing/incomplete"):
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
    # Two lines claiming (seed=1, seat=0) - satisfies the >= games_per_seed count check
    # (4 lines total) while still containing an actual duplicate pair.
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
    lines = (output_dir / "per_game.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 16
