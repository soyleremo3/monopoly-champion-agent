"""Compare two DDQN checkpoints (and optionally their training histories) for
reproducibility.

Deep-compares every field DDQNAgent.save() writes: metadata (format_version,
ruleset, state/action dims, player_id, hybrid, hidden_dim, step_count,
games_trained, epsilon, training_config), online/target network weights,
optimizer state, and replay buffer — tensors are compared bit-exact via
torch.equal, not with a tolerance. The checkpoint itself carries no
timing/memory fields, so there is nothing to exclude there.

For training histories (the *_history.json written alongside a checkpoint),
only deterministic fields are compared (win_rates, games, rewards, trade/
property counters, games-completed counters); elapsed_seconds,
seconds_per_game, peak_rss_gib, peak_cuda_gib, and training_segments (which
nests the same timing/memory fields) are intentionally ignored.

Usage:
    python scripts/compare_ddqn_checkpoints.py CKPT_A CKPT_B
    python scripts/compare_ddqn_checkpoints.py CKPT_A CKPT_B \\
        --history-a A_history.json --history-b B_history.json

Exits 0 and prints "MATCH: ..." if everything compared is identical; exits 1
and lists every mismatch otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

DETERMINISTIC_HISTORY_FIELDS = (
    "win_rates",
    "games",
    "rewards",
    "avg_trades_initiated",
    "avg_trades_accepted",
    "avg_trades_declined",
    "avg_properties_acquired",
    "resumed_from_games",
    "games_completed_this_run",
    "games_completed",
)
IGNORED_HISTORY_FIELDS = (
    "elapsed_seconds",
    "seconds_per_game",
    "peak_rss_gib",
    "peak_cuda_gib",
    "training_segments",
)


def _diff(path: str, a, b, out: list[str]) -> None:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            out.append(
                f"{path}: type mismatch ({type(a).__name__} vs {type(b).__name__})"
            )
            return
        if a.shape != b.shape:
            out.append(f"{path}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
            return
        if not torch.equal(a, b):
            diff = (a.float() - b.float()).abs()
            max_abs_diff = diff.max().item() if diff.numel() else 0.0
            out.append(f"{path}: tensor mismatch (max abs diff {max_abs_diff:.6g})")
        return
    if isinstance(a, dict) or isinstance(b, dict):
        if not isinstance(a, dict) or not isinstance(b, dict):
            out.append(
                f"{path}: type mismatch ({type(a).__name__} vs {type(b).__name__})"
            )
            return
        for key in sorted(set(a) | set(b)):
            if key not in a:
                out.append(f"{path}.{key}: missing in A")
            elif key not in b:
                out.append(f"{path}.{key}: missing in B")
            else:
                _diff(f"{path}.{key}", a[key], b[key], out)
        return
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
            out.append(
                f"{path}: type mismatch ({type(a).__name__} vs {type(b).__name__})"
            )
            return
        if len(a) != len(b):
            out.append(f"{path}: length mismatch {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(f"{path}[{i}]", x, y, out)
        return
    if a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def compare_checkpoints(path_a: Path, path_b: Path) -> list[str]:
    ckpt_a = torch.load(path_a, map_location="cpu", weights_only=True)
    ckpt_b = torch.load(path_b, map_location="cpu", weights_only=True)
    diffs: list[str] = []
    _diff("checkpoint", ckpt_a, ckpt_b, diffs)
    return diffs


def compare_history(path_a: Path, path_b: Path) -> tuple[list[str], list[str]]:
    """Returns (mismatches, ignored-fields-present) for deterministic fields only."""
    history_a = json.loads(path_a.read_text(encoding="utf-8"))
    history_b = json.loads(path_b.read_text(encoding="utf-8"))
    diffs: list[str] = []
    for field in DETERMINISTIC_HISTORY_FIELDS:
        if field not in history_a or field not in history_b:
            diffs.append(f"history.{field}: missing in one of the two histories")
            continue
        _diff(f"history.{field}", history_a[field], history_b[field], diffs)
    ignored_present = [
        field
        for field in IGNORED_HISTORY_FIELDS
        if field in history_a or field in history_b
    ]
    return diffs, ignored_present


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_a", type=Path)
    parser.add_argument("checkpoint_b", type=Path)
    parser.add_argument("--history-a", type=Path)
    parser.add_argument("--history-b", type=Path)
    args = parser.parse_args(argv)

    all_diffs = compare_checkpoints(args.checkpoint_a, args.checkpoint_b)

    if args.history_a and args.history_b:
        history_diffs, ignored_present = compare_history(args.history_a, args.history_b)
        all_diffs.extend(history_diffs)
        if ignored_present:
            print(
                "Ignored (timing/memory) history fields present, not compared: "
                f"{ignored_present}"
            )

    if all_diffs:
        print(f"MISMATCH: {len(all_diffs)} difference(s)")
        for line in all_diffs:
            print(f"  - {line}")
        return 1

    print("MATCH: checkpoints (and deterministic history fields, if provided) are identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
