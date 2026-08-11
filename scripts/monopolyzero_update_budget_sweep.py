"""ASU-import-free MonopolyZero update-budget sweep.

Generates ZERO new self-play games. Reuses the existing 32-game replay
buffer and pre-training baseline checkpoint from
013-monopolyzero-strength-pilot-training (see docs/EXPERIMENTS.md), and
trains three independent checkpoints — 100, 500, and 1000 updates — each
starting fresh from the SAME baseline checkpoint with the SAME deterministic
sampling seed, so the only variable across the three runs is update budget:
the first 100 updates are byte-identical across all three runs, the first
500 identical between the 500- and 1000-update runs, etc. None resumes from
another's checkpoint.

Refuses to run unless the existing replay buffer and baseline checkpoint are
present and the baseline checkpoint's SHA-256 matches the value recorded in
013's experiment log — if either check fails, stops before doing anything
else (no games generated, no training attempted).

Built on scripts/monopolyzero_common.py — no monopoly_bench.adapters/.arena
/.training import, no ASU. Refuses to run unless PYTHONHASHSEED=0 is set
and the git tree is clean. Stops immediately (no checkpoint saved for that
budget) if any update produces a non-finite loss.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import monopolyzero_common as common  # noqa: E402

EXPECTED_POSITIONS = 37_772
EXPECTED_BASELINE_SHA256 = "22ae2abea60478355f61ce0404a31de3b52fb97769d4f99f7e04f4c673629370"

GLOBAL_SEED = 42
BATCH_SIZE = 64
UPDATE_BUDGETS = (100, 500, 1000)
LOG_INTERVAL = 25

REPO_ROOT = common.REPO_ROOT
PILOT_DIR = REPO_ROOT / "artifacts" / "monopolyzero_strength_pilot"
REPLAY_DIR = PILOT_DIR / "replay"
BASELINE_CHECKPOINT = PILOT_DIR / "baseline_pretraining.pt"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_reused_artifacts() -> dict:
    """Checks the replay buffer and baseline checkpoint this sweep reuses.
    Raises SystemExit (does not proceed to any training) on any mismatch."""
    problems = []

    metadata_path = REPLAY_DIR / "metadata.json"
    if not metadata_path.is_file():
        problems.append(f"missing replay metadata: {metadata_path}")
        replay_size = None
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        replay_size = metadata.get("size")
        if replay_size != EXPECTED_POSITIONS:
            problems.append(
                f"replay size mismatch: metadata.json says {replay_size}, "
                f"expected {EXPECTED_POSITIONS}"
            )

    if not BASELINE_CHECKPOINT.is_file():
        problems.append(f"missing baseline checkpoint: {BASELINE_CHECKPOINT}")
        baseline_sha256 = None
    else:
        baseline_sha256 = _sha256(BASELINE_CHECKPOINT)
        if baseline_sha256 != EXPECTED_BASELINE_SHA256:
            problems.append(
                f"baseline checkpoint SHA-256 mismatch: got {baseline_sha256}, "
                f"expected {EXPECTED_BASELINE_SHA256}"
            )

    if problems:
        raise SystemExit(
            "monopolyzero_update_budget_sweep.py refuses to run: reused "
            "artifacts failed integrity check, no new games will be "
            "generated:\n  - " + "\n  - ".join(problems)
        )

    return {"replay_size": replay_size, "baseline_sha256": baseline_sha256}


def main(argv: list[str] | None = None) -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)

    integrity = verify_reused_artifacts()

    common.ensure_reference_on_path()
    import numpy as np
    import torch

    from monopoly_bench.config import TrainingConfig
    from monopoly_bench.model import MonopolyZeroNet
    from monopoly_bench.storage import ReplayBuffer

    training_defaults = TrainingConfig()

    replay = ReplayBuffer(REPLAY_DIR, create=False)
    if len(replay) != EXPECTED_POSITIONS:
        raise SystemExit(
            f"Loaded replay buffer size {len(replay)} != expected {EXPECTED_POSITIONS}"
        )

    results = []
    for budget in UPDATE_BUDGETS:
        started = time.perf_counter()
        with common.RssMonitor() as rss:
            model = MonopolyZeroNet.load_inference(BASELINE_CHECKPOINT)
            model.train()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training_defaults.learning_rate,
                weight_decay=training_defaults.weight_decay,
            )
            rng = np.random.default_rng(GLOBAL_SEED)

            interval_log = []
            all_finite = True
            completed_updates = 0
            for update_index in range(1, budget + 1):
                batch = replay.sample(BATCH_SIZE, rng)
                stats = common.local_training_update(
                    model, optimizer, batch, gradient_clip=training_defaults.gradient_clip
                )
                finite = all(
                    torch.isfinite(torch.tensor(value)).item()
                    for key, value in stats.items()
                    if key in ("loss", "policy_loss", "value_loss", "gradient_norm")
                )
                completed_updates = update_index
                if not finite:
                    all_finite = False
                    interval_log.append({"update": update_index, **stats, "finite": False})
                    break
                if update_index % LOG_INTERVAL == 0 or update_index == budget:
                    interval_log.append({"update": update_index, **stats, "finite": True})

            checkpoint_path = None
            checkpoint_sha256 = None
            if all_finite:
                model.eval()
                checkpoint_path = PILOT_DIR / f"trained_updates_{budget}.pt"
                model.save_inference(
                    checkpoint_path,
                    {
                        "stage": "post_training_update_budget_sweep",
                        "games_trained": 0,
                        "updates": completed_updates,
                        "seed": GLOBAL_SEED,
                        "reused_replay_positions": EXPECTED_POSITIONS,
                    },
                )
                checkpoint_sha256 = _sha256(checkpoint_path)

            asu_modules_loaded = common.loaded_asu_modules()

        elapsed_s = time.perf_counter() - started
        result = {
            "budget": budget,
            "completed_updates": completed_updates,
            "all_finite": all_finite,
            "interval_log": interval_log,
            "final_stats": interval_log[-1] if interval_log else None,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_sha256": checkpoint_sha256,
            "elapsed_s": elapsed_s,
            "peak_rss_gib": rss.peak_gib,
            "asu_modules_loaded": asu_modules_loaded,
            "asu_modules_loaded_count": len(asu_modules_loaded),
        }
        results.append(result)

        if not all_finite:
            payload = {
                "status": "STOPPED_NON_FINITE_LOSS",
                "git_head_sha": git_head_sha,
                "integrity": integrity,
                "results": results,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            raise RuntimeError(
                f"Stopping sweep: budget={budget} produced a non-finite loss "
                f"at update {completed_updates}"
            )
        if asu_modules_loaded:
            payload = {
                "status": "STOPPED_ASU_MODULE_LOADED",
                "git_head_sha": git_head_sha,
                "integrity": integrity,
                "results": results,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            raise RuntimeError(f"Stopping sweep: ASU modules loaded: {asu_modules_loaded}")

    payload = {
        "status": "OK",
        "git_head_sha": git_head_sha,
        "integrity": integrity,
        "config": {
            "batch_size": BATCH_SIZE,
            "learning_rate": training_defaults.learning_rate,
            "weight_decay": training_defaults.weight_decay,
            "gradient_clip": training_defaults.gradient_clip,
            "sampling_seed": GLOBAL_SEED,
            "log_interval": LOG_INTERVAL,
            "new_self_play_games": 0,
            "reused_replay_positions": EXPECTED_POSITIONS,
            "reused_baseline_checkpoint_sha256": EXPECTED_BASELINE_SHA256,
        },
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
