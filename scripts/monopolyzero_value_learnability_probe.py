"""ASU-import-free, policy-untouched value-learnability probe.

Before any new strength/policy training, measures whether the existing
300-dim state representation (`monopoly_game_engine/state.py::build_state_vector`)
carries the TRUE, full-horizon (`max_rounds=200`) final winner in a
learnable way at all. Generates 64 clean `POLICY_ONLY` self-play games
(zero MCTS, zero fixed agents, zero ASU) from `baseline_pretraining.pt`,
samples round-stratified decision states from each game, labels every
sampled state with that game's REAL eventual winner (not a round-50
truncated proxy - see `019`'s deprecation of the `013` replay for exactly
that flaw), and trains a small, separate, own-written `ValueProbe` MLP
(300 -> 256 -> 4, CPU) purely as a supervised final-winner classifier.

Does NOT touch `MonopolyZeroNet`'s weights, does NOT run PUCT/search, does
NOT evaluate win rate, and makes NO strength/policy claim - this is a
representation-learnability diagnostic only. Computes no promotion verdict
of its own; the numbers are for a human to read.

Built on scripts/monopolyzero_common.py and scripts/evaluation_protocol.py
- no monopoly_bench.adapters/.arena/.training import, no ASU. Refuses to
run unless PYTHONHASHSEED=0 is set, the git tree is clean, the checkpoint
SHA-256 matches, and every self-play seed is registered in the DEV pool
(`evaluation_protocol.require_seed_scope`).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluation_protocol as ep  # noqa: E402
import monopolyzero_common as common  # noqa: E402

BASELINE_CHECKPOINT_SHA256 = "22ae2abea60478355f61ce0404a31de3b52fb97769d4f99f7e04f4c673629370"

MAX_ROUNDS = 200
GAMES_TOTAL = 64
SELF_PLAY_SEEDS = tuple(range(42000, 42000 + GAMES_TOTAL))
TRAIN_SEEDS = SELF_PLAY_SEEDS[:48]
VALIDATION_SEEDS = SELF_PLAY_SEEDS[48:]

# (label, round_lo, round_hi_inclusive_or_None_for_open_ended)
ROUND_BUCKETS = (
    ("1-25", 1, 25),
    ("26-50", 26, 50),
    ("51-100", 51, 100),
    ("101-150", 101, 150),
    ("151-terminal", 151, None),
)
SAMPLES_PER_CELL = 3  # deterministic, bounded: first N non-forced decisions per (game, seat, bucket)

VALUE_PROBE_INPUT_DIM = 300
VALUE_PROBE_HIDDEN_DIM = 256
VALUE_PROBE_OUTPUT_DIM = 4
TRAINING_SEED = 0
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
LEARNING_CURVE_FRACTIONS = (0.25, 0.5, 1.0)

REPO_ROOT = common.REPO_ROOT
PILOT_DIR = REPO_ROOT / "artifacts" / "monopolyzero_strength_pilot"
CHECKPOINT_PATH = PILOT_DIR / "baseline_pretraining.pt"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path: Path = CHECKPOINT_PATH, expected_sha256: str = BASELINE_CHECKPOINT_SHA256) -> str:
    if not path.is_file():
        raise SystemExit(
            f"monopolyzero_value_learnability_probe.py refuses to run: missing checkpoint {path}"
        )
    actual = _sha256(path)
    if actual != expected_sha256:
        raise SystemExit(
            "monopolyzero_value_learnability_probe.py refuses to run: checkpoint "
            f"SHA-256 mismatch. Got {actual}, expected {expected_sha256}."
        )
    return actual


def bucket_for_round(round_: int) -> str | None:
    for label, lo, hi in ROUND_BUCKETS:
        if hi is None:
            if round_ >= lo:
                return label
        elif lo <= round_ <= hi:
            return label
    return None


# ── pure metric helpers ─────────────────────────────────────────────────


def cross_entropy(probs, true_classes, *, eps: float = 1e-12) -> float:
    import numpy as np

    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    true_classes = np.asarray(true_classes)
    n = len(true_classes)
    return float(-np.mean(np.log(probs[np.arange(n), true_classes])))


def brier_score(probs, true_classes) -> float:
    import numpy as np

    probs = np.asarray(probs, dtype=np.float64)
    true_classes = np.asarray(true_classes)
    n, k = probs.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), true_classes] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def top1_accuracy(probs, true_classes) -> float:
    import numpy as np

    preds = np.argmax(np.asarray(probs), axis=1)
    return float(np.mean(preds == np.asarray(true_classes)))


def confusion_matrix(probs, true_classes, num_classes: int = 4) -> list:
    import numpy as np

    preds = np.argmax(np.asarray(probs), axis=1)
    true_classes = np.asarray(true_classes)
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true_c, pred_c in zip(true_classes.tolist(), preds.tolist()):
        matrix[true_c][pred_c] += 1
    return matrix


def expected_calibration_error(probs, true_classes, *, n_bins: int = 10) -> dict:
    import numpy as np

    probs = np.asarray(probs)
    true_classes = np.asarray(true_classes)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    correct = (preds == true_classes).astype(float)
    n = len(true_classes)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin_range": [lo, hi], "count": 0, "mean_confidence": None, "accuracy": None})
            continue
        mean_conf = float(confidences[mask].mean())
        acc = float(correct[mask].mean())
        ece += (count / n) * abs(mean_conf - acc) if n else 0.0
        bins.append({"bin_range": [lo, hi], "count": count, "mean_confidence": mean_conf, "accuracy": acc})
    return {"ece": float(ece), "bins": bins}


def class_balance(true_classes, num_classes: int = 4) -> dict:
    import numpy as np

    true_classes = np.asarray(true_classes)
    n = len(true_classes)
    counts = np.bincount(true_classes, minlength=num_classes)
    return {
        str(c): {"count": int(counts[c]), "fraction": (float(counts[c]) / n) if n else None}
        for c in range(num_classes)
    }


def uniform_baseline_probs(n: int, num_classes: int = 4):
    import numpy as np

    return np.full((n, num_classes), 1.0 / num_classes)


def leader_baseline_probs(leader_classes, num_classes: int = 4, *, eps: float = 1e-6):
    """A hard 'predict the current net-worth leader' baseline, smoothed by
    a small epsilon so cross-entropy stays finite when the leader is NOT
    the eventual winner (a degenerate 1.0/0.0 distribution would score
    infinite CE on every miss) - same clamp-based convention this project
    already uses in monopolyzero_common.py::local_training_update."""
    import numpy as np

    leader_classes = np.asarray(leader_classes)
    n = len(leader_classes)
    probs = np.full((n, num_classes), eps)
    probs[np.arange(n), leader_classes] = 1.0 - (num_classes - 1) * eps
    return probs


def evaluate_predictor(probs, true_classes, buckets) -> dict:
    import numpy as np

    buckets_arr = np.asarray(buckets)
    true_arr = np.asarray(true_classes)
    probs_arr = np.asarray(probs)

    overall = {
        "cross_entropy": cross_entropy(probs_arr, true_arr),
        "brier_score": brier_score(probs_arr, true_arr),
        "top1_accuracy": top1_accuracy(probs_arr, true_arr),
        "confusion_matrix": confusion_matrix(probs_arr, true_arr),
        "calibration": expected_calibration_error(probs_arr, true_arr),
        "class_balance": class_balance(true_arr),
    }
    by_bucket = {}
    for label, _, _ in ROUND_BUCKETS:
        mask = buckets_arr == label
        count = int(mask.sum())
        if count == 0:
            by_bucket[label] = {"count": 0, "cross_entropy": None, "brier_score": None, "top1_accuracy": None}
            continue
        by_bucket[label] = {
            "count": count,
            "cross_entropy": cross_entropy(probs_arr[mask], true_arr[mask]),
            "brier_score": brier_score(probs_arr[mask], true_arr[mask]),
            "top1_accuracy": top1_accuracy(probs_arr[mask], true_arr[mask]),
        }
    return {"overall": overall, "by_bucket": by_bucket}


# ── ValueProbe: small, own-written, never touches MonopolyZeroNet ──────


def build_value_probe(
    input_dim: int = VALUE_PROBE_INPUT_DIM,
    hidden_dim: int = VALUE_PROBE_HIDDEN_DIM,
    output_dim: int = VALUE_PROBE_OUTPUT_DIM,
):
    import torch.nn as nn

    class ValueProbe(nn.Module):
        """Single-hidden-layer MLP, final-winner classifier only. Own
        architecture/training loop, not reused or imported from the
        reference's training module."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    return ValueProbe()


def train_value_probe(
    train_x, train_y, val_x, val_y,
    *, training_seed: int = TRAINING_SEED, max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE, lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
):
    """Deterministic, simple training loop with early stopping on
    validation loss (as requested - this is model-selection, not gradient
    signal from validation data)."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.manual_seed(training_seed)
    model = build_value_probe()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_x_t = torch.as_tensor(train_x, dtype=torch.float32)
    train_y_t = torch.as_tensor(train_y, dtype=torch.long)
    val_x_t = torch.as_tensor(val_x, dtype=torch.float32)
    val_y_t = torch.as_tensor(val_y, dtype=torch.long)

    n = train_x_t.shape[0]
    shuffle_rng = np.random.default_rng(training_seed)

    best_val_loss = float("inf")
    best_state = {key: value.clone() for key, value in model.state_dict().items()}
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        permutation = shuffle_rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = permutation[start:start + batch_size]
            batch_x = train_x_t[batch_idx]
            batch_y = train_y_t[batch_idx]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(F.cross_entropy(model(val_x_t), val_y_t))

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs_run": epochs_run, "best_val_loss": best_val_loss}


def value_probe_predict_proba(model, x):
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1)
    return probs.numpy()


# ── self-play game generation + round-stratified sampling ──────────────


def _generate_games(model):
    """64 POLICY_ONLY self-play games, all 4 seats the same checkpoint, no
    fixed agents at all (so fixed_fallbacks is always 0 - no contamination
    possible by construction), no PUCT/search. For each game, deterministically
    snapshots up to SAMPLES_PER_CELL non-forced decision states per
    (seat, round-bucket) cell, then labels ALL of that game's snapshots with
    the REAL final winner once the game actually finishes."""
    from monopoly_bench.engine import MAX_DECISIONS_PER_TURN, NUM_PLAYERS, SharedGame, clone_env

    policy = common.build_local_policy_only(model)  # stateless, safe to share across all 4 seats

    games = []
    for seed in SELF_PLAY_SEEDS:
        game = SharedGame.new(seed, MAX_ROUNDS)
        decision_budget = MAX_ROUNDS * NUM_PLAYERS * MAX_DECISIONS_PER_TURN
        collected = {seat: {label: [] for label, _, _ in ROUND_BUCKETS} for seat in range(NUM_PLAYERS)}

        turn_index = 0
        illegal_actions = 0
        crashed = False
        error = None
        try:
            while turn_index < decision_budget and not game.env.done:
                actor = game.env.whose_turn()
                legal = tuple(game.env.get_allowed_actions(actor))
                if len(legal) == 1:
                    action = legal[0]
                else:
                    current_round = game.env.round
                    bucket = bucket_for_round(current_round)
                    if bucket is not None and len(collected[actor][bucket]) < SAMPLES_PER_CELL:
                        collected[actor][bucket].append({"round": current_round, "env_clone": clone_env(game)})
                    decision_seed = common._mix_decision_seed(seed, turn_index, actor)
                    action, _, _, _ = common._invoke_policy(policy, game, actor, decision_seed)
                    if action not in legal:
                        raise RuntimeError(f"seat {actor} attempted illegal action {action}")
                game.step(action)
                turn_index += 1
        except RuntimeError as exc:
            illegal_actions = 1
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - intentional fail-closed boundary
            crashed = True
            error = f"{type(exc).__name__}: {exc}"

        finished = game.env.done
        winner = game.env.winner() if finished else None
        games.append(
            {
                "seed": seed, "collected": collected, "completed": finished,
                "winner": winner, "decisions": turn_index, "final_round": game.env.round,
                "illegal_actions": illegal_actions, "crashed": crashed, "error": error,
            }
        )
    return games


def _build_records(games):
    """Flattens every game's collected snapshots into per-state records,
    labeled with the actor-relative TRUE final winner class (0 = the
    deciding seat itself won, matching this project's existing
    actor-relative convention - see monopolyzero_common.py's
    local_training_update / model.py's state-relative ordering) and the
    actor-relative current-net-worth-leader class at that same state."""
    from monopoly_bench.engine import NUM_PLAYERS, actor_order

    records = []
    for g in games:
        winner = g["winner"]
        for seat in range(NUM_PLAYERS):
            order = actor_order(seat)
            for bucket_label, snapshots in g["collected"][seat].items():
                for snapshot in snapshots:
                    env_clone = snapshot["env_clone"]
                    state_vector = env_clone._get_state(seat)
                    net_worth = tuple(float(p.net_worth()) for p in env_clone.players)
                    leader_absolute = max(range(NUM_PLAYERS), key=lambda s: net_worth[s])
                    records.append(
                        {
                            "seed": g["seed"],
                            "seat": seat,
                            "round": snapshot["round"],
                            "bucket": bucket_label,
                            "state": state_vector,
                            "true_class_relative": order.index(winner),
                            "leader_class_relative": order.index(leader_absolute),
                        }
                    )
    return records


def _learning_curve_subset_seeds(train_seeds: tuple, fraction: float) -> tuple:
    import math

    count = max(1, math.ceil(len(train_seeds) * fraction))
    return train_seeds[:count]


def main(argv: list[str] | None = None) -> int:
    common.require_pinned_hash_seed(Path(__file__).name)
    git_head_sha = common.require_clean_git_tree(Path(__file__).name)
    checkpoint_sha256 = verify_checkpoint()
    ep.require_seed_scope(SELF_PLAY_SEEDS, ep.SEED_CLASS_DEV, context="monopolyzero_value_learnability_probe.py")

    common.ensure_reference_on_path()
    import random

    import numpy as np
    import torch

    from monopoly_bench.model import MonopolyZeroNet
    from monopoly_game_engine.state import STATE_DIM

    if STATE_DIM != VALUE_PROBE_INPUT_DIM:
        raise SystemExit(f"State dim mismatch: engine reports {STATE_DIM}, probe expects {VALUE_PROBE_INPUT_DIM}")

    started = time.perf_counter()
    with common.RssMonitor() as rss:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        model = MonopolyZeroNet.load_inference(CHECKPOINT_PATH)
        model.eval()

        games = _generate_games(model)
        total_illegal = sum(g["illegal_actions"] for g in games)
        total_crashed = sum(int(g["crashed"]) for g in games)
        if total_illegal or total_crashed:
            payload = {
                "status": "FAILED_DURING_GAME_GENERATION",
                "git_head_sha": git_head_sha,
                "games": [
                    {k: v for k, v in g.items() if k != "collected"} for g in games
                ],
                "total_illegal_actions": total_illegal,
                "total_crashed": total_crashed,
            }
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
            raise RuntimeError(f"Stopping: crashed={total_crashed} illegal={total_illegal}")

        records = _build_records(games)

        train_seed_set = set(TRAIN_SEEDS)
        val_seed_set = set(VALIDATION_SEEDS)
        seed_overlap = train_seed_set & val_seed_set
        if seed_overlap:
            raise RuntimeError(f"Leakage guard failed: train/validation seed overlap {sorted(seed_overlap)}")

        train_records = [r for r in records if r["seed"] in train_seed_set]
        val_records = [r for r in records if r["seed"] in val_seed_set]
        train_record_ids = {(r["seed"], r["seat"], r["round"]) for r in train_records}
        val_record_ids = {(r["seed"], r["seat"], r["round"]) for r in val_records}
        state_overlap = train_record_ids & val_record_ids
        if state_overlap:
            raise RuntimeError(f"Leakage guard failed: train/validation state overlap ({len(state_overlap)} states)")

        train_x = np.stack([r["state"] for r in train_records]).astype(np.float32)
        train_y = np.array([r["true_class_relative"] for r in train_records], dtype=np.int64)
        train_buckets = [r["bucket"] for r in train_records]
        train_leader_y = np.array([r["leader_class_relative"] for r in train_records], dtype=np.int64)

        val_x = np.stack([r["state"] for r in val_records]).astype(np.float32)
        val_y = np.array([r["true_class_relative"] for r in val_records], dtype=np.int64)
        val_buckets = [r["bucket"] for r in val_records]
        val_leader_y = np.array([r["leader_class_relative"] for r in val_records], dtype=np.int64)

        baselines = {
            "uniform": {
                "train": evaluate_predictor(uniform_baseline_probs(len(train_records)), train_y, train_buckets),
                "validation": evaluate_predictor(uniform_baseline_probs(len(val_records)), val_y, val_buckets),
            },
            "net_worth_leader": {
                "train": evaluate_predictor(leader_baseline_probs(train_leader_y), train_y, train_buckets),
                "validation": evaluate_predictor(leader_baseline_probs(val_leader_y), val_y, val_buckets),
            },
        }

        value_probe_model, training_stats = train_value_probe(train_x, train_y, val_x, val_y)
        learned = {
            "train": evaluate_predictor(value_probe_predict_proba(value_probe_model, train_x), train_y, train_buckets),
            "validation": evaluate_predictor(value_probe_predict_proba(value_probe_model, val_x), val_y, val_buckets),
        }

        learning_curve = []
        for fraction in LEARNING_CURVE_FRACTIONS:
            subset_seeds = set(_learning_curve_subset_seeds(TRAIN_SEEDS, fraction))
            subset_mask = [r["seed"] in subset_seeds for r in train_records]
            subset_x = train_x[subset_mask]
            subset_y = train_y[subset_mask]
            subset_model, subset_stats = train_value_probe(subset_x, subset_y, val_x, val_y)
            subset_val_probs = value_probe_predict_proba(subset_model, val_x)
            learning_curve.append(
                {
                    "fraction": fraction,
                    "train_games": len(subset_seeds),
                    "train_states": int(sum(subset_mask)),
                    "epochs_run": subset_stats["epochs_run"],
                    "validation_cross_entropy": cross_entropy(subset_val_probs, val_y),
                    "validation_brier_score": brier_score(subset_val_probs, val_y),
                    "validation_top1_accuracy": top1_accuracy(subset_val_probs, val_y),
                }
            )

        asu_modules_loaded = common.loaded_asu_modules()

    elapsed_s = time.perf_counter() - started

    bucket_counts_train: dict[str, int] = {}
    bucket_counts_val: dict[str, int] = {}
    for label, _, _ in ROUND_BUCKETS:
        bucket_counts_train[label] = sum(1 for b in train_buckets if b == label)
        bucket_counts_val[label] = sum(1 for b in val_buckets if b == label)

    payload = {
        "git_head_sha": git_head_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "config": {
            "self_play_seeds": list(SELF_PLAY_SEEDS),
            "train_seeds": list(TRAIN_SEEDS),
            "validation_seeds": list(VALIDATION_SEEDS),
            "max_rounds": MAX_ROUNDS,
            "policy": "POLICY_ONLY (no PUCT/search)",
            "opponents": "none - all 4 seats are the same checkpoint (clean self-play, zero fixed agents)",
            "round_buckets": [label for label, _, _ in ROUND_BUCKETS],
            "samples_per_cell": SAMPLES_PER_CELL,
            "value_probe": {
                "input_dim": VALUE_PROBE_INPUT_DIM, "hidden_dim": VALUE_PROBE_HIDDEN_DIM,
                "output_dim": VALUE_PROBE_OUTPUT_DIM, "max_epochs": MAX_EPOCHS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE, "lr": LEARNING_RATE,
                "batch_size": BATCH_SIZE, "training_seed": TRAINING_SEED,
            },
            "normalization": "none - build_state_vector's own [0,1]-scaled fields are used directly, so there are no TRAIN-fit statistics for VALIDATION to leak from",
            "no_policy_training": True,
            "no_puct": True,
            "no_win_rate_evaluation": True,
            "no_arbitrary_go_kill_threshold": True,
            "asu_involved": False,
        },
        "integrity": {
            "games_total": len(games),
            "total_illegal_actions": total_illegal,
            "total_crashed": total_crashed,
            "fixed_fallbacks": 0,
        },
        "leakage_guard": {
            "seed_overlap_train_validation": len(seed_overlap),
            "state_overlap_train_validation": len(state_overlap),
        },
        "dataset": {
            "train_states": len(train_records),
            "validation_states": len(val_records),
            "train_bucket_counts": bucket_counts_train,
            "validation_bucket_counts": bucket_counts_val,
        },
        "baselines": baselines,
        "value_probe_training": training_stats,
        "learned": learned,
        "learning_curve": learning_curve,
        "asu_modules_loaded": asu_modules_loaded,
        "asu_modules_loaded_count": len(asu_modules_loaded),
        "elapsed_s": elapsed_s,
        "peak_rss_gib": rss.peak_gib,
    }

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if asu_modules_loaded:
        raise RuntimeError(f"ASU modules loaded during probe: {asu_modules_loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
