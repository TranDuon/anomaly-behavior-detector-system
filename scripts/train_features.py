"""Train the LSTM Autoencoder from cached per-clip feature sequences."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomaly.train import train_with_validation
from core.config import settings


def feature_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.npz"))
    usable = []
    for path in files:
        with np.load(path) as data:
            count = len(data["sequences"])
        if count > 0:
            usable.append(path)
        else:
            print(f"[train-features] Ignore empty cache: {path.name}")
    return usable


def split_files(
    files: list[Path], val_ratio: float, seed: int
) -> tuple[list[Path], list[Path]]:
    if len(files) < 2:
        raise ValueError("At least two non-empty clip caches are required.")
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def compute_scaler(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    feature_sum = np.zeros(settings.feature_dim, dtype=np.float64)
    feature_squared_sum = np.zeros(settings.feature_dim, dtype=np.float64)
    timestep_count = 0

    for path in files:
        with np.load(path) as data:
            sequences = np.asarray(data["sequences"], dtype=np.float64)
        flattened = sequences.reshape(-1, settings.feature_dim)
        feature_sum += flattened.sum(axis=0)
        feature_squared_sum += np.square(flattened).sum(axis=0)
        timestep_count += len(flattened)

    mean = feature_sum / timestep_count
    variance = feature_squared_sum / timestep_count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-12))
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def load_sequences(
    files: list[Path], mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    arrays = []
    for path in files:
        with np.load(path) as data:
            sequences = np.asarray(data["sequences"], dtype=np.float32)
        arrays.append((sequences - mean) / std)
    return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LSTM Autoencoder from cached clip features"
    )
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:N")
    parser.add_argument("--dataset-name", default="custom-normal-video")
    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio must be between 0 and 1")

    features_dir = Path(args.features_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = feature_files(features_dir)
    train_files, val_files = split_files(files, args.val_ratio, args.seed)

    print(
        f"[train-features] Clips: {len(train_files)} train, "
        f"{len(val_files)} validation"
    )
    mean, std = compute_scaler(train_files)
    np.savez(output_dir / "feature_scaler.npz", mean=mean, std=std)

    train_sequences = load_sequences(train_files, mean, std)
    val_sequences = load_sequences(val_files, mean, std)
    print(
        f"[train-features] Sequences: {len(train_sequences)} train, "
        f"{len(val_sequences)} validation"
    )

    model, metadata = train_with_validation(
        train_sequences,
        val_sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
        patience=args.patience,
    )
    model.save(output_dir / "lstm_autoencoder.pt")

    threshold_payload = {
        "threshold": metadata["threshold"],
        "method": metadata["threshold_method"],
        "validation_error_mean": metadata["validation_error_mean"],
        "validation_error_std": metadata["validation_error_std"],
        "validation_error_p99": metadata["validation_error_p99"],
    }
    write_json(output_dir / "threshold.json", threshold_payload)
    write_json(output_dir / "training_history.json", {"history": metadata["history"]})
    write_json(
        output_dir / "split.json",
        {
            "seed": args.seed,
            "train_clips": [path.name for path in train_files],
            "validation_clips": [path.name for path in val_files],
        },
    )
    write_json(
        output_dir / "config.json",
        {
            "dataset": args.dataset_name,
            "seq_len": settings.seq_len,
            "feature_dim": settings.feature_dim,
            "lstm_hidden": settings.lstm_hidden,
            "lstm_layers": settings.lstm_layers,
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": args.seed,
            "device": metadata["device"],
            "best_epoch": metadata["best_epoch"],
            "best_val_loss": metadata["best_val_loss"],
        },
    )
    print(f"[train-features] Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
