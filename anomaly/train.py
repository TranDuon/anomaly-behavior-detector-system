"""
anomaly/train.py — Train the LSTM Autoencoder on NORMAL behaviour sequences.

Strategy:
  1. Feed ONLY normal behaviour video through the full pipeline to collect
     feature sequences (or use the synthetic generator below).
  2. Train to minimise MSE reconstruction loss.
  3. After training, calibrate the anomaly threshold on a held-out normal set.

Usage:
  python -m anomaly.train --source data/normal_walk.mp4 --epochs 30
  python -m anomaly.train --synthetic --epochs 30        # synthetic demo
"""

import argparse
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.config import settings
from anomaly.model import LSTMAutoencoder


# ── Synthetic dataset ────────────────────────────────────────────────────────

def generate_synthetic_normal(n_samples: int = 2000) -> np.ndarray:
    """
    Create fake NORMAL sequences: people walking in roughly straight lines
    with small, consistent velocities.

    Shape: (n_samples, seq_len, feature_dim)
    """
    seq_len    = settings.seq_len
    feat_dim   = settings.feature_dim
    sequences  = []

    for _ in range(n_samples):
        cx = random.uniform(0.1, 0.9)
        cy = random.uniform(0.2, 0.8)
        vx = random.uniform(-0.003, 0.003)
        vy = random.uniform(-0.001, 0.001)
        w  = random.uniform(0.04, 0.08)
        h  = random.uniform(0.12, 0.20)
        speed_base = np.sqrt(vx ** 2 + vy ** 2)

        frames = []
        prev_speed = speed_base
        for _ in range(seq_len):
            noise_v = random.gauss(0, 0.0005)
            cx = np.clip(cx + vx + random.gauss(0, 0.001), 0.02, 0.98)
            cy = np.clip(cy + vy + random.gauss(0, 0.0005), 0.02, 0.98)
            cur_speed = np.sqrt(vx ** 2 + vy ** 2) + abs(noise_v)
            accel     = abs(cur_speed - prev_speed)
            ar        = np.clip((w + random.gauss(0, 0.002)) /
                                (h + random.gauss(0, 0.002)), 0.3, 1.5)
            frames.append([cx, cy, w, h, vx + noise_v, vy, cur_speed, accel, ar])
            prev_speed = cur_speed

        sequences.append(frames)

    return np.array(sequences, dtype=np.float32)


def collect_sequences_from_video(video_path: str) -> np.ndarray:
    """
    Collect normal-behaviour sequences by running the full pipeline
    on a labelled-normal video (no anomaly detection, just feature extraction).
    """
    import cv2
    from detection.detector import PersonDetector
    from tracking.tracker import PersonTracker
    from anomaly.features import TrackFeatureBuffer

    detector = PersonDetector()
    tracker  = PersonTracker()
    buf      = TrackFeatureBuffer()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    sequences: List[np.ndarray] = []
    frame_no = 0

    print(f"[train] Extracting features from {video_path} …")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        h, w = frame.shape[:2]

        dets   = detector.detect(frame)
        tracks = tracker.update(dets, frame)

        for t in tracks:
            buf.update(t.track_id, t.bbox_xyxy, w, h)

        active_ids = {t.track_id for t in tracks}
        buf.prune(active_ids)

        if frame_no % settings.seq_len == 0:
            for t in tracks:
                seq = buf.get_sequence(t.track_id)
                if seq is not None:
                    sequences.append(seq)

    cap.release()
    print(f"[train] Collected {len(sequences)} sequences from {frame_no} frames.")

    if not sequences:
        raise RuntimeError("No sequences collected — video may have no persons or is too short.")

    return np.stack(sequences, axis=0)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_with_validation(
    train_sequences: np.ndarray,
    val_sequences: np.ndarray,
    epochs: int = 30,
    batch_size: int = 64,
    seed: int = 42,
    device_name: str | None = None,
    patience: int = 8,
) -> tuple[LSTMAutoencoder, dict]:
    """Train with an explicit clip-level validation split."""
    if len(train_sequences) == 0 or len(val_sequences) == 0:
        raise ValueError("Training and validation sequences must both be non-empty.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    requested_device = device_name or settings.yolo_device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    train_tensor = torch.tensor(train_sequences, dtype=torch.float32)
    val_tensor = torch.tensor(val_sequences, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    print(
        f"[train] Train: {len(train_sequences)} sequences; "
        f"validation: {len(val_sequences)}; device: {device}"
    )

    model = LSTMAutoencoder(
        input_size=settings.feature_dim,
        hidden_size=settings.lstm_hidden,
        num_layers=settings.lstm_layers,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[train] Model parameters: {parameter_count:,}; "
        f"batch size: {batch_size}; requested epochs: {epochs}",
        flush=True,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=3, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []
    training_started_at = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started_at = time.perf_counter()
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(train_sequences)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                reconstruction = model(batch)
                val_loss += criterion(reconstruction, batch).item() * len(batch)
        val_loss /= len(val_sequences)
        scheduler.step(val_loss)

        learning_rate = float(optimiser.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "learning_rate": learning_rate,
                "seconds": float(time.perf_counter() - epoch_started_at),
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"  Epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  "
            f"val={val_loss:.6f}  lr={learning_rate:.2e}  "
            f"time={history[-1]['seconds']:.2f}s",
            flush=True,
        )

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"[train] Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    errors = []
    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(device)
            reconstruction = model(batch)
            error = LSTMAutoencoder.reconstruction_error(batch, reconstruction)
            errors.extend(error.cpu().tolist())

    error_array = np.asarray(errors, dtype=np.float64)
    error_mean = float(error_array.mean())
    error_std = float(error_array.std())
    threshold = float(error_mean + 3 * error_std)
    metadata = {
        "seed": seed,
        "device": str(device),
        "parameter_count": parameter_count,
        "train_sequence_count": len(train_sequences),
        "validation_sequence_count": len(val_sequences),
        "epochs_completed": len(history),
        "training_seconds": float(time.perf_counter() - training_started_at),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "threshold": threshold,
        "threshold_method": "validation_mean_plus_3_std",
        "validation_error_mean": error_mean,
        "validation_error_std": error_std,
        "validation_error_p99": float(np.percentile(error_array, 99)),
        "history": history,
    }

    print(f"\n[train] Suggested anomaly_threshold = {threshold:.6f}", flush=True)
    print(
        f"        validation mean={error_mean:.6f}, std={error_std:.6f}",
        flush=True,
    )
    print(
        f"[train] Finished {len(history)} epochs in "
        f"{metadata['training_seconds']:.2f}s; best epoch={best_epoch}; "
        f"best val={best_val_loss:.6f}",
        flush=True,
    )
    return model, metadata


def train(sequences: np.ndarray, epochs: int = 30) -> LSTMAutoencoder:
    """Backward-compatible training helper for a single sequence array."""
    if len(sequences) < 2:
        raise ValueError("At least two sequences are required for training.")

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(sequences))
    val_size = max(1, int(len(sequences) * 0.1))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    model, _ = train_with_validation(
        sequences[train_indices],
        sequences[val_indices],
        epochs=epochs,
    )
    return model


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder")
    parser.add_argument("--source",    type=str, default=None, help="Path to normal-behaviour video")
    parser.add_argument("--synthetic", action="store_true",    help="Use synthetic data instead")
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--samples",   type=int, default=2000, help="Synthetic sample count")
    args = parser.parse_args()

    if args.synthetic or args.source is None:
        print("[train] Generating synthetic normal-behaviour dataset …")
        sequences = generate_synthetic_normal(n_samples=args.samples)
    else:
        sequences = collect_sequences_from_video(args.source)

    model = train(sequences, epochs=args.epochs)

    out_path = settings.anomaly_model_path
    model.save(out_path)
    print(f"\n[train] Model saved to {out_path}")


if __name__ == "__main__":
    main()
