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
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

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

def train(sequences: np.ndarray, epochs: int = 30) -> LSTMAutoencoder:
    """Train the LSTM Autoencoder on the provided normal sequences."""
    print(f"[train] Dataset: {sequences.shape[0]} sequences "
          f"× {sequences.shape[1]} frames × {sequences.shape[2]} features")

    X = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(X)

    val_size   = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    model = LSTMAutoencoder(
        input_size=settings.feature_dim,
        hidden_size=settings.lstm_hidden,
        num_layers=settings.lstm_layers,
    )

    device    = torch.device(settings.yolo_device if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item() * len(batch)

        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                val_loss += criterion(recon, batch).item() * len(batch)
        val_loss /= val_size

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  train={train_loss:.5f}  val={val_loss:.5f}")

    if best_state:
        model.load_state_dict(best_state)

    # Calibrate threshold on validation set
    model.eval()
    errors = []
    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(device)
            recon = model(batch)
            err   = LSTMAutoencoder.reconstruction_error(batch, recon)
            errors.extend(err.cpu().tolist())

    # Threshold = mean + 3 × std of reconstruction errors on normal data
    err_arr   = np.array(errors)
    threshold = float(err_arr.mean() + 3 * err_arr.std())
    print(f"\n[train] Suggested anomaly_threshold = {threshold:.5f}")
    print(f"        (mean={err_arr.mean():.5f}, std={err_arr.std():.5f})")
    print(f"        Update settings.anomaly_threshold in core/config.py")

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
