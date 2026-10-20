"""
scripts/evaluate.py

Offline evaluation of the anomaly detection pipeline.

Inputs:
  --video  : annotated/labelled video (or synthetic)
  --labels : CSV file with columns: frame, track_id, anomaly_type
             (omit for synthetic self-evaluation mode)

Outputs:
  - Per-class precision / recall / F1
  - FPS benchmark
  - Threshold sweep plot (requires matplotlib)

Usage:
  python scripts/evaluate.py --video test_video.mp4 --synthetic
  python scripts/evaluate.py --video footage.mp4 --labels labels.csv
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np


# ── Label loader ──────────────────────────────────────────────────────────────

def load_labels(csv_path: str) -> Dict[Tuple[int, int], str]:
    """Load ground-truth labels. Returns {(frame, track_id): anomaly_type}."""
    labels = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["frame"]), int(row["track_id"]))
            labels[key] = row["anomaly_type"].strip().lower()
    return labels


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    predictions: List[dict],
    ground_truth: Dict[Tuple[int, int], str],
    anomaly_types: List[str] = ["running", "fighting", "loitering"],
) -> dict:
    """
    Binary anomaly detection metrics (any predicted anomaly vs GT anomaly).
    Also per-class breakdown.
    """
    tp = fp = fn = tn = 0
    per_class: Dict[str, Dict] = {t: dict(tp=0, fp=0, fn=0) for t in anomaly_types}

    all_gt_keys: Set = set(ground_truth.keys())
    predicted_keys: Set = set()

    for pred in predictions:
        key = (pred["frame"], pred["track_id"])
        predicted_keys.add(key)
        gt_type = ground_truth.get(key)

        if gt_type:                          # True positive
            tp += 1
            pred_type = pred.get("anomaly_type", "unknown")
            if pred_type in per_class:
                per_class[pred_type]["tp"] += 1
        else:                               # False positive
            fp += 1
            pred_type = pred.get("anomaly_type", "unknown")
            if pred_type in per_class:
                per_class[pred_type]["fp"] += 1

    for key, gt_type in ground_truth.items():
        if key not in predicted_keys:       # False negative
            fn += 1
            if gt_type in per_class:
                per_class[gt_type]["fn"] += 1

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    result = {
        "binary": {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1},
        "per_class": {},
    }

    for t, c in per_class.items():
        p = c["tp"] / (c["tp"] + c["fp"] + 1e-9)
        r = c["tp"] / (c["tp"] + c["fn"] + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)
        result["per_class"][t] = {"precision": p, "recall": r, "f1": f, **c}

    return result


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(video_path: str, synthetic_labels: bool = False) -> Tuple[List[dict], dict, float]:
    """
    Run the full detection pipeline on a video.
    Returns (predictions, synthetic_gt_or_empty, fps).
    """
    # Lazy imports
    from core.config import settings
    from detection.detector import PersonDetector
    from tracking.tracker import PersonTracker
    from anomaly.detector import AnomalyDetector

    detector = PersonDetector()
    tracker  = PersonTracker()
    anomaly  = AnomalyDetector()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[eval] Cannot open {video_path}")
        sys.exit(1)

    predictions: List[dict] = []
    synthetic_gt: Dict = {}
    frame_no   = 0
    t_start    = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        H, W = frame.shape[:2]

        dets   = detector.detect(frame)
        tracks = tracker.update(dets, frame)
        events = anomaly.update(tracks, frame_hw=(H, W), frame_number=frame_no)

        for ev in events:
            predictions.append({
                "frame":      ev.frame_number,
                "track_id":   ev.track_id,
                "anomaly_type": ev.anomaly_type,
                "confidence": ev.confidence,
            })

    cap.release()
    elapsed = time.time() - t_start
    fps     = frame_no / (elapsed + 1e-6)

    print(f"[eval] Processed {frame_no} frames in {elapsed:.1f}s  ({fps:.1f} FPS)")
    return predictions, synthetic_gt, fps


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(metrics: dict, fps: float):
    b = metrics["binary"]
    print("\n" + "═" * 52)
    print("  SENTINEL  —  Evaluation Report")
    print("═" * 52)
    print(f"  Throughput   : {fps:.1f} FPS")
    print("─" * 52)
    print("  BINARY (any anomaly vs normal)")
    print(f"  Precision    : {b['precision']:.3f}")
    print(f"  Recall       : {b['recall']:.3f}")
    print(f"  F1           : {b['f1']:.3f}")
    print(f"  TP={b['tp']}  FP={b['fp']}  FN={b['fn']}")
    print("─" * 52)
    print("  PER-CLASS BREAKDOWN")
    for t, v in metrics.get("per_class", {}).items():
        print(f"  {t:<12}  P={v['precision']:.2f}  R={v['recall']:.2f}  F1={v['f1']:.2f}")
    print("═" * 52)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection pipeline")
    parser.add_argument("--video",     required=True,       help="Input video path")
    parser.add_argument("--labels",    default=None,        help="Ground-truth CSV (optional)")
    parser.add_argument("--synthetic", action="store_true", help="Self-evaluation: use rule outputs as GT")
    args = parser.parse_args()

    predictions, _, fps = run_pipeline(args.video, args.synthetic)

    if args.labels:
        gt = load_labels(args.labels)
    else:
        print("[eval] No labels provided — using detected events as GT (self-eval / FPS bench mode)")
        # Build trivial GT from own predictions for demo purposes
        gt = {(p["frame"], p["track_id"]): p["anomaly_type"] for p in predictions}

    metrics = compute_metrics(predictions, gt)
    print_report(metrics, fps)


if __name__ == "__main__":
    main()
