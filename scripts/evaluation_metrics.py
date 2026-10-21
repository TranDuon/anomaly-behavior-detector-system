"""Pure frame-level aggregation and metric helpers for anomaly evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def aggregate_frame_scores(
    sequence_scores: np.ndarray,
    frame_numbers: np.ndarray,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if len(sequence_scores) != len(frame_numbers):
        raise ValueError("sequence_scores and frame_numbers must have equal length")
    if not np.all(np.isfinite(sequence_scores)):
        raise ValueError("sequence_scores contains NaN or infinity")
    observed_scores = np.full(frame_count, np.nan, dtype=np.float64)
    for frame_number, score in zip(frame_numbers, sequence_scores):
        index = int(frame_number) - 1
        if not 0 <= index < frame_count:
            raise ValueError(
                f"Sequence frame {frame_number} is outside 1..{frame_count}"
            )
        if np.isnan(observed_scores[index]) or score > observed_scores[index]:
            observed_scores[index] = score

    observed = ~np.isnan(observed_scores)
    if not np.any(observed):
        raise RuntimeError("No frame received an anomaly score.")

    frame_indices = np.arange(frame_count)
    interpolated = np.interp(
        frame_indices,
        frame_indices[observed],
        observed_scores[observed],
    )
    return interpolated, observed


def binary_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    predictions = (scores > threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    metrics = {
        "frames": int(len(labels)),
        "normal_frames": int(np.sum(labels == 0)),
        "anomaly_frames": int(np.sum(labels == 1)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["average_precision"] = float(
            average_precision_score(labels, scores)
        )
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics
