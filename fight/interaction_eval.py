"""Cohort-aware calibration and explicit temporal event diagnostics."""

import numpy as np
from sklearn.metrics import average_precision_score

from scripts.fight_large_train import measured


def macro_ap(rows, scores):
    values = []
    for dataset in sorted({row["dataset"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["dataset"] == dataset]
        labels = [rows[index]["label"] for index in indices]
        if set(labels) != {0, 1}:
            raise ValueError("Both classes required in each selection cohort")
        values.append(
            float(average_precision_score(labels, np.asarray(scores)[indices]))
        )
    return float(np.mean(values))


def balanced_calibration(rows, scores, target=0.8):
    scores = np.asarray(scores, dtype=float)
    if (
        len(scores) != len(rows)
        or not len(rows)
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("Finite matching probability scores required")
    candidates = np.unique(np.clip(np.r_[scores, 0.5], 1e-6, 1 - 1e-6))
    cohort_values = []
    names = sorted({row["dataset"] for row in rows})
    for dataset in names:
        indices = [index for index, row in enumerate(rows) if row["dataset"] == dataset]
        labels = np.array([rows[index]["label"] for index in indices], dtype=bool)
        if set(labels) != {False, True}:
            raise ValueError("Both classes required in calibration cohorts")
        predictions = scores[indices, None] >= candidates
        true_positive = (predictions & labels[:, None]).sum(axis=0)
        precision = true_positive / np.maximum(1, predictions.sum(axis=0))
        recall = true_positive / labels.sum()
        cohort_values.append(np.minimum(precision, recall))
    worst = np.min(cohort_values, axis=0)
    selected = max(
        range(len(candidates)),
        key=lambda index: (
            worst[index],
            float(np.mean([values[index] for values in cohort_values])),
            candidates[index],
        ),
    )
    threshold = float(candidates[selected])
    return {
        "threshold": threshold,
        "target": target,
        "all_calibration_cohorts_meet_target": bool(worst[selected] >= target),
        "selection": "Maximize worst dataset precision/recall with one global threshold, calibration only",
        "cohorts": {
            dataset: measured(
                [row for row in rows if row["dataset"] == dataset],
                scores[[row["dataset"] == dataset for row in rows]],
                threshold,
            )
            for dataset in names
        },
    }


def event_metrics(events, intervals, fps, duration, tolerance=1.0, minimum_iou=0.1):
    positives = [
        (row["start"] / fps, (row["end"] + 1) / fps)
        for row in intervals
        if row["label"] == 1
    ]
    ignored = [
        (row["start"] / fps, (row["end"] + 1) / fps)
        for row in intervals
        if row["label"] is None
    ]
    matched, delays, false_positive, unscored = set(), [], 0, 0
    for event in sorted(events, key=lambda row: row["start"]):
        available = []
        for index, (start, end) in enumerate(positives):
            expanded_end = min(duration, end + tolerance)
            intersection = max(
                0.0, min(event["end"], expanded_end) - max(event["start"], start)
            )
            union = max(event["end"], expanded_end) - min(event["start"], start)
            if (
                index not in matched
                and event["start"] >= start - 0.5
                and intersection / max(union, 1e-9) >= minimum_iou
            ):
                available.append(index)
        if available:
            index = min(
                available, key=lambda index: abs(event["start"] - positives[index][0])
            )
            matched.add(index)
            delays.append(max(0.0, event["start"] - positives[index][0]))
        elif any(
            event["end"] >= start and event["start"] <= end for start, end in ignored
        ):
            unscored += 1
        else:
            false_positive += 1
    excluded = sorted(positives + ignored)
    merged = []
    for start, end in excluded:
        start, end = max(0, start), min(duration, end)
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        elif end > start:
            merged.append([start, end])
    normal_minutes = max(0.0, duration - sum(end - start for start, end in merged)) / 60
    true_positive = len(matched)
    return {
        "ground_truth_events": len(positives),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": len(positives) - true_positive,
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, len(positives)),
        "false_alerts_per_normal_minute": (
            false_positive / normal_minutes if normal_minutes > 0 else None
        ),
        "normal_minutes": normal_minutes,
        "ignored_events": unscored,
        "start_delays_seconds": delays,
        "end_tolerance_seconds": tolerance,
        "minimum_iou_with_end_tolerance": minimum_iou,
        "scope": "Small source-recording diagnostic; publisher action intervals, one-to-one event matching, not physical-phone acceptance",
    }
