"""Bounded larger training with separate development, calibration and frozen test."""

import argparse
import hashlib
import io
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from detection.detector import PersonDetector
from fight.model import VideoClassifier
from fight.vision import ClipConfig, DetectorConfig
from scripts.fight_ablation import cached_rows, export_runtime, stable_train
from scripts.fight_pilot import metrics, score_rows, write_json
from scripts.fight_public_data import SPLITS


def operating_point(rows, scores, target=0.8):
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    scores = np.asarray(scores, dtype=float)
    if (
        not len(labels)
        or len(labels) != len(scores)
        or not np.isfinite(scores).all()
        or set(labels) != {0, 1}
    ):
        raise ValueError("Calibration needs finite scores and both classes")
    candidates = np.unique(np.clip(np.r_[scores, 0.5], 1e-6, 1 - 1e-6))
    decisions = scores[:, None] >= candidates
    true_positive = (decisions & labels[:, None].astype(bool)).sum(axis=0)
    precision = true_positive / np.maximum(decisions.sum(axis=0), 1)
    recall = true_positive / labels.sum()
    feasible = np.flatnonzero((precision >= target) & (recall >= target))
    pool = feasible if len(feasible) else range(len(candidates))
    selected = max(
        pool,
        key=lambda index: (
            min(precision[index], recall[index]),
            2
            * precision[index]
            * recall[index]
            / max(precision[index] + recall[index], 1e-12),
            candidates[index],
        ),
    )
    threshold = float(candidates[selected])
    return {
        "threshold": threshold,
        "target_precision_and_recall": target,
        "target_feasible_on_calibration": bool(len(feasible)),
        "calibration": measured(rows, scores, threshold),
    }


def wilson(success, total):
    if total == 0:
        return None
    proportion = success / total
    denominator = 1 + 1.96**2 / total
    center = (proportion + 1.96**2 / (2 * total)) / denominator
    margin = (
        1.96
        * ((proportion * (1 - proportion) + 1.96**2 / (4 * total)) / total) ** 0.5
        / denominator
    )
    return [max(0, center - margin), min(1, center + margin)]


def measured(rows, scores, threshold):
    result = metrics(rows, scores, threshold)
    labels = np.asarray([row["label"] for row in rows], dtype=bool)
    predictions = np.asarray(scores) >= threshold
    true_positive = int((labels & predictions).sum())
    true_negative = int((~labels & ~predictions).sum())
    false_positive = int((~labels & predictions).sum())
    false_negative = int((labels & ~predictions).sum())
    return {
        **result,
        "metric_scope": "One central model window per public clip; publisher clip labels, not frame/event or phone accuracy",
        "accuracy": float((labels == predictions).mean()),
        "confusion_matrix": {
            "tp": true_positive,
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
        },
        "precision_wilson95": wilson(true_positive, true_positive + false_positive),
        "recall_wilson95": wilson(true_positive, true_positive + false_negative),
    }


def validate_manifest(manifest):
    rows = manifest["rows"]
    if len({row["video"] for row in rows}) != len(rows):
        raise ValueError("Duplicate video identities")
    for key in ("video", "scene_group", "sha256", "performance"):
        partitions = {}
        for row in rows:
            if row.get(key) is not None:
                partitions.setdefault(row[key], set()).add(row["split"])
        if any(len(values) > 1 for values in partitions.values()):
            raise ValueError(f"Split leakage: {key}")
    for split in SPLITS:
        if {row["label"] for row in rows if row["split"] == split} != {0, 1}:
            raise ValueError(f"Both classes required: {split}")


def people_focused_training(rows, cache):
    foreground, background = [], []
    for row in rows:
        with np.load(cache / f"{row['cache']}.npz") as cached:
            people = int(cached["people"])
        (background if row["label"] == 0 and people == 0 else foreground).append(row)
    limit = round(len(foreground) * 0.02 / 0.98)
    background.sort(key=lambda row: row["sha256"])
    kept = foreground + background[:limit]
    return kept, {
        "original_train_clips": len(rows),
        "actual_train_clips": len(kept),
        "positive_clips": sum(row["label"] for row in kept),
        "negative_without_people_kept": min(limit, len(background)),
        "negative_without_people_excluded": max(0, len(background) - limit),
        "note": "Only training background negatives capped at ~2%; positives retained even with missing detections. Development/calibration/test never filtered by detection.",
    }


def run(args):
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("Completed test already exists; no automatic reuse for tuning")
    if (output / "run.json").exists():
        raise ValueError(
            "Existing run requires explicit new output directory, not silent restart"
        )
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    run_state = {
        "status": "running",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "counts": manifest["counts"],
        "budget_minutes": args.budget_minutes,
        "approved_for_demo": False,
    }
    write_json(output / "run.json", run_state)
    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    config = ClipConfig(**manifest["clip"])
    detector_config = DetectorConfig()
    args.cache.mkdir(parents=True, exist_ok=True)
    model = VideoClassifier("x3d_s", pretrained=True).cuda().eval()
    detector = PersonDetector(**detector_config.to_dict(), device="cuda")
    try:
        development_rows = [row for row in manifest["rows"] if row["split"] != "test"]
        ready = cached_rows(
            development_rows,
            model,
            detector,
            config,
            detector_config,
            args.cache,
            "cuda",
            deadline - 600,
        )
        partitions = {
            split: [row for row in ready if row["split"] == split]
            for split in SPLITS[:-1]
        }
        partitions["train"], train_selection = people_focused_training(
            partitions["train"], args.cache
        )
        write_json(output / "train_selection.json", train_selection)
        print("TRAIN_SELECTION " + json.dumps(train_selection), flush=True)
        write_json(output / "cached_manifest.json", {**manifest, "rows": ready})
        try:
            stable_train(
                model,
                partitions["train"],
                partitions["development"],
                args.cache,
                "cuda",
                deadline - 300,
                output,
                epochs=(8, 8),
                camera_augmentation=True,
            )
        except TimeoutError:
            if not (output / "best.pt").exists():
                raise
            model.load_state_dict(
                torch.load(output / "best.pt", map_location="cuda", weights_only=True)[
                    "state_dict"
                ]
            )
            run_state["training_budget_exhausted_best_completed_epoch_restored"] = True
        calibration_scores = score_rows(
            model, partitions["calibration"], args.cache, "cuda", False
        )
        calibration = operating_point(partitions["calibration"], calibration_scores)
        print("CALIBRATION_FROZEN " + json.dumps(calibration), flush=True)
        write_json(output / "calibration_frozen.json", calibration)
        test_rows = cached_rows(
            [row for row in manifest["rows"] if row["split"] == "test"],
            model,
            detector,
            config,
            detector_config,
            args.cache,
            "cuda",
            deadline - 90,
        )
        test_scores = score_rows(model, test_rows, args.cache, "cuda", False)
        threshold = calibration["threshold"]
        report = {
            **calibration,
            "test": measured(test_rows, test_scores, threshold),
            "test_by_dataset": {
                dataset: measured(
                    [row for row in test_rows if row["dataset"] == dataset],
                    test_scores[[row["dataset"] == dataset for row in test_rows]],
                    threshold,
                )
                for dataset in sorted({row["dataset"] for row in test_rows})
            },
            "counts": manifest["counts"],
            "train_selection": train_selection,
            "label_provenance": manifest["label_provenance"],
            "grouping_note": manifest["grouping_note"],
            "manifest_sha256": run_state["manifest_sha256"],
            "approved_for_demo": False,
            "real_phone_tested": False,
            "scope": "Public clip-level grouped holdout; not guaranteed classroom/crowd/handheld event detection",
            "event_gate": {
                "confirm_seconds": 0.8,
                "release_seconds": 1.0,
                "max_observation_gap": 1.5,
            },
        }
        report["public_clip_target_met"] = bool(
            report["test"]["precision"] >= 0.8 and report["test"]["recall"] >= 0.8
        )
        report["decision"] = (
            "candidate_requires_video_event_and_real_phone_acceptance"
            if report["public_clip_target_met"]
            else "public_clip_target_not_met"
        )
        write_json(
            output / "test_predictions.json",
            [
                {
                    "video": row["video"],
                    "split": "test",
                    "label": row["label"],
                    "score": float(score),
                    "prediction": int(score >= threshold),
                }
                for row, score in zip(test_rows, test_scores)
            ],
        )
        export_runtime(model, config, detector_config, report, output)
        with zipfile.ZipFile(
            output / "frozen_features.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for row in ready:
                with np.load(args.cache / f"{row['cache']}.npz") as cached:
                    features = cached["features"]
                buffer = io.BytesIO()
                np.save(buffer, features, allow_pickle=False)
                archive.writestr(row["cache"] + ".npy", buffer.getvalue())
        report["seconds"] = time.monotonic() - started
        write_json(output / "report.json", report)
        run_state["status"] = "completed"
        print("LARGE_RESULT " + json.dumps(report), flush=True)
    except BaseException as error:
        run_state.update(status="incomplete", reason=str(error))
        raise
    finally:
        run_state["seconds"] = time.monotonic() - started
        write_json(output / "run.json", run_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--budget-minutes", type=float, default=45)
    args = parser.parse_args()
    if args.budget_minutes < 15 or not torch.cuda.is_available():
        parser.error("CUDA and at least 15 minutes required")
    run(args)


if __name__ == "__main__":
    main()
