"""Evaluate the visual fight head (stream A) on held-out test videos.

Same protocol as scripts/evaluate_fight.py (skeleton stream), so the two
streams compare directly:
  - frame level (UBI-Fights, UCF-Crime): each clip's probability goes to the
    frame where the clip ends (what a live system knows then); frames between
    are interpolated;
  - clip level (Surveillance Camera Fight, RLVS): highest clip score;
  - "<group>_short": videos without a fight cut to their first
    --short-normal-seconds, the shortened test set of skeleton runs 1-2;
  - "<group>_crowd": videos averaging >= --crowd-threshold tracked people,
    counted from the skeleton stream's pose caches when --pose-root is given;
  - the median AUC inside each fight video (does it find the fight, or only
    the fight video?).
Every metric is also reported for the motion-energy baseline
("motion_baseline_*"); a model that learnt fights, not movement, beats it
inside videos.

The threshold comes from validation; test labels never tune it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomaly.fight_data import load_items
from anomaly.visual_fight import ClipFeatures, FightTemporalHead, predict_video
from scripts.evaluate_fight import CLIP_LEVEL_DATASETS, FRAME_LEVEL_DATASETS, save_curves, save_timelines, write_json
from scripts.evaluation_metrics import aggregate_frame_scores, binary_metrics


def mean_people(pose_path: Path) -> float | None:
    """Tracked people per sampled frame from a skeleton-stream pose cache, if there is one."""
    if not pose_path.exists():
        return None
    with np.load(pose_path) as data:
        samples = len(data["sample_frames"])
        return float(len(data["track_ids"]) / samples) if samples else 0.0


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the visual fight head")
    parser.add_argument("--dataset", action="append", required=True, help="name:root[:split]")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pose-root", default=None, help="Skeleton pose caches, only to form the crowd groups")
    parser.add_argument("--crowd-threshold", type=float, default=5.0)
    parser.add_argument("--short-normal-seconds", type=float, default=None)
    args = parser.parse_args()

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[evaluate-visual] CUDA unavailable; falling back to CPU.", flush=True)
        device_name = "cpu"
    device = torch.device(device_name)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_root = Path(args.feature_root).resolve()
    pose_root = Path(args.pose_root).resolve() if args.pose_root else None

    model = FightTemporalHead.load(Path(args.model), device=str(device))
    metadata = torch.load(args.model, map_location="cpu", weights_only=True)["metadata"]
    threshold_payload = json.loads(Path(args.threshold).read_text(encoding="utf-8"))
    threshold = float(threshold_payload["threshold"])

    items = [item for spec in args.dataset for item in load_items(spec) if item.split == "test"]
    frame_data = defaultdict(lambda: {"labels": [], "scores": [], "motion": [], "video_aucs": [], "motion_aucs": []})
    clip_data = defaultdict(lambda: {"labels": [], "scores": [], "motion": []})
    timelines = defaultdict(list)
    video_rows, video_counts, missing = [], defaultdict(int), 0

    print(f"[evaluate-visual] {len(items)} test video(s), threshold={threshold:.4f}", flush=True)
    for index, item in enumerate(items, start=1):
        path = feature_root / item.dataset / f"{item.key}.npz"
        if not path.exists():
            missing += 1
            continue
        clips = ClipFeatures.load(path)
        if len(clips) == 0 or clips.frame_count == 0:
            missing += 1
            continue
        probabilities = predict_video(model, clips.features, device)
        motion = clips.motion.astype(np.float64)
        labels = item.labels_for(clips.frame_count)
        people = mean_people(pose_root / item.dataset / f"{item.key}.npz") if pose_root else None
        groups = [item.dataset]
        if people is not None and people >= args.crowd_threshold:
            groups.append(f"{item.dataset}_crowd")
        for group in groups:
            video_counts[group] += 1

        video_score = float(probabilities.max())
        row = {
            "dataset": item.dataset,
            "video": item.key,
            "fight_frames": int(labels.sum()),
            "frames": int(clips.frame_count),
            "mean_people": None if people is None else round(people, 2),
            "video_score": round(video_score, 6),
            "video_label": int(labels.any()),
        }
        if item.dataset in CLIP_LEVEL_DATASETS:
            for group in groups:
                clip_data[group]["labels"].append(int(labels.any()))
                clip_data[group]["scores"].append(video_score)
                clip_data[group]["motion"].append(float(motion.max()))
        if item.dataset in FRAME_LEVEL_DATASETS:
            ends = clips.last_frames + 1
            frame_scores, _ = aggregate_frame_scores(probabilities, ends, clips.frame_count)
            motion_scores, _ = aggregate_frame_scores(motion, ends, clips.frame_count)
            row["frame_auc"] = row["motion_baseline_frame_auc"] = None
            if 0 < labels.sum() < len(labels):
                row["frame_auc"] = round(safe_auc(labels, frame_scores), 6)
                row["motion_baseline_frame_auc"] = round(safe_auc(labels, motion_scores), 6)
            kept = len(labels)
            if args.short_normal_seconds and not labels.any():
                kept = min(kept, int(round(args.short_normal_seconds * clips.fps)))
            variants = [(group, len(labels)) for group in groups]
            if args.short_normal_seconds:
                variants += [(f"{group}_short", kept) for group in groups]
            for group, frames in variants:
                frame_data[group]["labels"].append(labels[:frames])
                frame_data[group]["scores"].append(frame_scores[:frames])
                frame_data[group]["motion"].append(motion_scores[:frames])
                if row["frame_auc"] is not None:
                    frame_data[group]["video_aucs"].append(row["frame_auc"])
                    frame_data[group]["motion_aucs"].append(row["motion_baseline_frame_auc"])
            if labels.any() and len(timelines[item.dataset]) < 9:
                timelines[item.dataset].append(
                    {"video": item.key, "labels": labels, "scores": frame_scores, "fps": clips.fps}
                )
        video_rows.append(row)
        if index % 20 == 0 or index == len(items):
            print(f"[{index}/{len(items)}] {item.uid} score={video_score:.3f}", flush=True)

    report = {
        "threshold": threshold,
        "threshold_source": threshold_payload.get("method", "validation"),
        "extraction": metadata.get("extraction"),
        "crowd_threshold_people": args.crowd_threshold,
        "crowd_groups": pose_root is not None,
        "videos_per_group": dict(video_counts),
        "missing_feature_caches": missing,
        "official_protocol": missing == 0,
        "frame_level": {},
        "clip_level": {},
    }
    for group, data in frame_data.items():
        labels = np.concatenate(data["labels"])
        scores = np.concatenate(data["scores"])
        metrics = binary_metrics(labels, scores, threshold)
        metrics["fight_videos"] = len(data["video_aucs"])
        metrics["fight_video_frame_auc_median"] = float(np.median(data["video_aucs"])) if data["video_aucs"] else None
        metrics["motion_baseline_roc_auc"] = safe_auc(labels, np.concatenate(data["motion"]))
        metrics["motion_baseline_fight_video_frame_auc_median"] = (
            float(np.median(data["motion_aucs"])) if data["motion_aucs"] else None
        )
        report["frame_level"][group] = metrics
        save_curves(output_dir, f"{group}_frame", labels, scores, metrics)
        save_timelines(output_dir, group, timelines[group], threshold)
    for group, data in clip_data.items():
        labels = np.asarray(data["labels"], dtype=np.int32)
        scores = np.asarray(data["scores"], dtype=np.float64)
        metrics = binary_metrics(labels, scores, threshold)
        metrics["motion_baseline_roc_auc"] = safe_auc(labels, np.asarray(data["motion"]))
        report["clip_level"][group] = metrics
        save_curves(output_dir, f"{group}_clip", labels, scores, metrics)

    write_json(output_dir / "evaluation.json", report)
    if video_rows:
        fieldnames = list(dict.fromkeys(key for row in video_rows for key in row))
        with (output_dir / "per_video.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(video_rows)

    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}"

    for level in ("frame_level", "clip_level"):
        for group, metrics in report[level].items():
            print(
                f"[evaluate-visual] {group} {level}: AUC={fmt(metrics['roc_auc'])} "
                f"(motion baseline {fmt(metrics['motion_baseline_roc_auc'])}) AP={fmt(metrics['average_precision'])} "
                f"balanced accuracy={metrics['balanced_accuracy']:.4f}",
                flush=True,
            )
            if metrics.get("fight_video_frame_auc_median") is not None:
                print(
                    f"[evaluate-visual]   in-video AUC (median of {metrics['fight_videos']} fight videos): "
                    f"model={fmt(metrics['fight_video_frame_auc_median'])} "
                    f"motion baseline={fmt(metrics['motion_baseline_fight_video_frame_auc_median'])}",
                    flush=True,
                )
    if missing:
        print(f"[evaluate-visual] WARNING: {missing} test video(s) without features.", flush=True)
    print(f"[evaluate-visual] Reports saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
