"""Evaluate the fight ST-GCN on held-out test videos.

Frame level (UBI-Fights test split, UCF-Crime Fighting + Normal test videos):
each window's fight probability is assigned to the frame where the window
ends (what a live system knows at that moment); frames between window ends
are interpolated, as in scripts/evaluation_metrics.py.

Clip level (every dataset): a video's score is its highest window score, its
label whether it contains any fight. This is the protocol of clip datasets
such as Surveillance Camera Fight.

Crowd subset: videos whose pose cache averages at least --crowd-threshold
tracked people per sampled frame are also reported on their own
("<dataset>_crowd"), since pairing and occlusion fail first in crowds.

The threshold comes from validation (fight_threshold.json); test labels are
never used to tune it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomaly.fight_data import load_items
from anomaly.skeleton import PoseTrack, window_starts
from anomaly.stgcn import PairBagClassifier, predict_windows
from scripts.evaluation_metrics import aggregate_frame_scores, binary_metrics


FRAME_LEVEL_DATASETS = {"ubi", "ucf"}


def save_curves(output_dir: Path, name: str, labels: np.ndarray, scores: np.ndarray, metrics: dict) -> None:
    if metrics.get("roc_auc") is None:
        return
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    left.plot(false_positive_rate, true_positive_rate, label=f"AUROC = {metrics['roc_auc']:.4f}")
    left.plot([0, 1], [0, 1], "--", color="gray")
    left.set_xlabel("False positive rate")
    left.set_ylabel("True positive rate")
    left.set_title(f"{name} ROC")
    right.plot(recall, precision, label=f"AP = {metrics['average_precision']:.4f}")
    right.set_xlabel("Recall")
    right.set_ylabel("Precision")
    right.set_title(f"{name} precision-recall")
    for axis in (left, right):
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}_curves.png", dpi=160)
    plt.close(figure)


def save_timelines(output_dir: Path, name: str, payloads: list[dict], threshold: float) -> None:
    if not payloads:
        return
    columns = 3
    rows = int(np.ceil(len(payloads) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 3.2 * rows), squeeze=False)
    for axis, payload in zip(axes.flat, payloads):
        seconds = np.arange(len(payload["scores"])) / payload["fps"]
        axis.plot(seconds, payload["scores"], color="tab:blue", linewidth=1)
        axis.axhline(threshold, color="tab:red", linestyle="--", linewidth=1)
        axis.fill_between(
            seconds, 0, 1, where=payload["labels"].astype(bool),
            color="orange", alpha=0.25, transform=axis.get_xaxis_transform(),
        )
        axis.set_ylim(0, 1)
        axis.set_title(payload["video"])
        axis.set_xlabel("Second")
        axis.set_ylabel("Fight probability")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(payloads):]:
        axis.axis("off")
    figure.suptitle(f"{name}: frame scores; orange = ground-truth fight", y=1.01)
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}_timelines.png", dpi=140, bbox_inches="tight")
    plt.close(figure)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fight ST-GCN")
    parser.add_argument("--dataset", action="append", required=True, help="name:root[:split]")
    parser.add_argument("--pose-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stride", type=int, default=1, help="Window stride in samples")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--crowd-threshold", type=float, default=5.0,
        help="Mean tracked people per sampled frame for the crowd subset",
    )
    args = parser.parse_args()

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[evaluate-fight] CUDA unavailable; falling back to CPU.", flush=True)
        device_name = "cpu"
    device = torch.device(device_name)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_root = Path(args.pose_root).resolve()

    model = PairBagClassifier.load(Path(args.model), device=str(device))
    window = torch.load(args.model, map_location="cpu", weights_only=True)["metadata"]["window"]
    threshold_payload = json.loads(Path(args.threshold).read_text(encoding="utf-8"))
    threshold = float(threshold_payload["threshold"])

    items = [item for spec in args.dataset for item in load_items(spec) if item.split == "test"]
    frame_data = defaultdict(lambda: {"labels": [], "scores": []})
    clip_data = defaultdict(lambda: {"labels": [], "scores": []})
    timelines = defaultdict(list)
    video_rows = []
    missing = 0

    print(f"[evaluate-fight] {len(items)} test video(s), threshold={threshold:.4f}", flush=True)
    for index, item in enumerate(items, start=1):
        pose_path = pose_root / item.dataset / f"{item.key}.npz"
        if not pose_path.exists():
            missing += 1
            continue
        pose = PoseTrack.load(pose_path)
        if pose.num_samples == 0 or pose.frame_count == 0:
            missing += 1
            continue
        starts = window_starts(pose.num_samples, window["length"], args.stride)
        probabilities = predict_windows(model, pose, starts, window, device, args.batch_size)
        end_samples = np.minimum(starts + window["length"] - 1, pose.num_samples - 1)
        end_frames = pose.sample_frames[end_samples] + 1
        labels = item.labels_for(pose.frame_count)
        mean_people = len(pose.track_ids) / pose.num_samples
        groups = [item.dataset]
        if mean_people >= args.crowd_threshold:
            groups.append(f"{item.dataset}_crowd")

        video_score = float(probabilities.max())
        for group in groups:
            clip_data[group]["labels"].append(int(labels.any()))
            clip_data[group]["scores"].append(video_score)
        row = {
            "dataset": item.dataset,
            "video": item.key,
            "fight_frames": int(labels.sum()),
            "frames": int(pose.frame_count),
            "mean_people": round(mean_people, 2),
            "video_score": round(video_score, 6),
            "video_label": int(labels.any()),
        }

        if item.dataset in FRAME_LEVEL_DATASETS:
            frame_scores, _ = aggregate_frame_scores(probabilities, end_frames, pose.frame_count)
            for group in groups:
                frame_data[group]["labels"].append(labels)
                frame_data[group]["scores"].append(frame_scores)
            row["frame_auc"] = None
            if 0 < labels.sum() < len(labels):
                row["frame_auc"] = round(binary_metrics(labels, frame_scores, threshold)["roc_auc"], 6)
            if labels.any() and len(timelines[item.dataset]) < 9:
                timelines[item.dataset].append(
                    {"video": item.key, "labels": labels, "scores": frame_scores, "fps": pose.fps}
                )
        video_rows.append(row)
        if index % 20 == 0 or index == len(items):
            print(f"[{index}/{len(items)}] {item.uid} score={video_score:.3f}", flush=True)

    report = {
        "threshold": threshold,
        "threshold_source": threshold_payload.get("method", "validation"),
        "window": window,
        "stride_samples": args.stride,
        "crowd_threshold_people": args.crowd_threshold,
        "videos_per_group": {group: len(data["labels"]) for group, data in clip_data.items()},
        "missing_pose_caches": missing,
        "frame_level": {},
        "clip_level": {},
    }
    for dataset, data in frame_data.items():
        labels = np.concatenate(data["labels"])
        scores = np.concatenate(data["scores"])
        metrics = binary_metrics(labels, scores, threshold)
        report["frame_level"][dataset] = metrics
        save_curves(output_dir, f"{dataset}_frame", labels, scores, metrics)
        save_timelines(output_dir, dataset, timelines[dataset], threshold)
    for dataset, data in clip_data.items():
        labels = np.asarray(data["labels"], dtype=np.int32)
        scores = np.asarray(data["scores"], dtype=np.float64)
        metrics = binary_metrics(labels, scores, threshold)
        report["clip_level"][dataset] = metrics
        save_curves(output_dir, f"{dataset}_clip", labels, scores, metrics)

    write_json(output_dir / "evaluation.json", report)
    if video_rows:
        fieldnames = list(dict.fromkeys(key for row in video_rows for key in row))
        with (output_dir / "per_video.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(video_rows)

    for level in ("frame_level", "clip_level"):
        for dataset, metrics in report[level].items():
            auc = metrics["roc_auc"]
            ap = metrics["average_precision"]
            print(
                f"[evaluate-fight] {dataset} {level}: AUC={auc if auc is None else round(auc, 4)} "
                f"AP={ap if ap is None else round(ap, 4)} F1={metrics['f1']:.4f} "
                f"accuracy={metrics['accuracy']:.4f}",
                flush=True,
            )
    print(f"[evaluate-fight] Reports saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
