"""Evaluate a trained LSTM Autoencoder on UCSD Ped2 at frame level."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomaly.model import LSTMAutoencoder
from scripts.evaluation_metrics import aggregate_frame_scores, binary_metrics


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def natural_key(path: Path) -> list:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def image_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_key,
    )


def load_frame_labels(test_root: Path, clip_name: str) -> np.ndarray:
    clip_dir = test_root / clip_name
    ground_truth_dir = test_root / f"{clip_name}_gt"
    if not clip_dir.is_dir():
        raise FileNotFoundError(f"Missing test clip: {clip_dir}")
    if not ground_truth_dir.is_dir():
        raise FileNotFoundError(f"Missing ground-truth masks: {ground_truth_dir}")

    frames = image_files(clip_dir)
    masks = image_files(ground_truth_dir)
    if not frames:
        raise RuntimeError(f"No test frames found in {clip_dir}")
    if len(frames) != len(masks):
        raise RuntimeError(
            f"Frame/mask count mismatch for {clip_name}: "
            f"{len(frames)} frames vs {len(masks)} masks"
        )

    labels = []
    for mask_path in masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read ground-truth mask: {mask_path}")
        labels.append(int(np.any(mask > 0)))
    return np.asarray(labels, dtype=np.int32)


def score_sequences(
    model: LSTMAutoencoder,
    sequences: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    standardized = ((sequences - mean) / std).astype(np.float32, copy=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(standardized), batch_size):
            batch = torch.from_numpy(standardized[start : start + batch_size]).to(
                device
            )
            reconstruction = model(batch)
            error = LSTMAutoencoder.reconstruction_error(batch, reconstruction)
            scores.extend(error.cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float64)


def save_plots(
    output_dir: Path,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    metrics: dict,
    clip_payloads: list[dict],
) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    plt.figure(figsize=(7, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"AUROC = {metrics['roc_auc']:.4f}",
    )
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("UCSD Ped2 frame-level ROC")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"AP = {metrics['average_precision']:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("UCSD Ped2 frame-level precision-recall")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close()

    matrix = np.asarray(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ]
    )
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["Normal", "Anomaly"])
    plt.yticks([0, 1], ["Normal", "Anomaly"])
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.title("Confusion matrix")
    for row in range(2):
        for column in range(2):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    columns = 3
    rows = int(np.ceil(len(clip_payloads) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 3.5 * rows), squeeze=False)
    for axis, payload in zip(axes.flat, clip_payloads):
        frame_axis = np.arange(1, len(payload["scores"]) + 1)
        axis.plot(frame_axis, payload["scores"], color="tab:blue", linewidth=1)
        axis.axhline(threshold, color="tab:red", linestyle="--", linewidth=1)
        axis.fill_between(
            frame_axis,
            0,
            1,
            where=payload["labels"].astype(bool),
            color="orange",
            alpha=0.2,
            transform=axis.get_xaxis_transform(),
        )
        axis.set_title(payload["clip"])
        axis.set_xlabel("Frame")
        axis.set_ylabel("Reconstruction error")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(clip_payloads) :]:
        axis.axis("off")
    figure.suptitle("Frame scores; orange = ground-truth anomaly", y=1.01)
    figure.tight_layout()
    figure.savefig(output_dir / "score_timelines.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frame-level evaluation of an LSTM checkpoint on UCSD Ped2"
    )
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--threshold", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    features_dir = Path(args.features_dir).resolve()
    test_root = Path(args.test_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[evaluate] CUDA unavailable; falling back to CPU.", flush=True)
        device_name = "cpu"
    device = torch.device(device_name)

    model = LSTMAutoencoder.load(Path(args.model), device=str(device))
    with np.load(args.scaler) as scaler:
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)
    if mean.shape != (model.input_size,) or std.shape != (model.input_size,):
        raise ValueError(
            f"Scaler shape mismatch: mean={mean.shape}, std={std.shape}, "
            f"model input={model.input_size}"
        )
    if np.any(std <= 0) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("Scaler contains invalid mean/std values")
    threshold_payload = json.loads(Path(args.threshold).read_text(encoding="utf-8"))
    threshold = float(threshold_payload["threshold"])

    cache_files = sorted(features_dir.glob("*.npz"), key=natural_key)
    if not cache_files:
        raise RuntimeError(f"No feature caches found in {features_dir}")

    all_labels = []
    all_scores = []
    frame_rows = []
    per_clip = {}
    clip_payloads = []

    print(
        f"[evaluate] Clips={len(cache_files)} device={device} "
        f"threshold={threshold:.6f}",
        flush=True,
    )
    for index, cache_path in enumerate(cache_files, start=1):
        clip_name = cache_path.stem
        labels = load_frame_labels(test_root, clip_name)
        with np.load(cache_path) as cache:
            if "frame_numbers" not in cache:
                raise RuntimeError(
                    f"{cache_path.name} uses the old cache format. "
                    "Delete the test feature directory and extract it again."
                )
            sequences = np.asarray(cache["sequences"], dtype=np.float32)
            frame_numbers = np.asarray(cache["frame_numbers"], dtype=np.int32)
        if len(sequences) != len(frame_numbers):
            raise RuntimeError(f"Sequence metadata mismatch in {cache_path}")
        if not len(sequences):
            raise RuntimeError(f"No sequences extracted for {clip_name}")

        sequence_scores = score_sequences(
            model, sequences, mean, std, device, args.batch_size
        )
        frame_scores, observed = aggregate_frame_scores(
            sequence_scores, frame_numbers, len(labels)
        )
        clip_metrics = binary_metrics(labels, frame_scores, threshold)
        clip_metrics["observed_frames"] = int(observed.sum())
        clip_metrics["coverage_before_interpolation"] = float(observed.mean())
        per_clip[clip_name] = clip_metrics
        clip_payloads.append(
            {"clip": clip_name, "labels": labels, "scores": frame_scores}
        )

        predictions = (frame_scores > threshold).astype(np.int32)
        for frame_index, (label, score, prediction, is_observed) in enumerate(
            zip(labels, frame_scores, predictions, observed), start=1
        ):
            frame_rows.append(
                {
                    "clip": clip_name,
                    "frame": frame_index,
                    "ground_truth": int(label),
                    "score": float(score),
                    "prediction": int(prediction),
                    "direct_score": int(is_observed),
                }
            )

        all_labels.append(labels)
        all_scores.append(frame_scores)
        roc_auc = clip_metrics["roc_auc"]
        roc_auc_text = f"{roc_auc:.4f}" if roc_auc is not None else "n/a"
        print(
            f"[{index}/{len(cache_files)}] {clip_name}: "
            f"frames={len(labels)} sequences={len(sequences)} "
            f"coverage={observed.mean():.1%} F1={clip_metrics['f1']:.4f} "
            f"AUROC={roc_auc_text}",
            flush=True,
        )

    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    overall = binary_metrics(labels, scores, threshold)
    overall["observed_frames"] = int(
        sum(metrics["observed_frames"] for metrics in per_clip.values())
    )
    overall["coverage_before_interpolation"] = float(
        overall["observed_frames"] / overall["frames"]
    )

    report = {
        "dataset": "UCSD Ped2",
        "evaluation_level": "frame",
        "frame_score": "maximum track reconstruction error ending at the frame",
        "missing_frame_policy": "linear interpolation; nearest value at clip edges",
        "ground_truth": "frame is anomalous when its pixel mask contains a nonzero value",
        "threshold": threshold,
        "threshold_source": threshold_payload.get("method", "validation calibration"),
        "device": str(device),
        "overall": overall,
        "per_clip": per_clip,
    }
    write_json(output_dir / "evaluation.json", report)

    with (output_dir / "frame_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)

    with (output_dir / "per_clip_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        metric_names = list(next(iter(per_clip.values())).keys())
        writer = csv.DictWriter(handle, fieldnames=["clip", *metric_names])
        writer.writeheader()
        for clip_name, metrics in per_clip.items():
            writer.writerow({"clip": clip_name, **metrics})

    save_plots(output_dir, labels, scores, threshold, overall, clip_payloads)

    print("\n[evaluate] Overall frame-level metrics", flush=True)
    for name in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "average_precision",
        "coverage_before_interpolation",
    ):
        print(f"  {name}: {overall[name]:.6f}", flush=True)
    print(
        f"  confusion: TN={overall['true_negative']} "
        f"FP={overall['false_positive']} FN={overall['false_negative']} "
        f"TP={overall['true_positive']}",
        flush=True,
    )
    print(f"[evaluate] Reports saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
