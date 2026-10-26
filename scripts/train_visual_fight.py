"""Train the visual fight head (stream A) on cached S3D clip features.

Each training example is a run of --sequence-length consecutive clips of one
video, labelled per clip from the dataset's frame labels: fight when at least
half of the clip is fight, normal when none of it is and no fight frame is
within --boundary-guard seconds, ignored otherwise.

What the skeleton stream taught (its model found fight *videos* — crowded
scenes, sharp poses — but not the fight inside a video, median in-video AUC
0.54):
  - --fight-share of the sequences come from videos with a fight, cut so
    that most contain fight and calm clips of the same scene;
  - the loss is a class-balanced BCE plus a ranking loss inside each
    sequence (every fight clip should outscore every calm clip of the same
    video), which only the action can satisfy;
  - validation scores whole held-out videos at frame level as the test
    does, and reports the median AUC inside each fight video; checkpoints
    are selected on --select-on (default: the mean of pooled and in-video
    AUC);
  - a motion-energy baseline ("much movement = fight") is scored on the same
    frames; the model has to beat it inside videos.

Test items never enter training or validation. A checkpoint is written after
every epoch, so --resume continues after a Colab disconnect.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomaly.fight_data import VideoItem, load_items
from anomaly.visual_fight import EXTRACTION_CONFIG, ClipFeatures, FightTemporalHead, predict_video
from scripts.evaluation_metrics import aggregate_frame_scores
from scripts.train_fight import best_threshold, split_videos, write_json


class VideoData:
    """One video's clip features with training labels."""

    def __init__(self, item: VideoItem, clips: ClipFeatures, guard_seconds: float):
        self.item = item
        self.clips = clips
        self.frame_labels = item.labels_for(clips.frame_count)
        self.labels = clips.clip_labels(self.frame_labels, int(round(guard_seconds * clips.fps)))
        self.fight_clips = np.flatnonzero(self.labels == 1)


def sample_sequence(video: VideoData, length: int, rng: np.random.Generator, fight_bias: float):
    """(features, labels) of `length` consecutive clips; centred on a fight with prob fight_bias."""
    count = len(video.clips)
    if count <= length:
        start = 0
    elif len(video.fight_clips) and rng.random() < fight_bias:
        anchor = int(rng.choice(video.fight_clips))
        start = int(np.clip(anchor - rng.integers(0, length), 0, count - length))
    else:
        start = int(rng.integers(0, count - length + 1))
    end = min(start + length, count)
    return video.clips.features[start:end], video.labels[start:end]


def make_batch(videos, fight_videos, normal_videos, batch_size, length, fight_share, rng):
    """Padded (B, T, D) features and (B, T) labels; padding is labelled -1."""
    rows = []
    for _ in range(batch_size):
        from_fight = fight_videos and (not normal_videos or rng.random() < fight_share)
        pool = fight_videos if from_fight else normal_videos
        rows.append(sample_sequence(videos[pool[int(rng.integers(len(pool)))]], length, rng, 0.8))
    width = max(len(labels) for _, labels in rows)
    features = np.zeros((batch_size, width, rows[0][0].shape[1]), dtype=np.float32)
    labels = np.full((batch_size, width), -1, dtype=np.int64)
    for index, (row_features, row_labels) in enumerate(rows):
        features[index, : len(row_labels)] = row_features
        labels[index, : len(row_labels)] = row_labels
    return torch.from_numpy(features), torch.from_numpy(labels)


def fight_loss(logits: torch.Tensor, labels: torch.Tensor, rank_weight: float) -> tuple[torch.Tensor, dict]:
    """Class-balanced BCE + pairwise ranking of fight over calm clips inside each sequence."""
    positive, negative = labels == 1, labels == 0
    targets = labels.clamp(min=0).float()
    terms = []
    for mask in (positive, negative):
        if mask.any():
            terms.append(F.binary_cross_entropy_with_logits(logits[mask], targets[mask]))
    bce = torch.stack(terms).mean() if terms else logits.sum() * 0.0
    ranks = []
    for row in range(len(labels)):
        if positive[row].any() and negative[row].any():
            gaps = logits[row][positive[row]][:, None] - logits[row][negative[row]][None, :]
            ranks.append(F.softplus(-gaps).mean())
    rank = torch.stack(ranks).mean() if ranks else logits.sum() * 0.0
    return bce + rank_weight * rank, {"bce": float(bce.item()), "rank": float(rank.item()), "ranked_rows": len(ranks)}


def frame_level(videos: list[VideoData], clip_scores: list[np.ndarray]) -> dict:
    """Pooled frame AUC/AP and median in-video AUC, frames scored as evaluate_visual_fight.py does."""
    all_labels, all_scores, video_aucs = [], [], []
    for video, scores in zip(videos, clip_scores):
        if len(scores) == 0:
            continue
        frame_scores, _ = aggregate_frame_scores(scores, video.clips.last_frames + 1, video.clips.frame_count)
        all_labels.append(video.frame_labels)
        all_scores.append(frame_scores)
        if 0 < video.frame_labels.sum() < len(video.frame_labels):
            video_aucs.append(float(roc_auc_score(video.frame_labels, frame_scores)))
    labels = np.concatenate(all_labels).astype(np.int32)
    scores = np.concatenate(all_scores)
    two_classes = len(np.unique(labels)) == 2
    return {
        "auc": float(roc_auc_score(labels, scores)) if two_classes else float("nan"),
        "ap": float(average_precision_score(labels, scores)) if two_classes else float("nan"),
        "video_auc_median": float(np.median(video_aucs)) if video_aucs else float("nan"),
        "labels": labels,
        "scores": scores,
    }


def selection_score(validation: dict, select_on: str) -> float:
    """Checkpoint selection value; NaN (no usable video) counts as 0."""
    pooled = float(np.nan_to_num(validation["auc"], nan=0.0))
    inside = float(np.nan_to_num(validation["video_auc_median"], nan=0.0))
    return {"frame_auc": pooled, "video_auc": inside, "mean": (pooled + inside) / 2}[select_on]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the visual fight head on S3D clip features")
    parser.add_argument("--dataset", action="append", required=True, help="name:root[:split]")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude", default=None, help="exclude.json from dedup_videos.py")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=64, help="Clips (seconds) per training sequence")
    parser.add_argument("--fight-share", type=float, default=0.6, help="Share of sequences from videos with a fight")
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--boundary-guard", type=float, default=1.0, help="Seconds around a fight where calm clips are ignored")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3, help="Causal blocks; history = 1 + 2*(2^layers - 1) clips")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--input-dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--select-on", choices=("mean", "video_auc", "frame_auc"), default="mean")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[train-visual] CUDA unavailable; falling back to CPU.", flush=True)
        device_name = "cpu"
    device = torch.device(device_name)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_root = Path(args.feature_root).resolve()
    excluded = set()
    if args.exclude:
        excluded = set(json.loads(Path(args.exclude).read_text(encoding="utf-8"))["exclude_from_training"])

    items = [item for spec in args.dataset for item in load_items(spec)]
    usable = [
        item for item in items
        if item.split == "train" and item.uid not in excluded
        and (feature_root / item.dataset / f"{item.key}.npz").exists()
    ]
    if len(usable) < 2:
        raise RuntimeError("Need at least two training videos with feature caches")
    train_items, val_items = split_videos(usable, args.val_ratio, args.seed)

    def load(group):
        videos = [VideoData(item, ClipFeatures.load(feature_root / item.dataset / f"{item.key}.npz"), args.boundary_guard)
                  for item in group]
        return [video for video in videos if len(video.clips)]

    train_videos, val_videos = load(train_items), load(val_items)
    fight_videos = [index for index, video in enumerate(train_videos) if video.item.has_positive]
    normal_videos = [index for index, video in enumerate(train_videos) if not video.item.has_positive]
    clip_labels = np.concatenate([video.labels for video in train_videos])
    print(
        f"[train-visual] videos: {len(train_videos)} train ({len(fight_videos)} with a fight), "
        f"{len(val_videos)} val ({len(excluded)} excluded as duplicates); train clips: "
        f"{int((clip_labels == 1).sum())} fight, {int((clip_labels == 0).sum())} normal, "
        f"{int((clip_labels < 0).sum())} ignored",
        flush=True,
    )
    if not fight_videos:
        raise RuntimeError("Training needs videos with a fight")

    # "Much movement = fight": the model must beat this on the same frames.
    motion = frame_level(val_videos, [video.clips.motion.astype(np.float64) for video in val_videos])
    baseline = {"frame_auc": motion["auc"], "video_auc_median": motion["video_auc_median"]}
    print(
        f"[train-visual] motion-energy baseline on validation: frameAUC={baseline['frame_auc']:.4f} "
        f"in-video AUC={baseline['video_auc_median']:.4f}",
        flush=True,
    )

    model = FightTemporalHead(
        hidden=args.hidden, layers=args.layers, dropout=args.dropout, input_dropout=args.input_dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "extraction": EXTRACTION_CONFIG,
        "receptive_field_clips": model.receptive_field,
    }

    checkpoint_path = output_dir / "checkpoint_last.pt"
    start_epoch, best_score, best_epoch, history, finished = 1, -1.0, 0, [], False
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_score, best_epoch = checkpoint["best_score"], checkpoint["best_epoch"]
        history = checkpoint["history"]
        rng = checkpoint.get("rng", rng)
        finished = bool(checkpoint.get("finished", False))
        state = "already finished; delete checkpoint_last.pt to retrain" if finished else "resuming"
        print(f"[train-visual] Checkpoint after epoch {checkpoint['epoch']}: {state}", flush=True)
    if finished:
        start_epoch = args.epochs + 1

    print(
        f"[train-visual] device={device} parameters={parameter_count:,} history={model.receptive_field} s "
        f"epochs={args.epochs} steps={args.steps_per_epoch} batch={args.batch_size}x{args.sequence_length} clips "
        f"select_on={args.select_on}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        losses, parts = [], {"bce": [], "rank": []}
        for _ in range(args.steps_per_epoch):
            features, labels = make_batch(
                train_videos, fight_videos, normal_videos, args.batch_size, args.sequence_length, args.fight_share, rng
            )
            logits = model(features.to(device))
            loss, detail = fight_loss(logits, labels.to(device), args.rank_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            parts["bce"].append(detail["bce"])
            parts["rank"].append(detail["rank"])
        scheduler.step()

        validation = frame_level(val_videos, [predict_video(model, video.clips.features, device) for video in val_videos])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_bce": float(np.mean(parts["bce"])),
            "train_rank": float(np.mean(parts["rank"])),
            "val_auc": validation["auc"],
            "val_ap": validation["ap"],
            "val_video_auc_median": validation["video_auc_median"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": round(time.perf_counter() - started, 2),
        }
        history.append(row)
        score = selection_score(validation, args.select_on)
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            model.save(output_dir / "visual_fight_head.pt", {**metadata, "epoch": epoch, args.select_on: score})
        stop = epoch == args.epochs or epoch - best_epoch >= args.patience
        torch.save(
            {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "best_score": best_score,
                "best_epoch": best_epoch, "history": history, "rng": rng, "finished": stop,
            },
            checkpoint_path,
        )
        print(
            f"  Epoch {epoch:3d}/{args.epochs} loss={row['train_loss']:.4f} (bce={row['train_bce']:.4f} "
            f"rank={row['train_rank']:.4f}) frameAUC={row['val_auc']:.4f} AP={row['val_ap']:.4f} "
            f"in-video AUC={row['val_video_auc_median']:.4f} time={row['seconds']:.1f}s{' *' if improved else ''}",
            flush=True,
        )
        if stop:
            if epoch < args.epochs:
                print(f"[train-visual] Early stopping at epoch {epoch}.", flush=True)
            break

    best = FightTemporalHead.load(output_dir / "visual_fight_head.pt", device=str(device))
    validation = frame_level(val_videos, [predict_video(best, video.clips.features, device) for video in val_videos])
    threshold, val_balanced = best_threshold(validation["labels"], validation["scores"])
    val_f1 = float(f1_score(validation["labels"], validation["scores"] > threshold, zero_division=0))
    write_json(
        output_dir / "visual_fight_threshold.json",
        {
            "threshold": threshold,
            "method": "max balanced accuracy (Youden J) on validation frames of whole videos",
            "val_balanced_accuracy": val_balanced,
            "val_f1": val_f1,
            "val_auc": validation["auc"],
            "val_ap": validation["ap"],
            "val_video_auc_median": validation["video_auc_median"],
            "val_motion_baseline": baseline,
        },
    )
    write_json(output_dir / "training_history.json", {"history": history})
    write_json(
        output_dir / "split.json",
        {
            "seed": args.seed,
            "train": [video.item.uid for video in train_videos],
            "validation": [video.item.uid for video in val_videos],
            "excluded_duplicates": sorted(excluded),
        },
    )
    write_json(
        output_dir / "config.json",
        {
            "datasets": args.dataset,
            "extraction": EXTRACTION_CONFIG,
            "model": best.config,
            "receptive_field_clips": best.receptive_field,
            "parameter_count": parameter_count,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "best_val_auc": validation["auc"],
            "best_val_video_auc_median": validation["video_auc_median"],
            "select_on": args.select_on,
            "val_motion_baseline": baseline,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "fight_share": args.fight_share,
            "rank_weight": args.rank_weight,
            "boundary_guard_seconds": args.boundary_guard,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "device": str(device),
            "train_videos": len(train_videos),
            "train_fight_videos": len(fight_videos),
            "train_fight_clips": int((clip_labels == 1).sum()),
            "train_normal_clips": int((clip_labels == 0).sum()),
            "val_videos": len(val_videos),
        },
    )
    print(
        f"[train-visual] best epoch {best_epoch}: val frameAUC={validation['auc']:.4f}, "
        f"in-video AUC={validation['video_auc_median']:.4f} (motion baseline {baseline['frame_auc']:.4f} / "
        f"{baseline['video_auc_median']:.4f}); threshold={threshold:.4f} "
        f"(val balanced accuracy={val_balanced:.4f}, F1={val_f1:.4f}) -> {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
