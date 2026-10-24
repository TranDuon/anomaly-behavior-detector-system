"""Train the fight ST-GCN (skeleton stream B) from cached pose files.

Windows of `settings.fight_window` sampled frames are labelled from each
dataset's frame labels: fight when at least half of the window is fight,
normal when none of it is, skipped when ambiguous. Normal windows closer than
--boundary-guard samples to a fight are skipped too: the frame annotations of
when a fight starts and ends are approximate. Each window is a bag of person
pairs (multiple-instance learning).

Sampling: half of every batch is fight windows; the other half is split
evenly between normal windows of videos that contain a fight (same scene and
people, most of the normal frames of the test set) and normal windows of
videos without one (50x more footage, which would otherwise drown the first
group). After --hard-negative-warmup epochs, a share --hard-negative-fraction
of each normal group is drawn in proportion to the window's last predicted
fight probability (hard-negative mining).

Validation scores every window of the held-out training videos at
--val-stride and spreads the scores over frames exactly as
scripts/evaluate_fight.py does, so the checkpoint and the threshold are
chosen on the same frame-level measure the test reports, on whole videos.

Test items (UBI-Fights test_videos.csv, Surveillance Camera Fight, UCF-Crime
test) never enter training or validation. Validation is a video-level split
of the training items. A checkpoint is written after every epoch, so
`--resume` continues after a Colab disconnect.
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
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomaly.fight_data import VideoItem, load_items
from anomaly.skeleton import PoseTrack, window_bag, window_starts
from anomaly.stgcn import PairBagClassifier, collate_bags
from core.config import settings
from scripts.evaluation_metrics import aggregate_frame_scores


WINDOW = {
    "length": settings.fight_window,
    "min_presence": settings.fight_min_presence,
    "max_pairs": settings.fight_max_pairs,
    "pair_radius": settings.fight_pair_radius,
}


def window_labels(sample_labels: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """1 = fight (>= half), 0 = normal (none), -1 = ambiguous."""
    labels = np.empty(len(starts), dtype=np.int64)
    for index, start in enumerate(starts):
        ratio = float(sample_labels[start : start + length].mean())
        labels[index] = 1 if ratio >= 0.5 else (0 if ratio == 0.0 else -1)
    return labels


def near_fight(sample_labels: np.ndarray, guard: int) -> np.ndarray:
    """True for samples within `guard` samples of a fight sample."""
    fight = np.asarray(sample_labels) > 0
    if guard <= 0 or not fight.any():
        return fight
    return np.convolve(fight.astype(np.int32), np.ones(2 * guard + 1, dtype=np.int32), mode="same") > 0


class WindowDataset(Dataset):
    def __init__(self, poses: list[PoseTrack], index: np.ndarray, augment: bool):
        self.poses = poses
        self.index = index            # (M, 4): pose id, start, label, source
        self.augment = augment
        self.rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, position: int):
        pose_id, start, label = (int(value) for value in self.index[position, :3])
        bag = window_bag(
            self.poses[pose_id],
            start,
            WINDOW["length"],
            WINDOW["min_presence"],
            WINDOW["max_pairs"],
            WINDOW["pair_radius"],
            rng=self.rng if self.augment else None,
        )
        return bag, label, position


def collate(batch):
    """Windows with people (scored) and their index positions; empty ones score 0."""
    full = [(bag, label, position) for bag, label, position in batch if len(bag)]
    if not full:
        return None, None, None, []
    bags, mask = collate_bags([bag for bag, _, _ in full])
    labels = torch.tensor([label for _, label, _ in full], dtype=torch.float32)
    return bags, mask, labels, [position for _, _, position in full]


def seed_worker(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % (2**32))


def build_index(
    items: list[VideoItem],
    poses: list[PoseTrack],
    positive_stride: int,
    normal_stride: int,
    boundary_guard: int = 0,
) -> np.ndarray:
    """(M, 4) training windows: pose id, start, label, source.

    source is 1 for windows of videos that contain a fight, 0 otherwise.
    """
    rows = []
    for pose_id, (item, pose) in enumerate(zip(items, poses)):
        frame_labels = item.labels_for(pose.frame_count)
        sample_labels = frame_labels[np.minimum(pose.sample_frames, pose.frame_count - 1)]
        stride = positive_stride if item.has_positive else normal_stride
        starts = window_starts(pose.num_samples, WINDOW["length"], stride)
        labels = window_labels(sample_labels, starts, WINDOW["length"])
        if boundary_guard > 0 and item.has_positive:
            near = near_fight(sample_labels, boundary_guard)
            close = np.array([near[s : s + WINDOW["length"]].any() for s in starts], dtype=bool)
            labels[(labels == 0) & close] = -1
        contiguous = np.array([pose.contiguous(int(s), WINDOW["length"]) for s in starts], dtype=bool)
        keep = (labels >= 0) & contiguous
        source = int(item.has_positive)
        rows.extend((pose_id, int(s), int(l), source) for s, l in zip(starts[keep], labels[keep]))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 4)


def build_eval_index(items: list[VideoItem], poses: list[PoseTrack], stride: int) -> np.ndarray:
    """(M, 4) every window of whole videos at `stride`, ambiguous ones labelled -1."""
    rows = []
    for pose_id, (item, pose) in enumerate(zip(items, poses)):
        if pose.num_samples == 0 or pose.frame_count == 0:
            continue
        frame_labels = item.labels_for(pose.frame_count)
        sample_labels = frame_labels[np.minimum(pose.sample_frames, pose.frame_count - 1)]
        starts = window_starts(pose.num_samples, WINDOW["length"], stride)
        starts = starts[[pose.contiguous(int(s), WINDOW["length"]) for s in starts]]
        labels = window_labels(sample_labels, starts, WINDOW["length"])
        source = int(item.has_positive)
        rows.extend((pose_id, int(s), int(l), source) for s, l in zip(starts, labels))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 4)


def sampling_weights(
    index: np.ndarray,
    scores: np.ndarray | None = None,
    hard_fraction: float = 0.0,
) -> np.ndarray:
    """Sampling probability of each training window (sums to 1).

    Fight windows share 0.5 uniformly. Normal windows share the other 0.5,
    split evenly between those from videos with a fight (source 1) and those
    from videos without (source 0). Inside each normal group a share
    `hard_fraction` follows `scores` (the last predicted fight probability;
    NaN = never scored yet, counted at the group's mean).
    """
    labels, sources = index[:, 2], index[:, 3]
    weights = np.zeros(len(index), dtype=np.float64)
    normal_groups = [group for group in ((labels == 0) & (sources == 1), (labels == 0) & (sources == 0)) if group.any()]
    groups = [(labels == 1, 0.5 if normal_groups else 1.0, False)]
    groups += [(group, 0.5 / len(normal_groups), True) for group in normal_groups]
    for group, mass, mine in groups:
        count = int(group.sum())
        if count == 0:
            continue
        share = np.full(count, 1.0 / count)
        if mine and scores is not None and hard_fraction > 0:
            group_scores = scores[group]
            seen = ~np.isnan(group_scores)
            if seen.any():
                group_scores = np.where(seen, group_scores, group_scores[seen].mean())
                total = group_scores.sum()
                if total > 0:
                    share = (1 - hard_fraction) * share + hard_fraction * group_scores / total
        weights[group] = mass * share
    return weights / weights.sum()


def split_videos(items: list[VideoItem], val_ratio: float, seed: int):
    """Video-level split, stratified by whether a video contains a fight."""
    rng = random.Random(seed)
    train, val = [], []
    for has_fight in (True, False):
        group = [item for item in items if item.has_positive == has_fight]
        rng.shuffle(group)
        count = int(round(len(group) * val_ratio))
        if len(group) > 1:
            count = min(max(count, 1), len(group) - 1)
        val.extend(group[:count])
        train.extend(group[count:])
    return train, val


def score_windows(model, loader, size: int, device) -> tuple[np.ndarray, np.ndarray]:
    """Fight probability per index row (0 for windows without people) and which rows had people."""
    model.eval()
    probabilities = np.zeros(size, dtype=np.float64)
    scored = np.zeros(size, dtype=bool)
    with torch.no_grad():
        for bags, mask, _, positions in loader:
            if bags is None:
                continue
            logits = model(bags.to(device), mask.to(device))[:, 0].float()
            probabilities[positions] = torch.sigmoid(logits).cpu().numpy()
            scored[positions] = True
    return probabilities, scored


def frame_level_validation(
    items: list[VideoItem],
    poses: list[PoseTrack],
    index: np.ndarray,
    probabilities: np.ndarray,
    scored: np.ndarray,
) -> dict:
    """Frame-level AUC/AP over whole validation videos, as evaluate_fight.py computes them."""
    all_labels, all_scores, video_aucs = [], [], []
    for pose_id, (item, pose) in enumerate(zip(items, poses)):
        rows = index[:, 0] == pose_id
        if not rows.any():
            continue
        starts = index[rows, 1]
        end_samples = np.minimum(starts + WINDOW["length"] - 1, pose.num_samples - 1)
        end_frames = pose.sample_frames[end_samples] + 1
        frame_scores, _ = aggregate_frame_scores(probabilities[rows], end_frames, pose.frame_count)
        labels = item.labels_for(pose.frame_count)
        all_labels.append(labels)
        all_scores.append(frame_scores)
        if 0 < labels.sum() < len(labels):
            video_aucs.append(float(roc_auc_score(labels, frame_scores)))
    labels = np.concatenate(all_labels).astype(np.int32)
    scores = np.concatenate(all_scores)
    two_classes = len(np.unique(labels)) == 2

    window_label = index[:, 2]
    usable = scored & (window_label >= 0)
    clipped = np.clip(probabilities[usable], 1e-7, 1 - 1e-7)
    targets = window_label[usable]
    loss = float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped))) if usable.any() else float("nan")
    return {
        "loss": loss,
        "auc": float(roc_auc_score(labels, scores)) if two_classes else float("nan"),
        "ap": float(average_precision_score(labels, scores)) if two_classes else float("nan"),
        "video_auc_median": float(np.median(video_aucs)) if video_aucs else float("nan"),
        "labels": labels,
        "scores": scores,
    }


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Threshold t maximising balanced accuracy (Youden J) of the rule score > t.

    Max-F1 favours recall when positives are frequent and produced a threshold
    that flagged 75% of normal frames; balanced accuracy weighs both classes.
    Returns (threshold, balanced accuracy).
    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    youden = true_positive_rate - false_positive_rate
    best = int(np.argmax(youden[1:])) + 1          # index 0 is the +inf threshold
    # roc_curve predicts positive for score >= t; nudge below t so that
    # score > threshold selects the same frames.
    balanced = (true_positive_rate[best] + 1 - false_positive_rate[best]) / 2
    return float(np.nextafter(thresholds[best], -np.inf)), float(balanced)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fight ST-GCN on pose caches")
    parser.add_argument("--dataset", action="append", required=True, help="name:root[:split]")
    parser.add_argument("--pose-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude", default=None, help="exclude.json from dedup_videos.py")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--samples-per-epoch", type=int, default=32768,
        help="Windows drawn per epoch; half are fight windows, ~1 pass over them per epoch",
    )
    parser.add_argument("--positive-stride", type=int, default=4, help="Window stride in videos with a fight")
    parser.add_argument("--normal-stride", type=int, default=10, help="Window stride in videos without a fight")
    parser.add_argument(
        "--boundary-guard", type=int, default=10,
        help="Skip normal windows within this many samples of a fight (approximate annotations)",
    )
    parser.add_argument(
        "--pool-r", type=float, default=5.0,
        help="Smooth-max sharpness for pooling pair logits; 0 = plain max (first two runs)",
    )
    parser.add_argument("--hard-negative-fraction", type=float, default=0.5)
    parser.add_argument("--hard-negative-warmup", type=int, default=2, help="Epochs of uniform sampling first")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-stride", type=int, default=5, help="Window stride for frame-level validation")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[train-fight] CUDA unavailable; falling back to CPU.", flush=True)
        device_name = "cpu"
    device = torch.device(device_name)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_root = Path(args.pose_root).resolve()
    excluded = set()
    if args.exclude:
        excluded = set(json.loads(Path(args.exclude).read_text(encoding="utf-8"))["exclude_from_training"])

    items = [item for spec in args.dataset for item in load_items(spec)]
    usable = []
    for item in items:
        if item.split != "train" or item.uid in excluded:
            continue
        if (pose_root / item.dataset / f"{item.key}.npz").exists():
            usable.append(item)
    if len(usable) < 2:
        raise RuntimeError("Need at least two training videos with pose caches")
    train_items, val_items = split_videos(usable, args.val_ratio, args.seed)
    print(
        f"[train-fight] videos: {len(train_items)} train, {len(val_items)} val "
        f"({len(excluded)} excluded as duplicates)",
        flush=True,
    )

    def load(group):
        return [PoseTrack.load(pose_root / item.dataset / f"{item.key}.npz") for item in group]

    train_poses, val_poses = load(train_items), load(val_items)
    train_index = build_index(
        train_items, train_poses, args.positive_stride, args.normal_stride, args.boundary_guard
    )
    val_index = build_eval_index(val_items, val_poses, args.val_stride)
    labels = train_index[:, 2]
    positives = int(labels.sum())
    same_scene = int(((labels == 0) & (train_index[:, 3] == 1)).sum())
    print(
        f"[train-fight] train windows: {len(train_index)} ({positives} fight, "
        f"{same_scene} normal in fight videos, {len(train_index) - positives - same_scene} in normal videos); "
        f"validation: {len(val_index)} windows over {len(val_items)} whole videos",
        flush=True,
    )
    if positives == 0 or positives == len(train_index):
        raise RuntimeError("Training windows need both fight and normal examples")

    sampler = WeightedRandomSampler(
        torch.from_numpy(sampling_weights(train_index)), num_samples=args.samples_per_epoch, replacement=True
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(WindowDataset(train_poses, train_index, True), sampler=sampler, **loader_args)
    val_loader = DataLoader(WindowDataset(val_poses, val_index, False), shuffle=False, **loader_args)

    model = PairBagClassifier(pool_r=args.pool_r or None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.BCEWithLogitsLoss()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    # Last fight probability seen in training for each window (hard-negative mining).
    seen_scores = np.full(len(train_index), np.nan, dtype=np.float32)

    checkpoint_path = output_dir / "checkpoint_last.pt"
    start_epoch, best_auc, best_epoch, history = 1, -1.0, 0, []
    finished = False
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_auc, best_epoch = checkpoint["best_auc"], checkpoint["best_epoch"]
        history = checkpoint["history"]
        if checkpoint.get("seen_scores") is not None and len(checkpoint["seen_scores"]) == len(train_index):
            seen_scores = checkpoint["seen_scores"]
        # A finished run (all epochs or early stop) is not trained further by a
        # rerun, so --resume is safe to repeat; delete the checkpoint to retrain.
        finished = bool(checkpoint.get("finished", False))
        state = "already finished; delete checkpoint_last.pt to retrain" if finished else "resuming"
        print(f"[train-fight] Checkpoint after epoch {checkpoint['epoch']}: {state}", flush=True)
    if finished:
        start_epoch = args.epochs + 1

    def save_checkpoint(epoch: int, done: bool) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_auc": best_auc,
                "best_epoch": best_epoch,
                "history": history,
                "seen_scores": seen_scores,
                "finished": done,
            },
            checkpoint_path,
        )

    print(
        f"[train-fight] device={device} parameters={parameter_count:,} pool_r={model.pool_r} "
        f"epochs={args.epochs} batch={args.batch_size} samples/epoch={args.samples_per_epoch}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        mining = args.hard_negative_fraction if epoch > args.hard_negative_warmup else 0.0
        sampler.weights = torch.from_numpy(sampling_weights(train_index, seen_scores, mining))
        model.train()
        losses = []
        for bags, mask, batch_labels, positions in train_loader:
            if bags is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(bags.to(device), mask.to(device))[:, 0]
                loss = criterion(logits.float(), batch_labels.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            seen_scores[positions] = torch.sigmoid(logits.detach().float()).cpu().numpy()
        scheduler.step()

        probabilities, scored = score_windows(model, val_loader, len(val_index), device)
        validation = frame_level_validation(val_items, val_poses, val_index, probabilities, scored)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_loss": validation["loss"],
            "val_auc": validation["auc"],
            "val_ap": validation["ap"],
            "val_video_auc_median": validation["video_auc_median"],
            "hard_negative_fraction": mining,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": round(time.perf_counter() - started, 2),
        }
        history.append(row)
        improved = validation["auc"] > best_auc
        if improved:
            best_auc, best_epoch = validation["auc"], epoch
            model.save(output_dir / "fight_stgcn.pt", {"window": WINDOW, "epoch": epoch, "val_auc": best_auc})
        stop = epoch == args.epochs or epoch - best_epoch >= args.patience
        save_checkpoint(epoch, done=stop)
        print(
            f"  Epoch {epoch:3d}/{args.epochs} train={row['train_loss']:.4f} "
            f"val={row['val_loss']:.4f} frameAUC={row['val_auc']:.4f} AP={row['val_ap']:.4f} "
            f"in-video AUC={row['val_video_auc_median']:.4f} mining={mining:.2f} "
            f"time={row['seconds']:.1f}s{' *' if improved else ''}",
            flush=True,
        )
        if epoch - best_epoch >= args.patience and epoch < args.epochs:
            print(f"[train-fight] Early stopping at epoch {epoch}.", flush=True)
        if stop:
            break

    best = PairBagClassifier.load(output_dir / "fight_stgcn.pt", device=str(device))
    probabilities, scored = score_windows(best, val_loader, len(val_index), device)
    validation = frame_level_validation(val_items, val_poses, val_index, probabilities, scored)
    threshold, val_balanced = best_threshold(validation["labels"], validation["scores"])
    val_f1 = float(f1_score(validation["labels"], validation["scores"] > threshold, zero_division=0))
    write_json(
        output_dir / "fight_threshold.json",
        {
            "threshold": threshold,
            "method": "max balanced accuracy (Youden J) on validation frames of whole videos",
            "val_balanced_accuracy": val_balanced,
            "val_f1": val_f1,
            "val_auc": validation["auc"],
            "val_ap": validation["ap"],
            "val_video_auc_median": validation["video_auc_median"],
        },
    )
    write_json(output_dir / "training_history.json", {"history": history})
    write_json(
        output_dir / "split.json",
        {
            "seed": args.seed,
            "train": [item.uid for item in train_items],
            "validation": [item.uid for item in val_items],
            "excluded_duplicates": sorted(excluded),
        },
    )
    write_json(
        output_dir / "config.json",
        {
            "datasets": args.dataset,
            "window": WINDOW,
            "pose_model": settings.pose_model,
            "pose_fps": settings.pose_fps,
            "model": best.config,
            "parameter_count": parameter_count,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_val_auc": best_auc,
            "selection_metric": "frame-level AUC on whole validation videos",
            "batch_size": args.batch_size,
            "samples_per_epoch": args.samples_per_epoch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "positive_stride": args.positive_stride,
            "normal_stride": args.normal_stride,
            "boundary_guard": args.boundary_guard,
            "hard_negative_fraction": args.hard_negative_fraction,
            "hard_negative_warmup": args.hard_negative_warmup,
            "val_stride": args.val_stride,
            "device": str(device),
            "train_windows": int(len(train_index)),
            "train_fight_windows": positives,
            "train_normal_windows_in_fight_videos": same_scene,
            "val_windows": int(len(val_index)),
        },
    )
    print(
        f"[train-fight] best epoch {best_epoch}: val frame AUC={best_auc:.4f}; "
        f"threshold={threshold:.4f} (val balanced accuracy={val_balanced:.4f}, F1={val_f1:.4f}) -> {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
