"""Train the fight ST-GCN (skeleton stream B) from cached pose files.

Windows of `settings.fight_window` sampled frames are labelled from each
dataset's frame labels: fight when at least half of the window is fight,
normal when none of it is, skipped when ambiguous. Each window is a bag of
person pairs (multiple-instance learning).

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


class WindowDataset(Dataset):
    def __init__(self, poses: list[PoseTrack], index: np.ndarray, augment: bool):
        self.poses = poses
        self.index = index            # (M, 3): pose id, start, label
        self.augment = augment
        self.rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, position: int):
        pose_id, start, label = (int(value) for value in self.index[position])
        bag = window_bag(
            self.poses[pose_id],
            start,
            WINDOW["length"],
            WINDOW["min_presence"],
            WINDOW["max_pairs"],
            WINDOW["pair_radius"],
            rng=self.rng if self.augment else None,
        )
        return bag, label


def collate(batch):
    """Split windows with people (scored) from empty ones (score 0)."""
    full = [(bag, label) for bag, label in batch if len(bag)]
    empty_labels = [label for bag, label in batch if not len(bag)]
    if full:
        bags, mask = collate_bags([bag for bag, _ in full])
        labels = torch.tensor([label for _, label in full], dtype=torch.float32)
    else:
        bags = mask = labels = None
    return bags, mask, labels, empty_labels


def seed_worker(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % (2**32))


def build_index(
    items: list[VideoItem],
    poses: list[PoseTrack],
    positive_stride: int,
    normal_stride: int,
) -> np.ndarray:
    rows = []
    for pose_id, (item, pose) in enumerate(zip(items, poses)):
        frame_labels = item.labels_for(pose.frame_count)
        sample_labels = frame_labels[np.minimum(pose.sample_frames, pose.frame_count - 1)]
        stride = positive_stride if item.has_positive else normal_stride
        starts = window_starts(pose.num_samples, WINDOW["length"], stride)
        labels = window_labels(sample_labels, starts, WINDOW["length"])
        contiguous = np.array([pose.contiguous(int(s), WINDOW["length"]) for s in starts], dtype=bool)
        keep = (labels >= 0) & contiguous
        rows.extend((pose_id, int(s), int(l)) for s, l in zip(starts[keep], labels[keep]))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


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


def evaluate(model, loader, device, criterion) -> dict:
    model.eval()
    scores, labels, losses = [], [], []
    with torch.no_grad():
        for bags, mask, batch_labels, empty_labels in loader:
            if bags is not None:
                logits = model(bags.to(device), mask.to(device))[:, 0]
                losses.append(float(criterion(logits, batch_labels.to(device)).item()))
                scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                labels.extend(batch_labels.numpy().tolist())
            scores.extend([0.0] * len(empty_labels))
            labels.extend(empty_labels)
    labels_array = np.asarray(labels, dtype=np.int32)
    scores_array = np.asarray(scores, dtype=np.float64)
    two_classes = len(np.unique(labels_array)) == 2
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "auc": float(roc_auc_score(labels_array, scores_array)) if two_classes else float("nan"),
        "ap": float(average_precision_score(labels_array, scores_array)) if two_classes else float("nan"),
        "scores": scores_array,
        "labels": labels_array,
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
    # score > threshold selects the same windows.
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
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--samples-per-epoch", type=int, default=16384)
    parser.add_argument("--positive-stride", type=int, default=4)
    parser.add_argument("--normal-stride", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=12)
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
    train_index = build_index(train_items, train_poses, args.positive_stride, args.normal_stride)
    val_index = build_index(val_items, val_poses, args.positive_stride, args.normal_stride)
    positives = int(train_index[:, 2].sum())
    print(
        f"[train-fight] windows: train {len(train_index)} ({positives} fight), "
        f"val {len(val_index)} ({int(val_index[:, 2].sum())} fight)",
        flush=True,
    )
    if positives == 0 or positives == len(train_index):
        raise RuntimeError("Training windows need both fight and normal examples")

    labels = train_index[:, 2]
    weights = np.where(labels == 1, 0.5 / positives, 0.5 / (len(labels) - positives))
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=args.samples_per_epoch, replacement=True
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

    model = PairBagClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.BCEWithLogitsLoss()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

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
                "finished": done,
            },
            checkpoint_path,
        )

    print(
        f"[train-fight] device={device} parameters={parameter_count:,} "
        f"epochs={args.epochs} batch={args.batch_size}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        for bags, mask, batch_labels, _ in train_loader:
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
        scheduler.step()

        validation = evaluate(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_loss": validation["loss"],
            "val_auc": validation["auc"],
            "val_ap": validation["ap"],
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
            f"val={row['val_loss']:.4f} AUC={row['val_auc']:.4f} AP={row['val_ap']:.4f} "
            f"time={row['seconds']:.1f}s{' *' if improved else ''}",
            flush=True,
        )
        if epoch - best_epoch >= args.patience and epoch < args.epochs:
            print(f"[train-fight] Early stopping at epoch {epoch}.", flush=True)
        if stop:
            break

    best = PairBagClassifier.load(output_dir / "fight_stgcn.pt", device=str(device))
    validation = evaluate(best, val_loader, device, criterion)
    threshold, val_balanced = best_threshold(validation["labels"], validation["scores"])
    val_f1 = float(f1_score(validation["labels"], validation["scores"] > threshold, zero_division=0))
    write_json(
        output_dir / "fight_threshold.json",
        {
            "threshold": threshold,
            "method": "max balanced accuracy (Youden J) on validation windows",
            "val_balanced_accuracy": val_balanced,
            "val_f1": val_f1,
            "val_auc": validation["auc"],
            "val_ap": validation["ap"],
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
            "batch_size": args.batch_size,
            "samples_per_epoch": args.samples_per_epoch,
            "lr": args.lr,
            "device": str(device),
            "train_windows": int(len(train_index)),
            "train_fight_windows": positives,
            "val_windows": int(len(val_index)),
        },
    )
    print(
        f"[train-fight] best epoch {best_epoch}: val AUC={best_auc:.4f}; "
        f"threshold={threshold:.4f} (val balanced accuracy={val_balanced:.4f}, F1={val_f1:.4f}) -> {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
