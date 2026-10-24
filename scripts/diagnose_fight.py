"""Diagnose why the fight ST-GCN learns little, using the cached pose files.

Three checks, on the same training windows train_fight.py builds:
  1. coverage — share of fight / normal windows that yield at least one
     person pair, pairs and tracks per window, keypoint confidence; a fight
     window without pairs is a label the model can never learn from;
  2. hand-made baseline — AUC of the pair-ranking interaction score used as a
     fight score; if it is as good as the model, the model adds nothing; if
     both are near 0.5 the pose input itself carries little signal;
  3. overfit test — train a fresh model without augmentation or dropout on a
     small balanced set; failing to memorise it (train AUC well below 1)
     points to a data or code bug rather than a weak method.

Writes diagnostics.json and prints a verdict per check.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomaly.fight_data import load_items
from anomaly.skeleton import (
    MIN_KEYPOINT_CONF,
    PoseTrack,
    _track_stats,
    proximity_matrix,
    select_pairs,
    window_bag,
)
from anomaly.stgcn import PairBagClassifier, collate_bags
from scripts.train_fight import WINDOW, build_index


def window_summary(pose: PoseTrack, start: int) -> dict:
    tracks = pose.window_tracks(start, WINDOW["length"], WINDOW["min_presence"])
    pairs = select_pairs(tracks, WINDOW["max_pairs"], WINDOW["pair_radius"])
    confidence = [
        float(kp[:, :, 2][h > 0].mean()) for kp, h in tracks if (h > 0).any()
    ]
    interaction = 0.0
    stats = [_track_stats(kp, h) for kp, h in tracks]
    valid = [index for index, item in enumerate(stats) if item is not None]
    if len(valid) >= 2:
        valid_stats = [stats[index] for index in valid]
        proximity = proximity_matrix(np.stack([tracks[index][0] for index in valid]), valid_stats)
        energy = np.asarray([item.limb_energy for item in valid_stats])
        score = (energy[:, None] + energy[None, :] + 1e-3) / (0.25 + proximity)
        close = np.triu(proximity <= WINDOW["pair_radius"], k=1)
        if close.any():
            interaction = float(score[close].max())
    return {
        "tracks": len(tracks),
        "pairs": len(pairs),
        "two_person_pairs": sum(1 for _, j in pairs if j is not None),
        "confidence": float(np.mean(confidence)) if confidence else 0.0,
        "interaction": interaction,
    }


def coverage(poses, index, sample: int, rng) -> tuple[dict, np.ndarray, np.ndarray]:
    rows = index[rng.choice(len(index), size=min(sample, len(index)), replace=False)]
    summaries = [window_summary(poses[int(row[0])], int(row[1])) for row in rows]
    labels = rows[:, 2].astype(np.int32)
    report = {}
    for label, name in ((1, "fight"), (0, "normal")):
        chosen = [item for item, value in zip(summaries, labels) if value == label]
        if not chosen:
            continue
        report[name] = {
            "windows_checked": len(chosen),
            "share_with_any_pair": float(np.mean([item["pairs"] > 0 for item in chosen])),
            "share_with_two_person_pair": float(np.mean([item["two_person_pairs"] > 0 for item in chosen])),
            "mean_pairs": float(np.mean([item["pairs"] for item in chosen])),
            "mean_tracks": float(np.mean([item["tracks"] for item in chosen])),
            "mean_keypoint_confidence": float(np.mean([item["confidence"] for item in chosen])),
        }
    scores = np.asarray([item["interaction"] for item in summaries])
    return report, labels, scores


def overfit_test(poses, index, per_class: int, epochs: int, device, rng) -> dict:
    bags, labels = [], []
    for label in (1, 0):
        candidates = index[index[:, 2] == label]
        order = rng.permutation(len(candidates))
        for row in candidates[order]:
            bag = window_bag(poses[int(row[0])], int(row[1]), WINDOW["length"],
                             WINDOW["min_presence"], WINDOW["max_pairs"], WINDOW["pair_radius"])
            if len(bag):
                bags.append(bag)
                labels.append(label)
            if labels.count(label) >= per_class:
                break
    labels_array = np.asarray(labels, dtype=np.float32)
    model = PairBagClassifier(dropout=0.0).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    order = np.arange(len(bags))
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(order)
        for start in range(0, len(order), 32):
            chosen = order[start:start + 32]
            batch, mask = collate_bags([bags[i] for i in chosen])
            target = torch.from_numpy(labels_array[chosen]).to(device)
            optimiser.zero_grad()
            loss = criterion(model(batch.to(device), mask.to(device))[:, 0], target)
            loss.backward()
            optimiser.step()
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            scores, losses = [], []
            with torch.no_grad():
                for start in range(0, len(bags), 64):
                    batch, mask = collate_bags(bags[start:start + 64])
                    logits = model(batch.to(device), mask.to(device))[:, 0]
                    target = torch.from_numpy(labels_array[start:start + 64]).to(device)
                    losses.append(float(criterion(logits, target).item()) * len(target))
                    scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            history.append({
                "epoch": epoch,
                "train_loss": sum(losses) / len(bags),
                "train_auc": float(roc_auc_score(labels_array, scores)),
            })
            print(f"  overfit epoch {epoch:3d}: loss={history[-1]['train_loss']:.4f} "
                  f"AUC={history[-1]['train_auc']:.4f}", flush=True)
    return {"windows": len(bags), "fight_windows": int(labels_array.sum()), "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the fight ST-GCN training data")
    parser.add_argument("--dataset", action="append", required=True, help="name:root[:split]")
    parser.add_argument("--pose-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--sample-windows", type=int, default=4000)
    parser.add_argument("--overfit-per-class", type=int, default=128)
    parser.add_argument("--overfit-epochs", type=int, default=60)
    parser.add_argument("--positive-stride", type=int, default=4)
    parser.add_argument("--normal-stride", type=int, default=10)
    parser.add_argument("--boundary-guard", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_root = Path(args.pose_root).resolve()
    excluded = set()
    if args.exclude:
        excluded = set(json.loads(Path(args.exclude).read_text(encoding="utf-8"))["exclude_from_training"])

    items = [
        item for spec in args.dataset for item in load_items(spec)
        if item.split == "train" and item.uid not in excluded
        and (pose_root / item.dataset / f"{item.key}.npz").exists()
    ]
    poses = [PoseTrack.load(pose_root / item.dataset / f"{item.key}.npz") for item in items]
    index = build_index(items, poses, args.positive_stride, args.normal_stride, args.boundary_guard)
    print(f"[diagnose] {len(items)} training videos, {len(index)} windows "
          f"({int(index[:, 2].sum())} fight)", flush=True)

    print("[diagnose] 1. coverage", flush=True)
    cover, labels, scores = coverage(poses, index, args.sample_windows, rng)
    for name, values in cover.items():
        print(f"  {name}: " + ", ".join(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                                        for key, value in values.items()), flush=True)

    baseline_auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
    print(f"[diagnose] 2. hand-made interaction score AUC = {baseline_auc:.4f}", flush=True)

    print("[diagnose] 3. overfit test", flush=True)
    overfit = overfit_test(poses, index, args.overfit_per_class, args.overfit_epochs, device, rng)
    final_auc = overfit["history"][-1]["train_auc"]

    verdicts = []
    fight_cover = cover.get("fight", {}).get("share_with_any_pair", 1.0)
    if fight_cover < 0.8:
        verdicts.append(f"{1 - fight_cover:.0%} of fight windows have no person pair: they carry no pose signal.")
    if final_auc < 0.9:
        verdicts.append(f"Overfit test reached only AUC {final_auc:.3f}: suspect a data/label or code problem.")
    else:
        verdicts.append(f"Overfit test reached AUC {final_auc:.3f}: the pipeline can learn; the weak result is generalisation/signal.")
    if baseline_auc is not None and baseline_auc < 0.6:
        verdicts.append(f"Hand-made interaction score AUC {baseline_auc:.3f}: pose input separates fights poorly.")

    report = {
        "window": WINDOW,
        "training_videos": len(items),
        "training_windows": int(len(index)),
        "coverage": cover,
        "interaction_baseline_auc": baseline_auc,
        "overfit_test": overfit,
        "min_keypoint_conf": MIN_KEYPOINT_CONF,
        "verdicts": verdicts,
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[diagnose] verdicts:", flush=True)
    for line in verdicts:
        print("  -", line, flush=True)


if __name__ == "__main__":
    main()
