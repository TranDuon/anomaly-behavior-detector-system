"""Colab pilot: add a running score without changing the fight/wrestle/fall head.

Frozen MMAction2 AVA features are reused. The running head starts from the
installed interaction head's hidden layer; its final logit is reset and trained
on fixed ground-camera running versus jogging/walking clips.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
import torch

from action_engine import (AUX_NAMES, ActionEngine, InteractionDistanceHead,
                           InteractionHead, SOURCE_SHA, SPEC, digest)
from prepare_fixed_running import DATASET_REVISION


PARENT_SHA = "ddb09180284c4093189c53b7decbfcc5b51fd2d9b7700536d104524bceb5c344"
CLASSES = ("walking", "jogging", "running")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def parent_head(weights):
    weights = Path(weights)
    metadata = json.loads((weights / "interaction_fall_wrestle.json").read_text())
    if (digest(weights / "interaction_fall_wrestle.pt") != PARENT_SHA
            or metadata["checkpoint_sha256"] != PARENT_SHA
            or metadata["source_sha256"] != SOURCE_SHA
            or metadata["spec"] != SPEC
            or metadata.get("head_version") != 1):
        raise ValueError("Wrong fight/wrestle/fall parent checkpoint")
    head = InteractionHead()
    head.load_state_dict(torch.load(weights / "interaction_fall_wrestle.pt",
                                    map_location="cpu", weights_only=True), strict=True)
    return head


def validate_manifest(manifest):
    if (manifest.get("dataset") != "SPHAR UCF-ARG ground only"
            or manifest.get("source_revision") != DATASET_REVISION
            or manifest.get("classes") != list(CLASSES)
            or len(manifest.get("rows", [])) != 143):
        raise ValueError("Unexpected fixed-camera dataset manifest")
    for row in manifest["rows"]:
        actor, action, split = row["actor"], row["action"], row["split"]
        expected = "train" if actor <= 8 else "development" if actor <= 10 else "test"
        if (split != expected or action not in CLASSES
                or row["label"] != int(action == "running")
                or "_ground_" not in row["source_path"]):
            raise ValueError("Invalid dataset row or actor split: " + str(row))
    return manifest["rows"]


def sampled_clip(path):
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not capture.isOpened() or not 10 <= fps <= 120:
        raise ValueError("Cannot decode video: " + str(path))
    frames = []
    while len(frames) <= fps * 30:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[1] > 640:
            frame = cv2.resize(frame, (640, round(frame.shape[0] * 640 / frame.shape[1])))
        frames.append(frame)
    capture.release()
    duration = len(frames) / fps
    # Some genuine running takes last 1.5-2 s. The 2.13 s AVA window uses
    # boundary frames for those clips, exactly as the source-video sampler does.
    if duration < 1.4 or duration > 30:
        raise ValueError(f"Unsupported clip duration {duration:.2f}s: {path}")
    anchor = duration / 2
    indexes = [min(len(frames) - 1, max(0, round((anchor + (i - 4) * 8 / 30) * fps)))
               for i in range(9)]
    return np.stack([frames[index] for index in indexes])


def feature_stage(args, rows, protocol):
    args.cache.mkdir(parents=True, exist_ok=True)
    marker = args.cache / "protocol.json"
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError("Feature protocol changed; select a new run folder")
    write_json(marker, protocol)
    engine = ActionEngine(weights=args.weights, device=args.device,
                          person_threshold=0.8)
    index = []
    for number, row in enumerate(rows, 1):
        key = hashlib.sha256(row["source_path"].encode()).hexdigest()[:24]
        feature_path = args.cache / (key + ".npz")
        if not feature_path.exists():
            bag, auxiliary, info = engine.extract_features(sampled_clip(row["path"]))
            # The first N bag rows are single people; later rows are pairs.
            singletons = bag[:info["people"]].astype(np.float32)
            singleton_motion = auxiliary[:info["people"]].astype(np.float32)
            temporary = args.cache / (key + ".partial.npz")
            np.savez_compressed(temporary, bag=singletons, auxiliary=singleton_motion,
                                people=np.int32(info["people"]))
            temporary.replace(feature_path)
        index.append({key: row[key] for key in
                      ("source_path", "actor", "take", "action", "label", "split")}
                     | {"key": key})
        if number % 16 == 0:
            print(f"Features {number}/{len(rows)}", flush=True)
    write_json(args.cache / "index.json", index)


def load_rows(cache):
    rows = json.loads((cache / "index.json").read_text())
    for row in rows:
        with np.load(cache / (row["key"] + ".npz"), allow_pickle=False) as data:
            row["bag"] = torch.from_numpy(data["bag"].copy())
            row["auxiliary"] = torch.from_numpy(data["auxiliary"].copy())
            row["people"] = int(data["people"])
    return rows


@torch.inference_mode()
def scores(head, rows, device):
    head.eval()
    return [float(head(row["bag"].to(device), row["auxiliary"].to(device))
                  .max().sigmoid().cpu())
            if row["people"] else 0.0 for row in rows]


def metrics(rows, values, threshold):
    labels = [row["label"] for row in rows]
    predicted = np.asarray(values) >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0)
    return {
        "ap": float(average_precision_score(labels, values)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "threshold": float(threshold), "windows": len(rows),
        "by_action": {
            action: {"total": sum(r["action"] == action for r in rows),
                     "predicted_running": int(sum(bool(p) for r, p in zip(rows, predicted)
                                                  if r["action"] == action))}
            for action in CLASSES
        },
    }


def choose_threshold(rows, values):
    candidates = np.unique(np.r_[0.05, np.asarray(values), 0.95])
    return float(max(candidates, key=lambda t: (
        metrics(rows, values, t)["f1"],
        metrics(rows, values, t)["precision"], float(t))))


def train_stage(args, protocol):
    if json.loads((args.cache / "protocol.json").read_text()) != protocol:
        raise ValueError("Train/feature protocol mismatch")
    rows = load_rows(args.cache)
    partitions = {split: [row for row in rows if row["split"] == split]
                  for split in ("train", "development", "test")}
    for split, part in partitions.items():
        expected = ({"walking": 32, "jogging": 31, "running": 32}
                    if split == "train" else {action: 8 for action in CLASSES})
        if {action: sum(row["action"] == action for row in part) for action in CLASSES} != expected:
            raise ValueError("Missing action or actor split: " + split)
    train_rows = [row for row in partitions["train"] if row["people"]]
    coverage = {
        split: {action: {
            "total": sum(row["action"] == action for row in part),
            "with_person": sum(row["action"] == action and row["people"] > 0 for row in part),
        } for action in CLASSES}
        for split, part in partitions.items()
    }
    if (coverage["train"]["running"]["with_person"] < 16
            or any(coverage["train"][action]["with_person"] < 16
                   for action in ("walking", "jogging"))):
        raise ValueError("Person detector found too few train clips; see coverage in feature cache")
    parent = parent_head(args.weights)
    head = InteractionDistanceHead().warm_start(parent).to(args.device)
    train_motion = torch.cat([row["auxiliary"] for row in train_rows])
    head.aux_mean.copy_(train_motion.mean(dim=0).to(args.device))
    head.aux_std.copy_(train_motion.std(dim=0).clamp_min(0.01).to(args.device))
    head.layers[-1].reset_parameters()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_file = args.output / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError("Output protocol changed; choose a new run folder")
    write_json(protocol_file, protocol)
    history, best, best_ap, stale = [], None, -1.0, 0
    started = time.monotonic()
    for epoch in range(args.epochs):
        if time.monotonic() - started > args.minutes * 60 - 120:
            break
        order = list(range(len(train_rows)))
        random.shuffle(order)
        head.train()
        losses = []
        for offset in range(0, len(order), 8):
            batch = [train_rows[i] for i in order[offset:offset + 8]]
            logits = torch.stack([
                head(row["bag"].to(args.device), row["auxiliary"].to(args.device)).max()
                for row in batch
            ])
            targets = logits.new_tensor([row["label"] for row in batch])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        dev_values = scores(head, partitions["development"], args.device)
        dev_ap = float(average_precision_score(
            [row["label"] for row in partitions["development"]], dev_values))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "dev_ap": dev_ap})
        write_json(args.output / "history.json", history)
        print("TRAIN", json.dumps(history[-1]), flush=True)
        if dev_ap > best_ap + 1e-4:
            best, best_ap, stale = copy.deepcopy(head.state_dict()), dev_ap, 0
        else:
            stale += 1
        if stale >= 8:
            break
    if best is None:
        raise TimeoutError("No checkpoint produced within training budget")
    head.load_state_dict(best)
    dev_values = scores(head, partitions["development"], args.device)
    threshold = choose_threshold(partitions["development"], dev_values)
    test_values = scores(head, partitions["test"], args.device)
    parent_state = {key: value.detach().cpu() for key, value in parent.state_dict().items()}
    running_state = {key: value.detach().cpu() for key, value in head.state_dict().items()}
    checkpoint = args.output / "interaction_fixed_running.pt"
    temporary = checkpoint.with_suffix(".partial.pt")
    torch.save({"parent": parent_state, "running": running_state}, temporary)
    temporary.replace(checkpoint)
    metadata = {
        "checkpoint_sha256": digest(checkpoint),
        "parent_checkpoint_sha256": PARENT_SHA,
        "source_sha256": SOURCE_SHA, "spec": SPEC,
        "dataset_revision": DATASET_REVISION,
        "person_threshold": 0.8,
        "architecture": "frozen fight/wrestle/fall head + running head warm-started from parent",
        "running_head_version": 2,
        "motion_features": list(AUX_NAMES),
        "labels": ["fight_wrestle_fall", "fast_running"],
        "running_threshold": threshold,
        "parent_threshold": 0.836290717124939,
        "approved_for_demo": False,
    }
    write_json(args.output / "interaction_fixed_running.json", metadata)
    report = {
        "status": "completed", "epochs": len(history),
        "dataset": "UCF-ARG fixed ground camera only",
        "split": "actor-disjoint 01-08 / 09-10 / 11-12",
        "detector_coverage": coverage,
        "development": metrics(partitions["development"], dev_values, threshold),
        "test": metrics(partitions["test"], test_values, threshold),
        "limit": "24 independent test clips, 8 running; outdoor camera differs from indoor demo",
    }
    write_json(args.output / "report.json", report)
    archive = args.output / "mmaction_fixed_running_pilot.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ("interaction_fixed_running.pt", "interaction_fixed_running.json",
                     "report.json", "protocol.json"):
            bundle.write(args.output / name, name)
    print("FINISHED", json.dumps({"zip": str(archive), "test": report["test"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "weights", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stage", choices=("features", "train"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--minutes", type=int, default=90)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    torch.set_num_threads(4); cv2.setNumThreads(1)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = validate_manifest(manifest)
    protocol = {
        "manifest_sha256": digest(args.manifest),
        "engine_sha256": digest(Path(__file__).with_name("action_engine.py")),
        "trainer_sha256": digest(Path(__file__)),
        "parent_checkpoint_sha256": PARENT_SHA,
        "source_sha256": SOURCE_SHA, "spec": SPEC,
        "dataset_revision": DATASET_REVISION, "seed": 42,
    }
    if args.stage == "features":
        feature_stage(args, rows, protocol)
    else:
        train_stage(args, protocol)


if __name__ == "__main__":
    main()
