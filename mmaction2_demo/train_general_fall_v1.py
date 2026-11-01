"""Colab-only pilot: warm-start interaction v1 with public fight and fall videos.

The AVA backbone and detector are frozen.  This trains only InteractionHead.
SCFD has a broad fight label (its publisher says wrestling is included), not a
verified wrestling subtype label.  The report never claims subtype accuracy.
"""

import argparse
from collections import Counter
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

from action_engine import ActionEngine, InteractionHead, SOURCE_SHA, SPEC, digest


PARENT_SHA = "8c5808db106aacd0f43b72a0e76178b2dec039e0a1766d996bb8374b7e4dddb7"
PREVIOUS_TRAINER_SHA = "2ecbd75660f4bde6410429f8e1599f993f3c19e78f1916568f1e19b897faa4cb"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sources(manifest, fall_root, scfd_root):
    """Exclude the one fall with unknown timing and keep subjects apart."""
    rows = []
    for item in manifest["sources"]:
        if item["annotation_status"] == "needs_temporal_review":
            continue
        if item["dataset"] != "gmdcsa24_v2_1" or item["split"] not in {"train", "development", "test"}:
            raise ValueError("Unexpected fall source")
        interval = item["intervals"][0]
        rows.append({"video": "gmdcsa24/" + item["video"],
                     "path": str(Path(fall_root) / item["video"]),
                     "dataset": "gmdcsa24", "group": item["group"],
                     "split": item["split"], "label": interval["label"],
                     "start": interval["start_seconds"], "end": interval["end_seconds"],
                     "publisher_duration": item["publisher_duration_seconds"]})
    for folder, label in (("fight", 1), ("noFight", 0)):
        clips = sorted((Path(scfd_root) / folder).glob("*.mp4"))
        if len(clips) != 150:
            raise ValueError(f"SCFD {folder}: expected 150 clips, found {len(clips)}")
        for clip in clips:
            bucket = int(hashlib.sha256((folder + "/" + clip.name).encode()).hexdigest()[:8], 16) % 10
            split = "development" if bucket == 0 else "test" if bucket == 1 else "train"
            rows.append({"video": "scfd/" + folder + "/" + clip.name,
                         "path": str(clip), "dataset": "scfd", "group": "scfd/" + folder + "/" + clip.stem,
                         "split": split, "label": label,
                         "start": 0.0, "end": None, "publisher_duration": None})
    if len(rows) != 459 or len({row["video"] for row in rows}) != len(rows):
        raise ValueError("Incomplete or duplicate public sources")
    groups = {}
    for row in rows:
        key = (row["dataset"], row["group"])
        if key in groups and groups[key] != row["split"]:
            raise ValueError("Subject/performance leakage")
        groups[key] = row["split"]
    return rows


def anchors(source, duration):
    if source["dataset"] == "scfd":
        return [duration / 2]
    if source["label"] == 0:
        return np.linspace(max(0.1, duration * .2), duration * .8, 3).tolist()
    start, end = source["start"], min(duration, source["end"])
    if not 0 <= start < end:
        raise ValueError("Fall interval outside decoded video: " + source["video"])
    # A short fall still contributes a centered positive window.  Never label
    # the unlabeled before/after portion of a fall video as negative.
    return np.linspace(start + (end - start) * .25,
                       start + (end - start) * .75, 2).tolist()


def frames_for_video(path):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not 1 <= fps <= 120:
        raise ValueError("Cannot decode: " + str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[1] > 640:
            frame = cv2.resize(frame, (640, round(frame.shape[0] * 640 / frame.shape[1])))
        frames.append(frame)
        if len(frames) > fps * 120:
            raise ValueError("Unexpectedly long video: " + str(path))
    capture.release()
    if len(frames) < 3:
        raise ValueError("Too few decoded frames: " + str(path))
    return frames, fps


def sampled_clip(frames, fps, anchor):
    indexes = [min(len(frames) - 1, max(0, round((anchor + (i - 4) * 8 / 30) * fps)))
               for i in range(9)]
    return np.stack([frames[index] for index in indexes])


def duration_decision(row, decoded_duration):
    expected = row["publisher_duration"]
    if expected is None or abs(decoded_duration - expected) <= 1.0:
        return "use"
    # ADL is labeled for the whole clip, so its decoded duration is sufficient.
    # Fall timestamps could point to a different moment: exclude those clips.
    return "use_decoded" if row["dataset"] == "gmdcsa24" and row["label"] == 0 else "skip"


def compatible_cache(previous, current):
    if previous == current:
        return True
    return (previous.get("trainer_sha256") == PREVIOUS_TRAINER_SHA
            and {k: v for k, v in previous.items() if k != "trainer_sha256"}
            == {k: v for k, v in current.items() if k != "trainer_sha256"})


def features(args, rows, protocol):
    args.cache.mkdir(parents=True, exist_ok=True)
    marker = args.cache / "protocol.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if not compatible_cache(previous, protocol):
            raise ValueError("Feature-cache protocol changed; choose a new cache folder")
        if previous != protocol:
            print("Reusing verified features from previous trainer revision", flush=True)
    write_json(marker, protocol)
    engine = ActionEngine(weights=args.weights, device=args.device)
    index = []
    skipped = []
    for number, row in enumerate(rows, 1):
        path = Path(row["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        frames, fps = frames_for_video(path)
        duration = len(frames) / fps
        decision = duration_decision(row, duration)
        if decision == "skip":
            skipped.append({"video": row["video"], "publisher_duration": row["publisher_duration"],
                            "decoded_duration": duration, "reason": "unsafe_temporal_labels"})
            print("SKIP duration mismatch:", row["video"], flush=True)
            continue
        if decision == "use_decoded":
            print("Using decoded duration for ADL:", row["video"], flush=True)
        for anchor in anchors(row, duration):
            key = hashlib.sha256((row["video"] + f"/{anchor:.6f}").encode()).hexdigest()[:24]
            feature_path = args.cache / (key + ".npz")
            if not feature_path.exists():
                bag, info = engine.extract(sampled_clip(frames, fps, anchor))
                temporary = args.cache / (key + ".partial.npz")
                np.savez_compressed(temporary, bag=bag.astype(np.float32),
                                    people=np.int32(info["people"]))
                temporary.replace(feature_path)
            index.append({"key": key, "video": row["video"], "dataset": row["dataset"],
                          "group": row["group"], "split": row["split"],
                          "label": row["label"], "anchor": anchor})
        if number % 10 == 0 or number == len(rows):
            print("FEATURES", number, "/", len(rows), flush=True)
    write_json(args.cache / "index.json", index)
    write_json(args.cache / "skipped_sources.json", skipped)


def load_rows(cache):
    rows = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    for row in rows:
        with np.load(cache / (row["key"] + ".npz"), allow_pickle=False) as value:
            row["bag"] = torch.from_numpy(value["bag"].copy())
            row["people"] = int(value["people"])
    return rows


@torch.inference_mode()
def scores(head, rows, device):
    head.eval()
    return [float(head(row["bag"].to(device)).max().sigmoid().cpu())
            if len(row["bag"]) else 0.0 for row in rows]


def report(rows, values, threshold):
    labels = [row["label"] for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError("Evaluation split needs both classes")
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, np.asarray(values) >= threshold, average="binary", zero_division=0)
    return {"ap": float(average_precision_score(labels, values)),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "windows": len(rows), "positives_without_person": sum(
                row["label"] == 1 and not row["people"] for row in rows)}


def choose_threshold(rows, values):
    choices = np.unique(np.r_[.05, np.asarray(values), .95])
    return float(max(choices, key=lambda t: (report(rows, values, float(t))["f1"], t)))


def train(args, protocol):
    rows = load_rows(args.cache)
    partitions = {split: [row for row in rows if row["split"] == split]
                  for split in ("train", "development", "test")}
    train_rows = [row for row in partitions["train"] if row["people"]]
    for split, part in (("train", train_rows), ("development", partitions["development"]),
                        ("test", partitions["test"])):
        if {row["label"] for row in part} != {0, 1}:
            raise ValueError("Missing a class in " + split)
    metadata = json.loads((args.weights / "interaction.json").read_text(encoding="utf-8"))
    if (digest(args.weights / "interaction.pt") != metadata["checkpoint_sha256"]
            or metadata["checkpoint_sha256"] != PARENT_SHA
            or metadata["source_sha256"] != SOURCE_SHA or metadata["spec"] != SPEC
            or int(metadata.get("head_version", 1)) != 1):
        raise ValueError("Wrong interaction-v1 parent checkpoint")
    head = InteractionHead().to(args.device)
    head.load_state_dict(torch.load(args.weights / "interaction.pt", map_location="cpu",
                                    weights_only=True), strict=True)
    base_dev = scores(head, partitions["development"], args.device)
    base_test = scores(head, partitions["test"], args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Training protocol changed; use a new output folder")
    write_json(protocol_path, protocol)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.01)
    counts = Counter((r["dataset"], r["label"]) for r in train_rows)
    weights = torch.tensor([1 / counts[(r["dataset"], r["label"])] for r in train_rows],
                           dtype=torch.double)
    weights /= weights.sum()
    best, best_ap, history, stale = None, -1.0, [], 0
    started = time.monotonic()
    for epoch in range(args.epochs):
        if time.monotonic() - started > args.minutes * 60 - 120:
            print("TRAIN budget reached at epoch", epoch, flush=True)
            break
        head.train()
        order = torch.multinomial(weights, max(128, len(train_rows)), replacement=True).tolist()
        losses = []
        for offset in range(0, len(order), 8):
            optimizer.zero_grad()
            batch = [train_rows[i] for i in order[offset:offset + 8]]
            logits = torch.stack([head(row["bag"].to(args.device)).max() for row in batch])
            labels = logits.new_tensor([row["label"] for row in batch])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        dev_scores = scores(head, partitions["development"], args.device)
        ap = report(partitions["development"], dev_scores, .5)["ap"]
        history.append({"epoch": epoch + 1, "loss": sum(losses) / len(losses), "dev_ap": ap})
        write_json(args.output / "history.json", history)
        if ap > best_ap + 1e-4:
            best, best_ap, stale = copy.deepcopy(head.state_dict()), ap, 0
            torch.save({k: v.detach().cpu() for k, v in best.items()}, args.output / "interaction.pt")
        else:
            stale += 1
        print("TRAIN", json.dumps(history[-1]), flush=True)
        if stale >= 8:
            break
    if best is None:
        raise TimeoutError("No checkpoint produced within the training budget")
    head.load_state_dict(best)
    selected_dev = scores(head, partitions["development"], args.device)
    threshold = choose_threshold(partitions["development"], selected_dev)
    test_scores = scores(head, partitions["test"], args.device)
    results = {
        "status": "completed", "parent_checkpoint_sha256": PARENT_SHA,
        "head_version": 1, "training_sources": len({r["video"] for r in train_rows}),
        "source_policy": "public SCFD broad fight + GMDCSA24 general fall/ADL",
        "skipped_sources": json.loads((args.cache / "skipped_sources.json").read_text(encoding="utf-8")),
        "scfd_clip_split_scene_independence_verified": False,
        "wrestle_specific_labels_available": False,
        "wrestle_specific_accuracy_measured": False,
        "development": {"parent": report(partitions["development"], base_dev, threshold),
                        "new": report(partitions["development"], selected_dev, threshold)},
        "test": {"parent": report(partitions["test"], base_test, threshold),
                 "new": report(partitions["test"], test_scores, threshold)},
        "test_by_dataset": {
            dataset: {
                "parent": report([r for r in partitions["test"] if r["dataset"] == dataset],
                                 [v for r, v in zip(partitions["test"], base_test) if r["dataset"] == dataset],
                                 threshold),
                "new": report([r for r in partitions["test"] if r["dataset"] == dataset],
                              [v for r, v in zip(partitions["test"], test_scores) if r["dataset"] == dataset],
                              threshold),
            }
            for dataset in ("gmdcsa24", "scfd")
        },
        "test_subject": 4, "threshold": threshold,
        "positive_without_person": sum(r["label"] == 1 and not r["people"] for r in rows),
        "epochs": len(history), "training_started_on_local_machine": False,
        "approved_for_demo": False,
    }
    write_json(args.output / "report.json", results)
    output_metadata = {**metadata, "checkpoint_sha256": digest(args.output / "interaction.pt"),
                       "threshold": threshold, "parent_checkpoint_sha256": PARENT_SHA,
                       "scope": "Public fight + fall pilot; wrestling subset not separately verified",
                       "approved_for_demo": False, "real_phone_tested": False,
                       "head_version": 1}
    write_json(args.output / "interaction.json", output_metadata)
    bundle = args.output / "mmaction_interaction_fall_wrestle_pilot.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as package:
        for name in ("interaction.pt", "interaction.json", "report.json", "protocol.json"):
            package.write(args.output / name, name)
    print("FINISHED", json.dumps({"checkpoint": str(bundle), "test": results["test"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "fall-root", "scfd-root", "weights", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stage", choices=("features", "train"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--minutes", type=int, default=90)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or not 5 <= args.minutes <= 600:
        parser.error("Bounded training budget required")
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    torch.set_num_threads(4); cv2.setNumThreads(1)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["publisher_tree_sha"] != "5abac7693229900cf80f722e878fbb119211fc1c":
        raise ValueError("Wrong GMDCSA24 v2.1 manifest")
    rows = sources(manifest, args.fall_root, args.scfd_root)
    protocol = {"manifest_sha256": digest(args.manifest),
                "engine_sha256": digest(Path(__file__).with_name("action_engine.py")),
                "trainer_sha256": digest(Path(__file__)),
                "parent_checkpoint_sha256": PARENT_SHA,
                "source_sha256": SOURCE_SHA, "spec": SPEC,
                "scfd_revision": args.scfd_root.name,
                "positive_class": "fight_or_wrestle_or_fall", "seed": 42}
    if args.stage == "features":
        features(args, rows, protocol)
    else:
        if json.loads((args.cache / "protocol.json").read_text(encoding="utf-8")) != protocol:
            raise ValueError("Feature/training protocol mismatch")
        train(args, protocol)


if __name__ == "__main__":
    main()
