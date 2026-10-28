"""Fine-tune only a MIL interaction head on reviewed public intervals; never rewrite AVA weights."""

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
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from action_engine import ActionEngine, InteractionHead, ROOT, SOURCE_SHA, SPEC, digest


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate(corpus):
    groups, videos = {}, set()
    for source in corpus["sources"]:
        if (
            source["dataset"] not in {"airtlab", "rlvs"}
            or source["video"] in videos
            or source["split"] not in {"train", "development", "calibration", "test"}
        ):
            raise ValueError("Public sources and explicit unique splits required")
        videos.add(source["video"])
        groups.setdefault(source["group"], set()).add(source["split"])
        previous = 0
        for interval in source["intervals"]:
            start, end = interval["start_seconds"], interval["end_seconds"]
            if (
                interval["action"]
                not in {
                    "fight",
                    "hug",
                    "handshake_greet",
                    "highfive_clap",
                    "gesture",
                    "solo_motion",
                    "normal",
                    "uncertain",
                }
                or not previous <= start < end <= source["duration"] + 0.15
            ):
                raise ValueError("Invalid reviewed interval")
            previous = end
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("Performance-group leakage")


def annotation_at(source, anchor):
    intervals = source["intervals"]
    center = next(
        (
            item
            for item in intervals
            if item["start_seconds"] <= anchor < item["end_seconds"]
        ),
        None,
    )
    if center is None or center["action"] == "uncertain":
        return None, "unlabeled"
    start, end = anchor - 32 / 30, anchor + 24 / 30
    if any(
        item["action"] == "uncertain"
        and item["start_seconds"] < end
        and item["end_seconds"] > start
        for item in intervals
    ):
        return None, "uncertain"
    if center["action"] == "fight":
        return 1, "fight"
    known = sum(
        max(0, min(end, item["end_seconds"]) - max(start, item["start_seconds"]))
        for item in intervals
        if item["action"] not in {"fight", "uncertain"}
    )
    return (0, center["action"]) if known >= end - start - 1e-6 else (None, "boundary")


def public_frames(path):
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not 1 <= fps <= 120:
        raise ValueError("Cannot decode reviewed public preview")
    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame.shape[1] > 640:
            frame = cv2.resize(
                frame, (640, round(frame.shape[0] * 640 / frame.shape[1]))
            )
        frames.append(frame)
        if len(frames) > fps * 90:
            raise ValueError("Unexpected long review source")
    capture.release()
    return frames, fps


def cache_sources(engine, sources, pack, cache, deadline):
    rows = []
    for source in sources:
        path = (pack / source["media"]).resolve()
        if (
            not path.is_relative_to(pack.resolve())
            or digest(path) != source["preview_sha256"]
        ):
            raise ValueError("Public preview path/hash mismatch")
        frames, fps = public_frames(path)
        duration = len(frames) / fps
        if abs(duration - source["duration"]) > max(0.2, 2 / fps):
            raise ValueError("Review preview timebase changed")
        anchor = 32 / 30
        while anchor + 32 / 30 < duration:
            if time.monotonic() >= deadline:
                raise TimeoutError("Budget exhausted; feature cache remains resumable")
            label, action = annotation_at(source, anchor)
            key = hashlib.sha256(
                (source["video"] + f"/{anchor:.6f}").encode()
            ).hexdigest()[:24]
            target = cache / (key + ".npz")
            if not target.exists():
                indices = [
                    min(
                        len(frames) - 1,
                        round((anchor - 32 / 30 + index * 8 / 30) * fps),
                    )
                    for index in range(9)
                ]
                bag, prediction = engine.extract(
                    np.stack([frames[index] for index in indices])
                )
                temporary = target.with_suffix(".partial.npz")
                np.savez_compressed(
                    temporary,
                    bag=bag,
                    base=np.float32(prediction["base_score"]),
                    people=np.int32(prediction["people"]),
                )
                temporary.replace(target)
            rows.append(
                {
                    "key": key,
                    "video": source["video"],
                    "dataset": source["dataset"],
                    "group": source["group"],
                    "split": source["split"],
                    "label": label,
                    "action": action,
                    "anchor": anchor,
                    "timestamp": anchor + 32 / 30,
                    "duration": duration,
                }
            )
            anchor += SPEC["decision_stride_seconds"]
        print(
            "FEATURES",
            source["split"],
            source["video"],
            "total_windows",
            len(rows),
            flush=True,
        )
    return rows


def load_features(rows, cache, device):
    values = []
    for row in rows:
        with np.load(cache / (row["key"] + ".npz"), allow_pickle=False) as record:
            values.append(
                {
                    **row,
                    "bag": torch.from_numpy(record["bag"]).to(device),
                    "base": float(record["base"]),
                    "people": int(record["people"]),
                }
            )
    return values


@torch.inference_mode()
def scores(head, rows):
    head.eval()
    return [
        float(head(row["bag"]).max().sigmoid().cpu()) if len(row["bag"]) else 0.0
        for row in rows
    ]


def metrics(rows, values, threshold):
    selected = [
        (row, score) for row, score in zip(rows, values) if row["label"] is not None
    ]
    labels = [row["label"] for row, _ in selected]
    actual = [score for _, score in selected]
    if set(labels) != {0, 1}:
        raise ValueError("Both reviewed classes required for evaluation")
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, np.asarray(actual) >= threshold, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ap": float(average_precision_score(labels, actual)),
        "auc": float(roc_auc_score(labels, actual)),
        "windows": len(labels),
        "groups": len({row["group"] for row, _ in selected}),
        "positives_without_person": sum(
            row["label"] == 1 and not row["people"] for row, _ in selected
        ),
    }


def development_rank(rows, values):
    reports = []
    for cohort in ("airtlab", "rlvs"):
        selected = [
            (row, score) for row, score in zip(rows, values) if row["dataset"] == cohort
        ]
        reports.append(
            metrics(
                [row for row, _ in selected], [score for _, score in selected], 0.5
            )["ap"]
        )
    return min(reports)


def threshold_on_calibration(rows, values):
    unique = np.unique(values)
    choices = np.clip(
        np.r_[1e-6, (unique[:-1] + unique[1:]) / 2, 1 - 1e-6], 1e-6, 1 - 1e-6
    )
    return float(
        max(
            choices,
            key=lambda threshold: (
                min(
                    metrics(rows, values, float(threshold))[key]
                    for key in ("precision", "recall")
                ),
                metrics(rows, values, float(threshold))["f1"],
                threshold,
            ),
        )
    )


def event_metrics(rows, values, sources, threshold):
    from fight.reviewed_events import reviewed_event_metrics, summarize_events
    from fight.runtime import EventGate

    reports = []
    for source in sources:
        points = [
            {"timestamp": row["timestamp"], "score": score, "people": row["people"]}
            for row, score in zip(rows, values)
            if row["video"] == source["video"]
        ]
        gate, events, active = (
            EventGate(threshold, threshold * 0.7, confirm_seconds=0.4),
            [],
            None,
        )
        for point in points:
            if not point["people"]:
                if active:
                    active["end"] = point["timestamp"]
                gate, active = gate.fresh(), None
                continue
            _, event = gate.update(point["score"], point["timestamp"])
            if event == "started":
                active = {"start": point["timestamp"], "end": source["duration"]}
                events.append(active)
            elif event == "ended" and active:
                active["end"], active = point["timestamp"], None
        reports.append(reviewed_event_metrics(events, source["intervals"]))
    return summarize_events(reports)


def run(args):
    started = time.monotonic()
    deadline = started + args.minutes * 60
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate(corpus)
    protocol = {
        "manifest_sha256": digest(args.manifest),
        "engine_sha256": digest(ROOT / "action_engine.py"),
        "training_code_sha256": digest(Path(__file__)),
        "source_sha256": SOURCE_SHA,
        "spec": SPEC,
        "seed": 42,
        "training_mode": "Frozen original backbone and AVA head; train a separate multiple-instance interaction head",
        "private_data_used": False,
    }
    saved = args.output / "protocol.json"
    if saved.exists() and json.loads(saved.read_text()) != protocol:
        raise ValueError("Run fingerprint changed; use a new output folder")
    write(saved, protocol)
    if (args.output / "report.json").is_file():
        print(
            "Run already completed; retaining its frozen selection and test report",
            flush=True,
        )
        return
    engine = ActionEngine(weights=args.weights, device=args.device)
    public = [source for source in corpus["sources"] if source["split"] != "test"]
    rows = cache_sources(engine, public, args.pack, args.cache, deadline)
    loaded = load_features(rows, args.cache, args.device)
    partitions = {
        split: [row for row in loaded if row["split"] == split]
        for split in ("train", "development", "calibration")
    }
    training = [
        row
        for row in partitions["train"]
        if row["label"] is not None and len(row["bag"])
    ]
    if {row["label"] for row in training} != {0, 1}:
        raise ValueError("Training requires both reviewed classes with person coverage")
    write(
        args.output / "preflight.json",
        {
            "scope": corpus["scope"],
            "windows": {split: len(values) for split, values in partitions.items()},
            "training_sources": len({row["video"] for row in training}),
            "training_groups": len({row["group"] for row in training}),
            "original_weights_trainable": False,
        },
    )
    head = InteractionHead().to(args.device)
    all_train_features = torch.cat([row["bag"] for row in training])
    head.mean.copy_(all_train_features.mean(0))
    head.std.copy_(all_train_features.std(0).clamp_min(0.05))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0003, weight_decay=0.01)
    counts = Counter((row["dataset"], row["label"], row["group"]) for row in training)
    group_counts = Counter((dataset, label) for dataset, label, _ in counts)
    weights = torch.tensor(
        [
            1
            / (
                counts[(row["dataset"], row["label"], row["group"])]
                * group_counts[(row["dataset"], row["label"])]
            )
            for row in training
        ],
        dtype=torch.float64,
    )
    frozen_path = args.output / "selection_frozen.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    best, best_rank, stale, history = None, -1.0, 0, []
    if frozen:
        if digest(args.output / "interaction.pt") != frozen["checkpoint_sha256"]:
            raise ValueError(
                "Frozen checkpoint changed; refusing to select again using test"
            )
        best = torch.load(
            args.output / "interaction.pt", map_location=args.device, weights_only=True
        )
        best_rank = frozen["development_rank"]
        history = json.loads((args.output / "history.json").read_text())
    for epoch in range(0 if frozen else args.epochs):
        if time.monotonic() >= deadline - 180:
            break
        head.train()
        order = torch.multinomial(
            weights, max(128, len(training)), replacement=True
        ).tolist()
        loss_sum = 0.0
        for offset in range(0, len(order), 8):
            optimizer.zero_grad()
            batch = [training[index] for index in order[offset : offset + 8]]
            logits = torch.stack([head(row["bag"]).max() for row in batch])
            target = logits.new_tensor([row["label"] for row in batch])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        values = scores(head, partitions["development"])
        rank = development_rank(partitions["development"], values)
        record = {
            "epoch": epoch + 1,
            "loss": loss_sum / len(order),
            "development_worst_cohort_ap": rank,
        }
        history.append(record)
        print("TRAIN", json.dumps(record), flush=True)
        if rank > best_rank + 1e-4:
            best, best_rank, stale = copy.deepcopy(head.state_dict()), rank, 0
            torch.save(
                {key: value.detach().cpu() for key, value in best.items()},
                args.output / "interaction.pt",
            )
        else:
            stale += 1
        write(args.output / "history.json", history)
        if stale >= 15:
            break
    if best is None:
        raise TimeoutError("No trained checkpoint within the budget")
    head.load_state_dict(best)
    calibration = partitions["calibration"]
    calibration_scores = scores(head, calibration)
    threshold = (
        frozen["threshold"]
        if frozen
        else threshold_on_calibration(calibration, calibration_scores)
    )
    metadata = {
        **protocol,
        "checkpoint_sha256": digest(args.output / "interaction.pt"),
        "threshold": threshold,
        "confirm_seconds": 0.4,
        "approved_for_demo": False,
        "real_phone_tested": False,
        "scope": "Small AI-assisted repeated-public pilot; original AVA head stays frozen; not camera acceptance",
        "epochs": len(history),
    }
    write(args.output / "interaction.json", metadata)
    development = partitions["development"]
    freeze = {
        "threshold": threshold,
        "development_rank": best_rank,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
    }
    write(args.output / "selection_frozen.json", freeze)
    testing_sources = [
        source for source in corpus["sources"] if source["split"] == "test"
    ]
    test_rows = cache_sources(engine, testing_sources, args.pack, args.cache, deadline)
    testing = load_features(test_rows, args.cache, args.device)
    report = {
        "status": "completed",
        "scope": metadata["scope"],
        "original_ava_parameters_changed": False,
        "training": metadata,
        "test": {},
        "seconds": time.monotonic() - started,
    }
    for variant, values, operating_point in (
        ("original", [row["base"] for row in testing], 0.4),
        ("interaction", scores(head, testing), threshold),
    ):
        report["test"][variant] = {
            "windows": metrics(testing, values, operating_point),
            "events": event_metrics(testing, values, testing_sources, operating_point),
        }
        report["test"][variant]["by_dataset"] = {}
        for dataset in ("airtlab", "rlvs"):
            subset = [
                (row, score)
                for row, score in zip(testing, values)
                if row["dataset"] == dataset
            ]
            report["test"][variant]["by_dataset"][dataset] = metrics(
                [row for row, _ in subset],
                [score for _, score in subset],
                operating_point,
            )
    report["development"] = {
        "original_worst_cohort_ap": development_rank(
            development, [row["base"] for row in development]
        ),
        "interaction_worst_cohort_ap": best_rank,
    }
    reference = next(row["bag"] for row in training if len(row["bag"]))
    head.eval()
    gpu_score = head(reference).detach().cpu()
    cpu_head = copy.deepcopy(head).cpu().eval()
    cpu_score = cpu_head(reference.cpu()).detach()
    report["head_gpu_cpu_maximum_logit_difference"] = float(
        (gpu_score - cpu_score).abs().max()
    )
    write(args.output / "report.json", report)
    with zipfile.ZipFile(
        args.output / "mmaction_interaction_pilot.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as package:
        for name in (
            "interaction.pt",
            "interaction.json",
            "report.json",
            "protocol.json",
        ):
            package.write(args.output / name, name)
    print("FINISHED", json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "pack", "cache", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--epochs", type=int, default=120)
    arguments = parser.parse_args()
    if not 1 <= arguments.minutes <= 60 or not 1 <= arguments.epochs <= 500:
        parser.error("Bounded budget and epoch count required")
    run(arguments)
