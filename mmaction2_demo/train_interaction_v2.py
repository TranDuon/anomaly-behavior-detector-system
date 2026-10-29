"""Train a public-only motion/base-aware interaction head with hard-negative mining."""

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import zipfile

os.environ["MPLBACKEND"] = "Agg"

import cv2
import numpy as np
import torch

from action_engine import (
    AUX_NAMES,
    AUX_VERSION,
    ActionEngine,
    InteractionHeadV2,
    ROOT,
    SOURCE_SHA,
    SPEC,
    digest,
)
from train_interaction import (
    annotation_at,
    event_metrics,
    metrics,
    public_frames,
    write,
)


INDOOR_DATASETS = {"airtlab", "sbu", "ntu_rgbd120", "vid_classroom"}
INDOOR_NEGATIVE_ACTIONS = {
    "hug",
    "handshake_greet",
    "highfive_clap",
    "gesture",
    "solo_motion",
    "normal",
    "approach",
    "depart",
    "stand_close",
    "walk_together",
    "pat_back",
    "point",
    "give_exchange",
    "follow",
    "whisper",
    "cheers",
}
HARD_NEGATIVE_ACTIONS = {
    "hug": 3.0,
    "handshake_greet": 3.0,
    "highfive_clap": 3.0,
    "stand_close": 3.5,
    "approach": 3.5,
    "depart": 2.5,
    "walk_together": 2.5,
    "pat_back": 3.0,
    "give_exchange": 2.5,
    "gesture": 2.0,
    "solo_motion": 2.0,
    "normal": 1.5,
}


def validate_indoor_corpus(corpus):
    if (
        corpus.get("schema_version") != 2
        or corpus.get("scope_profile") != "indoor_room_fight_v2"
        or corpus.get("readiness") != "ready"
    ):
        raise ValueError(
            "Indoor v2 manifest is not ready; curate/review the required datasets first"
        )
    groups, videos = {}, set()
    allowed_actions = INDOOR_NEGATIVE_ACTIONS | {"fight", "uncertain"}
    for source in corpus.get("sources", []):
        actions = [item.get("action") for item in source.get("intervals", [])]
        if (
            source.get("dataset") not in INDOOR_DATASETS
            or source.get("environment") != "indoor"
            or source.get("video") in videos
            or source.get("split")
            not in {"train", "development", "calibration", "test"}
            or any(action not in allowed_actions for action in actions)
            or (
                source.get("source_label") == "non_fight"
                and "fight" in actions
            )
        ):
            raise ValueError("Indoor-only source policy or labels are invalid")
        videos.add(source["video"])
        groups.setdefault(source["group"], set()).add(source["split"])
        previous = 0
        for interval in source["intervals"]:
            start, end = interval["start_seconds"], interval["end_seconds"]
            if not previous <= start < end <= source["duration"] + 0.15:
                raise ValueError("Invalid reviewed indoor interval")
            previous = end
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("Indoor performance-group leakage")


def indoor_annotation_at(source, anchor):
    """Label a window, including short Kinect clips padded at their boundaries."""
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
    start = max(0.0, anchor - 32 / 30)
    end = min(source["duration"], anchor + 24 / 30)
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
        half_window = 32 / 30
        anchors = []
        anchor = half_window
        while anchor + half_window < duration:
            anchors.append(anchor)
            anchor += SPEC["decision_stride_seconds"]
        if not anchors and duration >= 0.6:
            anchors.append(duration / 2)
        for anchor in anchors:
            if time.monotonic() >= deadline:
                raise TimeoutError("Budget exhausted; feature cache remains resumable")
            label, action = indoor_annotation_at(source, anchor)
            key = hashlib.sha256(
                (source["video"] + f"/{anchor:.6f}").encode()
            ).hexdigest()[:24]
            target = cache / (key + ".npz")
            indices = [
                    max(
                        0,
                        min(
                    len(frames) - 1,
                    round((anchor - 32 / 30 + index * 8 / 30) * fps),
                        ),
                )
                for index in range(9)
            ]
            window = np.stack([frames[index] for index in indices])
            cached = None
            if target.exists():
                with np.load(target, allow_pickle=False) as record:
                    cached = {name: record[name] for name in record.files}
            cached_version = int(cached.get("aux_version", -1)) if cached else -1
            if cached is None or cached_version != AUX_VERSION:
                bag, auxiliary, prediction = engine.extract_features(window)
                cached = {
                    "bag": bag,
                    "aux": auxiliary,
                    "base": np.float32(prediction["base_score"]),
                    "people": np.int32(prediction["people"]),
                    "aux_version": np.int32(AUX_VERSION),
                }
            if (
                int(cached["aux_version"]) != AUX_VERSION
                or len(cached["bag"]) != len(cached["aux"])
            ):
                raise ValueError("Incompatible v2 feature cache")
            temporary = target.with_suffix(".partial.npz")
            np.savez_compressed(temporary, **cached)
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
                    "timestamp": min(duration, anchor + 32 / 30),
                    "duration": duration,
                }
            )
        print("FEATURES_V2", source["split"], source["video"], flush=True)
    return rows


def load_features(rows, cache, device):
    loaded = []
    for row in rows:
        with np.load(cache / (row["key"] + ".npz"), allow_pickle=False) as record:
            loaded.append(
                {
                    **row,
                    "bag": torch.from_numpy(record["bag"]).to(device),
                    "aux": torch.from_numpy(record["aux"]).to(device),
                    "base": float(record["base"]),
                    "people": int(record["people"]),
                }
            )
    return loaded


@torch.inference_mode()
def scores(head, rows):
    head.eval()
    return [
        float(head(row["bag"], row["aux"]).max().sigmoid().cpu())
        if len(row["bag"])
        else 0.0
        for row in rows
    ]


def operating_point(rows, values):
    unique = np.unique(values)
    choices = np.clip(
        np.r_[1e-6, (unique[:-1] + unique[1:]) / 2, 1 - 1e-6],
        1e-6,
        1 - 1e-6,
    )
    candidates = []
    for threshold in choices:
        report = metrics(rows, values, float(threshold))
        negatives = [
            score
            for row, score in zip(rows, values)
            if row["label"] == 0
        ]
        report["false_positive_rate"] = float(
            np.mean(np.asarray(negatives) >= threshold)
        )
        report["threshold"] = float(threshold)
        report["target_feasible"] = (
            report["precision"] >= 0.75
            and report["recall"] >= 0.60
            and report["false_positive_rate"] <= 0.10
        )
        candidates.append(report)
    feasible = [item for item in candidates if item["target_feasible"]]
    selected = max(
        feasible or candidates,
        key=lambda item: (
            item["target_feasible"],
            min(item["precision"], item["recall"]),
            item["f1"],
            -item["false_positive_rate"],
            item["threshold"],
        ),
    )
    return selected


def hard_negative_report(rows, values, threshold):
    report = {}
    for action in sorted(HARD_NEGATIVE_ACTIONS):
        selected = [
            score
            for row, score in zip(rows, values)
            if row["label"] == 0 and row["action"] == action
        ]
        report[action] = {
            "windows": len(selected),
            "false_positives": sum(score >= threshold for score in selected),
            "maximum_score": max(selected, default=None),
        }
    return report


def require_both_classes(rows, purpose):
    labels = {row["label"] for row in rows if row["label"] is not None}
    if labels != {0, 1}:
        raise ValueError(f"{purpose} requires both reviewed classes; found {sorted(labels)}")


def development_rank(rows, values):
    """Rank the weakest evaluable indoor dataset, never an obsolete RLVS cohort."""
    cohorts = []
    for dataset in sorted({row["dataset"] for row in rows}):
        selected = [
            (row, score)
            for row, score in zip(rows, values)
            if row["dataset"] == dataset and row["label"] is not None
        ]
        if {row["label"] for row, _ in selected} == {0, 1}:
            cohorts.append(
                metrics(
                    [row for row, _ in selected],
                    [score for _, score in selected],
                    0.5,
                )["ap"]
            )
    if not cohorts:
        raise ValueError("Development split has no indoor dataset with both reviewed classes")
    return min(cohorts)


def train_loss(head, row):
    logits = head(row["bag"], row["aux"])
    target = logits.new_tensor(float(row["label"]))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.max(), target)
    if row["label"] == 0:
        loss = loss + 0.35 * torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits)
        )
    return loss


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
    validate_indoor_corpus(corpus)
    protocol = {
        "manifest_sha256": digest(args.manifest),
        "engine_sha256": digest(ROOT / "action_engine.py"),
        "training_code_sha256": digest(Path(__file__)),
        "source_sha256": SOURCE_SHA,
        "spec": SPEC,
        "head_version": 2,
        "aux_version": AUX_VERSION,
        "aux_names": list(AUX_NAMES),
        "seed": 42,
        "training_mode": "Frozen AVA features plus base score, people context and temporal motion; hard-negative-weighted MIL head",
        "private_data_used": False,
    }
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Run fingerprint changed; use a new output folder")
    write(protocol_path, protocol)
    if (args.output / "report.json").is_file():
        print("Run already completed; retaining frozen selection", flush=True)
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
        raise ValueError("Training requires both classes with person coverage")
    require_both_classes(partitions["development"], "Development split")
    require_both_classes(partitions["calibration"], "Calibration split")

    head = InteractionHeadV2().to(args.device)
    head.mean.copy_(torch.cat([row["bag"] for row in training]).mean(0))
    head.std.copy_(torch.cat([row["bag"] for row in training]).std(0).clamp_min(0.05))
    all_aux = torch.cat([row["aux"] for row in training])
    head.aux_mean.copy_(all_aux.mean(0))
    head.aux_std.copy_(all_aux.std(0).clamp_min(0.01))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.00025, weight_decay=0.02)
    counts = Counter((row["dataset"], row["label"], row["group"]) for row in training)
    group_counts = Counter((dataset, label) for dataset, label, _ in counts)
    base_weights = np.asarray(
        [
            HARD_NEGATIVE_ACTIONS.get(row["action"], 1.0)
            / (
                counts[(row["dataset"], row["label"], row["group"])]
                * group_counts[(row["dataset"], row["label"])]
            )
            for row in training
        ],
        dtype=np.float64,
    )
    frozen_path = args.output / "selection_frozen.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    best, best_rank, stale, history = None, -1.0, 0, []
    if frozen:
        if digest(args.output / "interaction.pt") != frozen["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint changed")
        best = torch.load(
            args.output / "interaction.pt", map_location=args.device, weights_only=True
        )
        best_rank = frozen["development_rank"]
        history = json.loads((args.output / "history.json").read_text())

    for epoch in range(0 if frozen else args.epochs):
        if time.monotonic() >= deadline - 180:
            break
        hardness = np.ones(len(training))
        if epoch >= 3:
            train_scores = scores(head, training)
            hardness = np.asarray(
                [
                    1 + 4 * score if row["label"] == 0 else 1
                    for row, score in zip(training, train_scores)
                ]
            )
        weights = torch.as_tensor(base_weights * hardness, dtype=torch.float64)
        order = torch.multinomial(
            weights, max(128, len(training)), replacement=True
        ).tolist()
        head.train()
        loss_sum = 0.0
        for offset in range(0, len(order), 8):
            optimizer.zero_grad()
            losses = [train_loss(head, training[index]) for index in order[offset : offset + 8]]
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(losses)
        values = scores(head, partitions["development"])
        rank = development_rank(partitions["development"], values)
        record = {
            "epoch": epoch + 1,
            "loss": loss_sum / len(order),
            "development_worst_cohort_ap": rank,
        }
        history.append(record)
        print("TRAIN_V2", json.dumps(record), flush=True)
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
    selected = (
        frozen["operating_point"]
        if frozen
        else operating_point(calibration, calibration_scores)
    )
    threshold = selected["threshold"]
    metadata = {
        **protocol,
        "checkpoint_sha256": digest(args.output / "interaction.pt"),
        "threshold": threshold,
        "confirm_seconds": 0.4,
        "operating_point_feasible": selected["target_feasible"],
        "approved_for_demo": False,
        "real_phone_tested": False,
        "scope": "Public-only hard-negative v2 pilot; private camera clips are local acceptance only",
        "epochs": len(history),
    }
    write(args.output / "interaction.json", metadata)
    freeze = {
        "operating_point": selected,
        "development_rank": best_rank,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
    }
    write(args.output / "selection_frozen.json", freeze)

    test_sources = [source for source in corpus["sources"] if source["split"] == "test"]
    test_rows = cache_sources(engine, test_sources, args.pack, args.cache, deadline)
    testing = load_features(test_rows, args.cache, args.device)
    development = partitions["development"]
    report = {
        "status": "completed",
        "scope": metadata["scope"],
        "training": metadata,
        "calibration": selected,
        "development": {
            "original_worst_cohort_ap": development_rank(
                development, [row["base"] for row in development]
            ),
            "fusion_v2_worst_cohort_ap": best_rank,
            "hard_negatives": hard_negative_report(
                development, scores(head, development), threshold
            ),
        },
        "test": {},
        "seconds": time.monotonic() - started,
    }
    for variant, values, point in (
        ("original", [row["base"] for row in testing], 0.4),
        ("fusion_v2", scores(head, testing), threshold),
    ):
        report["test"][variant] = {
            "windows": metrics(testing, values, point),
            "events": event_metrics(testing, values, test_sources, point),
            "hard_negatives": hard_negative_report(testing, values, point),
        }
    reference = next(row for row in training if len(row["bag"]))
    head.eval()
    gpu_score = head(reference["bag"], reference["aux"]).detach().cpu()
    cpu_head = copy.deepcopy(head).cpu().eval()
    cpu_score = cpu_head(reference["bag"].cpu(), reference["aux"].cpu()).detach()
    report["head_gpu_cpu_maximum_logit_difference"] = float(
        (gpu_score - cpu_score).abs().max()
    )
    write(args.output / "report.json", report)
    with zipfile.ZipFile(
        args.output / "mmaction_interaction_v2_pilot.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as package:
        for name in ("interaction.pt", "interaction.json", "report.json", "protocol.json"):
            package.write(args.output / name, name)
    print("FINISHED_V2", json.dumps(report), flush=True)


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
