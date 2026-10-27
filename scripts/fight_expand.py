"""Expanded UBI development pilot; no NTU dependency or automatic demo approval."""

import argparse
import json
import random
import re
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

from anomaly.fight_data import parse_ubi_annotation, ubi_fights_items
from detection.detector import PersonDetector
from fight.model import VideoClassifier
from fight.vision import ClipConfig, DetectorConfig
from scripts.fight_ablation import (
    audit_errors,
    cached_rows,
    export_runtime,
    row_id,
    stable_train,
    threshold_report,
)
from scripts.fight_pilot import metrics, scene_hash, score_rows, write_json


def expanded_config():
    return ClipConfig(region_mode="tubes", trim_black_borders=True, version=3)


def archive_selection(archive, original, videos=200, excluded=()):
    names = archive.namelist()
    test_name = next(
        name
        for name in names
        if name.endswith("/test_videos.csv") or name == "test_videos.csv"
    )
    official_test = {
        line.strip()
        for line in archive.read(test_name).decode().splitlines()
        if line.strip()
    }
    video_names = {
        Path(name).stem: name
        for name in names
        if Path(name).suffix.lower() in {".mp4", ".avi"} and "/videos/" in "/" + name
    }
    annotation_names = {
        Path(name).stem: name
        for name in names
        if "/annotation/" in "/" + name and name.endswith(".csv")
    }
    labels = {}
    for key in sorted(video_names.keys() & annotation_names.keys()):
        if key in official_test or "ubi/" + key in excluded:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError("Unsafe archive video key")
        labels[key] = bool(
            parse_ubi_annotation(archive.read(annotation_names[key]).decode()).any()
        )
    anchors = {row["key"] for row in original["rows"]}
    if not anchors <= labels.keys():
        raise ValueError(
            "Historical development videos missing or excluded; inspect selection"
        )
    rng = random.Random(42)
    selected = set(anchors)
    for positive in (False, True):
        pool = [
            key for key in labels if labels[key] == positive and key not in selected
        ]
        rng.shuffle(pool)
        needed = max(0, videos // 2 - sum(labels[key] == positive for key in selected))
        if len(pool) < needed:
            raise ValueError(
                "Not enough official-training source videos for requested expansion"
            )
        selected.update(pool[:needed])
    return sorted(selected), video_names, annotation_names, test_name


def grouped_partition(items, original, fingerprints, validation_per_class=20):
    parents = {item.uid: item.uid for item in items}

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def merge(first, second):
        parents[root(second)] = root(first)

    original_groups = {}
    old_splits = {}
    for row in original["rows"]:
        old_splits[row["video"]] = row["split"]
        original_groups.setdefault(row["scene_group"], set()).add(row["video"])
    for group in original_groups.values():
        members = sorted(group & parents.keys())
        for member in members[1:]:
            merge(members[0], member)
    for first_index, first in enumerate(items):
        for second in items[first_index + 1 :]:
            if (
                int(fingerprints[first.uid], 16) ^ int(fingerprints[second.uid], 16)
            ).bit_count() <= 6:
                merge(first.uid, second.uid)
    groups = {}
    for item in items:
        groups.setdefault(root(item.uid), []).append(item)
    splits, quarantined = {}, []
    for group, members in groups.items():
        known = {old_splits[item.uid] for item in members if item.uid in old_splits}
        if len(known) > 1:
            quarantined.extend(item.uid for item in members)
            splits[group] = "quarantine"
        elif known:
            splits[group] = known.pop()
    rng = random.Random(43)
    candidates = [group for group in sorted(groups) if group not in splits]
    rng.shuffle(candidates)
    for positive in (True, False):
        count = sum(
            item.has_positive == positive
            for group, members in groups.items()
            if splits.get(group) == "validation"
            for item in members
        )
        for group in candidates:
            if group in splits or count >= validation_per_class:
                continue
            matching = sum(item.has_positive == positive for item in groups[group])
            if matching:
                splits[group] = "validation"
                count += matching
    return {
        item.uid: (splits.get(root(item.uid), "train"), root(item.uid))
        for item in items
    }, quarantined


def sample_windows(item, split, group, config, clips_per_video=24):
    capture = cv2.VideoCapture(str(item.path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or abs(len(item.frame_labels) - count) > 2:
        raise ValueError(f"Timing/annotation mismatch: {item.uid}")
    labels = item.labels_for(count)
    span = round((config.frames - 1) / config.fps * fps) + 1
    starts = np.arange(0, max(0, count - span + 1), max(1, round(fps)))
    prefix = np.r_[0, np.cumsum(labels)]
    fractions = (prefix[starts + span] - prefix[starts]) / span
    rng = np.random.default_rng(42 + sum(item.key.encode()))
    rows, occupied = [], []
    for label in (1, 0):
        candidates = starts[fractions >= 0.7] if label else starts[fractions == 0]
        rng.shuffle(candidates)
        quota = clips_per_video // 2
        chosen = 0
        for start in candidates:
            if any(abs(int(start) - old) < span for old in occupied):
                continue
            rows.append(
                {
                    "video": item.uid,
                    "key": item.key,
                    "path": str(item.path),
                    "start": int(start),
                    "fps": fps,
                    "label": label,
                    "split": split,
                    "scene_group": group,
                }
            )
            occupied.append(int(start))
            chosen += 1
            if chosen >= quota:
                break
    return rows


def prepare(args, output):
    original = json.loads((Path(args.source_run) / "manifest.json").read_text())
    if original.get("official_test_used") is not False:
        raise ValueError("Original manifest must exclude official test")
    dataset = Path(args.dataset_root)
    excluded = set()
    if args.exclude and Path(args.exclude).exists():
        excluded = set(
            json.loads(Path(args.exclude).read_text())["exclude_from_training"]
        )
    with zipfile.ZipFile(args.archive) as archive:
        selected, videos, annotations, test_name = archive_selection(
            archive, original, args.videos, excluded
        )
        write_json(output / "selected_videos.json", selected)
        for index, key in enumerate(selected, 1):
            for member in (videos[key], annotations[key]):
                marker = "videos" if member == videos[key] else "annotation"
                folder = (
                    "fight" if marker == "videos" and key.startswith("F_") else "normal"
                )
                destination = (
                    dataset / marker / folder / Path(member).name
                    if marker == "videos"
                    else dataset / marker / Path(member).name
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                if (
                    not destination.exists()
                    or destination.stat().st_size != archive.getinfo(member).file_size
                ):
                    temporary = destination.with_suffix(destination.suffix + ".partial")
                    with archive.open(member) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    temporary.replace(destination)
            if index % 16 == 0:
                print(f"EXTRACT {index}/{len(selected)}", flush=True)
        (dataset / "test_videos.csv").write_bytes(archive.read(test_name))
    items = [
        item
        for item in ubi_fights_items(dataset)
        if item.key in selected and item.split == "train"
    ]
    fingerprints = {item.uid: scene_hash(item.path) for item in items}
    partitions, quarantined = grouped_partition(items, original, fingerprints)
    rows = [
        row
        for item in items
        if partitions[item.uid][0] != "quarantine"
        for row in sample_windows(
            item, *partitions[item.uid], expanded_config(), args.clips_per_video
        )
    ]
    manifest = {
        "rows": rows,
        "clip": expanded_config().to_dict(),
        "official_test_used": False,
        "seed": 42,
        "quarantined_videos": quarantined,
        "split_policy": "Original validation never enters train; old train groups never become new validation; transitive near-duplicate scene groups quarantined on conflicts.",
    }
    signature = {
        "videos": args.videos,
        "clips_per_video": args.clips_per_video,
        "clip": expanded_config().to_dict(),
    }
    manifest["run_settings"] = signature
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing manifest differs; use a new output directory")
    write_json(path, manifest)
    summary = {
        split: {
            "videos": len({row["video"] for row in rows if row["split"] == split}),
            "clips": sum(row["split"] == split for row in rows),
            "positive": sum(
                row["split"] == split and row["label"] == 1 for row in rows
            ),
        }
        for split in ("train", "validation")
    }
    write_json(
        output / "data_summary.json",
        {
            "counts": summary,
            "quarantined_videos": quarantined,
            "target_context": ["crowded_classroom", "crowded_hallway", "outdoor_crowd"],
            "context_note": "UBI source footage, not verified recordings of the user's demo environments.",
        },
    )
    print("DATA_SUMMARY " + json.dumps(summary), flush=True)


def train(args, output):
    manifest = json.loads((output / "manifest.json").read_text())
    config = ClipConfig(**manifest["clip"])
    if config.to_dict() != expanded_config().to_dict():
        raise ValueError("Expanded preprocessing configuration differs")
    if (output / "report.json").exists():
        print((output / "report.json").read_text(), flush=True)
        return
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    model = VideoClassifier("x3d_s", pretrained=True).cuda().eval()
    detector_config = DetectorConfig()
    detector = PersonDetector(**detector_config.to_dict(), device="cuda")
    try:
        ready = cached_rows(
            manifest["rows"],
            model,
            detector,
            config,
            detector_config,
            cache,
            "cuda",
            min(deadline - 300, started + args.budget_minutes * 36),
        )
        write_json(output / "cached_manifest.json", {**manifest, "rows": ready})
        feature_values, offsets = [], [0]
        for row in ready:
            with np.load(cache / f"{row['cache']}.npz") as values:
                feature_values.append(values["features"])
                offsets.append(offsets[-1] + len(values["features"]))
        np.savez_compressed(
            output / "pretrained_features.npz",
            values=np.concatenate(feature_values),
            offsets=np.array(offsets),
            cache_keys=np.array([row["cache"] for row in ready]),
        )
        train_rows = [row for row in ready if row["split"] == "train"]
        validation = [row for row in ready if row["split"] == "validation"]
        positive, human_normal, background = [], [], []
        for row in train_rows:
            with np.load(cache / f"{row['cache']}.npz") as values:
                people = int(values["people"])
            if row["label"]:
                positive.append(row)
            elif people:
                human_normal.append(row)
            else:
                background.append(row)
        random.shuffle(background)
        background_limit = int((len(positive) + len(human_normal)) * 0.05 / 0.95)
        train_rows = positive + human_normal + background[:background_limit]
        write_json(
            output / "training_selection.json",
            {
                "positive_retained_regardless_of_detector": len(positive),
                "human_normal": len(human_normal),
                "background_normal": min(len(background), background_limit),
                "rows": train_rows,
            },
        )
        for split_rows in (train_rows, validation):
            if any(
                sum(row["label"] == label for row in split_rows) < 40
                for label in (0, 1)
            ):
                raise ValueError(
                    "Too few clips per class for expanded development evaluation"
                )
        baseline_path = Path(args.baseline_model)
        baseline_metadata = json.loads(
            baseline_path.with_name("fight_runtime.json").read_text()
        )
        baseline_config = ClipConfig(**baseline_metadata["clip"])
        baseline_cache = cache / "baseline"
        baseline_cache.mkdir(exist_ok=True)
        baseline_rows = cached_rows(
            validation,
            model,
            detector,
            baseline_config,
            detector_config,
            baseline_cache,
            "cuda",
            deadline - 240,
        )
        baseline = torch.jit.load(str(baseline_path), map_location="cuda").eval()
        baseline_scores = score_rows(
            baseline, baseline_rows, baseline_cache, "cuda", False
        )
        baseline_report = threshold_report(validation, baseline_scores)
        write_json(output / "baseline.json", baseline_report)
        del baseline
        torch.cuda.empty_cache()
        budget_exhausted = False
        try:
            stable_train(
                model,
                train_rows,
                validation,
                cache,
                "cuda",
                deadline - 120,
                output,
                epochs=(8, 6),
                camera_augmentation=True,
            )
        except TimeoutError:
            if not (output / "best.pt").exists():
                raise
            budget_exhausted = True
            checkpoint = torch.load(
                output / "best.pt", map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint["state_dict"])
        history = json.loads((output / "history.json").read_text())
        scores = score_rows(model, validation, cache, "cuda", False)
        report = {
            **threshold_report(validation, scores),
            "baseline_on_same_validation": baseline_report,
            "official_test_used": False,
            "approved_for_demo": False,
            "scope": "Expanded UBI development validation; public data and simulated mild camera motion, not a real-phone acceptance test.",
            "train_clips": len(train_rows),
            "validation_clips": len(validation),
            "validation_videos": len({row["video"] for row in validation}),
            "target_context": ["crowded_classroom", "crowded_hallway", "outdoor_crowd"],
            "decision": "stop_and_review",
            "training_budget_exhausted": budget_exhausted,
            "completed_epochs": {
                stage: sum(row["stage"] == stage for row in history)
                for stage in ("probe", "finetune")
            },
        }
        result = report["validation"]
        if (
            report["best_recall_at_precision75"] >= 0.6
            and (result["in_video_auc"] or 0) >= 0.65
            and result["mixed_label_videos"] >= 3
            and result["ap"] > baseline_report["validation"]["ap"]
        ):
            report["decision"] = "promising"
        predictions = [
            {
                "video": row["video"],
                "start_seconds": row["start"] / row["fps"],
                "label": row["label"],
                "score": float(score),
            }
            for row, score in zip(validation, scores)
        ]
        write_json(output / "predictions.json", predictions)
        write_json(output / "evaluation_report.json", report)
        export_runtime(model, config, detector_config, report, output)
        report["seconds"] = time.monotonic() - started
        write_json(output / "report.json", report)
        print("EXPANDED_RESULT " + json.dumps(report), flush=True)
    except (TimeoutError, ValueError) as error:
        write_json(
            output / "execution_failure.json",
            {
                "decision": "inconclusive",
                "reason": str(error),
                "seconds": time.monotonic() - started,
                "approved_for_demo": False,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "audit"))
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive")
    parser.add_argument("--exclude")
    parser.add_argument("--videos", type=int, default=200)
    parser.add_argument("--clips-per-video", type=int, default=24)
    parser.add_argument("--cache-dir")
    parser.add_argument("--baseline-model")
    parser.add_argument("--budget-minutes", type=float, default=45)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (
        args.videos < 48
        or args.videos % 2
        or args.clips_per_video < 4
        or args.budget_minutes <= 5
    ):
        parser.error("Use an even video count >=48, >=4 clips/video and budget >5 min")
    if args.phase == "prepare":
        if not args.archive:
            parser.error("prepare requires --archive")
        prepare(args, output)
    elif args.phase == "train":
        if (
            not torch.cuda.is_available()
            or not args.cache_dir
            or not args.baseline_model
        ):
            parser.error("train requires CUDA, --cache-dir and --baseline-model")
        train(args, output)
    else:
        if not args.cache_dir:
            parser.error("audit requires --cache-dir")
        manifest = json.loads((output / "cached_manifest.json").read_text())
        rows = [row for row in manifest["rows"] if row["split"] == "validation"]
        predictions = json.loads((output / "predictions.json").read_text())
        if len(rows) != len(predictions) or any(
            row["video"] != prediction["video"]
            or abs(row["start"] / row["fps"] - prediction["start_seconds"]) > 1e-6
            for row, prediction in zip(rows, predictions)
        ):
            raise ValueError("Prediction order differs from validation manifest")
        report = json.loads((output / "report.json").read_text())
        audit_errors(
            rows,
            np.array([prediction["score"] for prediction in predictions]),
            report["threshold"],
            Path(args.cache_dir),
            output,
            ClipConfig(**manifest["clip"]),
        )


if __name__ == "__main__":
    main()
