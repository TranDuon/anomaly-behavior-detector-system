"""Matched 2x2 ROI/time pilot using unchanged labels and bounded GPU time."""

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from anomaly.fight_data import ubi_fights_items
from detection.detector import PersonDetector
from fight.model import VideoClassifier
from fight.vision import ClipConfig, DetectorConfig
from scripts.fight_ablation import (
    cached_rows,
    export_runtime,
    row_id,
    stable_train,
    threshold_report,
)
from scripts.fight_pilot import read_frames, score_rows, write_json


def variant_configs():
    baseline = ClipConfig(region_mode="tubes", trim_black_borders=True, version=3)
    return {
        "tubes_legacy": baseline,
        "pairs_legacy": replace(baseline, region_mode="pairs", version=4),
        "tubes_native": replace(baseline, frames=13, fps=5.0, version=4),
        "pairs_native": replace(
            baseline, frames=13, fps=5.0, region_mode="pairs", version=4
        ),
    }


def compatible_window(row, labels, configs):
    if row["fps"] <= 0 or row["start"] < 0:
        return False
    for config in configs:
        end = (
            int(np.rint(row["start"] + (config.frames - 1) * row["fps"] / config.fps))
            + 1
        )
        if end > len(labels):
            return False
        fraction = float(np.mean(labels[row["start"] : end]))
        if (row["label"] == 1 and fraction < 0.7) or (
            row["label"] == 0 and fraction != 0
        ):
            return False
    return True


def balanced_source_sample(rows, per_class=192):
    rng = random.Random(43)
    chosen = []
    for label in (0, 1):
        groups = defaultdict(list)
        for row in rows:
            if row["label"] == label:
                groups[row["video"]].append(row)
        if sum(map(len, groups.values())) < per_class:
            raise ValueError(
                "Insufficient training windows after shared time eligibility"
            )
        keys = sorted(groups)
        rng.shuffle(keys)
        for group in groups.values():
            rng.shuffle(group)
        selected = []
        while len(selected) < per_class:
            for key in keys:
                if groups[key]:
                    selected.append(groups[key].pop())
                if len(selected) == per_class:
                    break
        chosen.extend(selected)
    return chosen


def matched_manifest(source, dataset, per_class):
    original = json.loads((source / "cached_manifest.json").read_text())
    if original.get("official_test_used") is not False:
        raise ValueError("Only development source clips are allowed")
    selected_train = json.loads((source / "training_selection.json").read_text())[
        "rows"
    ]
    allowed = {row_id(row) for row in selected_train}
    items = {item.uid: item for item in ubi_fights_items(dataset)}
    eligible, excluded = [], []
    for row in original["rows"]:
        if row["split"] == "train" and row_id(row) not in allowed:
            continue
        item = items[row["video"]]
        if item.split != "train":
            raise ValueError("Official test cannot enter this experiment")
        if compatible_window(row, item.frame_labels, variant_configs().values()):
            eligible.append({**row, "path": str(item.path)})
        else:
            excluded.append(
                {"video": row["video"], "start": row["start"], "split": row["split"]}
            )
    train = balanced_source_sample(
        [row for row in eligible if row["split"] == "train"], per_class
    )
    validation = [row for row in eligible if row["split"] == "validation"]
    for key in ("video", "scene_group"):
        if {row[key] for row in train} & {row[key] for row in validation}:
            raise ValueError("Source/scene leakage across train and validation")
    if any(sum(row["label"] == label for row in validation) < 40 for label in (0, 1)):
        raise ValueError("Validation is too small after temporal eligibility checks")
    return {
        "rows": train + validation,
        "excluded_for_time_eligibility": excluded,
        "official_test_used": False,
        "labels_changed": False,
        "audit_judgments_used_for_training": False,
        "sampling": "Balanced round-robin across training source videos, same subset for all variants",
        "scope": "Repeated development comparison; not independent test or phone acceptance",
    }


def crop_gallery(rows, cache, config, output):
    ranked = []
    for row in rows:
        with np.load(cache / f"{row['cache']}.npz") as values:
            ranked.append((int(values["people"]), row))
    panels, seen = [], set()
    for people, row in sorted(ranked, key=lambda entry: entry[0], reverse=True):
        if people < 3 or row["video"] in seen:
            continue
        seen.add(row["video"])
        frames = read_frames(row, config)
        source = cv2.resize(frames[len(frames) // 2], (384, 216))
        panel = np.full(
            (246, 384 + config.size * config.max_regions, 3), 245, dtype=np.uint8
        )
        panel[30:246, :384] = source
        cv2.putText(
            panel,
            f"{people} detected people | source then model inputs",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        with np.load(cache / f"{row['cache']}.npz") as values:
            for index, clip in enumerate(values["clips"]):
                panel[
                    30 : 30 + config.size,
                    384 + index * config.size : 384 + (index + 1) * config.size,
                ] = cv2.cvtColor(clip[len(clip) // 2], cv2.COLOR_RGB2BGR)
        panels.append(panel)
        if len(panels) == 4:
            break
    if panels:
        cv2.imwrite(str(output / "crowd_crops.jpg"), np.concatenate(panels, axis=0))


def run(args):
    if not getattr(args, "acknowledge_unreviewed_labels", False):
        raise ValueError(
            "Original labels were flagged by review. Use the curated pilot, or explicitly acknowledge this is only an unreviewed-label diagnostic."
        )
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "comparison.json").exists():
        previous = json.loads((output / "comparison.json").read_text())
        if previous.get("status") == "completed":
            print(json.dumps(previous), flush=True)
            return
        raise ValueError(
            "Incomplete experiment exists; use a new output directory, no implicit retraining"
        )
    manifest = matched_manifest(args.source_run, args.dataset_root, args.per_class)
    write_json(output / "manifest.json", manifest)
    summary = {
        "status": "running",
        "variants": {},
        "official_test_used": False,
        "approved_for_demo": False,
        "labels_changed": False,
        "audit_judgments_used_for_training": False,
        "train_clips": sum(row["split"] == "train" for row in manifest["rows"]),
        "validation_clips": sum(
            row["split"] == "validation" for row in manifest["rows"]
        ),
        "removed_for_time_eligibility": len(manifest["excluded_for_time_eligibility"]),
        "scope": manifest["scope"],
        "decision": "inconclusive",
    }
    write_json(output / "comparison.json", summary)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = VideoClassifier("x3d_s", pretrained=True).cuda().eval()
    initial = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    detector_config = DetectorConfig()
    detector = PersonDetector(**detector_config.to_dict(), device="cuda")
    ready_by_variant = {}
    try:
        for name, config in variant_configs().items():
            destination = output / name
            destination.mkdir(exist_ok=True)
            cache = args.original_cache if name == "tubes_legacy" else args.cache_dir
            cache.mkdir(parents=True, exist_ok=True)
            print(f"VARIANT_CACHE {name}", flush=True)
            ready = cached_rows(
                manifest["rows"],
                model,
                detector,
                config,
                detector_config,
                cache,
                "cuda",
                min(deadline - 480, started + args.budget_minutes * 36),
            )
            ready_by_variant[name] = (ready, cache)
            write_json(
                destination / "cached_manifest.json",
                {**manifest, "rows": ready, "clip": config.to_dict()},
            )
            crop_gallery(
                [row for row in ready if row["split"] == "train"],
                cache,
                config,
                destination,
            )
        for name, config in variant_configs().items():
            if time.monotonic() >= deadline - 120:
                raise TimeoutError("No budget for another matched training variant")
            print(f"VARIANT_TRAIN {name}", flush=True)
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            model = model.cuda()
            model.load_state_dict(initial)
            ready, cache = ready_by_variant[name]
            train = [row for row in ready if row["split"] == "train"]
            validation = [row for row in ready if row["split"] == "validation"]
            destination = output / name
            history = stable_train(
                model,
                train,
                validation,
                cache,
                "cuda",
                deadline - 100,
                destination,
                epochs=(2, 2),
                camera_augmentation=True,
            )
            scores = score_rows(model, validation, cache, "cuda", False)
            report = {
                **threshold_report(validation, scores),
                "official_test_used": False,
                "approved_for_demo": False,
                "scope": manifest["scope"],
                "completed_epochs": len(history),
                "train_clips": len(train),
                "validation_clips": len(validation),
                "decision": "experimental_not_approved",
            }
            write_json(destination / "report.json", report)
            write_json(
                destination / "predictions.json",
                [
                    {
                        "video": row["video"],
                        "start_seconds": row["start"] / row["fps"],
                        "label": row["label"],
                        "score": float(score),
                    }
                    for row, score in zip(validation, scores)
                ],
            )
            export_runtime(model, config, detector_config, report, destination)
            summary["variants"][name] = report
            write_json(output / "comparison.json", summary)
        summary["status"] = "completed"
        summary["best_by_development_ap"] = max(
            summary["variants"],
            key=lambda name: summary["variants"][name]["validation"]["ap"],
        )
        baseline = summary["variants"]["tubes_legacy"]
        candidate = summary["variants"][summary["best_by_development_ap"]]
        summary["decision"] = "stop_and_review"
        if (
            candidate["validation"]["ap"] >= baseline["validation"]["ap"] + 0.03
            and (candidate["validation"]["in_video_auc"] or 0)
            >= (baseline["validation"]["in_video_auc"] or 0) + 0.05
            and candidate["best_recall_at_precision75"]
            >= baseline["best_recall_at_precision75"] + 0.05
        ):
            summary["decision"] = "promising_for_review_only"
    except KeyboardInterrupt:
        summary["status"] = "stopped_before_completion"
        summary["reason"] = "Interrupted; no automatic retry or runtime deletion"
        raise
    except Exception as error:
        summary["status"] = "incomplete"
        summary["reason"] = str(error)
        raise
    finally:
        summary["seconds"] = time.monotonic() - started
        write_json(output / "comparison.json", summary)
        print("MATCHED_RESULT " + json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-cache", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=192)
    parser.add_argument("--budget-minutes", type=float, default=25)
    parser.add_argument("--acknowledge-unreviewed-labels", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.per_class < 40 or args.budget_minutes < 10:
        parser.error("CUDA, >=40 clips/class and >=10 min budget required")
    run(args)


if __name__ == "__main__":
    main()
