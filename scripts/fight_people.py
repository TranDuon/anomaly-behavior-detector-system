"""People-centred development cohort with label-blind, auditable filtering."""

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from detection.detector import PersonDetector
from fight.model import VideoClassifier
from fight.vision import ClipConfig, DetectorConfig
from scripts.fight_ablation import (
    audit_errors,
    export_runtime,
    fixed_manifest,
    row_id,
    stable_train,
    threshold_report,
)
from scripts.fight_pilot import metrics, read_frames, score_rows, write_json


@dataclass(frozen=True)
class PeoplePolicy:
    min_people: int = 2
    min_height_ratio: float = 0.18
    min_area_ratio: float = 0.008
    support_frames: int = 2
    max_single_control_fraction: float = 0.15


def frame_people(boxes, width, height, policy):
    salient = []
    for box in boxes:
        coordinates = np.asarray(box, dtype=float)
        if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
            continue
        left, top, right, bottom = coordinates
        left, right = np.clip([left, right], 0, width)
        top, bottom = np.clip([top, bottom], 0, height)
        area = max(0, right - left) * max(0, bottom - top)
        if (
            bottom - top >= height * policy.min_height_ratio
            and area >= width * height * policy.min_area_ratio
        ):
            salient.append([float(left), float(top), float(right), float(bottom)])
    return salient


def decide_domain(salient_counts, raw_counts, policy):
    if (
        sum(count >= policy.min_people for count in salient_counts)
        >= policy.support_frames
    ):
        return "people_interaction"
    if (
        max(salient_counts, default=0) <= 1
        and sum(count == 1 for count in salient_counts) >= policy.support_frames
    ):
        return "single_person_control"
    if max(raw_counts, default=0) == 0:
        return "no_detected_people"
    return "small_or_unstable_people"


def select_people_rows(rows, audits, policy):
    main = [
        row for row in rows if audits[row_id(row)]["domain"] == "people_interaction"
    ]
    train = [row for row in main if row["split"] == "train"]
    validation = [row for row in main if row["split"] == "validation"]
    controls = [
        row
        for row in rows
        if row["split"] == "train"
        and row["label"] == 0
        and audits[row_id(row)]["domain"] == "single_person_control"
    ]
    random.Random(42).shuffle(controls)
    limit = int(
        len(train)
        * policy.max_single_control_fraction
        / (1 - policy.max_single_control_fraction)
    )
    return train + controls[:limit], validation


def coverage(rows, audits):
    result = {}
    for split in ("train", "validation"):
        result[split] = {}
        for label in (0, 1):
            selected = [
                row for row in rows if row["split"] == split and row["label"] == label
            ]
            counts = {}
            for row in selected:
                domain = audits[row_id(row)]["domain"]
                counts[domain] = counts.get(domain, 0) + 1
            result[split][str(label)] = {
                "source_clips": len(selected),
                "domains": counts,
            }
    return result


def preview_frames(frames, detections, path):
    panels = []
    for frame, boxes in zip(frames, detections):
        image = frame.copy()
        for left, top, right, bottom in boxes:
            cv2.rectangle(
                image,
                (round(left), round(top)),
                (round(right), round(bottom)),
                (0, 220, 0),
                2,
            )
        height, width = image.shape[:2]
        scale = min(320 / width, 220 / height)
        resized = cv2.resize(image, (round(width * scale), round(height * scale)))
        canvas = np.full((220, 320, 3), 114, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        panels.append(canvas)
    cv2.imwrite(str(path), np.concatenate(panels, axis=1))


def audit_domain(rows, config, detector_config, policy, output, deadline):
    destination = output / "domain_audit"
    destination.mkdir(parents=True, exist_ok=True)
    previews = output / "previews"
    previews.mkdir(exist_ok=True)
    detector = PersonDetector(**detector_config.to_dict(), device="cuda")
    audits, gallery, counts, seen = {}, [], {}, set()
    for index, row in enumerate(rows):
        if time.monotonic() >= deadline:
            raise TimeoutError("People audit budget exhausted")
        fingerprint = {
            "video": row["video"],
            "start": row["start"],
            "fps": row["fps"],
            "clip": config.to_dict(),
            "detector": detector_config.to_dict(),
            "policy": asdict(policy),
        }
        key = hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True).encode()
        ).hexdigest()[:24]
        path = destination / f"{key}.json"
        frames = None
        if path.exists():
            result = json.loads(path.read_text())
        else:
            all_frames = read_frames(row, config)
            frames = [
                all_frames[position]
                for position in (0, len(all_frames) // 2, len(all_frames) - 1)
            ]
            boxes = [
                [detection.bbox_xyxy for detection in detector.detect(frame)]
                for frame in frames
            ]
            salient = [
                frame_people(values, frame.shape[1], frame.shape[0], policy)
                for frame, values in zip(frames, boxes)
            ]
            raw_counts = [len(values) for values in boxes]
            salient_counts = [len(values) for values in salient]
            result = {
                "domain": decide_domain(salient_counts, raw_counts, policy),
                "raw_people_counts": raw_counts,
                "salient_people_counts": salient_counts,
                "salient_boxes": salient,
            }
            write_json(path, result)
        audits[row_id(row)] = {
            **result,
            "video": row["video"],
            "start": row["start"],
            "start_seconds": row["start"] / row["fps"],
            "label": row["label"],
            "split": row["split"],
        }
        category = (result["domain"], row["label"])
        identity = (row["video"], category)
        if counts.get(category, 0) < 4 and identity not in seen:
            if frames is None:
                all_frames = read_frames(row, config)
                frames = [
                    all_frames[position]
                    for position in (0, len(all_frames) // 2, len(all_frames) - 1)
                ]
            filename = f"{result['domain']}_label{row['label']}_{row['key']}_{row['start']}.jpg"
            preview_frames(frames, result["salient_boxes"], previews / filename)
            gallery.append({"image": "previews/" + filename, **audits[row_id(row)]})
            counts[category] = counts.get(category, 0) + 1
            seen.add(identity)
        if (index + 1) % 32 == 0:
            print(f"PEOPLE_AUDIT {index + 1}/{len(rows)}", flush=True)
    write_json(output / "domain_audit.json", list(audits.values()))
    write_json(output / "gallery.json", gallery)
    return audits


def original_predictions(path, rows):
    predictions = json.loads(Path(path).read_text())
    lookup = {
        (row["video"], round(row["start_seconds"], 6), row["label"]): row["score"]
        for row in predictions
    }
    return np.array(
        [
            lookup[(row["video"], round(row["start"] / row["fps"], 6), row["label"])]
            for row in rows
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--ablation-run", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-minutes", type=float, default=12)
    args = parser.parse_args()
    if args.budget_minutes <= 0 or not torch.cuda.is_available():
        raise ValueError("Positive budget and a Colab GPU are required")
    source, previous, cache, output = map(
        Path, (args.source_run, args.ablation_run, args.cache_dir, args.output_dir)
    )
    output.mkdir(parents=True, exist_ok=True)
    config, detector_config, policy = ClipConfig(), DetectorConfig(), PeoplePolicy()
    manifest = fixed_manifest(source / "manifest.json", args.dataset_root)
    signature = {
        "policy": asdict(policy),
        "clip": config.to_dict(),
        "detector": detector_config.to_dict(),
        "source_rows": [
            {
                key: row[key]
                for key in ("video", "start", "label", "split", "scene_group")
            }
            for row in manifest["rows"]
        ],
    }
    signature_path = output / "run_config.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Configuration changed; choose a new output directory")
    write_json(signature_path, signature)
    if (output / "report.json").exists():
        print((output / "report.json").read_text(), flush=True)
        return
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    try:
        rows = manifest["rows"]
        audits = audit_domain(
            rows, config, detector_config, policy, output, deadline - 60
        )
        cache_manifest = json.loads(
            (previous / "interaction" / "manifest.json").read_text()
        )
        if ClipConfig(**cache_manifest["clip"]).to_dict() != config.to_dict():
            raise ValueError("Interaction cache configuration differs")
        cache_ids = {row_id(row): row["cache"] for row in cache_manifest["rows"]}
        rows = [{**row, "cache": cache_ids[row_id(row)]} for row in rows]
        if not all((cache / f"{row['cache']}.npz").exists() for row in rows):
            raise FileNotFoundError("V2 interaction clip caches are required")
        train, validation = select_people_rows(rows, audits, policy)
        all_validation = [row for row in rows if row["split"] == "validation"]
        report = {
            "scope": "People-centred development cohort; changed validation scope, not comparable directly with old headline metrics.",
            "policy": asdict(policy),
            "coverage": coverage(rows, audits),
            "train_clips": len(train),
            "validation_clips": len(validation),
            "original_validation_clips": len(all_validation),
            "official_test_used": False,
            "approved_for_demo": False,
            "decision": "inconclusive",
            "reason": "Audit completed; training not completed yet.",
        }
        write_json(
            output / "cohort_manifest.json", {**manifest, "rows": train + validation}
        )
        write_json(output / "cohort_report.json", report)
        print("COHORT " + json.dumps(report), flush=True)
        if any(
            sum(row["label"] == label for row in train) < 12 for label in (0, 1)
        ) or any(
            sum(row["label"] == label for row in validation) < 8 for label in (0, 1)
        ):
            report["reason"] = (
                "Too few clips per class after label-blind people filtering; expand training-only source sampling rather than fabricate success."
            )
            write_json(output / "report.json", report)
            print(json.dumps(report), flush=True)
            return
        baselines = {}
        paths = [
            ("baseline_v1", source),
            ("interaction_v2", previous / "interaction"),
            ("full_hardneg_v2", previous / "full_hardneg"),
        ]
        for name, path in paths:
            metadata = json.loads((path / "report.json").read_text())
            target_scores = original_predictions(path / "predictions.json", validation)
            full_scores = original_predictions(
                path / "predictions.json", all_validation
            )
            baselines[name] = {
                "people_scope": threshold_report(validation, target_scores),
                "people_at_original_threshold": metrics(
                    validation, target_scores, metadata["threshold"]
                ),
                "original_scope_at_original_threshold": metrics(
                    all_validation, full_scores, metadata["threshold"]
                ),
            }
        report["baselines_on_same_cohort"] = baselines
        write_json(output / "cohort_report.json", report)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = VideoClassifier("x3d_s", pretrained=True).cuda().eval()
        stable_train(
            model,
            train,
            validation,
            cache,
            "cuda",
            deadline - 60,
            output,
            epochs=(10, 6),
        )
        target_scores = score_rows(model, validation, cache, "cuda", False)
        full_scores = score_rows(model, all_validation, cache, "cuda", False)
        target = threshold_report(validation, target_scores)
        report.update(target)
        report["original_scope_at_people_threshold"] = metrics(
            all_validation, full_scores, target["threshold"]
        )
        report["decision"] = (
            "promising"
            if target["best_recall_at_precision75"] >= 0.6
            and (target["validation"]["in_video_auc"] or 0) >= 0.65
            and target["validation"]["mixed_label_videos"] >= 3
            else "stop_and_review"
        )
        report["reason"] = (
            "Independent test and full-video event evaluation still required; no automatic demo promotion."
        )
        report["seconds"] = time.monotonic() - started
        write_json(output / "evaluation_report.json", report)
        predictions = [
            {
                "video": row["video"],
                "start_seconds": row["start"] / row["fps"],
                "label": row["label"],
                "score": float(score),
                "people_scope": audits[row_id(row)]["domain"] == "people_interaction",
            }
            for row, score in zip(all_validation, full_scores)
        ]
        write_json(output / "predictions.json", predictions)
        audit_errors(
            validation, target_scores, target["threshold"], cache, output, config
        )
        export_runtime(model, config, detector_config, report, output)
        report["seconds"] = time.monotonic() - started
        write_json(output / "report.json", report)
        print("PEOPLE_RESULT " + json.dumps(report), flush=True)
    except TimeoutError as error:
        write_json(
            output / "report.json",
            {
                "decision": "inconclusive",
                "reason": str(error),
                "approved_for_demo": False,
                "seconds": time.monotonic() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
