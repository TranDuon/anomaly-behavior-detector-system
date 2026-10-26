"""Fixed-split, bounded development experiments; never approves a demo model."""

import argparse
import copy
import hashlib
import json
import random
import subprocess
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from anomaly.fight_data import ubi_fights_items
from detection.detector import PersonDetector
from fight.model import VideoClassifier, prepare_export
from fight.vision import (
    ClipConfig,
    DetectorConfig,
    clip_regions,
    clip_tensor,
    prepare_clip,
)
from scripts.fight_pilot import metrics, read_frames, score_rows, write_json


def row_id(row):
    return (row["video"], row["start"], row["label"])


def fixed_manifest(source, dataset_root):
    manifest = json.loads(Path(source).read_text())
    if manifest.get("official_test_used") is not False:
        raise ValueError("Source must explicitly exclude official test")
    items = {item.uid: item for item in ubi_fights_items(Path(dataset_root))}
    rows = copy.deepcopy(manifest["rows"])
    partitions = {}
    for row in rows:
        item = items[row["video"]]
        if item.split != "train":
            raise ValueError("Official test found in development manifest")
        row["path"] = str(item.path)
        partitions.setdefault(row["scene_group"], set()).add(row["split"])
    if any(len(values) != 1 for values in partitions.values()):
        raise ValueError("Scene group crosses partitions")
    if len({row_id(row) for row in rows}) != len(rows):
        raise ValueError("Duplicate source rows")
    return {**manifest, "rows": rows}


def threshold_report(rows, scores):
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    values = [
        (metrics(rows, scores, float(value)), float(value)) for value in candidates
    ]
    feasible = [entry for entry in values if entry[0]["recall"] >= 0.7]
    result, threshold = max(
        feasible or values,
        key=lambda entry: (entry[0]["f1"], entry[0]["precision"], entry[1]),
    )
    precise = [entry for entry in values if entry[0]["precision"] >= 0.75]
    sensitive = [entry for entry in values if entry[0]["recall"] >= 0.6]
    return {
        "validation": result,
        "threshold": threshold,
        "at_half": metrics(rows, scores, 0.5),
        "best_recall_at_precision75": max(
            (entry[0]["recall"] for entry in precise), default=0
        ),
        "best_precision_at_recall60": max(
            (entry[0]["precision"] for entry in sensitive), default=0
        ),
        "threshold_note": "Development validation only; thresholds are not independent test results.",
    }


def extra_negative_candidates(rows, dataset_root, config, maximum=96):
    items = {item.uid: item for item in ubi_fights_items(Path(dataset_root))}
    train_videos = sorted({row["video"] for row in rows if row["split"] == "train"})
    validation_videos = {row["video"] for row in rows if row["split"] == "validation"}
    candidates = []
    rng = np.random.default_rng(43)
    for video in train_videos:
        if video in validation_videos:
            raise ValueError("Video crosses partitions")
        item = items[video]
        if item.split != "train":
            raise ValueError("Hard-negative mining cannot use official test")
        reference = next(row for row in rows if row["video"] == video)
        capture = cv2.VideoCapture(str(item.path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        fps = reference["fps"]
        span = round((config.frames - 1) / config.fps * fps) + 1
        labels = item.labels_for(count)
        occupied = [row["start"] for row in rows if row["video"] == video]
        starts = list(range(0, max(0, count - span + 1), max(span, round(fps * 3))))
        rng.shuffle(starts)
        selected = 0
        for start in starts:
            if labels[start : start + span].any() or any(
                abs(start - old) < span for old in occupied
            ):
                continue
            candidates.append({**reference, "start": start, "label": 0, "mined": True})
            occupied.append(start)
            selected += 1
            if selected >= 4:
                break
    rng.shuffle(candidates)
    return candidates[:maximum]


def cached_rows(
    rows, model, detector, config, detector_config, cache, device, deadline
):
    ready = []
    for index, original in enumerate(rows):
        if time.monotonic() >= deadline:
            raise TimeoutError("Cache budget exhausted")
        identity = {
            "id": row_id(original),
            "fps": original["fps"],
            "clip": config.to_dict(),
            "detector": detector_config.to_dict(),
            "backbone": "x3d_s_kinetics400",
            "cache_version": 2,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
            :24
        ]
        destination = cache / f"{key}.npz"
        row = {**original, "cache": key}
        if not destination.exists():
            frames = read_frames(row, config)
            boxes, people = [], 0
            for frame_index in (0, len(frames) // 2, len(frames) - 1):
                detections = detector.detect(frames[frame_index])
                boxes.extend(detection.bbox_xyxy for detection in detections)
                people = max(people, len(detections))
            height, width = frames[0].shape[:2]
            regions = clip_regions(boxes, width, height, config)
            clips = np.stack(
                [prepare_clip(frames, region, config.size) for region in regions]
            )
            with torch.inference_mode():
                features = (
                    model.features(clip_tensor(clips, device))
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
            temporary = destination.with_suffix(".partial.npz")
            np.savez_compressed(
                temporary,
                clips=clips,
                features=features,
                regions=np.array(regions),
                people=people,
            )
            temporary.replace(destination)
        ready.append(row)
        if (index + 1) % 32 == 0:
            print(f"CACHE {config.region_mode} {index + 1}/{len(rows)}", flush=True)
    return ready


def stable_train(
    model, train, validation, cache, device, deadline, output, epochs=(10, 6)
):
    history, best, best_state = [], -1.0, None
    counts = np.bincount([row["label"] for row in train], minlength=2)
    weights = len(train) / (2 * np.maximum(counts, 1))
    for frozen, total_epochs in zip((True, False), epochs):
        if best_state is not None:
            model.load_state_dict(best_state)
        model.set_stage("probe" if frozen else "finetune")
        groups = [{"params": model.head.parameters(), "lr": 1e-3 if frozen else 1e-4}]
        if not frozen:
            groups.append(
                {
                    "params": [
                        parameter
                        for parameter in model.backbone.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": 1e-5,
                }
            )
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs)
        stale = 0
        for epoch in range(total_epochs):
            order = list(train)
            random.shuffle(order)
            model.train()
            model.backbone.eval()
            losses = []
            for batch_start in range(0, len(order), 8):
                batch = order[batch_start : batch_start + 8]
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Training budget exhausted; completed checkpoints retained"
                    )
                optimizer.zero_grad()
                for row in batch:
                    with np.load(cache / f"{row['cache']}.npz") as data:
                        if frozen:
                            features = torch.from_numpy(
                                data["features"].astype(np.float32)
                            ).to(device)
                            logits = model.head(features).flatten()
                        else:
                            clips = data["clips"]
                            if random.random() < 0.5:
                                clips = clips[:, :, :, ::-1].copy()
                            logits = model(clip_tensor(clips, device))
                    target = torch.tensor(float(row["label"]), device=device)
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits.max(), target
                    ) * float(weights[row["label"]])
                    (loss / len(batch)).backward()
                    losses.append(float(loss.detach()))
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            scores = score_rows(model, validation, cache, device, frozen)
            report = threshold_report(validation, scores)
            selection = report["validation"]["ap"]
            record = {
                "stage": "probe" if frozen else "finetune",
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                **report,
            }
            history.append(record)
            write_json(output / "history.json", history)
            print(json.dumps(record), flush=True)
            if selection > best:
                best, stale = selection, 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                torch.save(
                    {"state_dict": best_state, "architecture": "x3d_s"},
                    output / "best.pt",
                )
            else:
                stale += 1
            if stale >= 3:
                break
    if best_state is None:
        raise RuntimeError("No completed validation epoch")
    model.load_state_dict(best_state)
    return history


def browser_video(path):
    converted = path.with_name(path.stem + "_h264.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-movflags",
            "+faststart",
            str(converted),
        ],
        check=True,
        timeout=60,
    )
    converted.replace(path)


def audit_errors(rows, scores, threshold, cache, output, config):
    records = []
    preview_counts = {"false_positive": 0, "false_negative": 0}
    destination = output / "errors"
    destination.mkdir(parents=True, exist_ok=True)
    errors = sorted(
        zip(rows, scores),
        key=lambda entry: abs(float(entry[1]) - entry[0]["label"]),
        reverse=True,
    )
    for row, score in errors:
        predicted = int(score >= threshold)
        if predicted == row["label"]:
            continue
        kind = "false_positive" if predicted else "false_negative"
        name = f"{kind}_{row['key']}_{row['start']}"
        record = {
            "video": row["video"],
            "start_seconds": row["start"] / row["fps"],
            "label": row["label"],
            "score": float(score),
            "kind": kind,
        }
        with np.load(cache / f"{row['cache']}.npz") as data:
            record["people"] = int(data["people"]) if "people" in data else None
            record["regions"] = data["regions"].tolist() if "regions" in data else None
            if preview_counts[kind] < 6:
                preview_counts[kind] += 1
                clips = data["clips"]
                writer = cv2.VideoWriter(
                    str(destination / f"{name}_crop.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    config.fps,
                    (config.size * len(clips), config.size),
                )
                for frame_index in range(config.frames):
                    writer.write(
                        cv2.cvtColor(
                            np.concatenate(clips[:, frame_index], axis=1),
                            cv2.COLOR_RGB2BGR,
                        )
                    )
                writer.release()
                browser_video(destination / f"{name}_crop.mp4")
                frames = read_frames(row, config)
                height, width = frames[0].shape[:2]
                writer = cv2.VideoWriter(
                    str(destination / f"{name}_source.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    config.fps,
                    (width, height),
                )
                for frame in frames:
                    writer.write(frame)
                writer.release()
                browser_video(destination / f"{name}_source.mp4")
                record["source_clip"] = f"errors/{name}_source.mp4"
                record["crop_clip"] = f"errors/{name}_crop.mp4"
        records.append(record)
    write_json(output / "errors.json", records)


def export_runtime(model, config, detector_config, report, output):
    model = prepare_export(model.cpu().eval())
    torch.set_num_threads(4)
    example = torch.zeros(1, 3, config.frames, config.size, config.size)
    destination = output / "runtime" / "models"
    destination.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, check_trace=False)
        traced.save(str(destination / "fight_model.ts"))
        restored = torch.jit.load(str(destination / "fight_model.ts"))
        torch.testing.assert_close(
            restored(example), model(example), atol=1e-4, rtol=1e-4
        )
    metadata = {
        **report,
        "schema_version": 1,
        "architecture": "x3d_s",
        "clip": config.to_dict(),
        "person_detector": detector_config.to_dict(),
        "approved_for_demo": False,
    }
    write_json(destination / "fight_runtime.json", metadata)
    with zipfile.ZipFile(
        output / "sentinel_fight_only_runtime.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in ("fight_model.ts", "fight_runtime.json"):
            archive.write(destination / name, "models/" + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--budget-minutes", type=float, default=25)
    parser.add_argument("--baseline-cache")
    args = parser.parse_args()
    if args.budget_minutes <= 0 or not torch.cuda.is_available():
        raise ValueError("Positive budget and a Colab GPU are required")
    source, output, cache = (
        Path(args.source_run),
        Path(args.output_dir),
        Path(args.cache_dir),
    )
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = fixed_manifest(source / "manifest.json", args.dataset_root)
    write_json(output / "source_manifest.json", manifest)
    detector_config = DetectorConfig()
    detector = PersonDetector(**detector_config.to_dict(), device="cuda")
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    summary = {
        "official_test_used": False,
        "approved_for_demo": False,
        "scope": "Repeated development validation on the original fixed split; independent test still required.",
        "baseline": json.loads((source / "report.json").read_text()),
        "variants": {},
    }
    write_json(output / "comparison.json", summary)
    if args.baseline_cache and not (output / "baseline" / "errors.json").exists():
        old_rows = json.loads((source / "cached_manifest.json").read_text())["rows"]
        cache_ids = {row_id(row): row["cache"] for row in old_rows}
        validation = [
            {**row, "cache": cache_ids[row_id(row)]}
            for row in manifest["rows"]
            if row["split"] == "validation"
        ]
        old_predictions = json.loads((source / "predictions.json").read_text())
        lookup = {
            (row["video"], round(row["start_seconds"], 6)): row["score"]
            for row in old_predictions
        }
        scores = np.array(
            [
                lookup[(row["video"], round(row["start"] / row["fps"], 6))]
                for row in validation
            ]
        )
        audit_errors(
            validation,
            scores,
            summary["baseline"]["threshold"],
            Path(args.baseline_cache),
            output / "baseline",
            ClipConfig(**manifest["clip"]),
        )
    for variant in ("interaction", "full", "full_hardneg"):
        destination = output / variant
        destination.mkdir(exist_ok=True)
        if (destination / "report.json").exists():
            summary["variants"][variant] = json.loads(
                (destination / "report.json").read_text()
            )
            continue
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        variant_deadline = min(
            deadline - 30, time.monotonic() + args.budget_minutes * 60 / 3 - 10
        )
        config = ClipConfig(
            region_mode="interaction" if variant == "interaction" else "full"
        )
        model = VideoClassifier("x3d_s", pretrained=True).cuda().eval()
        try:
            rows = cached_rows(
                manifest["rows"],
                model,
                detector,
                config,
                detector_config,
                cache,
                "cuda",
                variant_deadline,
            )
            train = [row for row in rows if row["split"] == "train"]
            validation = [row for row in rows if row["split"] == "validation"]
            expected = {
                row_id(row) for row in manifest["rows"] if row["split"] == "validation"
            }
            assert {row_id(row) for row in validation} == expected
            if variant == "full_hardneg":
                preceding = output / "full" / "best.pt"
                if not preceding.exists():
                    raise ValueError(
                        "Full-frame checkpoint required for train-only negative mining"
                    )
                mining_model = VideoClassifier("x3d_s", pretrained=False).cuda().eval()
                mining_model.load_state_dict(
                    torch.load(preceding, map_location="cuda", weights_only=True)[
                        "state_dict"
                    ]
                )
                candidates = extra_negative_candidates(
                    manifest["rows"], args.dataset_root, config
                )
                candidates = cached_rows(
                    candidates,
                    model,
                    detector,
                    config,
                    detector_config,
                    cache,
                    "cuda",
                    variant_deadline,
                )
                scores = score_rows(mining_model, candidates, cache, "cuda", False)
                hard = [
                    row
                    for row, score in sorted(
                        zip(candidates, scores),
                        key=lambda entry: entry[1],
                        reverse=True,
                    )[:32]
                ]
                train.extend(hard)
                write_json(destination / "hard_negatives.json", hard)
                del mining_model
            write_json(
                destination / "manifest.json",
                {**manifest, "rows": train + validation, "clip": config.to_dict()},
            )
            stable_train(
                model, train, validation, cache, "cuda", variant_deadline, destination
            )
            scores = score_rows(model, validation, cache, "cuda", False)
            report = {
                **threshold_report(validation, scores),
                "variant": variant,
                "train_clips": len(train),
                "validation_clips": len(validation),
                "approved_for_demo": False,
                "official_test_used": False,
                "selection_metric": "validation average_precision",
                "clip": config.to_dict(),
                "person_detector": detector_config.to_dict(),
            }
            values = report["validation"]
            report["decision"] = (
                "promising"
                if report["best_recall_at_precision75"] >= 0.6
                and (values["in_video_auc"] or 0) >= 0.65
                else "stop_and_review"
            )
            predictions = [
                {
                    "video": row["video"],
                    "start_seconds": row["start"] / row["fps"],
                    "label": row["label"],
                    "score": float(score),
                }
                for row, score in zip(validation, scores)
            ]
            write_json(destination / "predictions.json", predictions)
            audit_errors(
                validation,
                scores,
                report["threshold"],
                cache,
                destination,
                config,
            )
            export_runtime(model, config, detector_config, report, destination)
            write_json(destination / "report.json", report)
            summary["variants"][variant] = report
        except TimeoutError as error:
            summary["variants"][variant] = {
                "decision": "inconclusive",
                "reason": str(error),
            }
        finally:
            del model
            torch.cuda.empty_cache()
        write_json(output / "comparison.json", summary)
        print("VARIANT_DONE " + json.dumps(summary["variants"][variant]), flush=True)
    summary["seconds"] = time.monotonic() - started
    write_json(output / "comparison.json", summary)
    print("COMPARISON " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
