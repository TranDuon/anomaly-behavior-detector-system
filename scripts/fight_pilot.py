"""Budgeted fight-only pilot. Official test videos are never used for selection."""

import argparse
import hashlib
import json
import random
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn

from anomaly.fight_data import ubi_fights_items
from fight.model import VideoClassifier
from fight.vision import (
    ClipConfig,
    clip_tensor,
    motion_score,
    prepare_clip,
    regions_for,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def auc(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None


def metrics(rows, scores, threshold):
    labels = np.array([row["label"] for row in rows])
    predictions = np.asarray(scores) >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    per_video = []
    for video in sorted({row["video"] for row in rows}):
        selected = [index for index, row in enumerate(rows) if row["video"] == video]
        video_auc = auc(labels[selected], np.asarray(scores)[selected])
        if video_auc is not None:
            per_video.append(video_auc)
    normal = labels == 0
    return {
        "metric_scope": "selected pilot clips, not full-video frame-level evaluation",
        "auc": auc(labels, scores),
        "ap": float(average_precision_score(labels, scores)) if labels.any() else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": (
            float(predictions[normal].mean()) if normal.any() else None
        ),
        "in_video_auc": float(np.median(per_video)) if per_video else None,
        "mixed_label_videos": len(per_video),
        "clips": len(rows),
    }


def select_threshold(rows, scores):
    candidates = np.unique(np.concatenate([[0.5], np.linspace(0.05, 0.99, 60)]))
    values = [
        (metrics(rows, scores, float(threshold)), float(threshold))
        for threshold in candidates
    ]
    feasible = [value for value in values if value[0]["recall"] >= 0.7]
    return max(
        feasible or values,
        key=lambda value: (value[0]["f1"], value[0]["precision"], value[1]),
    )


def scene_hash(path):
    capture = cv2.VideoCapture(str(path))
    try:
        for seconds in (0, 2, 5):
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if grey.std() < 8:
                continue
            tiny = cv2.resize(grey, (9, 8))
            return np.packbits(tiny[:, 1:] > tiny[:, :-1]).tobytes().hex()
        raise ValueError(
            f"No usable scene frame in {path}; inspect video before training"
        )
    finally:
        capture.release()


def build_manifest(items, videos, clips_per_video, config, seed, excluded=()):
    rng = np.random.default_rng(seed)
    available = [
        item for item in items if item.split == "train" and item.uid not in excluded
    ]
    selected = []
    for positive in (False, True):
        group = [item for item in available if item.has_positive == positive]
        rng.shuffle(group)
        selected.extend(group[: videos // 2])
    if len(selected) < 8:
        raise ValueError("Need at least eight usable training videos for a pilot")
    hashes, groups = [], {}
    for item in selected:
        value = int(scene_hash(item.path), 16)
        nearby = [
            group
            for fingerprint, group in hashes
            if (value ^ fingerprint).bit_count() <= 6
        ]
        group_id = nearby[0] if nearby else item.uid
        hashes.append((value, group_id))
        groups[item.uid] = group_id
    group_names = sorted(set(groups.values()))
    partition = None
    for attempt in range(100):
        rng.shuffle(group_names)
        val_groups = set(group_names[: max(2, round(len(group_names) * 0.3))])
        candidate = {
            item.uid: "validation" if groups[item.uid] in val_groups else "train"
            for item in selected
        }
        if all(
            len(
                {item.has_positive for item in selected if candidate[item.uid] == split}
            )
            == 2
            for split in ("train", "validation")
        ):
            partition = candidate
            break
    if partition is None:
        raise ValueError("Cannot form scene-grouped splits containing both classes")
    rows = []
    for item in selected:
        capture = cv2.VideoCapture(str(item.path))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        labels = item.labels_for(count)
        span = int(round((config.frames - 1) / config.fps * fps)) + 1
        candidates = np.arange(0, max(0, count - span + 1), max(1, int(fps)))
        if not len(candidates):
            continue
        fractions = np.array(
            [labels[start : start + span].mean() for start in candidates]
        )
        for label in (0, 1):
            valid = (
                candidates[fractions == 0]
                if label == 0
                else candidates[fractions >= 0.7]
            )
            maximum = clips_per_video // 2 if item.has_positive else clips_per_video
            if len(valid):
                choices = rng.choice(
                    valid, size=min(maximum, len(valid)), replace=False
                )
                for start in sorted(choices):
                    rows.append(
                        {
                            "video": item.uid,
                            "key": item.key,
                            "path": str(item.path),
                            "start": int(start),
                            "fps": fps,
                            "label": label,
                            "split": partition[item.uid],
                            "scene_group": groups[item.uid],
                        }
                    )
    if any(
        len({row["label"] for row in rows if row["split"] == split}) != 2
        for split in ("train", "validation")
    ):
        raise ValueError("Selected clips do not contain both classes in each partition")
    return {
        "seed": seed,
        "clip": config.to_dict(),
        "rows": rows,
        "split_policy": "official train only; near-duplicate first-frame hashes grouped before splitting; visual scene audit still required",
        "official_test_used": False,
        "metric_scope": "selected pilot clips from official training videos only",
    }


def read_frames(row, config):
    capture = cv2.VideoCapture(row["path"])
    frames = []
    try:
        positions = np.rint(
            row["start"] + np.arange(config.frames) * row["fps"] / config.fps
        ).astype(int)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(positions[0]))
        current = int(positions[0])
        for position in positions:
            while current < position:
                if not capture.grab():
                    raise ValueError("Video ended before the selected clip")
                current += 1
            ok, frame = capture.read()
            current += 1
            if not ok:
                raise ValueError("Cannot decode selected clip")
            frames.append(frame)
        return frames
    finally:
        capture.release()


def cache_key(row, config, architecture):
    identity = {key: row[key] for key in ("video", "start", "fps", "label")}
    identity.update(
        clip=config.to_dict(),
        architecture=architecture,
        weights="kinetics400",
        detector="yolov8n.pt@640:0.4",
    )
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :24
    ]


def score_rows(model, rows, cache, device, frozen):
    scores = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            with np.load(cache / f"{row['cache']}.npz") as data:
                values = (
                    torch.from_numpy(data["features"].astype(np.float32)).to(device)
                    if frozen
                    else clip_tensor(data["clips"], device)
                )
                logits = model.head(values).flatten() if frozen else model(values)
                scores.append(float(torch.sigmoid(logits.max()).cpu()))
    return np.asarray(scores)


def train_stage(
    model, train, validation, cache, device, deadline, epochs, frozen, history, output
):
    model.set_stage("probe" if frozen else "finetune")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3 if frozen else 1e-4,
        weight_decay=0.01,
    )
    labels = np.array([row["label"] for row in train])
    class_counts = np.bincount(labels, minlength=2)
    weights = torch.tensor(
        [len(labels) / (2 * max(1, count)) for count in class_counts], device=device
    )
    best = max((entry.get("selection", -1.0) for entry in history), default=-1.0)
    for epoch in range(epochs):
        random.shuffle(train)
        total, completed = 0.0, 0
        model.train()
        model.backbone.eval()
        for row in train:
            if time.monotonic() >= deadline:
                break
            with np.load(cache / f"{row['cache']}.npz") as data:
                if frozen:
                    values = torch.from_numpy(data["features"].astype(np.float32)).to(
                        device
                    )
                    logits = model.head(values).flatten()
                else:
                    clips = data["clips"]
                    if random.random() < 0.5:
                        clips = clips[:, :, :, ::-1].copy()
                    logits = model(clip_tensor(clips, device))
            target = torch.tensor(float(row["label"]), device=device)
            loss = (
                nn.functional.binary_cross_entropy_with_logits(logits.max(), target)
                * weights[row["label"]]
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            completed += 1
        if not completed or time.monotonic() >= deadline:
            break
        scores = score_rows(model, validation, cache, device, frozen)
        result, threshold = select_threshold(validation, scores)
        inside = result["in_video_auc"]
        selection = (
            (inside if inside is not None else 0.0) + (result["auc"] or 0.0)
        ) / 2
        record = {
            "stage": "probe" if frozen else "finetune",
            "epoch": epoch + 1,
            "loss": total / completed,
            "validation": result,
            "threshold": threshold,
            "selection": selection,
        }
        history.append(record)
        write_json(output / "history.json", history)
        print(json.dumps(record), flush=True)
        if selection > best:
            best = selection
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": model.architecture,
                    "threshold": threshold,
                },
                output / "best.pt",
            )
        elif epoch >= 3:
            break
    checkpoint = output / "best.pt"
    if checkpoint.exists():
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"]
        )


def sanity_check(model, rows, cache, device, deadline):
    original = {
        key: value.detach().clone() for key, value in model.head.state_dict().items()
    }
    tiny = [
        row
        for label in (0, 1)
        for row in [row for row in rows if row["label"] == label][:4]
    ]
    values = []
    for row in tiny:
        with np.load(cache / f"{row['cache']}.npz") as data:
            values.append(
                torch.from_numpy(data["features"].astype(np.float32)).to(device)
            )
    optimizer = torch.optim.Adam(model.head.parameters(), lr=0.01)
    initial = final = None
    for step in range(100):
        if time.monotonic() >= deadline:
            break
        logits = torch.stack([model.head(features).max() for features in values])
        labels = torch.tensor(
            [row["label"] for row in tiny], dtype=torch.float32, device=device
        )
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        initial = float(loss.detach()) if initial is None else initial
        final = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.head.load_state_dict(original)
    return {
        "clips": len(tiny),
        "initial_loss": initial,
        "final_loss": final,
        "passed": initial is not None and final < initial * 0.5,
        "meaning": "Training plumbing only; does not measure generalization.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--architecture", choices=["s3d", "x3d_s"], default="x3d_s")
    parser.add_argument("--budget-minutes", type=float, default=30)
    parser.add_argument("--videos", type=int, default=48)
    parser.add_argument("--clips-per-video", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.budget_minutes <= 0 or args.videos < 8 or args.clips_per_video < 4:
        parser.error(
            "Budget must be positive; use at least 8 videos and 4 clips per video"
        )
    output, cache = Path(args.output_dir), Path(args.cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Pilot requires a Colab GPU. No silent CPU fallback.")
    config = ClipConfig()
    run_settings = {
        "architecture": args.architecture,
        "videos": args.videos,
        "clips_per_video": args.clips_per_video,
    }
    excluded = set()
    if args.exclude and Path(args.exclude).exists():
        excluded = set(
            json.loads(Path(args.exclude).read_text())["exclude_from_training"]
        )
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("clip") != config.to_dict()
            or manifest.get("seed") != args.seed
            or manifest.get("run_settings") != run_settings
        ):
            raise ValueError("Manifest settings changed; choose a new run directory")
    else:
        manifest = build_manifest(
            ubi_fights_items(Path(args.dataset_root)),
            args.videos,
            args.clips_per_video,
            config,
            args.seed,
            excluded,
        )
        manifest["run_settings"] = run_settings
        write_json(manifest_path, manifest)
    print(
        f"PILOT: {len(manifest['rows'])} clips; budget={args.budget_minutes} min; architecture={args.architecture}",
        flush=True,
    )
    model = VideoClassifier(args.architecture).to(device).eval()
    from ultralytics import YOLO

    detector = YOLO("yolov8n.pt")
    ready = []
    (output / "previews").mkdir(exist_ok=True)
    for index, row in enumerate(manifest["rows"]):
        if time.monotonic() >= deadline:
            break
        row = dict(row, cache=cache_key(row, config, args.architecture))
        destination = cache / f"{row['cache']}.npz"
        if not destination.exists():
            frames = read_frames(row, config)
            boxes = []
            for frame_index in (0, len(frames) // 2, len(frames) - 1):
                prediction = detector.predict(
                    frames[frame_index],
                    classes=[0],
                    conf=0.4,
                    imgsz=640,
                    device=str(device),
                    verbose=False,
                )[0]
                if prediction.boxes is not None:
                    boxes.extend(prediction.boxes.xyxy.cpu().numpy().tolist())
            height, width = frames[0].shape[:2]
            regions = regions_for(boxes, width, height, config.max_regions)
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
                motion=np.float32(motion_score(frames)),
            )
            temporary.replace(destination)
            if index < 24:
                preview = np.concatenate(
                    [clip[len(clip) // 2] for clip in clips], axis=1
                )
                cv2.imwrite(
                    str(
                        output
                        / "previews"
                        / f"{index:03d}_{row['key']}_label{row['label']}.jpg"
                    ),
                    cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
                )
        with np.load(destination) as data:
            row["motion"] = float(data["motion"])
        ready.append(row)
        if (index + 1) % 8 == 0:
            print(
                f"CACHE {index + 1}/{len(manifest['rows'])}; remaining={max(0, deadline - time.monotonic()):.0f}s",
                flush=True,
            )
    write_json(output / "cached_manifest.json", {**manifest, "rows": ready})
    train = [row for row in ready if row["split"] == "train"]
    validation = [row for row in ready if row["split"] == "validation"]
    history = []
    sanity = None
    complete = len(ready) == len(manifest["rows"])
    if complete and all(
        len({row["label"] for row in rows}) == 2 for rows in (train, validation)
    ):
        sanity = sanity_check(
            model, train, cache, device, min(deadline, time.monotonic() + 60)
        )
        write_json(output / "sanity.json", sanity)
        if sanity["passed"]:
            train_stage(
                model,
                train,
                validation,
                cache,
                device,
                min(deadline, time.monotonic() + 180),
                6,
                True,
                history,
                output,
            )
        if sanity["passed"] and time.monotonic() < deadline - 30:
            train_stage(
                model,
                train,
                validation,
                cache,
                device,
                deadline - 20,
                3,
                False,
                history,
                output,
            )
    report = {
        "schema_version": 1,
        "architecture": args.architecture,
        "clip": config.to_dict(),
        "approved_for_demo": False,
        "decision": "inconclusive",
        "reason": "Budget exhausted or no completed validation epoch.",
        "budget_minutes": args.budget_minutes,
        "seconds": time.monotonic() - started,
        "cached_clips": len(ready),
        "planned_clips": len(manifest["rows"]),
        "official_test_used": False,
        "sanity": sanity,
        "caveat": "Pilot validation only. Scene grouping is approximate; inspect previews before full training.",
    }
    if history and (output / "best.pt").exists():
        checkpoint = torch.load(
            output / "best.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["state_dict"])
        scores = score_rows(model, validation, cache, device, False)
        result, threshold = select_threshold(validation, scores)
        baseline = metrics(
            validation, np.array([row["motion"] for row in validation]), 0.05
        )
        inside, motion_inside = result["in_video_auc"], baseline["in_video_auc"]
        promising = (
            inside is not None
            and motion_inside is not None
            and result["mixed_label_videos"] >= 3
            and inside >= 0.65
            and inside >= motion_inside + 0.05
            and result["precision"] >= 0.75
            and result["recall"] >= 0.6
        )
        report.update(
            validation=result,
            motion_baseline=baseline,
            threshold=threshold,
            decision="promising" if promising else "stop_and_review",
            reason=(
                "Pilot gate passed; independent final evaluation still required."
                if promising
                else "Pilot has not demonstrated sufficient generalization; inspect labels, crops and errors before spending more GPU time."
            ),
        )
        model = model.cpu().eval()
        torch.set_num_threads(4)
        example = torch.zeros(1, 3, config.frames, config.size, config.size)
        with torch.inference_mode():
            scripted = torch.jit.trace(model, example, check_trace=False)
            torch.testing.assert_close(
                scripted(example), model(example), atol=1e-4, rtol=1e-4
            )
            measured = []
            for iteration in range(4):
                tick = time.monotonic()
                scripted(example)
                if iteration:
                    measured.append(time.monotonic() - tick)
        report["cpu_single_region_seconds"] = float(np.median(measured))
        report["cpu_measurement_note"] = (
            "Colab CPU, 4 threads, classifier only; local CPU + person detector must be measured separately."
        )
        export_dir = output / "runtime" / "models"
        export_dir.mkdir(parents=True, exist_ok=True)
        scripted.save(str(export_dir / "fight_model.ts"))
        write_json(export_dir / "fight_runtime.json", report)
        with zipfile.ZipFile(
            output / "sentinel_fight_only_runtime.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for file in export_dir.iterdir():
                archive.write(file, "models/" + file.name)
        timeline = [
            {
                "video": row["video"],
                "start_seconds": row["start"] / row["fps"],
                "label": row["label"],
                "score": float(score),
            }
            for row, score in zip(validation, scores)
        ]
        write_json(output / "predictions.json", timeline)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(12, 4))
        axis.plot(scores, label="fight score")
        axis.step(
            range(len(validation)),
            [row["label"] for row in validation],
            label="ground truth",
            alpha=0.6,
        )
        axis.axhline(
            threshold, color="red", linestyle="--", label="validation threshold"
        )
        axis.set(
            xlabel="Validation clips (see predictions.json for video/timestamp)",
            ylabel="Score",
            ylim=(-0.05, 1.05),
        )
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "validation.png")
        plt.close(figure)
    report["seconds"] = time.monotonic() - started
    write_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
