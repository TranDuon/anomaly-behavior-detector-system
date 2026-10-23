"""Extract tracked YOLOv8-pose keypoints from the fight datasets.

One compressed .npz per video in <pose-root>/<dataset>/<key>.npz, so a
Colab disconnect only loses the video being processed. Frames are sampled at
--fps (default settings.pose_fps) whatever the source frame rate; the same
model, rate and tracker must be used at runtime.

Tracking is sequential within a video, so one process leaves the GPU mostly
idle; --shard k/n runs the k-th of n interleaved slices of the video list,
and n such processes can share one GPU. Each shard writes its own manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomaly.fight_data import VideoItem, load_items
from core.config import settings


def reset_tracker(model) -> None:
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", None) or []:
        tracker.reset()


def extract_video(
    item: VideoItem,
    model,
    target_fps: float,
    conf: float,
    imgsz: int,
    device: str,
    tracker_config: str,
    max_seconds: float | None,
) -> dict:
    capture = cv2.VideoCapture(str(item.path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {item.path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
    step = max(source_fps / target_fps, 1.0)
    max_frames = int(max_seconds * source_fps) if max_seconds else None

    reset_tracker(model)
    sample_frames, frame_index, track_ids, boxes, keypoints = [], [], [], [], []
    width = height = 0
    decoded = 0
    next_sample = 0.0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            current = decoded
            decoded += 1
            if max_frames is not None and decoded > max_frames:
                decoded -= 1
                break
            if current + 1e-6 < next_sample:
                continue
            next_sample += step
            height, width = frame.shape[:2]
            sample_frames.append(current)
            result = model.track(
                frame,
                persist=True,
                tracker=tracker_config,
                conf=conf,
                imgsz=imgsz,
                device=device,
                classes=[settings.person_class_id],
                verbose=False,
            )[0]
            if result.boxes is None or result.boxes.id is None or result.keypoints is None:
                continue
            ids = result.boxes.id.int().cpu().numpy()
            frame_index.extend([current] * len(ids))
            track_ids.append(ids)
            boxes.append(result.boxes.xyxy.cpu().numpy())
            keypoints.append(result.keypoints.data.cpu().numpy())
    finally:
        capture.release()

    def stack(parts, shape, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(shape, dtype)

    return {
        "sample_frames": np.asarray(sample_frames, dtype=np.int32),
        "frame_index": np.asarray(frame_index, dtype=np.int32),
        "track_ids": stack(track_ids, (0,), np.int32),
        "boxes": stack(boxes, (0, 4), np.float16),
        "keypoints": stack(keypoints, (0, 17, 3), np.float16),
        "frame_count": np.int32(decoded),
        "fps": np.float32(source_fps),
        "width": np.int32(width),
        "height": np.int32(height),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract tracked pose keypoints")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="name:root[:split], name in ubi|scf|rlvs|ucf; repeatable",
    )
    parser.add_argument("--pose-root", required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--model", default=settings.pose_model)
    parser.add_argument("--fps", type=float, default=settings.pose_fps)
    parser.add_argument("--conf", type=float, default=settings.pose_conf_threshold)
    parser.add_argument("--imgsz", type=int, default=settings.pose_imgsz)
    parser.add_argument("--device", default=settings.yolo_device)
    parser.add_argument("--tracker", default=str(settings.pose_tracker_config))
    parser.add_argument(
        "--max-seconds-normal",
        type=float,
        default=None,
        help="Cap train videos without any positive label (long normal footage)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", default=None, help="k/n: process every n-th video starting at k")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    pose_root = Path(args.pose_root).resolve()
    items = [item for spec in args.dataset for item in load_items(spec)]
    if args.split != "all":
        items = [item for item in items if item.split == args.split]
    if args.limit is not None:
        items = items[: args.limit]
    manifest_name = "manifest.json"
    if args.shard:
        index_text, _, count_text = args.shard.partition("/")
        shard_index, shard_count = int(index_text), int(count_text)
        if not 0 <= shard_index < shard_count:
            parser.error("--shard must be k/n with 0 <= k < n")
        items = items[shard_index::shard_count]
        manifest_name = f"manifest_shard{shard_index}of{shard_count}.json"
    if not items:
        raise RuntimeError("No videos found for the given datasets")

    config = {
        "pose_model": args.model,
        "fps": args.fps,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "tracker": Path(args.tracker).name,
    }
    model = YOLO(args.model)
    print(f"[pose] {len(items)} video(s) -> {pose_root}", flush=True)

    manifests: dict[str, dict] = {}
    for index, item in enumerate(items, start=1):
        dataset_dir = pose_root / item.dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = dataset_dir / manifest_name
        if item.dataset not in manifests:
            manifest = {"config": config, "videos": {}}
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("config") != config and not args.force:
                    raise RuntimeError(
                        f"Pose cache {dataset_dir} was built with {manifest.get('config')}; "
                        "use a new --pose-root or --force."
                    )
                manifest["config"] = config
            manifests[item.dataset] = manifest
        manifest = manifests[item.dataset]

        output_path = dataset_dir / f"{item.key}.npz"
        if output_path.exists() and not args.force:
            print(f"[{index}/{len(items)}] Skip {item.uid}: cached", flush=True)
            continue

        max_seconds = None
        if item.split == "train" and not item.has_positive:
            max_seconds = args.max_seconds_normal
        started = time.perf_counter()
        payload = extract_video(
            item,
            model,
            target_fps=args.fps,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            tracker_config=args.tracker,
            max_seconds=max_seconds,
        )
        temporary_path = output_path.with_name(output_path.stem + ".partial.npz")
        np.savez_compressed(temporary_path, **payload)
        temporary_path.replace(output_path)
        seconds = time.perf_counter() - started
        manifest["videos"][item.key] = {
            "source": str(item.path),
            "split": item.split,
            "frames": int(payload["frame_count"]),
            "samples": int(len(payload["sample_frames"])),
            "person_rows": int(len(payload["track_ids"])),
            "tracks": int(len(np.unique(payload["track_ids"]))),
            "max_seconds": max_seconds,
            "seconds": round(seconds, 2),
        }
        write_manifest(manifest_path, manifest)
        rate = len(payload["sample_frames"]) / max(seconds, 1e-6)
        print(
            f"[{index}/{len(items)}] {item.uid}: samples={len(payload['sample_frames'])} "
            f"rows={len(payload['track_ids'])} ({rate:.1f} samples/s)",
            flush=True,
        )

    for dataset, manifest in manifests.items():
        write_manifest(pose_root / dataset / manifest_name, manifest)
    print("[pose] Complete.", flush=True)


if __name__ == "__main__":
    main()
