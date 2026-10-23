"""Extract tracked YOLOv8-pose keypoints from the fight datasets.

One compressed .npz per video in <pose-root>/<dataset>/<key>.npz, so a
Colab disconnect only loses the video being processed. Frames are sampled at
--fps (default settings.pose_fps) whatever the source frame rate; the same
model, rate and tracker must be used at runtime.

Tracking is sequential within a video, so one process leaves the GPU mostly
idle; --shard k/n runs the k-th of n interleaved slices of the video list,
and n such processes can share one GPU. Each shard writes its own manifest.

Cutting extraction time (each option only changes which frames are read):
  --fight-margin-seconds  training videos with a fight keep only the fight
                          runs plus this margin; the tracker restarts per range
  --max-seconds-normal    training videos without a fight keep their start
  --max-seconds-test-normal  same for test videos without a fight — a
                          development shortcut; evaluation then reports that
                          the official protocol was not followed
A cached video is re-extracted when its stored options differ from the ones
requested, so dropping a cap later (final evaluation) extends only the
videos that were capped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomaly.fight_data import VideoItem, load_items, training_ranges
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
    ranges: Optional[List[Tuple[int, int]]] = None,
    half: bool = False,
) -> dict:
    """Sample frames on a fixed fps grid; inside `ranges` only when given."""
    capture = cv2.VideoCapture(str(item.path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {item.path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
    step = max(source_fps / target_fps, 1.0)
    max_frames = int(max_seconds * source_fps) if max_seconds else None
    last_wanted = ranges[-1][1] if ranges else None

    sample_frames, segment_ids, frame_index, track_ids, boxes, keypoints = [], [], [], [], [], []
    width = height = 0
    decoded = 0
    next_sample = 0.0
    range_index = 0
    current_segment = -1
    truncated = False
    try:
        while True:
            current = decoded
            if max_frames is not None and current >= max_frames:
                truncated = capture.grab()   # stopped by the cap only if frames remain
                break
            if last_wanted is not None and current > last_wanted:
                break
            on_grid = current + 1e-6 >= next_sample
            if ranges:
                while range_index < len(ranges) and current > ranges[range_index][1]:
                    range_index += 1
                inside = range_index < len(ranges) and current >= ranges[range_index][0]
            else:
                inside = True
            wanted = on_grid and inside
            # grab() skips colour conversion for frames that are not sampled.
            if wanted:
                success, frame = capture.read()
            else:
                success, frame = capture.grab(), None
            if not success:
                break
            decoded += 1
            if on_grid:
                next_sample += step
            if not wanted:
                continue

            segment = range_index if ranges else 0
            if segment != current_segment:
                reset_tracker(model)
                current_segment = segment
            height, width = frame.shape[:2]
            sample_frames.append(current)
            segment_ids.append(segment)
            result = model.track(
                frame,
                persist=True,
                tracker=tracker_config,
                conf=conf,
                imgsz=imgsz,
                device=device,
                half=half,
                classes=[settings.person_class_id],
                verbose=False,
            )[0]
            if result.boxes is None or result.boxes.id is None or result.keypoints is None:
                continue
            ids = result.boxes.id.int().cpu().numpy()
            frame_index.extend([current] * len(ids))
            track_ids.append(ids)
            boxes.append(result.boxes.xyxy.float().cpu().numpy())
            keypoints.append(result.keypoints.data.float().cpu().numpy())
    finally:
        capture.release()

    def stack(parts, shape, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(shape, dtype)

    return {
        "sample_frames": np.asarray(sample_frames, dtype=np.int32),
        "segment_ids": np.asarray(segment_ids, dtype=np.int16),
        "frame_index": np.asarray(frame_index, dtype=np.int32),
        "track_ids": stack(track_ids, (0,), np.int32),
        "boxes": stack(boxes, (0, 4), np.float16),
        "keypoints": stack(keypoints, (0, 17, 3), np.float16),
        "frame_count": np.int32(decoded),
        "truncated": np.bool_(truncated),
        "fps": np.float32(source_fps),
        "width": np.int32(width),
        "height": np.int32(height),
    }


def extraction_options(item: VideoItem, args) -> dict:
    """Per-video options that decide which frames are read (stored in the cache)."""
    max_seconds = None
    if not item.has_positive:
        max_seconds = args.max_seconds_normal if item.split == "train" else args.max_seconds_test_normal
    margin = args.fight_margin_seconds
    crop = margin is not None and item.split == "train" and item.has_positive
    return {
        "max_seconds": float(max_seconds) if max_seconds else 0.0,
        "fight_margin_seconds": float(margin) if crop else -1.0,
    }


def cache_matches(path: Path, requested: dict) -> bool:
    """True when the cached extraction already covers what `requested` asks for.

    A cache that holds more than requested is kept: an uncut video satisfies
    any cap (a cap that never cut the video counts as uncut), a longer cap a
    shorter one, the whole video or a wider margin a narrower margin.
    """
    try:
        with np.load(path) as data:
            margin = float(data["fight_margin_seconds"]) if "fight_margin_seconds" in data else -1.0
            cap = float(data["max_seconds"]) if "max_seconds" in data else 0.0
            truncated = bool(data["truncated"]) if "truncated" in data else cap > 0
    except (OSError, ValueError, KeyError):
        return False
    wanted_margin, wanted_cap = requested["fight_margin_seconds"], requested["max_seconds"]
    margin_covered = margin < 0 or (wanted_margin >= 0 and wanted_margin <= margin)
    cap_covered = not truncated or (wanted_cap > 0 and wanted_cap <= cap)
    return margin_covered and cap_covered


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
        "--half", choices=("auto", "on", "off"), default="auto",
        help="FP16 inference; auto = on for CUDA devices",
    )
    parser.add_argument("--max-seconds-normal", type=float, default=None)
    parser.add_argument("--max-seconds-test-normal", type=float, default=None)
    parser.add_argument("--fight-margin-seconds", type=float, default=None)
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
    half = args.half == "on" or (args.half == "auto" and str(args.device).startswith("cuda"))

    config = {
        "pose_model": args.model,
        "fps": args.fps,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "tracker": Path(args.tracker).name,
    }
    model = YOLO(args.model)
    print(f"[pose] {len(items)} video(s) -> {pose_root} (half={half})", flush=True)

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

        options = extraction_options(item, args)
        output_path = dataset_dir / f"{item.key}.npz"
        if output_path.exists() and not args.force and cache_matches(output_path, options):
            print(f"[{index}/{len(items)}] Skip {item.uid}: cached", flush=True)
            continue

        ranges = None
        started = time.perf_counter()
        if options["fight_margin_seconds"] >= 0:
            capture = cv2.VideoCapture(str(item.path))
            source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            capture.release()
            ranges = training_ranges(item, int(round(options["fight_margin_seconds"] * source_fps)))
        payload = extract_video(
            item,
            model,
            target_fps=args.fps,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            tracker_config=args.tracker,
            max_seconds=options["max_seconds"] or None,
            ranges=ranges,
            half=half,
        )
        payload["max_seconds"] = np.float32(options["max_seconds"])
        payload["fight_margin_seconds"] = np.float32(options["fight_margin_seconds"])
        temporary_path = output_path.with_name(output_path.stem + ".partial.npz")
        np.savez_compressed(temporary_path, **payload)
        temporary_path.replace(output_path)
        seconds = time.perf_counter() - started
        manifest["videos"][item.key] = {
            "source": str(item.path),
            "split": item.split,
            "frames": int(payload["frame_count"]),
            "samples": int(len(payload["sample_frames"])),
            "segments": int(len(np.unique(payload["segment_ids"]))),
            "person_rows": int(len(payload["track_ids"])),
            "tracks": int(len(np.unique(payload["track_ids"]))),
            **options,
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
