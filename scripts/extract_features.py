"""Extract per-track feature sequences and frame metadata from video clips."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomaly.features import TrackFeatureBuffer
from core.config import settings
from detection.detector import PersonDetector
from tracking.tracker import PersonTracker


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def natural_key(path: Path) -> list:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def direct_images(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_key,
    )


def discover_clips(input_path: Path) -> list[Path]:
    """Return video files and image-sequence directories below input_path."""
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported input file: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    clips: list[Path] = []
    if direct_images(input_path):
        clips.append(input_path)

    for path in sorted(input_path.rglob("*")):
        if path.name.lower().endswith("_gt"):
            continue
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            clips.append(path)
        elif path.is_dir() and direct_images(path):
            clips.append(path)

    unique = {str(path.resolve()): path.resolve() for path in clips}
    return sorted(unique.values(), key=lambda path: str(path).lower())


def iter_frames(clip_path: Path, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Yield frames from a video or a directory containing an image sequence."""
    yielded = 0
    if clip_path.is_dir():
        for image_path in direct_images(clip_path):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            yield frame
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
        return

    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {clip_path}")
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            yield frame
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
    finally:
        capture.release()


def extract_clip(
    clip_path: Path,
    detector: PersonDetector,
    stride: int,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extract full-length rolling sequences from one independent clip."""
    tracker = PersonTracker()
    feature_buffer = TrackFeatureBuffer()
    track_steps: dict[int, int] = {}
    sequences: list[np.ndarray] = []
    frame_numbers: list[int] = []
    track_ids: list[int] = []
    frame_count = 0
    detection_count = 0
    track_observations = 0

    for frame in iter_frames(clip_path, max_frames=max_frames):
        frame_count += 1
        height, width = frame.shape[:2]
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        detection_count += len(detections)
        track_observations += len(tracks)

        for track in tracks:
            feature_buffer.update(
                track.track_id,
                track.bbox_xyxy,
                width,
                height,
            )
            track_steps[track.track_id] = track_steps.get(track.track_id, 0) + 1
            step = track_steps[track.track_id]
            if (
                feature_buffer.buffer_len(track.track_id) >= settings.seq_len
                and (step - settings.seq_len) % stride == 0
            ):
                sequence = feature_buffer.get_sequence(track.track_id)
                if sequence is not None:
                    sequences.append(sequence.copy())
                    frame_numbers.append(frame_count)
                    track_ids.append(track.track_id)

        active_ids = {track.track_id for track in tracks}
        feature_buffer.prune(active_ids)
        for track_id in set(track_steps) - active_ids:
            del track_steps[track_id]

        if frame_count % 100 == 0:
            print(
                f"    frames={frame_count} detections={detection_count} "
                f"sequences={len(sequences)}",
                flush=True,
            )

    if sequences:
        sequence_array = np.stack(sequences).astype(np.float32, copy=False)
    else:
        sequence_array = np.empty(
            (0, settings.seq_len, settings.feature_dim), dtype=np.float32
        )
    stats = {
        "frames": frame_count,
        "detections": detection_count,
        "track_observations": track_observations,
        "sequences": int(len(sequence_array)),
    }
    return (
        sequence_array,
        np.asarray(frame_numbers, dtype=np.int32),
        np.asarray(track_ids, dtype=np.int32),
        stats,
    )


def clip_key(clip_path: Path, input_path: Path) -> str:
    try:
        relative = clip_path.relative_to(input_path)
    except ValueError:
        relative = Path(clip_path.name)
    if str(relative) == ".":
        relative = Path(clip_path.name)
    if relative.suffix.lower() in VIDEO_EXTENSIONS:
        relative = relative.with_suffix("")
    return "__".join(relative.parts)


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract LSTM feature sequences from normal video clips"
    )
    parser.add_argument(
        "--input", required=True, help="Video, image clip, or dataset directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Persistent feature cache directory"
    )
    parser.add_argument("--stride", type=int, default=10, help="Sliding-window stride")
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing clip caches"
    )
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be at least 1")

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = discover_clips(input_path)
    if args.limit_clips is not None:
        clips = clips[: args.limit_clips]
    if not clips:
        raise RuntimeError(f"No supported clips found below {input_path}")

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"input": str(input_path), "clips": {}}
    extraction_config = {
        "seq_len": settings.seq_len,
        "feature_dim": settings.feature_dim,
        "stride": args.stride,
        "yolo_model": settings.yolo_model,
        "yolo_device": settings.yolo_device,
        "yolo_conf_threshold": settings.yolo_conf_threshold,
    }
    previous_config = manifest.get("config")
    if previous_config and previous_config != extraction_config and not args.force:
        raise RuntimeError(
            "Feature cache configuration changed. Use a new output directory or "
            "rerun with --force."
        )
    manifest["config"] = extraction_config

    print(f"[extract] Found {len(clips)} clip(s). Output: {output_dir}")
    detector = PersonDetector()
    for index, clip_path in enumerate(clips, start=1):
        key = clip_key(clip_path, input_path)
        output_path = output_dir / f"{key}.npz"
        if output_path.exists() and not args.force:
            with np.load(output_path) as cached:
                count = int(len(cached["sequences"]))
            manifest["clips"].setdefault(
                key,
                {
                    "source": str(clip_path),
                    "cache": str(output_path),
                    "sequences": count,
                },
            )
            print(f"[{index}/{len(clips)}] Skip {key}: cached {count} sequences")
            continue

        print(f"[{index}/{len(clips)}] Extracting {clip_path}")
        sequences, frame_numbers, track_ids, stats = extract_clip(
            clip_path,
            detector,
            stride=args.stride,
            max_frames=args.max_frames,
        )
        np.savez_compressed(
            output_path,
            sequences=sequences,
            frame_numbers=frame_numbers,
            track_ids=track_ids,
        )
        manifest["clips"][key] = {
            "source": str(clip_path),
            "cache": str(output_path),
            **stats,
        }
        write_manifest(manifest_path, manifest)
        print(f"    saved {len(sequences)} sequences -> {output_path}")

    write_manifest(manifest_path, manifest)
    total_sequences = sum(
        int(item.get("sequences", 0)) for item in manifest["clips"].values()
    )
    print(f"[extract] Complete. Cached sequences: {total_sequences}")


if __name__ == "__main__":
    main()
