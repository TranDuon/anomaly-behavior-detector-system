"""Shared sparse pose extraction for cached clips and causal CPU streams."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from fight.vision import DetectionCache


@dataclass(frozen=True)
class PoseConfig:
    model_name: str = "yolov8n-pose.pt"
    image_size: int = 320
    confidence: float = 0.3
    keypoint_confidence: float = 0.3
    samples: int = 8
    version: int = 1

    def __post_init__(self):
        if self.model_name != "yolov8n-pose.pt" or self.samples != 8:
            raise ValueError("Unsupported pose preprocessing schema")
        if (
            self.image_size not in {320, 640}
            or not 0 < self.confidence < 1
            or not 0 < self.keypoint_confidence < 1
        ):
            raise ValueError("Invalid pose configuration")


class PoseDetector:
    def __init__(self, config=None, device="cpu", model_path=None):
        from ultralytics import YOLO

        self.config = config or PoseConfig()
        self.device = device
        self.model = YOLO(str(model_path or self.config.model_name))
        self.cache = {}
        self.batch_detection = False

    def retain(self, frames):
        owners = {id(DetectionCache.owner(frame)) for frame in frames}
        self.cache = {
            key: value for key, value in self.cache.items() if key[0] in owners
        }

    @staticmethod
    def frame_key(frame):
        owner = DetectionCache.owner(frame)
        return (
            id(owner),
            frame.__array_interface__["data"][0],
            frame.shape,
            frame.strides,
        )

    @staticmethod
    def records(result):
        records = []
        if result.keypoints is not None:
            for box, points in zip(
                result.boxes.xyxy.cpu().numpy(), result.keypoints.data.cpu().numpy()
            ):
                if (
                    points.shape != (17, 3)
                    or not np.isfinite(points).all()
                    or not np.isfinite(box).all()
                ):
                    continue
                records.append(
                    {
                        "box": np.asarray(box, dtype=np.float32),
                        "points": np.asarray(points, dtype=np.float32),
                    }
                )
        return records

    def prefetch(self, frames):
        if not getattr(self, "batch_detection", False):
            return
        missing = {
            self.frame_key(frame): frame
            for frame in frames
            if self.frame_key(frame) not in self.cache
        }
        if len(missing) < 2 or len({frame.shape for frame in missing.values()}) != 1:
            return
        results = self.model.predict(
            list(missing.values()),
            imgsz=self.config.image_size,
            conf=self.config.confidence,
            device=self.device,
            verbose=False,
        )
        if len(results) != len(missing):
            raise ValueError("Batched pose output count differs from source frames")
        for (key, frame), result in zip(missing.items(), results):
            self.cache[key] = (DetectionCache.owner(frame), self.records(result))
        while len(self.cache) > 64:
            del self.cache[next(iter(self.cache))]

    def poses(self, frame):
        owner = DetectionCache.owner(frame)
        key = self.frame_key(frame)
        if key not in self.cache:
            result = self.model.predict(
                frame,
                imgsz=self.config.image_size,
                conf=self.config.confidence,
                device=self.device,
                verbose=False,
            )[0]
            self.cache[key] = (owner, self.records(result))
            if len(self.cache) > 64:
                del self.cache[next(iter(self.cache))]
        return self.cache[key][1]

    def detect(self, frame):
        return [
            SimpleNamespace(bbox_xyxy=record["box"]) for record in self.poses(frame)
        ]

    def warmup(self, image_size=320):
        self.poses(np.zeros((image_size, image_size, 3), dtype=np.uint8))
        self.cache.clear()


def pair_pose_sequence(detections, region, confidence=0.3):
    output = np.zeros((len(detections), 2, 17, 3), dtype=np.float32)
    previous = [None, None]
    left, top, right, bottom = region
    for frame_index, records in enumerate(detections):
        visible = []
        for record in records:
            box, points = record["box"], record["points"]
            if (
                np.shape(points) != (17, 3)
                or np.shape(box) != (4,)
                or not np.isfinite(points).all()
                or not np.isfinite(box).all()
            ):
                continue
            center = (box[:2] + box[2:]) / 2
            if (
                left <= center[0] <= right
                and top <= center[1] <= bottom
                and box[3] > box[1]
                and box[2] > box[0]
            ):
                visible.append(record)
        visible.sort(
            key=lambda record: -float(np.prod(record["box"][2:] - record["box"][:2]))
        )
        selected = [None, None]
        remaining = list(visible)
        for slot, old in enumerate(previous):
            if old is None or not remaining:
                continue
            distances = [
                np.linalg.norm((record["box"][:2] + record["box"][2:]) / 2 - old[:2])
                / max(old[2], 1)
                for record in remaining
            ]
            closest = int(np.argmin(distances))
            if distances[closest] < 1.5:
                selected[slot] = remaining.pop(closest)
        for slot in range(2):
            if selected[slot] is None and remaining:
                selected[slot] = remaining.pop(0)
        active = [record for record in selected if record is not None]
        if not active:
            previous = [None, None]
            continue
        centers = [(record["box"][:2] + record["box"][2:]) / 2 for record in active]
        origin = np.mean(centers, axis=0)
        scale = max(float(record["box"][3] - record["box"][1]) for record in active)
        for slot, record in enumerate(selected):
            if record is None:
                previous[slot] = None
                continue
            box, points = record["box"], record["points"]
            valid = points[:, 2] >= confidence
            normalized = np.clip((points[:, :2] - origin) / scale, -4, 4)
            output[frame_index, slot, valid, :2] = normalized[valid]
            output[frame_index, slot, valid, 2] = np.clip(points[valid, 2], 0, 1)
            previous[slot] = np.r_[(box[:2] + box[2:]) / 2, box[3] - box[1]]
    return output


def pose_inputs(frames, regions, detector):
    indices = (
        np.array([0, 2, 4, 6, 8, 10, 12, 15])
        if len(frames) == 16
        else np.rint(np.linspace(0, len(frames) - 1, detector.config.samples)).astype(
            int
        )
    )
    detections = [detector.poses(frames[index]) for index in indices]
    return np.stack(
        [
            pair_pose_sequence(detections, region, detector.config.keypoint_confidence)
            for region in regions
        ]
    )


def flip_pose(poses):
    flipped = poses.copy()
    flipped[..., 0] *= -1
    return flipped[..., [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], :]
