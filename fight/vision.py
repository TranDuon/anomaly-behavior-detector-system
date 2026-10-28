from dataclasses import asdict, dataclass
from collections import OrderedDict

import cv2
import numpy as np
import torch


class DetectionCache:
    def __init__(self, detector, capacity=48):
        self.detector = detector
        self.capacity = capacity
        self.entries = OrderedDict()

    @staticmethod
    def owner(frame):
        owner = frame
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        return owner

    def retain(self, frames):
        active = {id(self.owner(frame)) for frame in frames}
        for key in list(self.entries):
            if key[0] not in active:
                del self.entries[key]

    def detect(self, frame):
        owner = self.owner(frame)
        key = (
            id(owner),
            frame.__array_interface__["data"][0],
            frame.shape,
            frame.strides,
        )
        if key not in self.entries:
            self.entries[key] = (owner, self.detector.detect(frame))
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        self.entries.move_to_end(key)
        return self.entries[key][1]

    def prefetch(self, frames):
        if hasattr(self.detector, "prefetch"):
            self.detector.prefetch(frames)


@dataclass(frozen=True)
class ClipConfig:
    frames: int = 16
    fps: float = 8.0
    size: int = 182
    stride_seconds: float = 0.5
    max_regions: int = 3
    version: int = 1
    region_mode: str = "interaction"
    trim_black_borders: bool = False

    def __post_init__(self):
        if self.region_mode not in {"interaction", "full", "tubes", "pairs"}:
            raise ValueError("Unsupported region mode")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class DetectorConfig:
    model_name: str = "yolov8n.pt"
    confidence: float = 0.4
    iou: float = 0.45
    image_size: int = 640

    def to_dict(self):
        return asdict(self)


def clip_regions(boxes, width, height, config):
    if config.region_mode == "full":
        return [(0, 0, width, height)]
    if config.region_mode == "pairs":
        return pair_regions(boxes, width, height, config.max_regions)
    return regions_for(boxes, width, height, config.max_regions)


def pair_regions(boxes, width, height, limit=3):
    full = (0, 0, width, height)
    if limit <= 1:
        return [full]
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes = boxes[np.isfinite(boxes).all(axis=1)]
    boxes[:, :2] = np.maximum(boxes[:, :2], 0)
    boxes[:, 2:] = np.minimum(boxes[:, 2:], [width, height])
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    boxes = sorted(
        boxes,
        key=lambda box: float((box[2] - box[0]) * (box[3] - box[1])),
        reverse=True,
    )[:64]
    candidates = []
    for first_index, first in enumerate(boxes):
        for second in boxes[first_index + 1 :]:
            scale = max(first[3] - first[1], second[3] - second[1])
            gap_x = max(0, first[0] - second[2], second[0] - first[2])
            gap_y = max(0, first[1] - second[3], second[1] - first[3])
            if gap_x > scale * 0.6 or gap_y > scale * 0.3:
                continue
            lower = np.minimum(first[:2], second[:2])
            upper = np.maximum(first[2:], second[2:])
            margin = max(upper - lower) * 0.2
            region = (
                max(0, int(lower[0] - margin)),
                max(0, int(lower[1] - margin)),
                min(width, int(upper[0] + margin)),
                min(height, int(upper[1] + margin)),
            )
            distance = np.linalg.norm(
                (first[:2] + first[2:] - second[:2] - second[2:]) / 2
            )
            priority = (gap_x + gap_y + 0.25 * distance) / scale - 0.15 * scale / height
            candidates.append((float(priority), region))
    selected = []
    for priority, region in sorted(candidates):
        duplicate = False
        for previous in selected:
            intersection = max(
                0, min(region[2], previous[2]) - max(region[0], previous[0])
            ) * max(0, min(region[3], previous[3]) - max(region[1], previous[1]))
            union = (
                (region[2] - region[0]) * (region[3] - region[1])
                + (previous[2] - previous[0]) * (previous[3] - previous[1])
                - intersection
            )
            if intersection / max(1, union) > 0.6:
                duplicate = True
                break
        if not duplicate and region != full:
            selected.append(region)
        if len(selected) == limit - 1:
            break
    return selected + [full]


def content_bounds(frames):
    height, width = frames[0].shape[:2]
    sampled = [frames[index] for index in (0, len(frames) // 2, len(frames) - 1)]
    active = np.maximum.reduce([np.max(frame, axis=2) > 18 for frame in sampled])
    rows = np.flatnonzero(active.mean(axis=1) > 0.02)
    columns = np.flatnonzero(active.mean(axis=0) > 0.02)
    full = (0, 0, width, height)
    if not len(rows) or not len(columns):
        return full
    left, right = int(columns[0]), int(columns[-1]) + 1
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    if (
        right - left < width * 0.35
        or bottom - top < height * 0.35
        or (right - left) * (bottom - top) < width * height * 0.2
    ):
        return full
    return (
        left if left >= 8 else 0,
        top if top >= 8 else 0,
        right if width - right >= 8 else width,
        bottom if height - bottom >= 8 else height,
    )


def person_tubes(sampled_boxes):
    tracks = []
    for boxes in sampled_boxes:
        used = set()
        existing = len(tracks)
        for raw in boxes:
            box = np.asarray(raw, dtype=np.float32)
            if (
                box.shape != (4,)
                or not np.isfinite(box).all()
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                continue
            candidates = []
            for index in range(existing):
                if index in used:
                    continue
                previous = tracks[index]["last"]
                intersection = np.maximum(
                    0,
                    np.minimum(box[2:], previous[2:])
                    - np.maximum(box[:2], previous[:2]),
                ).prod()
                area = np.maximum(0, box[2:] - box[:2]).prod()
                old_area = np.maximum(0, previous[2:] - previous[:2]).prod()
                iou = float(intersection / max(1, area + old_area - intersection))
                distance = np.linalg.norm(
                    (box[:2] + box[2:] - previous[:2] - previous[2:]) / 2
                )
                scale = max(box[3] - box[1], previous[3] - previous[1])
                if iou >= 0.05 or distance <= scale * 0.6:
                    candidates.append((iou - float(distance / scale) * 0.1, index))
            if candidates:
                index = max(candidates)[1]
                track = tracks[index]
                track["bounds"] = np.r_[
                    np.minimum(track["bounds"][:2], box[:2]),
                    np.maximum(track["bounds"][2:], box[2:]),
                ]
                track["last"] = box
                used.add(index)
            else:
                tracks.append({"last": box, "bounds": box.copy()})
    return [track["bounds"].tolist() for track in tracks]


def build_clip_inputs(frames, detector, config, *, include_person_boxes=False):
    height, width = frames[0].shape[:2]
    bounds = (
        content_bounds(frames) if config.trim_black_borders else (0, 0, width, height)
    )
    left, top, right, bottom = bounds
    content = [frame[top:bottom, left:right] for frame in frames]
    if hasattr(detector, "prefetch"):
        detector.prefetch(
            [content[index] for index in (0, len(content) // 2, len(content) - 1)]
        )
    sampled_boxes = [
        [detection.bbox_xyxy for detection in detector.detect(content[index])]
        for index in (0, len(content) // 2, len(content) - 1)
    ]
    counts = [len(boxes) for boxes in sampled_boxes]
    boxes = (
        person_tubes(sampled_boxes)
        if config.region_mode in {"tubes", "pairs"}
        else [box for boxes in sampled_boxes for box in boxes]
    )
    regions = clip_regions(boxes, right - left, bottom - top, config)
    clips = np.stack([prepare_clip(content, region, config.size) for region in regions])
    absolute = [
        [region[0] + left, region[1] + top, region[2] + left, region[3] + top]
        for region in regions
    ]
    result = (clips, absolute, max(counts, default=0), bounds, counts)
    if not include_person_boxes:
        return result
    latest_boxes = []
    for raw in sampled_boxes[-1]:
        box = np.asarray(raw, dtype=np.float32)
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        translated = np.clip(
            box + np.array([left, top, left, top]),
            [0, 0, 0, 0],
            [width - 1, height - 1, width - 1, height - 1],
        ).astype(int)
        if translated[2] > translated[0] and translated[3] > translated[1]:
            latest_boxes.append(translated.tolist())
    return (*result, latest_boxes)


def regions_for(boxes, width, height, limit=3):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes = boxes[np.isfinite(boxes).all(axis=1)]
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    if not len(boxes):
        return [(0, 0, width, height)]
    groups = [{index} for index in range(len(boxes))]
    changed = True
    while changed:
        changed = False
        for first in range(len(groups)):
            if changed:
                break
            for second in range(first + 1, len(groups)):
                linked = False
                for first_index in groups[first]:
                    for second_index in groups[second]:
                        first_box, second_box = boxes[first_index], boxes[second_index]
                        gap = max(
                            0,
                            first_box[0] - second_box[2],
                            second_box[0] - first_box[2],
                        )
                        vertical = max(
                            0,
                            first_box[1] - second_box[3],
                            second_box[1] - first_box[3],
                        )
                        scale = max(
                            first_box[3] - first_box[1], second_box[3] - second_box[1]
                        )
                        linked |= gap < scale * 0.6 and vertical < scale * 0.3
                if linked:
                    groups[first] |= groups.pop(second)
                    changed = True
                    break
    groups.sort(
        key=lambda group: (
            -len(group),
            -float(
                np.sum(
                    (boxes[list(group), 2] - boxes[list(group), 0])
                    * (boxes[list(group), 3] - boxes[list(group), 1])
                )
            ),
        )
    )
    regions = []
    for group in groups[:limit]:
        selected = boxes[list(group)]
        lower, upper = selected[:, :2].min(axis=0), selected[:, 2:].max(axis=0)
        margin = max(upper - lower) * 0.2
        regions.append(
            (
                max(0, int(lower[0] - margin)),
                max(0, int(lower[1] - margin)),
                min(width, int(upper[0] + margin)),
                min(height, int(upper[1] + margin)),
            )
        )
    return regions


def prepare_clip(frames, region, size):
    images = []
    for frame in frames:
        height, width = frame.shape[:2]
        left, top, right, bottom = region
        patch = frame[
            max(0, top) : min(height, bottom), max(0, left) : min(width, right)
        ]
        if patch.size == 0:
            raise ValueError("Empty interaction region")
        patch_height, patch_width = patch.shape[:2]
        scale = size / max(patch_height, patch_width)
        resized = cv2.resize(
            patch,
            (max(1, round(patch_width * scale)), max(1, round(patch_height * scale))),
        )
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        offset_y, offset_x = (size - resized.shape[0]) // 2, (
            size - resized.shape[1]
        ) // 2
        canvas[
            offset_y : offset_y + resized.shape[0],
            offset_x : offset_x + resized.shape[1],
        ] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        images.append(canvas)
    return np.stack(images)


def clip_tensor(clips, device="cpu"):
    tensor = torch.from_numpy(np.asarray(clips)).to(device=device, dtype=torch.float32)
    return (tensor.permute(0, 4, 1, 2, 3) / 255.0 - 0.45) / 0.225


def augment_handheld(clips, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    result = np.empty_like(clips)
    height, width = clips.shape[2:4]
    shift = rng.uniform(-0.025, 0.025, size=2) * np.array([width, height])
    angle = float(rng.uniform(-3, 3))
    gain = float(rng.uniform(0.85, 1.15))
    for frame_index in range(clips.shape[1]):
        progress = frame_index / max(1, clips.shape[1] - 1) - 0.5
        transform = cv2.getRotationMatrix2D(
            (width / 2, height / 2), angle * progress, 1.0
        )
        transform[:, 2] += shift * progress
        for region_index in range(clips.shape[0]):
            warped = cv2.warpAffine(
                clips[region_index, frame_index],
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            result[region_index, frame_index] = np.clip(
                warped.astype(np.float32) * gain, 0, 255
            ).astype(np.uint8)
    return result


def motion_score(frames):
    greys = np.stack(
        [
            cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
            for frame in frames
        ]
    ).astype(np.float32)
    return float(np.abs(np.diff(greys, axis=0)).mean() / 255.0)
