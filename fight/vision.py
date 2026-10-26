from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class ClipConfig:
    frames: int = 16
    fps: float = 8.0
    size: int = 182
    stride_seconds: float = 0.5
    max_regions: int = 3
    version: int = 1

    def to_dict(self):
        return asdict(self)


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


def motion_score(frames):
    greys = np.stack(
        [
            cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
            for frame in frames
        ]
    ).astype(np.float32)
    return float(np.abs(np.diff(greys, axis=0)).mean() / 255.0)
