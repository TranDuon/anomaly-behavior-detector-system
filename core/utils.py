"""
core/utils.py — Shared utility helpers.
"""

import cv2
import numpy as np
from typing import Tuple, List


def letterbox_resize(frame: np.ndarray, target: int = 640) -> Tuple[np.ndarray, float, Tuple]:
    """Resize keeping aspect ratio, pad to square. Returns (resized, scale, pad)."""
    h, w = frame.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    pad_h = (target - nh) // 2
    pad_w = (target - nw) // 2
    padded = cv2.copyMakeBorder(resized, pad_h, target - nh - pad_h,
                                pad_w, target - nw - pad_w,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, (pad_w, pad_h)


def xyxy_to_ltwh(xyxy: List[float]) -> List[float]:
    """Convert [x1, y1, x2, y2] → [left, top, width, height]."""
    x1, y1, x2, y2 = xyxy
    return [x1, y1, x2 - x1, y2 - y1]


def ltwh_to_xyxy(ltwh: List[float]) -> List[float]:
    """Convert [left, top, width, height] → [x1, y1, x2, y2]."""
    l, t, w, h = ltwh
    return [l, t, l + w, t + h]


def normalise_bbox(bbox_xyxy: List[float], frame_w: int, frame_h: int) -> List[float]:
    """Return bbox coordinates normalised to [0, 1]."""
    x1, y1, x2, y2 = bbox_xyxy
    return [x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h]


def centroid(bbox_xyxy: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2


def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def draw_bbox(
    frame: np.ndarray,
    bbox_xyxy: List[float],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw a bounding box + label on frame (in-place)."""
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label background
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, label,
        (x1 + 2, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


def draw_overlay_stats(frame: np.ndarray, stats: dict) -> np.ndarray:
    """Draw a semi-transparent stats overlay in the top-left corner."""
    lines = [f"{k}: {v}" for k, v in stats.items()]
    padding = 8
    line_h = 20
    overlay_h = len(lines) * line_h + padding * 2
    overlay_w = 220

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (overlay_w, overlay_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(
            frame, line,
            (padding, padding + (i + 1) * line_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (200, 255, 200), 1, cv2.LINE_AA,
        )
    return frame


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
