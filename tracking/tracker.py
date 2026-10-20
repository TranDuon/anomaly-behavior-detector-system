"""
tracking/tracker.py — DeepSORT multi-object tracker.

Wraps deep-sort-realtime to produce stable, ID-consistent track objects
from per-frame YOLO detections.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.config import settings
from detection.detector import DetectionResult


@dataclass
class Track:
    """
    Represents one confirmed track at the current frame.

    bbox_xyxy : absolute pixel [x1,y1,x2,y2]
    age       : frames since track was first confirmed
    """
    track_id: int
    bbox_xyxy: List[float]
    confidence: float
    age: int = 0
    is_anomaly: bool = False
    anomaly_type: Optional[str] = None
    anomaly_score: float = 0.0

    # --- Derived helpers ---

    @property
    def centroid(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def wh(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        return x2 - x1, y2 - y1

    @property
    def bbox_ltwh(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        return [x1, y1, x2 - x1, y2 - y1]


class PersonTracker:
    """
    Wraps DeepSort from deep-sort-realtime.

    Usage:
        tracker = PersonTracker()
        tracks = tracker.update(detections, frame)
    """

    def __init__(self):
        self._tracker = None
        self._track_ages: dict[int, int] = {}

    def _init_tracker(self):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self._tracker = DeepSort(
            max_age=settings.deepsort_max_age,
            n_init=settings.deepsort_n_init,
            nms_max_overlap=settings.deepsort_nms_max_overlap,
            max_cosine_distance=settings.deepsort_max_cosine_distance,
            max_iou_distance=settings.deepsort_max_iou_distance,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def update(
        self,
        detections: List[DetectionResult],
        frame: np.ndarray,
    ) -> List[Track]:
        """
        Feed current-frame detections into DeepSORT.

        Returns a list of confirmed Track objects.
        """
        if self._tracker is None:
            self._init_tracker()

        # DeepSORT input format: list of ([l, t, w, h], conf, class_id_str)
        raw = [
            (d.bbox_ltwh, d.confidence, str(d.class_id))
            for d in detections
        ]

        ds_tracks = self._tracker.update_tracks(raw, frame=frame)

        tracks: List[Track] = []
        active_ids = set()

        for t in ds_tracks:
            if not t.is_confirmed():
                continue

            tid = int(t.track_id)
            active_ids.add(tid)
            self._track_ages[tid] = self._track_ages.get(tid, 0) + 1

            ltrb = t.to_ltrb()
            conf = t.det_conf if t.det_conf is not None else 0.5

            tracks.append(
                Track(
                    track_id=tid,
                    bbox_xyxy=list(ltrb),
                    confidence=float(conf),
                    age=self._track_ages[tid],
                )
            )

        # Prune ages for dead tracks
        dead = set(self._track_ages.keys()) - active_ids
        for d in dead:
            del self._track_ages[d]

        return tracks

    def reset(self):
        """Reset tracker state (e.g., between videos)."""
        self._tracker = None
        self._track_ages.clear()
