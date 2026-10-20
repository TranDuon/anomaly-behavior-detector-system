"""
detection/detector.py — YOLOv8 person detector.

Returns a list of DetectionResult objects for every frame.
Only persons (COCO class 0) are returned.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np

from core.config import settings


@dataclass
class DetectionResult:
    """Single detected person in one frame."""
    bbox_xyxy: List[float]      # absolute pixel coords [x1, y1, x2, y2]
    confidence: float
    class_id: int = 0
    class_name: str = "person"

    @property
    def bbox_ltwh(self) -> List[float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return [x1, y1, x2 - x1, y2 - y1]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x2 - x1) * (y2 - y1)


class PersonDetector:
    """
    Thin wrapper around YOLOv8.

    Lazy-loads the model on first call so it doesn't block import.

    Usage:
        detector = PersonDetector()
        results = detector.detect(frame)   # frame: BGR np.ndarray
    """

    def __init__(self):
        self._model = None

    def _load(self):
        # Deferred import — ultralytics is heavy
        from ultralytics import YOLO
        self._model = YOLO(settings.yolo_model)
        self._model.to(settings.yolo_device)

    # ── Public API ──────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Run inference on a single BGR frame.

        Returns a list of DetectionResult, one per detected person.
        Filters by confidence and class (person only).
        """
        if self._model is None:
            self._load()

        results = self._model.predict(
            frame,
            conf=settings.yolo_conf_threshold,
            iou=settings.yolo_iou_threshold,
            imgsz=settings.yolo_imgsz,
            device=settings.yolo_device,
            classes=[settings.person_class_id],
            verbose=False,
        )

        detections: List[DetectionResult] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id != settings.person_class_id:
                    continue
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    DetectionResult(
                        bbox_xyxy=[x1, y1, x2, y2],
                        confidence=conf,
                        class_id=cls_id,
                    )
                )
        return detections

    def warmup(self, size: int = 320) -> None:
        """Run a dummy inference to pre-load model weights."""
        if self._model is None:
            self._load()
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        self.detect(dummy)
