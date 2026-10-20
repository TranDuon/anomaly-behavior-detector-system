"""
api/processor.py — VideoProcessor orchestrates the full pipeline.

Pipeline per frame:
  BGR frame → YOLOv8 detection → DeepSORT tracking → Anomaly detection
           → Annotated frame (JPEG bytes) + events

The processor runs in a background thread. Consumers read annotated
frames from `frame_queue` and alerts from the event logger callbacks.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Callable, Optional

import cv2
import numpy as np

from core.config import settings
from core.logger import event_logger, AnomalyEvent
from core.utils import draw_bbox, draw_overlay_stats, encode_jpeg
from detection.detector import PersonDetector
from tracking.tracker import PersonTracker
from anomaly.detector import AnomalyDetector


# ── Annotation helpers ───────────────────────────────────────────────────────

ANOMALY_LABELS = {
    "running":   "RUNNING",
    "fighting":  "FIGHT",
    "loitering": "LOITER",
    "unknown":   "ANOMALY",
}


def _annotate(frame: np.ndarray, tracks, events: list, frame_no: int) -> np.ndarray:
    """Draw bounding boxes, labels, and a HUD overlay on the frame."""
    anomaly_ids = {e.track_id for e in events}

    for track in tracks:
        color = settings.anomaly_color if track.track_id in anomaly_ids else settings.normal_color
        label_parts = [f"#{track.track_id}"]
        if track.is_anomaly and track.anomaly_type:
            label_parts.append(ANOMALY_LABELS.get(track.anomaly_type, "!"))
            label_parts.append(f"{track.anomaly_score:.2f}")
        label = " ".join(label_parts)
        draw_bbox(frame, track.bbox_xyxy, label, color, settings.box_thickness)

    # Flash red border on anomaly frames
    if events:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), settings.anomaly_color, 6)

    draw_overlay_stats(
        frame,
        {
            "Frame":   frame_no,
            "Tracks":  len(tracks),
            "Alerts":  len(events),
        },
    )
    return frame


# ── Main processor class ─────────────────────────────────────────────────────

class VideoProcessor:
    """
    Manages the full detection → tracking → anomaly pipeline.

    Usage:
        proc = VideoProcessor()
        proc.start("path/to/video.mp4")          # starts background thread
        for jpeg_bytes in proc.iter_frames():    # MJPEG consumer
            ...
        proc.stop()
    """

    def __init__(self, alert_callback: Optional[Callable] = None):
        self.detector  = PersonDetector()
        self.tracker   = PersonTracker()
        self.anomaly   = AnomalyDetector()

        self.frame_queue: Queue[Optional[bytes]] = Queue(maxsize=60)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._frame_count   = 0
        self._alert_count   = 0
        self._track_count   = 0
        self._source: Optional[str] = None

        if alert_callback:
            event_logger.register_callback(alert_callback)

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self, source: str | int) -> None:
        """Start background processing thread. `source` is a path or camera index."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                self.stop()
            self._stop_event.clear()
            self._source = str(source)
            self._thread = threading.Thread(
                target=self._run, args=(source,), daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        # Drain queue and push sentinel
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        self.frame_queue.put(None)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def iter_frames(self):
        """Generator that yields JPEG bytes, terminates on None sentinel."""
        while True:
            item = self.frame_queue.get(timeout=10)
            if item is None:
                break
            yield item

    def status(self) -> dict:
        return {
            "running":      self.is_running(),
            "source":       self._source,
            "frame_count":  self._frame_count,
            "alert_count":  self._alert_count,
            "track_count":  self._track_count,
            "model_loaded": self.anomaly._model_loaded,
        }

    # ── Pipeline ─────────────────────────────────────────────────────────────

    def _run(self, source) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[Processor] Cannot open source: {source}")
            self.frame_queue.put(None)
            return

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        delay  = max(1.0 / settings.stream_fps_cap, 1.0 / (fps + 1))

        self.tracker.reset()

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    # Video file ended — loop or stop
                    if isinstance(source, str) and Path(source).is_file():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop file
                        continue
                    break

                self._frame_count += 1
                t0 = time.time()

                annotated, events = self._process_frame(frame)

                self._alert_count += len(events)
                for ev in events:
                    event_logger.log(ev)

                # Encode & push (drop frame if queue full to avoid memory build-up)
                jpeg = encode_jpeg(annotated, settings.jpeg_quality)
                try:
                    self.frame_queue.put_nowait(jpeg)
                except Exception:
                    pass  # queue full → drop frame

                # Throttle to stream_fps_cap
                elapsed = time.time() - t0
                sleep   = delay - elapsed
                if sleep > 0:
                    time.sleep(sleep)

        finally:
            cap.release()
            self.frame_queue.put(None)  # signal end-of-stream

    def _process_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        dets   = self.detector.detect(frame)
        tracks = self.tracker.update(dets, frame)

        events = self.anomaly.update(
            tracks, frame_hw=(h, w), frame_number=self._frame_count
        )

        self._track_count = len(tracks)
        annotated = _annotate(frame.copy(), tracks, events, self._frame_count)
        return annotated, events
