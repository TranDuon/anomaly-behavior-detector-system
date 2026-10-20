"""
core/logger.py — Structured event logger.

Writes anomaly events to a .jsonl file and keeps an in-memory ring-buffer
for the API to serve without disk reads.
"""

import json
import time
import threading
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import List, Optional
from pathlib import Path

from core.config import settings


@dataclass
class AnomalyEvent:
    track_id: int
    anomaly_type: str                   # "running" | "fighting" | "loitering" | "unknown"
    confidence: float                   # 0-1
    timestamp: float = field(default_factory=time.time)
    frame_number: int = 0
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2] normalised
    reconstruction_error: float = 0.0
    rule_triggered: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)
        )
        return d


class EventLogger:
    """
    Thread-safe event logger.

    Usage:
        logger = EventLogger()
        logger.log(event)
        recent = logger.get_recent(50)
    """

    _MAX_BUFFER = 500          # keep last N events in memory

    def __init__(self, log_path: Path = settings.event_log_path):
        self._path = log_path
        self._buffer: deque[dict] = deque(maxlen=self._MAX_BUFFER)
        self._lock = threading.Lock()
        self._alert_callbacks: List = []

        # Open log file (append mode)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "a", buffering=1)  # line-buffered

    # ── Public API ──────────────────────────────────────────────────────────

    def log(self, event: AnomalyEvent) -> None:
        d = event.to_dict()
        with self._lock:
            self._buffer.append(d)
            self._fh.write(json.dumps(d) + "\n")
        self._fire_callbacks(d)

    def get_recent(self, n: int = 100) -> List[dict]:
        with self._lock:
            items = list(self._buffer)
        return items[-n:]

    def register_callback(self, fn) -> None:
        """Register a callable(event_dict) invoked on every new event."""
        self._alert_callbacks.append(fn)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    # ── Internal ────────────────────────────────────────────────────────────

    def _fire_callbacks(self, event_dict: dict) -> None:
        for cb in self._alert_callbacks:
            try:
                cb(event_dict)
            except Exception:
                pass  # never let a bad callback crash the logger


# Module-level singleton
event_logger = EventLogger()
