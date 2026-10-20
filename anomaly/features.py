"""
anomaly/features.py — Per-track feature extraction.

For each frame a track is alive, we compute a 9-dimensional feature vector.
The feature buffer maintains a rolling window of length seq_len, which is
fed to the LSTM Autoencoder.

Feature vector (9-dim):
  [0] cx_norm       — centroid x / frame_width
  [1] cy_norm       — centroid y / frame_height
  [2] w_norm        — bbox width / frame_width
  [3] h_norm        — bbox height / frame_height
  [4] vx            — delta cx (pixels/frame, normalised by frame diag)
  [5] vy            — delta cy
  [6] speed         — sqrt(vx² + vy²)
  [7] acceleration  — delta speed
  [8] aspect_ratio  — w/h (clamped)
"""

from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.config import settings


class TrackFeatureBuffer:
    """
    Maintains a rolling feature buffer for every active track.

    buffer[track_id] → deque of feature vectors, maxlen = seq_len
    """

    def __init__(self):
        self._buffers: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=settings.seq_len)
        )
        self._prev_state: Dict[int, dict] = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def update(
        self,
        track_id: int,
        bbox_xyxy: List[float],
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Compute feature vector and push it to the track's buffer."""
        feat = self._compute_feature(track_id, bbox_xyxy, frame_w, frame_h)
        self._buffers[track_id].append(feat)

    def get_sequence(self, track_id: int) -> Optional[np.ndarray]:
        """
        Return the feature sequence as a (seq_len, feature_dim) array.

        Returns None if buffer is shorter than min_track_frames.
        Pads with the first frame if buffer < seq_len.
        """
        buf = self._buffers.get(track_id)
        if buf is None or len(buf) < settings.min_track_frames:
            return None

        seq = list(buf)
        # Left-pad with the earliest frame to reach seq_len
        if len(seq) < settings.seq_len:
            pad = [seq[0]] * (settings.seq_len - len(seq))
            seq = pad + seq

        return np.array(seq, dtype=np.float32)  # (seq_len, feature_dim)

    def prune(self, active_ids: set) -> None:
        """Remove buffers for tracks that are no longer alive."""
        dead = set(self._buffers.keys()) - active_ids
        for tid in dead:
            del self._buffers[tid]
            self._prev_state.pop(tid, None)

    def buffer_len(self, track_id: int) -> int:
        return len(self._buffers.get(track_id, []))

    def get_current_features(self, track_id: int) -> Optional[np.ndarray]:
        """Return the most recent feature vector for this track."""
        buf = self._buffers.get(track_id)
        if not buf:
            return None
        return buf[-1]

    # ── Internal ────────────────────────────────────────────────────────────

    def _compute_feature(
        self,
        track_id: int,
        bbox_xyxy: List[float],
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = x2 - x1
        bh = max(y2 - y1, 1e-6)

        cx_n = cx / frame_w
        cy_n = cy / frame_h
        w_n  = bw / frame_w
        h_n  = bh / frame_h

        diag = np.sqrt(frame_w ** 2 + frame_h ** 2) + 1e-9

        prev = self._prev_state.get(track_id)
        if prev is None:
            vx, vy, speed, accel = 0.0, 0.0, 0.0, 0.0
        else:
            vx    = (cx - prev["cx"]) / diag
            vy    = (cy - prev["cy"]) / diag
            speed = float(np.sqrt(vx ** 2 + vy ** 2))
            accel = float(abs(speed - prev["speed"]))

        ar = float(np.clip(bw / bh, 0.1, 5.0))

        self._prev_state[track_id] = {"cx": cx, "cy": cy, "speed": speed}

        return np.array(
            [cx_n, cy_n, w_n, h_n, vx, vy, speed, accel, ar],
            dtype=np.float32,
        )
