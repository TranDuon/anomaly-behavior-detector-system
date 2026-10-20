"""
anomaly/detector.py — Per-frame anomaly detection engine.

Combines:
  1. LSTM Autoencoder (if a trained model exists)
  2. Rule-based heuristics (always active, used for type classification)

Rule logic:
  running   → high speed sustained over recent frames
  loitering → low displacement over many frames
  fighting  → two+ tracks in close proximity both with elevated speed
"""

from __future__ import annotations
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch

from core.config import settings
from core.logger import AnomalyEvent
from anomaly.features import TrackFeatureBuffer
from anomaly.model import LSTMAutoencoder
from tracking.tracker import Track


class AnomalyDetector:
    """
    Usage:
        detector = AnomalyDetector()
        events = detector.update(tracks, frame_hw=(H, W), frame_number=n)
    """

    def __init__(self):
        self._model: Optional[LSTMAutoencoder] = None
        self._model_loaded: bool = False
        self._features = TrackFeatureBuffer()
        self._device = settings.yolo_device

        # Per-track loiter counter: {track_id: (frames_in_zone, anchor_pos)}
        self._loiter: Dict[int, dict] = {}

        self._try_load_model()

    # ── Public API ──────────────────────────────────────────────────────────

    def update(
        self,
        tracks: List[Track],
        frame_hw: Tuple[int, int],
        frame_number: int = 0,
    ) -> List[AnomalyEvent]:
        """
        Process one frame of tracks. Returns a list of AnomalyEvent (may be empty).
        Also annotates each Track in-place (track.is_anomaly, track.anomaly_type).
        """
        frame_h, frame_w = frame_hw
        active_ids = {t.track_id for t in tracks}

        # Update feature buffers
        for track in tracks:
            self._features.update(track.track_id, track.bbox_xyxy, frame_w, frame_h)

        self._features.prune(active_ids)

        events: List[AnomalyEvent] = []

        for track in tracks:
            recon_err, anomaly_score, is_anomaly_model = self._score_model(track.track_id)
            rule_type, rule_triggered, rule_conf = self._rule_check(
                track, tracks, frame_w, frame_h
            )

            is_anomaly = is_anomaly_model or (rule_type is not None)

            if not is_anomaly:
                track.is_anomaly = False
                continue

            anomaly_type = rule_type if rule_type else "unknown"
            confidence   = max(anomaly_score, rule_conf)

            track.is_anomaly     = True
            track.anomaly_type   = anomaly_type
            track.anomaly_score  = confidence

            events.append(
                AnomalyEvent(
                    track_id=track.track_id,
                    anomaly_type=anomaly_type,
                    confidence=round(confidence, 3),
                    frame_number=frame_number,
                    bbox=self._norm_bbox(track.bbox_xyxy, frame_w, frame_h),
                    reconstruction_error=round(recon_err, 5),
                    rule_triggered=rule_triggered,
                )
            )

        return events

    # ── Model scoring ────────────────────────────────────────────────────────

    def _score_model(self, track_id: int) -> Tuple[float, float, bool]:
        """
        Returns (reconstruction_error, normalised_score, is_anomaly).
        Falls back to (0, 0, False) if model unavailable or buffer too short.
        """
        if not self._model_loaded:
            return 0.0, 0.0, False

        seq = self._features.get_sequence(track_id)
        if seq is None:
            return 0.0, 0.0, False

        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            x_hat = self._model(x)
            err = float(LSTMAutoencoder.reconstruction_error(x, x_hat).item())

        # Normalise score: sigmoid-like mapping around threshold
        score = float(1 / (1 + np.exp(-10 * (err - settings.anomaly_threshold))))
        is_anomaly = err > settings.anomaly_threshold
        return err, score, is_anomaly

    # ── Rule-based heuristics ────────────────────────────────────────────────

    def _rule_check(
        self,
        track: Track,
        all_tracks: List[Track],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Returns (anomaly_type | None, rule_name | None, confidence 0-1).
        """
        feat = self._features.get_current_features(track.track_id)
        if feat is None:
            return None, None, 0.0

        speed = float(feat[6])

        # 1. Running — sustained high speed
        if speed > settings.speed_run_threshold:
            conf = min(1.0, speed / (settings.speed_run_threshold * 2))
            return "running", "speed_threshold", conf

        # 2. Loitering — low displacement over many frames
        loiter_result = self._check_loiter(track, frame_w, frame_h)
        if loiter_result:
            return "loitering", "loiter_displacement", 0.7

        # 3. Fighting — two close, fast-moving tracks
        fight_result = self._check_fight(track, all_tracks, frame_w, frame_h)
        if fight_result:
            return "fighting", "proximity_speed", 0.85

        return None, None, 0.0

    def _check_loiter(self, track: Track, frame_w: int, frame_h: int) -> bool:
        diag = np.sqrt(frame_w ** 2 + frame_h ** 2) + 1e-9
        cx, cy = track.centroid

        state = self._loiter.get(track.track_id)
        if state is None:
            self._loiter[track.track_id] = {"anchor": (cx, cy), "count": 1}
            return False

        anchor = state["anchor"]
        dist   = np.sqrt((cx - anchor[0]) ** 2 + (cy - anchor[1]) ** 2) / diag

        if dist < settings.loiter_displacement:
            state["count"] += 1
        else:
            # Moved — reset anchor
            state["anchor"] = (cx, cy)
            state["count"]  = 1

        return state["count"] >= settings.loiter_frames

    def _check_fight(
        self,
        track: Track,
        all_tracks: List[Track],
        frame_w: int,
        frame_h: int,
    ) -> bool:
        diag  = np.sqrt(frame_w ** 2 + frame_h ** 2) + 1e-9
        feat1 = self._features.get_current_features(track.track_id)
        if feat1 is None or float(feat1[6]) < settings.fight_speed_threshold:
            return False

        cx1, cy1 = track.centroid
        for other in all_tracks:
            if other.track_id == track.track_id:
                continue
            feat2 = self._features.get_current_features(other.track_id)
            if feat2 is None:
                continue
            cx2, cy2 = other.centroid
            dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / diag
            if (
                dist < settings.fight_proximity
                and float(feat2[6]) >= settings.fight_speed_threshold
            ):
                return True
        return False

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _try_load_model(self) -> None:
        path = settings.anomaly_model_path
        if path.exists():
            try:
                self._model = LSTMAutoencoder.load(path, device=self._device)
                self._model_loaded = True
                print(f"[AnomalyDetector] Loaded model from {path}")
            except Exception as e:
                print(f"[AnomalyDetector] Model load failed: {e} — using rules only")
        else:
            print("[AnomalyDetector] No trained model found — using rule-based detection only.")

    @staticmethod
    def _norm_bbox(bbox_xyxy, frame_w, frame_h) -> List[float]:
        x1, y1, x2, y2 = bbox_xyxy
        return [x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h]
