import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch

from fight.vision import (
    ClipConfig,
    DetectorConfig,
    build_clip_inputs,
    clip_tensor,
)


class EventGate:
    def __init__(
        self,
        threshold=0.8,
        release_threshold=0.5,
        confirm_seconds=0.8,
        release_seconds=1.0,
    ):
        self.threshold = threshold
        self.release_threshold = release_threshold
        self.confirm_seconds = confirm_seconds
        self.release_seconds = release_seconds
        self.pending_since = None
        self.low_since = None
        self.active = False

    def update(self, score, timestamp):
        event = None
        if score >= self.threshold:
            self.low_since = None
            if self.pending_since is None:
                self.pending_since = timestamp
            if (
                not self.active
                and timestamp - self.pending_since >= self.confirm_seconds
            ):
                self.active = True
                event = "started"
        else:
            self.pending_since = None
            if self.active and score < self.release_threshold:
                if self.low_since is None:
                    self.low_since = timestamp
                if timestamp - self.low_since >= self.release_seconds:
                    self.active = False
                    self.low_since = None
                    event = "ended"
            else:
                self.low_since = None
        state = (
            "fighting"
            if self.active
            else ("verifying" if self.pending_since is not None else "normal")
        )
        return state, event


class FightRuntime:
    def __init__(
        self,
        models_dir=Path("models"),
        allow_pilot=False,
        predictor=None,
        config=None,
        detector=None,
    ):
        self.config = config or ClipConfig()
        self.predictor = predictor
        self.detector = detector
        self.model_ready = predictor is not None
        self.message = "Chưa có checkpoint đánh nhau được kiểm thử."
        self.metadata = {}
        self.detector_config = DetectorConfig()
        self.gate = EventGate()
        model_path = Path(models_dir) / "fight_model.ts"
        metadata_path = Path(models_dir) / "fight_runtime.json"
        if predictor is None and model_path.exists() and metadata_path.exists():
            try:
                self.metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8-sig")
                )
                if self.metadata.get("schema_version") != 1:
                    raise ValueError("Unsupported runtime schema")
                self.config = ClipConfig(**self.metadata["clip"])
                self.detector_config = DetectorConfig(
                    **self.metadata.get("person_detector", {})
                )
                if (
                    not self.metadata.get("approved_for_demo", False)
                    and not allow_pilot
                ):
                    raise ValueError(
                        "Checkpoint pilot chưa đạt cổng kiểm thử; chỉ bật chế độ thử nghiệm khi cần."
                    )
                threshold = float(self.metadata["threshold"])
                if not 0 < threshold < 1:
                    raise ValueError("Invalid threshold")
                self.predictor = torch.jit.load(
                    str(model_path), map_location="cpu"
                ).eval()
                self.gate = EventGate(threshold, threshold * 0.7)
                self.model_ready = True
                self.message = (
                    "Checkpoint thử nghiệm"
                    if not self.metadata.get("approved_for_demo")
                    else "Checkpoint đã qua cổng kiểm thử"
                )
            except Exception as error:
                self.message = str(error)
                self.model_ready = False
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=self.config.frames)
        self.sample_times = deque(maxlen=self.config.frames)
        self.next_sample = None
        self.last_input = None
        self.last_prediction = None
        self.gate = EventGate(
            self.gate.threshold,
            self.gate.release_threshold,
            self.gate.confirm_seconds,
            self.gate.release_seconds,
        )
        self.result = {
            "state": "warming_up" if self.model_ready else "unavailable",
            "score": None,
            "regions": [],
            "people": 0,
            "event": None,
            "message": self.message,
        }

    def push(self, frame, timestamp, predict=True):
        if not math.isfinite(timestamp) or (
            self.last_input is not None and timestamp < self.last_input
        ):
            raise ValueError("Frame timestamps must be finite and monotonic")
        self.result = {**self.result, "event": None}
        if not self.model_ready:
            return dict(self.result)
        if self.last_input is not None and timestamp - self.last_input > max(
            1.0, 3 / self.config.fps
        ):
            was_active = self.gate.active
            self.reset()
            self.result["message"] = "Nguồn hình bị gián đoạn; đang thu lại clip."
            self.result["event"] = "ended" if was_active else None
        self.last_input = timestamp
        if self.next_sample is not None and timestamp + 1e-6 < self.next_sample:
            return dict(self.result)
        if self.next_sample is None:
            self.next_sample = timestamp
        while self.next_sample <= timestamp + 1e-6:
            self.next_sample += 1 / self.config.fps
        self.samples.append(frame.copy())
        self.sample_times.append(timestamp)
        if not predict:
            return dict(self.result)
        if len(self.samples) < self.config.frames:
            return dict(self.result)
        expected_span = (self.config.frames - 1) / self.config.fps
        if self.sample_times[-1] - self.sample_times[0] > expected_span * 1.5:
            was_active = self.gate.active
            self.gate = EventGate(self.gate.threshold, self.gate.release_threshold)
            self.result = {
                **self.result,
                "state": "insufficient",
                "score": None,
                "event": "ended" if was_active else None,
                "message": "FPS nguồn hình quá thấp hoặc mất nhiều frame; chưa đủ dữ liệu theo thời gian.",
            }
            return dict(self.result)
        if (
            self.last_prediction is not None
            and timestamp - self.last_prediction + 1e-6 < self.config.stride_seconds
        ):
            return dict(self.result)
        self.last_prediction = timestamp
        frames = list(self.samples)
        if self.detector is None:
            from detection.detector import PersonDetector

            self.detector = PersonDetector(
                **self.detector_config.to_dict(), device="cpu"
            )
        clips, regions, people, content_bounds, people_counts = build_clip_inputs(
            frames, self.detector, self.config
        )
        with torch.inference_mode():
            probabilities = (
                torch.sigmoid(self.predictor(clip_tensor(clips)))
                .cpu()
                .numpy()
                .reshape(-1)
            )
        if not np.isfinite(probabilities).all():
            raise ValueError("Model returned a non-finite prediction")
        selected = int(np.argmax(probabilities))
        score = float(probabilities[selected])
        if people == 0 and not self.gate.active:
            state = "insufficient"
            self.gate.pending_since = None
            event = None
        else:
            state, event = self.gate.update(score, timestamp)
        self.result = {
            "state": state,
            "score": score,
            "regions": [list(regions[selected])],
            "people": people,
            "event": event,
            "message": self.message,
        }
        return dict(self.result)
