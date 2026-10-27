import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from fight.vision import (
    ClipConfig,
    DetectorConfig,
    DetectionCache,
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
        max_observation_gap=1.5,
    ):
        values = (
            threshold,
            release_threshold,
            confirm_seconds,
            release_seconds,
            max_observation_gap,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or not 0 < release_threshold <= threshold < 1
            or min(confirm_seconds, release_seconds) < 0
            or max_observation_gap <= 0
        ):
            raise ValueError("Invalid event gate configuration")
        self.threshold = threshold
        self.release_threshold = release_threshold
        self.confirm_seconds = confirm_seconds
        self.release_seconds = release_seconds
        self.max_observation_gap = max_observation_gap
        self.last_observation = None
        self.pending_since = None
        self.low_since = None
        self.active = False

    def update(self, score, timestamp):
        if (
            not math.isfinite(score)
            or not 0 <= score <= 1
            or not math.isfinite(timestamp)
            or (self.last_observation is not None and timestamp < self.last_observation)
        ):
            raise ValueError("Finite probabilities and monotonic timestamps required")
        stale = (
            self.last_observation is not None
            and timestamp - self.last_observation > self.max_observation_gap
        )
        self.last_observation = timestamp
        if stale:
            was_active = self.active
            self.pending_since = self.low_since = None
            self.active = False
            if was_active:
                return "insufficient", "ended"
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

    def fresh(self):
        return EventGate(
            self.threshold,
            self.release_threshold,
            self.confirm_seconds,
            self.release_seconds,
            self.max_observation_gap,
        )


class FightRuntime:
    def __init__(
        self,
        models_dir=Path("models"),
        allow_pilot=False,
        predictor=None,
        config=None,
        detector=None,
        pose_detector=None,
        input_kind="rgb",
    ):
        self.config = config or ClipConfig()
        self.predictor = predictor
        self.detector = detector
        self.model_ready = predictor is not None
        self.pilot_required = False
        self.message = "Chưa có checkpoint đánh nhau được kiểm thử."
        self.metadata = {}
        self.models_dir = Path(models_dir)
        self.input_kind = input_kind
        self.pose_detector = pose_detector
        self.pose_config = None
        self.motion_config = None
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
                self.input_kind = self.metadata.get("input_kind", "rgb")
                if self.input_kind not in {"rgb", "rgb_pose", "rgb_pose_motion"}:
                    raise ValueError("Unsupported model inputs")
                if self.input_kind in {"rgb_pose", "rgb_pose_motion"}:
                    from fight.pose import PoseConfig

                    self.pose_config = PoseConfig(**self.metadata["pose"])
                    if not (self.models_dir / self.pose_config.model_name).is_file():
                        raise ValueError("Pose weights missing from runtime bundle")
                if self.input_kind == "rgb_pose_motion":
                    from fight.pair_motion import MotionConfig

                    self.motion_config = MotionConfig(**self.metadata["motion"])
                    if self.config.frames != 16 or self.config.fps != 12:
                        raise ValueError("Motion runtime requires 16 frames at 12fps")
                self.detector_config = DetectorConfig(
                    **self.metadata.get("person_detector", {})
                )
                if (
                    not self.metadata.get("approved_for_demo", False)
                    and not allow_pilot
                ):
                    self.pilot_required = True
                    raise ValueError(
                        "Đã có checkpoint thử nghiệm; hãy tích Cho phép checkpoint pilot để chạy. Chưa nghiệm thu demo."
                    )
                threshold = float(self.metadata["threshold"])
                if not 0 < threshold < 1:
                    raise ValueError("Invalid threshold")
                self.predictor = torch.jit.load(
                    str(model_path), map_location="cpu"
                ).eval()
                self.gate = EventGate(
                    threshold, threshold * 0.7, **self.metadata.get("event_gate", {})
                )
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
        if self.pose_detector is not None and hasattr(self.pose_detector, "cache"):
            self.pose_detector.cache.clear()
        self.detection_cache = None
        self.samples = deque(maxlen=self.config.frames)
        self.sample_times = deque(maxlen=self.config.frames)
        self.next_sample = None
        self.last_input = None
        self.last_prediction = None
        self.gate = self.gate.fresh()
        self.result = {
            "state": "warming_up" if self.model_ready else "unavailable",
            "score": None,
            "regions": [],
            "person_boxes": [],
            "visible_people": None,
            "people": 0,
            "event": None,
            "message": self.message,
            "prediction_timestamp": None,
            "inference_seconds": None,
        }

    def push(self, frame, timestamp, predict=True, *, sampled=False):
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
        if (
            not sampled
            and self.next_sample is not None
            and timestamp + 1e-6 < self.next_sample
        ):
            return dict(self.result)
        if self.next_sample is None:
            self.next_sample = timestamp
        while self.next_sample <= timestamp + 1e-6:
            self.next_sample += 1 / self.config.fps
        self.samples.append(frame.copy())
        self.sample_times.append(timestamp)
        if self.detection_cache is not None:
            self.detection_cache.retain(self.samples)
        if self.pose_detector is not None:
            self.pose_detector.retain(self.samples)
        if not predict:
            return dict(self.result)
        if len(self.samples) < self.config.frames:
            return dict(self.result)
        expected_span = (self.config.frames - 1) / self.config.fps
        if self.sample_times[-1] - self.sample_times[0] > expected_span * 1.5:
            was_active = self.gate.active
            self.gate = self.gate.fresh()
            self.result = {
                **self.result,
                "state": "insufficient",
                "score": None,
                "person_boxes": [],
                "visible_people": None,
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
        inference_started = time.monotonic()
        frames = list(self.samples)
        if self.input_kind in {"rgb_pose", "rgb_pose_motion"}:
            self._ensure_pose()
        if self.detector is None:
            from detection.detector import PersonDetector

            self.detector = PersonDetector(
                **self.detector_config.to_dict(), device="cpu"
            )
        if self.detection_cache is None:
            self.detection_cache = DetectionCache(self.detector)
        clips, regions, people, content_bounds, people_counts, person_boxes = (
            build_clip_inputs(
                frames, self.detection_cache, self.config, include_person_boxes=True
            )
        )
        with torch.inference_mode():
            if self.input_kind in {"rgb_pose", "rgb_pose_motion"}:
                from fight.pose import pose_inputs

                poses = (
                    torch.zeros(len(clips), 8, 2, 17, 3)
                    if self.metadata.get("use_pose") is False
                    else torch.from_numpy(
                        pose_inputs(frames, regions, self.pose_detector)
                    )
                )
                if self.input_kind == "rgb_pose_motion":
                    from fight.pair_motion import motion_inputs

                    motion = (
                        torch.zeros(len(clips), 2, 15, 32, 32)
                        if self.metadata.get("use_motion") is False
                        else torch.from_numpy(motion_inputs(clips, self.motion_config))
                    )
                    logits = self.predictor(clip_tensor(clips), poses, motion)
                else:
                    logits = self.predictor(clip_tensor(clips), poses)
            else:
                logits = self.predictor(clip_tensor(clips))
            probabilities = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        if not np.isfinite(probabilities).all():
            raise ValueError("Model returned a non-finite prediction")
        selected = int(np.argmax(probabilities))
        score = float(probabilities[selected])
        if people == 0:
            was_active = self.gate.active
            state = "insufficient"
            self.gate = self.gate.fresh()
            event = "ended" if was_active else None
        else:
            state, event = self.gate.update(score, timestamp)
        self.result = {
            "state": state,
            "score": score,
            "regions": [list(regions[selected])],
            "people": people,
            "person_boxes": person_boxes,
            "visible_people": len(person_boxes),
            "event": event,
            "message": self.message,
            "prediction_timestamp": timestamp,
            "inference_seconds": time.monotonic() - inference_started,
        }
        return dict(self.result)

    def warmup(self):
        if not self.model_ready:
            return
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        if self.input_kind in {"rgb_pose", "rgb_pose_motion"}:
            self._ensure_pose()
        if self.detector is None:
            from detection.detector import PersonDetector

            self.detector = PersonDetector(
                **self.detector_config.to_dict(), device="cpu"
            )
        if hasattr(self.detector, "warmup"):
            self.detector.warmup(self.detector_config.image_size)
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        example = torch.zeros(
            1, 3, self.config.frames, self.config.size, self.config.size
        )
        with torch.inference_mode():
            for iteration in range(2):
                if self.input_kind == "rgb_pose_motion":
                    self.predictor(
                        example,
                        torch.zeros(1, 8, 2, 17, 3),
                        torch.zeros(1, 2, 15, 32, 32),
                    )
                elif self.input_kind == "rgb_pose":
                    self.predictor(example, torch.zeros(1, 8, 2, 17, 3))
                else:
                    self.predictor(example)

    def _ensure_pose(self):
        if self.pose_detector is None:
            from fight.pose import PoseConfig, PoseDetector

            self.pose_config = self.pose_config or PoseConfig()
            self.pose_detector = PoseDetector(
                self.pose_config,
                model_path=self.models_dir / self.pose_config.model_name,
            )
        self.detector = self.pose_detector
