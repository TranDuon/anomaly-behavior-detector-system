import json
import math
import queue
import threading
import uuid
from pathlib import Path

import cv2


class SessionRecording:
    def __init__(self, directory, fps, metadata, limit_seconds=30, capacity=12):
        if not 0 < fps <= 60 or not 0 < limit_seconds <= 30 or capacity < 1:
            raise ValueError("Invalid recording limits")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identifier = "session_" + uuid.uuid4().hex
        self.video_path = self.directory / (self.identifier + ".mp4")
        self.report_path = self.directory / (self.identifier + ".json")
        self.fps = fps
        self.limit_seconds = limit_seconds
        self.metadata = metadata
        self.pending = queue.Queue(maxsize=capacity)
        self.closing = threading.Event()
        self.lock = threading.Lock()
        self.phase = "recording"
        self.error = None
        self.source_time = 0.0
        self.dropped_frames = 0
        self.repeated_frames = 0
        self.saved_times = []
        self.predictions = []
        self.state_changes = []
        self.video_ready = self.report_ready = False
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def offer(self, frame, timestamp):
        with self.lock:
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("Finite nonnegative source timestamps required")
            if self.closing.is_set() or self.error or timestamp >= self.limit_seconds:
                return False
            self.source_time = timestamp
            try:
                self.pending.put_nowait((frame, timestamp))
                return True
            except queue.Full:
                self.dropped_frames += 1
                return False

    def observe(self, result, processing_time, source_time=None):
        timestamp = result.get("prediction_timestamp")
        with self.lock:
            state = {"state": result.get("state"), "message": result.get("message")}
            if len(self.state_changes) < 2000 and (
                not self.state_changes
                or any(
                    self.state_changes[-1].get(key) != value
                    for key, value in state.items()
                )
            ):
                self.state_changes.append(
                    {
                        **state,
                        "source_timestamp": source_time,
                        "processing_elapsed_seconds": processing_time,
                    }
                )
            if timestamp is None:
                return
            if (
                self.predictions
                and self.predictions[-1]["prediction_timestamp"] == timestamp
            ):
                return
            if len(self.predictions) >= 2000:
                return
            self.predictions.append(
                {
                    **{
                        key: result.get(key)
                        for key in (
                            "prediction_timestamp",
                            "score",
                            "state",
                            "event",
                            "people",
                            "visible_people",
                            "person_boxes",
                            "regions",
                            "actions",
                            "base_score",
                            "fusion_context",
                            "inference_seconds",
                            "message",
                        )
                    },
                    "processing_elapsed_seconds": processing_time,
                    "prediction_age_seconds": max(0.0, processing_time - timestamp),
                }
            )

    def finish(self, timeout=5, outcome=None):
        with self.lock:
            if outcome is not None:
                self.metadata = {**self.metadata, "session_end": dict(outcome)}
            if self.phase == "recording":
                self.phase = "finishing"
        self.closing.set()
        self.thread.join(timeout=timeout)

    def status(self):
        with self.lock:
            return {
                "id": self.identifier,
                "phase": self.phase,
                "error": self.error,
                "limit_seconds": self.limit_seconds,
                "source_seconds": self.source_time,
                "fps": self.fps,
                "video_frames": len(self.saved_times),
                "repeated_frames": self.repeated_frames,
                "recording_dropped_frames": self.dropped_frames,
                "predictions": len(self.predictions),
                "video": (
                    "/api/evidence/" + self.video_path.name
                    if self.video_ready
                    else None
                ),
                "report": (
                    "/api/evidence/" + self.report_path.name
                    if self.report_ready
                    else None
                ),
            }

    def _write(self):
        writer = None
        previous = None
        start_time = previous_time = None
        dimensions = None
        maximum_frames = math.ceil(self.fps * self.limit_seconds)
        try:
            while not self.closing.is_set() or not self.pending.empty():
                try:
                    frame, timestamp = self.pending.get(timeout=0.05)
                except queue.Empty:
                    continue
                height, width = frame.shape[:2]
                if writer is None:
                    if width * height > 1920 * 1080:
                        raise ValueError(
                            "Ghi test hỗ trợ tối đa 1080p; hãy hạ độ phân giải nguồn."
                        )
                    dimensions = (width, height)
                    writer = cv2.VideoWriter(
                        str(self.video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.fps,
                        dimensions,
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Không mở được file ghi video test.")
                    start_time = timestamp
                if dimensions != (width, height):
                    raise ValueError("Độ phân giải nguồn thay đổi trong phiên ghi.")
                target = min(
                    maximum_frames - 1,
                    max(0, round((timestamp - start_time) * self.fps)),
                )
                while len(self.saved_times) < target and previous is not None:
                    writer.write(previous)
                    with self.lock:
                        if self.saved_times and self.saved_times[-1] == previous_time:
                            self.repeated_frames += 1
                        self.saved_times.append(previous_time)
                if len(self.saved_times) <= target:
                    writer.write(frame)
                    with self.lock:
                        self.saved_times.append(timestamp)
                previous, previous_time = frame, timestamp
            if (
                writer is not None
                and previous_time is not None
                and len(self.saved_times) < maximum_frames
                and self.saved_times[-1] != previous_time
            ):
                writer.write(previous)
                with self.lock:
                    self.saved_times.append(previous_time)
        except Exception as error:
            with self.lock:
                self.error = str(error)
            self.closing.set()
        finally:
            if writer is not None:
                writer.release()
            while not self.pending.empty():
                try:
                    self.pending.get_nowait()
                    with self.lock:
                        self.dropped_frames += 1
                except queue.Empty:
                    break
            with self.lock:
                if not self.saved_times and not self.error:
                    self.error = "Không nhận được frame để lưu video test."
                if not self.error and (
                    not self.video_path.is_file() or self.video_path.stat().st_size == 0
                ):
                    self.error = "File video test trống hoặc chưa được tạo."
                self.video_ready = (
                    not self.error
                    and bool(self.saved_times)
                    and self.video_path.is_file()
                    and self.video_path.stat().st_size > 0
                )
                self.phase = "failed" if self.error else "saved"
                scores = [
                    row["score"] for row in self.predictions if row["score"] is not None
                ]
                report = {
                    "schema_version": 1,
                    "scope": "Local camera diagnostic, no audio; not labeled action accuracy",
                    "metadata": self.metadata,
                    "recording": self.status_unlocked(),
                    "video_start_source_seconds": start_time,
                    "frame_source_timestamps": list(self.saved_times),
                    "predictions": [dict(row) for row in self.predictions],
                    "state_changes": [dict(row) for row in self.state_changes],
                    "maximum_observed_score": max(scores, default=None),
                    "events": [row for row in self.predictions if row["event"]],
                    "timing_note": "Video uses constant sampling FPS; gaps repeat the previous frame. Use source timestamps, repeated/drop counts and live predictions when diagnosing; lossy MP4 replay is not bit-identical model input.",
                    "video_dimensions": dimensions,
                }
            try:
                partial = self.report_path.with_suffix(".partial.json")
                partial.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                partial.replace(self.report_path)
                with self.lock:
                    self.report_ready = True
            except Exception as error:
                with self.lock:
                    self.error = "Không lưu được log test: " + str(error)
                    self.phase = "failed"

    def status_unlocked(self):
        return {
            "id": self.identifier,
            "phase": self.phase,
            "error": self.error,
            "limit_seconds": self.limit_seconds,
            "fps": self.fps,
            "source_seconds": self.source_time,
            "video_frames": len(self.saved_times),
            "recording_dropped_frames": self.dropped_frames,
            "repeated_frames": self.repeated_frames,
            "video_file": self.video_path.name if self.video_ready else None,
        }
