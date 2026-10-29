import json
import hashlib
import math
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from core.config import BASE_DIR, settings
from fight.runtime import FightRuntime
from fight.session_recording import SessionRecording

UI_PATH = BASE_DIR / "frontend" / "fight.html"
SERVER_INSTANCE = uuid.uuid4().hex[:12]
BACKEND_REVISION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def ui_revision(source=None):
    source = UI_PATH.read_text(encoding="utf-8") if source is None else source
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def server_identity():
    return {
        "instance_id": SERVER_INSTANCE,
        "process_id": os.getpid(),
        "project_dir": str(BASE_DIR),
        "backend_revision": BACKEND_REVISION,
        "ui_revision": ui_revision(),
    }


LABELS = {
    "normal": "NO FIGHT CONFIRMED",
    "fighting": "FIGHT DETECTED",
    "verifying": "VERIFYING",
    "warming_up": "COLLECTING CLIP",
    "unavailable": "MODEL NOT READY",
    "insufficient": "CHECK CAMERA VIEW",
    "preview": "PREVIEW ONLY",
}


def bounded_image(image, maximum_width):
    height, width = image.shape[:2]
    if width <= maximum_width:
        return image
    return cv2.resize(
        image,
        (maximum_width, max(1, round(height * maximum_width / width))),
        interpolation=cv2.INTER_AREA,
    )


def open_capture(source):
    if isinstance(source, int) and os.name == "nt":
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return capture
        capture.release()
    if isinstance(source, str) and hasattr(cv2, "CAP_PROP_N_THREADS"):
        try:
            capture = cv2.VideoCapture(
                source,
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_N_THREADS, settings.fight_decode_threads],
            )
            if capture.isOpened():
                return capture
            capture.release()
        except (cv2.error, TypeError):
            pass
    return cv2.VideoCapture(source)


def annotate(frame, result):
    image = frame.copy()
    color = (
        (50, 65, 235)
        if result["state"] == "fighting"
        else (
            (40, 195, 215)
            if result["state"] in ("verifying", "insufficient")
            else (175, 195, 90)
        )
    )
    if not result.get("score_stale", False):
        for box in result.get("person_boxes", []):
            left, top, right, bottom = map(int, box)
            cv2.rectangle(image, (left, top), (right, bottom), (40, 200, 60), 2)
            cv2.putText(
                image,
                "PERSON",
                (left, max(48, top - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 200, 60),
                1,
            )
        for person in result.get("actions", []):
            left, top, _, _ = map(int, person["box"])
            for index, action in enumerate(person["labels"][:3]):
                cv2.putText(
                    image,
                    f"{action['name']}: {action['score']:.2f}",
                    (max(0, left), max(52, top) + index * 17),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (40, 220, 180),
                    1,
                )
        if result["state"] in {"fighting", "verifying"}:
            for region in result.get("regions", []):
                left, top, right, bottom = map(int, region)
                cv2.rectangle(image, (left, top), (right, bottom), color, 3)
    cv2.rectangle(image, (0, 0), (image.shape[1], 36), (25, 30, 28), -1)
    label = LABELS.get(result["state"], result["state"])
    visible = result.get("visible_people")
    if visible is not None and not result.get("score_stale", False):
        label += f" | PERSONS: {visible}"
    cv2.putText(
        image,
        label,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
    return image


class FightProcessor:
    def __init__(self):
        self.runtime = FightRuntime(
            settings.models_dir,
            settings.fight_allow_pilot,
            cpu_threads=settings.fight_cpu_threads,
            batch_detection=True,
        )
        self.thread = None
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.lifecycle = threading.Lock()
        self.jpeg = None
        self.revision = 0
        self.events = []
        self.phase = "idle"
        self.error = None
        self.source = None
        self.replay = False
        self.preview = False
        self.record_test = False
        self.recording = None
        self.session_started = None
        self.dropped_samples = 0
        self.preview_frame_count = 0
        self.frame_count = self.total_frames = 0
        self.timestamp = self.latency = 0.0
        self.output = None
        self.active_event = None
        self.event_frames = []
        self.before = deque(maxlen=24)
        self.evidence_dir = BASE_DIR / "uploads" / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=8)
        if not self.is_running() and hasattr(self.runtime, "close"):
            self.runtime.close()

    def start(
        self,
        source,
        allow_pilot=False,
        preview=False,
        replay=False,
        record_test=False,
        backend="legacy",
    ):
        with self.lifecycle:
            if preview and record_test:
                raise ValueError(
                    "Ghi test phải chạy cùng nhận diện, không dùng chế độ Xem thử."
                )
            if self.recording and self.recording.thread.is_alive():
                raise RuntimeError(
                    "Phiên ghi trước chưa kết thúc; hãy Dừng/chờ hoàn tất rồi thử lại."
                )
            if backend not in {"legacy", "mmaction_original", "mmaction_tuned"}:
                raise ValueError("Model không hợp lệ.")
            if backend == "legacy":
                candidate = FightRuntime(
                    settings.models_dir,
                    allow_pilot,
                    cpu_threads=settings.fight_cpu_threads,
                    batch_detection=True,
                )
            else:
                from fight.mmaction_runtime import MMActionRuntime

                candidate = MMActionRuntime(
                    tuned=backend == "mmaction_tuned", allow_pilot=allow_pilot
                )
            if not candidate.model_ready and not preview:
                raise ValueError(candidate.message)
            self.stop()
            if self.is_running():
                raise RuntimeError("Nguồn trước chưa dừng; vui lòng thử lại.")
            cv2.setNumThreads(1)
            if not preview:
                candidate.warmup()
            self.runtime = candidate
            self.stop_event.clear()
            self.preview = preview
            self.replay = replay
            self.record_test = record_test
            self.recording = None
            self.source, self.phase, self.error = source, "opening", None
            self.frame_count = self.total_frames = 0
            self.timestamp = self.latency = 0.0
            self.session_started = time.monotonic()
            self.dropped_samples = 0
            self.preview_frame_count = 0
            self.output = self.jpeg = None
            self.events = []
            self.active_event = None
            self.event_frames = []
            self.before.clear()
            self.thread = threading.Thread(target=self.run, args=(source,), daemon=True)
            self.thread.start()

    def status(self):
        prediction = dict(self.runtime.result)
        live_clock = self.replay or not self.total_frames
        current_time = (
            time.monotonic() - self.session_started
            if live_clock and self.session_started is not None and self.is_running()
            else self.timestamp
        )
        measured_seconds = max(0.0, current_time)
        observed_at = prediction.get("prediction_timestamp")
        age = max(0.0, current_time - observed_at) if observed_at is not None else None
        stale = (
            age is not None
            and live_clock
            and self.is_running()
            and age > self.runtime.gate.max_observation_gap
        )
        if stale:
            prediction.update(
                state="insufficient",
                person_boxes=[],
                visible_people=None,
                event=None,
                message="Kết quả suy luận đã cũ; CPU hoặc nguồn hình đang chậm, chưa xác nhận hành vi hiện tại.",
            )
        prediction["score_stale"] = stale
        return {
            "server": server_identity(),
            "running": self.is_running(),
            "preview_mode": self.preview,
            "phase": self.phase,
            "source": str(self.source) if self.source is not None else None,
            "source_kind": (
                "replay"
                if self.replay
                else ("video" if self.total_frames else "camera")
            ),
            "frame_count": self.frame_count,
            "total_frames": self.total_frames,
            "source_fps_actual": (
                round(self.frame_count / measured_seconds, 2)
                if live_clock and measured_seconds >= 1
                else None
            ),
            "preview_fps_actual": (
                round(self.preview_frame_count / measured_seconds, 2)
                if live_clock and measured_seconds >= 1
                else None
            ),
            "timestamp": self.timestamp,
            "latency_seconds": round(
                max(self.latency, current_time - self.timestamp), 3
            ),
            "prediction_age_seconds": round(age, 3) if age is not None else None,
            "dropped_samples": self.dropped_samples,
            "analysis_incomplete": self.dropped_samples > 0,
            "model_loaded": self.runtime.model_ready,
            "model": {
                "backend": getattr(self.runtime, "backend_name", "legacy"),
                "device": getattr(self.runtime, "device", "cpu"),
                "architecture": self.runtime.metadata.get("architecture"),
                "input_kind": self.runtime.input_kind,
                "cpu_threads": getattr(
                    self.runtime, "cpu_threads", settings.fight_cpu_threads
                ),
                "cpu_optimization": getattr(self.runtime, "cpu_optimization", "none"),
                "pose_prefetch": getattr(self.runtime, "batch_detection", False),
                "pilot_required": getattr(self.runtime, "pilot_required", False),
                "pilot_default_allowed": settings.fight_allow_pilot,
                "deployment": self.runtime.metadata.get("deployment", {}),
                "approved_for_demo": self.runtime.metadata.get(
                    "approved_for_demo", False
                ),
                "message": self.runtime.message,
                "clip_test": self.runtime.metadata.get("test"),
                "hard_case_test": self.runtime.metadata.get("test_by_dataset", {}).get(
                    "airtlab"
                ),
                "real_phone_tested": self.runtime.metadata.get(
                    "real_phone_tested", False
                ),
            },
            "prediction": prediction,
            "error": self.error,
            "event_count": len(self.events),
            "output": self.output,
            "recording": self.recording.status() if self.recording else None,
        }

    def frames(self):
        revision = -1
        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: self.revision != revision or not self.is_running(),
                    timeout=5,
                )
                jpeg, revision = self.jpeg, self.revision
            if jpeg is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            if not self.is_running():
                break

    def publish_frame(self, frame, result):
        image = annotate(frame, result)
        preview_image = bounded_image(image, settings.fight_preview_max_width)
        ok, encoded = cv2.imencode(
            ".jpg",
            preview_image,
            [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality],
        )
        if ok:
            with self.condition:
                self.jpeg = encoded.tobytes()
                self.revision += 1
                self.condition.notify_all()

    def live_preview_result(self, timestamp):
        if self.preview:
            return {
                "state": "preview",
                "score": None,
                "regions": [],
                "person_boxes": [],
                "actions": [],
                "visible_people": None,
                "people": 0,
                "event": None,
                "message": "Chỉ xem trước camera",
            }
        result = dict(self.runtime.result)
        observed_at = result.get("prediction_timestamp")
        if (
            observed_at is not None
            and timestamp - observed_at > self.runtime.gate.max_observation_gap
        ):
            result.update(
                state="insufficient",
                person_boxes=[],
                regions=[],
                actions=[],
                visible_people=None,
                event=None,
                score_stale=True,
            )
        return result

    def record(self, frame, result):
        self.before.append((self.timestamp, frame.copy()))
        if result.get("event") == "started" and self.active_event is None:
            identifier = uuid.uuid4().hex
            self.active_event = {
                "id": identifier,
                "type": "fighting",
                "started_at": self.timestamp,
                "ended_at": None,
                "score": result["score"],
                "snapshot": f"/api/evidence/{identifier}.jpg",
                "clip": None,
            }
            self.events.append(self.active_event)
            self.event_frames = list(self.before)
            cv2.imwrite(
                str(self.evidence_dir / f"{identifier}.jpg"), annotate(frame, result)
            )
        elif self.active_event is not None:
            self.event_frames.append((self.timestamp, frame.copy()))
        if self.active_event is not None:
            self.active_event["score"] = max(
                self.active_event["score"], result.get("score") or 0
            )
            if (
                result.get("event") == "ended"
                or self.timestamp - self.event_frames[0][0] >= 30
            ):
                self.finish_event()

    def finish_event(self):
        if self.active_event is None:
            return
        event = self.active_event
        event["ended_at"] = self.timestamp
        height, width = self.event_frames[0][1].shape[:2]
        event["clip_started_at"] = self.event_frames[0][0]
        event["clip_ended_at"] = self.event_frames[-1][0]
        event["source_samples"] = len(self.event_frames)
        event["max_sample_gap_seconds"] = max(
            (
                current[0] - previous[0]
                for previous, current in zip(self.event_frames, self.event_frames[1:])
            ),
            default=0.0,
        )
        path = self.evidence_dir / f"{event['id']}.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.runtime.config.fps,
            (width, height),
        )
        if writer.isOpened():
            sample_index = 0
            total = (
                round(
                    (self.event_frames[-1][0] - self.event_frames[0][0])
                    * self.runtime.config.fps
                )
                + 1
            )
            for frame_index in range(min(total, round(self.runtime.config.fps * 35))):
                target_time = (
                    self.event_frames[0][0] + frame_index / self.runtime.config.fps
                )
                while (
                    sample_index + 1 < len(self.event_frames)
                    and self.event_frames[sample_index + 1][0] <= target_time + 1e-6
                ):
                    sample_index += 1
                writer.write(
                    cv2.resize(self.event_frames[sample_index][1], (width, height))
                )
            event["clip"] = f"/api/evidence/{path.name}"
        writer.release()
        (self.evidence_dir / f"{event['id']}.json").write_text(
            json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.active_event, self.event_frames = None, []

    def rebuild_file_event_clips(self, source, fps):
        if not self.events:
            return
        capture = open_capture(source)
        if not capture.isOpened():
            raise ValueError("Không mở lại được video nguồn để dựng clip sự kiện.")
        writers = {}
        frame_counts = {event["id"]: 0 for event in self.events}
        try:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = index / fps
                index += 1
                for event in self.events:
                    if not (
                        event["clip_started_at"] - 1e-6
                        <= timestamp
                        <= event["clip_ended_at"] + 1e-6
                    ):
                        continue
                    output = bounded_image(frame, settings.fight_output_max_width)
                    writer = writers.get(event["id"])
                    if writer is None:
                        path = self.evidence_dir / f"{event['id']}.mp4"
                        writer = cv2.VideoWriter(
                            str(path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (output.shape[1], output.shape[0]),
                        )
                        if not writer.isOpened():
                            raise RuntimeError("Không tạo được clip sự kiện full-FPS.")
                        writers[event["id"]] = writer
                    writer.write(output)
                    frame_counts[event["id"]] += 1
        finally:
            capture.release()
            for writer in writers.values():
                writer.release()
        for event in self.events:
            if frame_counts[event["id"]]:
                event["clip_fps"] = fps
                event["clip_frames"] = frame_counts[event["id"]]
                (self.evidence_dir / f"{event['id']}.json").write_text(
                    json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8"
                )

    def reset_after_sample_loss(self):
        self.finish_event()
        self.runtime.reset()
        self.runtime.result["message"] = (
            "CPU xử lý không kịp, đã mất mẫu; đang thu lại clip, không kết luận bình thường."
        )

    def run(self, source):
        is_file = isinstance(source, str) and Path(source).is_file()
        live_mode = not is_file or self.replay
        capture = open_capture(source)
        writer = reader = None
        previewer = None
        fps = None
        pending = deque(maxlen=self.runtime.config.frames * 2)
        observed_drops = 0
        live_lock = threading.Lock()
        preview_lock = threading.Lock()
        preview_event = threading.Event()
        preview_latest = {"packet": None}
        latest = {"frame": None, "ended": False, "count": 0, "limit_reached": False}
        started = self.session_started or time.monotonic()
        try:
            if not capture.isOpened():
                raise ValueError("Không mở được nguồn hình. Kiểm tra camera hoặc URL.")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            fps = fps if 0 < fps <= 240 else 25.0
            self.total_frames = (
                int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
            )
            self.phase = "processing"
            output_path = self.evidence_dir / f"{uuid.uuid4().hex}.mp4"
            recording = None
            if self.record_test:
                if not live_mode:
                    raise ValueError(
                        "Ghi test chỉ dành cho luồng camera hoặc phát lại realtime."
                    )
                recording_fps = min(60.0, fps)
                recording = self.recording = SessionRecording(
                    self.evidence_dir,
                    recording_fps,
                    {
                        "deployment": self.runtime.metadata.get("deployment", {}),
                        "clip": self.runtime.config.to_dict(),
                        "input_kind": self.runtime.input_kind,
                        "use_pose": self.runtime.metadata.get("use_pose"),
                        "pose": self.runtime.metadata.get("pose"),
                        "use_motion": self.runtime.metadata.get("use_motion"),
                        "motion": self.runtime.metadata.get("motion"),
                        "threshold": self.runtime.gate.threshold,
                        "event_gate": {
                            "confirm_seconds": self.runtime.gate.confirm_seconds,
                            "release_seconds": self.runtime.gate.release_seconds,
                            "release_threshold": self.runtime.gate.release_threshold,
                            "max_observation_gap": self.runtime.gate.max_observation_gap,
                        },
                        "source_kind": "replay" if self.replay else "camera",
                        "reported_source_fps": fps,
                    },
                    limit_seconds=settings.fight_test_record_seconds,
                    capacity=max(12, math.ceil(recording_fps * 2)),
                )
            if live_mode:

                # Opening a physical/network camera can take seconds. Source timestamps,
                # recording limits and freshness begin when capture is actually ready,
                # not while the driver is still opening.
                started = time.monotonic()
                self.session_started = started

                def publish_live():
                    while not self.stop_event.is_set() or preview_latest["packet"]:
                        preview_event.wait(0.1)
                        preview_event.clear()
                        with preview_lock:
                            packet = preview_latest["packet"]
                            preview_latest["packet"] = None
                        if packet is None:
                            continue
                        preview_frame, preview_timestamp = packet
                        self.publish_frame(
                            preview_frame,
                            self.live_preview_result(preview_timestamp),
                        )
                        self.preview_frame_count += 1

                def read_live():
                    next_sample = 0.0
                    next_preview = 0.0
                    preview_interval = 1 / settings.stream_fps_cap
                    source_index = 0
                    while not self.stop_event.is_set():
                        ok, frame = capture.read()
                        if not ok:
                            break
                        timestamp = (
                            source_index / fps
                            if is_file
                            else time.monotonic() - started
                        )
                        source_index += 1
                        if is_file and self.stop_event.wait(
                            max(0.0, started + timestamp - time.monotonic())
                        ):
                            break
                        if recording and timestamp >= recording.limit_seconds:
                            with live_lock:
                                latest["limit_reached"] = True
                            break
                        if recording:
                            recording.offer(frame, timestamp)
                        with live_lock:
                            latest.update(frame=frame, count=latest["count"] + 1)
                            if timestamp + 1e-6 >= next_sample:
                                if len(pending) == pending.maxlen:
                                    self.dropped_samples += 1
                                pending.append((frame, timestamp))
                                while next_sample <= timestamp + 1e-6:
                                    next_sample += 1 / self.runtime.config.fps
                        if timestamp + 1e-6 >= next_preview:
                            with preview_lock:
                                preview_latest["packet"] = (frame, timestamp)
                            preview_event.set()
                            while next_preview <= timestamp + 1e-6:
                                next_preview += preview_interval
                    with live_lock:
                        latest["ended"] = True

                previewer = threading.Thread(target=publish_live, daemon=True)
                previewer.start()
                reader = threading.Thread(target=read_live, daemon=True)
                reader.start()
            last_evidence = -1.0
            while not self.stop_event.is_set():
                if not live_mode:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    timestamp = self.frame_count / fps
                    samples = [(frame, timestamp)]
                    self.frame_count += 1
                else:
                    with live_lock:
                        frame, samples, ended = (
                            latest["frame"],
                            list(pending),
                            latest["ended"],
                        )
                        pending.clear()
                        self.frame_count = latest["count"]
                        sampled_drops = self.dropped_samples
                    if not samples:
                        if ended:
                            if is_file or latest["limit_reached"]:
                                break
                            raise ValueError("Camera mất kết nối.")
                        self.stop_event.wait(0.02)
                        continue
                    timestamp = samples[-1][1]
                    if sampled_drops > observed_drops and not self.preview:
                        self.reset_after_sample_loss()
                    observed_drops = sampled_drops
                for sample, sample_time in samples:
                    result = (
                        {
                            "state": "unavailable",
                            "score": None,
                            "regions": [],
                            "people": 0,
                            "event": None,
                            "message": "Chỉ xem trước camera",
                        }
                        if self.preview
                        else self.runtime.push(
                            sample,
                            sample_time,
                            predict=True,
                            sampled=live_mode,
                        )
                    )
                    if recording:
                        recording.observe(
                            result, time.monotonic() - started, sample_time
                        )
                    self.timestamp = sample_time
                    if (
                        sample_time - last_evidence + 1e-6
                        >= 1 / self.runtime.config.fps
                        or result.get("event")
                    ):
                        self.record(sample, result)
                        last_evidence = sample_time
                if self.preview:
                    self.runtime.result = result
                self.timestamp = timestamp
                self.latency = (
                    max(0.0, time.monotonic() - started - timestamp)
                    if live_mode
                    else 0.0
                )
                if not live_mode:
                    image = annotate(frame, result)
                    output_image = bounded_image(image, settings.fight_output_max_width)
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (output_image.shape[1], output_image.shape[0]),
                        )
                        if not writer.isOpened():
                            raise RuntimeError("Không tạo được video kết quả.")
                    writer.write(output_image)
                    self.publish_frame(frame, result)
            self.phase = "stopped" if self.stop_event.is_set() else "completed"
            if is_file and writer is not None:
                self.output = f"/api/evidence/{output_path.name}"
        except Exception as error:
            self.phase, self.error = "error", str(error)
        finally:
            self.stop_event.set()
            preview_event.set()
            capture.release()
            if reader:
                reader.join(timeout=2)
            if previewer:
                previewer.join(timeout=2)
            if writer:
                writer.release()
            self.finish_event()
            if is_file and not self.replay and fps is not None and self.events:
                try:
                    self.rebuild_file_event_clips(source, fps)
                except Exception as error:
                    self.error = "Không dựng được clip sự kiện full-FPS: " + str(error)
                    self.phase = "error"
            if self.recording:
                self.recording.finish(
                    outcome={
                        "phase": self.phase,
                        "error": self.error,
                        "dropped_model_samples": self.dropped_samples,
                        "source_frames": self.frame_count,
                        "source_seconds": self.timestamp,
                    }
                )
            if hasattr(self.runtime, "close"):
                self.runtime.close()
            with self.condition:
                self.condition.notify_all()


app = FastAPI(title="FightWatch", version="2.0.0")
processor = FightProcessor()


@app.middleware("http")
async def prevent_stale_local_state(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-FightWatch-Instance"] = SERVER_INSTANCE
    return response


@app.get("/", response_class=HTMLResponse)
def frontend():
    source = UI_PATH.read_text(encoding="utf-8")
    revision = ui_revision(source)
    source = source.replace("__FIGHTWATCH_UI_VERSION__", revision)
    source = source.replace("__FIGHTWATCH_SERVER_INSTANCE__", SERVER_INSTANCE)
    return HTMLResponse(source, headers={"X-FightWatch-UI-Version": revision})


@app.get("/api/status")
def status():
    return processor.status()


@app.get("/api/events")
def events():
    return {"events": [dict(event) for event in processor.events[-100:]]}


def start_source(
    source,
    allow_pilot,
    preview=False,
    replay=False,
    record_test=False,
    backend="legacy",
):
    try:
        if backend != "legacy":
            processor.start(
                source,
                allow_pilot,
                preview,
                replay=replay,
                record_test=record_test,
                backend=backend,
            )
        elif record_test:
            processor.start(
                source, allow_pilot, preview, replay=replay, record_test=True
            )
        elif replay:
            processor.start(source, allow_pilot, preview, replay=True)
        else:
            processor.start(source, allow_pilot, preview)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(409, str(error)) from error
    return {"status": "started"}


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    allow_pilot: bool = False,
    replay: bool = False,
    backend: str = "legacy",
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(400, "Chọn một file video hợp lệ.")
    destination = BASE_DIR / "uploads" / f"{uuid.uuid4().hex}{suffix}"
    with destination.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            handle.write(chunk)
    return start_source(str(destination), allow_pilot, replay=replay, backend=backend)


@app.post("/api/stream/camera")
def camera(
    index: int = 0,
    allow_pilot: bool = False,
    preview: bool = False,
    record_test: bool = False,
    backend: str = "legacy",
):
    if not 0 <= index <= 20:
        raise HTTPException(400, "Camera index không hợp lệ.")
    return start_source(
        index, allow_pilot, preview, record_test=record_test, backend=backend
    )


@app.get("/api/cameras")
def cameras():
    found = []
    for index in range(6):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            found.append({"index": index, "label": f"Camera {index}"})
        capture.release()
    return {"cameras": found}


@app.post("/api/stream/url")
def stream_url(url: str, allow_pilot: bool = False, backend: str = "legacy"):
    if not url.lower().startswith(("rtsp://", "http://", "https://")):
        raise HTTPException(400, "Cần URL RTSP hoặc HTTP(S).")
    return start_source(url, allow_pilot, backend=backend)


@app.post("/api/stop")
def stop():
    processor.stop()
    return {"status": "stopped"}


@app.get("/api/stream")
def stream():
    return StreamingResponse(
        processor.frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/report")
def report():
    return {
        "system": "FightWatch",
        "status": processor.status(),
        "events": processor.events,
    }


@app.get("/api/evidence/{name}")
def evidence(name: str):
    if Path(name).name != name or Path(name).suffix not in {".jpg", ".mp4", ".json"}:
        raise HTTPException(404)
    path = processor.evidence_dir / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)
